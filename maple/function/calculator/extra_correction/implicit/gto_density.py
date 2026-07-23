"""Electrostatic observables from the MACE-POLAR l<=1 GTO density."""

from __future__ import annotations

import math

import numpy as np
from ase.units import Bohr
from scipy.special import erf


MACE_POLAR_DENSITY_SIGMA_ANGSTROM = 1.5


def cartesian_multipoles(
    density_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return atom-centred charges and Cartesian dipoles.

    MACE-POLAR stores the l=1 real-spherical components in its e3nn phase
    convention.  Upstream converts columns 1:4 to Cartesian x/y/z with the
    permutation ``[2, 0, 1]`` when reporting the molecular dipole.
    """

    coefficients = np.asarray(density_coefficients, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[1] != 4:
        raise ValueError(
            "MACE-POLAR Route 2 requires density coefficients with shape "
            f"(n_atoms, 4); received {coefficients.shape}."
        )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("MACE-POLAR density coefficients contain non-finite values.")
    charges = coefficients[:, 0].copy()
    dipoles_e_angstrom = coefficients[:, 1:4][:, [2, 0, 1]].copy()
    return charges, dipoles_e_angstrom


def point_multipole_potential(
    points_bohr: np.ndarray,
    atom_positions_angstrom: np.ndarray,
    density_coefficients: np.ndarray,
) -> np.ndarray:
    """Evaluate the cavity-exterior point-multipole MEP in atomic units.

    MACE-POLAR's monopole and dipole coefficients are retained, but the
    model-internal Gaussian smearing is not extended across the PCM dielectric
    boundary. PCMSolver cavity points are outside the atomic centres, so the
    exterior l<=1 multipole expansion is finite there.
    """

    points = np.asarray(points_bohr, dtype=float)
    positions = np.asarray(atom_positions_angstrom, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("MEP points must have shape (n_points, 3).")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Atom positions must have shape (n_atoms, 3).")
    charges, dipoles_angstrom = cartesian_multipoles(density_coefficients)
    if positions.shape[0] != charges.shape[0]:
        raise ValueError("Density coefficient count does not match atom positions.")

    positions_bohr = positions / Bohr
    dipoles_bohr = dipoles_angstrom / Bohr
    potential = np.zeros(points.shape[0], dtype=float)
    for center, charge, dipole in zip(
        positions_bohr, charges, dipoles_bohr, strict=True
    ):
        displacement = points - center
        radius = np.linalg.norm(displacement, axis=1)
        if np.any(radius <= 1.0e-14):
            raise ValueError("A PCM surface point coincides with an atomic centre.")
        potential += charge / radius
        potential += np.einsum("ij,j->i", displacement, dipole) / (radius**3)
    return potential


def gaussian_multipole_potential(
    points_bohr: np.ndarray,
    atom_positions_angstrom: np.ndarray,
    density_coefficients: np.ndarray,
    *,
    sigma_angstrom: float = MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
) -> np.ndarray:
    """Evaluate the solute MEP at arbitrary points in atomic units.

    The l=0/l=1 coefficients are normalized total charge and centred dipole
    moments.  A common spherical Gaussian width is used by MACE-POLAR-1.
    Input points are in bohr because that is PCMSolver's API convention;
    returned values are Hartree per elementary charge.
    """

    points = np.asarray(points_bohr, dtype=float)
    positions = np.asarray(atom_positions_angstrom, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("MEP points must have shape (n_points, 3).")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Atom positions must have shape (n_atoms, 3).")
    charges, dipoles_angstrom = cartesian_multipoles(density_coefficients)
    if positions.shape[0] != charges.shape[0]:
        raise ValueError("Density coefficient count does not match atom positions.")
    if sigma_angstrom <= 0:
        raise ValueError("The Gaussian density width must be positive.")

    positions_bohr = positions / Bohr
    dipoles_bohr = dipoles_angstrom / Bohr
    sigma_bohr = float(sigma_angstrom) / Bohr
    root_two_sigma = math.sqrt(2.0) * sigma_bohr
    root_pi = math.sqrt(math.pi)

    potential = np.zeros(points.shape[0], dtype=float)
    for center, charge, dipole in zip(
        positions_bohr, charges, dipoles_bohr, strict=True
    ):
        displacement = points - center
        radius = np.linalg.norm(displacement, axis=1)
        nonzero = radius > 1.0e-14

        x = np.zeros_like(radius)
        x[nonzero] = radius[nonzero] / root_two_sigma
        monopole_kernel = np.full(
            radius.shape,
            math.sqrt(2.0 / math.pi) / sigma_bohr,
            dtype=float,
        )
        monopole_kernel[nonzero] = erf(x[nonzero]) / radius[nonzero]
        potential += charge * monopole_kernel

        # -p.grad[erf(r/(sqrt(2)sigma))/r], expressed so that it
        # approaches the ordinary point-dipole potential p.r/r^3.
        dipole_kernel = np.zeros_like(radius)
        dipole_kernel[nonzero] = (
            erf(x[nonzero])
            - (2.0 * x[nonzero] / root_pi) * np.exp(-(x[nonzero] ** 2))
        ) / (radius[nonzero] ** 3)
        potential += np.einsum("ij,j->i", displacement, dipole) * dipole_kernel

    return potential


def point_asc_reaction_potential_gradient(
    atom_positions_angstrom: np.ndarray,
    surface_centers_bohr: np.ndarray,
    apparent_surface_charges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ASC onto atom-centred point monopoles and dipoles.

    The returned potential and gradient use the same exterior Coulomb kernel as
    :func:`point_multipole_potential`, preserving
    ``sum(q*V + p.grad(V)) == dot(MEP, ASC)``.
    """

    positions_bohr = np.asarray(atom_positions_angstrom, dtype=float) / Bohr
    centers = np.asarray(surface_centers_bohr, dtype=float)
    asc = np.asarray(apparent_surface_charges, dtype=float)
    if positions_bohr.ndim != 2 or positions_bohr.shape[1] != 3:
        raise ValueError("Atom positions must have shape (n_atoms, 3).")
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("Surface centers must have shape (n_surface, 3).")
    if asc.shape != (centers.shape[0],):
        raise ValueError("ASC vector length does not match surface centers.")

    potential = np.empty(positions_bohr.shape[0], dtype=float)
    gradient = np.empty_like(positions_bohr)
    for index, position in enumerate(positions_bohr):
        displacement = centers - position
        radius = np.linalg.norm(displacement, axis=1)
        if np.any(radius <= 1.0e-14):
            raise ValueError("A PCM surface point coincides with an atomic centre.")
        potential[index] = np.dot(asc, 1.0 / radius)
        gradient[index] = np.sum(
            (asc / (radius**3))[:, None] * displacement,
            axis=0,
        )
    return potential, gradient


def asc_reaction_potential_gradient(
    atom_positions_angstrom: np.ndarray,
    surface_centers_bohr: np.ndarray,
    apparent_surface_charges: np.ndarray,
    *,
    sigma_angstrom: float = MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the ASC reaction potential onto the MACE-POLAR GTO moments.

    The returned potential is in Hartree/e and the gradient is in
    Hartree/(e bohr).  The same Gaussian width as the MACE-POLAR density is
    used, making

    ``sum(q*V + p.grad(V)) == dot(MEP, ASC)``

    for the discrete l<=1 representation (up to floating-point roundoff).
    PCMSolver ASC values are integrated tessera charges, not charge densities,
    so no surface-area factor is applied.
    """

    positions_bohr = np.asarray(atom_positions_angstrom, dtype=float) / Bohr
    centers = np.asarray(surface_centers_bohr, dtype=float)
    asc = np.asarray(apparent_surface_charges, dtype=float)
    if positions_bohr.ndim != 2 or positions_bohr.shape[1] != 3:
        raise ValueError("Atom positions must have shape (n_atoms, 3).")
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("Surface centers must have shape (n_surface, 3).")
    if asc.shape != (centers.shape[0],):
        raise ValueError("ASC vector length does not match surface centers.")
    if sigma_angstrom <= 0:
        raise ValueError("The Gaussian density width must be positive.")

    potential = np.empty(positions_bohr.shape[0], dtype=float)
    gradient = np.empty_like(positions_bohr)
    sigma_bohr = float(sigma_angstrom) / Bohr
    root_two_sigma = math.sqrt(2.0) * sigma_bohr
    root_pi = math.sqrt(math.pi)
    for index, position in enumerate(positions_bohr):
        displacement = centers - position
        radius = np.linalg.norm(displacement, axis=1)
        nonzero = radius > 1.0e-14
        x = np.zeros_like(radius)
        x[nonzero] = radius[nonzero] / root_two_sigma

        monopole_kernel = np.full(
            radius.shape,
            math.sqrt(2.0 / math.pi) / sigma_bohr,
            dtype=float,
        )
        monopole_kernel[nonzero] = erf(x[nonzero]) / radius[nonzero]

        dipole_kernel = np.zeros_like(radius)
        dipole_kernel[nonzero] = (
            erf(x[nonzero])
            - (2.0 * x[nonzero] / root_pi) * np.exp(-(x[nonzero] ** 2))
        ) / (radius[nonzero] ** 3)

        potential[index] = np.dot(asc, monopole_kernel)
        gradient[index] = np.sum(
            (asc * dipole_kernel)[:, None] * displacement,
            axis=0,
        )
    return potential, gradient


def density_reaction_coupling(
    density_coefficients: np.ndarray,
    reaction_potential_hartree: np.ndarray,
    reaction_gradient_hartree_per_bohr: np.ndarray,
) -> float:
    """Return ``<rho,V_reac>`` in Hartree for l<=1 atom-centred multipoles."""

    charges, dipoles_angstrom = cartesian_multipoles(density_coefficients)
    potential = np.asarray(reaction_potential_hartree, dtype=float)
    gradient = np.asarray(reaction_gradient_hartree_per_bohr, dtype=float)
    if potential.shape != charges.shape:
        raise ValueError("Reaction potential length does not match the density.")
    if gradient.shape != (charges.size, 3):
        raise ValueError("Reaction gradient shape does not match the density.")
    dipoles_bohr = dipoles_angstrom / Bohr
    return float(
        np.dot(charges, potential) + np.einsum("ij,ij->", dipoles_bohr, gradient)
    )
