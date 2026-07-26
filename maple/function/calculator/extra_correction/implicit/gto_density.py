"""Electrostatic observables from the MACE-POLAR l<=1 GTO density."""

from __future__ import annotations

import math

import numpy as np
from ase.units import Bohr
from scipy.special import erf

from .electrostatic_pairing import MACE_POLAR_L1_PAIRING


MACE_POLAR_DENSITY_SIGMA_ANGSTROM = 1.5


def external_field_to_density_order(values: np.ndarray) -> np.ndarray:
    """Reorder a Cartesian node field into the raw density-dual convention.

    MACE-POLAR stores its real-spherical ``l=1`` coefficients so that raw
    columns ``[1, 2, 3]`` map to Cartesian ``[y, z, x]`` in the dual pairing.
    Consequently an external field ``[V, gx, gy, gz]`` pairs with the raw
    density as ``[V, gy, gz, gx]``.  This function changes order only; it does
    not introduce a unit conversion.
    """

    return MACE_POLAR_L1_PAIRING.field_to_density_order(values)


def density_to_external_field_order(values: np.ndarray) -> np.ndarray:
    """Reorder a raw density-dual block into external Cartesian field order.

    This is the inverse/transpose of :func:`external_field_to_density_order`.
    It maps raw MACE-POLAR coefficients ``[q, l1_0, l1_1, l1_2]`` to the
    external dual order ``[q, x, y, z] = [q, l1_2, l1_0, l1_1]`` without
    changing units.
    """

    return MACE_POLAR_L1_PAIRING.density_to_field_order(values)


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


def point_multipole_potential_position_vjp(
    points_bohr: np.ndarray,
    atom_positions_angstrom: np.ndarray,
    density_coefficients: np.ndarray,
    surface_cotangent: np.ndarray,
) -> np.ndarray:
    """Contract the point-multipole MEP position derivative with a cotangent.

    Density coefficients and surface points are held fixed.  This evaluates
    ``(dV_surface/dR).T @ surface_cotangent`` directly, without constructing a
    dense surface-by-coordinate Jacobian.  The returned array has shape
    ``(n_atoms, 3)``.  With an integrated-charge cotangent, its units are
    Hartree/Angstrom.  This is one explicit kernel term, not a total PCM or
    self-consistent solvent derivative.
    """

    points = np.asarray(points_bohr, dtype=float)
    positions = np.asarray(atom_positions_angstrom, dtype=float)
    cotangent = np.asarray(surface_cotangent, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("MEP points must have shape (n_points, 3).")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Atom positions must have shape (n_atoms, 3).")
    if cotangent.shape != (points.shape[0],):
        raise ValueError("Surface cotangent length does not match MEP points.")
    charges, dipoles_angstrom = cartesian_multipoles(density_coefficients)
    if positions.shape[0] != charges.shape[0]:
        raise ValueError("Density coefficient count does not match atom positions.")

    positions_bohr = positions / Bohr
    dipoles_bohr = dipoles_angstrom / Bohr
    position_vjp = np.empty_like(positions)
    for atom_index, (center, charge, dipole) in enumerate(
        zip(positions_bohr, charges, dipoles_bohr, strict=True)
    ):
        displacement = points - center
        radius = np.linalg.norm(displacement, axis=1)
        if np.any(radius <= 1.0e-14):
            raise ValueError("A PCM surface point coincides with an atomic centre.")
        dipole_projection = np.einsum("ij,j->i", displacement, dipole)
        derivative_bohr = (
            (charge / radius**3)[:, None] * displacement
            - dipole[None, :] / radius[:, None] ** 3
            + (3.0 * dipole_projection / radius**5)[:, None] * displacement
        )
        position_vjp[atom_index] = np.einsum(
            "s,si->i", cotangent, derivative_bohr
        ) / Bohr
    return position_vjp


def point_multipole_potential_surface_position_vjp(
    points_bohr: np.ndarray,
    atom_positions_angstrom: np.ndarray,
    density_coefficients: np.ndarray,
    surface_cotangent: np.ndarray,
) -> np.ndarray:
    """Contract the point-multipole MEP derivative over surface coordinates.

    Atom positions and density coefficients are held fixed.  This evaluates
    ``(dV_surface/dS).T @ surface_cotangent`` directly for the surface-point
    coordinates ``S`` supplied in bohr.  The returned array therefore has shape
    ``(n_surface, 3)`` and is differentiated per bohr.  It is the moving-node
    counterpart of :func:`point_multipole_potential_position_vjp`, not a
    continuum-operator or total solvent derivative.
    """

    points = np.asarray(points_bohr, dtype=float)
    positions = np.asarray(atom_positions_angstrom, dtype=float)
    cotangent = np.asarray(surface_cotangent, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("MEP points must be finite with shape (n_points, 3).")
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("Atom positions must be finite with shape (n_atoms, 3).")
    if cotangent.shape != (points.shape[0],):
        raise ValueError("Surface cotangent length does not match MEP points.")
    if not np.all(np.isfinite(cotangent)):
        raise ValueError("Surface cotangent must be finite.")
    charges, dipoles_angstrom = cartesian_multipoles(density_coefficients)
    if positions.shape[0] != charges.shape[0]:
        raise ValueError("Density coefficient count does not match atom positions.")

    positions_bohr = positions / Bohr
    dipoles_bohr = dipoles_angstrom / Bohr
    surface_position_vjp = np.zeros_like(points)
    for center, charge, dipole in zip(
        positions_bohr,
        charges,
        dipoles_bohr,
        strict=True,
    ):
        displacement = points - center
        radius = np.linalg.norm(displacement, axis=1)
        if np.any(radius <= 1.0e-14):
            raise ValueError("A PCM surface point coincides with an atomic centre.")
        dipole_projection = np.einsum("ij,j->i", displacement, dipole)
        atom_position_derivative_bohr = (
            (charge / radius**3)[:, None] * displacement
            - dipole[None, :] / radius[:, None] ** 3
            + (3.0 * dipole_projection / radius**5)[:, None] * displacement
        )
        surface_position_vjp -= (
            cotangent[:, None] * atom_position_derivative_bohr
        )
    return surface_position_vjp


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


def point_asc_reaction_position_vjp(
    atom_positions_angstrom: np.ndarray,
    surface_centers_bohr: np.ndarray,
    apparent_surface_charges: np.ndarray,
    potential_cotangent: np.ndarray,
    gradient_cotangent: np.ndarray,
) -> np.ndarray:
    """Contract the point-ASC reaction-field position derivative.

    Surface centres and integrated ASC values are held fixed.  This evaluates
    the vector-Jacobian product for atom-centred reaction potential and
    gradient directly, without constructing a dense Hessian.  The returned
    array has shape ``(n_atoms, 3)``.  Potential cotangents are in elementary
    charge and gradient cotangents in elementary-charge bohr, giving
    Hartree/Angstrom.  This is an explicit kernel term only.
    """

    positions_bohr = np.asarray(atom_positions_angstrom, dtype=float) / Bohr
    centers = np.asarray(surface_centers_bohr, dtype=float)
    asc = np.asarray(apparent_surface_charges, dtype=float)
    potential_weights = np.asarray(potential_cotangent, dtype=float)
    gradient_weights = np.asarray(gradient_cotangent, dtype=float)
    if positions_bohr.ndim != 2 or positions_bohr.shape[1] != 3:
        raise ValueError("Atom positions must have shape (n_atoms, 3).")
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("Surface centers must have shape (n_surface, 3).")
    if asc.shape != (centers.shape[0],):
        raise ValueError("ASC vector length does not match surface centers.")
    if potential_weights.shape != (positions_bohr.shape[0],):
        raise ValueError("Potential cotangent length does not match atom positions.")
    if gradient_weights.shape != positions_bohr.shape:
        raise ValueError("Gradient cotangent shape does not match atom positions.")

    position_vjp = np.empty_like(positions_bohr)
    identity = np.eye(3)
    for atom_index, position in enumerate(positions_bohr):
        displacement = centers - position
        radius = np.linalg.norm(displacement, axis=1)
        if np.any(radius <= 1.0e-14):
            raise ValueError("A PCM surface point coincides with an atomic centre.")
        inverse_radius_cubed = asc / radius**3
        potential_derivative_bohr = np.sum(
            inverse_radius_cubed[:, None] * displacement,
            axis=0,
        )
        hessian_bohr = (
            -np.sum(inverse_radius_cubed) * identity
            + 3.0
            * np.einsum(
                "s,si,sj->ij",
                asc / radius**5,
                displacement,
                displacement,
            )
        )
        position_vjp[atom_index] = (
            potential_weights[atom_index] * potential_derivative_bohr
            + gradient_weights[atom_index] @ hessian_bohr
        ) / Bohr
    return position_vjp


def asc_reaction_potential_gradient(
    atom_positions_angstrom: np.ndarray,
    surface_centers_bohr: np.ndarray,
    apparent_surface_charges: np.ndarray,
    *,
    sigma_angstrom: float = MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the requested-width Gaussian-smoothed ASC potential and gradient.

    The returned potential is in Hartree/e and the gradient is in
    Hartree/(e bohr). ``sigma_angstrom`` selects the receiver smoothing width;
    it need not equal the MACE-POLAR source-density width. The corresponding
    finite-width ``l<=1`` reciprocity identity applies only when the solute
    source uses the same Gaussian width. The current Route-2 point-multipole
    source instead checks its energy reciprocity with the unsmoothed point-ASC
    field.

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
