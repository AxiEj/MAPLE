"""Analytic coordinate pullbacks for the legacy normalized Gaussian MEP.

This module extends the immutable legacy forward kernel without modifying its
P0-certified source file. It owns only derivative contractions needed by the
vNext conjugate radial-GTO operator.
"""

from __future__ import annotations

import math

import numpy as np
from ase.units import Bohr
from scipy.special import erf

from maple.function.calculator.extra_correction.implicit.gto_density import (
    MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
    cartesian_multipoles,
)


def gaussian_multipole_potential_displacement_gradient(
    displacement_bohr: np.ndarray,
    charge: float,
    dipole_bohr: np.ndarray,
    *,
    sigma_bohr: float,
) -> np.ndarray:
    """Return ``d V / d displacement`` for one normalized Gaussian multipole."""

    displacement = np.asarray(displacement_bohr, dtype=float)
    dipole = np.asarray(dipole_bohr, dtype=float)
    if (
        displacement.ndim != 2
        or displacement.shape[1] != 3
        or not np.all(np.isfinite(displacement))
    ):
        raise ValueError("Gaussian displacement must be finite with shape (n, 3).")
    if dipole.shape != (3,) or not np.all(np.isfinite(dipole)):
        raise ValueError("Gaussian dipole must be finite with shape (3,).")
    if not math.isfinite(charge) or not math.isfinite(sigma_bohr) or sigma_bohr <= 0:
        raise ValueError("Gaussian charge/width must be finite and width positive.")

    radius = np.linalg.norm(displacement, axis=1)
    scaled_radius = radius / sigma_bohr
    small = scaled_radius <= 1.0e-4
    regular = ~small
    root_two_over_pi = math.sqrt(2.0 / math.pi)
    dipole_kernel = np.empty_like(radius)
    radial_coefficient = np.empty_like(radius)

    if np.any(regular):
        regular_radius = radius[regular]
        x = regular_radius / (math.sqrt(2.0) * sigma_bohr)
        exponential = np.exp(-(x**2))
        b = root_two_over_pi * exponential / sigma_bohr
        dipole_kernel[regular] = (erf(x) - b * regular_radius) / regular_radius**3
        radial_coefficient[regular] = (
            b / sigma_bohr**2 - 3.0 * dipole_kernel[regular]
        ) / regular_radius**2

    if np.any(small):
        t2 = scaled_radius[small] ** 2
        dipole_kernel[small] = (
            root_two_over_pi / sigma_bohr**3 * (1.0 / 3.0 - t2 / 10.0 + t2**2 / 56.0)
        )
        radial_coefficient[small] = (
            root_two_over_pi / sigma_bohr**5 * (-1.0 / 5.0 + t2 / 14.0)
        )

    projection = np.einsum("si,i->s", displacement, dipole)
    return (
        -charge * dipole_kernel[:, None] * displacement
        + dipole_kernel[:, None] * dipole[None, :]
        + (projection * radial_coefficient)[:, None] * displacement
    )


def _validated_inputs(
    points_bohr: object,
    atom_positions_angstrom: object,
    density_coefficients: object,
    surface_cotangent: object,
    sigma_angstrom: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    points = np.asarray(points_bohr, dtype=float)
    positions = np.asarray(atom_positions_angstrom, dtype=float)
    cotangent = np.asarray(surface_cotangent, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("MEP points must be finite with shape (n_points, 3).")
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("Atom positions must be finite with shape (n_atoms, 3).")
    if cotangent.shape != (points.shape[0],) or not np.all(np.isfinite(cotangent)):
        raise ValueError("Surface cotangent must be finite with one value per point.")
    if not math.isfinite(sigma_angstrom) or sigma_angstrom <= 0.0:
        raise ValueError("The Gaussian density width must be finite and positive.")
    charges, dipoles_angstrom = cartesian_multipoles(density_coefficients)
    if positions.shape[0] != charges.shape[0]:
        raise ValueError("Density coefficient count does not match atom positions.")
    return points, positions, charges, dipoles_angstrom, float(sigma_angstrom)


def gaussian_multipole_potential_position_vjp(
    points_bohr: np.ndarray,
    atom_positions_angstrom: np.ndarray,
    density_coefficients: np.ndarray,
    surface_cotangent: np.ndarray,
    *,
    sigma_angstrom: float = MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
) -> np.ndarray:
    """Contract the atom-centre derivative at fixed surface points, per A."""

    points, positions, charges, dipoles_angstrom, sigma = _validated_inputs(
        points_bohr,
        atom_positions_angstrom,
        density_coefficients,
        surface_cotangent,
        sigma_angstrom,
    )
    cotangent = np.asarray(surface_cotangent, dtype=float)
    positions_bohr = positions / Bohr
    dipoles_bohr = dipoles_angstrom / Bohr
    sigma_bohr = sigma / Bohr
    result = np.empty_like(positions)
    for atom_index, (center, charge, dipole) in enumerate(
        zip(positions_bohr, charges, dipoles_bohr, strict=True)
    ):
        displacement_gradient = gaussian_multipole_potential_displacement_gradient(
            points - center, float(charge), dipole, sigma_bohr=sigma_bohr
        )
        result[atom_index] = (
            -np.einsum("s,si->i", cotangent, displacement_gradient) / Bohr
        )
    return result


def gaussian_multipole_potential_surface_position_vjp(
    points_bohr: np.ndarray,
    atom_positions_angstrom: np.ndarray,
    density_coefficients: np.ndarray,
    surface_cotangent: np.ndarray,
    *,
    sigma_angstrom: float = MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
) -> np.ndarray:
    """Contract the surface-point derivative, per bohr."""

    points, positions, charges, dipoles_angstrom, sigma = _validated_inputs(
        points_bohr,
        atom_positions_angstrom,
        density_coefficients,
        surface_cotangent,
        sigma_angstrom,
    )
    cotangent = np.asarray(surface_cotangent, dtype=float)
    positions_bohr = positions / Bohr
    dipoles_bohr = dipoles_angstrom / Bohr
    sigma_bohr = sigma / Bohr
    result = np.zeros_like(points)
    for center, charge, dipole in zip(
        positions_bohr, charges, dipoles_bohr, strict=True
    ):
        displacement_gradient = gaussian_multipole_potential_displacement_gradient(
            points - center, float(charge), dipole, sigma_bohr=sigma_bohr
        )
        result += cotangent[:, None] * displacement_gradient
    return result


__all__ = [
    "gaussian_multipole_potential_displacement_gradient",
    "gaussian_multipole_potential_position_vjp",
    "gaussian_multipole_potential_surface_position_vjp",
]
