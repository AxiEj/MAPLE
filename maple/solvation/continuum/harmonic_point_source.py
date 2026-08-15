"""Exterior point ``l<=1`` source map in harmonic coefficient space.

The MACE-MDP permanent source is a genuine atom-centred point monopole/dipole
source, whereas the MACE-POLAR induced increment uses the checkpoint's
Gaussian density convention.  This module keeps those kernels distinct while
projecting the point potential into the same fixed, complete real-harmonic
coefficient representation used by the smooth Galerkin continuum.

Only invariant one-dimensional radial quadrature and complete SO(3) irreps are
used.  There is no laboratory-fixed surface grid and no active point set.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from ase.units import Bohr

from maple.solvation.api.units import HARTREE_TO_EV

from .harmonic_coefficients import (
    _bounded_lmax,
    _positive_int,
    _real_harmonic_design,
    real_wigner_generators,
)
from .harmonic_exposure import _legendre_rule

HARMONIC_POINT_SOURCE_CONTRACT_ID = (
    "maple.route2.coupling.point-l1-to-harmonic-test-basis.v1"
)
HARMONIC_POINT_SOURCE_PROVIDER_ID = (
    "maple.route2.coupling.harmonic-point-source.impl.v1"
)


def harmonic_point_source_implementation_sha256() -> str:
    """Return the current implementation digest used by profile identities."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=float)
    result.setflags(write=False)
    return result


def _geometry(
    positions_angstrom: object, radii_angstrom: object
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions_angstrom, dtype=float)
    radii = np.asarray(radii_angstrom, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[0] < 1
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError(
            "positions_angstrom must be finite with shape (atom_count, 3)."
        )
    if (
        radii.shape != (len(positions),)
        or not np.all(np.isfinite(radii))
        or np.any(radii <= 0.0)
    ):
        raise ValueError(
            "radii_angstrom must be finite and positive with shape (atom_count,)."
        )
    return (
        np.ascontiguousarray(positions, dtype=float),
        np.ascontiguousarray(radii, dtype=float),
    )


def _point_kernel_and_radial_derivative(
    radius_angstrom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    radius = np.asarray(radius_angstrom, dtype=float)
    if not np.all(np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("point-source radii must be finite and strictly positive.")
    # 1 / r_bohr and its derivative with respect to r_angstrom.
    return Bohr / radius, -Bohr / (radius**2)


def _zonal_coefficients_and_distance_derivative(
    *,
    target_radius_angstrom: float,
    distance_angstrom: float,
    lmax: int,
    radial_quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    if abs(distance_angstrom - target_radius_angstrom) <= 1.0e-12:
        raise ValueError("a distinct point source lies on a target sphere.")
    cosine, weights = _legendre_rule(radial_quadrature_order)
    radius_squared = (
        target_radius_angstrom**2
        + distance_angstrom**2
        - 2.0 * target_radius_angstrom * distance_angstrom * cosine
    )
    radius = np.sqrt(np.maximum(0.0, radius_squared))
    kernel, radial_derivative = _point_kernel_and_radial_derivative(radius)
    radial_direction = (distance_angstrom - target_radius_angstrom * cosine) / radius
    legendre = np.polynomial.legendre.legvander(cosine, lmax)
    coefficients = 2.0 * np.pi * ((weights * kernel) @ legendre)
    derivatives = (
        2.0 * np.pi * ((weights * radial_derivative * radial_direction) @ legendre)
    )
    return coefficients, derivatives


def _distinct_pair_block(
    *,
    displacement_angstrom: np.ndarray,
    target_radius_angstrom: float,
    lmax: int,
    radial_quadrature_order: int,
) -> np.ndarray:
    distance = float(np.linalg.norm(displacement_angstrom))
    if distance <= 1.0e-14:
        raise ValueError("distinct point-source centres must not coincide.")
    direction = displacement_angstrom / distance
    zonal, zonal_derivative = _zonal_coefficients_and_distance_derivative(
        target_radius_angstrom=target_radius_angstrom,
        distance_angstrom=distance,
        lmax=lmax,
        radial_quadrature_order=radial_quadrature_order,
    )
    harmonics = _real_harmonic_design(direction[None, :], lmax=lmax)[0]
    generators = real_wigner_generators(lmax=lmax)
    dimension = (lmax + 1) ** 2
    coefficient = np.empty(dimension, dtype=float)
    jacobian = np.empty((dimension, 3), dtype=float)
    for ell in range(lmax + 1):
        section = slice(ell * ell, (ell + 1) * (ell + 1))
        coefficient[section] = zonal[ell] * harmonics[section]
    for axis in range(3):
        radial_change = direction[axis]
        direction_change = (np.eye(3)[axis] - direction * radial_change) / distance
        rotation_vector = np.cross(direction, direction_change)
        harmonic_change = sum(
            rotation_vector[generator_axis] * (generators[generator_axis] @ harmonics)
            for generator_axis in range(3)
        )
        for ell in range(lmax + 1):
            section = slice(ell * ell, (ell + 1) * (ell + 1))
            jacobian[section, axis] = (
                zonal_derivative[ell] * radial_change * harmonics[section]
                + zonal[ell] * harmonic_change[section]
            )
    block = np.empty((dimension, 4), dtype=float)
    block[:, 0] = coefficient
    # Authoritative raw l=1 order is (m0,m1,m-1)=(y,z,x).
    block[:, 1] = jacobian[:, 1]
    block[:, 2] = jacobian[:, 2]
    block[:, 3] = jacobian[:, 0]
    return block


def _self_pair_block(*, target_radius_angstrom: float, lmax: int) -> np.ndarray:
    kernel, radial_derivative = _point_kernel_and_radial_derivative(
        np.asarray([target_radius_angstrom], dtype=float)
    )
    block = np.zeros(((lmax + 1) ** 2, 4), dtype=float)
    block[0, 0] = np.sqrt(4.0 * np.pi) * kernel[0]
    if lmax >= 1:
        dipole_amplitude = -radial_derivative[0] * np.sqrt(4.0 * np.pi / 3.0)
        block[1:4, 1:4] = dipole_amplitude * np.eye(3)
    return block


def point_harmonic_source_operator(
    *,
    positions_angstrom: object,
    radii_angstrom: object,
    surface_lmax: int,
    radial_quadrature_order: int = 128,
) -> np.ndarray:
    """Return point ``(q,p_raw)`` to harmonic MEP coefficients in eV units."""

    positions, radii = _geometry(positions_angstrom, radii_angstrom)
    maximum = _bounded_lmax(surface_lmax)
    order = _positive_int(radial_quadrature_order, name="radial_quadrature_order")
    if order > 4096:
        raise ValueError("radial_quadrature_order exceeds the bounded contract.")
    atom_count = len(positions)
    block_dimension = (maximum + 1) ** 2
    operator = np.zeros((atom_count * block_dimension, atom_count * 4), dtype=float)
    for target in range(atom_count):
        row = slice(target * block_dimension, (target + 1) * block_dimension)
        for source in range(atom_count):
            displacement = positions[source] - positions[target]
            if target == source:
                block = _self_pair_block(
                    target_radius_angstrom=float(radii[target]), lmax=maximum
                )
            else:
                block = _distinct_pair_block(
                    displacement_angstrom=displacement,
                    target_radius_angstrom=float(radii[target]),
                    lmax=maximum,
                    radial_quadrature_order=order,
                )
            operator[row, source * 4 : (source + 1) * 4] = HARTREE_TO_EV * block
    return _readonly(operator)


__all__ = [
    "HARMONIC_POINT_SOURCE_CONTRACT_ID",
    "HARMONIC_POINT_SOURCE_PROVIDER_ID",
    "harmonic_point_source_implementation_sha256",
    "point_harmonic_source_operator",
]
