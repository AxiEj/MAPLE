"""Finite complete real-harmonic coefficient representations for Route 2.

The only quadrature in this module contracts a known finite band-limited
representation exactly.  It never samples a cavity mask, overlap field, or
continuum surface operator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
from scipy.special import lpmv

HARMONIC_GALERKIN_COEFFICIENT_CONTRACT_ID = (
    "maple.route2.harmonic-coefficients.complete-real-irreps.v1"
)
HARMONIC_GALERKIN_MAXIMUM_LMAX = 16


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _positive_int(value: object, *, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    minimum = 0 if allow_zero else 1
    if result < minimum:
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparison}.")
    return result


def _bounded_lmax(value: object) -> int:
    maximum = _positive_int(value, name="lmax", allow_zero=True)
    if maximum > HARMONIC_GALERKIN_MAXIMUM_LMAX:
        raise ValueError(
            "lmax exceeds the bounded harmonic coefficient contract "
            f"({HARMONIC_GALERKIN_MAXIMUM_LMAX})."
        )
    return maximum


def _proper_rotation(rotation: object) -> np.ndarray:
    values = np.asarray(rotation, dtype=float)
    tolerance = 5.0e-13
    if (
        values.shape != (3, 3)
        or not np.all(np.isfinite(values))
        or not np.allclose(values @ values.T, np.eye(3), atol=tolerance, rtol=0.0)
        or not np.isclose(np.linalg.det(values), 1.0, atol=tolerance, rtol=0.0)
    ):
        raise ValueError("rotation must be a finite proper rotation matrix.")
    return np.array(values, dtype=float, copy=True)


@dataclass(frozen=True, slots=True)
class PerAtomHarmonicSpace:
    """Fixed complete real-harmonic coefficient blocks on labelled atoms."""

    atom_count: int
    lmax: int
    coefficient_contract_id: str = HARMONIC_GALERKIN_COEFFICIENT_CONTRACT_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "atom_count", _positive_int(self.atom_count, name="atom_count")
        )
        maximum = _bounded_lmax(self.lmax)
        object.__setattr__(self, "lmax", maximum)
        if (
            not isinstance(self.coefficient_contract_id, str)
            or not self.coefficient_contract_id.strip()
        ):
            raise ValueError("coefficient_contract_id must be a non-empty string.")
        object.__setattr__(
            self, "coefficient_contract_id", self.coefficient_contract_id.strip()
        )

    @property
    def single_atom_dimension(self) -> int:
        return (self.lmax + 1) ** 2

    @property
    def dimension(self) -> int:
        return self.atom_count * self.single_atom_dimension

    @property
    def labels(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (atom, ell, order)
            for atom in range(self.atom_count)
            for ell in range(self.lmax + 1)
            for order in range(-ell, ell + 1)
        )

    def metadata(self) -> dict[str, object]:
        return {
            "coefficient_contract_id": self.coefficient_contract_id,
            "basis": "orthonormal-real-spherical-harmonics-condon-shortley",
            "atom_count": self.atom_count,
            "lmax": self.lmax,
            "maximum_lmax": HARMONIC_GALERKIN_MAXIMUM_LMAX,
            "single_atom_dimension": self.single_atom_dimension,
            "dimension": self.dimension,
            "ordering": "atom-major,l-major,m=-l..l",
            "representation": "complete SO(3) irreps for every retained l",
        }

    def metadata_sha256(self) -> str:
        return _sha(self.metadata())

    def representation_matrix(self, rotation: object) -> np.ndarray:
        single_atom = real_wigner_matrix(rotation, lmax=self.lmax)
        result = np.kron(np.eye(self.atom_count), single_atom)
        result.setflags(write=False)
        return result


def _exact_bandlimited_sphere_rule(lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a rule exact for harmonics through degree ``2*lmax``."""

    maximum = _bounded_lmax(lmax)
    polar_order = maximum + 1
    azimuthal_order = 2 * maximum + 1
    cos_theta, polar_weights = np.polynomial.legendre.leggauss(polar_order)
    phi = 2.0 * np.pi * np.arange(azimuthal_order) / azimuthal_order
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    directions = np.stack(
        (
            np.repeat(sin_theta, azimuthal_order) * np.tile(np.cos(phi), polar_order),
            np.repeat(sin_theta, azimuthal_order) * np.tile(np.sin(phi), polar_order),
            np.repeat(cos_theta, azimuthal_order),
        ),
        axis=1,
    )
    weights = np.repeat(
        polar_weights * (2.0 * np.pi / azimuthal_order), azimuthal_order
    )
    return directions, weights


def _real_harmonic_design(directions: object, *, lmax: int) -> np.ndarray:
    values = np.asarray(directions, dtype=float)
    maximum = _bounded_lmax(lmax)
    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] != 3
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("directions must be finite with shape (point_count, 3).")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, np.ones_like(norms), atol=5.0e-13, rtol=0.0):
        raise ValueError("directions must lie on the unit sphere.")
    theta = np.arccos(np.clip(values[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(values[:, 1], values[:, 0]), 2.0 * np.pi)
    columns: list[np.ndarray] = []
    for ell in range(maximum + 1):
        for order in range(-ell, ell + 1):
            if order < 0:
                complex_values = _complex_spherical_harmonic(ell, -order, theta, phi)
                column = np.sqrt(2.0) * ((-1) ** order) * complex_values.imag
            elif order == 0:
                column = _complex_spherical_harmonic(ell, 0, theta, phi).real
            else:
                complex_values = _complex_spherical_harmonic(ell, order, theta, phi)
                column = np.sqrt(2.0) * ((-1) ** order) * complex_values.real
            columns.append(np.asarray(column, dtype=float))
    return np.column_stack(columns)


def _complex_spherical_harmonic(
    ell: int, order: int, theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """SciPy-1.7-compatible orthonormal Condon-Shortley harmonic."""

    log_ratio = math.lgamma(ell - order + 1) - math.lgamma(ell + order + 1)
    normalization = math.sqrt((2 * ell + 1) * math.exp(log_ratio) / (4.0 * np.pi))
    associated_legendre = lpmv(order, ell, np.cos(theta))
    return normalization * associated_legendre * np.exp(1j * order * phi)


def real_wigner_matrix(rotation: object, *, lmax: int) -> np.ndarray:
    """Return the real coefficient action of ``f(u) -> f(Q^-1 u)``.

    Complete irrep blocks are contracted independently, so cross-``l``
    entries are structurally zero rather than numerical projection noise.
    """

    transform = _proper_rotation(rotation)
    maximum = _bounded_lmax(lmax)
    directions, weights = _exact_bandlimited_sphere_rule(maximum)
    base = _real_harmonic_design(directions, lmax=maximum)
    # Row-vector directions transform as (Q^-1 u)^T = u^T Q.
    rotated = _real_harmonic_design(directions @ transform, lmax=maximum)
    dimension = (maximum + 1) ** 2
    result = np.zeros((dimension, dimension), dtype=float)
    offset = 0
    for ell in range(maximum + 1):
        width = 2 * ell + 1
        section = slice(offset, offset + width)
        result[section, section] = base[:, section].T @ (
            weights[:, None] * rotated[:, section]
        )
        offset += width
    tolerance = 2.0e-12 * (maximum + 1)
    if not np.allclose(
        result @ result.T,
        np.eye(dimension),
        atol=tolerance,
        rtol=0.0,
    ):
        raise RuntimeError("finite harmonic rotation contraction lost orthogonality.")
    result.setflags(write=False)
    return result


def radial_gto_source_rotation_matrix(
    rotation: object, *, atom_count: int
) -> np.ndarray:
    """Return the active rotation in the authoritative raw eight-channel order."""

    transform = _proper_rotation(rotation)
    count = _positive_int(atom_count, name="atom_count")
    block = np.zeros((8, 8), dtype=float)
    block[0, 0] = 1.0
    block[1, 1] = 1.0
    # Raw l=1 order is (m0,m1,m-1); Cartesian is (m-1,m0,m1)=(x,y,z).
    raw_to_cartesian = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    raw_rotation = raw_to_cartesian.T @ transform @ raw_to_cartesian
    block[2:5, 2:5] = raw_rotation
    block[5:8, 5:8] = raw_rotation
    result = np.kron(np.eye(count), block)
    result.setflags(write=False)
    return result


__all__ = [
    "HARMONIC_GALERKIN_COEFFICIENT_CONTRACT_ID",
    "HARMONIC_GALERKIN_MAXIMUM_LMAX",
    "PerAtomHarmonicSpace",
    "radial_gto_source_rotation_matrix",
    "real_wigner_matrix",
]
