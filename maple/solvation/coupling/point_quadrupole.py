"""Orthonormal Cartesian representation of atom-centred traceless quadrupoles.

The five fixed basis tensors span the real l=2 irreducible representation.
Coefficients carry physical raw quadrupole units ``e*angstrom^2`` and the
surface operator returns Hartree per positive test charge.  This is an analytic
source-map primitive; it does not by itself define a trained head, continuum,
or admitted capability.
"""

from __future__ import annotations

import numpy as np
from ase.units import Bohr


_SQRT2 = np.sqrt(2.0)
_SQRT6 = np.sqrt(6.0)
TRACELESS_QUADRUPOLE_BASIS = np.asarray(
    [
        np.diag((1.0, -1.0, 0.0)) / _SQRT2,
        np.diag((1.0, 1.0, -2.0)) / _SQRT6,
        np.asarray(((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        / _SQRT2,
        np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        / _SQRT2,
        np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
        / _SQRT2,
    ],
    dtype=np.float64,
)
TRACELESS_QUADRUPOLE_BASIS.setflags(write=False)


def traceless_quadrupole_tensors(coefficients_eangstrom2: object) -> np.ndarray:
    coefficients = np.asarray(coefficients_eangstrom2, dtype=np.float64)
    if (
        coefficients.ndim != 2
        or coefficients.shape[1] != 5
        or not np.all(np.isfinite(coefficients))
    ):
        raise ValueError("quadrupole coefficients must have finite shape (N,5).")
    return np.einsum(
        "na,aij->nij",
        coefficients,
        TRACELESS_QUADRUPOLE_BASIS,
        optimize=True,
    )


def traceless_quadrupole_coefficients(tensors_eangstrom2: object) -> np.ndarray:
    tensors = np.asarray(tensors_eangstrom2, dtype=np.float64)
    if (
        tensors.ndim != 3
        or tensors.shape[1:] != (3, 3)
        or not np.all(np.isfinite(tensors))
    ):
        raise ValueError("quadrupole tensors must have finite shape (N,3,3).")
    symmetric = 0.5 * (tensors + tensors.swapaxes(-1, -2))
    traceless = symmetric - np.trace(symmetric, axis1=1, axis2=2)[:, None, None] * (
        np.eye(3)[None, :, :] / 3.0
    )
    return np.einsum(
        "nij,aij->na",
        traceless,
        TRACELESS_QUADRUPOLE_BASIS,
        optimize=True,
    )


def quadrupole_rotation_matrix(rotation: object) -> np.ndarray:
    values = np.asarray(rotation, dtype=np.float64)
    if (
        values.shape != (3, 3)
        or not np.all(np.isfinite(values))
        or not np.allclose(values @ values.T, np.eye(3), atol=2.0e-12)
        or not np.isclose(np.linalg.det(values), 1.0, atol=2.0e-12)
    ):
        raise ValueError("rotation must be a finite proper orthogonal matrix.")
    rotated = np.einsum(
        "ij,ajk,lk->ail",
        values,
        TRACELESS_QUADRUPOLE_BASIS,
        values,
        optimize=True,
    )
    result = np.einsum(
        "aij,bij->ab",
        TRACELESS_QUADRUPOLE_BASIS,
        rotated,
        optimize=True,
    )
    return np.asarray(result, dtype=np.float64)


def point_traceless_quadrupole_surface_operator(
    *,
    points_bohr: object,
    centers_angstrom: object,
    minimum_distance_bohr: float = 1.0e-8,
) -> np.ndarray:
    """Return ``B`` such that ``MEP = B @ coefficients.reshape(-1)``."""

    points = np.asarray(points_bohr, dtype=np.float64)
    centers = np.asarray(centers_angstrom, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("points_bohr must have finite nonempty shape (M,3).")
    if (
        centers.ndim != 2
        or centers.shape[0] == 0
        or centers.shape[1] != 3
        or not np.all(np.isfinite(centers))
    ):
        raise ValueError("centers_angstrom must have finite nonempty shape (N,3).")
    threshold = float(minimum_distance_bohr)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("minimum_distance_bohr must be finite and positive.")
    displacement = points[:, None, :] - centers[None, :, :] / Bohr
    radius2 = np.einsum("pni,pni->pn", displacement, displacement)
    if np.any(radius2 < threshold * threshold):
        raise ValueError("A surface point is too close to a quadrupole centre.")
    contraction = np.einsum(
        "pni,aij,pnj->pna",
        displacement,
        TRACELESS_QUADRUPOLE_BASIS,
        displacement,
        optimize=True,
    )
    # Every basis tensor is traceless, so the delta_ij term vanishes.  Divide
    # by Bohr^2 because public coefficients are in e*angstrom^2.
    operator = 1.5 * contraction / radius2[:, :, None] ** 2.5 / Bohr**2
    result = np.ascontiguousarray(operator.reshape(len(points), -1))
    if not np.all(np.isfinite(result)):
        raise RuntimeError("quadrupole surface operator is nonfinite.")
    return result


__all__ = [
    "TRACELESS_QUADRUPOLE_BASIS",
    "point_traceless_quadrupole_surface_operator",
    "quadrupole_rotation_matrix",
    "traceless_quadrupole_coefficients",
    "traceless_quadrupole_tensors",
]
