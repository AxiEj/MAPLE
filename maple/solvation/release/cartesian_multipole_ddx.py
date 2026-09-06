"""Exact numerical basis change from raw Cartesian q/p/Q/O to ddX l<=3.

SPICE stores raw atom-centred Cartesian moments, whereas ddX consumes real
spherical multipoles.  This module derives the *linear basis transformation*
from the two independently implemented exterior potentials on a one-sphere
Lebedev grid.  It is not a regression or model fit: both matrices are fixed
analytic basis functions, and the solved change of basis is required to replay
the Cartesian potential to floating-point precision on unrelated geometries.

The optional pyddx dependency is imported lazily so dependency-light MAPLE
modules remain importable without the native continuum runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib

import numpy as np

from .cartesian_multipole_mep import cartesian_atomic_multipole_potential

_CARTESIAN_COMPONENT_COUNT = 1 + 3 + 9 + 27
_PYDDX_L3_COMPONENT_COUNT = 16


def _runtime():
    try:
        module = importlib.import_module("pyddx")
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("The Cartesian l<=3 oracle requires pyddx.") from exc
    if str(getattr(module, "__version__", "unknown")) != "0.8.0":
        raise RuntimeError("The Cartesian l<=3 oracle is frozen to pyddx 0.8.0.")
    return module


def _finite(values: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return np.ascontiguousarray(array)


@dataclass(frozen=True, slots=True)
class CartesianToPyDDXTransform:
    """One validated, immutable Cartesian-to-real-spherical basis change."""

    values: np.ndarray
    collocation_rank: int
    collocation_condition_number: float
    maximum_collocation_residual: float

    def __post_init__(self) -> None:
        values = _finite(
            self.values,
            name="Cartesian-to-pyDDX transform",
            shape=(_PYDDX_L3_COMPONENT_COUNT, _CARTESIAN_COMPONENT_COUNT),
        )
        if self.collocation_rank != _PYDDX_L3_COMPONENT_COUNT:
            raise ValueError("Cartesian-to-pyDDX collocation is rank deficient.")
        condition = float(self.collocation_condition_number)
        residual = float(self.maximum_collocation_residual)
        if (
            not np.isfinite(condition)
            or condition > 1.0e4
            or not np.isfinite(residual)
            or residual > 5.0e-13
        ):
            raise ValueError("Cartesian-to-pyDDX basis change is numerically invalid.")
        readonly = np.frombuffer(values.tobytes(), dtype=np.float64).reshape(
            values.shape
        )
        object.__setattr__(self, "values", readonly)
        object.__setattr__(self, "collocation_condition_number", condition)
        object.__setattr__(self, "maximum_collocation_residual", residual)


def _cartesian_unit_potential(points_bohr: np.ndarray, component: int) -> np.ndarray:
    charge = np.zeros(1)
    dipole = np.zeros((1, 3))
    quadrupole = np.zeros((1, 3, 3))
    octupole = np.zeros((1, 3, 3, 3))
    if component == 0:
        charge[0] = 1.0
    elif component < 4:
        dipole.reshape(-1)[component - 1] = 1.0
    elif component < 13:
        quadrupole.reshape(-1)[component - 4] = 1.0
    else:
        octupole.reshape(-1)[component - 13] = 1.0
    return cartesian_atomic_multipole_potential(
        points_bohr=points_bohr,
        centers_angstrom=np.zeros((1, 3)),
        charges_e=charge,
        dipoles_eangstrom=dipole,
        quadrupoles_eangstrom2=quadrupole,
        octupoles_eangstrom3=octupole,
    )


@lru_cache(maxsize=1)
def cartesian_to_pyddx_l3_transform() -> CartesianToPyDDXTransform:
    """Derive and cache the target-independent l<=3 basis transformation."""

    pyddx = _runtime()
    model = pyddx.Model(
        "pcm",
        np.zeros((3, 1)),
        np.asarray([4.0]),
        78.39,
        lmax=8,
        n_lebedev=194,
        enable_fmm=False,
        n_proc=1,
    )
    points = np.asarray(model.cavity, dtype=np.float64).T
    spherical = np.empty((len(points), _PYDDX_L3_COMPONENT_COUNT))
    for component in range(_PYDDX_L3_COMPONENT_COUNT):
        multipoles = np.zeros((_PYDDX_L3_COMPONENT_COUNT, 1))
        multipoles[component, 0] = 1.0
        spherical[:, component] = np.asarray(
            model.multipole_electrostatics(multipoles, derivative_order=0)["phi"],
            dtype=np.float64,
        )
    cartesian = np.column_stack(
        [
            _cartesian_unit_potential(points, component)
            for component in range(_CARTESIAN_COMPONENT_COUNT)
        ]
    )
    transform, _residuals, rank, singular_values = np.linalg.lstsq(
        spherical,
        cartesian,
        rcond=None,
    )
    condition = float(singular_values[0] / singular_values[-1])
    residual = float(np.max(np.abs(spherical @ transform - cartesian)))
    return CartesianToPyDDXTransform(
        values=transform,
        collocation_rank=int(rank),
        collocation_condition_number=condition,
        maximum_collocation_residual=residual,
    )


def cartesian_atomic_multipoles_to_pyddx_l3(
    *,
    charges_e: object,
    dipoles_eangstrom: object,
    quadrupoles_eangstrom2: object,
    octupoles_eangstrom3: object,
) -> np.ndarray:
    """Return ddX real-spherical multipoles with shape ``(16,N)``."""

    charges = np.asarray(charges_e, dtype=np.float64)
    if charges.ndim != 1 or len(charges) < 1 or not np.all(np.isfinite(charges)):
        raise ValueError("charges_e must be a finite nonempty vector.")
    count = len(charges)
    dipoles = _finite(
        dipoles_eangstrom,
        name="dipoles_eangstrom",
        shape=(count, 3),
    )
    quadrupoles = _finite(
        quadrupoles_eangstrom2,
        name="quadrupoles_eangstrom2",
        shape=(count, 3, 3),
    )
    octupoles = _finite(
        octupoles_eangstrom3,
        name="octupoles_eangstrom3",
        shape=(count, 3, 3, 3),
    )
    cartesian = np.concatenate(
        (
            charges[:, None],
            dipoles,
            quadrupoles.reshape(count, 9),
            octupoles.reshape(count, 27),
        ),
        axis=1,
    )
    result = cartesian_to_pyddx_l3_transform().values @ cartesian.T
    if result.shape != (_PYDDX_L3_COMPONENT_COUNT, count) or not np.all(
        np.isfinite(result)
    ):
        raise RuntimeError("Converted pyddx l<=3 multipoles are invalid.")
    return np.ascontiguousarray(result)


__all__ = [
    "CartesianToPyDDXTransform",
    "cartesian_atomic_multipoles_to_pyddx_l3",
    "cartesian_to_pyddx_l3_transform",
]
