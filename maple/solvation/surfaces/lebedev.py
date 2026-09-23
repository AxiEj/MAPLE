"""Ordered, immutable Lebedev quadrature used by Torch solvation kernels.

The node ordering is part of the numerical method: ddX 0.8.0 and PySCF
2.13.1 use the same Lebedev-Laikov tables for the supported grids.  This
module loads only the constant angular rule; it does not call either
package's cavity, PCM, or solvation implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib

import numpy as np

SUPPORTED_LEBEDEV_ORDERS = (50, 194, 1202)
LEBEDEV_GRID_CONTRACT = "lebedev-laikov-ordered-pyscf-2.13.1-ddx-0.8.0-v1"


@dataclass(frozen=True, slots=True)
class LebedevGrid:
    """One content-addressed ordered rule with weights normalized to ``4*pi``."""

    directions: np.ndarray
    weights: np.ndarray
    point_count: int
    contract: str
    sha256: str


def _array_digest(directions: np.ndarray, weights: np.ndarray) -> str:
    digest = hashlib.sha256(LEBEDEV_GRID_CONTRACT.encode())
    for array in (directions, weights):
        value = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(value.shape).encode())
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _immutable_float64(values: object, shape: tuple[int, ...]) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    if contiguous.shape != shape:
        raise ValueError(f"immutable array must have shape {shape}.")
    # A bytes object does not export a writable buffer.  Unlike merely setting
    # ndarray.flags.writeable=False, callers therefore cannot re-enable writes.
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64).reshape(shape)


@lru_cache(maxsize=len(SUPPORTED_LEBEDEV_ORDERS))
def ordered_lebedev_grid(point_count: int) -> LebedevGrid:
    """Return an immutable ddX-ordered Lebedev rule.

    PySCF's pure-Python ``MakeAngularGrid`` is used solely as the packaged
    constant-table loader.  Its weights sum to one and are converted here to
    the spherical integration convention used by ddX (sum ``4*pi``).
    """

    if isinstance(point_count, bool) or point_count not in SUPPORTED_LEBEDEV_ORDERS:
        raise ValueError(
            "point_count must be one of "
            f"{SUPPORTED_LEBEDEV_ORDERS}; received {point_count!r}."
        )
    try:
        pyscf = importlib.import_module("pyscf")
        gen_grid = importlib.import_module("pyscf.dft.gen_grid")
    except ImportError as exc:  # pragma: no cover - optional runtime boundary
        raise RuntimeError("ordered Lebedev loading requires PySCF 2.13.1.") from exc
    if str(getattr(pyscf, "__version__", "unknown")) != "2.13.1":
        raise RuntimeError("ordered Lebedev loading is pinned to PySCF 2.13.1.")

    raw = np.asarray(gen_grid.MakeAngularGrid(point_count), dtype=np.float64)
    if raw.shape != (point_count, 4) or not np.all(np.isfinite(raw)):
        raise RuntimeError("PySCF returned an invalid Lebedev table.")
    directions = _immutable_float64(raw[:, :3], (point_count, 3))
    weights = _immutable_float64(raw[:, 3] * (4.0 * np.pi), (point_count,))
    if not np.allclose(np.linalg.norm(directions, axis=1), 1.0, rtol=0.0, atol=2.0e-15):
        raise RuntimeError("Lebedev directions are not unit vectors.")
    if np.any(weights <= 0.0) or not np.isclose(
        weights.sum(), 4.0 * np.pi, rtol=0.0, atol=3.0e-14
    ):
        raise RuntimeError("Lebedev weights violate the 4*pi normalization.")
    return LebedevGrid(
        directions=directions,
        weights=weights,
        point_count=point_count,
        contract=LEBEDEV_GRID_CONTRACT,
        sha256=_array_digest(directions, weights),
    )


__all__ = [
    "LEBEDEV_GRID_CONTRACT",
    "SUPPORTED_LEBEDEV_ORDERS",
    "LebedevGrid",
    "ordered_lebedev_grid",
]
