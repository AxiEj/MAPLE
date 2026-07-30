"""Open-domain cubic B-spline deposition for isolated Route-2 V0 sources.

The periodic deposition map is appropriate for bulk liquid controls, but an
isolated molecular AO/grid calculation must not wrap a nuclear source through
the open boundary.  This module uses the same nonnegative cardinal cubic
B-spline basis on a regular grid and fails closed when its four-point support
would leave the declared box.  A physical calculation must therefore enlarge
its zero-boundary buffer rather than silently clip, wrap, or renormalise a
nuclear charge.

The stencil preserves the exact discrete source-count and field/deposition
adjoint identities while the support remains in the box.  It supplies no
electronic functional, solvent asset, force/PES certificate, or accuracy
result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .route2_v0_structured_solvent import RegularCartesianGrid

V0_OPEN_CUBIC_BSPLINE_CONSTRUCTION = "route2-v0-open-cubic-bspline-v1"
V0_OPEN_CUBIC_BSPLINE_STENCIL_SIZE = 64


def _immutable_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    integer: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    """Return one finite immutable array with its exact declared shape."""

    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        qualifier = "real integer-valued" if integer else "real-valued"
        raise ValueError(f"{name} must be {qualifier}.")
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric.") from error
    if numeric.shape != shape or not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    if integer:
        rounded = np.rint(numeric)
        if not np.array_equal(numeric, rounded):
            raise ValueError(f"{name} must be integer-valued.")
        result = rounded.astype(np.int64, copy=True)
    else:
        if nonnegative and np.any(numeric < 0.0):
            raise ValueError(f"{name} must be nonnegative.")
        result = np.array(numeric, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _positions(values: np.ndarray) -> np.ndarray:
    """Validate one nonempty Cartesian position matrix in Bohr."""

    if np.iscomplexobj(values):
        raise ValueError("Open cubic B-spline positions must be real-valued.")
    positions = np.asarray(values, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[0] == 0
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError(
            "Open cubic B-spline positions must be finite with shape (n_points, 3)."
        )
    result = np.array(positions, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _cubic_weights(fractional: np.ndarray) -> np.ndarray:
    """Return the four cardinal cubic B-spline weights at each fraction."""

    values = np.asarray(fractional, dtype=float)
    if (
        values.ndim != 1
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values >= 1.0)
    ):
        raise ValueError("Open cubic B-spline fractions must lie in [0, 1).")
    one_minus = 1.0 - values
    result = np.stack(
        (
            one_minus**3 / 6.0,
            (3.0 * values**3 - 6.0 * values**2 + 4.0) / 6.0,
            (-3.0 * values**3 + 3.0 * values**2 + 3.0 * values + 1.0) / 6.0,
            values**3 / 6.0,
        ),
        axis=1,
    )
    if np.any(result < -64.0 * np.finfo(float).eps) or not np.allclose(
        np.sum(result, axis=1),
        1.0,
        rtol=0.0,
        atol=64.0 * np.finfo(float).eps,
    ):
        raise RuntimeError("Open cubic B-spline weights violate partition of unity.")
    return np.maximum(result, 0.0)


def _stencil_arrays(
    *,
    grid: RegularCartesianGrid,
    positions_bohr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return non-wrapping cubic B-spline entries for in-box source positions."""

    positions = _positions(positions_bohr)
    scaled = (positions - grid.origin_bohr) / grid.spacing_bohr
    if not np.all(np.isfinite(scaled)):
        raise ValueError("Open cubic B-spline positions overflow grid coordinates.")
    lower = np.floor(scaled)
    if np.any(np.abs(lower) > np.iinfo(np.int64).max - 2):
        raise ValueError("Open cubic B-spline positions exceed integer grid range.")
    lower_indices = lower.astype(np.int64)
    fractional = scaled - lower_indices
    carry = fractional >= 1.0
    if np.any(carry):
        lower_indices[carry] += 1
        fractional[carry] -= 1.0
    if np.any(fractional < 0.0) or np.any(fractional >= 1.0):
        raise RuntimeError("Open cubic B-spline coordinate decomposition is invalid.")

    offsets = np.array((-1, 0, 1, 2), dtype=np.int64)
    axis_indices: list[np.ndarray] = []
    axis_weights: list[np.ndarray] = []
    for axis, size in enumerate(grid.shape):
        indices = lower_indices[:, axis, None] + offsets
        if np.any(indices < 0) or np.any(indices >= size):
            raise ValueError(
                "Open cubic B-spline support leaves the declared box; enlarge the "
                "open-boundary buffer instead of clipping or wrapping the source."
            )
        axis_indices.append(indices)
        axis_weights.append(_cubic_weights(fractional[:, axis]))

    count = positions.shape[0]
    node_indices = np.empty((count, V0_OPEN_CUBIC_BSPLINE_STENCIL_SIZE), dtype=np.int64)
    weights = np.empty((count, V0_OPEN_CUBIC_BSPLINE_STENCIL_SIZE), dtype=float)
    column = 0
    for x_index in range(4):
        for y_index in range(4):
            for z_index in range(4):
                node_indices[:, column] = np.ravel_multi_index(
                    (
                        axis_indices[0][:, x_index],
                        axis_indices[1][:, y_index],
                        axis_indices[2][:, z_index],
                    ),
                    grid.shape,
                )
                weights[:, column] = (
                    axis_weights[0][:, x_index]
                    * axis_weights[1][:, y_index]
                    * axis_weights[2][:, z_index]
                )
                column += 1
    if column != V0_OPEN_CUBIC_BSPLINE_STENCIL_SIZE:
        raise RuntimeError("Open cubic B-spline stencil size is inconsistent.")
    return node_indices, weights


@dataclass(frozen=True)
class Route2V0OpenCubicBSplineStencil:
    """Compact non-wrapping cubic B-spline deposition and exact grid adjoint."""

    grid: RegularCartesianGrid
    positions_bohr: np.ndarray
    node_linear_indices: np.ndarray
    weights: np.ndarray
    construction: str = V0_OPEN_CUBIC_BSPLINE_CONSTRUCTION
    _point_count: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.grid, RegularCartesianGrid):
            raise TypeError("Open cubic B-spline stencil requires a regular grid.")
        if self.construction != V0_OPEN_CUBIC_BSPLINE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 open cubic B-spline construction.")
        positions = _positions(self.positions_bohr)
        count = positions.shape[0]
        indices = _immutable_array(
            self.node_linear_indices,
            name="Open cubic B-spline node indices",
            shape=(count, V0_OPEN_CUBIC_BSPLINE_STENCIL_SIZE),
            integer=True,
        )
        if np.any(indices < 0) or np.any(indices >= self.grid.point_count):
            raise ValueError("Open cubic B-spline node indices leave the grid.")
        weights = _immutable_array(
            self.weights,
            name="Open cubic B-spline weights",
            shape=(count, V0_OPEN_CUBIC_BSPLINE_STENCIL_SIZE),
            nonnegative=True,
        )
        if not np.allclose(
            np.sum(weights, axis=1),
            1.0,
            rtol=0.0,
            atol=128.0 * np.finfo(float).eps,
        ):
            raise ValueError("Open cubic B-spline weights must sum to one.")
        expected_indices, expected_weights = _stencil_arrays(
            grid=self.grid,
            positions_bohr=positions,
        )
        if not np.array_equal(indices, expected_indices) or not np.allclose(
            weights,
            expected_weights,
            rtol=0.0,
            atol=128.0 * np.finfo(float).eps,
        ):
            raise ValueError(
                "Open cubic B-spline stencil does not match its declared grid and "
                "positions."
            )
        object.__setattr__(self, "positions_bohr", positions)
        object.__setattr__(self, "node_linear_indices", indices)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "_point_count", count)

    @property
    def point_count(self) -> int:
        """Return the number of deposited point sources."""

        return self._point_count

    def deposit(self, values: np.ndarray) -> np.ndarray:
        """Return the exact open-grid deposition ``D values``."""

        coefficients = _immutable_array(
            values,
            name="Open cubic B-spline deposition values",
            shape=(self.point_count,),
        )
        result = np.zeros(self.grid.point_count, dtype=float)
        np.add.at(
            result,
            self.node_linear_indices.reshape(-1),
            (self.weights * coefficients[:, None]).reshape(-1),
        )
        field = result.reshape(self.grid.shape)
        field.setflags(write=False)
        return field

    def field_adjoint(self, field: np.ndarray) -> np.ndarray:
        """Return the exact Euclidean grid/deposition adjoint ``D.T field``."""

        values = _immutable_array(
            field,
            name="Open cubic B-spline grid field",
            shape=self.grid.shape,
        ).reshape(self.grid.point_count)
        result = np.sum(self.weights * values[self.node_linear_indices], axis=1)
        result.setflags(write=False)
        return result


def build_route2_v0_open_cubic_bspline_stencil(
    *,
    grid: RegularCartesianGrid,
    positions_bohr: np.ndarray,
) -> Route2V0OpenCubicBSplineStencil:
    """Build the non-wrapping cubic B-spline source map for one open box."""

    if not isinstance(grid, RegularCartesianGrid):
        raise TypeError("Open cubic B-spline construction requires a regular grid.")
    positions = _positions(positions_bohr)
    node_indices, weights = _stencil_arrays(grid=grid, positions_bohr=positions)
    return Route2V0OpenCubicBSplineStencil(
        grid=grid,
        positions_bohr=positions,
        node_linear_indices=node_indices,
        weights=weights,
    )


__all__ = [
    "V0_OPEN_CUBIC_BSPLINE_CONSTRUCTION",
    "V0_OPEN_CUBIC_BSPLINE_STENCIL_SIZE",
    "Route2V0OpenCubicBSplineStencil",
    "build_route2_v0_open_cubic_bspline_stencil",
]
