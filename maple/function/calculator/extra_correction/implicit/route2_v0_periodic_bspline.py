"""Periodic cubic B-spline deposition for Route-2 V0 liquid controls.

The molecular liquid scalar needs one fixed map from rigid molecular positions
to a periodic Cartesian grid.  A nearest-grid-point map would make that map
discontinuous, while a cloud-in-cell map is only ``C0`` in a moving position.
This module instead uses the nonnegative cardinal cubic B-spline stencil.  It
has four nodes per Cartesian direction, is a partition of unity, and is
``C2`` as a function of every deposited position.  Consequently it preserves
both the discrete particle-count identity and the field/deposition adjoint
without introducing a physical length, a fitted switch, or a solvation label.

The compact stencil stores 64 entries per point and exposes both its dense
reference matrix and its exact forward/adjoint actions.  Existing V0 controls
still use dense occupancy tensors on deliberately small grids.  The stencil is
also the source-preserving representation a later matrix-free production
implementation must reproduce; this module itself is not a production liquid,
force, solvation-free-energy, or accuracy claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .route2_v0_structured_solvent import RegularCartesianGrid

V0_PERIODIC_CUBIC_BSPLINE_CONSTRUCTION = "route2-v0-periodic-cubic-bspline-v1"
V0_PERIODIC_CUBIC_BSPLINE_STENCIL_SIZE = 64


def _immutable_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    integer: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    """Return one finite immutable array with the declared exact shape."""

    raw = np.asarray(values)
    if integer:
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must be real integer-valued.")
        try:
            numeric = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be integer-valued.") from exc
        rounded = np.rint(numeric)
        if (
            numeric.shape != shape
            or not np.all(np.isfinite(numeric))
            or not np.array_equal(numeric, rounded)
        ):
            raise ValueError(
                f"{name} must be finite integer-valued with shape {shape}."
            )
        result = rounded.astype(np.int64, copy=True)
    else:
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must be real-valued.")
        result = np.asarray(raw, dtype=float)
        if result.shape != shape or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be finite with shape {shape}.")
        if nonnegative and np.any(result < 0.0):
            raise ValueError(f"{name} must be nonnegative.")
        result = np.array(result, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _positions(values: np.ndarray) -> np.ndarray:
    """Validate one nonempty matrix of Cartesian positions in Bohr."""

    if np.iscomplexobj(values):
        raise ValueError("Periodic B-spline positions must be real-valued.")
    positions = np.asarray(values, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[0] == 0
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError(
            "Periodic B-spline positions must be finite with shape (n_points, 3)."
        )
    result = np.array(positions, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _cubic_weights(fractional: np.ndarray) -> np.ndarray:
    """Return the four cardinal cubic B-spline weights for ``fractional``."""

    values = np.asarray(fractional, dtype=float)
    if (
        values.ndim != 1
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values >= 1.0)
    ):
        raise ValueError("Cubic B-spline fractions must lie in [0, 1).")
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
    if not np.all(np.isfinite(result)) or np.any(result < -64.0 * np.finfo(float).eps):
        raise RuntimeError("Periodic cubic B-spline weights are invalid.")
    result = np.maximum(result, 0.0)
    if not np.allclose(
        np.sum(result, axis=1),
        1.0,
        rtol=0.0,
        atol=64.0 * np.finfo(float).eps,
    ):
        raise RuntimeError(
            "Periodic cubic B-spline weights violate partition of unity."
        )
    return result


def _stencil_arrays(
    *,
    grid: RegularCartesianGrid,
    positions_bohr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the unique periodic cubic B-spline entries for valid positions."""

    positions = _positions(positions_bohr)
    scaled = (positions - grid.origin_bohr) / grid.spacing_bohr
    if not np.all(np.isfinite(scaled)):
        raise ValueError("Periodic cubic B-spline positions overflow grid coordinates.")
    lower = np.floor(scaled)
    if np.any(np.abs(lower) > np.iinfo(np.int64).max - 2):
        raise ValueError("Periodic cubic B-spline positions exceed integer grid range.")
    lower_indices = lower.astype(np.int64)
    fractional = scaled - lower_indices
    # Rounding can produce exactly 1.0 for a very large finite coordinate.
    carry = fractional >= 1.0
    if np.any(carry):
        lower_indices[carry] += 1
        fractional[carry] -= 1.0
    if np.any(fractional < 0.0) or np.any(fractional >= 1.0):
        raise RuntimeError(
            "Periodic cubic B-spline coordinate decomposition is invalid."
        )

    axis_indices: list[np.ndarray] = []
    axis_weights: list[np.ndarray] = []
    offsets = np.array((-1, 0, 1, 2), dtype=np.int64)
    for axis, size in enumerate(grid.shape):
        axis_indices.append((lower_indices[:, axis, None] + offsets) % size)
        axis_weights.append(_cubic_weights(fractional[:, axis]))

    count = positions.shape[0]
    node_indices = np.empty(
        (count, V0_PERIODIC_CUBIC_BSPLINE_STENCIL_SIZE), dtype=np.int64
    )
    weights = np.empty((count, V0_PERIODIC_CUBIC_BSPLINE_STENCIL_SIZE), dtype=float)
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
    if column != V0_PERIODIC_CUBIC_BSPLINE_STENCIL_SIZE:
        raise RuntimeError("Periodic cubic B-spline stencil size is inconsistent.")
    return node_indices, weights


@dataclass(frozen=True)
class Route2V0PeriodicCubicBSplineStencil:
    """Compact periodic ``C2`` deposition matrix for positions on one grid.

    ``node_linear_indices[p, q]`` and ``weights[p, q]`` encode the 64 grid
    nodes of position ``p``.  Repeated nodes are intentionally retained for
    one- and two-point periodic axes; :meth:`deposit` and
    :meth:`dense_occupancy_weights` use accumulation rather than assignment,
    so the same count and adjoint identities hold on controlled tiny grids.
    """

    grid: RegularCartesianGrid
    positions_bohr: np.ndarray
    node_linear_indices: np.ndarray
    weights: np.ndarray
    construction: str = V0_PERIODIC_CUBIC_BSPLINE_CONSTRUCTION
    _point_count: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.grid, RegularCartesianGrid):
            raise TypeError("Periodic cubic B-spline stencil requires a regular grid.")
        if self.construction != V0_PERIODIC_CUBIC_BSPLINE_CONSTRUCTION:
            raise ValueError(
                "Unsupported Route-2 V0 periodic cubic B-spline construction."
            )
        positions = _positions(self.positions_bohr)
        count = positions.shape[0]
        indices = _immutable_array(
            self.node_linear_indices,
            name="Periodic cubic B-spline node indices",
            shape=(count, V0_PERIODIC_CUBIC_BSPLINE_STENCIL_SIZE),
            integer=True,
        )
        if np.any(indices < 0) or np.any(indices >= self.grid.point_count):
            raise ValueError("Periodic cubic B-spline node indices leave the grid.")
        weights = _immutable_array(
            self.weights,
            name="Periodic cubic B-spline weights",
            shape=(count, V0_PERIODIC_CUBIC_BSPLINE_STENCIL_SIZE),
            nonnegative=True,
        )
        if not np.allclose(
            np.sum(weights, axis=1),
            1.0,
            rtol=0.0,
            atol=128.0 * np.finfo(float).eps,
        ):
            raise ValueError("Periodic cubic B-spline weights must sum to one.")
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
                "Periodic cubic B-spline stencil does not match its declared grid "
                "and positions."
            )
        object.__setattr__(self, "positions_bohr", positions)
        object.__setattr__(self, "node_linear_indices", indices)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "_point_count", count)

    @property
    def point_count(self) -> int:
        """Return the number of deposited Cartesian points."""

        return self._point_count

    def dense_occupancy_weights(self) -> np.ndarray:
        """Materialize ``D[g,p]`` only for controlled dense-grid references."""

        result = np.zeros((self.grid.point_count, self.point_count), dtype=float)
        columns = np.repeat(
            np.arange(self.point_count, dtype=np.intp),
            V0_PERIODIC_CUBIC_BSPLINE_STENCIL_SIZE,
        )
        np.add.at(
            result,
            (self.node_linear_indices.reshape(-1), columns),
            self.weights.reshape(-1),
        )
        dense = result.reshape((*self.grid.shape, self.point_count))
        dense.setflags(write=False)
        return dense

    def deposit(self, values: np.ndarray) -> np.ndarray:
        """Return the exact dense-grid action ``D values``."""

        coefficients = _immutable_array(
            values,
            name="Periodic cubic B-spline deposition values",
            shape=(self.point_count,),
        )
        result = np.zeros(self.grid.point_count, dtype=float)
        np.add.at(
            result,
            self.node_linear_indices.reshape(-1),
            (self.weights * coefficients[:, None]).reshape(-1),
        )
        field_values = result.reshape(self.grid.shape)
        field_values.setflags(write=False)
        return field_values

    def field_adjoint(self, field: np.ndarray) -> np.ndarray:
        """Return the exact Euclidean adjoint ``D.T field`` at each point."""

        values = _immutable_array(
            field,
            name="Periodic cubic B-spline grid field",
            shape=self.grid.shape,
        ).reshape(self.grid.point_count)
        result = np.sum(self.weights * values[self.node_linear_indices], axis=1)
        result.setflags(write=False)
        return result


def build_route2_v0_periodic_cubic_bspline_stencil(
    *,
    grid: RegularCartesianGrid,
    positions_bohr: np.ndarray,
) -> Route2V0PeriodicCubicBSplineStencil:
    """Build the exact periodic cubic B-spline stencil for declared positions."""

    if not isinstance(grid, RegularCartesianGrid):
        raise TypeError("Periodic cubic B-spline construction requires a regular grid.")
    positions = _positions(positions_bohr)
    node_indices, weights = _stencil_arrays(
        grid=grid,
        positions_bohr=positions,
    )
    return Route2V0PeriodicCubicBSplineStencil(
        grid=grid,
        positions_bohr=positions,
        node_linear_indices=node_indices,
        weights=weights,
    )


__all__ = [
    "V0_PERIODIC_CUBIC_BSPLINE_CONSTRUCTION",
    "V0_PERIODIC_CUBIC_BSPLINE_STENCIL_SIZE",
    "Route2V0PeriodicCubicBSplineStencil",
    "build_route2_v0_periodic_cubic_bspline_stencil",
]
