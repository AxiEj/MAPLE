from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_periodic_bspline import (
    V0_PERIODIC_CUBIC_BSPLINE_STENCIL_SIZE,
    build_route2_v0_periodic_cubic_bspline_stencil,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)


def _grid() -> RegularCartesianGrid:
    return RegularCartesianGrid(
        origin_bohr=np.array([-1.0, 0.5, 2.0]),
        spacing_bohr=np.array([0.5, 0.75, 1.25]),
        shape=(5, 4, 3),
    )


def _positions() -> np.ndarray:
    return np.array(
        [
            [-0.8, 0.6, 2.1],
            [0.27, 1.73, 3.88],
            [1.41, -0.32, 4.14],
        ]
    )


def test_periodic_cubic_bspline_stencil_preserves_count_periodicity_and_adjoint():
    grid = _grid()
    positions = _positions()
    stencil = build_route2_v0_periodic_cubic_bspline_stencil(
        grid=grid,
        positions_bohr=positions,
    )

    assert stencil.point_count == len(positions)
    assert stencil.weights.shape == (
        len(positions),
        V0_PERIODIC_CUBIC_BSPLINE_STENCIL_SIZE,
    )
    assert np.all(stencil.weights >= 0.0)
    np.testing.assert_allclose(
        np.sum(stencil.weights, axis=1),
        np.ones(len(positions)),
        rtol=0.0,
        atol=1.0e-14,
    )

    dense = stencil.dense_occupancy_weights()
    np.testing.assert_allclose(
        np.sum(dense.reshape(grid.point_count, len(positions)), axis=0),
        np.ones(len(positions)),
        rtol=0.0,
        atol=1.0e-14,
    )

    values = np.array([0.3, -0.7, 1.2])
    field = np.arange(grid.point_count, dtype=float).reshape(grid.shape) / 11.0
    deposited = stencil.deposit(values)
    adjoint = stencil.field_adjoint(field)
    assert float(np.sum(field * deposited)) == pytest.approx(
        float(np.dot(adjoint, values)),
        rel=0.0,
        abs=2.0e-14,
    )

    shifted = build_route2_v0_periodic_cubic_bspline_stencil(
        grid=grid,
        positions_bohr=positions + grid.spacing_bohr * np.asarray(grid.shape),
    )
    np.testing.assert_allclose(
        shifted.dense_occupancy_weights(),
        dense,
        rtol=0.0,
        atol=2.0e-14,
    )


def test_periodic_cubic_bspline_accumulates_repeated_nodes_on_small_periodic_axes():
    grid = RegularCartesianGrid(
        origin_bohr=np.zeros(3),
        spacing_bohr=np.ones(3),
        shape=(1, 2, 1),
    )
    stencil = build_route2_v0_periodic_cubic_bspline_stencil(
        grid=grid,
        positions_bohr=np.array([[0.2, 0.7, -0.4]]),
    )

    dense = stencil.dense_occupancy_weights()
    assert dense.shape == (*grid.shape, 1)
    assert float(np.sum(dense)) == pytest.approx(1.0, rel=0.0, abs=1.0e-14)
    np.testing.assert_allclose(
        stencil.deposit(np.array([2.5])),
        2.5 * dense[..., 0],
        rtol=0.0,
        atol=1.0e-14,
    )


def test_periodic_cubic_bspline_stencil_fails_closed_for_invalid_ledgers():
    stencil = build_route2_v0_periodic_cubic_bspline_stencil(
        grid=_grid(),
        positions_bohr=_positions(),
    )
    with pytest.raises(ValueError, match="shape"):
        build_route2_v0_periodic_cubic_bspline_stencil(
            grid=_grid(),
            positions_bohr=np.zeros((2, 2)),
        )
    with pytest.raises(ValueError, match="weights must sum to one"):
        replace(stencil, weights=stencil.weights * 0.5)
    with pytest.raises(ValueError, match="does not match its declared grid"):
        replace(stencil, weights=np.roll(stencil.weights, 1, axis=1))
    with pytest.raises(ValueError, match="node indices leave the grid"):
        replace(stencil, node_linear_indices=stencil.node_linear_indices + 1000)
    with pytest.raises(ValueError, match="Unsupported Route-2 V0 periodic"):
        replace(stencil, construction="another-stencil")
