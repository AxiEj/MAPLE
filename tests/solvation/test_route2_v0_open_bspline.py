from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_open_bspline import (
    V0_OPEN_CUBIC_BSPLINE_CONSTRUCTION,
    Route2V0OpenCubicBSplineStencil,
    build_route2_v0_open_cubic_bspline_stencil,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)


def _grid() -> RegularCartesianGrid:
    return RegularCartesianGrid(
        origin_bohr=np.array([-3.0, -2.5, -2.0]),
        spacing_bohr=np.array([0.7, 0.8, 0.9]),
        shape=(10, 9, 8),
    )


def test_open_bspline_preserves_charge_and_grid_field_adjoint_without_wrapping():
    grid = _grid()
    stencil = build_route2_v0_open_cubic_bspline_stencil(
        grid=grid,
        positions_bohr=np.array(
            [
                [-1.21, -0.93, -0.62],
                [0.36, 0.27, 0.41],
            ]
        ),
    )
    charges = np.array([1.6, 4.4])
    field = np.linspace(-0.4, 0.3, grid.point_count).reshape(grid.shape)
    deposited = stencil.deposit(charges)

    assert stencil.construction == V0_OPEN_CUBIC_BSPLINE_CONSTRUCTION
    assert float(np.sum(deposited)) == pytest.approx(float(np.sum(charges)), abs=2e-14)
    assert float(np.sum(deposited * field)) == pytest.approx(
        float(charges @ stencil.field_adjoint(field)),
        abs=2e-14,
    )
    assert np.all(stencil.node_linear_indices >= 0)
    assert np.all(stencil.node_linear_indices < grid.point_count)


def test_open_bspline_fails_instead_of_clipping_or_wrapping_boundary_support():
    grid = _grid()
    with pytest.raises(ValueError, match="enlarge the open-boundary buffer"):
        build_route2_v0_open_cubic_bspline_stencil(
            grid=grid,
            positions_bohr=np.array([grid.origin_bohr]),
        )
    stencil = build_route2_v0_open_cubic_bspline_stencil(
        grid=grid,
        positions_bohr=np.array([[-1.21, -0.93, -0.62]]),
    )
    with pytest.raises(ValueError, match="integer-valued"):
        Route2V0OpenCubicBSplineStencil(
            grid=grid,
            positions_bohr=stencil.positions_bohr,
            node_linear_indices=stencil.node_linear_indices.astype(float) + 0.5,
            weights=stencil.weights,
        )
    with pytest.raises(ValueError, match="real-valued"):
        stencil.deposit(np.array([1.0 + 0.0j]))
