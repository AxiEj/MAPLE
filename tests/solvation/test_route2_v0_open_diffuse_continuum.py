from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_open_diffuse_continuum import (
    V0_OPEN_DIFFUSE_DIELECTRIC_BOUNDARY,
    V0_OPEN_DIFFUSE_DIELECTRIC_CONSTRUCTION,
    V0_OPEN_DIFFUSE_DIELECTRIC_SCOPE,
    Route2V0OpenDiffuseLocalDielectricOperator,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_auxiliary_ao_grid import (
    Route2V0AuxiliaryAODensityGridProjection,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_open_bspline import (
    build_route2_v0_open_cubic_bspline_stencil,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)


def _grid() -> RegularCartesianGrid:
    return RegularCartesianGrid(
        origin_bohr=np.array([-2.8, -2.1, -1.7]),
        spacing_bohr=np.array([0.7, 0.9, 1.1]),
        shape=(7, 6, 5),
    )


def _charged(grid: RegularCartesianGrid, rng: np.random.Generator) -> np.ndarray:
    """Return a deliberately non-neutral finite-box source."""

    values = rng.normal(size=grid.shape)
    values += 0.17
    return values


def _operator(
    occupancy: np.ndarray,
) -> Route2V0OpenDiffuseLocalDielectricOperator:
    return Route2V0OpenDiffuseLocalDielectricOperator(
        grid=_grid(),
        occupancy=occupancy,
        bulk_dielectric_constant=28.4,
        cg_relative_tolerance=1.0e-13,
    )


def _ao_projection(
    grid: RegularCartesianGrid,
) -> Route2V0AuxiliaryAODensityGridProjection:
    points = grid.points_bohr()
    raw = np.column_stack(
        (
            np.exp(-0.55 * np.sum((points - np.array([0.15, -0.1, 0.0])) ** 2, axis=1)),
            (points[:, 0] - 0.35)
            * np.exp(
                -0.42 * np.sum((points - np.array([-0.25, 0.2, 0.1])) ** 2, axis=1)
            ),
        )
    )
    orthonormal, _ = np.linalg.qr(np.sqrt(grid.volume_element_bohr3) * raw)
    return Route2V0AuxiliaryAODensityGridProjection(
        grid=grid,
        ao_values=orthonormal / np.sqrt(grid.volume_element_bohr3),
        overlap_matrix=np.eye(2),
    )


def test_open_diffuse_operator_accepts_a_charged_source_and_matches_uniform_scaling():
    grid = _grid()
    operator = _operator(np.ones(grid.shape))
    density = _charged(grid, np.random.default_rng(20260731))

    state = operator.solve(density)

    scale = 1.0 / operator.bulk_dielectric_constant - 1.0
    np.testing.assert_allclose(
        state.reaction_potential_hartree_per_e,
        scale * state.vacuum_potential_hartree_per_e,
        rtol=0.0,
        atol=5.0e-12,
    )
    assert abs(state.total_grid_charge_e) > 1.0e-3
    assert state.reaction_energy_hartree < 0.0
    assert state.vacuum_residual_inf < 1.0e-10
    assert state.solvated_residual_inf < 1.0e-10
    assert state.boundary_condition == V0_OPEN_DIFFUSE_DIELECTRIC_BOUNDARY
    assert state.grid_fingerprint == operator.grid_fingerprint
    assert operator.construction == V0_OPEN_DIFFUSE_DIELECTRIC_CONSTRUCTION
    assert operator.response_scope == V0_OPEN_DIFFUSE_DIELECTRIC_SCOPE
    assert operator.is_total_solvation_asset is False
    with pytest.raises(ValueError, match="numerical control"):
        operator.require_bulk_state_source()


def test_open_diffuse_reaction_block_is_reciprocal_passive_and_density_conjugate():
    grid = _grid()
    axes = np.indices(grid.shape, dtype=float)
    occupancy = 0.55 + 0.18 * np.sin(2.0 * math.pi * axes[0] / grid.shape[0])
    operator = _operator(occupancy)
    rng = np.random.default_rng(20260801)
    density = _charged(grid, rng)
    left = _charged(grid, rng)
    right = _charged(grid, rng)

    state = operator.solve(density)
    left_right = operator.reaction_pairing_hartree(left, right)
    right_left = operator.reaction_pairing_hartree(right, left)
    scale = max(1.0, abs(left_right), abs(right_left))
    assert abs(left_right - right_left) < 3.0e-10 * scale
    assert state.reaction_energy_hartree <= 1.0e-10

    direction = _charged(grid, rng)
    direction /= float(np.linalg.norm(direction))
    step = 1.0e-6
    finite_difference = (
        operator.reaction_energy_hartree(density + step * direction)
        - operator.reaction_energy_hartree(density - step * direction)
    ) / (2.0 * step)
    analytic = grid.volume_element_bohr3 * float(
        np.sum(direction * state.reaction_potential_hartree_per_e)
    )
    assert finite_difference == pytest.approx(analytic, rel=3.0e-6, abs=3.0e-8)


def test_open_diffuse_occupancy_derivative_contains_exterior_face_terms():
    grid = _grid()
    rng = np.random.default_rng(20260802)
    axes = np.indices(grid.shape, dtype=float)
    base_occupancy = (
        0.53
        + 0.12 * np.sin(2.0 * math.pi * axes[0] / grid.shape[0])
        + 0.08 * np.cos(2.0 * math.pi * axes[1] / grid.shape[1])
    )
    density = _charged(grid, rng)
    direction = rng.normal(size=grid.shape)
    direction /= float(np.max(np.abs(direction)))
    operator = _operator(base_occupancy)
    state = operator.solve(density)
    analytic = float(
        np.sum(operator.occupancy_energy_gradient_hartree(state) * direction)
    )
    step = 2.0e-6
    plus = _operator(base_occupancy + step * direction).reaction_energy_hartree(density)
    minus = _operator(base_occupancy - step * direction).reaction_energy_hartree(
        density
    )
    finite_difference = (plus - minus) / (2.0 * step)

    assert finite_difference == pytest.approx(analytic, rel=4.0e-6, abs=3.0e-8)
    gradient = operator.occupancy_energy_gradient_hartree(state)
    assert float(np.max(np.abs(gradient[0, :, :]))) > 1.0e-12
    assert float(np.max(np.abs(gradient[-1, :, :]))) > 1.0e-12


def test_open_diffuse_reaction_pulls_back_exactly_to_the_ao_density_dual():
    grid = _grid()
    projection = _ao_projection(grid)
    density_matrix = np.diag([1.16, 0.84])
    density_direction = np.diag([0.19, -0.19])
    electron_state = projection.evaluate(density_matrix)
    nuclear_stencil = build_route2_v0_open_cubic_bspline_stencil(
        grid=grid,
        positions_bohr=np.array([[0.12, -0.18, 0.08]]),
    )
    nuclear_density = (
        nuclear_stencil.deposit(np.array([2.0])) / grid.volume_element_bohr3
    )
    occupancy = 0.55 + 0.1 * np.sin(
        2.0 * math.pi * np.indices(grid.shape, dtype=float)[0] / grid.shape[0]
    )
    operator = _operator(occupancy)
    state = operator.solve(
        nuclear_density - electron_state.electron_density_e_per_bohr3
    )
    electron_fock = projection.density_potential_matrix_hartree(
        -state.reaction_potential_hartree_per_e
    )
    analytic = float(np.einsum("ij,ji->", density_direction, electron_fock))

    step = 1.0e-6
    plus = projection.evaluate(density_matrix + step * density_direction)
    minus = projection.evaluate(density_matrix - step * density_direction)
    finite_difference = (
        operator.reaction_energy_hartree(
            nuclear_density - plus.electron_density_e_per_bohr3
        )
        - operator.reaction_energy_hartree(
            nuclear_density - minus.electron_density_e_per_bohr3
        )
    ) / (2.0 * step)

    assert electron_state.grid_electron_count_e == pytest.approx(2.0, abs=2.0e-14)
    assert finite_difference == pytest.approx(analytic, rel=4.0e-6, abs=3.0e-8)
    assert (
        projection.density_potential_pairing_error_hartree(
            density_direction,
            -state.reaction_potential_hartree_per_e,
        )
        < 2.0e-14
    )


def test_open_diffuse_block_rejects_bad_domain_or_crossed_state_without_neutrality_rule():
    grid = _grid()
    operator = _operator(np.full(grid.shape, 0.5))
    charged_state = operator.solve(np.ones(grid.shape))
    assert charged_state.total_grid_charge_e > 0.0
    with pytest.raises(ValueError, match="at least two cells"):
        Route2V0OpenDiffuseLocalDielectricOperator(
            grid=RegularCartesianGrid(
                origin_bohr=np.zeros(3),
                spacing_bohr=np.ones(3),
                shape=(1, 3, 3),
            ),
            occupancy=np.ones((1, 3, 3)),
            bulk_dielectric_constant=10.0,
        )
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        _operator(np.full(grid.shape, 1.01))
    with pytest.raises(ValueError, match="occupancy does not match"):
        _operator(np.full(grid.shape, 0.6)).occupancy_energy_gradient_hartree(
            charged_state
        )
    with pytest.raises(ValueError, match="grid representation does not match"):
        Route2V0OpenDiffuseLocalDielectricOperator(
            grid=RegularCartesianGrid(
                origin_bohr=grid.origin_bohr,
                spacing_bohr=1.1 * grid.spacing_bohr,
                shape=grid.shape,
            ),
            occupancy=np.full(grid.shape, 0.5),
            bulk_dielectric_constant=28.4,
        ).occupancy_energy_gradient_hartree(charged_state)
    with pytest.raises(ValueError, match="total grid charge is inconsistent"):
        operator.occupancy_energy_gradient_hartree(
            replace(
                charged_state,
                total_grid_charge_e=charged_state.total_grid_charge_e + 0.1,
            )
        )
    with pytest.raises(ValueError, match="real-valued"):
        operator.solve(np.ones(grid.shape, dtype=complex))
