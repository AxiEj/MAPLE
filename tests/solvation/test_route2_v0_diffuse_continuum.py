from __future__ import annotations

import math

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_diffuse_continuum import (
    V0_DIFFUSE_DIELECTRIC_CONSTRUCTION,
    V0_DIFFUSE_DIELECTRIC_SCOPE,
    Route2V0DiffuseLocalDielectricOperator,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)


def _grid() -> RegularCartesianGrid:
    return RegularCartesianGrid(
        origin_bohr=np.array([-3.0, -2.5, -2.0]),
        spacing_bohr=np.array([0.8, 0.9, 1.1]),
        shape=(8, 7, 6),
    )


def _neutral(grid: RegularCartesianGrid, rng: np.random.Generator) -> np.ndarray:
    values = rng.normal(size=grid.shape)
    values -= float(np.mean(values))
    return values


def _operator(occupancy: np.ndarray) -> Route2V0DiffuseLocalDielectricOperator:
    return Route2V0DiffuseLocalDielectricOperator(
        grid=_grid(),
        occupancy=occupancy,
        bulk_dielectric_constant=24.7,
        cg_relative_tolerance=1.0e-13,
    )


def test_uniform_diffuse_dielectric_matches_its_same_discrete_vacuum_operator():
    grid = _grid()
    operator = _operator(np.ones(grid.shape))
    density = _neutral(grid, np.random.default_rng(20260730))

    state = operator.solve(density)

    scale = 1.0 / operator.bulk_dielectric_constant - 1.0
    np.testing.assert_allclose(
        state.reaction_potential_hartree_per_e,
        scale * state.vacuum_potential_hartree_per_e,
        rtol=0.0,
        atol=4.0e-12,
    )
    assert state.reaction_energy_hartree < 0.0
    assert state.vacuum_residual_inf < 1.0e-10
    assert state.solvated_residual_inf < 1.0e-10
    assert operator.construction == V0_DIFFUSE_DIELECTRIC_CONSTRUCTION
    assert operator.response_scope == V0_DIFFUSE_DIELECTRIC_SCOPE
    assert operator.is_total_solvation_asset is False
    with pytest.raises(ValueError, match="numerical control"):
        operator.require_bulk_state_source()


def test_diffuse_reaction_block_is_reciprocal_passive_and_density_conjugate():
    grid = _grid()
    axes = np.indices(grid.shape, dtype=float)
    occupancy = 0.5 + 0.2 * np.sin(2.0 * math.pi * axes[0] / grid.shape[0])
    operator = _operator(occupancy)
    rng = np.random.default_rng(20260731)
    density = _neutral(grid, rng)
    left = _neutral(grid, rng)
    right = _neutral(grid, rng)

    state = operator.solve(density)
    left_right = operator.reaction_pairing_hartree(left, right)
    right_left = operator.reaction_pairing_hartree(right, left)
    scale = max(1.0, abs(left_right), abs(right_left))
    assert abs(left_right - right_left) < 2.0e-10 * scale
    assert state.reaction_energy_hartree <= 1.0e-10

    direction = _neutral(grid, rng)
    direction /= float(np.linalg.norm(direction))
    step = 1.0e-6
    finite_difference = (
        operator.reaction_energy_hartree(density + step * direction)
        - operator.reaction_energy_hartree(density - step * direction)
    ) / (2.0 * step)
    analytic = grid.volume_element_bohr3 * float(
        np.sum(direction * state.reaction_potential_hartree_per_e)
    )
    assert finite_difference == pytest.approx(analytic, rel=2.0e-6, abs=2.0e-8)


def test_diffuse_occupancy_derivative_is_the_envelope_derivative_of_the_energy():
    grid = _grid()
    rng = np.random.default_rng(20260732)
    axes = np.indices(grid.shape, dtype=float)
    base_occupancy = (
        0.5
        + 0.12 * np.sin(2.0 * math.pi * axes[0] / grid.shape[0])
        + 0.08 * np.cos(2.0 * math.pi * axes[1] / grid.shape[1])
    )
    density = _neutral(grid, rng)
    direction = _neutral(grid, rng)
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

    assert finite_difference == pytest.approx(analytic, rel=3.0e-6, abs=2.0e-8)


def test_diffuse_block_fails_closed_on_nonneutral_density_and_mismatched_state():
    grid = _grid()
    occupancy = np.full(grid.shape, 0.5)
    operator = _operator(occupancy)
    with pytest.raises(ValueError, match="must be neutral"):
        operator.solve(np.ones(grid.shape))
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        _operator(np.full(grid.shape, 1.01))

    state = operator.solve(_neutral(grid, np.random.default_rng(20260734)))
    other = _operator(np.full(grid.shape, 0.6))
    with pytest.raises(ValueError, match="occupancy does not match"):
        other.occupancy_energy_gradient_hartree(state)
    with pytest.raises(ValueError, match="read-only"):
        state.reaction_potential_hartree_per_e[0, 0, 0] = 0.0
