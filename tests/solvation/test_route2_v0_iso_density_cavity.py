from __future__ import annotations

import math

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_iso_density_cavity import (
    V0_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION,
    V0_ISO_DENSITY_PRODUCT_CAVITY_SCOPE,
    Route2V0IsoDensityProductCavity,
    Route2V0IsoDensityProductReactionOperator,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)


def _grid() -> RegularCartesianGrid:
    return RegularCartesianGrid(
        origin_bohr=np.array([-3.0, -2.4, -2.0]),
        spacing_bohr=np.array([0.8, 0.8, 0.8]),
        shape=(8, 7, 6),
    )


def _periodic_gaussian(
    grid: RegularCartesianGrid,
    *,
    width_bohr: float,
    centre_index: tuple[int, int, int] = (0, 0, 0),
    normalize: bool = True,
) -> np.ndarray:
    indices = np.indices(grid.shape, dtype=float)
    distance_squared = np.zeros(grid.shape, dtype=float)
    for axis, length in enumerate(grid.shape):
        delta = np.abs(indices[axis] - centre_index[axis])
        delta = np.minimum(delta, length - delta) * grid.spacing_bohr[axis]
        distance_squared += delta**2
    values = np.exp(-distance_squared / (2.0 * width_bohr**2))
    if normalize:
        values /= grid.volume_element_bohr3 * float(np.sum(values))
    return values


def _cavity() -> Route2V0IsoDensityProductCavity:
    grid = _grid()
    return Route2V0IsoDensityProductCavity(
        grid=grid,
        solvent_electron_density_kernel_e_per_bohr3=_periodic_gaussian(
            grid,
            width_bohr=0.85,
        ),
        overlap_threshold_e2_per_bohr3=0.030,
    )


def _electron_density(grid: RegularCartesianGrid) -> np.ndarray:
    return 0.010 + 0.150 * _periodic_gaussian(
        grid,
        width_bohr=1.15,
        centre_index=(4, 3, 3),
        normalize=False,
    )


def _zero_mean_direction(
    grid: RegularCartesianGrid,
    rng: np.random.Generator,
) -> np.ndarray:
    direction = rng.normal(size=grid.shape)
    direction -= float(np.mean(direction))
    direction /= float(np.max(np.abs(direction)))
    return direction


def test_iso_density_product_cavity_is_smooth_and_has_an_exact_vjp():
    cavity = _cavity()
    grid = cavity.grid
    electron_density = _electron_density(grid)
    state = cavity.evaluate(electron_density)

    assert np.min(state.occupancy) < 0.10
    assert np.max(state.occupancy) > 0.90
    assert np.all(state.occupancy_derivative_per_overlap_bohr3_per_e2 <= 0.0)
    assert cavity.construction == V0_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION
    assert cavity.response_scope == V0_ISO_DENSITY_PRODUCT_CAVITY_SCOPE
    assert cavity.is_total_solvation_asset is False

    rng = np.random.default_rng(20260731)
    direction = _zero_mean_direction(grid, rng)
    weights = rng.normal(size=grid.shape)
    coordinate_gradient = grid.volume_element_bohr3 * weights
    potential = cavity.occupancy_coordinate_vjp(state, coordinate_gradient)
    analytic = grid.volume_element_bohr3 * float(np.sum(direction * potential))
    step = 1.0e-5
    plus = cavity.evaluate(electron_density + step * direction)
    minus = cavity.evaluate(electron_density - step * direction)
    finite_difference = (
        grid.volume_element_bohr3
        * float(np.sum(weights * (plus.occupancy - minus.occupancy)))
        / (2.0 * step)
    )

    assert finite_difference == pytest.approx(analytic, rel=2.0e-6, abs=2.0e-8)


def test_composed_reaction_scalar_includes_the_nonlocal_cavity_chain_rule():
    cavity = _cavity()
    grid = cavity.grid
    electron_density = _electron_density(grid)
    rng = np.random.default_rng(20260732)
    charge = 0.006 * _zero_mean_direction(grid, rng)
    nuclear_density = electron_density + charge
    operator = Route2V0IsoDensityProductReactionOperator(
        cavity=cavity,
        bulk_dielectric_constant=31.2,
        cg_relative_tolerance=1.0e-13,
    )

    state = operator.solve(nuclear_density, electron_density)
    potential = operator.electron_density_reaction_potential_hartree_per_e(state)
    direction = _zero_mean_direction(grid, rng)
    analytic = grid.volume_element_bohr3 * float(np.sum(direction * potential))
    step = 1.0e-5
    finite_difference = (
        operator.reaction_energy_hartree(
            nuclear_density, electron_density + step * direction
        )
        - operator.reaction_energy_hartree(
            nuclear_density,
            electron_density - step * direction,
        )
    ) / (2.0 * step)

    assert state.continuum_state.reaction_energy_hartree < 0.0
    assert finite_difference == pytest.approx(analytic, rel=4.0e-6, abs=3.0e-8)
    direct_charge_term = -state.continuum_state.reaction_potential_hartree_per_e
    assert float(np.max(np.abs(potential - direct_charge_term))) > 1.0e-8


def test_composed_nonlocal_cavity_is_periodically_translation_covariant():
    cavity = _cavity()
    grid = cavity.grid
    electron_density = _electron_density(grid)
    nuclear_density = electron_density + 0.005 * _zero_mean_direction(
        grid,
        np.random.default_rng(20260734),
    )
    operator = Route2V0IsoDensityProductReactionOperator(
        cavity=cavity,
        bulk_dielectric_constant=31.2,
        cg_relative_tolerance=1.0e-13,
    )
    shift = (2, -1, 1)
    axes = (0, 1, 2)

    state = operator.solve(nuclear_density, electron_density)
    shifted_state = operator.solve(
        np.roll(nuclear_density, shift, axis=axes),
        np.roll(electron_density, shift, axis=axes),
    )
    potential = operator.electron_density_reaction_potential_hartree_per_e(state)
    shifted_potential = operator.electron_density_reaction_potential_hartree_per_e(
        shifted_state
    )

    assert shifted_state.continuum_state.reaction_energy_hartree == pytest.approx(
        state.continuum_state.reaction_energy_hartree,
        rel=0.0,
        abs=2.0e-12,
    )
    np.testing.assert_allclose(
        shifted_state.cavity_state.occupancy,
        np.roll(state.cavity_state.occupancy, shift, axis=axes),
        rtol=0.0,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        shifted_potential,
        np.roll(potential, shift, axis=axes),
        rtol=0.0,
        atol=4.0e-11,
    )


def test_iso_density_product_cavity_and_reaction_fail_closed_on_invalid_or_mismatched_states():
    cavity = _cavity()
    grid = cavity.grid
    electron_density = _electron_density(grid)
    with pytest.raises(ValueError, match="nonnegative"):
        cavity.evaluate(-electron_density)
    with pytest.raises(ValueError, match="must not vanish"):
        Route2V0IsoDensityProductCavity(
            grid=grid,
            solvent_electron_density_kernel_e_per_bohr3=np.zeros(grid.shape),
            overlap_threshold_e2_per_bohr3=0.030,
        )

    other = Route2V0IsoDensityProductCavity(
        grid=grid,
        solvent_electron_density_kernel_e_per_bohr3=_periodic_gaussian(
            grid,
            width_bohr=1.10,
        ),
        overlap_threshold_e2_per_bohr3=0.030,
    )
    state = cavity.evaluate(electron_density)
    with pytest.raises(ValueError, match="does not match"):
        other.occupancy_coordinate_vjp(state, np.ones(grid.shape))

    reaction = Route2V0IsoDensityProductReactionOperator(
        cavity=cavity,
        bulk_dielectric_constant=31.2,
    )
    nuclear_density = electron_density + 0.004 * _zero_mean_direction(
        grid,
        np.random.default_rng(20260735),
    )
    reaction_state = reaction.solve(nuclear_density, electron_density)
    different_dielectric = Route2V0IsoDensityProductReactionOperator(
        cavity=cavity,
        bulk_dielectric_constant=20.0,
    )
    with pytest.raises(ValueError, match="bulk dielectric does not match"):
        different_dielectric.electron_density_reaction_potential_hartree_per_e(
            reaction_state
        )
    with pytest.raises(ValueError, match="read-only"):
        reaction_state.cavity_state.occupancy[0, 0, 0] = math.nan
