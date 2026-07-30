from __future__ import annotations

import math

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_weighted_density_cavity import (
    V0_WEIGHTED_DENSITY_CAVITY_CONSTRUCTION,
    V0_WEIGHTED_DENSITY_CAVITY_SCOPE,
    Route2V0WeightedDensityCavityFunctional,
    Route2V0WeightedDensityCavityThermodynamics,
)


def _grid() -> RegularCartesianGrid:
    return RegularCartesianGrid(
        origin_bohr=np.array([-3.2, -2.8, -2.4]),
        spacing_bohr=np.array([0.7, 0.8, 0.9]),
        shape=(8, 7, 6),
    )


def _thermodynamics() -> Route2V0WeightedDensityCavityThermodynamics:
    return Route2V0WeightedDensityCavityThermodynamics.from_si(
        temperature_kelvin=298.15,
        pressure_pascal=101_325.0,
        molecular_number_density_angstrom3=0.03336,
        vapor_pressure_pascal=3_169.0,
        surface_tension_newton_per_meter=0.07197,
        solvent_vdw_radius_angstrom=1.385,
    )


def _kernel(grid: RegularCartesianGrid) -> np.ndarray:
    """Return an exact periodic, positive, normalized shell-control kernel."""

    weights = np.zeros(grid.shape, dtype=float)
    weights[0, 0, 0] = 0.40
    for axis in range(3):
        positive = [0, 0, 0]
        negative = [0, 0, 0]
        positive[axis] = 1
        negative[axis] = -1
        weights[tuple(positive)] = 0.10
        weights[tuple(negative)] = 0.10
    assert float(np.sum(weights)) == pytest.approx(1.0)
    return weights / grid.volume_element_bohr3


def _functional() -> Route2V0WeightedDensityCavityFunctional:
    grid = _grid()
    return Route2V0WeightedDensityCavityFunctional(
        grid=grid,
        thermodynamics=_thermodynamics(),
        nearest_neighbor_shell_kernel_per_bohr3=_kernel(grid),
    )


def _occupancy(grid: RegularCartesianGrid) -> np.ndarray:
    rng = np.random.default_rng(20260801)
    return 0.20 + 0.60 * rng.random(grid.shape)


def test_weighted_density_cavity_recovers_bulk_and_empty_volume_limits():
    functional = _functional()
    grid = functional.grid
    thermodynamics = functional.thermodynamics

    bulk = functional.evaluate(np.ones(grid.shape))
    cavity = functional.evaluate(np.zeros(grid.shape))

    assert functional.construction == V0_WEIGHTED_DENSITY_CAVITY_CONSTRUCTION
    assert functional.response_scope == V0_WEIGHTED_DENSITY_CAVITY_SCOPE
    assert functional.is_total_solvation_asset is False
    assert bulk.cavity_energy_hartree == pytest.approx(0.0, abs=2.0e-15)
    assert cavity.cavity_energy_hartree == pytest.approx(
        thermodynamics.pressure_hartree_per_bohr3
        * grid.volume_element_bohr3
        * np.prod(grid.shape),
        rel=0.0,
        abs=2.0e-15,
    )
    assert np.allclose(bulk.weighted_occupancy, 1.0)
    assert np.allclose(cavity.weighted_occupancy, 0.0)
    assert math.isfinite(thermodynamics.small_droplet_log_term)
    assert math.isfinite(thermodynamics.surface_coefficient)


def test_weighted_density_cavity_has_an_exact_occupancy_derivative():
    functional = _functional()
    grid = functional.grid
    occupancy = _occupancy(grid)
    state = functional.evaluate(occupancy)
    rng = np.random.default_rng(20260802)
    direction = rng.normal(size=grid.shape)
    direction /= float(np.max(np.abs(direction)))
    coordinate_gradient = functional.occupancy_coordinate_gradient_hartree(state)
    functional_gradient = functional.occupancy_functional_derivative_hartree_per_bohr3(
        state
    )
    np.testing.assert_allclose(
        coordinate_gradient,
        grid.volume_element_bohr3 * functional_gradient,
        rtol=0.0,
        atol=2.0e-15,
    )

    analytic = float(np.sum(coordinate_gradient * direction))
    step = 1.0e-6
    finite_difference = (
        functional.evaluate(occupancy + step * direction).cavity_energy_hartree
        - functional.evaluate(occupancy - step * direction).cavity_energy_hartree
    ) / (2.0 * step)
    assert finite_difference == pytest.approx(analytic, rel=2.0e-7, abs=2.0e-11)


def test_weighted_density_cavity_is_periodically_translation_covariant_and_reciprocal():
    functional = _functional()
    grid = functional.grid
    occupancy = _occupancy(grid)
    shifted = np.roll(occupancy, (2, -1, 1), axis=(0, 1, 2))

    state = functional.evaluate(occupancy)
    shifted_state = functional.evaluate(shifted)
    gradient = functional.occupancy_functional_derivative_hartree_per_bohr3(state)
    shifted_gradient = functional.occupancy_functional_derivative_hartree_per_bohr3(
        shifted_state
    )
    assert shifted_state.cavity_energy_hartree == pytest.approx(
        state.cavity_energy_hartree,
        rel=0.0,
        abs=3.0e-15,
    )
    np.testing.assert_allclose(
        shifted_state.weighted_occupancy,
        np.roll(state.weighted_occupancy, (2, -1, 1), axis=(0, 1, 2)),
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        shifted_gradient,
        np.roll(gradient, (2, -1, 1), axis=(0, 1, 2)),
        rtol=0.0,
        atol=2.0e-14,
    )

    rng = np.random.default_rng(20260803)
    left = rng.normal(size=grid.shape)
    right = rng.normal(size=grid.shape)
    left_convolved = functional._convolution(left)
    right_convolved = functional._convolution(right)
    lhs = grid.volume_element_bohr3 * float(np.sum(left * right_convolved))
    rhs = grid.volume_element_bohr3 * float(np.sum(left_convolved * right))
    assert lhs == pytest.approx(rhs, rel=0.0, abs=5.0e-15)


def test_weighted_density_cavity_fails_closed_on_invalid_kernel_or_crossed_state():
    functional = _functional()
    grid = functional.grid
    with pytest.raises(ValueError, match="integrate to one"):
        Route2V0WeightedDensityCavityFunctional(
            grid=grid,
            thermodynamics=_thermodynamics(),
            nearest_neighbor_shell_kernel_per_bohr3=np.ones(grid.shape),
        )

    non_even = _kernel(grid).copy()
    non_even[1, 0, 0] += 0.01 / grid.volume_element_bohr3
    non_even[0, 0, 0] -= 0.01 / grid.volume_element_bohr3
    with pytest.raises(ValueError, match="must be even"):
        Route2V0WeightedDensityCavityFunctional(
            grid=grid,
            thermodynamics=_thermodynamics(),
            nearest_neighbor_shell_kernel_per_bohr3=non_even,
        )

    state = functional.evaluate(_occupancy(grid))
    other_thermodynamics = Route2V0WeightedDensityCavityThermodynamics(
        pressure_hartree_per_bohr3=functional.thermodynamics.pressure_hartree_per_bohr3,
        molecular_number_density_per_bohr3=(
            functional.thermodynamics.molecular_number_density_per_bohr3 * 1.01
        ),
        thermal_energy_hartree=functional.thermodynamics.thermal_energy_hartree,
        vapor_pressure_hartree_per_bohr3=(
            functional.thermodynamics.vapor_pressure_hartree_per_bohr3
        ),
        surface_tension_hartree_per_bohr2=(
            functional.thermodynamics.surface_tension_hartree_per_bohr2
        ),
        solvent_vdw_radius_bohr=functional.thermodynamics.solvent_vdw_radius_bohr,
    )
    other = Route2V0WeightedDensityCavityFunctional(
        grid=grid,
        thermodynamics=other_thermodynamics,
        nearest_neighbor_shell_kernel_per_bohr3=_kernel(grid),
    )
    with pytest.raises(ValueError, match="does not match"):
        other.occupancy_coordinate_gradient_hartree(state)
    with pytest.raises(ValueError, match="lie in"):
        functional.evaluate(np.full(grid.shape, 1.01))
    with pytest.raises(ValueError, match="read-only"):
        state.weighted_occupancy[0, 0, 0] = math.nan
