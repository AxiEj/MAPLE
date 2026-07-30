from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_open_weighted_density_cavity import (
    V0_OPEN_WEIGHTED_DENSITY_CAVITY_BOUNDARY,
    V0_OPEN_WEIGHTED_DENSITY_CAVITY_CONSTRUCTION,
    V0_OPEN_WEIGHTED_DENSITY_CAVITY_SCOPE,
    Route2V0OpenWeightedDensityCavityFunctional,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_weighted_density_cavity import (
    Route2V0WeightedDensityCavityThermodynamics,
)


def _grid() -> RegularCartesianGrid:
    return RegularCartesianGrid(
        origin_bohr=np.array([-3.2, -2.8, -2.4]),
        spacing_bohr=np.array([0.7, 0.8, 0.9]),
        shape=(9, 8, 7),
    )


def _thermodynamics() -> Route2V0WeightedDensityCavityThermodynamics:
    """Return fixed pure-liquid structural inputs, not a solvent asset."""

    return Route2V0WeightedDensityCavityThermodynamics.from_si(
        temperature_kelvin=298.15,
        pressure_pascal=101_325.0,
        molecular_number_density_angstrom3=0.03336,
        vapor_pressure_pascal=3_169.0,
        surface_tension_newton_per_meter=0.07197,
        solvent_vdw_radius_angstrom=1.385,
    )


def _kernel(grid: RegularCartesianGrid) -> np.ndarray:
    """Return a centered positive shell-control kernel of unit grid integral."""

    weights = np.zeros((3, 3, 3), dtype=float)
    weights[1, 1, 1] = 0.40
    for axis in range(3):
        positive = [1, 1, 1]
        negative = [1, 1, 1]
        positive[axis] += 1
        negative[axis] -= 1
        weights[tuple(positive)] = 0.10
        weights[tuple(negative)] = 0.10
    assert float(np.sum(weights)) == pytest.approx(1.0)
    return weights / grid.volume_element_bohr3


def _functional() -> Route2V0OpenWeightedDensityCavityFunctional:
    grid = _grid()
    return Route2V0OpenWeightedDensityCavityFunctional(
        grid=grid,
        thermodynamics=_thermodynamics(),
        nearest_neighbor_shell_kernel_per_bohr3=_kernel(grid),
    )


def _direct_zero_extended_full_convolution(
    grid: RegularCartesianGrid,
    kernel: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Reference the complete relative convolution by an explicit no-wrap loop."""

    result = np.zeros(
        tuple(grid.shape[axis] + kernel.shape[axis] - 1 for axis in range(3)),
        dtype=float,
    )
    for output_index in np.ndindex(result.shape):
        output = np.asarray(output_index, dtype=int)
        for source_index in np.ndindex(grid.shape):
            relative = output - np.asarray(source_index, dtype=int)
            if np.all(relative >= 0) and np.all(relative < kernel.shape):
                result[output_index] += kernel[tuple(relative)] * values[source_index]
    return grid.volume_element_bohr3 * result


def _occupancy(grid: RegularCartesianGrid) -> np.ndarray:
    rng = np.random.default_rng(20260806)
    return 0.20 + 0.60 * rng.random(grid.shape)


def test_open_shell_convolution_matches_direct_reference_and_is_self_adjoint():
    functional = _functional()
    grid = functional.grid
    rng = np.random.default_rng(20260807)
    values = rng.normal(size=grid.shape)

    actual_full = functional._full_linear_convolution(values)
    expected_full = _direct_zero_extended_full_convolution(
        grid,
        functional.nearest_neighbor_shell_kernel_per_bohr3,
        values,
    )
    np.testing.assert_allclose(actual_full, expected_full, rtol=0.0, atol=3.0e-14)
    np.testing.assert_allclose(
        functional._convolution(values),
        expected_full[functional.interior_slices],
        rtol=0.0,
        atol=3.0e-14,
    )

    lower_face_delta = np.zeros(grid.shape)
    lower_face_delta[0, 3, 3] = 1.0
    lower_face_response = functional._convolution(lower_face_delta)
    assert lower_face_response[-1, 3, 3] == pytest.approx(0.0, abs=3.0e-15)
    assert lower_face_response[0, 3, 3] > 0.0

    left = rng.normal(size=grid.shape)
    right = rng.normal(size=functional.extended_shape)
    left_right = grid.volume_element_bohr3 * float(
        np.sum(functional._full_linear_convolution(left) * right)
    )
    right_left = grid.volume_element_bohr3 * float(
        np.sum(left * functional._full_adjoint_convolution(right))
    )
    assert left_right == pytest.approx(right_left, rel=0.0, abs=5.0e-14)


def test_open_cavitation_uses_bulk_outside_the_box_not_periodic_or_vacuum_extension():
    functional = _functional()
    grid = functional.grid

    bulk = functional.evaluate(np.ones(grid.shape))
    finite_void = functional.evaluate(np.zeros(grid.shape))

    assert functional.construction == V0_OPEN_WEIGHTED_DENSITY_CAVITY_CONSTRUCTION
    assert functional.response_scope == V0_OPEN_WEIGHTED_DENSITY_CAVITY_SCOPE
    assert functional.is_total_solvation_asset is False
    assert bulk.boundary_condition == V0_OPEN_WEIGHTED_DENSITY_CAVITY_BOUNDARY
    assert bulk.cavity_energy_hartree == pytest.approx(0.0, abs=2.0e-15)
    np.testing.assert_allclose(bulk.weighted_occupancy, 1.0, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        bulk.extended_weighted_occupancy,
        1.0,
        rtol=0.0,
        atol=2.0e-15,
    )

    centre = tuple(length // 2 for length in grid.shape)
    assert finite_void.weighted_occupancy[centre] == pytest.approx(0.0, abs=2.0e-15)
    assert finite_void.weighted_occupancy[0, 3, 3] > 0.0
    assert finite_void.extended_weighted_occupancy[0, 4, 4] > 0.0
    assert finite_void.cavity_energy_hartree > 0.0


def test_open_cavitation_has_an_exact_bulk_continued_occupancy_derivative():
    functional = _functional()
    grid = functional.grid
    occupancy = _occupancy(grid)
    state = functional.evaluate(occupancy)
    rng = np.random.default_rng(20260808)
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
    assert finite_difference == pytest.approx(analytic, rel=3.0e-7, abs=2.0e-11)


def test_open_cavitation_fails_closed_on_bad_kernel_and_crossed_state():
    functional = _functional()
    grid = functional.grid
    with pytest.raises(ValueError, match="integrate to one"):
        Route2V0OpenWeightedDensityCavityFunctional(
            grid=grid,
            thermodynamics=_thermodynamics(),
            nearest_neighbor_shell_kernel_per_bohr3=np.ones((3, 3, 3)),
        )

    non_even = _kernel(grid).copy()
    non_even[0, 1, 1] += 0.01 / grid.volume_element_bohr3
    non_even[1, 1, 1] -= 0.01 / grid.volume_element_bohr3
    with pytest.raises(ValueError, match="centrosymmetric"):
        Route2V0OpenWeightedDensityCavityFunctional(
            grid=grid,
            thermodynamics=_thermodynamics(),
            nearest_neighbor_shell_kernel_per_bohr3=non_even,
        )

    state = functional.evaluate(_occupancy(grid))
    different_thermodynamics = Route2V0WeightedDensityCavityThermodynamics(
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
    other = Route2V0OpenWeightedDensityCavityFunctional(
        grid=grid,
        thermodynamics=different_thermodynamics,
        nearest_neighbor_shell_kernel_per_bohr3=_kernel(grid),
    )
    with pytest.raises(ValueError, match="does not match"):
        other.occupancy_coordinate_gradient_hartree(state)
    with pytest.raises(ValueError, match="extended support"):
        functional.occupancy_coordinate_gradient_hartree(
            replace(
                state,
                extended_weighted_occupancy=state.extended_weighted_occupancy[
                    :-1,
                    :,
                    :,
                ],
            )
        )
    with pytest.raises(ValueError, match="lie in"):
        functional.evaluate(np.full(grid.shape, 1.01))
    with pytest.raises(ValueError, match="read-only"):
        state.weighted_occupancy[0, 0, 0] = math.nan
