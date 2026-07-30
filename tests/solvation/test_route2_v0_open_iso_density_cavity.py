from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_auxiliary_ao_grid import (
    Route2V0AuxiliaryAODensityGridProjection,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_open_bspline import (
    build_route2_v0_open_cubic_bspline_stencil,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_open_iso_density_cavity import (
    V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION,
    V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_SCOPE,
    V0_OPEN_ISO_DENSITY_PRODUCT_CONVOLUTION,
    Route2V0OpenIsoDensityProductCavity,
    Route2V0OpenIsoDensityProductReactionOperator,
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


def _relative_gaussian_kernel(grid: RegularCartesianGrid) -> np.ndarray:
    """Return an arbitrary frozen relative kernel for structural tests only."""

    shape = (5, 5, 3)
    indices = np.indices(shape, dtype=float)
    distance_squared = np.zeros(shape, dtype=float)
    for axis, length in enumerate(shape):
        offset = (indices[axis] - length // 2) * grid.spacing_bohr[axis]
        distance_squared += offset**2
    values = np.exp(-distance_squared / (2.0 * 0.82**2))
    values /= grid.volume_element_bohr3 * float(np.sum(values))
    return values


def _cavity() -> Route2V0OpenIsoDensityProductCavity:
    grid = _grid()
    return Route2V0OpenIsoDensityProductCavity(
        grid=grid,
        solvent_electron_density_kernel_e_per_bohr3=_relative_gaussian_kernel(grid),
        overlap_threshold_e2_per_bohr3=0.020,
    )


def _electron_density(grid: RegularCartesianGrid) -> np.ndarray:
    points = grid.points_bohr().reshape((*grid.shape, 3))
    distance_squared = np.sum(
        (points - np.array([0.10, -0.15, 0.05])) ** 2,
        axis=-1,
    )
    return 0.003 + 0.180 * np.exp(-distance_squared / (2.0 * 0.85**2))


def _bounded_direction(
    grid: RegularCartesianGrid,
    rng: np.random.Generator,
) -> np.ndarray:
    direction = rng.normal(size=grid.shape)
    direction /= float(np.max(np.abs(direction)))
    return direction


def _direct_zero_extended_convolution(
    grid: RegularCartesianGrid,
    kernel: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Reference the declared relative-offset map without FFTs or wrapping."""

    centre = np.asarray(kernel.shape, dtype=int) // 2
    result = np.zeros(grid.shape, dtype=float)
    for output_index in np.ndindex(grid.shape):
        output = np.asarray(output_index, dtype=int)
        for source_index in np.ndindex(grid.shape):
            relative = centre + output - np.asarray(source_index, dtype=int)
            if np.all(relative >= 0) and np.all(relative < kernel.shape):
                result[output_index] += kernel[tuple(relative)] * values[source_index]
    return grid.volume_element_bohr3 * result


def _ao_projection(
    grid: RegularCartesianGrid,
) -> Route2V0AuxiliaryAODensityGridProjection:
    points = grid.points_bohr()
    raw = np.column_stack(
        (
            np.exp(-0.55 * np.sum((points - np.array([0.15, -0.1, 0.0])) ** 2, axis=1)),
            (points[:, 0] - 0.35)
            * np.exp(
                -0.42
                * np.sum(
                    (points - np.array([-0.25, 0.2, 0.1])) ** 2,
                    axis=1,
                )
            ),
        )
    )
    orthonormal, _ = np.linalg.qr(np.sqrt(grid.volume_element_bohr3) * raw)
    return Route2V0AuxiliaryAODensityGridProjection(
        grid=grid,
        ao_values=orthonormal / np.sqrt(grid.volume_element_bohr3),
        overlap_matrix=np.eye(2),
    )


def test_open_convolution_matches_direct_zero_extension_and_never_wraps():
    cavity = _cavity()
    grid = cavity.grid
    rng = np.random.default_rng(20260802)
    values = rng.normal(size=grid.shape)

    actual = cavity._convolution(values)
    expected = _direct_zero_extended_convolution(
        grid,
        cavity.solvent_electron_density_kernel_e_per_bohr3,
        values,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3.0e-14)

    lower_face_delta = np.zeros(grid.shape)
    lower_face_delta[0, 2, 2] = 1.0
    lower_face_response = cavity._convolution(lower_face_delta)
    assert lower_face_response[-1, 2, 2] == pytest.approx(0.0, abs=3.0e-15)
    assert lower_face_response[0, 2, 2] > 0.0

    left = rng.normal(size=grid.shape)
    right = rng.normal(size=grid.shape)
    left_right = grid.volume_element_bohr3 * float(
        np.sum(cavity._convolution(left) * right)
    )
    right_left = grid.volume_element_bohr3 * float(
        np.sum(left * cavity._adjoint_convolution(right))
    )
    assert left_right == pytest.approx(right_left, rel=0.0, abs=5.0e-14)
    assert V0_OPEN_ISO_DENSITY_PRODUCT_CONVOLUTION == "zero-extended-linear-v1"


def test_open_iso_density_cavity_is_smooth_and_has_an_exact_vjp():
    cavity = _cavity()
    grid = cavity.grid
    electron_density = _electron_density(grid)
    state = cavity.evaluate(electron_density)

    assert np.min(state.occupancy) < 0.10
    assert np.max(state.occupancy) > 0.90
    assert np.all(state.occupancy_derivative_per_overlap_bohr3_per_e2 <= 0.0)
    assert cavity.construction == V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION
    assert cavity.response_scope == V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_SCOPE
    assert cavity.is_total_solvation_asset is False

    rng = np.random.default_rng(20260803)
    direction = _bounded_direction(grid, rng)
    weights = rng.normal(size=grid.shape)
    coordinate_gradient = grid.volume_element_bohr3 * weights
    potential = cavity.occupancy_coordinate_vjp(state, coordinate_gradient)
    analytic = grid.volume_element_bohr3 * float(np.sum(direction * potential))
    step = 1.0e-6
    plus = cavity.evaluate(electron_density + step * direction)
    minus = cavity.evaluate(electron_density - step * direction)
    finite_difference = (
        grid.volume_element_bohr3
        * float(np.sum(weights * (plus.occupancy - minus.occupancy)))
        / (2.0 * step)
    )

    assert finite_difference == pytest.approx(analytic, rel=3.0e-6, abs=3.0e-8)


def test_open_composed_reaction_scalar_includes_cavity_chain_rule_without_neutrality():
    cavity = _cavity()
    grid = cavity.grid
    electron_density = _electron_density(grid)
    nuclear_density = electron_density + 0.008
    operator = Route2V0OpenIsoDensityProductReactionOperator(
        cavity=cavity,
        bulk_dielectric_constant=31.2,
        cg_relative_tolerance=1.0e-13,
    )

    state = operator.solve(nuclear_density, electron_density)
    potential = operator.electron_density_reaction_potential_hartree_per_e(state)
    direction = _bounded_direction(grid, np.random.default_rng(20260804))
    analytic = grid.volume_element_bohr3 * float(np.sum(direction * potential))
    step = 1.0e-6
    finite_difference = (
        operator.reaction_energy_hartree(
            nuclear_density, electron_density + step * direction
        )
        - operator.reaction_energy_hartree(
            nuclear_density, electron_density - step * direction
        )
    ) / (2.0 * step)

    assert state.continuum_state.reaction_energy_hartree < 0.0
    assert abs(state.continuum_state.total_grid_charge_e) > 1.0e-3
    assert finite_difference == pytest.approx(analytic, rel=5.0e-6, abs=3.0e-8)
    direct_charge_term = -state.continuum_state.reaction_potential_hartree_per_e
    assert float(np.max(np.abs(potential - direct_charge_term))) > 1.0e-8


def test_open_composed_reaction_pulls_back_to_the_ao_density_dual():
    cavity = _cavity()
    grid = cavity.grid
    projection = _ao_projection(grid)
    density_matrix = np.diag([1.16, 0.84])
    density_direction = np.diag([0.19, -0.19])
    electron_state = projection.evaluate(density_matrix)
    nuclear_stencil = build_route2_v0_open_cubic_bspline_stencil(
        grid=grid,
        positions_bohr=np.array([[0.12, -0.18, 0.08]]),
    )
    nuclear_density = (
        nuclear_stencil.deposit(np.array([2.4])) / grid.volume_element_bohr3
    )
    operator = Route2V0OpenIsoDensityProductReactionOperator(
        cavity=cavity,
        bulk_dielectric_constant=31.2,
        cg_relative_tolerance=1.0e-13,
    )
    state = operator.solve(nuclear_density, electron_state.electron_density_e_per_bohr3)
    potential = operator.electron_density_reaction_potential_hartree_per_e(state)
    electron_fock = projection.density_potential_matrix_hartree(potential)
    analytic = float(np.einsum("ij,ji->", density_direction, electron_fock))

    step = 1.0e-6
    plus = projection.evaluate(density_matrix + step * density_direction)
    minus = projection.evaluate(density_matrix - step * density_direction)
    finite_difference = (
        operator.reaction_energy_hartree(
            nuclear_density,
            plus.electron_density_e_per_bohr3,
        )
        - operator.reaction_energy_hartree(
            nuclear_density,
            minus.electron_density_e_per_bohr3,
        )
    ) / (2.0 * step)

    assert electron_state.grid_electron_count_e == pytest.approx(2.0, abs=2.0e-14)
    assert abs(state.continuum_state.total_grid_charge_e) > 1.0e-3
    assert finite_difference == pytest.approx(analytic, rel=5.0e-6, abs=3.0e-8)
    assert (
        projection.density_potential_pairing_error_hartree(
            density_direction,
            potential,
        )
        < 2.0e-14
    )


def test_open_iso_density_cavity_and_reaction_fail_closed_on_invalid_states():
    cavity = _cavity()
    grid = cavity.grid
    electron_density = _electron_density(grid)
    with pytest.raises(ValueError, match="odd shape"):
        Route2V0OpenIsoDensityProductCavity(
            grid=grid,
            solvent_electron_density_kernel_e_per_bohr3=np.ones((4, 3, 3)),
            overlap_threshold_e2_per_bohr3=0.020,
        )
    asymmetric = _relative_gaussian_kernel(grid)
    asymmetric[0, 0, 0] *= 1.1
    with pytest.raises(ValueError, match="centrosymmetric"):
        Route2V0OpenIsoDensityProductCavity(
            grid=grid,
            solvent_electron_density_kernel_e_per_bohr3=asymmetric,
            overlap_threshold_e2_per_bohr3=0.020,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        cavity.evaluate(-electron_density)

    operator = Route2V0OpenIsoDensityProductReactionOperator(
        cavity=cavity,
        bulk_dielectric_constant=31.2,
    )
    state = operator.solve(electron_density + 0.005, electron_density)
    other = Route2V0OpenIsoDensityProductReactionOperator(
        cavity=cavity,
        bulk_dielectric_constant=20.0,
    )
    with pytest.raises(ValueError, match="bulk dielectric does not match"):
        other.electron_density_reaction_potential_hartree_per_e(state)
    with pytest.raises(ValueError, match="read-only"):
        state.cavity_state.occupancy[0, 0, 0] = math.nan
    with pytest.raises(ValueError, match="charge split"):
        replace(
            state,
            total_charge_density_e_per_bohr3=state.total_charge_density_e_per_bohr3
            + 0.1,
        )
    with pytest.raises(ValueError, match="electron density/cavity mismatch"):
        replace(
            state,
            nuclear_charge_density_e_per_bohr3=state.nuclear_charge_density_e_per_bohr3
            + 0.001,
            solute_electron_density_e_per_bohr3=(
                state.solute_electron_density_e_per_bohr3 + 0.001
            ),
        )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        replace(state, cavity_operator_fingerprint="0" * 64)
