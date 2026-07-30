from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_auxiliary_ao_grid import (
    V0_AUXILIARY_AO_GRID_CONSTRUCTION,
    V0_AUXILIARY_AO_GRID_SCOPE,
    Route2V0AuxiliaryAODensityGridProjection,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_iso_density_cavity import (
    Route2V0IsoDensityProductCavity,
    Route2V0IsoDensityProductReactionOperator,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_periodic_bspline import (
    build_route2_v0_periodic_cubic_bspline_stencil,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)


def _grid() -> RegularCartesianGrid:
    return RegularCartesianGrid(
        origin_bohr=np.array([-2.4, -2.0, -1.6]),
        spacing_bohr=np.array([0.8, 0.8, 0.8]),
        shape=(7, 6, 5),
    )


def _projection() -> Route2V0AuxiliaryAODensityGridProjection:
    grid = _grid()
    points = grid.points_bohr()
    raw = np.column_stack(
        (
            np.exp(-0.55 * np.sum((points - np.array([0.2, -0.1, 0.0])) ** 2, axis=1)),
            (points[:, 0] - 0.35)
            * np.exp(
                -0.42 * np.sum((points - np.array([-0.3, 0.2, 0.1])) ** 2, axis=1)
            ),
        )
    )
    # This is a controlled discrete AO basis.  Its overlap is exactly the
    # uniform-grid quadrature overlap, so count error tests do not hide behind
    # a finite-box integration mismatch.
    orthonormal, _ = np.linalg.qr(np.sqrt(grid.volume_element_bohr3) * raw)
    ao_values = orthonormal / np.sqrt(grid.volume_element_bohr3)
    return Route2V0AuxiliaryAODensityGridProjection(
        grid=grid,
        ao_values=ao_values,
        overlap_matrix=np.eye(2),
    )


def _periodic_gaussian(grid: RegularCartesianGrid, width_bohr: float) -> np.ndarray:
    indices = np.indices(grid.shape, dtype=float)
    distance_squared = np.zeros(grid.shape, dtype=float)
    for axis, length in enumerate(grid.shape):
        delta = np.minimum(indices[axis], length - indices[axis])
        distance_squared += (delta * grid.spacing_bohr[axis]) ** 2
    values = np.exp(-distance_squared / (2.0 * width_bohr**2))
    values /= grid.volume_element_bohr3 * float(np.sum(values))
    return values


def test_auxiliary_ao_grid_has_exact_density_potential_duality_and_exposes_count_error():
    projection = _projection()
    density_matrix = np.array([[1.2, 0.16], [0.16, 0.8]])
    potential = np.linspace(-0.11, 0.17, projection.grid.point_count).reshape(
        projection.grid.shape
    )

    state = projection.evaluate(density_matrix)
    pullback = projection.density_potential_matrix_hartree(potential)

    assert state.ao_electron_count_e == pytest.approx(2.0, abs=2.0e-14)
    assert state.grid_electron_count_e == pytest.approx(2.0, abs=2.0e-14)
    assert state.grid_electron_count_error_e == pytest.approx(0.0, abs=2.0e-14)
    assert np.min(state.electron_density_e_per_bohr3) >= 0.0
    np.testing.assert_allclose(pullback, pullback.T, rtol=0.0, atol=2.0e-14)
    assert (
        projection.density_potential_pairing_error_hartree(density_matrix, potential)
        < 2.0e-14
    )
    assert projection.construction == V0_AUXILIARY_AO_GRID_CONSTRUCTION
    assert projection.scope == V0_AUXILIARY_AO_GRID_SCOPE
    projection.validate_state(state)

    # A real electronic method supplies its AO overlap independently of this
    # finite box.  The bridge must reveal the mismatch instead of changing the
    # grid density to make the two counts agree.
    finite_box_projection = Route2V0AuxiliaryAODensityGridProjection(
        grid=projection.grid,
        ao_values=projection.ao_values,
        overlap_matrix=1.15 * np.eye(2),
    )
    finite_box_state = finite_box_projection.evaluate(np.eye(2))
    assert finite_box_state.ao_electron_count_e == pytest.approx(2.3)
    assert finite_box_state.grid_electron_count_e == pytest.approx(2.0)
    assert finite_box_state.grid_electron_count_error_e == pytest.approx(-0.3)


def test_auxiliary_ao_grid_pulls_back_the_full_density_defined_reaction_derivative():
    projection = _projection()
    density_matrix = np.diag([1.12, 0.88])
    density_direction = np.diag([0.23, -0.23])
    electron_state = projection.evaluate(density_matrix)

    nuclear_stencil = build_route2_v0_periodic_cubic_bspline_stencil(
        grid=projection.grid,
        positions_bohr=np.array([[0.15, -0.20, 0.10]]),
    )
    nuclear_density = (
        nuclear_stencil.deposit(np.array([2.0])) / projection.grid.volume_element_bohr3
    )
    cavity = Route2V0IsoDensityProductCavity(
        grid=projection.grid,
        solvent_electron_density_kernel_e_per_bohr3=_periodic_gaussian(
            projection.grid,
            width_bohr=0.95,
        ),
        overlap_threshold_e2_per_bohr3=0.028,
    )
    reaction = Route2V0IsoDensityProductReactionOperator(
        cavity=cavity,
        bulk_dielectric_constant=31.2,
        cg_relative_tolerance=1.0e-13,
    )
    reaction_state = reaction.solve(
        nuclear_density,
        electron_state.electron_density_e_per_bohr3,
    )
    electron_potential = reaction.electron_density_reaction_potential_hartree_per_e(
        reaction_state
    )
    reaction_fock = projection.density_potential_matrix_hartree(electron_potential)
    analytic = float(np.einsum("ij,ji->", density_direction, reaction_fock))

    step = 1.0e-6
    plus = projection.evaluate(density_matrix + step * density_direction)
    minus = projection.evaluate(density_matrix - step * density_direction)
    finite_difference = (
        reaction.reaction_energy_hartree(
            nuclear_density, plus.electron_density_e_per_bohr3
        )
        - reaction.reaction_energy_hartree(
            nuclear_density, minus.electron_density_e_per_bohr3
        )
    ) / (2.0 * step)

    assert reaction_state.continuum_state.reaction_energy_hartree < 0.0
    assert finite_difference == pytest.approx(analytic, rel=8.0e-6, abs=4.0e-8)
    assert (
        projection.density_potential_pairing_error_hartree(
            density_direction,
            electron_potential,
        )
        < 2.0e-14
    )
    direct_charge_term = (
        -reaction_state.continuum_state.reaction_potential_hartree_per_e
    )
    assert float(np.max(np.abs(electron_potential - direct_charge_term))) > 1.0e-8


def test_auxiliary_ao_grid_rejects_nonphysical_or_crossed_representation_states():
    projection = _projection()
    state = projection.evaluate(np.diag([1.0, 1.0]))

    with pytest.raises(ValueError, match="real-valued"):
        projection.evaluate(np.eye(2, dtype=complex))
    with pytest.raises(ValueError, match="symmetric"):
        projection.evaluate(np.array([[1.0, 0.2], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="shape"):
        projection.density_potential_matrix_hartree(np.ones((2, 2)))
    with pytest.raises(ValueError, match="positive definite"):
        Route2V0AuxiliaryAODensityGridProjection(
            grid=projection.grid,
            ao_values=projection.ao_values,
            overlap_matrix=np.array([[1.0, 2.0], [2.0, 1.0]]),
        )
    with pytest.raises(ValueError, match="does not match"):
        Route2V0AuxiliaryAODensityGridProjection(
            grid=projection.grid,
            ao_values=np.roll(projection.ao_values, 1, axis=0),
            overlap_matrix=projection.overlap_matrix,
        ).validate_state(state)
    with pytest.raises(ValueError, match="inconsistent"):
        projection.validate_state(
            replace(
                state,
                electron_density_e_per_bohr3=np.roll(
                    state.electron_density_e_per_bohr3,
                    1,
                    axis=0,
                ),
            )
        )
    with pytest.raises(ValueError, match="grid_electron_count_e is inconsistent"):
        projection.validate_state(
            replace(
                state,
                grid_electron_count_e=state.grid_electron_count_e + 0.1,
            )
        )
