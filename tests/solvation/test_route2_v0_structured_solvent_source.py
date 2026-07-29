from __future__ import annotations

import math

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.gto_density import (
    MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
    gaussian_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
    V0_STRUCTURED_SOLVENT_ELECTROSTATIC_GRID_CONSTRUCTION,
    V0_STRUCTURED_SOLVENT_GRID_LAYOUT,
    evaluate_route2_v0_structured_solvent_electrostatic_source,
)


def _grid() -> RegularCartesianGrid:
    return RegularCartesianGrid(
        origin_bohr=np.array([-2.0, -1.5, -1.0]),
        spacing_bohr=np.array([0.7, 0.5, 0.9]),
        shape=(4, 5, 3),
    )


def _source(grid: RegularCartesianGrid | None = None):
    density = np.array(
        [
            [0.35, 0.08, -0.04, 0.03],
            [-0.35, -0.06, 0.02, -0.05],
        ]
    )
    positions = np.array(
        [[-0.4, 0.2, 0.1], [0.7, -0.3, 0.5]],
        dtype=float,
    )
    return evaluate_route2_v0_structured_solvent_electrostatic_source(
        density_coefficients=density,
        atom_positions_angstrom=positions,
        grid=_grid() if grid is None else grid,
    ), density, positions


def test_regular_cartesian_grid_coordinates_follow_declared_c_order_layout():
    grid = _grid()

    assert grid.layout == V0_STRUCTURED_SOLVENT_GRID_LAYOUT
    assert grid.point_count == 60
    assert grid.volume_element_bohr3 == pytest.approx(0.315)
    points = grid.points_bohr()
    assert points.flags.writeable is False
    assert points[0] == pytest.approx(grid.origin_bohr)
    assert points[1] == pytest.approx(
        grid.origin_bohr + np.array([0.0, 0.0, grid.spacing_bohr[2]])
    )
    assert points[grid.shape[2]] == pytest.approx(
        grid.origin_bohr + np.array([0.0, grid.spacing_bohr[1], 0.0])
    )


def test_structured_solvent_source_is_exactly_the_gaussian_mace_grid_source():
    source, density, positions = _source()

    expected = gaussian_multipole_potential(
        source.grid.points_bohr(),
        positions,
        density,
        sigma_angstrom=MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
    ).reshape(source.grid.shape)
    np.testing.assert_allclose(
        source.potential_hartree_per_e,
        expected,
        rtol=0.0,
        atol=2.0e-15,
    )
    assert source.construction == V0_STRUCTURED_SOLVENT_ELECTROSTATIC_GRID_CONSTRUCTION
    assert source.interaction_scope == "electrostatic-grid-source-only-v1"
    assert source.density_coefficients.flags.writeable is False
    assert source.atom_positions_angstrom.flags.writeable is False
    assert source.potential_hartree_per_e.flags.writeable is False


def test_gaussian_grid_source_is_translation_covariant():
    source, density, positions = _source()
    shift_angstrom = np.array([0.17, -0.29, 0.41])
    shifted_grid = RegularCartesianGrid(
        origin_bohr=source.grid.origin_bohr + shift_angstrom / Bohr,
        spacing_bohr=source.grid.spacing_bohr,
        shape=source.grid.shape,
    )
    shifted = evaluate_route2_v0_structured_solvent_electrostatic_source(
        density_coefficients=density,
        atom_positions_angstrom=positions + shift_angstrom,
        grid=shifted_grid,
    )

    np.testing.assert_allclose(
        shifted.potential_hartree_per_e,
        source.potential_hartree_per_e,
        rtol=0.0,
        atol=2.0e-15,
    )


def test_gaussian_source_is_finite_at_an_atomic_center_and_has_the_known_limit():
    grid = RegularCartesianGrid(
        origin_bohr=np.zeros(3),
        spacing_bohr=np.ones(3),
        shape=(1, 1, 1),
    )
    source = evaluate_route2_v0_structured_solvent_electrostatic_source(
        density_coefficients=np.array([[1.0, 0.0, 0.0, 0.0]]),
        atom_positions_angstrom=np.zeros((1, 3)),
        grid=grid,
        target_total_charge_e=1.0,
    )

    sigma_bohr = MACE_POLAR_DENSITY_SIGMA_ANGSTROM / Bohr
    assert source.potential_hartree_per_e[0, 0, 0] == pytest.approx(
        math.sqrt(2.0 / math.pi) / sigma_bohr,
        rel=0.0,
        abs=2.0e-15,
    )


def test_site_energy_is_only_the_charge_scaled_electrostatic_summand():
    source, _, _ = _source()

    energy = source.site_electrostatic_energy_hartree(-0.834)
    np.testing.assert_allclose(
        energy,
        -0.834 * source.potential_hartree_per_e,
        rtol=0.0,
        atol=0.0,
    )
    assert energy.flags.writeable is False
    with pytest.raises(ValueError, match="finite"):
        source.site_electrostatic_energy_hartree(float("nan"))


def test_source_fails_closed_on_charge_and_grid_contract_violations():
    grid = _grid()
    with pytest.raises(ValueError, match="total-charge constraint"):
        evaluate_route2_v0_structured_solvent_electrostatic_source(
            density_coefficients=np.array([[0.2, 0.0, 0.0, 0.0]]),
            atom_positions_angstrom=np.zeros((1, 3)),
            grid=grid,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        RegularCartesianGrid(
            origin_bohr=np.zeros(3),
            spacing_bohr=np.array([1.0, 0.0, 1.0]),
            shape=(1, 1, 1),
        )
    with pytest.raises(ValueError, match="positive integers"):
        RegularCartesianGrid(
            origin_bohr=np.zeros(3),
            spacing_bohr=np.ones(3),
            shape=(1, 0, 1),
        )
