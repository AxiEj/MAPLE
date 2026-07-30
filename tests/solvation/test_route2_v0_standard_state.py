from __future__ import annotations

import math

import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_standard_state import (
    V0_STANDARD_STATE_CONSTRUCTION,
    V0_STANDARD_STATE_SCOPE,
    evaluate_route2_v0_standard_state_correction,
)


def test_one_atmosphere_to_one_molar_conversion_has_the_exact_thermodynamic_sign():
    state = evaluate_route2_v0_standard_state_correction(temperature_kelvin=298.15)

    expected_ratio = 1_000.0 * 8.31446261815324 * 298.15 / 101_325.0
    expected_joule_per_mol = 8.31446261815324 * 298.15 * math.log(expected_ratio)

    assert state.construction == V0_STANDARD_STATE_CONSTRUCTION
    assert state.scope == V0_STANDARD_STATE_SCOPE
    assert state.dimensionless_standard_ratio == pytest.approx(expected_ratio)
    assert state.correction_joule_per_mol == pytest.approx(expected_joule_per_mol)
    assert state.correction_kcal_per_mol == pytest.approx(1.893, abs=0.002)
    assert state.correction_joule_per_mol > 0.0
    assert state.correction_hartree_per_molecule > 0.0
    assert state.coordinate_gradient_hartree_per_bohr == 0.0
    assert state.is_total_solvation_asset is False


def test_matched_ideal_gas_concentration_standard_has_zero_conversion():
    temperature = 310.0
    pressure = 100_000.0
    concentration_mol_per_liter = pressure / (1_000.0 * 8.31446261815324 * temperature)

    state = evaluate_route2_v0_standard_state_correction(
        temperature_kelvin=temperature,
        gas_standard_pressure_pascal=pressure,
        solution_standard_concentration_mol_per_liter=concentration_mol_per_liter,
    )

    assert state.dimensionless_standard_ratio == pytest.approx(1.0)
    assert state.correction_joule_per_mol == pytest.approx(0.0, abs=1.0e-12)
    assert state.correction_kcal_per_mol == pytest.approx(0.0, abs=1.0e-15)
    assert state.temperature_derivative_joule_per_mol_kelvin == pytest.approx(
        8.31446261815324
    )


def test_standard_state_temperature_derivative_matches_central_difference():
    temperature = 298.15
    step = 1.0e-3
    state = evaluate_route2_v0_standard_state_correction(temperature_kelvin=temperature)
    lower = evaluate_route2_v0_standard_state_correction(
        temperature_kelvin=temperature - step
    )
    upper = evaluate_route2_v0_standard_state_correction(
        temperature_kelvin=temperature + step
    )

    finite_difference = (
        upper.correction_joule_per_mol - lower.correction_joule_per_mol
    ) / (2.0 * step)
    assert state.temperature_derivative_joule_per_mol_kelvin == pytest.approx(
        finite_difference,
        rel=2.0e-11,
        abs=1.0e-9,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature_kelvin": 0.0}, "temperature_kelvin"),
        ({"gas_standard_pressure_pascal": float("nan")}, "pressure"),
        ({"solution_standard_concentration_mol_per_liter": -1.0}, "concentration"),
    ],
)
def test_standard_state_rejects_nonphysical_standards(
    kwargs: dict[str, float], message: str
):
    parameters = {"temperature_kelvin": 298.15}
    parameters.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        evaluate_route2_v0_standard_state_correction(**parameters)
