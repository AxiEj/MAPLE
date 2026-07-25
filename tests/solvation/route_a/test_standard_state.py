from __future__ import annotations

import math

from .conftest import FIXTURE_DIR, load_json

STANDARD_FIX = FIXTURE_DIR / "standard_state" / "standard_state.json"


def _rt(t, r):
    return t * r


def test_standard_state_formula_uses_molecule_reference_volume():
    spec = load_json(STANDARD_FIX)
    assert spec["standard_state_volume_A3_per_molecule"] > 0.0
    assert math.isclose(spec["standard_state_volume_A3_per_molecule"], 1660.5390671738467, abs_tol=0.0)


def test_standard_state_uses_deltaG_volume_reference_and_density_relation():
    spec = load_json(STANDARD_FIX)
    t = spec["temperature_k"]
    r = spec["gas_constant_kcal_per_mol_k"]
    c0 = spec["C_number_0_per_A3"]

    # C0 is by number density; zero correction at V0
    delta_g0 = -_rt(t, r) * math.log(spec["standard_state_volume_A3_per_molecule"] * c0)
    assert math.isclose(delta_g0, 0.0, abs_tol=1e-12)

    rho_w_number = spec["rho_W_number_per_A3"]
    rho_ratio = spec["rho_W_over_C0"]
    water_bulk_concentration = spec["water_bulk_concentration_molar"]
    solution_reference = spec["solution_reference_concentration_molar"]

    assert math.isclose(rho_w_number, 0.03332953803622, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(rho_ratio, 55.345, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(rho_w_number, water_bulk_concentration * c0 / solution_reference, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(rho_ratio, rho_w_number / c0, rel_tol=0.0, abs_tol=1e-12)

    # only the dimensionless rho_W/C0 ratio enters the logarithm
    density_term = -_rt(t, r) * math.log(rho_ratio)
    assert density_term < 0


def test_standard_state_unit_notes_are_explicit_and_distinct():
    spec = load_json(STANDARD_FIX)
    assert spec["unit_notes"]["volume_in"] == "angstrom3"
    assert spec["unit_notes"]["water_bulk_concentration_molar_units"] == "mol/L"
    assert spec["unit_notes"]["rho_W_number_per_A3_units"] == "A^-3"
    assert spec["unit_notes"]["rho_W_over_C0_units"] == "dimensionless"
    assert "deltaG_vol" in spec["formula"]
