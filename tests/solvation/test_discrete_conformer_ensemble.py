from __future__ import annotations

import numpy as np
import pytest


def _analyze(**kwargs):
    from maple.function.free_energy.discrete_conformers import (
        analyze_discrete_conformer_ensemble,
    )

    return analyze_discrete_conformer_ensemble(**kwargs)


def test_discrete_conformer_ensemble_uses_only_relative_gas_energies():
    arguments = {
        "gas_energy_kcal_mol": [-10.0, -7.0],
        "solvent_correction_kcal_mol": [-3.0, -6.0],
        "temperature_kelvin": 298.15,
        "state_ids": ["a", "b"],
    }
    result = _analyze(**arguments)
    shifted = _analyze(
        **{
            **arguments,
            "gas_energy_kcal_mol": [999990.0, 999993.0],
        }
    )

    assert result["delta_g_discrete_kcal_mol"] == pytest.approx(-3.406, abs=0.002)
    assert result["gas_weights"][1] == pytest.approx(0.0063, abs=0.0002)
    assert result["solution_weights"] == pytest.approx([0.5, 0.5], abs=1.0e-12)
    assert shifted["delta_g_discrete_kcal_mol"] == pytest.approx(
        result["delta_g_discrete_kcal_mol"],
        abs=1.0e-10,
    )
    assert result["gas_energy_zero_kcal_mol"] == -10.0
    assert result["gas_relative_energy_kcal_mol"] == [0.0, 3.0]


def test_single_conformer_reduces_to_endpoint_but_cannot_open_ensemble_claim():
    result = _analyze(
        gas_energy_kcal_mol=[-123.456],
        solvent_correction_kcal_mol=[-4.25],
        temperature_kelvin=298.15,
        state_ids=["only"],
    )

    assert result["delta_g_discrete_kcal_mol"] == pytest.approx(-4.25)
    assert result["gas_weights"] == [1.0]
    assert result["solution_weights"] == [1.0]
    assert result["gates"]["checks"]["minimum_state_count"] is False
    assert result["gates"]["ensemble_diagnostic_passed"] is False
    assert result["claim_boundary"]["public_solvfe_eligible"] is False
    assert result["claim_boundary"]["hydration_free_energy_claim"] is False


def test_distribution_diagnostics_fail_closed_on_disjoint_dominant_states():
    result = _analyze(
        gas_energy_kcal_mol=[0.0, 20.0],
        solvent_correction_kcal_mol=[0.0, -40.0],
        temperature_kelvin=298.15,
        state_ids=["gas", "solution"],
    )

    assert result["gas_dominant_state_id"] == "gas"
    assert result["solution_dominant_state_id"] == "solution"
    assert result["distribution_overlap"] < 1.0e-6
    assert result["gates"]["checks"]["minimum_distribution_overlap"] is False
    assert result["gates"]["checks"]["maximum_gas_weight"] is False
    assert result["gates"]["checks"]["maximum_solution_weight"] is False
    assert result["gates"]["ensemble_diagnostic_passed"] is False


def test_discrete_result_is_permutation_invariant_and_bounded_by_endpoints():
    gas = np.asarray([4.0, -2.0, 1.0, 8.0])
    solvent = np.asarray([-7.0, 2.0, -1.5, 4.0])
    state_ids = ["a", "b", "c", "d"]
    result = _analyze(
        gas_energy_kcal_mol=gas,
        solvent_correction_kcal_mol=solvent,
        temperature_kelvin=310.0,
        state_ids=state_ids,
    )
    permutation = [2, 0, 3, 1]
    permuted = _analyze(
        gas_energy_kcal_mol=gas[permutation],
        solvent_correction_kcal_mol=solvent[permutation],
        temperature_kelvin=310.0,
        state_ids=[state_ids[index] for index in permutation],
    )

    assert permuted["delta_g_discrete_kcal_mol"] == pytest.approx(
        result["delta_g_discrete_kcal_mol"],
        abs=1.0e-12,
    )
    assert solvent.min() <= result["delta_g_discrete_kcal_mol"] <= solvent.max()
    assert result["identities"]["endpoint_bounds_satisfied"] is True
    assert result["identities"]["partition_ratio_closure_abs_kcal_mol"] < 1.0e-12


def test_logsumexp_path_remains_finite_for_extreme_energy_separation():
    result = _analyze(
        gas_energy_kcal_mol=[0.0, 1.0e6],
        solvent_correction_kcal_mol=[0.0, -2.0e6],
        temperature_kelvin=298.15,
        state_ids=["gas", "solution"],
    )

    assert np.isfinite(result["delta_g_discrete_kcal_mol"])
    assert np.isfinite(result["gas_weights"]).all()
    assert np.isfinite(result["solution_weights"]).all()
    assert sum(result["gas_weights"]) == pytest.approx(1.0)
    assert sum(result["solution_weights"]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {
                "gas_energy_kcal_mol": [],
                "solvent_correction_kcal_mol": [],
                "temperature_kelvin": 298.15,
            },
            "non-empty",
        ),
        (
            {
                "gas_energy_kcal_mol": [0.0, 1.0],
                "solvent_correction_kcal_mol": [0.0],
                "temperature_kelvin": 298.15,
            },
            "equal length",
        ),
        (
            {
                "gas_energy_kcal_mol": [0.0, float("nan")],
                "solvent_correction_kcal_mol": [0.0, 1.0],
                "temperature_kelvin": 298.15,
            },
            "finite",
        ),
        (
            {
                "gas_energy_kcal_mol": [0.0],
                "solvent_correction_kcal_mol": [0.0],
                "temperature_kelvin": 0.0,
            },
            "temperature_kelvin",
        ),
        (
            {
                "gas_energy_kcal_mol": [0.0, 1.0],
                "solvent_correction_kcal_mol": [0.0, 1.0],
                "temperature_kelvin": 298.15,
                "state_ids": ["duplicate", "duplicate"],
            },
            "unique",
        ),
        (
            {
                "gas_energy_kcal_mol": [0.0, 1.0],
                "solvent_correction_kcal_mol": [0.0, 1.0],
                "temperature_kelvin": 298.15,
                "minimum_distribution_overlap": 1.1,
            },
            "minimum_distribution_overlap",
        ),
    ],
)
def test_discrete_conformer_inputs_fail_closed(arguments, message):
    with pytest.raises(ValueError, match=message):
        _analyze(**arguments)


def test_result_excludes_labels_residuals_and_unearned_thermodynamic_claims():
    result = _analyze(
        gas_energy_kcal_mol=[0.0, 0.2, 0.5],
        solvent_correction_kcal_mol=[-2.0, -2.1, -1.8],
        temperature_kelvin=298.15,
        state_ids=["a", "b", "c"],
    )

    assert result["route_contract"] == {
        "gas_phase_mm_energy": False,
        "hydration_label_residual": False,
        "mlip_retraining": False,
        "fixed_charge_solvent_correction": True,
    }
    assert result["claim_boundary"]["equal_basin_measure_assumed"] is True
    assert result["claim_boundary"]["basin_volumes_included"] is False
    assert result["claim_boundary"]["vibrational_free_energies_included"] is False
    assert result["claim_boundary"]["experimental_labels_used"] is False
    assert result["uncertainty_kcal_mol"] is None
