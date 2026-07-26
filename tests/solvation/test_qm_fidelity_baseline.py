from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = (
    REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/route2-qm-fidelity-v1.json"
)
GAS_CONSTANT_KCAL_MOL_K = 0.00198720425864083


def _load_baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _free_energy(energies_kcal_mol: list[float], temperature_k: float) -> float:
    minimum = min(energies_kcal_mol)
    thermal = GAS_CONSTANT_KCAL_MOL_K * temperature_k
    return minimum - thermal * math.log(
        sum(math.exp(-(energy - minimum) / thermal) for energy in energies_kcal_mol)
    )


def _summary(errors: list[float]) -> dict[str, float]:
    return {
        "mae_kcal_mol": sum(errors) / len(errors),
        "rmse_kcal_mol": math.sqrt(
            sum(error * error for error in errors) / len(errors)
        ),
        "maximum_absolute_error_kcal_mol": max(errors),
    }


def _assert_sha256(value: str) -> None:
    assert len(value) == 64
    int(value, 16)


def _assert_git_sha(value: str) -> None:
    assert len(value) == 40
    int(value, 16)


def test_qm_fidelity_baseline_recomputes_fixed_conformer_metrics():
    baseline = _load_baseline()
    panel = baseline["fixed_conformer_panel"]
    records = panel["records"]

    assert baseline["schema_version"] == 2
    assert baseline["status"] == "frozen-bounded-pilot"
    assert baseline["route2_profiles"] == [
        "smd-ddpcm-l15-n1202-gaff2-o-mace-kspace40-v1",
        "smd-ddpcm-l15-n1202-v1",
    ]
    current_alignment = baseline["current_checkout_alignment"]
    latest_checked_head = current_alignment["latest_checked_git_head"]
    runtime_reference_head = current_alignment["runtime_code_reference_git_head"]
    _assert_git_sha(latest_checked_head)
    _assert_git_sha(runtime_reference_head)
    assert current_alignment["runtime_source_scope"] == "maple/"
    assert current_alignment["runtime_code_unchanged_between_canary_heads"] is True
    canary_execution_heads = current_alignment["canary_execution_heads"]
    assert canary_execution_heads == {
        "2-acetoxyethyl acetate": latest_checked_head,
        "acetone": "f3e989245189f742a03a61a19a848df54035c96f",
        "methanol": runtime_reference_head,
    }
    for execution_head in canary_execution_heads.values():
        _assert_git_sha(execution_head)
    assert current_alignment["fixed_records_confirmed_by_current_head_canary"] == [
        "methanol",
        "acetone",
        "2-acetoxyethyl acetate",
    ]
    assert current_alignment["fixed_records_not_rerun"] == []
    assert current_alignment["flexible_panel_current_checkout_match"] is None
    assert len(records) == panel["summary"]["record_count"] == 3
    assert len({record["compound_id"] for record in records}) == len(records)

    route2_qm_errors = []
    route2_experiment_errors = []
    for record in records:
        assert record["same_geometry"] is True
        assert record["training_membership_confirmed_excluded"] is False
        assert record["observed_timing"]["hardware_normalized"] is False
        assert record["observed_timing"]["speedup_certified"] is False
        evidence = record["source_evidence"]
        _assert_sha256(evidence["sha256"])
        _assert_git_sha(evidence["evidence_git_head"])
        assert evidence["status"] == "comparison-valid"

        route2 = record["route2"]
        qm = record["qm"]
        experiment = record["experiment"]
        differences = record["differences_kcal_mol"]
        route2_components = route2["components_kcal_mol"]
        qm_components = qm["components_kcal_mol"]

        assert sum(route2_components.values()) == pytest.approx(
            route2["delta_g_solv_kcal_mol"], abs=1.0e-9
        )
        assert sum(qm_components.values()) == pytest.approx(
            qm["delta_g_solv_kcal_mol"], abs=1.0e-9
        )
        assert differences["route2_minus_qm"] == pytest.approx(
            route2["delta_g_solv_kcal_mol"] - qm["delta_g_solv_kcal_mol"],
            abs=1.0e-12,
        )
        assert differences["route2_minus_experiment"] == pytest.approx(
            route2["delta_g_solv_kcal_mol"] - experiment["delta_g_solv_kcal_mol"],
            abs=1.0e-12,
        )
        assert differences["qm_minus_experiment"] == pytest.approx(
            qm["delta_g_solv_kcal_mol"] - experiment["delta_g_solv_kcal_mol"],
            abs=1.0e-12,
        )
        assert record["observed_timing"][
            "qm_over_route2_evaluation_ratio"
        ] == pytest.approx(
            qm["gas_plus_solution_seconds"] / route2["evaluation_seconds"],
            abs=1.0e-12,
        )

        component_differences = differences["route2_minus_qm_components"]
        for name in ("delta_e_solute", "u_pol", "g_cds"):
            assert component_differences[name] == pytest.approx(
                route2_components[name] - qm_components[name], abs=1.0e-12
            )
        assert component_differences["delta_g_solv"] == pytest.approx(
            differences["route2_minus_qm"], abs=1.0e-12
        )
        component_sum = sum(
            component_differences[name] for name in ("delta_e_solute", "u_pol", "g_cds")
        )
        assert component_sum == pytest.approx(
            differences["route2_minus_qm"], abs=1.0e-10
        )
        assert component_differences["g_cds"] == pytest.approx(0.0)
        assert (
            component_differences["delta_e_solute"] * component_differences["u_pol"]
            < 0.0
        )

        if record["name"] in canary_execution_heads:
            assert evidence["current_checkout_match"] is True
            assert evidence["alignment_status"] == (
                "confirmed-by-current-head-energy-canary"
            )
            canary = evidence["current_head_canary"]
            assert canary["status"] == "pass"
            assert canary["git_head"] == canary_execution_heads[record["name"]]
            assert evidence["current_checkout_head"] == canary["git_head"]
            assert canary["root_scf_iterations"] == route2["root_scf_iterations"]
            assert canary["evaluation_seconds"] > 0.0
            assert canary["used_for_timing_claim"] is False
            for hash_name in (
                "sha256",
                "force_reference_sha256",
                "mol2_sha256",
                "runner_sha256",
            ):
                _assert_sha256(canary[hash_name])
            assert canary[
                "absolute_difference_from_historical_correction_ev"
            ] == pytest.approx(
                abs(
                    canary["correction_energy_ev"]
                    - canary["historical_correction_energy_ev"]
                ),
                abs=1.0e-18,
            )
        else:
            assert evidence["current_checkout_match"] is None
            assert evidence["alignment_status"] == "not-rerun-on-current-checkout"
            assert evidence["current_checkout_head"] == latest_checked_head

        route2_qm_errors.append(abs(differences["route2_minus_qm"]))
        route2_experiment_errors.append(abs(differences["route2_minus_experiment"]))

    assert panel["summary"]["route2_vs_qm"] == pytest.approx(
        _summary(route2_qm_errors), abs=1.0e-12
    )
    assert panel["summary"]["route2_vs_experiment"] == pytest.approx(
        _summary(route2_experiment_errors), abs=1.0e-12
    )


def test_qm_fidelity_baseline_recomputes_bounded_electronic_ensemble():
    baseline = _load_baseline()
    panel = baseline["flexible_conformer_panel"]
    records = panel["records"]
    summary = panel["summary"]
    temperature = panel["temperature_k"]

    assert panel["geometry_count"] == len(records) == 4
    assert panel["geometry_generation"].startswith(
        "One original source geometry plus three ETKDGv3/MMFF"
    )
    assert panel["predeclared_primary_gate"] == {
        "metric": "mean_absolute_route2_minus_qm_kcal_mol",
        "status": "pass",
        "threshold_kcal_mol": 1.0,
    }
    evidence = panel["source_evidence"]
    _assert_sha256(evidence["sha256"])
    assert evidence["status"] == "complete-valid"
    assert (
        evidence["current_checkout_head"]
        == baseline["current_checkout_alignment"]["latest_checked_git_head"]
    )
    assert evidence["current_checkout_match"] is None
    assert evidence["alignment_status"] == "not-rerun-on-current-checkout"
    for evidence_head in evidence["evidence_git_heads"]:
        _assert_git_sha(evidence_head)

    signed_errors = []
    for record in records:
        route2 = record["route2"]
        qm = record["qm"]
        _assert_sha256(record["mol2_sha256"])
        assert sum(route2["components_kcal_mol"].values()) == pytest.approx(
            route2["delta_g_solv_kcal_mol"], abs=1.0e-9
        )
        assert sum(qm["components_kcal_mol"].values()) == pytest.approx(
            qm["delta_g_solv_kcal_mol"], abs=1.0e-9
        )
        assert (
            route2["aqueous_energy_kcal_mol"] - route2["gas_energy_kcal_mol"]
        ) == pytest.approx(route2["delta_g_solv_kcal_mol"], abs=1.0e-9)
        assert (
            qm["aqueous_energy_kcal_mol"] - qm["gas_energy_kcal_mol"]
        ) == pytest.approx(qm["delta_g_solv_kcal_mol"], abs=1.0e-9)
        signed_error = route2["delta_g_solv_kcal_mol"] - qm["delta_g_solv_kcal_mol"]
        assert record["route2_minus_qm_kcal_mol"] == pytest.approx(
            signed_error, abs=1.0e-12
        )
        assert record["absolute_route2_minus_qm_kcal_mol"] == pytest.approx(
            abs(signed_error), abs=1.0e-12
        )
        component_differences = record["route2_minus_qm_components_kcal_mol"]
        for name in ("delta_e_solute", "u_pol", "g_cds"):
            assert component_differences[name] == pytest.approx(
                route2["components_kcal_mol"][name] - qm["components_kcal_mol"][name],
                abs=1.0e-12,
            )
        assert sum(component_differences.values()) == pytest.approx(
            signed_error, abs=1.0e-10
        )
        assert component_differences["g_cds"] == pytest.approx(0.0)
        assert (
            component_differences["delta_e_solute"] * component_differences["u_pol"] < 0
        )
        signed_errors.append(signed_error)

    route2_ensemble = _free_energy(
        [record["route2"]["aqueous_energy_kcal_mol"] for record in records],
        temperature,
    ) - _free_energy(
        [record["route2"]["gas_energy_kcal_mol"] for record in records],
        temperature,
    )
    qm_ensemble = _free_energy(
        [record["qm"]["aqueous_energy_kcal_mol"] for record in records],
        temperature,
    ) - _free_energy(
        [record["qm"]["gas_energy_kcal_mol"] for record in records],
        temperature,
    )

    assert summary["mean_absolute_route2_minus_qm_kcal_mol"] == pytest.approx(
        sum(abs(error) for error in signed_errors) / len(signed_errors),
        abs=1.0e-12,
    )
    assert summary["maximum_absolute_route2_minus_qm_kcal_mol"] == pytest.approx(
        max(abs(error) for error in signed_errors)
    )
    assert summary["mean_signed_route2_minus_qm_kcal_mol"] == pytest.approx(
        sum(signed_errors) / len(signed_errors), abs=1.0e-12
    )
    assert summary["route2_minus_qm_bias_range_kcal_mol"] == pytest.approx(
        max(signed_errors) - min(signed_errors), abs=1.0e-12
    )
    assert summary["route2_bounded_electronic_ensemble_kcal_mol"] == pytest.approx(
        route2_ensemble, abs=1.0e-9
    )
    assert summary["qm_bounded_electronic_ensemble_kcal_mol"] == pytest.approx(
        qm_ensemble, abs=1.0e-9
    )
    assert summary["absolute_ensemble_difference_kcal_mol"] == pytest.approx(
        abs(route2_ensemble - qm_ensemble), abs=1.0e-9
    )
    assert summary[
        "route2_ensemble_absolute_experimental_error_kcal_mol"
    ] == pytest.approx(
        abs(route2_ensemble - summary["experimental_kcal_mol"]), abs=1.0e-9
    )

    gates = panel["certification_gates"]
    primary_gate = panel["predeclared_primary_gate"]
    primary_gate_passed = (
        summary[primary_gate["metric"]] <= primary_gate["threshold_kcal_mol"]
    )
    assert (primary_gate["status"] == "pass") is primary_gate_passed
    assert gates["all_four_locked_geometries_complete"] is True
    assert gates["predeclared_four_geometry_mae_pass"] is primary_gate_passed
    assert gates["complete_conformer_thermochemistry"] is False
    assert gates["solution_phase_geometry_relaxation"] is False
    assert gates["training_membership_confirmed_excluded"] is False


def test_qm_fidelity_baseline_preserves_reference_and_claim_boundaries():
    baseline = _load_baseline()
    diagnostic = baseline["acetone_same_cavity_ao_reference_diagnostic"]

    assert "not CI inputs" in baseline["source_artifact_scope"]
    assert baseline["qm_reference"]["comparison_level"] == (
        "chemical-level same-geometry reference, not operator-identical"
    )
    assert diagnostic["status"] == "acetone-specific-invalid-reference"
    assert diagnostic["production_change_required"] is False
    assert "Do not enlarge Route-2 radii" in diagnostic["forbidden_inference"]
    assert "does not prove" in diagnostic["scope_boundary"]
    invalid_case = diagnostic["numerically_completed_but_physically_invalid_case"]
    assert invalid_case["status"] == "completed-but-physically-invalid"
    assert invalid_case["dd_solvation_energy_hartree"] == pytest.approx(
        -114.16103810006106
    )
    _assert_sha256(diagnostic["source_evidence"]["sha256"])
    assert diagnostic["source_evidence"]["status"] == "source-diagnostic-valid"
    assert (
        diagnostic["source_evidence"]["current_checkout_head"]
        == baseline["current_checkout_alignment"]["latest_checked_git_head"]
    )
    assert diagnostic["source_evidence"]["current_checkout_match"] is None
    assert "No vibrational thermochemistry" in baseline["claim_boundary"]
    assert "hardware-normalized speedup" in baseline["claim_boundary"]
