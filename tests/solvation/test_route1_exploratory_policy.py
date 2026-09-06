import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs" / "implicit-solvation" / "benchmarks"
POLICY_PATH = BENCHMARKS / "route1_exploratory_policy_v1.json"
STATUS_PATH = ROOT / "docs" / "implicit-solvation" / "EXPLORATORY_STATUS_2026-09-05.md"
VALIDATION_STATUS_PATH = ROOT / "docs" / "implicit-solvation" / "VALIDATION_STATUS.md"
HISTORICAL_CONTRACT_PATH = BENCHMARKS / "route1_multisolvent_accuracy_contract_v1.json"


def _policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def test_experimental_accuracy_threshold_is_explicitly_null_and_descriptive() -> None:
    policy = _policy()
    accuracy = policy["experimental_accuracy"]

    assert accuracy["experimental_accuracy_threshold_kcal_mol"] is None
    assert accuracy["accuracy_based_method_promotion_allowed"] is False
    assert accuracy["accuracy_based_method_rejection_required"] is False
    assert set(accuracy["required_metrics"]) >= {
        "bias_kcal_mol",
        "mae_kcal_mol",
        "rmse_kcal_mol",
        "maximum_absolute_error_kcal_mol",
        "successful_record_count",
        "failed_record_count",
    }


def test_relaxed_accuracy_does_not_relax_correctness_gates() -> None:
    gates = _policy()["correctness_gates"]

    assert gates["numerical_validity_retained"] is True
    assert gates["force_energy_consistency_retained"] is True
    assert gates["hessian_validity_retained"] is True
    assert gates["provenance_and_seal_validation_retained"] is True
    checks = " ".join(gates["required_checks"])
    assert "all-3N finite-difference" in checks
    assert "one imaginary vibrational mode" in checks
    assert "manifest seal verification" in checks


def test_policy_does_not_falsely_promote_force_or_requested_tasks() -> None:
    policy = _policy()
    tasks = policy["requested_task_scope"]
    probe = policy["current_force_probe"]

    assert tasks["goals"] == ["SP", "OPT", "FREQ", "TS"]
    assert tasks["chagb_pbsa_currently_delivered"] == ["SP"]
    assert tasks["chagb_pbsa_not_delivered"] == ["OPT", "FREQ", "TS"]
    assert tasks["runtime_force_enabled"] is False
    assert tasks["runtime_hessian_enabled"] is False
    assert tasks["runtime_ts_enabled"] is False
    assert probe["accuracy_threshold_used"] is False
    assert probe["force_promotion_allowed"] is False
    assert probe["adjacent_step_force_rms_kcal_mol_angstrom"] == [
        3.600651332203967,
        1.5543473137167199,
    ]


def test_multisolvent_counts_are_honest_and_not_energy_claims() -> None:
    pilot = _policy()["multisolvent_pilot"]
    records = pilot["per_solvent_counts"]

    assert len(records) == pilot["requested_solvents"] == 15
    assert sum(record["requested"] for record in records) == 300
    assert sum(record["selected"] for record in records) == 222
    assert sum(record["selected"] == 20 for record in records) == 8
    assert sum(record["shortfall"] > 0 for record in records) == 7
    assert all(record["selected"] == min(20, record["eligible"]) for record in records)
    assert all(record["shortfall"] == 20 - record["selected"] for record in records)
    assert pilot["energy_calculations_completed"] is False
    assert pilot["accuracy_scoring_completed"] is False
    assert pilot["runtime_nonwater_solvents_enabled"] is False
    assert pilot["current_product_parameters"] == "water only"


def test_supersession_is_prospective_and_preserves_historical_contract() -> None:
    policy = _policy()
    historical = policy["historical_evidence"]
    status = STATUS_PATH.read_text(encoding="utf-8")
    validation_status = VALIDATION_STATUS_PATH.read_text(encoding="utf-8")

    assert HISTORICAL_CONTRACT_PATH.is_file()
    assert historical["supersession_kind"] == "prospective_policy_only"
    assert historical["historical_protocols_resealed"] is False
    assert historical["historical_results_relabelled"] is False
    assert "route1_multisolvent_accuracy_contract_v1.json" in " ".join(
        historical["preserved_references"]
    )
    assert "not edited, resealed, or retrospectively relabelled" in status
    assert "Current exploratory-policy notice (2026-09-05)" in validation_status


def test_status_page_separates_completed_policy_from_unfinished_science() -> None:
    status = STATUS_PATH.read_text(encoding="utf-8")
    normalized_status = " ".join(status.split())

    assert (
        "No new solvent energies or accuracy scores have been produced."
        in normalized_status
    )
    assert "**not delivered**" in status
    assert "The current product parameters are water-only." in status
    assert "222 selectable pairs out of 300 requested" in normalized_status
