from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
HISTORICAL12 = (
    BENCHMARKS / "route2-legacy-exact-gto-historical-freesolv12-diagnostic-v1.json"
)
CONFIRMATION123 = (
    BENCHMARKS / "route2-legacy-exact-gto-confirmation123-diagnostic-v1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_legacy_only(artifact: dict) -> None:
    assert artifact["schema_version"] == 1
    assert artifact["status"] == "diagnostic-only-not-route2v-acceptance"
    assert "not a Route-2 V0/2V result" in artifact["claim_boundary"]

    method = artifact["current_method"]
    assert method["profile"] == (
        "macepolar-mlpcm-smdcds-iefpcm-intrinsic-cavity-exact-gto-v1"
    )
    assert method["scientific_status"] == "legacy-nonvariational-fixedpoint-diagnostic"
    assert method["public_capability"] == "experimental-energy-only"
    assert method["source_receiver_contract"] == (
        "known-nonconjugate-point-source-gto-receiver"
    )
    assert "Route-2 V0 accuracy" in method["prohibited_claims"]
    assert "acceptance-panel pass" in method["prohibited_claims"]

    no_target = artifact["no_target_policy"]
    assert all(
        no_target[key] is False
        for key in (
            "post_training",
            "fine_tuning",
            "experimental_solvation_fit",
            "map_or_uq_calibration",
            "radius_adjustment",
            "half_coupling_adjustment",
            "post_selection_parameter_change",
            "experimental_labels_used_to_rank_current_method",
            "experimental_labels_used_to_select_panel",
        )
    )


def test_frozen_historical_freesolv12_exposes_the_acid_failure():
    artifact = _load(HISTORICAL12)
    _assert_legacy_only(artifact)
    assert artifact["artifact"] == "route2-legacy-exact-gto-historical-freesolv12-diagnostic-v1"
    assert artifact["panel"]["kind"] == (
        "frozen-historical-freesolv12-functional-group-regression"
    )
    assert artifact["panel"]["expected_record_count"] == 12

    summary = artifact["summary"]
    assert summary["success_count"] == 12
    assert summary["failure_count"] == 0
    assert summary["mae_kcal_mol"] == pytest.approx(0.7895271292049749, abs=1e-12)
    assert summary["rmse_kcal_mol"] == pytest.approx(1.1256273000770607, abs=1e-12)
    assert summary["maximum_absolute_error_kcal_mol"] == pytest.approx(
        3.2473279957153336, abs=1e-12
    )
    assert summary["records_strictly_below_1_5_kcal_mol"] == 11
    assert summary["records_at_or_above_1_5_kcal_mol"] == 1
    assert summary["all_records_strictly_below_1_5_kcal_mol"] is False

    acid = next(record for record in artifact["records"] if record["name"] == "acetic acid")
    result = acid["legacy_exact_gto_diagnostic"]
    assert result["absolute_error_kcal_mol"] == pytest.approx(
        3.2473279957153336, abs=1e-12
    )
    assert result["density_residual_inf_e"] < 1.0e-5


def test_all_deterministic_confirmation_records_are_preserved_including_failures():
    artifact = _load(CONFIRMATION123)
    _assert_legacy_only(artifact)
    assert artifact["artifact"] == "route2-legacy-exact-gto-confirmation123-diagnostic-v1"
    assert artifact["panel"]["kind"] == (
        "all-pre-existing-deterministic-confirmation-partition-members"
    )
    assert artifact["panel"]["expected_record_count"] == 123
    assert artifact["panel"]["source_prepared_manifest"]["candidate_count"] == 642
    assert artifact["panel"]["source_prepared_manifest"]["partition_counts"] == {
        "confirmation": 123,
        "development": 519,
    }

    summary = artifact["summary"]
    assert summary["success_count"] == 77
    assert summary["failure_count"] == 46
    assert summary["failure_rate"] == pytest.approx(46 / 123, abs=1e-15)
    assert summary["mae_kcal_mol"] == pytest.approx(1.2431537584357881, abs=1e-12)
    assert summary["rmse_kcal_mol"] == pytest.approx(1.5700939863262517, abs=1e-12)
    assert summary["maximum_absolute_error_kcal_mol"] == pytest.approx(
        3.9737772496012496, abs=1e-12
    )
    assert summary["worst_name"] == "2-iodophenol"
    assert summary["records_at_or_above_1_5_kcal_mol"] == 24
    assert summary["all_records_strictly_below_1_5_kcal_mol"] is False

    records = artifact["records"]
    assert len(records) == 123
    assert len({record["compound_id"] for record in records}) == 123
    failures = [
        record
        for record in records
        if record["legacy_exact_gto_diagnostic"].get("status")
        == "provider-rejected-no-published-energy"
    ]
    assert len(failures) == 46
    assert all(
        "PCMSolver emitted a warning for every cavity attempt"
        in record["legacy_exact_gto_diagnostic"]["reason"]
        for record in failures
    )
    assert artifact["provider_failure_summary"]["count"] == len(failures)

    worst = artifact["worst_successful_records"][0]
    assert worst == {
        "absolute_error_kcal_mol": pytest.approx(3.9737772496012496, abs=1e-12),
        "compound_id": "mobley_8789465",
        "functional_groups": [
            "aromatic",
            "aryl iodide",
            "phenol or hydroxyhetarene",
        ],
        "name": "2-iodophenol",
    }
