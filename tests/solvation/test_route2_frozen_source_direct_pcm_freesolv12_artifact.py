from __future__ import annotations

import json
from pathlib import Path

import pytest
from artifact_source_binding import assert_source_files_match_execution_commit

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-frozen-source-direct-pcm-freesolv12-ddpcm-ddcosmo-v1-"
    "execution-ae427ea7.json"
)


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value))
    return set()


def test_frozen_source_freesolv12_keeps_the_strict_all_record_failure():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact"] == (
        "route2-frozen-source-direct-pcm-freesolv12-ddpcm-ddcosmo-v1-"
        "execution-ae427ea7"
    )
    assert artifact["execution_git_head"] == (
        "ae427ea7537c5d50673d99a566a250642b45707b"
    )
    assert artifact["status"] == "complete-strict-freesolv12-fail"
    assert artifact["disposition"]["strict_gate_passed"] is False

    method = artifact["frozen_source_method"]
    identity = artifact["scientific_identity"]
    assert method["response_mode"] == "frozen"
    assert method["fixed_point_applicable"] is False
    assert method["field_conditioned_model_evaluated"] is False
    assert method["receiver"] is None
    assert method["energy_ledger"] == "pcm-half-coupling-only-v1"
    assert identity["mutual_ml_continuum_polarization"] is False
    assert identity["forces_available"] is False
    assert {
        "scf_iterations",
        "unmixed_density_residual_inf_e",
        "experimental_kcal_mol",
        "predicted_kcal_mol",
    }.isdisjoint(_all_keys(artifact))

    panel = artifact["frozen_panel"]
    records = artifact["records"]
    actual_groups = {
        row["functional_group"]
        for row in records
        if not row["is_nonfunctional_control"]
    }
    assert panel["record_count"] == len(records) == 12
    assert panel["distinct_actual_functional_groups"] == len(actual_groups) == 10
    assert sum(row["is_nonfunctional_control"] for row in records) == 2
    assert panel["historical_ethyl_acetate_regression_included"] in {
        row["compound_id"] for row in records
    }

    expected = {
        "ddpcm": {
            "mae": 1.0842909974701032,
            "rmse": 1.2546769632514712,
            "maximum": 1.9910283638282094,
            "failed": 4,
            "worst": "mobley_8578590",
        },
        "ddcosmo": {
            "mae": 1.0722607290154251,
            "rmse": 1.236159890195475,
            "maximum": 1.9549099146690123,
            "failed": 3,
            "worst": "mobley_3034976",
        },
    }
    for equation, reference in expected.items():
        metrics = artifact["results_kcal_mol"][equation]
        assert metrics["record_count"] == 12
        assert metrics["all_records_strictly_below_1_5_kcal_mol"] is False
        assert metrics["mae_kcal_mol"] == pytest.approx(reference["mae"], abs=1.0e-12)
        assert metrics["rmse_kcal_mol"] == pytest.approx(reference["rmse"], abs=1.0e-12)
        assert metrics["maximum_absolute_error_kcal_mol"] == pytest.approx(
            reference["maximum"], abs=1.0e-12
        )
        assert metrics["records_at_or_above_1_5_kcal_mol"] == (reference["failed"])
        assert metrics["worst_compound_id"] == reference["worst"]

        error_key = f"{equation}_absolute_error_kcal_mol"
        pass_key = f"{equation}_passes_strict_threshold"
        assert sum(not row[pass_key] for row in records) == reference["failed"]
        assert max(row[error_key] for row in records) == pytest.approx(
            reference["maximum"], abs=1.0e-12
        )

    historical = artifact["historical_scf_source_direct_ledger_comparison"]
    assert historical["ddpcm_maximum_absolute_error_kcal_mol"] == pytest.approx(
        7.330262771684603
    )
    assert historical["ddcosmo_maximum_absolute_error_kcal_mol"] == pytest.approx(
        7.447874329470056
    )
    assert "response=frozen" in historical["clarification"]

    assert_source_files_match_execution_commit(ROOT, artifact)
