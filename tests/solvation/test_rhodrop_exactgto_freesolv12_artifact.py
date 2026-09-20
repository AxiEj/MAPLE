from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = REPO_ROOT / "docs/implicit-solvation/benchmarks"
PREREGISTRATIONS = REPO_ROOT / "docs/route2/preregistrations"
RESULT_PATH = BENCHMARKS / "route2-rhodrop-exactgto-freesolv12-total-v1.json"
SELECTION_PATH = BENCHMARKS / "route2-v0-freesolv12-functional-groups-v1.json"
COMPARATOR_PATH = (
    BENCHMARKS
    / "route2-legacy-exact-gto-historical-freesolv12-diagnostic-v1.json"
)
PREREGISTRATION_PATH = (
    PREREGISTRATIONS / "rhodrop-exactgto-freesolv12-total-v1.json"
)
AMENDMENT_PATH = (
    PREREGISTRATIONS / "rhodrop-exactgto-freesolv12-runtime-amendment-v1.json"
)


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preregistration_remains_bound_to_the_executed_sources() -> None:
    preregistration = _load(PREREGISTRATION_PATH)
    source_hashes = preregistration["source_sha256"]
    assert isinstance(source_hashes, dict)
    assert source_hashes
    for relative_path, expected_sha256 in source_hashes.items():
        assert _sha256(REPO_ROOT / relative_path) == expected_sha256

    amendment = _load(AMENDMENT_PATH)
    assert amendment["parent_preregistration_sha256"] == _sha256(
        PREREGISTRATION_PATH
    )
    assert amendment["amendment"]["scientific_method_changed"] is False
    assert amendment["amendment"]["source_files_changed"] is False


def test_complete_negative_result_recomputes_from_all_frozen_records() -> None:
    result = _load(RESULT_PATH)
    selection = _load(SELECTION_PATH)
    records = result["records"]
    locked_records = selection["locked_records"]
    assert len(records) == len(locked_records) == 12

    signed_errors = []
    for index, (record, locked) in enumerate(zip(records, locked_records, strict=True)):
        assert record["record_index"] == index
        assert record["compound_id"] == locked["compound_id"]
        assert record["experimental_kcal_mol"] == locked["experimental_kcal_mol"]
        assert record["signed_error_kcal_mol"] == pytest.approx(
            record["predicted_kcal_mol"] - record["experimental_kcal_mol"],
            abs=1.0e-12,
        )
        assert record["absolute_error_kcal_mol"] == pytest.approx(
            abs(record["signed_error_kcal_mol"]),
            abs=1.0e-12,
        )
        assert record["exact_cold_replay_residual_max"] <= 2.0e-12
        signed_errors.append(record["signed_error_kcal_mol"])

    errors = np.asarray(signed_errors, dtype=float)
    summary = result["summary"]
    assert summary["success_count"] == 12
    assert summary["failure_count"] == 0
    assert summary["mae_kcal_mol"] == pytest.approx(np.mean(np.abs(errors)))
    assert summary["rmse_kcal_mol"] == pytest.approx(
        np.sqrt(np.mean(np.square(errors)))
    )
    assert summary["mean_signed_error_kcal_mol"] == pytest.approx(np.mean(errors))
    assert summary["maximum_absolute_error_kcal_mol"] == pytest.approx(
        np.max(np.abs(errors))
    )
    assert summary["records_strictly_below_1_0_kcal_mol"] == int(
        np.sum(np.abs(errors) < 1.0)
    )
    assert result["status"] == "complete-negative-accuracy-result"
    assert result["decision"] == {
        "accuracy_retention_mae_below_1_passed": False,
        "meaningful_accuracy_improvement_passed": False,
        "method_may_be_retuned_on_this_panel": False,
        "numerical_completion_passed": True,
    }


def test_historical_exact_gto_comparison_is_recomputed_not_hand_entered() -> None:
    result = _load(RESULT_PATH)
    comparator = _load(COMPARATOR_PATH)
    old_records = {
        record["compound_id"]: record["legacy_exact_gto_diagnostic"]
        for record in comparator["records"]
    }
    new_absolute_errors = np.asarray(
        [record["absolute_error_kcal_mol"] for record in result["records"]]
    )
    old_absolute_errors = np.asarray(
        [
            old_records[record["compound_id"]]["absolute_error_kcal_mol"]
            for record in result["records"]
        ]
    )
    comparison = result["comparison"]
    assert comparison["new_minus_historical_mae_kcal_mol"] == pytest.approx(
        np.mean(new_absolute_errors) - np.mean(old_absolute_errors)
    )
    assert comparison["new_minus_historical_count_below_1_0"] == (
        int(np.sum(new_absolute_errors < 1.0))
        - int(np.sum(old_absolute_errors < 1.0))
    )
    assert result["panel_manifest_sha256"] == _sha256(SELECTION_PATH)
    assert result["historical_comparator_sha256"] == _sha256(COMPARATOR_PATH)
