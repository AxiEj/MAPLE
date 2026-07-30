from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"

if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from route2_v0_historical_freesolv10 import (
    HISTORICAL_FREE_SOLV10_IDS,
    HISTORICAL_FUNCTIONAL_GROUP_OR_SCAFFOLD_CLASSES,
    HISTORICAL_LOCKED_RECORDS_SHA256,
    HISTORICAL_SOURCE_BLOB_OID,
    HISTORICAL_SOURCE_COMMIT,
    HISTORICAL_SOURCE_CONTENT_SHA256,
    HISTORICAL_WORST_ABSOLUTE_ERROR_KCAL_MOL,
    HISTORICAL_WORST_RECORD_ID,
    MINIMUM_DISTINCT_FUNCTIONAL_GROUP_OR_SCAFFOLD_CLASSES,
    STRICT_ABSOLUTE_ERROR_THRESHOLD_KCAL_MOL,
    evaluate_historical_freesolv10_predictions,
    load_historical_freesolv10_manifest,
    main,
)

MANIFEST = BENCHMARKS / "route2-v0-historical-freesolv10-regression-v1.json"


def _predictions(
    error_by_id: dict[str, float] | None = None,
) -> list[dict[str, float | str]]:
    manifest = load_historical_freesolv10_manifest()
    error_by_id = {} if error_by_id is None else error_by_id
    return [
        {
            "compound_id": record["compound_id"],
            "predicted_kcal_mol": record["experimental_kcal_mol"]
            + error_by_id.get(record["compound_id"], 0.0),
        }
        for record in manifest["locked_records"]
    ]


def test_historical_freesolv10_manifest_preserves_the_actual_seven_kcal_outlier_panel():
    manifest = load_historical_freesolv10_manifest()

    assert manifest["panel_id"] == "route2-v0-historical-freesolv10-regression-v1"
    assert manifest["historical_source"]["git_commit"] == HISTORICAL_SOURCE_COMMIT
    assert manifest["historical_source"]["git_blob_oid"] == HISTORICAL_SOURCE_BLOB_OID
    assert (
        manifest["historical_source"]["content_sha256"]
        == HISTORICAL_SOURCE_CONTENT_SHA256
    )
    assert (
        manifest["locked_records_canonical_sha256"] == HISTORICAL_LOCKED_RECORDS_SHA256
    )
    assert tuple(record["compound_id"] for record in manifest["locked_records"]) == (
        HISTORICAL_FREE_SOLV10_IDS
    )
    assert tuple(record["chemical_class"] for record in manifest["locked_records"]) == (
        HISTORICAL_FUNCTIONAL_GROUP_OR_SCAFFOLD_CLASSES
    )
    assert len(HISTORICAL_FUNCTIONAL_GROUP_OR_SCAFFOLD_CLASSES) == (
        MINIMUM_DISTINCT_FUNCTIONAL_GROUP_OR_SCAFFOLD_CLASSES
    )
    assert len(set(HISTORICAL_FUNCTIONAL_GROUP_OR_SCAFFOLD_CLASSES)) == (
        MINIMUM_DISTINCT_FUNCTIONAL_GROUP_OR_SCAFFOLD_CLASSES
    )
    assert manifest["acceptance_gate"] == {
        "expected_record_count": 10,
        "functional_group_or_scaffold_field": "chemical_class",
        "minimum_distinct_functional_group_or_scaffold_classes": 10,
        "functional_group_or_scaffold_diversity_is_mandatory": True,
        "all_locked_records_required": True,
        "extra_records_fail": True,
        "maximum_absolute_error_threshold_kcal_mol": 1.5,
        "comparison": "strictly_less_than",
        "all_records_must_individually_satisfy_threshold": True,
        "mae_or_rmse_alone_is_never_sufficient": True,
        "missing_record_is_failure": True,
        "historical_maximum_error_record_must_remain_included": "mobley_6973347",
        "validator_module": "route2_v0_historical_freesolv10.py",
    }
    outlier = manifest["historical_observed_maximum_error"]
    assert outlier["compound_id"] == HISTORICAL_WORST_RECORD_ID
    assert outlier["name"] == "ethyl acetate"
    assert outlier["absolute_error_kcal_mol"] == pytest.approx(
        HISTORICAL_WORST_ABSOLUTE_ERROR_KCAL_MOL
    )


def test_historical_freesolv10_requires_every_original_record_below_one_point_five():
    passed = evaluate_historical_freesolv10_predictions(
        _predictions({HISTORICAL_WORST_RECORD_ID: 1.499999999})
    )
    assert passed.status == "pass"
    assert passed.strict_all_records_under_threshold is True
    assert passed.maximum_absolute_error_kcal_mol == pytest.approx(1.499999999)
    assert passed.worst_compound_id == HISTORICAL_WORST_RECORD_ID
    assert len(passed.records) == 10
    assert passed.distinct_functional_group_or_scaffold_class_count == 10

    failed = evaluate_historical_freesolv10_predictions(
        _predictions(
            {HISTORICAL_WORST_RECORD_ID: STRICT_ABSOLUTE_ERROR_THRESHOLD_KCAL_MOL}
        )
    )
    assert failed.status == "fail"
    assert failed.strict_all_records_under_threshold is False
    assert failed.maximum_absolute_error_kcal_mol == pytest.approx(1.5)


def test_historical_freesolv10_rejects_missing_extra_duplicate_and_nonfinite_records():
    good = _predictions()
    with pytest.raises(ValueError, match="missing=.*mobley_6973347"):
        evaluate_historical_freesolv10_predictions(
            [
                record
                for record in good
                if record["compound_id"] != HISTORICAL_WORST_RECORD_ID
            ]
        )
    with pytest.raises(ValueError, match="extra=.*not-in-panel"):
        evaluate_historical_freesolv10_predictions(
            [*good, {"compound_id": "not-in-panel", "predicted_kcal_mol": 0.0}]
        )
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_historical_freesolv10_predictions([*good, good[0]])

    nonfinite = [dict(record) for record in good]
    nonfinite[0]["predicted_kcal_mol"] = float("nan")
    with pytest.raises(ValueError, match="finite real"):
        evaluate_historical_freesolv10_predictions(nonfinite)


def test_historical_freesolv10_manifest_loader_rejects_a_shrunken_or_weakened_panel(
    tmp_path,
):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["locked_records"] = manifest["locked_records"][:-1]
    path = tmp_path / "shrunken.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="locked record identity changed"):
        load_historical_freesolv10_manifest(path)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["acceptance_gate"]["comparison"] = "less_than_or_equal"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="strict maximum-error gate"):
        load_historical_freesolv10_manifest(path)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["acceptance_gate"][
        "minimum_distinct_functional_group_or_scaffold_classes"
    ] = 9
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="functional-group/scaffold gate"):
        load_historical_freesolv10_manifest(path)


def test_historical_freesolv10_cli_writes_the_full_failure_report(tmp_path):
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            {
                "predictions": _predictions(
                    {
                        HISTORICAL_WORST_RECORD_ID: (
                            STRICT_ABSOLUTE_ERROR_THRESHOLD_KCAL_MOL
                        )
                    }
                )
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evaluation.json"

    assert main(["--predictions", str(predictions), "--output", str(output)]) == 1

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    assert report["acceptance"]["strict_all_records_under_threshold"] is False
    assert report["acceptance"]["maximum_absolute_error_kcal_mol"] == pytest.approx(
        STRICT_ABSOLUTE_ERROR_THRESHOLD_KCAL_MOL
    )
    assert len(report["records"]) == 10
    assert (
        report["acceptance"]["distinct_functional_group_or_scaffold_class_count"] == 10
    )
