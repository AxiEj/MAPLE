from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"

if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from route2_v0_freesolv12_functional_groups import (
    FUNCTIONAL_GROUP_PANEL_IDS,
    FUNCTIONAL_GROUP_PANEL_LABELS,
    FUNCTIONAL_GROUP_PANEL_NONFUNCTIONAL_CONTROL_IDS,
    HISTORICAL_FREE_SOLV10_IDS,
    MINIMUM_DISTINCT_FUNCTIONAL_GROUPS,
    STRICT_ABSOLUTE_ERROR_THRESHOLD_KCAL_MOL,
    evaluate_freesolv12_functional_group_predictions,
    load_freesolv12_functional_group_manifest,
    main,
)


def _predictions(
    error_by_id: dict[str, float] | None = None,
) -> list[dict[str, float | str]]:
    manifest = load_freesolv12_functional_group_manifest()
    error_by_id = {} if error_by_id is None else error_by_id
    return [
        {
            "compound_id": record["compound_id"],
            "predicted_kcal_mol": record["experimental_kcal_mol"]
            + error_by_id.get(record["compound_id"], 0.0),
        }
        for record in manifest["locked_records"]
    ]


def test_functional_group_panel_requires_ten_actual_groups_and_embeds_the_7kcal_history():
    manifest = load_freesolv12_functional_group_manifest()

    records = manifest["locked_records"]
    assert (
        tuple(record["compound_id"] for record in records) == FUNCTIONAL_GROUP_PANEL_IDS
    )
    assert tuple(record["functional_group"] for record in records) == (
        FUNCTIONAL_GROUP_PANEL_LABELS
    )
    assert tuple(record["compound_id"] for record in records[:10]) == (
        HISTORICAL_FREE_SOLV10_IDS
    )
    assert (
        tuple(
            record["compound_id"]
            for record in records
            if record["functional_group"] is None
        )
        == FUNCTIONAL_GROUP_PANEL_NONFUNCTIONAL_CONTROL_IDS
    )
    assert (
        len(
            {
                record["functional_group"]
                for record in records
                if record["functional_group"] is not None
            }
        )
        == MINIMUM_DISTINCT_FUNCTIONAL_GROUPS
    )

    gate = manifest["acceptance_gate"]
    assert gate["expected_record_count"] == 12
    assert gate["minimum_distinct_functional_groups"] == 10
    assert gate["nonfunctional_controls_do_not_count_toward_functional_group_minimum"]
    assert gate["historical_freesolv10_regression_is_embedded"]
    assert gate["maximum_absolute_error_threshold_kcal_mol"] == 1.5
    assert gate["comparison"] == "strictly_less_than"


def test_functional_group_panel_requires_every_record_to_be_strictly_below_one_point_five():
    passed = evaluate_freesolv12_functional_group_predictions(
        _predictions({"mobley_6973347": 1.499999999})
    )
    assert passed.status == "pass"
    assert passed.strict_all_records_under_threshold
    assert passed.record_count == 12
    assert passed.distinct_functional_group_count == 10
    assert passed.nonfunctional_control_count == 2
    assert passed.historical_freesolv10_regression_embedded
    assert passed.maximum_absolute_error_kcal_mol == pytest.approx(1.499999999)
    assert passed.worst_compound_id == "mobley_6973347"

    failed = evaluate_freesolv12_functional_group_predictions(
        _predictions({"mobley_6973347": STRICT_ABSOLUTE_ERROR_THRESHOLD_KCAL_MOL})
    )
    assert failed.status == "fail"
    assert not failed.strict_all_records_under_threshold
    assert failed.maximum_absolute_error_kcal_mol == pytest.approx(1.5)


def test_functional_group_panel_rejects_missing_extra_duplicate_and_nonfinite_records():
    good = _predictions()
    with pytest.raises(ValueError, match="missing=.*mobley_6973347"):
        evaluate_freesolv12_functional_group_predictions(
            [record for record in good if record["compound_id"] != "mobley_6973347"]
        )
    with pytest.raises(ValueError, match="extra=.*not-in-panel"):
        evaluate_freesolv12_functional_group_predictions(
            [*good, {"compound_id": "not-in-panel", "predicted_kcal_mol": 0.0}]
        )
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_freesolv12_functional_group_predictions([*good, good[0]])

    nonfinite = [dict(record) for record in good]
    nonfinite[0]["predicted_kcal_mol"] = float("nan")
    with pytest.raises(ValueError, match="finite real"):
        evaluate_freesolv12_functional_group_predictions(nonfinite)


def test_functional_group_panel_loader_rejects_a_weakened_or_narrowed_gate(tmp_path):
    manifest = load_freesolv12_functional_group_manifest()
    manifest["acceptance_gate"]["minimum_distinct_functional_groups"] = 9
    path = tmp_path / "weakened.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="acceptance gate changed"):
        load_freesolv12_functional_group_manifest(path)

    manifest = load_freesolv12_functional_group_manifest()
    manifest["locked_records"] = manifest["locked_records"][:-1]
    path = tmp_path / "narrowed.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="record identity changed"):
        load_freesolv12_functional_group_manifest(path)


def test_functional_group_panel_cli_writes_the_full_per_record_failure_report(tmp_path):
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            {
                "predictions": _predictions(
                    {"mobley_6973347": STRICT_ABSOLUTE_ERROR_THRESHOLD_KCAL_MOL}
                )
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evaluation.json"

    assert main(["--predictions", str(predictions), "--output", str(output)]) == 1

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    assert report["acceptance"]["record_count"] == 12
    assert report["acceptance"]["distinct_functional_group_count"] == 10
    assert report["acceptance"]["nonfunctional_control_count"] == 2
    assert report["acceptance"]["historical_freesolv10_regression_embedded"]
    assert len(report["records"]) == 12
