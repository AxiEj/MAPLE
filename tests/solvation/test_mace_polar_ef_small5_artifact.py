"""Internal integrity of historical diagnostics, not runtime admission."""

import hashlib
import json
import math
from pathlib import Path

import pytest


ARCHIVE = Path(__file__).resolve().parents[2] / (
    "docs/implicit-solvation/benchmarks/"
    "mace-polar-ef-v2-freesolv-small5-20260905"
)


def _load(name):
    return json.loads((ARCHIVE / name).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_original_evidence_bytes_and_preregistration_are_preserved():
    for relative, expected in _load("archive-sha256.json").items():
        assert _sha(ARCHIVE / relative) == expected
    lock = _load("preregistered-panel.json")
    assert _sha(ARCHIVE / "parent-panel.json") == lock["parent_manifest_sha256"]
    assert _sha(ARCHIVE / "preregistered-panel.json") == _load("results.json")[
        "lock_sha256"
    ]


def test_selection_and_failed_records_are_not_replaced():
    lock = _load("preregistered-panel.json")
    parent = _load("parent-panel.json")
    selected = sorted(
        parent["locked_records"], key=lambda r: (r["natoms"], r["compound_id"])
    )[:5]
    assert lock["records"] == selected
    report = _load("results.json")
    assert [r["compound_id"] for r in report["records"]] == [
        r["compound_id"] for r in selected
    ]
    assert report["n_selected"] == report["n_attempted"] == 5
    assert report["n_success"] == 3
    assert report["n_failed"] == 2
    assert report["complete"] is True
    assert report["scientifically_valid"] is False
    assert report["metrics_scope"] == "successful_subset_only"
    for row in report["records"]:
        if row["status"] != "ok":
            assert row["returncode"] == 1
            assert "predicted_kcal_mol" not in row
            console = ARCHIVE / "raw" / row["compound_id"] / "console.log"
            assert "Error during calculation:" in console.read_text()


def test_successful_subset_metrics_recompute_from_raw_components():
    lock = _load("preregistered-panel.json")
    report = _load("results.json")
    references = {r["compound_id"]: r for r in lock["records"]}
    errors = []
    for row in report["records"]:
        if row["status"] != "ok":
            continue
        folder = f"raw/{row['compound_id']}"
        result_path = ARCHIVE / folder / "coupling-result.json"
        assert _sha(result_path) == row["raw_result_sha256"]
        assert _sha(ARCHIVE / folder / "job.out") == row["output_sha256"]
        raw = _load(f"{folder}/coupling-result.json")
        assert raw["scientifically_valid"] is False
        assert raw["profile"] == lock["profile"]
        parts = raw["components_hartree"]
        assert parts["delta_g_solv"] == pytest.approx(
            parts["electrostatic"] + parts["cds"], abs=1e-12
        )
        predicted = parts["delta_g_solv"] * lock["hartree_to_kcal_mol"]
        reference = references[row["compound_id"]]["experimental_kcal_mol"]
        assert row["experimental_kcal_mol"] == reference
        assert row["predicted_kcal_mol"] == pytest.approx(predicted, abs=1e-12)
        error = predicted - reference
        assert row["error_kcal_mol"] == pytest.approx(error, abs=1e-12)
        assert row["absolute_error_kcal_mol"] == pytest.approx(abs(error))
        errors.append(error)
    metrics = report["metrics"]
    assert metrics["n"] == len(errors) == 3
    assert metrics["mae_kcal_mol"] == pytest.approx(sum(map(abs, errors)) / 3)
    assert metrics["rmse_kcal_mol"] == pytest.approx(
        math.sqrt(sum(e * e for e in errors) / 3)
    )
    assert metrics["mean_signed_error_kcal_mol"] == pytest.approx(sum(errors) / 3)
    assert metrics["maximum_absolute_error_kcal_mol"] == pytest.approx(
        max(map(abs, errors))
    )
