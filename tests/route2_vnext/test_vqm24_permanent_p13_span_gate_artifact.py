from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "vqm24-permanent-p13-span-gate-20260827"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_point_p13_span_gate_preserves_its_generalization_failure() -> None:
    result = json.loads((EVIDENCE / "result.json").read_text())
    source = result["source"]

    assert result["status"] == "fail"
    assert source["runner_sha256"] == _sha256(ROOT / source["runner_path"])
    assert source["preregistration_file_sha256"] == _sha256(
        ROOT / source["preregistration_path"]
    )
    failures = [record for record in result["records"] if record["status"] == "fail"]
    assert len(failures) == 8
    assert all(
        [name for name, passed in record["gates"].items() if not passed]
        == ["audit_to_fit_ratio"]
        for record in failures
    )
    assert all(record["gates"]["audit_relative_mep"] for record in result["records"])
    assert all(record["gates"]["audit_minus_fit"] for record in result["records"])
    assert result["aggregate"] == {
        "all_records_pass": False,
        "maximum_audit_minus_fit_relative": pytest.approx(
            0.006066452396396542, abs=0.0
        ),
        "maximum_audit_relative_mep_error": pytest.approx(
            0.015528422478868148, abs=0.0
        ),
        "maximum_audit_to_fit_ratio": pytest.approx(
            3.654632627895447, abs=0.0
        ),
        "record_count": 32,
    }


def test_point_p13_failure_does_not_emit_coefficients_or_admit_training() -> None:
    result = json.loads((EVIDENCE / "result.json").read_text())

    assert all(
        record["metrics"]["coefficient_label_emitted"] is False
        and record["metrics"]["model_fit_performed"] is False
        for record in result["records"]
    )
    assert result["claim_boundary"] == {
        "capability_admitted": False,
        "density_or_partition_coefficient_label_emitted": False,
        "experimental_solvation_target_read": False,
        "fit_or_training_performed": False,
        "model_accuracy_measured": False,
        "pcm_or_cavity_used": False,
        "point_p13_permanent_span_admitted_for_head_training": False,
        "vqm24_energy_target_used": False,
    }
