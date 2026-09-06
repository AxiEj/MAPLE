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
    / "auxiliary-density-representation-gate-20260824"
)
RUNNER = ROOT / "tools" / "route2_release" / (
    "run_auxiliary_density_representation_gate.py"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str) -> dict[str, object]:
    value = json.loads((EVIDENCE / name).read_text())
    assert isinstance(value, dict)
    return value


def test_preregistered_runner_and_result_are_content_bound() -> None:
    preregistration = _load("preregistration.json")
    result = _load("result.json")

    assert preregistration["source_files_sha256"] == {
        "tools/route2_release/run_auxiliary_density_representation_gate.py": (
            _sha256(RUNNER)
        )
    }
    assert result["preregistration_file_sha256"] == _sha256(
        EVIDENCE / "preregistration.json"
    )
    assert result["preregistration_sha256"] == preregistration[
        "preregistration_sha256"
    ]
    assert result["record_count"] == 12


def test_locked_gate_remains_failed_without_reinterpreting_methane() -> None:
    result = _load("result.json")
    aggregate = result["aggregate"]
    gates = result["gates"]
    records = result["records"]

    assert result["status"] == "fail"
    assert gates == {"constraints": True, "maximum": False, "mean": True}
    assert aggregate["mean_constrained_area_relative_error"] == pytest.approx(
        0.03669208835457475,
        abs=0.0,
    )
    assert aggregate["maximum_constrained_area_relative_error"] == pytest.approx(
        0.19631874998988177,
        abs=0.0,
    )
    methane = next(
        record for record in records if record["compound_id"] == "mobley_9055303"
    )
    assert methane["name"] == "methane"
    assert methane["constrained"]["weighted_relative_l2"] == pytest.approx(
        aggregate["maximum_constrained_area_relative_error"],
        abs=0.0,
    )
    assert methane["constrained"][
        "maximum_absolute_error_hartree_per_e"
    ] == pytest.approx(0.0006943590977188663, abs=0.0)
    assert methane["constrained"]["correlation"] == pytest.approx(
        0.9969757262195107,
        abs=0.0,
    )


def test_evidence_claim_boundary_forbids_accuracy_or_capability_promotion() -> None:
    result = _load("result.json")

    assert result["claim_boundary"] == {
        "allowed_decision": "reject or retain this basis for later head training",
        "capability_admitted": False,
        "experimental_solvation_target_read": False,
        "fit_or_training_performed": False,
    }
