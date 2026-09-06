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
    / "local-whitened-etb2-condition-gate-20260824"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str) -> dict[str, object]:
    value = json.loads((EVIDENCE / name).read_text())
    assert isinstance(value, dict)
    return value


def test_local_whitening_gate_replays_bound_sources_and_preregistration() -> None:
    preregistration = _load("preregistration.json")
    result = _load("result.json")

    for relative, expected in preregistration["source_files_sha256"].items():
        assert _sha256(ROOT / relative) == expected
    assert result["preregistration_file_sha256"] == _sha256(
        EVIDENCE / "preregistration.json"
    )
    assert result["preregistration_sha256"] == preregistration[
        "preregistration_sha256"
    ]


def test_local_whitening_fails_only_benzene_and_aniline_global_condition() -> None:
    result = _load("result.json")
    failed = {
        record["name"]: record["global_restricted_condition_number"]
        for record in result["records"]
        if not record["case_passed"]
    }

    assert result["status"] == "fail"
    assert result["gates"] == {
        "all_cases": False,
        "constraint_condition": True,
        "constraint_residual": True,
        "global_condition": False,
    }
    assert failed == {
        "benzene": pytest.approx(31812563063.39607, abs=0.0),
        "aniline": pytest.approx(38525665813.60515, abs=0.0),
    }
    assert result["aggregate"]["maximum_constraint_gram_condition_number"] < 50.0
    assert result["aggregate"]["maximum_constraint_absolute_residual"] < 1.0e-10
    assert result["claim_boundary"]["capability_admitted"] is False
