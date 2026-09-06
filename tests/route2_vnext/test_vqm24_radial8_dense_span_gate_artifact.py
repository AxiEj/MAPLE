from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/route2/evidence/vqm24-radial8-dense-span-gate-20260827"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_radial8_dense_span_gate_closes_the_complete_quantitative_source() -> None:
    result = json.loads((EVIDENCE / "result.json").read_text())
    source = result["source"]

    assert result["status"] == "fail"
    assert source["runner_sha256"] == _sha256(ROOT / source["runner_path"])
    assert source["preregistration_file_sha256"] == _sha256(
        ROOT / source["preregistration_path"]
    )
    failures = [record for record in result["records"] if record["status"] == "fail"]
    assert len(failures) == 6
    bromine = next(
        record
        for record in failures
        if record["record_id"] == "bromine-63727-71eb4b0867c9"
    )
    assert [name for name, passed in bromine["gates"].items() if not passed] == [
        "response_mep_audit_absolute"
    ]
    assert bromine["metrics"][
        "response_mep_audit_max_absolute_hartree_per_e_per_source_e"
    ] == pytest.approx(0.0033968300301452925, abs=0.0)
    assert result["aggregate"]["maximum_zero_mep_audit_relative"] == pytest.approx(
        0.08293519807849942, abs=0.0
    )
    assert result["claim_boundary"][
        "radial8_source_span_admitted_for_observable_head_training"
    ] is False
    assert all(
        record["metrics"]["coefficient_label_emitted"] is False
        for record in result["records"]
    )
