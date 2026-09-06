from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/route2/evidence/vqm24-observable-training-batch-20260827"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_observable_batch_preserves_enthalpy_conjugacy_failure() -> None:
    result = json.loads((EVIDENCE / "result.json").read_text())
    source = result["source"]

    assert result["status"] == "fail"
    assert source["sealer_sha256"] == _sha256(ROOT / source["sealer_path"])
    assert source["preregistration_file_sha256"] == _sha256(
        ROOT / source["preregistration_path"]
    )
    assert [name for name, passed in result["gates"].items() if not passed] == [
        "enthalpy_mep_conjugacy"
    ]
    assert result["aggregate"][
        "maximum_enthalpy_mep_conjugacy_relative"
    ] == pytest.approx(1.3367431176735478e-05, abs=0.0)
    assert result["aggregate"]["maximum_mep_nonlinearity"] == pytest.approx(
        0.0003384574689567132, abs=0.0
    )
    assert result["aggregate"]["maximum_dipole_nonlinearity"] == pytest.approx(
        0.00030263854647953067, abs=0.0
    )
    assert result["claim_boundary"]["quadratic_response_adequacy_passed"] is False
    assert result["claim_boundary"]["fit_or_training_performed"] is False

