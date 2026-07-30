from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
RESULT = BENCHMARKS / "route2-v0-atomic-displacement-source-acetone-v1.json"
PREREG = BENCHMARKS / "route2-v0-atomic-displacement-source-acetone-prereg-v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_atomic_displacement_source_result_passes_only_the_registered_qm_mep_gate():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))

    assert result["artifact"] == "route2-v0-atomic-displacement-source-acetone-v1"
    assert result["status"] == "pass"
    assert result["preregistration"] == {
        "path": (
            "docs/implicit-solvation/benchmarks/"
            "route2-v0-atomic-displacement-source-acetone-prereg-v1.json"
        ),
        "protocol_id": "route2-v0-atomic-displacement-source-acetone-prereg-v1",
        "sha256": _sha256(PREREG),
    }
    assert result["input_files_sha256"] == prereg["execution_contract"]["input_sha256"]
    assert (
        result["source_files_sha256"] == prereg["execution_contract"]["source_sha256"]
    )
    assert result["system"]["source_point_count"] == 516
    assert all(check["passes"] for check in result["numerical_checks"].values())
    assert all(
        check["passes"]
        for check in result["scientific_falsification"]["checks"].values()
    )
    assert result["scientific_falsification"]["verdict"] == (
        "admit-frozen-atomic-displacement-source-to-same-basis-common-scalar-kkt-gates-only"
    )


def test_atomic_displacement_source_qm_mep_metrics_are_not_an_accuracy_claim():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    checks = result["scientific_falsification"]["checks"]

    assert checks["mep_response_relative_frobenius"]["value"] == pytest.approx(
        0.1484314508938934
    )
    assert checks["mep_response_relative_direction_max"]["value"] == pytest.approx(
        0.1611208414744773
    )
    assert checks["induced_dipole_response_relative_frobenius"][
        "value"
    ] == pytest.approx(0.003867862176415407)
    assert all(value is False for value in result["hard_constraints"].values())
    assert "not an end-to-end Route-2" in result["runtime"]["boundary"]
    assert (
        "does not establish" in result["scientific_falsification"]["admission_boundary"]
    )
