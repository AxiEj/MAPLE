from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
RESULT = (
    BENCHMARKS / "route2-v0-atomic-independent-particle-source-acetone-v1.json"
)
PREREG = (
    BENCHMARKS
    / "route2-v0-atomic-independent-particle-source-acetone-prereg-v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_atomic_independent_particle_source_result_passes_only_registered_qm_mep_gate():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))

    assert result["artifact"] == "route2-v0-atomic-independent-particle-source-acetone-v1"
    assert result["status"] == "pass"
    assert result["execution_git_head"] == (
        "f5e9939d93f7e0d519143d14d5187d329e81aa27"
    )
    assert result["preregistration"] == {
        "path": (
            "docs/implicit-solvation/benchmarks/"
            "route2-v0-atomic-independent-particle-source-acetone-prereg-v1.json"
        ),
        "protocol_id": "route2-v0-atomic-independent-particle-source-acetone-prereg-v1",
        "sha256": _sha256(PREREG),
    }
    assert result["input_files_sha256"] == prereg["execution_contract"]["input_sha256"]
    assert result["source_files_sha256"] == prereg["execution_contract"][
        "source_sha256"
    ]
    assert result["system"]["source_point_count"] == 516
    assert result["response_kernel"]["coefficient_count"] == 727
    assert result["response_kernel"]["support_constraint_count"] == 27
    assert result["response_kernel"]["charge_constraint_vector"] is None
    assert result["response_kernel"]["completed_minimum_eigenvalue"] >= -1.0e-12
    assert result["source_representation"]["source_potential_mode_matrix_sha256"] == (
        "ab5ae1abb2dfdea4f82394119c184a87e4ceeafc787e812be1bc6ad16ea3c7b9"
    )
    assert all(check["passes"] for check in result["numerical_checks"].values())
    assert all(
        check["passes"]
        for check in result["scientific_falsification"]["checks"].values()
    )
    assert result["scientific_falsification"]["verdict"] == (
        "admit-frozen-atomic-independent-particle-source-to-same-basis-"
        "common-scalar-kkt-gates-only"
    )


def test_atomic_independent_particle_source_metrics_are_not_a_solvation_claim():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    checks = result["scientific_falsification"]["checks"]

    assert checks["mep_response_relative_frobenius"]["value"] == pytest.approx(
        0.09667637936478042
    )
    assert checks["mep_response_relative_direction_max"]["value"] == pytest.approx(
        0.11009245472840734
    )
    assert checks["induced_dipole_response_relative_frobenius"]["value"] == (
        pytest.approx(0.0038678621764154297)
    )
    assert all(value is False for value in result["hard_constraints"].values())
    assert "not an end-to-end Route-2" in result["runtime"]["boundary"]
    assert "does not establish" in result["scientific_falsification"][
        "admission_boundary"
    ]
    assert "cannot establish chemical accuracy" in result["claim_boundary"]
