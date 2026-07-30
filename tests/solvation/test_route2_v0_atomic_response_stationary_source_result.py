from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
RESULT = BENCHMARKS / "route2-v0-atomic-response-stationary-source-acetone-v2.json"
PREREG = (
    BENCHMARKS / "route2-v0-atomic-response-stationary-source-acetone-prereg-v2.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_atomic_response_stationary_source_v2_rejection_is_frozen():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))

    assert _sha256(RESULT) == (
        "a10a7c7211899143f8bfb1ef22053a247ab0fb840c508ba0532ddc1bb31c6c8f"
    )
    assert (
        result["artifact"] == "route2-v0-atomic-response-stationary-source-acetone-v2"
    )
    assert result["status"] == "reject"
    assert result["execution_git_head"] == "1901b7dd7cb89320901cac0954feaaad9f818d2e"
    assert result["preregistration"] == {
        "path": (
            "docs/implicit-solvation/benchmarks/"
            "route2-v0-atomic-response-stationary-source-acetone-prereg-v2.json"
        ),
        "protocol_id": "route2-v0-atomic-response-stationary-source-acetone-prereg-v2",
        "sha256": _sha256(PREREG),
    }
    assert result["input_files_sha256"] == prereg["execution_contract"]["input_sha256"]
    assert (
        result["source_files_sha256"] == prereg["execution_contract"]["source_sha256"]
    )
    assert result["system"]["source_point_count"] == 516


def test_atomic_response_stationary_source_v2_fails_independent_physical_gates():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    checks = result["scientific_falsification"]["checks"]
    numerical = result["numerical_checks"]

    assert result["stationary_reference"]["stationarity_residual_inf_hartree"] == (
        pytest.approx(4.0419056990259605e-15)
    )
    assert result["stationary_reference"]["support_constraint_residual_inf"] == (
        pytest.approx(9.540979117872439e-17)
    )
    assert numerical["permanent_density_grid_minimum_electron_number_density"] == {
        "minimum": -1.0e-10,
        "passes": False,
        "value": pytest.approx(-8.643801638337562e-06),
    }
    assert checks["static_mep_relative_frobenius"] == {
        "maximum": 0.2,
        "passes": False,
        "value": pytest.approx(1.208788374621507),
    }
    assert checks["static_mep_relative_max_abs"] == {
        "maximum": 0.3,
        "passes": False,
        "value": pytest.approx(1.2708329141597103),
    }
    assert checks["static_dipole_relative_frobenius"] == {
        "maximum": 0.2,
        "passes": False,
        "value": pytest.approx(1.0841449738576339),
    }
    assert result["scientific_falsification"]["verdict"] == (
        "reject-this-exact-atomic-response-stationary-permanent-source"
    )
    assert result["candidate"]["response_or_continuum"] == "not invoked"
    assert (
        "does not establish" in result["scientific_falsification"]["admission_boundary"]
    )


def test_atomic_response_stationary_source_v2_retains_its_nondecisive_checkpoint_count_defect():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    checkpoint = result["qm_checkpoint"]
    numerical = result["numerical_checks"]

    assert checkpoint["electron_count_e"] == pytest.approx(32.0)
    assert checkpoint["total_charge_e"] == pytest.approx(0.0)
    assert numerical["qm_checkpoint_electron_count_error"] == {
        "maximum": 1.0e-07,
        "passes": False,
        "value": pytest.approx(16.0),
    }
