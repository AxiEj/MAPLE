"""Regression guards for the frozen GFN2 MOLDEN permanent-source evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARKS = _REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
_ARTIFACT_PATH = _BENCHMARKS / "route2-v0-gfn2-molden-permanent-source-acetone-v2.json"
_PREREGISTRATION_PATH = (
    _BENCHMARKS / "route2-v0-gfn2-molden-permanent-source-acetone-prereg-v2.json"
)
_V1_PREFLIGHT_FAILURE_PATH = (
    _BENCHMARKS
    / "route2-v0-gfn2-molden-permanent-source-acetone-preflight-failure-v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_gfn2_molden_v2_artifact_is_source_bound_and_not_an_accuracy_claim():
    artifact = _load(_ARTIFACT_PATH)
    preregistration = _load(_PREREGISTRATION_PATH)

    assert artifact["schema_version"] == 1
    assert artifact["artifact"] == "route2-v0-gfn2-molden-permanent-source-acetone-v2"
    assert artifact["status"] == "pass"
    assert artifact["execution_git_head"] == "b0a4a52617e86dae619383d74814d896eccb8755"
    assert artifact["preregistration"] == {
        "path": (
            "docs/implicit-solvation/benchmarks/"
            "route2-v0-gfn2-molden-permanent-source-acetone-prereg-v2.json"
        ),
        "sha256": _sha256(_PREREGISTRATION_PATH),
    }
    assert artifact["hard_constraints"] == preregistration["hard_constraints"]
    assert artifact["decision"]["verdict"] == (
        "admit-only-to-preregistered-static-qm-mep-source-gate"
    )
    boundary = artifact["claim_boundary"].lower()
    assert "solvation energy" in boundary
    assert "experimental accuracy" in boundary

    source_hashes = artifact["source_files_sha256"]
    assert source_hashes == preregistration["execution_contract"]["source_sha256"]
    for relative_path, digest in source_hashes.items():
        assert _sha256(_REPOSITORY_ROOT / relative_path) == digest

    geometry_relative_path = (
        "docs/implicit-solvation/benchmarks/reproducers/"
        "route2-v0-gfn2-molden-permanent-source-acetone-v1/acetone.xyz"
    )
    assert artifact["input_files_sha256"][geometry_relative_path] == _sha256(
        _REPOSITORY_ROOT / geometry_relative_path
    )
    assert artifact["candidate"] == {
        "construction": "route2-v0-gfn2-molden-permanent-source-v1",
        "electronic_method": "GFN2-xTB",
        "parameter_file_sha256": (
            "f6f05c6c5264cb2f75c54f8ac6838044541c1e493b16f1a4d4ee4fa64de5f545"
        ),
        "scope": "zero-field-permanent-source-only-not-xtb-response-v1",
        "xtb_version": "6.7.1 (edcfbbe)",
    }

    round_trip = artifact["permanent_source_round_trip"]
    gates = preregistration["numerical_gates"]
    assert round_trip["ao_count"] == 22
    assert round_trip["mo_metric_error"] < gates["mo_metric_relative_operator_max"]
    assert (
        round_trip["valence_electron_count_error_e"]
        < gates["valence_electron_count_absolute_max"]
    )
    assert (
        abs(round_trip["total_effective_charge_e"])
        < gates["effective_charge_absolute_max"]
    )
    assert (
        round_trip["dipole_round_trip_error_e_bohr"]
        < gates["dipole_round_trip_e_bohr_max"]
    )
    assert math.isclose(
        round_trip["valence_electron_count_e"], 24.0, rel_tol=0.0, abs_tol=5.0e-8
    )


def test_gfn2_molden_v1_preflight_failure_is_preserved_not_overwritten():
    failure = _load(_V1_PREFLIGHT_FAILURE_PATH)

    assert failure["status"] == "preflight-failed-no-source-gate-evaluated"
    assert failure["stage"] == "runner-source-module-import"
    assert failure["failure"]["exception_type"] == "ImportError"
    assert failure["result_handling"] == {
        "decision_artifact_written": False,
        "numerical_gates_evaluated": False,
        "raw_xtb_output_used_for_acceptance_or_selection": False,
        "source_representation_evaluated": False,
        "successor_protocol": (
            "route2-v0-gfn2-molden-permanent-source-acetone-prereg-v2"
        ),
    }
