from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "run_route2_v0_atomic_response_stationary_source_acetone.py"
)
V1_PREREGISTRATION_PATH = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-response-stationary-source-acetone-prereg-v1.json"
)
V2_PREREGISTRATION_PATH = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-response-stationary-source-acetone-prereg-v2.json"
)
V1_PREFLIGHT_FAILURE_PATH = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-response-stationary-source-acetone-preflight-failure-v1.json"
)
V2_COMPLETED_ARTIFACT_PATH = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-response-stationary-source-acetone-v2.json"
)
THEORY_PATH = (
    ROOT / "docs/implicit-solvation/"
    "ROUTE2_V0_ATOMIC_RESPONSE_STATIONARY_PERMANENT_SOURCE_THEORY.md"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha256(revision: str, relative_path: str) -> str:
    payload = subprocess.check_output(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=ROOT,
    )
    return hashlib.sha256(payload).hexdigest()


def test_atomic_response_stationary_source_v1_preflight_failure_is_preserved():
    preregistration = json.loads(V1_PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    failure = json.loads(V1_PREFLIGHT_FAILURE_PATH.read_text(encoding="utf-8"))

    assert preregistration["status"] == "frozen-before-execution"
    assert failure["status"] == (
        "preflight-failed-after-helper-evaluation-no-final-gate-verdict-v1"
    )
    assert failure["stage"] == "parent-scientific-gate-key-lookup"
    assert failure["failure"] == {
        "cause": (
            "The V1 parent runner registered static gate names but indexed those names "
            "directly in the helper comparison record, whose deliberately shorter fields "
            "are mep_relative_frobenius, mep_relative_max_abs, and "
            "dipole_relative_frobenius."
        ),
        "exception_type": "KeyError",
        "message": "static_mep_relative_frobenius",
    }
    assert failure["result_handling"] == {
        "candidate_definition_changed": False,
        "density_grid_changed": False,
        "final_registered_gate_values_accessed_or_decided": False,
        "helper_output_preserved": True,
        "source_points_changed": False,
        "thresholds_changed": False,
        "v1_final_artifact_written": False,
    }
    assert (
        failure["source_files_sha256"]
        == preregistration["execution_contract"]["source_sha256"]
    )
    for relative_path, digest in failure["source_files_sha256"].items():
        assert _git_blob_sha256(failure["execution_git_head"], relative_path) == digest
    sealed = ROOT / failure["sealed_helper_output"]["path"]
    assert _sha256(sealed) == failure["sealed_helper_output"]["sha256"]
    assert sealed.stat().st_size == failure["sealed_helper_output"]["bytes"]


def test_atomic_response_stationary_source_v2_preregistration_freezes_the_candidate():
    namespace = runpy.run_path(str(RUNNER_PATH))
    preregistration = json.loads(V2_PREREGISTRATION_PATH.read_text(encoding="utf-8"))

    assert preregistration["schema_version"] == 1
    assert (
        preregistration["protocol_id"]
        == "route2-v0-atomic-response-stationary-source-acetone-prereg-v2"
    )
    assert preregistration["status"] == "frozen-before-execution"
    assert preregistration["hard_constraints"] == namespace["HARD_CONSTRAINTS"]
    assert all(
        value is False
        for key, value in preregistration["hard_constraints"].items()
        if key != "all_electron_atomic_reference_only"
    )
    assert preregistration["hard_constraints"]["all_electron_atomic_reference_only"]
    assert preregistration["scientific_falsification_gates"] == {
        "static_dipole_relative_frobenius_max": 0.2,
        "static_mep_relative_frobenius_max": 0.2,
        "static_mep_relative_max_abs_max": 0.3,
    }
    assert preregistration["source_definition"]["density_positivity_grid"] == {
        "buffer_bohr": 5.0,
        "construction": (
            "Cartesian nuclear-coordinate bounding box plus buffer, evaluated "
            "without clipping, projection, or electron-count renormalization."
        ),
        "spacing_bohr": 0.4,
    }
    assert preregistration["protocol_revision"] == {
        "candidate_definition_changed": False,
        "density_grid_changed": False,
        "preflight_failure": {
            "path": namespace["PREFLIGHT_FAILURE_RELATIVE_PATH"],
            "sha256": _sha256(ROOT / namespace["PREFLIGHT_FAILURE_RELATIVE_PATH"]),
        },
        "qm_comparator_changed": False,
        "reason": (
            "V1 parent runner completed its helper then raised KeyError while looking up "
            "registered static gate names directly in helper fields. V2 maps the unchanged "
            "registered names to the helper fields before indexing. The V1 helper output is "
            "retained but not used for a final decision or candidate selection."
        ),
        "response_or_continuum_added": False,
        "scientific_thresholds_changed": False,
        "sealed_unread_helper_output": {
            "path": (
                "docs/implicit-solvation/benchmarks/"
                "route2-v0-atomic-response-stationary-source-acetone-helper-preflight-v1.json"
            ),
            "sha256": _sha256(
                ROOT / "docs/implicit-solvation/benchmarks/"
                "route2-v0-atomic-response-stationary-source-acetone-helper-preflight-v1.json"
            ),
        },
        "source_points_changed": False,
        "supersedes_protocol_id": (
            "route2-v0-atomic-response-stationary-source-acetone-prereg-v1"
        ),
    }

    source_hashes = preregistration["execution_contract"]["source_sha256"]
    completed = json.loads(V2_COMPLETED_ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert completed["status"] == "reject"
    assert completed["source_files_sha256"] == source_hashes
    assert completed["preregistration"] == {
        "path": str(V2_PREREGISTRATION_PATH.relative_to(ROOT)),
        "protocol_id": preregistration["protocol_id"],
        "sha256": _sha256(V2_PREREGISTRATION_PATH),
    }
    for relative, digest in source_hashes.items():
        assert _git_blob_sha256(completed["execution_git_head"], relative) == digest
    assert source_hashes != {
        relative: _sha256(ROOT / relative)
        for relative in namespace["SOURCE_RELATIVE_PATHS"]
    }
    input_hashes = preregistration["execution_contract"]["input_sha256"]
    assert completed["input_files_sha256"] == input_hashes
    assert namespace["SCIENTIFIC_GATE_HELPER_FIELDS"] == {
        "static_dipole_relative_frobenius": "dipole_relative_frobenius",
        "static_mep_relative_frobenius": "mep_relative_frobenius",
        "static_mep_relative_max_abs": "mep_relative_max_abs",
    }


def test_atomic_response_stationary_source_theory_forbids_a_hidden_charge_fix():
    theory = THEORY_PATH.read_text(encoding="utf-8")

    assert "x_0=-Cb" in theory
    assert "QEq" in theory
    assert "not a claimed ensemble one-particle density matrix" in theory
    assert "not a Harris-like nonselfconsistent energy evaluation" in theory
    assert "Route-2 solvation method" in theory
