from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "run_route2_v0_atomic_response_stationary_source_acetone.py"
)
PREREGISTRATION_PATH = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-response-stationary-source-acetone-prereg-v1.json"
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


def test_atomic_response_stationary_source_preregistration_freezes_the_candidate():
    namespace = runpy.run_path(str(RUNNER_PATH))
    preregistration = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))

    assert preregistration["schema_version"] == 1
    assert (
        preregistration["protocol_id"]
        == "route2-v0-atomic-response-stationary-source-acetone-prereg-v1"
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

    source_hashes = preregistration["execution_contract"]["source_sha256"]
    assert source_hashes == {
        relative: _sha256(ROOT / relative)
        for relative in namespace["SOURCE_RELATIVE_PATHS"]
    }
    input_hashes = preregistration["execution_contract"]["input_sha256"]
    assert input_hashes == {
        relative: _sha256(ROOT / relative)
        for relative in namespace["INPUT_RELATIVE_PATHS"]
    }


def test_atomic_response_stationary_source_theory_forbids_a_hidden_charge_fix():
    theory = THEORY_PATH.read_text(encoding="utf-8")

    assert "x_0=-Cb" in theory
    assert "QEq" in theory
    assert "not a claimed ensemble one-particle density matrix" in theory
    assert "not a Harris-like nonselfconsistent energy evaluation" in theory
    assert "Route-2 solvation method" in theory
