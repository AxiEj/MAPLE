from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
PREREGISTRATION = (
    BENCHMARKS / "route2-v0-promolecular-atomic-hf-def2-tzvpd-prereg-v1.json"
)
MANIFEST = BENCHMARKS / "route2-v0-promolecular-atomic-hf-def2-tzvpd-v1.json"
TABLE = BENCHMARKS / "route2-v0-promolecular-atomic-hf-def2-tzvpd-v1.npz"
RUNNER = BENCHMARKS / "generate_route2_v0_promolecular_density.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_promolecular_density_artifact_is_bound_to_its_frozen_generator_and_input():
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    artifact = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert artifact["artifact"] == "route2-v0-promolecular-atomic-hf-def2-tzvpd-v1"
    assert artifact["status"] == "pass"
    assert artifact["preregistration"] == {
        "path": "docs/implicit-solvation/benchmarks/"
        "route2-v0-promolecular-atomic-hf-def2-tzvpd-prereg-v1.json",
        "sha256": _sha256(PREREGISTRATION),
    }
    runner_key = "docs/implicit-solvation/benchmarks/generate_route2_v0_promolecular_density.py"
    assert preregistration["source_sha256"][runner_key] == _sha256(RUNNER)
    assert artifact["source_files_sha256"] == {runner_key: _sha256(RUNNER)}
    assert artifact["table"] == {
        "path": "docs/implicit-solvation/benchmarks/"
        "route2-v0-promolecular-atomic-hf-def2-tzvpd-v1.npz",
        "sha256": _sha256(TABLE),
    }


def test_promolecular_density_artifact_passes_each_preregistered_mathematical_gate():
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    artifact = json.loads(MANIFEST.read_text(encoding="utf-8"))
    gates = preregistration["generation_contract"]["validation_gates"]

    assert [result["symbol"] for result in artifact["results"]] == [
        entry["symbol"]
        for entry in preregistration["generation_contract"]["elements"]
    ]
    assert all(result["scf_converged"] for result in artifact["results"])
    assert max(
        result["electron_count_integral_absolute_error"]
        for result in artifact["results"]
    ) <= gates["electron_count_absolute_error_max"]
    assert max(
        result["spherical_probe_maximum_difference_e_per_bohr3"]
        for result in artifact["results"]
    ) <= gates["spherical_probe_maximum_difference_max"]
    assert max(
        result["radial_tail_density_e_per_bohr3"]
        for result in artifact["results"]
    ) <= gates["radial_tail_density_max_e_per_bohr3"]
    assert min(
        result["minimum_density_e_per_bohr3"] for result in artifact["results"]
    ) >= 0.0
