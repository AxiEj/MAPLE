from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs" / "implicit-solvation" / "benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core

VALIDATION_PATH = (
    BENCHMARK_DIR / "route1-torch-obc2-analytic-derivatives-2026-09-22.json"
)
PERFORMANCE_PATH = (
    BENCHMARK_DIR / "route1-torch-obc2-derivative-performance-2026-09-22.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_validation_artifact_is_self_hashed_source_bound_and_label_free():
    artifact = _load(VALIDATION_PATH)

    assert core.artifact_content_sha256(artifact) == artifact["content_sha256"]
    assert artifact["scientific_identity"]["new_physics"] is False
    assert artifact["scientific_identity"]["parameters_changed"] is False
    assert artifact["scientific_identity"]["hydration_accuracy_changed"] is False
    assert artifact["gate_a_reference_parity"]["passed"] is True
    assert artifact["gate_a_reference_parity"]["comparison_count"] == 48
    assert artifact["analytic_workflow_reference_cells"]["passed"] is True
    assert artifact["production_cpu_gate_c_diagnostic"]["passed"] is False
    assert (
        "Reference platform" in artifact["production_cpu_gate_c_diagnostic"]["decision"]
    )
    assert "experimental" in json.dumps(artifact["performance_decision"]).lower()

    for relative, expected in artifact["implementation_source_sha256"].items():
        assert core.sha256_file(REPOSITORY_ROOT / relative) == expected


def test_performance_artifact_is_self_hashed_and_linked_from_validation():
    performance = _load(PERFORMANCE_PATH)
    validation = _load(VALIDATION_PATH)
    link = validation["performance_artifact"]

    assert core.artifact_content_sha256(performance) == performance["content_sha256"]
    assert core.sha256_file(PERFORMANCE_PATH) == link["file_sha256"]
    assert performance["content_sha256"] == link["content_sha256"]
    assert performance["label_reads"] is False
    assert performance["interpretation"]["universal_performance_claim"] is False
    assert performance["interpretation"]["openmm_runtime_default_changed"] is False
    assert performance["records"]["methyl-hexanoate"]["direct_hvp_speedup"] > 5.0
    assert (
        performance["records"]["methyl-hexanoate"]["analytic_over_numerical_speedup"]
        < 3.0
    )

    script = REPOSITORY_ROOT / performance["command_provenance"]["script"]
    assert (
        core.sha256_file(script) == performance["command_provenance"]["script_sha256"]
    )
