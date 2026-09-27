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
REVALIDATION_PATH = (
    BENCHMARK_DIR / "route1-torch-post-audit-revalidation-2026-09-27.json"
)
HISTORICAL_VALIDATION_SHA256 = (
    "961ba66a97a7b9a0c6d80640409bfb1218de496d42757c06e8dfd2e1e81947f5"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_validation_artifact_is_self_hashed_source_bound_and_label_free():
    artifact = _load(VALIDATION_PATH)

    # The old observation is immutable history, not a certificate for revised
    # source. A fresh independently executed artifact below binds current code.
    assert core.sha256_file(VALIDATION_PATH) == HISTORICAL_VALIDATION_SHA256
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

    revalidation = _load(REVALIDATION_PATH)
    assert revalidation["supersedes_current_source_binding_of"] == {
        "file": VALIDATION_PATH.name,
        "file_sha256": HISTORICAL_VALIDATION_SHA256,
        "content_sha256": artifact["content_sha256"],
    }
    assert set(artifact["implementation_source_sha256"]) <= set(
        revalidation["implementation_and_test_sha256"]
    )


def test_post_audit_revalidation_binds_current_sources_and_fresh_results():
    artifact = _load(REVALIDATION_PATH)
    assert core.artifact_content_sha256(artifact) == artifact["content_sha256"]
    assert artifact["passed"] is artifact["source_unchanged"] is True
    assert artifact["historical_evidence_bytes_unchanged"] is True
    assert artifact["label_reads"] is artifact["new_qm"] is False
    assert (
        artifact["experimental_accuracy_claim"]
        is artifact["full_torch_cha_complete"]
        is False
    )
    for path, expected in artifact["implementation_and_test_sha256"].items():
        assert core.sha256_file(REPOSITORY_ROOT / path) == expected
        assert artifact["source_sha256_after"][path] == expected
    assert artifact["qualified_workflow_cell"] == {
        "gas": "ani2x",
        "device": "cpu",
        "dtype": "float64",
        "d4": False,
        "solvent": "obc2",
        "openmm_platform": "Reference",
    }
    records = artifact["corpus"]["records"]
    assert artifact["corpus"]["comparison_count"] == len(records) == 48
    assert len({r["case"] for r in records}) == len(artifact["corpus"]["cases"]) == 12
    assert all(r["passed"] for r in records)
    assert all(r["force_max_delta_hartree_per_angstrom"] <= 2e-7 for r in records)
    assert artifact["regression"]["exit_code"] == 0
    cells = artifact["regression"]["cases"]
    assert cells and all(cell["status"] == "passed" for cell in cells)
    assert any("test_actual_engine_analytic_frequency" in c["name"] for c in cells)
    assert any(
        "test_actual_engine_analytic_ts_derivative_paths" in c["name"] for c in cells
    )


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
