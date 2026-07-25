from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
ARTIFACT_PATH = (
    BENCHMARK_DIR / "route1-discrete-conformer-core-replay-2026-07-25.json"
)
RUNNER_PATH = BENCHMARK_DIR / "run_mlip_conformer_weighting.py"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core


def _artifact() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_discrete_conformer_core_replay_is_sealed_and_exact():
    artifact = _artifact()
    records = artifact["records"]

    assert artifact["content_sha256"] == core.artifact_content_sha256(artifact)
    assert artifact["command_provenance"]["script_sha256"] == core.sha256_file(
        RUNNER_PATH
    )
    assert artifact["record_count"] == len(records) == 40
    assert artifact["state_count"] == sum(row["state_count"] for row in records) == 2596
    assert artifact["all_legacy_fields_exact"] is True
    assert {row["model"] for row in records} == {"maceoff23m"}
    assert all(row["legacy_field_max_abs_difference"] == 0.0 for row in records)
    assert all(
        difference == 0.0
        for row in records
        for difference in row["legacy_field_differences"].values()
    )
    assert all(len(row["source_record_sha256"]) == 64 for row in records)


def test_discrete_conformer_core_replay_locks_weight_diagnostics():
    summaries = _artifact()["method_summaries"]

    assert set(summaries) == {"abcg2/obc2", "am1bcc/obc2"}
    for summary in summaries.values():
        assert summary["record_count"] == 20
        assert summary["state_count"] == 1298
        assert summary["weight_diagnostic_pass_count"] == 14
        assert summary["gas_effective_conformer_count"]["minimum"] == 1.0
        assert summary["solution_effective_conformer_count"]["minimum"] == 1.0

    assert summaries["abcg2/obc2"]["distribution_overlap"]["median"] == pytest.approx(
        0.9461890588948296
    )
    assert summaries["am1bcc/obc2"]["distribution_overlap"][
        "median"
    ] == pytest.approx(0.9424539385294269)


def test_discrete_conformer_core_replay_preserves_route_and_claim_boundaries():
    artifact = _artifact()

    assert artifact["route"] == {
        "name": "Additive fixed-charge PB/GB implicit solvation",
        "formula": (
            "E_solution(R)=E_MLIP,gas(R)+"
            "G_polar(R,q_fixed)+G_nonpolar(R)"
        ),
        "gas_phase_mm_energy": False,
        "hydration_label_residual": False,
        "retraining": False,
    }
    assert artifact["claim_boundary"][
        "historical_records_contain_experimental_fields"
    ] is True
    assert artifact["claim_boundary"][
        "experimental_fields_used_by_core_or_replay_metrics"
    ] is False
    assert artifact["claim_boundary"]["public_solvfe_eligible"] is False


def test_route1_docs_bind_discrete_core_evidence_and_limits():
    paths = [
        REPOSITORY_ROOT / "docs/implicit-solvation/FORMULAS_AND_REFERENCES.md",
        REPOSITORY_ROOT / "docs/implicit-solvation/README.md",
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md",
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md",
        BENCHMARK_DIR / "README.md",
    ]
    normalized = " ".join(
        " ".join(path.read_text(encoding="utf-8").split()) for path in paths
    )

    assert "analyze_discrete_conformer_ensemble" in normalized
    assert "route1-discrete-conformer-core-replay-2026-07-25.json" in normalized
    assert "40 historical MACE-OFF23m records" in normalized
    assert "2,596 states" in normalized
    assert "14/20 AM1-BCC" in normalized
    assert "14/20 ABCG2" in normalized
    assert "MACE-only" in normalized
    assert "no public `#solvfe` task is opened" in normalized
