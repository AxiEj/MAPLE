from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_core import sha256_file
from source_compatibility import (
    load_source_compatibility,
    require_exact_frozen_sources,
    validate_frozen_source,
)


def test_production_safety_ledger_preserves_history_and_retained_sources():
    artifact = load_source_compatibility()
    ledger = (
        BENCHMARK_DIR / "route1-production-safety-source-compatibility-2026-09-27.json"
    )
    assert sha256_file(ledger) == (
        "4658b7dc8e0d7c1fa510e62e391d04aab9d076b084183b590384055b651fc17d"
    )
    # Explicitly retired research is still present in the immutable ledger,
    # not silently removed from history or accepted as an active source file.
    retired = {
        "maple/function/calculator/extra_correction/implicit/" + name + ".py"
        for name in (
            "continuum_chagb_inputs",
            "torch_continuum_sav",
            "torch_sphere_union_geometry",
        )
    }

    assert artifact["historical_evidence_bytes_unchanged"] is True
    assert artifact["frozen_results_recomputed"] is False
    assert artifact["scientific_claim_promoted"] is False
    assert artifact["route1_formula_changed"] is False
    assert artifact["gas_phase_mm_energy_added"] is False
    assert artifact["hydration_label_residual_added"] is False
    assert artifact["qrrho_v8_status_remains_failed_closed"] is True

    records = artifact["source_changes"]
    assert records
    assert retired <= {record["path"] for record in records}
    assert len({record["path"] for record in records}) == len(records)
    for record in records:
        path = REPOSITORY_ROOT / record["path"]
        if record["path"] in retired:
            assert not path.exists()
        else:
            assert path.is_file()
            assert sha256_file(path) == record["current_sha256"]
        assert record["historical_numerical_results_recomputed"] is False
        assert record["scientific_claim_reuse_authorized"] is False
        assert all(len(value) == 64 for value in record["historical_sha256"])


def test_documented_source_drift_never_authorizes_execution_or_sealing():
    artifact = load_source_compatibility()
    changed = next(
        record for record in artifact["source_changes"] if record["historical_sha256"]
    )

    with pytest.raises(ValueError, match="historical-audit-only"):
        require_exact_frozen_sources(
            REPOSITORY_ROOT,
            [
                {
                    "path": changed["path"],
                    "sha256": changed["historical_sha256"][0],
                }
            ],
        )


@pytest.mark.parametrize(
    "path",
    [
        "maple/function/read/command_control.py",
        "maple/function/calculator/extra_correction/implicit/charges.py",
    ],
)
def test_runtime_selection_changes_are_archival_only_not_old_evidence_reuse(path):
    record = next(
        record
        for record in load_source_compatibility()["source_changes"]
        if record["path"] == path
    )
    assert record["qeq_cqeq_runtime_enabled"] is False
    assert record["scientific_claim_reuse_authorized"] is False
    for historical in record["historical_sha256"]:
        status = validate_frozen_source(REPOSITORY_ROOT, record["path"], historical)
        assert status["mode"] == "documented-postexecution-production-safety-change"
        with pytest.raises(ValueError, match="historical-audit-only"):
            require_exact_frozen_sources(
                REPOSITORY_ROOT, [{"path": record["path"], "sha256": historical}]
            )
