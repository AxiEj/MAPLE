from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_core import sha256_file
from source_compatibility import (
    load_source_compatibility,
    require_exact_frozen_sources,
)


def test_production_safety_source_compatibility_is_narrow_and_current():
    artifact = load_source_compatibility()

    assert artifact["historical_evidence_bytes_unchanged"] is True
    assert artifact["frozen_results_recomputed"] is False
    assert artifact["scientific_claim_promoted"] is False
    assert artifact["route1_formula_changed"] is False
    assert artifact["gas_phase_mm_energy_added"] is False
    assert artifact["hydration_label_residual_added"] is False
    assert artifact["qrrho_v8_status_remains_failed_closed"] is True

    records = artifact["source_changes"]
    assert records
    assert len({record["path"] for record in records}) == len(records)
    for record in records:
        path = REPOSITORY_ROOT / record["path"]
        assert path.is_file()
        assert sha256_file(path) == record["current_sha256"]
        assert record["historical_numerical_results_recomputed"] is False
        assert record["scientific_claim_reuse_authorized"] is False
        assert all(len(value) == 64 for value in record["historical_sha256"])


def test_documented_source_drift_never_authorizes_execution_or_sealing():
    artifact = load_source_compatibility()
    changed = next(
        record
        for record in artifact["source_changes"]
        if record["historical_sha256"]
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
