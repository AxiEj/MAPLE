from __future__ import annotations

import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core


def test_route1_runtime_and_formula_ledger_exclude_invalid_r6_doi():
    invalid_doi = "10.1021/acs.jctc.4c01471"
    current_files = [
        REPOSITORY_ROOT
        / "maple/function/calculator/extra_correction/implicit/amber_chagb.py",
        REPOSITORY_ROOT / "docs/implicit-solvation/FORMULAS_AND_REFERENCES.md",
    ]

    assert all(
        invalid_doi not in path.read_text(encoding="utf-8")
        for path in current_files
    )


def test_route1_citation_erratum_is_self_hashed_and_scoped_to_metadata():
    path = BENCHMARK_DIR / "route1-citation-errata-2026-07-26.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))

    assert artifact["content_sha256"] == core.artifact_content_sha256(artifact)
    assert artifact["historical_metadata_only"] is True
    assert artifact["scientific_results_changed"] is False
    assert artifact["formula_or_parameter_changed"] is False
    assert artifact["invalid_historical_record"]["resolution_status"] == "not-found"
    assert artifact["replacement_record"]["doi"] == "10.1021/ct200786m"
    assert artifact["replacement_record"]["resolution_status"] == "verified"
    assert artifact["cha_gb_record"]["doi"] == "10.1021/ct4010917"
    assert artifact["cha_gb_record"]["resolution_status"] == "verified"
