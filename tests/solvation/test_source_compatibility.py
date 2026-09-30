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
    COMPATIBILITY_ARTIFACT,
    PREVIOUS_COMPATIBILITY_ARTIFACT,
    PREVIOUS_COMPATIBILITY_SHA256,
    load_source_compatibility,
    require_exact_frozen_sources,
    validate_frozen_source,
)
from generate_source_compatibility import build_receipt

SEPTEMBER_27_RECEIPT = (
    BENCHMARK_DIR / "route1-production-safety-source-compatibility-2026-09-27.json"
)
SEPTEMBER_27_RECEIPT_SHA256 = (
    "4658b7dc8e0d7c1fa510e62e391d04aab9d076b084183b590384055b651fc17d"
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


def test_new_receipt_supersedes_immutable_september_27_receipt():
    assert COMPATIBILITY_ARTIFACT.name == (
        "route1-production-safety-source-compatibility-2026-09-30.json"
    )
    assert PREVIOUS_COMPATIBILITY_ARTIFACT == SEPTEMBER_27_RECEIPT
    assert PREVIOUS_COMPATIBILITY_SHA256 == SEPTEMBER_27_RECEIPT_SHA256
    assert sha256_file(SEPTEMBER_27_RECEIPT) == SEPTEMBER_27_RECEIPT_SHA256

    artifact = load_source_compatibility()
    assert artifact["supersedes"] == {
        "file": SEPTEMBER_27_RECEIPT.name,
        "file_sha256": SEPTEMBER_27_RECEIPT_SHA256,
    }


def test_prior_records_roll_forward_without_losing_metadata_or_strict_denial():
    import json

    previous = json.loads(SEPTEMBER_27_RECEIPT.read_text(encoding="utf-8"))
    current = load_source_compatibility()
    current_by_path = {record["path"]: record for record in current["source_changes"]}

    for old in previous["source_changes"]:
        new = current_by_path[old["path"]]
        for key, value in old.items():
            if key not in {"current_sha256", "historical_sha256"}:
                assert new[key] == value
        expected_history = list(old["historical_sha256"])
        if old["current_sha256"] != new["current_sha256"]:
            expected_history.append(old["current_sha256"])
        assert new["historical_sha256"] == list(dict.fromkeys(expected_history))
        for historical in new["historical_sha256"]:
            with pytest.raises(ValueError, match="historical-audit-only"):
                require_exact_frozen_sources(
                    REPOSITORY_ROOT,
                    [{"path": new["path"], "sha256": historical}],
                )


def test_receipt_is_the_deterministic_current_source_snapshot():
    assert build_receipt() == load_source_compatibility()


@pytest.mark.parametrize(
    "path",
    [
        "maple/function/calculator/extra_correction/implicit/data/__init__.py",
        "maple/function/calculator/extra_correction/implicit/data/obc2_profile_manifest.json",
        "maple/function/calculator/extra_correction/implicit/data/OPENMM_CUSTOMGBFORCES_LICENSE.txt",
    ],
)
def test_packaged_obc2_data_is_provenance_only(path):
    record = next(
        record
        for record in load_source_compatibility()["source_changes"]
        if record["path"] == path
    )
    assert record["historical_sha256"] == []
    assert "provenance" in record["change_class"]
    assert record["historical_numerical_results_recomputed"] is False
    assert record["scientific_claim_reuse_authorized"] is False


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
