from __future__ import annotations

import hashlib
import json
from pathlib import Path


BASELINE_SHA = "15777aadf92e8a14419e5ff5d8ac6b3cc17aa482"
EVIDENCE = (
    Path(__file__).parents[2]
    / "docs"
    / "route2"
    / "evidence"
    / "baseline-15777aad"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase0_manifest_binds_exact_audited_baseline_and_real_results():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())

    assert manifest["schema_version"] == "route2-phase0-baseline-evidence-v1"
    assert manifest["audited_baseline"]["git_head"] == BASELINE_SHA
    assert manifest["audited_baseline"]["worktree_clean_before_capture"] is True
    assert manifest["test_scope"] == {
        "chemical_accuracy_evidence": False,
        "collected": 1445,
        "passed": 1430,
        "production_pes_evidence": False,
        "skipped": 15,
        "skipped_by_missing_runtime": {"moist": 5, "pyddx": 4, "pyscf": 6},
    }
    commands = {entry["command"]: entry for entry in manifest["commands"]}
    assert commands["python -m pytest -q -rs"]["exit_code"] == 0
    assert commands["python -m pip check"]["exit_code"] == 1


def test_phase0_manifest_hashes_every_bound_raw_file():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())

    for filename, record in manifest["files"].items():
        path = EVIDENCE / filename
        assert path.stat().st_size == record["bytes"]
        assert _sha256(path) == record["sha256"]


def test_legacy_capabilities_are_not_promoted_into_vnext():
    matrix = json.loads((EVIDENCE / "capability-matrix.json").read_text())

    assert matrix["baseline_git_head"] == BASELINE_SHA
    assert matrix["summary"]["profile_count"] == 21
    assert matrix["summary"]["vnext_admitted_profile_count"] == 0
    for profile in matrix["profiles"]:
        assert profile["strict_common_variational"] is False
        assert not any(
            profile[f"vnext_tier_{tier}"] for tier in ("E", "F", "H", "V", "M")
        )


def test_inventory_is_source_bound_and_nonempty():
    inventory = json.loads((EVIDENCE / "inventory.json").read_text())

    assert inventory["baseline_git_head"] == BASELINE_SHA
    assert inventory["counts"]["production_modules"] >= 100
    assert inventory["counts"]["tests"] >= 150
    for group in ("production_modules", "tests", "evidence_artifacts"):
        for record in inventory[group]:
            assert len(record["sha256"]) == 64
            assert record["bytes"] > 0
