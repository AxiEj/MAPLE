from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
WORK_DIR = (
    REPOSITORY_ROOT
    / ".omx/benchmarks/apbs-mol-grid-followup-route1-20260725"
)
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core
import run_apbs_mol_grid_followup as followup


def test_followup_protocol_is_label_free_and_cannot_reopen_parent_decision():
    path = BENCHMARK_DIR / "apbs_mol_grid_followup_protocol.json"
    protocol, parent, fingerprint = followup.load_followup_protocol(path)
    manifest = followup.load_source_manifest(path, protocol)

    assert len(fingerprint) == 64
    assert parent["protocol_id"] == "maple-route1-am1bcc-apbs-mol-ace-v1"
    assert protocol["execution_boundary"] == {
        "reads_target_properties": False,
        "changes_parent_accuracy_decision": False,
        "no_fit_or_residual_model": True,
        "no_gas_phase_mm_energy": True,
        "force_claim": False,
    }
    assert protocol["apbs"]["parent_grid"] == {
        "points_per_axis": 129,
        "spacing_angstrom": 0.25,
        "levels": 4,
        "center": "molecule",
    }
    assert protocol["apbs"]["followup_grid"] == {
        "points_per_axis": 161,
        "spacing_angstrom": 0.2,
        "levels": 4,
        "center": "molecule",
    }
    assert manifest["case_count"] == 20
    assert "experimental" not in json.dumps(manifest, sort_keys=True).lower()


def test_followup_summary_passes_only_the_fine_grid_numerical_gate():
    protocol, _parent, fingerprint = followup.load_followup_protocol(
        BENCHMARK_DIR / "apbs_mol_grid_followup_protocol.json"
    )
    summary = core.load_json(
        BENCHMARK_DIR / "apbs-mol-grid-followup-2026-07-25.json"
    )

    assert summary["protocol_fingerprint"] == fingerprint
    assert summary["case_count"] == 20
    assert len(summary["record_sha256"]) == 20
    assert summary["mean_absolute_difference_kcal_mol"] == pytest.approx(
        0.0749298936511534
    )
    assert summary["p90_absolute_difference_kcal_mol"] == pytest.approx(
        0.14847398274880616
    )
    assert summary["maximum_absolute_difference_kcal_mol"] == pytest.approx(
        0.17498562571701726
    )
    assert summary["gate"]["checks"] == {"maximum": True, "p90": True}
    assert summary["gate"]["passed"] is True
    assert summary["parent_accuracy_decision_unchanged"] is True
    assert summary["label_use_boundary"] == {
        "target_properties_read": False,
        "fit_or_residual_model": False,
        "gas_phase_mm_energy": False,
    }


def test_followup_records_are_label_free_and_hash_reconciled():
    records = sorted((WORK_DIR / "records").glob("*.json"))
    summary = core.load_json(
        BENCHMARK_DIR / "apbs-mol-grid-followup-2026-07-25.json"
    )

    assert len(records) == 20
    for path in records:
        record = core.load_json(path)
        assert "experimental" not in json.dumps(record, sort_keys=True).lower()
        assert record["status"] == "success"
        assert core.sha256_file(path) == summary["record_sha256"][
            record["compound_id"]
        ]
