from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
WORK_DIR = (
    REPOSITORY_ROOT / ".omx/benchmarks/am1bcc-apbs-ace-route1-20260725"
)
FINE_WORK_DIR = (
    REPOSITORY_ROOT
    / ".omx/benchmarks/am1bcc-apbs-ace-fine-route1-20260725"
)
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core
import run_apbs_ace_screen as evaluation


def test_protocol_freezes_route1_boundaries_and_sp_only_pb_endpoint():
    protocol_path = BENCHMARK_DIR / "apbs_ace_protocol.json"
    protocol, fingerprint = evaluation.load_evaluation_protocol(protocol_path)
    manifest = evaluation.load_label_free_manifest(protocol_path, protocol)

    assert len(fingerprint) == 64
    assert protocol["source_partition"] == "development"
    assert protocol["charge"] == {
        "method": "AM1-BCC",
        "lifecycle": (
            "One frozen charge vector per molecule, reused unchanged by APBS "
            "and the source OBC-II/ACE calculation."
        ),
        "geometry_policy": "keep",
    }
    assert protocol["execution_boundary"] == {
        "energy_phase_reads_experimental_labels": False,
        "energy_phase_inputs": [
            "protocol",
            "label-free source manifest",
            "frozen FreeSolv MOL2",
        ],
        "summary_phase_may_read_experimental_labels": True,
        "confirmation_remains_closed": True,
        "no_experimental_fit_or_residual_model": True,
        "no_gas_phase_mm_energy": True,
        "no_mlip_retraining": True,
    }
    assert protocol["polar"]["surface_definition"] == "mol"
    assert protocol["polar"]["calculation"] == {
        "energy": "total",
        "force": "no",
    }
    assert protocol["nonpolar_endpoints"]["openmm_ace_candidate"][
        "double_counting_policy"
    ].startswith("Discard the APBS APOLAR")
    assert protocol["execution_controls"]["maximum_concurrent_apbs_processes"] == 6
    assert protocol["execution_controls"]["subprocess_environment"][
        "OMP_NUM_THREADS"
    ] == "1"
    assert manifest["case_count"] == 526
    assert len(manifest["grid_sensitivity_selection"]["compound_ids"]) == 20
    assert "experimental" not in json.dumps(manifest, sort_keys=True).lower()


def test_manifest_ace_components_are_hash_pinned_and_finite():
    protocol_path = BENCHMARK_DIR / "apbs_ace_protocol.json"
    protocol, _fingerprint = evaluation.load_evaluation_protocol(protocol_path)
    manifest = evaluation.load_label_free_manifest(protocol_path, protocol)

    assert core.sha256_file(BENCHMARK_DIR / "apbs_ace_source_manifest.json") == (
        protocol["source_evidence"]["source_manifest_sha256"]
    )
    assert all(
        isinstance(row["openmm_ace_nonpolar_kcal_mol"], float)
        and row["openmm_ace_nonpolar_kcal_mol"] > 0.0
        for row in manifest["records"]
    )
    assert all(len(row["am1bcc_charges_e"]) > 0 for row in manifest["records"])


def test_fine_grid_protocol_is_a_disclosed_development_followup():
    protocol_path = BENCHMARK_DIR / "apbs_ace_fine_protocol.json"
    protocol, fingerprint = evaluation.load_evaluation_protocol(protocol_path)
    manifest = evaluation.load_label_free_manifest(protocol_path, protocol)

    assert len(fingerprint) == 64
    assert protocol["protocol_id"] == (
        "maple-route1-am1bcc-apbs-mol-ace-fine-v1"
    )
    assert protocol["polar"]["grid"] == {
        "mode": "mg-manual",
        "points_per_axis": 129,
        "spacing_angstrom": 0.25,
        "levels": 4,
        "center": "molecule",
    }
    assert protocol["grid_sensitivity"]["reference_grid"] == {
        "points_per_axis": 161,
        "spacing_angstrom": 0.2,
        "levels": 4,
        "center": "molecule",
    }
    assert protocol["parent_numerical_evidence"]["selection_was_label_free"] is True
    assert protocol["prospective_decision_gates"]["product_role"][
        "runtime_promotion_requires_performance_gate"
    ] is True
    assert manifest["case_count"] == 526


def test_paired_gain_uses_paired_absolute_errors():
    gain = evaluation.paired_absolute_error_gain(
        baseline_errors=[2.0, -1.0, 3.0],
        candidate_errors=[1.0, -0.5, 2.0],
        resamples=1000,
        confidence=0.95,
        seed=9,
    )

    assert gain["mean_mae_gain_kcal_mol"] == pytest.approx(5.0 / 6.0)
    assert gain["case_outcomes"] == {
        "improved": 3,
        "unchanged": 0,
        "worsened": 0,
    }


def test_frozen_energy_records_are_label_free_and_reconcile_components():
    records = sorted((WORK_DIR / "records").glob("*.json"))
    summary = core.load_json(
        BENCHMARK_DIR / "freesolv-am1bcc-apbs-ace-2026-07-25.json"
    )
    assert len(records) == 526

    for path in records:
        record = core.load_json(path)
        assert "experimental" not in json.dumps(record, sort_keys=True).lower()
        assert record["provider_provenance"]["ace"]["gas_phase_mm_energy_used"] is False
        components = record["components_kcal_mol"]
        predictions = record["predictions_kcal_mol"]
        assert predictions["am1bcc_apbs_mol_sasa"] == pytest.approx(
            components["apbs_mol_lpb_polar"]
            + components["apbs_sasa_nonpolar"]
        )
        assert predictions["am1bcc_apbs_mol_ace"] == pytest.approx(
            components["apbs_mol_lpb_polar"]
            + components["openmm_ace_nonpolar"]
        )
        assert core.sha256_file(path) == summary["record_sha256"][
            record["compound_id"]
        ]


def test_frozen_summary_rejects_the_candidate_on_materiality_and_grid_stability():
    protocol, fingerprint = evaluation.load_evaluation_protocol(
        BENCHMARK_DIR / "apbs_ace_protocol.json"
    )
    summary = core.load_json(
        BENCHMARK_DIR / "freesolv-am1bcc-apbs-ace-2026-07-25.json"
    )

    assert summary["protocol_fingerprint"] == fingerprint
    assert summary["case_count"] == 526
    assert summary["success_count"] == 526
    assert summary["failure_count"] == 0
    assert len(summary["record_sha256"]) == 526
    assert len(summary["grid_record_sha256"]) == 20
    assert summary["methods"]["am1bcc_obc2_ace"]["mae"] == pytest.approx(
        1.7603512076636891
    )
    assert summary["methods"]["am1bcc_apbs_mol_sasa"]["mae"] == pytest.approx(
        4.41490913322249
    )
    assert summary["methods"]["am1bcc_apbs_mol_ace"]["mae"] == pytest.approx(
        1.6583080285056173
    )
    assert summary["methods"]["am1bcc_apbs_mol_ace"]["rmse"] == pytest.approx(
        2.535162160920558
    )
    assert summary["paired_absolute_error_gain"]["obc2_ace_to_apbs_mol_ace"][
        "mean_mae_gain_kcal_mol"
    ] == pytest.approx(0.10204317915807191)
    assert summary["grid_sensitivity"][
        "maximum_absolute_difference_kcal_mol"
    ] == pytest.approx(0.6705209766491436)
    assert summary["grid_sensitivity"]["p90_absolute_difference_kcal_mol"] == (
        pytest.approx(0.4546563579808768)
    )
    assert summary["decision"] == {
        "result": "rejected_for_runtime_promotion",
        "checks": {
            "coverage": True,
            "minimum_mae_gain": False,
            "paired_ci_lower_above_zero": True,
            "rmse_not_above_obc2_ace": True,
            "mae_below_apbs_sasa": True,
            "rmse_below_apbs_sasa": True,
            "grid_sensitivity": False,
        },
        "scope": "SP energy only; APBS molecular-surface forces are not exposed.",
        "confirmation_remains_closed": True,
    }
    assert summary["label_use_boundary"] == {
        "energy_phase": "No experimental labels read.",
        "summary_phase": (
            "Experimental FreeSolv values read only after all energy and grid "
            "records existed."
        ),
        "experimental_fit_or_residual_model": False,
        "gas_phase_mm_energy": False,
        "mlip_retraining": False,
    }


def test_fine_grid_summary_still_rejects_the_candidate_on_materiality():
    protocol, fingerprint = evaluation.load_evaluation_protocol(
        BENCHMARK_DIR / "apbs_ace_fine_protocol.json"
    )
    summary = core.load_json(
        BENCHMARK_DIR / "freesolv-am1bcc-apbs-ace-fine-2026-07-25.json"
    )

    assert summary["protocol_fingerprint"] == fingerprint
    assert summary["case_count"] == 526
    assert summary["success_count"] == 526
    assert summary["failure_count"] == 0
    assert len(summary["record_sha256"]) == 526
    assert len(summary["grid_record_sha256"]) == 20
    assert len(list((FINE_WORK_DIR / "records").glob("*.json"))) == 526
    assert summary["methods"]["am1bcc_apbs_mol_ace"]["mae"] == pytest.approx(
        1.6288990168391229
    )
    assert summary["methods"]["am1bcc_apbs_mol_ace"]["rmse"] == pytest.approx(
        2.46221030127189
    )
    assert summary["paired_absolute_error_gain"]["obc2_ace_to_apbs_mol_ace"][
        "mean_mae_gain_kcal_mol"
    ] == pytest.approx(0.13145219082456652)
    assert summary["paired_absolute_error_gain"]["obc2_ace_to_apbs_mol_ace"][
        "bootstrap_ci"
    ] == pytest.approx([0.050131871579939924, 0.21665365264400133])
    assert summary["grid_sensitivity"]["gate"]["passed"] is True
    assert summary["decision"]["checks"]["minimum_mae_gain"] is False
    assert summary["decision"]["checks"]["grid_sensitivity"] is True
    assert summary["decision"]["result"] == "rejected_for_runtime_promotion"
    assert summary["parent_numerical_evidence"][
        "selection_was_label_free"
    ] is True
