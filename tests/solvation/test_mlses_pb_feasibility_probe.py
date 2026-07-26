from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core

ARTIFACT = (
    BENCHMARK_DIR / "route1-mlses-pb-feasibility-probe-2026-07-26.json"
)
RUNNER = BENCHMARK_DIR / "run_mlses_pb_feasibility_probe.py"


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "mlses_probe",
        RUNNER,
    )
    assert specification is not None
    assert specification.loader is not None
    runner = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(runner)
    return runner


def test_mlses_probe_is_self_hashed_and_reproducible():
    artifact = _artifact()
    recorded = artifact.pop("content_sha256")

    assert benchmark_core.artifact_content_sha256(artifact) == recorded
    assert artifact["schema_version"] == 1
    assert artifact["protocol"]["protocol_id"] == (
        "route1-mlses-pb-feasibility-v2"
    )
    assert artifact["protocol"]["amendment_from"] == (
        "route1-mlses-pb-feasibility-v1"
    )
    assert artifact["protocol"]["canonical_inputs"]
    assert artifact["command_provenance"]["script_sha256"] == (
        benchmark_core.sha256_file(RUNNER)
    )


def test_mlses_surface_is_not_a_hydration_residual():
    artifact = _artifact()
    boundary = artifact["route1_boundary"]
    executed = artifact["source_basis"]["executed_surface_model"]
    predecessor = artifact["source_basis"]["mlses_predecessor"]

    assert boundary["candidate_changes_only_pb_dielectric_surface_generation"]
    assert boundary["hydration_labels_read"] is False
    assert boundary["hydration_label_fit_or_residual_model"] is False
    assert boundary["gas_phase_mm_energy_in_reported_potential"] is False
    assert boundary["gas_mlip_retrained"] is False
    assert executed["name"] == "GENIUSES"
    assert executed["runtime_mapping"]["mlses_opt"] == 0
    assert executed["primary_paper"]["doi"] == (
        "10.1021/acs.jpclett.3c02176"
    )
    assert "not hydration free energies" in executed["primary_paper"][
        "training_target"
    ]
    assert predecessor["doi"] == "10.1021/acs.jctc.1c00492"
    assert "not the mlses_opt=0 runtime" in predecessor["relationship"]


def test_runtime_pair_discovery_and_controls_isolate_missing_geniuses_force():
    capability = _artifact()["force_capability"]
    discovery = capability["pair_discovery_records"]
    records = capability["records"]
    classical = [
        record for record in records if record["surface_id"] == "classical-ses"
    ]
    geniuses = [
        record
        for record in records
        if record["surface_id"] == "geniuses-mlses-opt0"
    ]

    accepted = [
        {"eneopt": 1, "frcopt": 1},
        {"eneopt": 2, "frcopt": 2},
        {"eneopt": 2, "frcopt": 3},
        {"eneopt": 2, "frcopt": 4},
        {"eneopt": 3, "frcopt": 2},
    ]
    assert len(discovery) == 20
    assert _artifact()["protocol"]["force_pair_discovery"][
        "expected_runtime_accepted_pairs"
    ] == accepted
    assert capability["runtime_accepted_energy_force_pairs"] == accepted
    assert [
        {"eneopt": record["eneopt"], "frcopt": record["frcopt"]}
        for record in discovery
        if record["returncode"] == 0
        and record["force_file"]["exists"]
        and record["force_file"]["size_bytes"] > 0
    ] == accepted
    assert len(classical) == len(geniuses) == 10
    assert capability["classical_ses_controls_pass"]
    assert all(
        record["returncode"] == 0
        and record["force_file"]["exists"]
        and record["force_file"]["size_bytes"] > 0
        for record in classical
    )
    assert capability[
        "geniuses_all_runtime_accepted_force_pairs_pass"
    ] is False
    assert capability["finite_difference_gate_eligible"] is False
    assert capability["finite_difference_gate_executed"] is False
    assert capability["finite_difference_gate_passed"] is None
    assert all(
        record["returncode"] != 0
        and record["force_file"]["size_bytes"] == 0
        for record in geniuses
    )
    assert {
        (record["eneopt"], record["frcopt"]) for record in classical
    } == {(1, 1), (2, 2), (2, 3), (2, 4), (3, 2)}


def test_local_energy_screen_supplies_no_small_molecule_speed_advantage():
    timing = _artifact()["energy_timing"]

    assert timing["all_ratios_finite"]
    assert timing["geniuses_faster_than_classical_at_every_grid"] is False
    assert len(timing["records"]) == 2
    for record in timing["records"]:
        assert record["repeats_per_surface"] == 3
        assert record["classical_over_geniuses_speed_ratio"] < 1.0
        assert abs(record["geniuses_minus_classical_energy_kcal_mol"]) < 0.1


def test_mlses_probe_rejects_product_integration_without_calling_it_cheating():
    decision = _artifact()["decision"]

    assert decision["status"] == (
        "rejected-no-atom-resolved-force-or-local-small-molecule-speedup"
    )
    assert decision["learned_surface_is_route1_residual_cheating"] is False
    assert decision["atom_resolved_force_output_supported"] is False
    assert decision["force_consistency_established"] is False
    assert decision["local_small_molecule_speed_gate_passed"] is False
    assert decision["eligible_for_further_provider_validation"] is False
    assert decision["optimization"] is False
    assert decision["relaxed_scan"] is False
    assert decision["md"] is False
    assert decision["single_point_provider_added"] is False
    assert decision["full_freesolv_screen_opened"] is False
    assert decision["new_dependency_added"] is False
    assert decision["default_provider_changed"] is False


def test_decision_is_derived_from_force_and_speed_observations():
    runner = _load_runner()

    assert runner._derive_decision(
        force_supported=False,
        faster_at_every_grid=False,
    )["status"] == (
        "rejected-no-atom-resolved-force-or-local-small-molecule-speedup"
    )
    assert runner._derive_decision(
        force_supported=False,
        faster_at_every_grid=True,
    )["status"] == "rejected-no-atom-resolved-force"
    assert runner._derive_decision(
        force_supported=True,
        faster_at_every_grid=False,
    )["status"] == "rejected-no-local-small-molecule-speedup"
    assert runner._derive_decision(
        force_supported=True,
        faster_at_every_grid=True,
    )["status"] == "eligible-for-further-provider-validation"


def test_noncanonical_probe_cannot_overwrite_canonical_artifact(tmp_path):
    runner = _load_runner()

    import pytest

    with pytest.raises(ValueError, match="require an explicit --output"):
        runner._validate_output_boundary(
            canonical_inputs=False,
            output=runner.DEFAULT_OUTPUT.resolve(),
        )
    runner._validate_output_boundary(
        canonical_inputs=False,
        output=(tmp_path / "comparison.json").resolve(),
    )


def test_route1_docs_preserve_the_mlses_decision_boundary():
    documents = [
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md",
        REPOSITORY_ROOT / "docs/implicit-solvation/FORMULAS_AND_REFERENCES.md",
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md",
        BENCHMARK_DIR / "README.md",
    ]
    normalized = " ".join(
        "\n".join(path.read_text(encoding="utf-8") for path in documents).split()
    )

    assert "MLSES PB surface feasibility boundary" in normalized
    assert "10.1021/acs.jctc.1c00492" in normalized
    assert "10.1021/acs.jpclett.3c02176" in normalized
    assert "five runtime-accepted" in normalized
    assert "no atom-resolved MLSES force" in normalized
    assert "no MLSES runtime provider" in normalized
