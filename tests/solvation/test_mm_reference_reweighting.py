from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[2] / "docs" / "implicit-solvation" / "benchmarks"
)
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
from source_compatibility import validate_frozen_source

RUNNER_PATH = BENCHMARK_DIR / "run_mlip_mm_reference_reweighting.py"
PROTOCOL_PATH = BENCHMARK_DIR / "mlip_mm_reference_reweighting_protocol.json"
ANALYSIS_PATH = BENCHMARK_DIR / "route1-mm-reference-reweighting-2026-07-25.json"
SCORE_PATH = BENCHMARK_DIR / "route1-mm-reference-reweighting-score-2026-07-25.json"


def _runner():
    name = "route1_mm_reference_reweighting_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_reference_reweighting_protocol_preserves_route1_boundary():
    runner = _runner()
    protocol, fingerprint = runner.load_protocol(PROTOCOL_PATH)

    assert len(fingerprint) == 64
    assert protocol["route_contract"]["reference_mm_role"] == "sampling_only"
    assert protocol["route_contract"]["reference_mm_energy_in_target"] is False
    assert protocol["route_contract"]["no_hydration_residual_model"] is True
    assert protocol["route_contract"]["no_mlip_retraining"] is True
    assert (
        protocol["execution_boundary"][
            "sampling_or_energy_phase_reads_experimental_labels"
        ]
        is False
    )
    assert protocol["execution_boundary"]["promotion_allowed"] is False
    assert protocol["decision_policy"]["promotion_allowed"] is False


def test_reference_reweighting_cycle_has_correct_phase_signs():
    runner = _runner()

    assert runner.indirect_cycle(-2.0, -100.2, -100.0) == pytest.approx(-2.2)


def test_reference_reweighting_protocol_rejects_target_mm_energy(tmp_path):
    runner = _runner()
    protocol = runner.load_json(PROTOCOL_PATH)
    protocol["route_contract"]["reference_mm_energy_in_target"] = True
    candidate = tmp_path / "protocol.json"
    runner.write_json_atomic(candidate, protocol)

    with pytest.raises(ValueError, match="target may not contain an MM energy"):
        runner.load_protocol(candidate)


def test_reference_reweighting_artifact_fails_closed():
    runner = _runner()
    analysis = runner.load_json(ANALYSIS_PATH)
    _, fingerprint = runner.load_protocol(PROTOCOL_PATH)

    assert analysis["content_sha256"] == runner.artifact_content_sha256(analysis)
    assert analysis["protocol_fingerprint"] == fingerprint
    assert analysis["record_count"] == 9
    assert analysis["aggregate_diagnostics"]["numerical_pass_count"] == 0
    assert analysis["aggregate_diagnostics"]["solver_convergence_pass_count"] == 7
    assert analysis["decision"]["promotion_allowed"] is False
    assert analysis["label_boundary"]["experimental_labels_read"] is False
    assert analysis["claim_boundary"]["reference_mm_energy_in_target"] is False
    assert (
        analysis["aggregate_diagnostics"]["maximum_provider_parity_difference_kcal_mol"]
        < 0.0001
    )
    assert (
        analysis["aggregate_diagnostics"]["minimum_gas_mbar_directional_overlap"] < 0.03
    )
    assert (
        analysis["aggregate_diagnostics"]["minimum_solution_mbar_directional_overlap"]
        < 0.03
    )
    assert all(
        row["target_energy_only_evaluations"] == 400 for row in analysis["records"]
    )
    for row in analysis["records"]:
        assert "bidirectional_mbar_uncertainty_kcal_mol" not in row
        assert "gas_mbar_directional_overlap" in row
        assert "solution_mbar_directional_overlap" in row
        assert {
            "reference_solvation_uncorrelated_samples",
            "gas_correction_uncorrelated_samples",
            "solution_correction_uncorrelated_samples",
            "direct_endpoint_uncorrelated_samples",
            "reference_solvation_effective_samples",
            "gas_correction_effective_samples",
            "solution_correction_effective_samples",
            "direct_endpoint_effective_samples",
            "reference_solvation_solver_convergence",
            "gas_correction_solver_convergence",
            "solution_correction_solver_convergence",
            "direct_endpoint_solver_convergence",
            "reference_solvation_bar_solver_convergence",
            "gas_correction_bar_solver_convergence",
            "solution_correction_bar_solver_convergence",
            "direct_endpoint_bar_solver_convergence",
            "reference_solvation_mbar_directional_overlap",
            "gas_correction_mbar_directional_overlap",
            "solution_correction_mbar_directional_overlap",
            "direct_endpoint_mbar_directional_overlap",
            "reference_solvation_bar_uncertainty",
            "gas_correction_bar_uncertainty",
            "solution_correction_bar_uncertainty",
            "direct_endpoint_bar_uncertainty",
        } <= row["numerical_checks"].keys()
    validation = validate_frozen_source(
        Path(__file__).resolve().parents[2],
        analysis["implementation_provenance"]["mbar_analysis_module"],
        analysis["implementation_provenance"]["mbar_analysis_module_sha256"],
    )
    assert validation["mode"] == (
        "documented-postexecution-production-safety-change"
    )


def test_reference_reweighting_raw_manifest_is_self_consistent():
    runner = _runner()
    analysis = runner.load_json(ANALYSIS_PATH)
    manifest_path = runner._resolve_record_path(analysis["raw_manifest"]["path"])
    manifest = runner.load_json(manifest_path)

    assert runner.sha256_file(manifest_path) == analysis["raw_manifest"]["file_sha256"]
    assert manifest["content_sha256"] == runner.artifact_content_sha256(manifest)
    rows = manifest["reference_records"] + manifest["model_records"]
    assert len(rows) == 12
    for row in rows:
        path = runner._resolve_record_path(row["relative_path"])
        record = runner.load_json(path)
        assert runner.sha256_file(path) == row["file_sha256"]
        assert record["content_sha256"] == row["content_sha256"]
        assert record["content_sha256"] == runner.artifact_content_sha256(record)
        assert "experimental_kcal_mol" not in path.read_text(encoding="utf-8").lower()
        if record["artifact_type"] == "route1-mmgb-reference-mlip-reweighting-case":
            cycle = record["thermodynamic_cycle"]
            assert "bidirectional_mbar_uncertainty_kcal_mol" not in cycle
            assert "reference_only_accelerated_uncertainty_kcal_mol" not in cycle
            for diagnostics in record["reweighting_diagnostics"].values():
                assert "mbar_solver_diagnostics" in diagnostics
                assert "bar_solver_diagnostics" in diagnostics


def test_reference_reweighting_scoring_is_separate_and_negative():
    runner = _runner()
    score = runner.load_json(SCORE_PATH)

    assert score["decision"]["promotion_allowed"] is False
    assert score["label_use_boundary"]["analysis_sealed_before_scoring"] is True
    assert (
        score["label_use_boundary"][
            "labels_changed_neither_protocol_nor_energy_estimates"
        ]
        is True
    )
    assert all(
        summary["reference_only_accelerated_minus_fixed_geometry_mae_kcal_mol"] > 0.0
        for summary in score["model_summaries"].values()
    )
