from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs" / "implicit-solvation" / "benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
from source_compatibility import validate_frozen_source

RUNNER = BENCHMARK_DIR / "run_mlip_mm_nonequilibrium_switching.py"
PROTOCOL = BENCHMARK_DIR / "mlip_mm_nonequilibrium_switching_protocol.json"
SCORER = BENCHMARK_DIR / "score_mlip_mm_nonequilibrium_switching.py"
ANALYSIS = BENCHMARK_DIR / "route1-mm-mlip-nonequilibrium-switching-2026-07-25.json"
SCORE = BENCHMARK_DIR / "route1-mm-mlip-nonequilibrium-switching-score-2026-07-25.json"
ENDPOINT_ANALYSIS = BENCHMARK_DIR / "route1-mm-reference-reweighting-2026-07-25.json"
LABEL_SUMMARY = BENCHMARK_DIR / "route1-multi-mlip-obc2-ti-2026-07-25.json"


def _module():
    spec = importlib.util.spec_from_file_location(
        "run_mlip_mm_nonequilibrium_switching",
        RUNNER,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _scorer():
    spec = importlib.util.spec_from_file_location(
        "score_mlip_mm_nonequilibrium_switching",
        SCORER,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_switching_protocol_preserves_route1_and_frozen_cost_contract():
    module = _module()
    protocol, fingerprint = module.load_protocol(PROTOCOL)

    assert len(fingerprint) == 64
    assert protocol["route_contract"]["reference_mm_energy_in_target"] is False
    assert protocol["route_contract"]["no_hydration_residual_model"] is True
    assert protocol["route_contract"]["no_mlip_retraining"] is True
    assert (
        protocol["execution_boundary"][
            "sampling_or_switching_reads_experimental_labels"
        ]
        is False
    )
    assert protocol["execution_boundary"]["promotion_allowed"] is False
    assert (
        protocol["switching"]["expected_target_energy_force_evaluations_per_model_case"]
        == 3264
    )
    assert (
        protocol["switching"]["expected_target_energy_force_evaluations_full_matrix"]
        == 29376
    )
    assert "experimental_kcal_mol" not in json.dumps(protocol).lower()


def test_switching_protocol_rejects_mm_energy_in_target(tmp_path):
    module = _module()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(protocol)
    tampered["route_contract"]["reference_mm_energy_in_target"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="may not contain an MM energy"):
        module.load_protocol(path)


def test_common_alignment_constant_cancels_from_indirect_cycle():
    module = _module()
    reference = -2.0
    aligned_gas = 0.4
    aligned_solution = -0.3
    common_offset = -153000.0

    aligned_cycle = module.aligned_indirect_cycle(
        reference,
        aligned_solution,
        aligned_gas,
    )
    physical_cycle = module.aligned_indirect_cycle(
        reference,
        module.physical_endpoint_correction(aligned_solution, common_offset),
        module.physical_endpoint_correction(aligned_gas, common_offset),
    )
    assert physical_cycle == pytest.approx(aligned_cycle, abs=1.0e-10)


def test_switching_seed_is_deterministic_and_unique_for_frozen_dimensions():
    module = _module()
    seeds = []
    for model_index in range(3):
        for phase_index in range(2):
            for direction_index in range(2):
                for length_index in range(2):
                    for replicate_index in range(2):
                        for frame_index in (5, 31, 57, 83):
                            seeds.append(
                                module.switching_seed(
                                    case_seed=2026072501,
                                    model_index=model_index,
                                    phase_index=phase_index,
                                    direction_index=direction_index,
                                    length_index=length_index,
                                    replicate_index=replicate_index,
                                    frame_index=frame_index,
                                )
                            )

    assert len(seeds) == len(set(seeds))
    assert seeds[0] == module.switching_seed(
        case_seed=2026072501,
        model_index=0,
        phase_index=0,
        direction_index=0,
        length_index=0,
        replicate_index=0,
        frame_index=5,
    )


def test_switching_artifact_is_complete_sealed_and_fail_closed():
    module = _module()
    analysis = module.load_json(ANALYSIS)
    _, fingerprint = module.load_protocol(PROTOCOL)

    assert analysis["content_sha256"] == module.artifact_content_sha256(analysis)
    assert analysis["protocol_fingerprint"] == fingerprint
    assert analysis["record_count"] == 9
    assert analysis["model_count"] == 3
    assert analysis["case_count"] == 3
    assert analysis["matrix_complete"] is True
    assert analysis["aggregate_diagnostics"]["numerical_gate_pass_count"] == 0
    assert (
        analysis["aggregate_diagnostics"]["total_target_energy_force_evaluations"]
        == 29376
    )
    assert analysis["decision"]["promotion_allowed"] is False
    assert analysis["label_boundary"]["experimental_labels_read"] is False
    assert analysis["claim_boundary"]["acceleration_established"] is False
    assert all(
        row["target_energy_force_evaluations"] == 3264 for row in analysis["records"]
    )
    for name, provenance in analysis["implementation_provenance"].items():
        validation = validate_frozen_source(
            ROOT,
            provenance["path"],
            provenance["sha256"],
        )
        assert validation["mode"] in {
            "exact-historical-freeze",
            "documented-postexecution-production-safety-change",
        }, name


def test_switching_raw_manifest_closes_hash_chain():
    module = _module()
    analysis = module.load_json(ANALYSIS)
    manifest_path = module._resolve_path(analysis["raw_manifest"]["path"])
    manifest = module.load_json(manifest_path)

    assert module.sha256_file(manifest_path) == analysis["raw_manifest"]["file_sha256"]
    assert manifest["content_sha256"] == module.artifact_content_sha256(manifest)
    assert len(manifest["records"]) == 9
    for row in manifest["records"]:
        path = module._resolve_path(row["relative_path"])
        record = module.load_json(path)
        assert module.sha256_file(path) == row["file_sha256"]
        assert record["content_sha256"] == row["content_sha256"]
        assert record["content_sha256"] == module.artifact_content_sha256(record)
        assert record["label_boundary"]["experimental_labels_read"] is False
        assert record["route_contract"]["reference_mm_energy_in_target"] is False


def test_switching_scoring_is_separate_and_negative():
    scorer = _scorer()
    score = scorer.load_json(SCORE)

    assert score["content_sha256"] == scorer.artifact_content_sha256(score)
    assert score["decision"]["promotion_allowed"] is False
    assert score["label_use_boundary"]["analysis_sealed_before_scoring"] is True
    assert (
        score["label_use_boundary"][
            "labels_changed_neither_protocol_nor_energy_estimates"
        ]
        is True
    )
    assert len(score["case_results"]) == 9
    metrics = score["overall_metrics"]
    assert metrics["fixed_geometry_obc2_ace"]["mae_kcal_mol"] == pytest.approx(
        2.1503519846347534
    )
    assert metrics["nonequilibrium_switching_20fs"]["mae_kcal_mol"] == (
        pytest.approx(2.3301672605233694)
    )
    assert metrics["direct_target_multistate_mbar"]["mae_kcal_mol"] == (
        pytest.approx(2.2125115549734473)
    )
    assert metrics["endpoint_reweighting_bidirectional"]["mae_kcal_mol"] == (
        pytest.approx(2.6652612195411023)
    )
    assert all(
        summary["nonequilibrium_minus_fixed_geometry_mae_kcal_mol"] > 0.0
        and summary["nonequilibrium_minus_endpoint_reweighting_mae_kcal_mol"] < 0.0
        and summary["switching_numerical_gate_pass_count"] == 0
        for summary in score["model_summaries"].values()
    )


def test_switching_scorer_rejects_an_incomplete_matrix(tmp_path):
    scorer = _scorer()
    analysis = scorer.load_json(ANALYSIS)
    incomplete = copy.deepcopy(analysis)
    incomplete["matrix_complete"] = False
    incomplete = scorer.seal_artifact(incomplete)
    path = tmp_path / "incomplete.json"
    scorer.write_json_atomic(path, incomplete)

    with pytest.raises(ValueError, match="matrix is incomplete"):
        scorer.score(path, ENDPOINT_ANALYSIS, LABEL_SUMMARY)
