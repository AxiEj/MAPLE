import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs" / "implicit-solvation" / "benchmarks"
SPEC = importlib.util.spec_from_file_location(
    "run_mlip_obc2_mbar",
    BENCHMARK_DIR / "run_mlip_obc2_mbar.py",
)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)
SCORE_SPEC = importlib.util.spec_from_file_location(
    "score_mlip_obc2_mbar",
    BENCHMARK_DIR / "score_mlip_obc2_mbar.py",
)
SCORER = importlib.util.module_from_spec(SCORE_SPEC)
assert SCORE_SPEC.loader is not None
SCORE_SPEC.loader.exec_module(SCORER)
PROTOCOL_PATH = BENCHMARK_DIR / "mlip_obc2_mbar_protocol.json"
ARTIFACT_PATH = BENCHMARK_DIR / "route1-multi-mlip-obc2-mbar-2026-07-25.json"
LABEL_SUMMARY_PATH = BENCHMARK_DIR / "route1-multi-mlip-obc2-ti-2026-07-25.json"
SCORE_PATH = BENCHMARK_DIR / "route1-multi-mlip-obc2-mbar-score-2026-07-25.json"


def test_protocol_is_label_free_upstream_mbar_and_nonpromotable():
    protocol, fingerprint = RUNNER.load_protocol(PROTOCOL_PATH)
    manifest = RUNNER.load_source_manifest(PROTOCOL_PATH, protocol)

    assert len(fingerprint) == 64
    assert protocol["dependency"]["name"] == "pymbar"
    assert protocol["dependency"]["handwritten_substitute_allowed"] is False
    assert protocol["route_contract"]["no_gas_phase_mm_energy"] is True
    assert protocol["route_contract"]["no_hydration_residual_model"] is True
    assert protocol["route_contract"]["no_mlip_retraining"] is True
    assert protocol["route_contract"]["experimental_labels_read"] is False
    assert protocol["analysis"]["equilibrium_claim"] is False
    assert protocol["decision_policy"]["promotion_allowed"] is False
    assert manifest["record_count"] == 9
    assert len({(row["model"], row["compound_id"]) for row in manifest["records"]}) == 9
    assert "experimental_kcal_mol" not in json.dumps(manifest).lower()


def test_source_matrix_preserves_two_repeats_and_all_lambda_energies():
    protocol, _ = RUNNER.load_protocol(PROTOCOL_PATH)
    manifest = RUNNER.load_source_manifest(PROTOCOL_PATH, protocol)
    record = RUNNER.load_source_record(manifest["records"][0])
    states = RUNNER.state_replicates(
        record,
        protocol["analysis"]["lambda_values"],
    )

    assert len(states) == 5
    assert all(len(replicates) == 2 for replicates in states)
    assert all(
        matrix.shape == (40, 5) for replicates in states for matrix in replicates
    )
    assert all(
        np.all(np.isfinite(matrix)) for replicates in states for matrix in replicates
    )


def test_frozen_mbar_artifact_is_self_consistent_and_fail_closed():
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    expected_hash = artifact["content_sha256"]

    assert RUNNER.artifact_content_sha256(artifact) == expected_hash
    assert artifact["record_count"] == 9
    assert artifact["decision"]["result"] == "diagnostic_only_not_promotable"
    assert artifact["decision"]["promotion_allowed"] is False
    assert artifact["label_boundary"] == {
        "experimental_labels_read": False,
        "experimental_scoring_performed": False,
    }
    assert artifact["claim_boundary"]["upstream_mbar_path_exercised"] is True
    assert artifact["claim_boundary"]["overlap_matrix_reported"] is True
    assert artifact["claim_boundary"]["equilibrated_sampling_proven"] is False
    assert (
        artifact["claim_boundary"]["product_solvation_free_energy_established"] is False
    )
    assert artifact["implementation_provenance"][
        "analysis_module_sha256"
    ] == RUNNER.sha256_file(
        REPOSITORY_ROOT / "maple" / "function" / "free_energy" / "mbar.py"
    )

    aggregate = artifact["aggregate_diagnostics"]
    assert aggregate["adjacent_overlap_gate_pass_count"] == 9
    assert aggregate["solver_convergence_gate_pass_count"] == 9
    assert aggregate["minimum_effective_sample_gate_pass_count"] == 9
    assert aggregate["mbar_ti_agreement_pass_count"] == 9
    assert aggregate["independent_repeat_agreement_pass_count"] == 9
    assert aggregate["minimum_uncorrelated_sample_gate_pass_count"] == 0
    assert aggregate["diagnostic_pass_count"] == 0
    assert aggregate["minimum_directional_adjacent_overlap"] > 0.03
    assert aggregate["maximum_absolute_mbar_ti_difference_kcal_mol"] < 0.25
    assert aggregate["maximum_independent_repeat_difference_kcal_mol"] < 0.25

    assert all(row["promotion_allowed"] is False for row in artifact["records"])
    assert all(
        row["diagnostic_checks"]["equilibrium_proven"] is False
        for row in artifact["records"]
    )
    assert all(
        row["diagnostic_checks"]["minimum_uncorrelated_samples_per_state"] is False
        for row in artifact["records"]
    )
    assert all(
        row["diagnostic_checks"]["solver_convergence"] is True
        for row in artifact["records"]
    )
    assert all(
        row["mbar"]["estimator"]["handwritten_estimator"] is False
        for row in artifact["records"]
    )
    assert "experimental_kcal_mol" not in json.dumps(artifact).lower()


def test_scoring_is_separate_reproducible_and_negative_for_all_three_mlips():
    frozen = json.loads(SCORE_PATH.read_text(encoding="utf-8"))
    recomputed = SCORER.score(ARTIFACT_PATH, LABEL_SUMMARY_PATH)

    assert recomputed == frozen
    assert SCORER.artifact_content_sha256(frozen) == frozen["content_sha256"]
    assert frozen["decision"]["result"] == "negative_development_diagnostic"
    assert frozen["decision"]["promotion_allowed"] is False
    assert frozen["label_use_boundary"] == {
        "analysis_sealed_before_scoring": True,
        "labels_read_only_by_scoring_phase": True,
        "confirmation_partition": False,
    }
    assert set(frozen["model_summaries"]) == {
        "aimnet2",
        "ani2x",
        "maceoff23m",
    }
    for summary in frozen["model_summaries"].values():
        assert summary["mbar_minus_fixed_geometry_mae_kcal_mol"] > 0.0
        assert summary["promotion_allowed"] is False


@pytest.mark.skipif(
    importlib.util.find_spec("pymbar") is None,
    reason="optional PyMBAR dependency is not installed",
)
def test_runner_recomputes_the_frozen_artifact_with_upstream_pymbar():
    recomputed = RUNNER.run(PROTOCOL_PATH)
    frozen = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    assert recomputed == frozen
