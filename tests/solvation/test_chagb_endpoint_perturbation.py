from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from maple.function.free_energy import analyze_one_sided_perturbation

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs" / "implicit-solvation" / "benchmarks"
RUNNER_PATH = BENCHMARK_DIR / "run_chagb_endpoint_perturbation.py"
PROTOCOL_PATH = BENCHMARK_DIR / "chagb_endpoint_perturbation_protocol.json"
ENERGY_PATH = (
    BENCHMARK_DIR
    / "route1-multi-mlip-chagb-endpoint-perturbation-2026-07-25.json"
)
SCORE_PATH = (
    BENCHMARK_DIR
    / "route1-multi-mlip-chagb-endpoint-perturbation-score-2026-07-25.json"
)
DOCUMENTATION_PATHS = (
    REPOSITORY_ROOT / "docs" / "implicit-solvation" / "README.md",
    REPOSITORY_ROOT
    / "docs"
    / "implicit-solvation"
    / "FORMULAS_AND_REFERENCES.md",
    REPOSITORY_ROOT
    / "docs"
    / "implicit-solvation"
    / "ROUTE1_PRODUCT_SPEC.md",
    REPOSITORY_ROOT
    / "docs"
    / "implicit-solvation"
    / "VALIDATION_STATUS.md",
    BENCHMARK_DIR / "README.md",
)

SPEC = importlib.util.spec_from_file_location(
    "run_chagb_endpoint_perturbation",
    RUNNER_PATH,
)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_preserves_the_route1_additive_boundary():
    protocol, fingerprint = RUNNER.load_protocol(PROTOCOL_PATH)
    route = protocol["route_contract"]

    assert len(fingerprint) == 64
    assert route["gas_phase_mm_energy"] is False
    assert route["hydration_label_residual"] is False
    assert route["mlip_retraining"] is False
    assert route["gas_correction_kcal_mol"] == 0.0
    assert route["fixed_am1bcc_charges"] is True
    assert route["low_solution_potential"] == (
        "U_low(R)=E_MLIP,gas(R)+W_OBC2_ACE(R)"
    )
    assert route["high_solution_potential"] == (
        "U_high(R)=E_MLIP,gas(R)+W_CHAGB_PBSA(R)"
    )
    assert protocol["sampling_reuse"]["reference_equilibrium_claim"] is False
    assert protocol["sampling_reuse"]["target_ensemble_sampled"] is False
    assert (
        protocol["sampling_reuse"]["expected_high_endpoint_evaluations_total"]
        == 720
    )
    assert protocol["decision_policy"]["promotion_allowed"] is False
    assert not RUNNER._contains_forbidden_label_key(protocol)


def test_source_trajectory_replay_is_hash_checked_and_has_40_frames():
    protocol, _ = RUNNER.load_protocol(PROTOCOL_PATH)
    sources = RUNNER._load_energy_sources(PROTOCOL_PATH, protocol)
    records = RUNNER._load_source_records(sources["record_manifest"])
    first = records[sorted(records)[0]]["record"]
    window = RUNNER._lambda_one_window(first["chains"][0])

    positions = RUNNER._production_positions(window)

    assert len(positions) == 40
    assert all(frame.shape == positions[0].shape for frame in positions)
    assert all(frame.shape[1] == 3 for frame in positions)


@pytest.mark.skipif(
    importlib.util.find_spec("pymbar") is None,
    reason="optional PyMBAR dependency is not installed",
)
def test_energy_artifact_replays_analysis_and_fails_closed():
    protocol, fingerprint = RUNNER.load_protocol(PROTOCOL_PATH)
    artifact = _load(ENERGY_PATH)

    assert RUNNER.artifact_content_sha256(artifact) == artifact["content_sha256"]
    assert artifact["protocol_fingerprint"] == fingerprint
    assert artifact["command_provenance"]["script_sha256"] == RUNNER.sha256_file(
        RUNNER_PATH
    )
    assert artifact["implementation_provenance"][
        "analysis_module_sha256"
    ] == RUNNER.sha256_file(
        REPOSITORY_ROOT
        / "maple"
        / "function"
        / "free_energy"
        / "perturbation.py"
    )
    assert len(artifact["records"]) == 9
    assert (
        artifact["aggregate_diagnostics"]["high_endpoint_evaluation_count"]
        == 720
    )
    assert artifact["aggregate_diagnostics"]["numerical_gate_pass_count"] == 0
    assert artifact["aggregate_diagnostics"]["scientific_gate_pass_count"] == 0
    assert artifact["promotion_allowed"] is False
    assert not RUNNER._contains_forbidden_label_key(artifact)

    options = dict(protocol["analysis"])
    temperature = options.pop("temperature_kelvin")
    for record in artifact["records"]:
        differences = [
            chain["high_minus_low_kcal_mol"] for chain in record["chains"]
        ]
        replay = analyze_one_sided_perturbation(
            differences,
            temperature_kelvin=temperature,
            reference_equilibrium_claim=False,
            **options,
        )
        assert replay == record["analysis"]
        assert math.isclose(
            record["corrected_high_delta_g_kcal_mol"],
            record["source_low_mbar_delta_g_kcal_mol"]
            + record["endpoint_correction_kcal_mol"],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        assert record["all_numerical_checks_pass"] is False
        assert record["all_scientific_checks_pass"] is False
        assert record["promotion_allowed"] is False


def test_score_is_separate_linked_and_nonpromotable():
    energy = _load(ENERGY_PATH)
    score = _load(SCORE_PATH)

    assert RUNNER.artifact_content_sha256(score) == score["content_sha256"]
    assert score["energy_artifact"]["file_sha256"] == RUNNER.sha256_file(
        ENERGY_PATH
    )
    assert score["energy_artifact"]["content_sha256"] == energy["content_sha256"]
    assert score["label_boundary"] == {
        "energy_artifact_sealed_before_scoring": True,
        "labels_read_only_by_score_phase": True,
        "labels_changed_neither_cases_parameters_nor_energies": True,
        "confirmation_partition": False,
    }
    assert set(score["model_summaries"]) == {
        "aimnet2",
        "ani2x",
        "maceoff23m",
    }
    for summary in score["model_summaries"].values():
        assert summary["case_count"] == 3
        assert summary["corrected_mae_gain_vs_low_mbar_kcal_mol"] > 0.0
        assert summary["corrected_mae_gain_vs_fixed_geometry_kcal_mol"] > 0.0
        assert summary["numerical_gate_pass_count"] == 0
        assert summary["scientific_gate_pass_count"] == 0
        assert summary["promotion_allowed"] is False
    assert score["decision"]["result"] == "development_score_not_promotable"
    assert score["promotion_allowed"] is False


def test_documentation_binds_the_result_to_the_route1_boundary():
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in DOCUMENTATION_PATHS
    )

    assert "route1-multi-mlip-chagb-endpoint-perturbation-2026-07-25.json" in (
        combined
    )
    assert (
        "route1-multi-mlip-chagb-endpoint-perturbation-score-2026-07-25.json"
        in combined
    )
    assert "0/9" in combined
    assert "gas correction is exactly zero" in combined
    assert "speed comparison with bare MM" in combined
