from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs" / "implicit-solvation" / "benchmarks"
RUNNER_PATH = BENCHMARK_DIR / "run_multi_mlip_chagb_discrete_conformers.py"
PROTOCOL_PATH = BENCHMARK_DIR / "multi_mlip_chagb_discrete_conformer_protocol.json"
ENERGY_PATH = (
    BENCHMARK_DIR / "route1-multi-mlip-chagb-discrete-conformer-2026-07-25.json"
)
SCORE_PATH = (
    BENCHMARK_DIR / "route1-multi-mlip-chagb-discrete-conformer-score-2026-07-25.json"
)

SPEC = importlib.util.spec_from_file_location(
    "run_multi_mlip_chagb_discrete_conformers",
    RUNNER_PATH,
)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_is_label_free_and_preserves_route1():
    protocol, fingerprint = RUNNER.load_protocol(PROTOCOL_PATH)
    route = protocol["route_contract"]

    assert len(fingerprint) == 64
    assert route["gas_phase_mm_energy"] is False
    assert route["hydration_label_residual"] is False
    assert route["mlip_retraining"] is False
    assert route["fixed_am1bcc_charges"] is True
    assert route["gas_mlip_energies_reused_without_modification"] is True
    assert protocol["matrix"] == {
        "model_count": 3,
        "case_count": 19,
        "model_case_record_count": 57,
        "unique_state_count": 1270,
        "high_endpoint_repeat_count": 2,
        "expected_high_endpoint_evaluation_count": 2540,
        "models": ["maceoff23m", "aimnet2", "ani2x"],
    }
    assert protocol["execution_boundary"]["promotion_allowed"] is False
    assert not RUNNER._contains_forbidden_label_key(protocol)


def test_frozen_sources_cover_the_complete_common_matrix():
    protocol, _ = RUNNER.load_protocol(PROTOCOL_PATH)
    sources = RUNNER._load_energy_sources(PROTOCOL_PATH, protocol)
    manifest = sources["manifest"]
    energy = sources["source_energy"]

    assert len(manifest["cases"]) == 19
    assert sum(case["state_count"] for case in manifest["cases"]) == 1270
    assert len(energy["records"]) == 57
    assert {
        (record["model"], record["compound_id"]) for record in energy["records"]
    } == {
        (model, case["compound_id"])
        for model in protocol["matrix"]["models"]
        for case in manifest["cases"]
    }


def test_reused_gas_energies_reproduce_the_source_low_analysis():
    protocol, _ = RUNNER.load_protocol(PROTOCOL_PATH)
    sources = RUNNER._load_energy_sources(PROTOCOL_PATH, protocol)
    case = next(
        case for case in sources["manifest"]["cases"] if case["state_count"] > 1
    )
    source = next(
        record
        for record in sources["source_energy"]["records"]
        if record["model"] == "maceoff23m"
        and record["compound_id"] == case["compound_id"]
    )

    replay = RUNNER._analysis(
        gas_energy_hartree=source["energy_hartree_by_repeat"][0],
        solvent_kcal_mol=case["solvent_correction_kcal_mol"],
        protocol=protocol,
    )

    assert replay["delta_g_discrete_kcal_mol"] == pytest.approx(
        source["analysis"]["delta_g_discrete_kcal_mol"],
        abs=1.0e-12,
    )
    assert replay["gas_weights"] == pytest.approx(
        source["analysis"]["gas_weights"],
        abs=1.0e-12,
    )
    assert replay["solution_weights"] == pytest.approx(
        source["analysis"]["solution_weights"],
        abs=1.0e-12,
    )


def test_generated_artifacts_are_complete_self_hashed_and_fail_closed():
    protocol, fingerprint = RUNNER.load_protocol(PROTOCOL_PATH)
    energy = _load(ENERGY_PATH)
    score = _load(SCORE_PATH)

    assert RUNNER.artifact_content_sha256(energy) == energy["content_sha256"]
    assert RUNNER.artifact_content_sha256(score) == score["content_sha256"]
    assert energy["protocol_fingerprint"] == fingerprint
    assert score["protocol_fingerprint"] == fingerprint
    assert energy["command_provenance"]["script_sha256"] == RUNNER.sha256_file(
        RUNNER_PATH
    )
    assert score["command_provenance"]["script_sha256"] == RUNNER.sha256_file(
        RUNNER_PATH
    )
    assert (
        energy["aggregate_diagnostics"]["high_endpoint_evaluation_count"]
        == protocol["matrix"]["expected_high_endpoint_evaluation_count"]
    )
    assert energy["aggregate_diagnostics"]["case_count"] == 19
    assert energy["aggregate_diagnostics"]["model_record_count"] == 57
    assert energy["aggregate_diagnostics"]["unique_state_count"] == 1270
    assert (
        energy["aggregate_diagnostics"]["maximum_endpoint_repeat_difference_kcal_mol"]
        == 0.0
    )
    assert (
        energy["aggregate_diagnostics"]["maximum_delta_g_repeat_difference_kcal_mol"]
        == 0.0
    )
    assert all(
        value
        for key, value in energy["engineering_gates"].items()
        if key
        not in {
            "endpoint_repeat_limit_kcal_mol",
            "delta_g_repeat_limit_kcal_mol",
        }
    )
    assert not RUNNER._contains_forbidden_label_key(energy)
    assert energy["promotion_allowed"] is False
    assert score["promotion_allowed"] is False
    assert score["energy_artifact"]["file_sha256"] == RUNNER.sha256_file(ENERGY_PATH)
    assert score["energy_artifact"]["content_sha256"] == energy["content_sha256"]
    assert score["decision"]["result"] == ("long_sampling_candidate_not_supported")
    assert score["long_sampling_signal_passed_for_all_models"] is False


def test_all_high_endpoint_analyses_replay_from_frozen_arrays():
    protocol, _ = RUNNER.load_protocol(PROTOCOL_PATH)
    sources = RUNNER._load_energy_sources(PROTOCOL_PATH, protocol)
    energy = _load(ENERGY_PATH)
    source_by_id = {
        (record["model"], record["compound_id"]): record
        for record in sources["source_energy"]["records"]
    }
    endpoint_by_id = {
        record["compound_id"]: record for record in energy["case_endpoints"]
    }

    for record in energy["model_records"]:
        identity = (record["model"], record["compound_id"])
        source = source_by_id[identity]
        endpoint = endpoint_by_id[record["compound_id"]]
        replay = RUNNER._analysis(
            gas_energy_hartree=source["energy_hartree_by_repeat"][0],
            solvent_kcal_mol=endpoint["high_solvent_kcal_mol_by_repeat"][0],
            protocol=protocol,
        )

        assert RUNNER._compact_analysis(replay) == record["analysis"]
        assert record["high_discrete_kcal_mol_by_repeat"] == [
            record["analysis"]["delta_g_discrete_kcal_mol"],
            record["analysis"]["delta_g_discrete_kcal_mol"],
        ]


def test_scoring_rejects_long_sampling_signal_for_every_model():
    score = _load(SCORE_PATH)
    expected_mae = {
        "aimnet2": 1.698949709831669,
        "ani2x": 1.720109513137733,
        "maceoff23m": 1.7093455634514296,
    }

    for model, summary in score["model_summaries"].items():
        assert summary["methods"]["discrete_chagb_pbsa"][
            "mae_kcal_mol"
        ] == pytest.approx(expected_mae[model], abs=1.0e-12)
        assert summary["long_sampling_signal_passed"] is False
        checks = summary["long_sampling_signal_checks"]
        assert checks["material_gain_vs_discrete_obc2"] is True
        assert checks["material_gain_vs_fixed_chagb"] is False
        assert checks["paired_interval_vs_fixed_excludes_zero"] is False
        assert checks["majority_of_cases_improve_vs_fixed"] is False
        assert checks["all_high_weight_diagnostics_pass"] is False
        assert summary["promotion_allowed"] is False


def test_documentation_binds_the_negative_long_sampling_decision():
    documentation = [
        REPOSITORY_ROOT / "docs" / "implicit-solvation" / name
        for name in (
            "ROUTE1_PRODUCT_SPEC.md",
            "FORMULAS_AND_REFERENCES.md",
            "README.md",
            "VALIDATION_STATUS.md",
            "benchmarks/README.md",
        )
    ]

    for path in documentation:
        text = path.read_text(encoding="utf-8")
        assert ENERGY_PATH.name in text
        assert SCORE_PATH.name in text
        assert "long_sampling_candidate_not_supported" in text


def test_documentation_uses_current_high_endpoint_timings():
    energy = _load(ENERGY_PATH)
    diagnostics = energy["aggregate_diagnostics"]
    total = f"`{diagnostics['total_wall_seconds']:.2f} s`"
    documents = [
        REPOSITORY_ROOT / "docs" / "implicit-solvation" / name
        for name in (
            "ROUTE1_PRODUCT_SPEC.md",
            "README.md",
            "VALIDATION_STATUS.md",
            "benchmarks/README.md",
        )
    ]

    for path in documents:
        assert total in path.read_text(encoding="utf-8")

    detailed = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md",
            REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/README.md",
        )
    )
    for seconds in diagnostics["high_endpoint_seconds_by_repeat"]:
        assert f"{seconds:.2f}" in detailed
