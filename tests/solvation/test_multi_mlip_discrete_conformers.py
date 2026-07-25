from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from maple.function.free_energy import analyze_discrete_conformer_ensemble

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
PROTOCOL_PATH = BENCHMARK_DIR / "multi_mlip_discrete_conformer_protocol.json"
MANIFEST_PATH = BENCHMARK_DIR / "multi_mlip_discrete_conformer_source_manifest.json"
ENERGY_PATH = BENCHMARK_DIR / "route1-multi-mlip-discrete-conformer-2026-07-25.json"
SCORE_PATH = (
    BENCHMARK_DIR / "route1-multi-mlip-discrete-conformer-score-2026-07-25.json"
)
RUNNER_PATH = BENCHMARK_DIR / "run_multi_mlip_discrete_conformers.py"
REFERENCE_UNION_PATH = BENCHMARK_DIR / "run_mlip_conformer_weighting.py"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core
import run_multi_mlip_discrete_conformers as runner


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _contains_key(value, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(
            _contains_key(item, target) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(item, target) for item in value)
    return False


def test_multi_mlip_protocol_freezes_common_domain_without_labels():
    protocol, fingerprint = runner.load_protocol(PROTOCOL_PATH)

    assert len(fingerprint) == 64
    assert [model["name"] for model in protocol["models"]] == [
        "maceoff23m",
        "aimnet2",
        "ani2x",
    ]
    assert protocol["selection"]["common_atomic_numbers"] == [
        1,
        6,
        7,
        8,
        9,
        16,
        17,
    ]
    assert protocol["selection"]["expected_selected_case_count"] == 19
    assert protocol["selection"]["expected_selected_state_count"] == 1270
    assert protocol["selection"]["expected_exclusions"] == [
        {
            "compound_id": "mobley_1770205",
            "unsupported_atomic_numbers": [15],
            "reason": "ANI2x checkpoint domain excludes phosphorus.",
        }
    ]
    assert protocol["selection"]["labels_used"] is False
    assert protocol["route"]["gas_phase_mm_energy"] is False
    assert protocol["route"]["hydration_label_residual"] is False
    assert protocol["route"]["mlip_retraining"] is False
    assert protocol["solvent_endpoint"]["charge_method"] == "am1bcc"


def test_source_manifest_is_sealed_label_free_and_self_contained():
    protocol, fingerprint = runner.load_protocol(PROTOCOL_PATH)
    manifest = _load(MANIFEST_PATH)

    assert manifest["content_sha256"] == core.artifact_content_sha256(manifest)
    assert manifest["protocol_fingerprint"] == fingerprint
    assert manifest["command_provenance"]["script_sha256"] == core.sha256_file(
        RUNNER_PATH
    )
    assert manifest["source_evidence"][
        "reference_union_script_sha256"
    ] == core.sha256_file(REFERENCE_UNION_PATH)
    assert len(manifest["cases"]) == 19
    assert sum(case["state_count"] for case in manifest["cases"]) == 1270
    assert manifest["selection"]["observed_exclusions"] == [
        {
            "compound_id": "mobley_1770205",
            "unsupported_atomic_numbers": [15],
        }
    ]
    for forbidden in runner.FORBIDDEN_LABEL_KEYS:
        assert not _contains_key(manifest, forbidden)

    for case in manifest["cases"]:
        state_path = REPOSITORY_ROOT / case["state_file"]
        assert core.sha256_file(state_path) == case["state_file_sha256"]
        states = np.load(state_path, allow_pickle=False)
        assert states.dtype == np.float64
        assert list(states.shape) == case["state_shape"]
        assert states.shape[0] == case["state_count"]
        assert len(case["solvent_correction_kcal_mol"]) == case["state_count"]
        assert len(case["historical_mace_energy_hartree"]) == case["state_count"]


def test_energy_artifact_locks_three_real_model_runs_and_failed_repeat_gate():
    energy = _load(ENERGY_PATH)
    manifest = _load(MANIFEST_PATH)

    assert energy["content_sha256"] == core.artifact_content_sha256(energy)
    assert energy["command_provenance"]["script_sha256"] == core.sha256_file(
        RUNNER_PATH
    )
    assert energy["source_manifest_file_sha256"] == core.sha256_file(MANIFEST_PATH)
    assert energy["source_manifest_content_sha256"] == manifest["content_sha256"]
    assert len(energy["records"]) == 57
    assert set(energy["model_summaries"]) == {
        "maceoff23m",
        "aimnet2",
        "ani2x",
    }
    assert all(
        summary["case_count"] == 19 and summary["state_count"] == 1270
        for summary in energy["model_summaries"].values()
    )
    assert {
        name: summary["weight_diagnostic_pass_count"]
        for name, summary in energy["model_summaries"].items()
    } == {
        "maceoff23m": 13,
        "aimnet2": 4,
        "ani2x": 12,
    }
    gates = energy["engineering_gates"]
    assert gates["all_model_case_records_present"] is True
    assert gates["historical_mace_relative_energy_parity_within_limit"] is True
    assert gates["all_repeat_differences_within_limit"] is False
    assert gates["maximum_historical_mace_relative_energy_difference_kcal_mol"] < 1.0e-9
    assert 1.0e-6 < gates["maximum_observed_repeat_difference_kcal_mol"] < 2.0e-4
    assert gates["maximum_observed_repeat_delta_g_difference_kcal_mol"] < 1.0e-5
    assert energy["promotion_allowed"] is False
    for environment in energy["model_environments"].values():
        for path_key, hash_key in (
            ("calculator_source", "calculator_source_sha256"),
            ("set_calculator_source", "set_calculator_source_sha256"),
            ("calculator_base_source", "calculator_base_source_sha256"),
            ("discrete_core_source", "discrete_core_source_sha256"),
        ):
            source_path = REPOSITORY_ROOT / environment[path_key]
            assert source_path.is_file()
            assert len(environment[hash_key]) == 64
            if path_key in {"set_calculator_source", "discrete_core_source"}:
                assert core.sha256_file(source_path) == environment[hash_key]
    # Calculator/base sources intentionally evolved after this historical
    # serial artifact to add the batch API. Their current hashes are locked by
    # the separate batch-parity artifact rather than rewriting this evidence.
    for forbidden in runner.FORBIDDEN_LABEL_KEYS:
        assert not _contains_key(energy, forbidden)


def test_energy_artifact_recomputes_with_public_discrete_core():
    protocol = _load(PROTOCOL_PATH)
    manifest = _load(MANIFEST_PATH)
    energy = _load(ENERGY_PATH)
    cases = {case["compound_id"]: case for case in manifest["cases"]}

    for record in energy["records"]:
        case = cases[record["compound_id"]]
        raw = np.asarray(record["energy_hartree_by_repeat"][0])
        relative = (raw - float(np.min(raw))) * runner.KCAL_PER_HARTREE
        analysis = analyze_discrete_conformer_ensemble(
            gas_energy_kcal_mol=relative,
            solvent_correction_kcal_mol=case["solvent_correction_kcal_mol"],
            temperature_kelvin=protocol["estimator"]["temperature_kelvin"],
            minimum_effective_conformer_count=protocol["estimator"][
                "minimum_effective_conformer_count"
            ],
            maximum_dominant_weight=protocol["estimator"]["maximum_dominant_weight"],
            minimum_distribution_overlap=protocol["estimator"][
                "minimum_distribution_overlap"
            ],
        )

        assert analysis["delta_g_discrete_kcal_mol"] == pytest.approx(
            record["analysis"]["delta_g_discrete_kcal_mol"],
            abs=1.0e-12,
        )
        assert analysis["gas_weights"] == pytest.approx(
            record["analysis"]["gas_weights"],
            abs=1.0e-15,
        )
        assert analysis["solution_weights"] == pytest.approx(
            record["analysis"]["solution_weights"],
            abs=1.0e-15,
        )
        assert analysis["gates"] == record["analysis"]["gates"]


def test_cross_model_result_exposes_weight_disagreement_and_alachlor_outlier():
    energy = _load(ENERGY_PATH)
    sensitivity = energy["model_sensitivity"]
    largest = max(
        sensitivity["cases"],
        key=lambda row: row["model_range_kcal_mol"],
    )

    assert sensitivity["case_range_kcal_mol"]["median"] == pytest.approx(
        0.035971210642677054,
        rel=0.0,
        abs=1.0e-12,
    )
    assert sensitivity["case_range_kcal_mol"]["p90"] == pytest.approx(
        0.11968818444069881,
        rel=0.0,
        abs=1.0e-12,
    )
    assert largest["compound_id"] == "mobley_8124669"
    assert largest["model_range_kcal_mol"] == pytest.approx(
        1.41021224908578,
        rel=0.0,
        abs=1.0e-12,
    )
    aimnet_ani = energy["pairwise_model_comparisons"]["aimnet2__ani2x"]
    assert aimnet_ani["gas_weight_overlap"]["median"] == pytest.approx(
        0.21412706261854236,
        rel=0.0,
        abs=1.0e-12,
    )
    assert aimnet_ani["solution_weight_overlap"]["median"] == pytest.approx(
        0.20132870113547413,
        rel=0.0,
        abs=1.0e-12,
    )


def test_separate_development_score_does_not_support_accuracy_promotion():
    score = _load(SCORE_PATH)
    energy = _load(ENERGY_PATH)

    assert score["content_sha256"] == core.artifact_content_sha256(score)
    assert score["command_provenance"]["script_sha256"] == core.sha256_file(RUNNER_PATH)
    assert score["energy_artifact_file_sha256"] == core.sha256_file(ENERGY_PATH)
    assert score["energy_artifact_content_sha256"] == energy["content_sha256"]
    assert score["case_count"] == 19
    assert score["fixed_geometry"]["mae_kcal_mol"] == pytest.approx(1.9302190744806231)
    assert score["models"]["aimnet2"]["metrics"]["mae_kcal_mol"] == pytest.approx(
        1.9099029720136953,
        rel=0.0,
        abs=1.0e-12,
    )
    assert score["models"]["maceoff23m"]["metrics"]["mae_kcal_mol"] == pytest.approx(
        1.9686755659553825,
        rel=0.0,
        abs=1.0e-12,
    )
    assert score["models"]["ani2x"]["metrics"]["mae_kcal_mol"] == pytest.approx(
        1.993306317848252,
        rel=0.0,
        abs=1.0e-12,
    )
    for model in score["models"].values():
        lower, upper = model["paired_vs_fixed_geometry"]["bootstrap_ci_kcal_mol"]
        assert lower < 0.0 < upper
    assert score["label_boundary"]["energy_artifact_sealed_before_scoring"] is True
    assert score["label_boundary"]["selection_used_labels"] is False
    assert score["label_boundary"]["independent_confirmation"] is False
    assert score["promotion_allowed"] is False


def test_route1_docs_bind_multi_mlip_discrete_evidence_and_failures():
    paths = [
        REPOSITORY_ROOT / "docs/implicit-solvation/FORMULAS_AND_REFERENCES.md",
        REPOSITORY_ROOT / "docs/implicit-solvation/README.md",
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md",
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md",
        BENCHMARK_DIR / "README.md",
    ]
    normalized = " ".join(
        " ".join(path.read_text(encoding="utf-8").split()) for path in paths
    )

    assert "multi_mlip_discrete_conformer_protocol.json" in normalized
    assert "route1-multi-mlip-discrete-conformer-2026-07-25.json" in normalized
    assert "route1-multi-mlip-discrete-conformer-score-2026-07-25.json" in normalized
    assert "1,270 states" in normalized
    assert "13/19" in normalized
    assert "4/19" in normalized
    assert "12/19" in normalized
    assert "1.41020 kcal/mol" in normalized
    assert "1.910" in normalized
    assert "1.969" in normalized
    assert "1.993" in normalized
    assert "Every interval crosses zero" in normalized
    assert "public `#solvfe` task remains closed" in normalized
