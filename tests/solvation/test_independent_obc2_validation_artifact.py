from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs" / "implicit-solvation" / "benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import validate_independent_obc2 as validation


def test_validation_output_uses_the_current_run_date_without_rewriting_stage0():
    assert validation.RECORDED_DATE == "2026-09-30"
    assert validation.OUTPUT_REL.endswith(
        "route1-independent-obc2-validation-2026-09-30.json"
    )
    assert validation.STAGE0_MAIN_REL.endswith(
        "route1-obc2-am1bcc-526-reference-2026-09-27.json"
    )


def test_sealed_stage0_inputs_are_exact_and_complete():
    main, portable = validation.load_stage0()

    assert main["content_sha256"] == validation.STAGE0_MAIN_CONTENT_SHA256
    assert portable["content_sha256"] == validation.STAGE0_PORTABLE_CONTENT_SHA256
    assert main["record_count"] == 1052
    assert portable["comparison_count"] == 48
    assert main["semantic_decoder_guard"]["runtime_violation_count"] == 0


def test_full_h_selection_was_frozen_before_provider_calculations():
    _main, portable = validation.load_stage0()
    selection = validation.select_full_h_cases(portable)

    assert selection["selected_cases"] == [
        "water",
        "methanol",
        "methyl-hexanoate",
        "nitralin",
    ]
    assert selection["selected_largest_case"] == "nitralin"
    assert selection["maximum_atom_count"] == 42
    assert selection["rule"] == validation.FULL_H_SELECTION_RULE


def test_hvp_directions_are_seeded_normalized_and_replay_exactly():
    first_seed, first = validation.normalized_direction(
        case_index=4,
        geometry_index=1,
        direction_index=2,
        size=69,
    )
    second_seed, second = validation.normalized_direction(
        case_index=4,
        geometry_index=1,
        direction_index=2,
        size=69,
    )

    assert first_seed == second_seed
    np.testing.assert_array_equal(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0, rtol=0.0, atol=1.0e-15)


def test_fd_ladder_uses_approved_d64_steps_and_both_finest_pair_gate():
    analytic = np.zeros((3, 3), dtype=np.float64)
    matrices = {
        0.001: np.full((3, 3), 4.0e-6),
        0.0005: np.full((3, 3), 2.0e-6),
        0.00025: np.full((3, 3), 1.0e-6),
    }
    result = validation.assess_fd_ladder(analytic, matrices)

    assert result["step_order"] == [0.001, 0.0005, 0.00025]
    assert result["all_three_pass_observed"] is True
    assert result["both_finest_pass"] is True
    assert result["finest_pair_pass"] is True
    assert result["passed"] is True


def test_fd_ladder_logs_coarse_without_turning_it_into_an_extra_gate():
    analytic = np.zeros((3, 3), dtype=np.float64)
    matrices = {
        0.001: np.full((3, 3), 3.0e-5),
        0.0005: np.full((3, 3), 2.0e-6),
        0.00025: np.full((3, 3), 1.0e-6),
    }
    result = validation.assess_fd_ladder(analytic, matrices)

    assert result["both_finest_pass"] is True
    assert result["coarse_pass_observed"] is False
    assert result["all_three_pass_observed"] is False
    assert result["passed"] is True


def test_fd_ladder_rejects_when_either_predeclared_finest_step_fails():
    analytic = np.zeros((3, 3), dtype=np.float64)
    matrices = {
        0.001: np.zeros((3, 3)),
        0.0005: np.full((3, 3), 3.0e-5),
        0.00025: np.full((3, 3), 1.0e-6),
    }
    result = validation.assess_fd_ladder(analytic, matrices)

    assert result["both_finest_pass"] is False
    assert result["passed"] is False


def test_fresh_validation_artifact_is_self_consistent_when_sealed():
    path = validation.ROOT / validation.OUTPUT_REL
    if not path.is_file():
        pytest.skip(
            "final validation waits for a stable independent provider source tree"
        )
    artifact = json.loads(path.read_text(encoding="utf-8"))

    validation.validate_artifact(artifact)
    assert artifact["passed"] is True
    assert artifact["replay_526"]["mae_drift_upper_bound_hartree"] == pytest.approx(
        artifact["replay_526"]["mean_absolute_prediction_drift_hartree"],
        rel=0.0,
        abs=0.0,
    )
    assert artifact["direct_hvp_matrix"]["record_count"] == 144
    assert all(
        row["dense_hessian_forbidden_during_call"]
        for row in artifact["direct_hvp_matrix"]["records"]
    )
