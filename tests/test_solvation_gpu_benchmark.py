from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from maple.function.benchmarking.solvation_gpu import (
    FROZEN_PANEL_SHA256,
    collect_numerical_panel,
    load_frozen_panel,
)

PANEL = Path(".omx/validation/gpu-fp32-20261025/frozen-panel-v2.json")
PANEL_SHA256 = "fbfeb576eb37c7444190c497a9b0c459387d1a7ffd7395424779ea854713ec70"


def test_module_binds_the_preregistered_panel_identity():
    assert FROZEN_PANEL_SHA256 == PANEL_SHA256


@pytest.fixture(scope="module")
def frozen_panel():
    return load_frozen_panel(PANEL, PANEL_SHA256)


def _pairwise_model(numbers, positions, solvent, need_forces):
    del numbers
    centered = positions - positions.mean(axis=0, keepdims=True)
    energy = 0.5 * float(np.square(centered).sum())
    if solvent == "vacuum":
        offset = 0.0
    else:
        offset = {"water": 0.1, "ethanol": 0.2, "toluene": 0.3}[solvent]
    result = {"energy_ev": energy + offset}
    if need_forces:
        result["forces_ev_per_angstrom"] = -centered
    return result


def test_collects_exact_scalar_and_vacuum_denominators(frozen_panel):
    anysolv = collect_numerical_panel(_pairwise_model, frozen_panel, "anysolv")
    consolv = collect_numerical_panel(_pairwise_model, frozen_panel, "consolv")

    assert anysolv["counts"]["scalar"] == 18
    assert consolv["counts"]["scalar"] == 12
    assert anysolv["counts"]["vacuum"] + consolv["counts"]["vacuum"] == 12


def test_collects_all_frozen_central_difference_components(frozen_panel):
    result = collect_numerical_panel(_pairwise_model, frozen_panel, "consolv")

    records = result["finite_difference_records"]
    assert len(records) == 234
    assert {record["status"] for record in records} == {"success"}
    assert max(record["absolute_error_ev_per_angstrom"] for record in records) < 1.0e-10


def test_motion_comparisons_apply_force_covariance(frozen_panel):
    result = collect_numerical_panel(_pairwise_model, frozen_panel, "consolv")

    records = result["motion_records"]
    assert len(records) == 18
    assert max(record["absolute_energy_error_ev"] for record in records) < 1.0e-12
    assert (
        max(
            record["maximum_force_covariance_error_ev_per_angstrom"]
            for record in records
        )
        < 1.0e-12
    )


def test_force_replays_retain_every_step_position(frozen_panel):
    result = collect_numerical_panel(_pairwise_model, frozen_panel, "consolv")

    replays = result["sampled_force_records"]
    assert len(replays) == 2
    assert sum(len(record["steps"]) for record in replays) == 20
    assert (
        sum(
            np.asarray(step["next_positions_angstrom"]).size
            for record in replays
            for step in record["steps"]
        )
        == 360
    )


def test_context_matrix_reuses_exact_solution_denominator_without_task_admission(
    frozen_panel,
):
    result = collect_numerical_panel(
        _pairwise_model, frozen_panel, "consolv", inference_mode="optimized"
    )

    rows = result["internal_context_coverage"]
    assert [row["task"] for row in rows] == ["sp", "opt", "scan", "ts", "irc", "md"]
    assert {row["inference_mode"] for row in rows} == {"optimized"}
    assert {row["expected_energy_pairs"] for row in rows} == {12}
    assert {row["expected_force_components"] for row in rows} == {234}
    assert all("public tasks unchanged" in row["scope"] for row in rows)


def test_timing_executes_and_separates_changed_from_identical_inputs(frozen_panel):
    calls = []
    synchronization_count = 0

    def evaluate(numbers, positions, solvent, need_forces):
        calls.append(np.asarray(positions).copy())
        return _pairwise_model(numbers, positions, solvent, need_forces)

    def synchronize():
        nonlocal synchronization_count
        synchronization_count += 1

    result = collect_numerical_panel(
        evaluate,
        frozen_panel,
        "anysolv",
        synchronize=synchronize,
        include_timing=True,
    )

    timing = result["timing"]
    assert timing["fresh_inference"]["count"] == 10
    assert len(set(timing["fresh_inference"]["geometry_sha256s"])) == 10
    assert timing["exact_input_repetition"]["count"] == 10
    assert timing["exact_input_repetition"]["classification"] == (
        "identical-input evaluation; cache hit not assumed; not inference acceleration"
    )
    assert synchronization_count == 43
    assert (
        len(calls) == 47
    )  # 24 panel evaluations + 3 warmups + 10 fresh + 10 identical
    assert not np.array_equal(calls[-20], calls[-19])
    assert all(np.array_equal(calls[-10], positions) for positions in calls[-10:])
    assert np.array_equal(calls[-10], calls[-1])


def test_evaluation_failures_remain_in_named_ledger(frozen_panel):
    def failing_model(numbers, positions, solvent, need_forces):
        if solvent == "water" and len(numbers) == 3:
            raise RuntimeError("frozen injected failure")
        return _pairwise_model(numbers, positions, solvent, need_forces)

    result = collect_numerical_panel(failing_model, frozen_panel, "anysolv")

    assert result["counts"]["scalar"] == 18
    assert result["counts"]["failed_records"] == 1
    failed = [
        record for record in result["scalar_records"] if record["status"] == "failed"
    ]
    assert failed == [
        {
            "case_id": "anysolv:H2O:water",
            "status": "failed",
            "error_type": "RuntimeError",
            "error": "frozen injected failure",
        }
    ]


def test_loader_rejects_wrong_panel_identity():
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_frozen_panel(PANEL, "0" * 64)


def test_collector_rejects_mutated_panel(frozen_panel):
    mutated = {**frozen_panel, "rotation_angle_radians": 0.0}

    with pytest.raises(ValueError, match="exact preregistered frozen panel"):
        collect_numerical_panel(_pairwise_model, mutated, "consolv")
