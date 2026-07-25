from __future__ import annotations

import numpy as np
import pytest

from maple.function.dispatcher.solvfe.analysis import (
    ReducedPotentialTable,
    diagnose_independent_replicas,
    diagnose_mbar,
    estimate_packing_probability,
    estimate_adjacent_bar,
    estimate_mbar,
    subsample_state_series,
)


def _constant_offset_table():
    samples_per_state = 100
    coordinates = np.linspace(-2.0, 2.0, 2 * samples_per_state)
    base = 0.5 * coordinates**2
    reduced = np.vstack([base, base + 2.0])
    return ReducedPotentialTable.create(
        u_kn=reduced,
        N_k=(samples_per_state, samples_per_state),
        row_labels=("state-0", "state-1"),
        frame_ids=tuple(f"frame-{index}" for index in range(len(coordinates))),
        beta=1.0,
        measure_id="toy-canonical",
        boundary_conditions="nonperiodic",
    )


def test_reduced_potential_table_is_dimensionless_hashed_and_immutable():
    table = _constant_offset_table()

    assert table.u_kn.shape == (2, 200)
    assert table.u_kn.flags.writeable is False
    assert len(table.state_hashes) == 2
    assert table.state_hash
    with pytest.raises(ValueError):
        table.u_kn[0, 0] = 99.0

    with pytest.raises(ValueError, match="dimensionless"):
        ReducedPotentialTable.create(
            u_kn=np.ones((2, 2)),
            N_k=(1, 1),
            row_labels=("a", "b"),
            frame_ids=("0", "1"),
            beta=1.0,
            measure_id="toy",
            boundary_conditions="nonperiodic",
            units="eV",
        )


def test_pymbar_and_adjacent_bar_recover_constant_free_energy_offset():
    table = _constant_offset_table()
    mbar = estimate_mbar(table)
    bar = estimate_adjacent_bar(table)

    assert mbar["delta_f"][0, 1] == pytest.approx(2.0, abs=1.0e-10)
    assert mbar["overlap_matrix"].shape == (2, 2)
    assert mbar["overlap_scalar"] > 0.99
    assert bar[0]["delta_f"] == pytest.approx(2.0, abs=1.0e-10)
    assert bar[0]["from"] == "state-0"
    assert bar[0]["to"] == "state-1"


def test_timeseries_subsampling_returns_ordered_state_local_indices():
    rng = np.random.default_rng(20260725)
    values = np.zeros(500)
    for index in range(1, len(values)):
        values[index] = 0.95 * values[index - 1] + rng.normal()

    result = subsample_state_series(values)

    assert result["equilibrated_start"] >= 0
    assert result["statistical_inefficiency"] >= 1.0
    assert result["effective_sample_count"] == len(result["indices"])
    assert result["indices"] == sorted(set(result["indices"]))
    assert result["indices"][-1] < len(values)


def test_mbar_diagnostics_fail_closed_when_effective_samples_are_below_gate():
    table = _constant_offset_table()
    diagnostics = diagnose_mbar(
        table,
        overlap_min=0.03,
        effective_samples_min=500.0,
        bar_disagreement_kcal_max=0.2,
        kcal_per_dimensionless=0.592,
    )

    assert diagnostics["minimum_adjacent_overlap"] > 0.03
    assert diagnostics["status"] == "failed"
    assert diagnostics["failure_codes"] == ["MBAR_ESS_TOO_LOW"]


def test_independent_replicas_pass_before_inverse_variance_pooling():
    diagnostics = diagnose_independent_replicas(
        estimates_kcal_mol=(1.0, 1.1, 0.9),
        standard_errors_kcal_mol=(0.1, 0.1, 0.1),
        replica_ids=("replica-a", "replica-b", "replica-c"),
        seeds=(11, 22, 33),
        per_replica_status=("passed", "passed", "passed"),
        minimum_replicas=3,
        pairwise_z_max=2.0,
    )

    assert diagnostics["status"] == "passed"
    assert diagnostics["failure_codes"] == []
    assert diagnostics["pooled_estimate_kcal_mol"] == pytest.approx(1.0)
    assert diagnostics["maximum_pairwise_z"] < 2.0
    assert diagnostics["diagnostic_input_sha256"]


@pytest.mark.parametrize(
    ("estimates", "ids", "seeds", "statuses", "failure_code"),
    [
        (
            (1.0, 1.1),
            ("a", "b"),
            (11, 22),
            ("passed", "passed"),
            "REPLICA_COUNT_TOO_LOW",
        ),
        (
            (1.0, 1.1, 0.9),
            ("a", "b", "c"),
            (11, 11, 33),
            ("passed", "passed", "passed"),
            "REPLICA_NOT_INDEPENDENT",
        ),
        (
            (1.0, 1.1, 0.9),
            ("a", "b", "c"),
            (11, 22, 33),
            ("passed", "failed", "passed"),
            "REPLICA_INTERNAL_GATE_FAILED",
        ),
        (
            (1.0, 1.5, 0.9),
            ("a", "b", "c"),
            (11, 22, 33),
            ("passed", "passed", "passed"),
            "REPLICA_DISAGREEMENT",
        ),
    ],
)
def test_independent_replicas_fail_closed(
    estimates,
    ids,
    seeds,
    statuses,
    failure_code,
):
    diagnostics = diagnose_independent_replicas(
        estimates_kcal_mol=estimates,
        standard_errors_kcal_mol=(0.1,) * len(estimates),
        replica_ids=ids,
        seeds=seeds,
        per_replica_status=statuses,
        minimum_replicas=3,
        pairwise_z_max=2.0,
    )

    assert diagnostics["status"] == "failed"
    assert failure_code in diagnostics["failure_codes"]
    assert diagnostics["pooled_estimate_kcal_mol"] is None
    assert diagnostics["pooled_standard_error_kcal_mol"] is None


def test_staged_bias_mbar_estimates_unbiased_empty_volume_probability():
    sample_count = 200
    table = ReducedPotentialTable.create(
        u_kn=np.zeros((2, sample_count)),
        N_k=(100, 100),
        row_labels=("bias-0", "bias-max"),
        frame_ids=tuple(f"packing-{index}" for index in range(sample_count)),
        beta=1.0,
        measure_id="pure-water-ghost-cavity-v1",
        boundary_conditions="periodic",
    )
    indicator = np.tile(np.array([True, False]), sample_count // 2)

    result = estimate_packing_probability(
        table,
        empty_indicator=indicator,
        kcal_per_dimensionless=0.592,
        minimum_final_state_empty_samples=50,
        packing_se_kcal_max=0.2,
    )

    assert result["status"] == "passed"
    assert result["p0"] == pytest.approx(0.5)
    assert result["packing_free_energy_kcal_mol"] == pytest.approx(
        -0.592 * np.log(0.5)
    )
    assert result["final_state_empty_sample_count"] == 50


def test_packing_probability_fails_when_max_bias_has_no_empty_support():
    table = ReducedPotentialTable.create(
        u_kn=np.zeros((2, 20)),
        N_k=(10, 10),
        row_labels=("bias-0", "bias-max"),
        frame_ids=tuple(f"packing-{index}" for index in range(20)),
        beta=1.0,
        measure_id="pure-water-ghost-cavity-v1",
        boundary_conditions="periodic",
    )
    indicator = np.array([True, False] * 5 + [False] * 10)

    result = estimate_packing_probability(
        table,
        empty_indicator=indicator,
        kcal_per_dimensionless=0.592,
        minimum_final_state_empty_samples=1,
        packing_se_kcal_max=10.0,
    )

    assert result["status"] == "failed"
    assert "PACKING_EMPTY_SUPPORT_TOO_LOW" in result["failure_codes"]
