from __future__ import annotations

from copy import deepcopy

from tools.route2_release import run_auxiliary_density_basis_ladder as ladder


def _record(*, bound: float = 0.20, condition: float = 1.0e9):
    return {
        "auxiliary_dimension": 100,
        "actual_energy_error_kcal_per_mol": 0.05,
        "energy_error_upper_bound_kcal_per_mol": bound,
        "constrained_moment_max_absolute_error": 1.0e-12,
        "reciprocity_relative_defect": 1.0e-15,
        "metric_condition_number": condition,
        "case_passed": bool(
            bound <= ladder.PER_CASE_ENERGY_BOUND_MAX_KCAL_PER_MOL
            and condition <= ladder.METRIC_CONDITION_NUMBER_MAXIMUM
        ),
    }


def test_candidate_order_is_cost_increasing_and_fixed() -> None:
    assert ladder.CANDIDATE_ORDER == (
        "etb-beta-2p0",
        "autoaux",
        "etb-beta-1p5",
    )


def test_candidate_summary_requires_both_mean_and_tail_bound() -> None:
    passing = ladder._candidate_summary(
        "etb-beta-2p0",
        [_record() for _ in range(12)],
    )
    assert passing["status"] == "pass"
    assert all(passing["gates"].values())

    mean_failure_records = [_record(bound=0.26) for _ in range(12)]
    mean_failure = ladder._candidate_summary(
        "etb-beta-2p0",
        mean_failure_records,
    )
    assert mean_failure["status"] == "fail"
    assert mean_failure["gates"]["mean_bound"] is False
    assert mean_failure["gates"]["maximum_bound"] is True

    tail_records = [_record() for _ in range(12)]
    tail_records[-1] = _record(bound=0.51)
    tail_failure = ladder._candidate_summary("autoaux", tail_records)
    assert tail_failure["status"] == "fail"
    assert tail_failure["gates"]["maximum_bound"] is False


def test_candidate_summary_fails_an_ill_conditioned_basis() -> None:
    records = [_record() for _ in range(12)]
    records[-1] = deepcopy(records[-1])
    records[-1]["metric_condition_number"] = (
        ladder.METRIC_CONDITION_NUMBER_MAXIMUM * 1.01
    )
    records[-1]["case_passed"] = False

    summary = ladder._candidate_summary("autoaux", records)

    assert summary["status"] == "fail"
    assert summary["gates"]["conditioning"] is False
    assert summary["gates"]["all_cases"] is False
