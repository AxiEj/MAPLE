from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from maple.solvation.release.residual_force import (
    ADJOINT_REFINEMENT_ABSOLUTE_TOLERANCES,
    ADJOINT_REFINEMENT_RELATIVE_TOLERANCES,
    PRIMAL_REFINEMENT_TOLERANCES,
    RESIDUAL_FORCE_CONTRACT_VERSION,
    RESIDUAL_FORCE_ERROR_BUDGET_EV_PER_A,
    summarize_residual_force_panel,
    summarize_residual_force_refinement,
)


def _levels(*, scale: float = 1.0):
    reference = np.asarray([[0.2, -0.3, 0.1], [-0.4, 0.5, -0.2]])
    primal_offsets = (4.0e-7, 1.0e-7, 2.0e-8)
    adjoint_offsets = (3.0e-7, 8.0e-8, 0.0)
    primal = [
        {
            "primal_tolerance": tolerance,
            "adjoint_relative_tolerance": (
                ADJOINT_REFINEMENT_RELATIVE_TOLERANCES[-1]
            ),
            "adjoint_absolute_tolerance": (
                ADJOINT_REFINEMENT_ABSOLUTE_TOLERANCES[-1]
            ),
            "actual_primal_residual": tolerance * 0.8,
            "actual_adjoint_residual": 1.0e-15,
            "forces_eV_per_A": reference + scale * offset,
        }
        for tolerance, offset in zip(PRIMAL_REFINEMENT_TOLERANCES, primal_offsets)
    ]
    # The tightest adjoint/release-primal force is exactly the first primal
    # level.  This is the required bridge between the two independent series.
    adjoint_offsets = (*adjoint_offsets[:-1], primal_offsets[0])
    adjoint = [
        {
            "primal_tolerance": PRIMAL_REFINEMENT_TOLERANCES[0],
            "adjoint_relative_tolerance": relative,
            "adjoint_absolute_tolerance": absolute,
            "actual_primal_residual": 8.0e-13,
            "actual_adjoint_residual": residual,
            "forces_eV_per_A": reference + scale * offset,
        }
        for relative, absolute, residual, offset in zip(
            ADJOINT_REFINEMENT_RELATIVE_TOLERANCES,
            ADJOINT_REFINEMENT_ABSOLUTE_TOLERANCES,
            (8.0e-12, 8.0e-13, 8.0e-14),
            adjoint_offsets,
        )
    ]
    # Make the adjoint series contract towards the bridge from the other side.
    adjoint[0]["forces_eV_per_A"] = reference + scale * 8.0e-7
    adjoint[1]["forces_eV_per_A"] = reference + scale * 5.0e-7
    return primal, adjoint


def test_three_level_separated_residual_refinement_passes_with_safety_factor():
    primal, adjoint = _levels()
    summary = summarize_residual_force_refinement(primal, adjoint)
    assert summary["contract_version"] == RESIDUAL_FORCE_CONTRACT_VERSION
    assert summary["all_gates_passed"] is True
    assert all(summary["gates"].values())
    assert summary["estimated_maximum_error_eV_per_A"] < 4.0e-6
    assert summary["estimated_maximum_error_eV_per_A"] >= summary[
        "observed_release_to_reference_maximum_eV_per_A"
    ]
    assert "not a rigorous mathematical upper bound" in summary["claim_boundary"]


def test_numerical_plateau_does_not_manufacture_a_refinement_order():
    primal, adjoint = _levels(scale=1.0e-4)
    summary = summarize_residual_force_refinement(primal, adjoint)
    assert summary["primal_contribution"]["numerical_plateau"] is True
    assert summary["adjoint_contribution"]["numerical_plateau"] is True
    assert summary["primal_contribution"]["tail_multiplier"] is None
    assert summary["all_gates_passed"] is True


def test_refinement_rejects_noncontracting_or_over_budget_force_error():
    primal, adjoint = _levels()
    primal[2]["forces_eV_per_A"] = np.asarray(
        primal[1]["forces_eV_per_A"]
    ) + 2.0 * (
        np.asarray(primal[1]["forces_eV_per_A"])
        - np.asarray(primal[0]["forces_eV_per_A"])
    )
    failed = summarize_residual_force_refinement(primal, adjoint)
    assert failed["primal_contribution"]["gate_passed"] is False
    assert failed["all_gates_passed"] is False

    primal, adjoint = _levels(scale=100.0)
    large = summarize_residual_force_refinement(primal, adjoint)
    assert large["estimated_maximum_error_eV_per_A"] > (
        RESIDUAL_FORCE_ERROR_BUDGET_EV_PER_A
    )
    assert large["gates"]["estimated_rms_and_maximum_le_5e-5_eV_per_A"] is False
    assert large["all_gates_passed"] is False


def test_refinement_rejects_mixed_series_wrong_tolerances_and_bad_residuals():
    primal, adjoint = _levels()
    broken = deepcopy(adjoint)
    broken[-1]["forces_eV_per_A"] = np.asarray(
        broken[-1]["forces_eV_per_A"]
    ) + 1.0e-12
    with pytest.raises(ValueError, match="bridge force"):
        summarize_residual_force_refinement(primal, broken)

    broken = deepcopy(primal)
    broken[1]["primal_tolerance"] = 2.0e-13
    with pytest.raises(ValueError, match="changed from"):
        summarize_residual_force_refinement(broken, adjoint)

    broken = deepcopy(primal)
    broken[2]["actual_primal_residual"] = 2.0e-14
    summary = summarize_residual_force_refinement(broken, adjoint)
    assert summary["gates"]["all_residuals_within_separate_level_ceilings"] is False
    assert summary["all_gates_passed"] is False


def test_panel_summary_requires_exact_unique_coverage_and_never_hides_failure():
    first_primal, first_adjoint = _levels()
    second_primal, second_adjoint = _levels(scale=0.5)
    records = []
    for molecule_id, primal, adjoint in (
        ("water", first_primal, first_adjoint),
        ("methanol", second_primal, second_adjoint),
    ):
        record = summarize_residual_force_refinement(primal, adjoint)
        record["molecule_id"] = molecule_id
        records.append(record)
    panel = summarize_residual_force_panel(records, ("water", "methanol"))
    assert panel["molecule_count"] == 2
    assert panel["all_gates_passed"] is True

    failed = deepcopy(records)
    failed[0]["all_gates_passed"] = False
    assert summarize_residual_force_panel(
        failed, ("water", "methanol")
    )["all_gates_passed"] is False
    with pytest.raises(ValueError, match="coverage"):
        summarize_residual_force_panel(records[:1], ("water", "methanol"))
