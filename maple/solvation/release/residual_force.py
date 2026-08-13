"""Preregistered residual-refinement estimate for Route-2 force error.

This module deliberately keeps the primal and adjoint residuals in separate
spaces.  It estimates the algebraic force error by tightening one solve at a
time and never forms a mixed-unit residual norm.  The result is an empirical
a-posteriori estimate with a frozen safety factor, not a rigorous theorem or a
capability-admission decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

RESIDUAL_FORCE_CONTRACT_VERSION = (
    "route2-fixedbox590-residual-refinement-force-error-contract-v1"
)
RESIDUAL_FORCE_SCHEMA_VERSION = (
    "route2-fixedbox590-residual-refinement-force-error-summary-v1"
)

# These are solver controls, not measured residual values.  Three levels are
# required so that a declining refinement sequence can be distinguished from
# one fortuitously small difference.
PRIMAL_REFINEMENT_TOLERANCES = (1.0e-12, 1.0e-13, 1.0e-14)
ADJOINT_REFINEMENT_RELATIVE_TOLERANCES = (1.0e-11, 1.0e-12, 1.0e-13)
ADJOINT_REFINEMENT_ABSOLUTE_TOLERANCES = (1.0e-13, 1.0e-14, 1.0e-15)

# Measured primal and adjoint norms remain separate because they are not the
# same physical/numerical quantity.  The loose release ceilings agree with the
# original force gate; the tighter levels must meet successively smaller caps.
MAXIMUM_PRIMAL_RESIDUALS = (1.0e-12, 1.0e-13, 1.0e-14)
MAXIMUM_ADJOINT_RESIDUALS = (1.0e-10, 1.0e-11, 1.0e-12)

# Ten percent of the preregistered 5e-4 eV/A Cartesian RMS tolerance.  Applying
# this same stricter value to the component maximum is more conservative than
# ten percent of the separate 2e-3 eV/A component tolerance.
RESIDUAL_FORCE_ERROR_BUDGET_EV_PER_A = 5.0e-5
RESIDUAL_FORCE_SAFETY_FACTOR = 2.0
RESIDUAL_FORCE_MAXIMUM_REFINEMENT_RATIO = 0.5
RESIDUAL_FORCE_NUMERICAL_PLATEAU_EV_PER_A = 1.0e-8


def _finite_nonnegative(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _force(values: object, shape: tuple[int, int] | None, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if (
        result.ndim != 2
        or result.shape[0] < 1
        or result.shape[1] != 3
        or not np.all(np.isfinite(result))
        or (shape is not None and result.shape != shape)
    ):
        expected = "(N, 3)" if shape is None else str(shape)
        raise ValueError(f"{name} must be finite with shape {expected}.")
    return np.array(result, copy=True)


def _exact_float(record: Mapping[str, Any], name: str, expected: float) -> float:
    value = float(record.get(name))
    if not math.isfinite(value) or value != expected:
        raise ValueError(f"{name} changed from the preregistered contract.")
    return value


def _refinement_estimate(
    forces: Sequence[np.ndarray], *, contribution: str
) -> dict[str, object]:
    first_difference = np.abs(forces[0] - forces[1])
    second_difference = np.abs(forces[1] - forces[2])
    first_maximum = float(np.max(first_difference))
    second_maximum = float(np.max(second_difference))
    plateau = max(first_maximum, second_maximum) <= (
        RESIDUAL_FORCE_NUMERICAL_PLATEAU_EV_PER_A
    )
    if first_maximum == 0.0:
        ratio = 0.0 if second_maximum == 0.0 else math.inf
    else:
        ratio = second_maximum / first_maximum
    contracting = plateau or ratio <= RESIDUAL_FORCE_MAXIMUM_REFINEMENT_RATIO
    if plateau:
        # At a preregistered numerical plateau, manufacturing a convergence
        # ratio from roundoff is less honest than retaining both observed
        # differences and the full safety factor.
        component_estimate = RESIDUAL_FORCE_SAFETY_FACTOR * (
            first_difference + second_difference
        )
        tail_multiplier = None
    elif contracting:
        tail_multiplier = 1.0 / (1.0 - ratio)
        component_estimate = RESIDUAL_FORCE_SAFETY_FACTOR * (
            first_difference + tail_multiplier * second_difference
        )
    else:
        tail_multiplier = None
        component_estimate = np.full(first_difference.shape, math.inf)

    finite = bool(np.all(np.isfinite(component_estimate)))
    rms = (
        float(np.sqrt(np.mean(component_estimate**2))) if finite else math.inf
    )
    maximum = float(np.max(component_estimate)) if finite else math.inf
    return {
        "contribution": contribution,
        "first_refinement_difference_rms_eV_per_A": float(
            np.sqrt(np.mean(first_difference**2))
        ),
        "first_refinement_difference_maximum_eV_per_A": first_maximum,
        "second_refinement_difference_rms_eV_per_A": float(
            np.sqrt(np.mean(second_difference**2))
        ),
        "second_refinement_difference_maximum_eV_per_A": second_maximum,
        "maximum_refinement_ratio": ratio,
        "maximum_allowed_refinement_ratio": (
            RESIDUAL_FORCE_MAXIMUM_REFINEMENT_RATIO
        ),
        "numerical_plateau": plateau,
        "numerical_plateau_threshold_eV_per_A": (
            RESIDUAL_FORCE_NUMERICAL_PLATEAU_EV_PER_A
        ),
        "tail_multiplier": tail_multiplier,
        "safety_factor": RESIDUAL_FORCE_SAFETY_FACTOR,
        "componentwise_estimate_eV_per_A": component_estimate.tolist(),
        "estimated_rms_error_eV_per_A": rms,
        "estimated_maximum_error_eV_per_A": maximum,
        "gates": {
            "contracting_or_numerical_plateau": contracting,
            "finite_estimate": finite,
        },
        "gate_passed": contracting and finite,
    }


def summarize_residual_force_refinement(
    primal_levels: Sequence[Mapping[str, Any]],
    adjoint_levels: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Estimate release-force error from separated three-level refinements.

    ``adjoint_levels`` all use the release primal state and tighten only the
    adjoint.  ``primal_levels`` all use the tightest adjoint and tighten only
    the primal.  Consequently ``adjoint_levels[-1]`` and
    ``primal_levels[0]`` are the same bridge evaluation.  This decomposition
    prevents an arbitrary mixed norm of primal and adjoint residuals.
    """

    primal = tuple(primal_levels)
    adjoint = tuple(adjoint_levels)
    if len(primal) != 3 or len(adjoint) != 3:
        raise ValueError("Residual-force refinement requires exactly three levels.")
    if not all(isinstance(item, Mapping) for item in (*primal, *adjoint)):
        raise TypeError("Residual-force refinement levels must be mappings.")

    shape: tuple[int, int] | None = None
    primal_forces: list[np.ndarray] = []
    primal_records: list[dict[str, object]] = []
    for index, (raw, requested, residual_limit) in enumerate(
        zip(primal, PRIMAL_REFINEMENT_TOLERANCES, MAXIMUM_PRIMAL_RESIDUALS)
    ):
        _exact_float(raw, "primal_tolerance", requested)
        _exact_float(
            raw,
            "adjoint_relative_tolerance",
            ADJOINT_REFINEMENT_RELATIVE_TOLERANCES[-1],
        )
        _exact_float(
            raw,
            "adjoint_absolute_tolerance",
            ADJOINT_REFINEMENT_ABSOLUTE_TOLERANCES[-1],
        )
        actual_primal = _finite_nonnegative(
            raw.get("actual_primal_residual"), "actual_primal_residual"
        )
        actual_adjoint = _finite_nonnegative(
            raw.get("actual_adjoint_residual"), "actual_adjoint_residual"
        )
        values = _force(raw.get("forces_eV_per_A"), shape, "primal-level force")
        shape = values.shape
        primal_forces.append(values)
        primal_records.append(
            {
                "level": index,
                "primal_tolerance": requested,
                "actual_primal_residual": actual_primal,
                "maximum_primal_residual": residual_limit,
                "adjoint_relative_tolerance": (
                    ADJOINT_REFINEMENT_RELATIVE_TOLERANCES[-1]
                ),
                "adjoint_absolute_tolerance": (
                    ADJOINT_REFINEMENT_ABSOLUTE_TOLERANCES[-1]
                ),
                "actual_adjoint_residual": actual_adjoint,
                "maximum_adjoint_residual": MAXIMUM_ADJOINT_RESIDUALS[-1],
                "forces_eV_per_A": values.tolist(),
                "gates": {
                    "primal_residual_within_level_ceiling": (
                        actual_primal <= residual_limit
                    ),
                    "adjoint_residual_within_reference_ceiling": (
                        actual_adjoint <= MAXIMUM_ADJOINT_RESIDUALS[-1]
                    ),
                },
            }
        )

    adjoint_forces: list[np.ndarray] = []
    adjoint_records: list[dict[str, object]] = []
    release_primal_residual: float | None = None
    for index, (raw, relative, absolute, residual_limit) in enumerate(
        zip(
            adjoint,
            ADJOINT_REFINEMENT_RELATIVE_TOLERANCES,
            ADJOINT_REFINEMENT_ABSOLUTE_TOLERANCES,
            MAXIMUM_ADJOINT_RESIDUALS,
        )
    ):
        _exact_float(raw, "primal_tolerance", PRIMAL_REFINEMENT_TOLERANCES[0])
        _exact_float(raw, "adjoint_relative_tolerance", relative)
        _exact_float(raw, "adjoint_absolute_tolerance", absolute)
        actual_primal = _finite_nonnegative(
            raw.get("actual_primal_residual"), "actual_primal_residual"
        )
        if release_primal_residual is None:
            release_primal_residual = actual_primal
        elif actual_primal != release_primal_residual:
            raise ValueError("Adjoint refinement must reuse one exact primal state.")
        actual_adjoint = _finite_nonnegative(
            raw.get("actual_adjoint_residual"), "actual_adjoint_residual"
        )
        values = _force(raw.get("forces_eV_per_A"), shape, "adjoint-level force")
        adjoint_forces.append(values)
        adjoint_records.append(
            {
                "level": index,
                "primal_tolerance": PRIMAL_REFINEMENT_TOLERANCES[0],
                "actual_primal_residual": actual_primal,
                "maximum_primal_residual": MAXIMUM_PRIMAL_RESIDUALS[0],
                "adjoint_relative_tolerance": relative,
                "adjoint_absolute_tolerance": absolute,
                "actual_adjoint_residual": actual_adjoint,
                "maximum_adjoint_residual": residual_limit,
                "forces_eV_per_A": values.tolist(),
                "gates": {
                    "primal_residual_within_release_ceiling": (
                        actual_primal <= MAXIMUM_PRIMAL_RESIDUALS[0]
                    ),
                    "adjoint_residual_within_level_ceiling": (
                        actual_adjoint <= residual_limit
                    ),
                },
            }
        )

    bridge_equal = np.array_equal(primal_forces[0], adjoint_forces[-1])
    if not bridge_equal:
        raise ValueError(
            "Primal and adjoint refinement series do not share one bridge force."
        )

    primal_estimate = _refinement_estimate(
        primal_forces, contribution="primal-root residual"
    )
    adjoint_estimate = _refinement_estimate(
        adjoint_forces, contribution="adjoint-linear-solve residual"
    )
    total_components = np.asarray(
        primal_estimate["componentwise_estimate_eV_per_A"], dtype=float
    ) + np.asarray(adjoint_estimate["componentwise_estimate_eV_per_A"], dtype=float)
    finite_total = bool(np.all(np.isfinite(total_components)))
    estimated_rms = (
        float(np.sqrt(np.mean(total_components**2))) if finite_total else math.inf
    )
    estimated_maximum = (
        float(np.max(total_components)) if finite_total else math.inf
    )
    observed_release_to_reference = np.abs(adjoint_forces[0] - primal_forces[-1])
    observed_rms = float(np.sqrt(np.mean(observed_release_to_reference**2)))
    observed_maximum = float(np.max(observed_release_to_reference))
    residual_gates = all(
        all(bool(value) for value in record["gates"].values())
        for record in (*primal_records, *adjoint_records)
    )
    convergence_gates = bool(primal_estimate["gate_passed"]) and bool(
        adjoint_estimate["gate_passed"]
    )
    estimate_dominates_observed = finite_total and bool(
        np.all(
            observed_release_to_reference
            <= total_components + 10.0 * np.finfo(float).eps
        )
    )
    budget_gate = (
        estimated_rms <= RESIDUAL_FORCE_ERROR_BUDGET_EV_PER_A
        and estimated_maximum <= RESIDUAL_FORCE_ERROR_BUDGET_EV_PER_A
    )
    gates = {
        "all_residuals_within_separate_level_ceilings": residual_gates,
        "both_refinements_contract_or_reach_numerical_plateau": convergence_gates,
        "bridge_force_is_bitwise_identical": bridge_equal,
        "estimate_dominates_observed_release_to_reference_change": (
            estimate_dominates_observed
        ),
        "estimated_rms_and_maximum_le_5e-5_eV_per_A": budget_gate,
    }
    return {
        "schema_version": RESIDUAL_FORCE_SCHEMA_VERSION,
        "contract_version": RESIDUAL_FORCE_CONTRACT_VERSION,
        "estimator_kind": (
            "separated-three-level-residual-refinement-a-posteriori-estimate"
        ),
        "claim_boundary": (
            "Empirical residual-refinement estimate with a frozen safety factor; "
            "not a rigorous mathematical upper bound and not capability admission."
        ),
        "primal_levels": primal_records,
        "adjoint_levels": adjoint_records,
        "primal_contribution": primal_estimate,
        "adjoint_contribution": adjoint_estimate,
        "release_force_eV_per_A": adjoint_forces[0].tolist(),
        "reference_force_eV_per_A": primal_forces[-1].tolist(),
        "observed_release_to_reference_rms_eV_per_A": observed_rms,
        "observed_release_to_reference_maximum_eV_per_A": observed_maximum,
        "componentwise_total_estimate_eV_per_A": total_components.tolist(),
        "estimated_rms_error_eV_per_A": estimated_rms,
        "estimated_maximum_error_eV_per_A": estimated_maximum,
        "force_error_budget_eV_per_A": RESIDUAL_FORCE_ERROR_BUDGET_EV_PER_A,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def summarize_residual_force_panel(
    records: Sequence[Mapping[str, Any]],
    expected_molecule_ids: Sequence[str],
) -> dict[str, object]:
    """Require exactly one passing residual-refinement record per molecule."""

    expected = tuple(str(value) for value in expected_molecule_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Expected molecule IDs must be non-empty and unique.")
    values = tuple(records)
    by_id = {str(item.get("molecule_id")): item for item in values}
    if len(values) != len(expected) or len(by_id) != len(values) or set(by_id) != set(
        expected
    ):
        raise ValueError("Residual-force panel molecule coverage is incomplete.")

    maximum_rms = 0.0
    maximum_component = 0.0
    maximum_observed = 0.0
    maximum_primal_by_level = [0.0, 0.0, 0.0]
    maximum_adjoint_by_level = [0.0, 0.0, 0.0]
    all_pass = True
    for molecule_id in expected:
        record = by_id[molecule_id]
        if record.get("contract_version") != RESIDUAL_FORCE_CONTRACT_VERSION:
            raise ValueError("Residual-force record has the wrong contract version.")
        maximum_rms = max(maximum_rms, float(record.get("estimated_rms_error_eV_per_A")))
        maximum_component = max(
            maximum_component,
            float(record.get("estimated_maximum_error_eV_per_A")),
        )
        maximum_observed = max(
            maximum_observed,
            float(record.get("observed_release_to_reference_maximum_eV_per_A")),
        )
        for index, level in enumerate(record.get("primal_levels", ())):
            maximum_primal_by_level[index] = max(
                maximum_primal_by_level[index],
                float(level.get("actual_primal_residual")),
            )
        for index, level in enumerate(record.get("adjoint_levels", ())):
            maximum_adjoint_by_level[index] = max(
                maximum_adjoint_by_level[index],
                float(level.get("actual_adjoint_residual")),
            )
        all_pass &= bool(record.get("all_gates_passed")) and all(
            bool(value) for value in record.get("gates", {}).values()
        )

    gates = {
        "all_molecules_pass_residual_refinement": all_pass,
        "panel_maximum_rms_le_5e-5_eV_per_A": (
            maximum_rms <= RESIDUAL_FORCE_ERROR_BUDGET_EV_PER_A
        ),
        "panel_maximum_component_le_5e-5_eV_per_A": (
            maximum_component <= RESIDUAL_FORCE_ERROR_BUDGET_EV_PER_A
        ),
    }
    return {
        "schema_version": (
            "route2-fixedbox590-residual-refinement-force-panel-summary-v1"
        ),
        "contract_version": RESIDUAL_FORCE_CONTRACT_VERSION,
        "molecule_count": len(expected),
        "molecule_ids": list(expected),
        "maximum_estimated_rms_error_eV_per_A": maximum_rms,
        "maximum_estimated_component_error_eV_per_A": maximum_component,
        "maximum_observed_release_to_reference_error_eV_per_A": maximum_observed,
        "maximum_actual_primal_residual_by_level": maximum_primal_by_level,
        "maximum_actual_adjoint_residual_by_level": maximum_adjoint_by_level,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


__all__ = [
    "ADJOINT_REFINEMENT_ABSOLUTE_TOLERANCES",
    "ADJOINT_REFINEMENT_RELATIVE_TOLERANCES",
    "MAXIMUM_ADJOINT_RESIDUALS",
    "MAXIMUM_PRIMAL_RESIDUALS",
    "PRIMAL_REFINEMENT_TOLERANCES",
    "RESIDUAL_FORCE_CONTRACT_VERSION",
    "RESIDUAL_FORCE_ERROR_BUDGET_EV_PER_A",
    "RESIDUAL_FORCE_MAXIMUM_REFINEMENT_RATIO",
    "RESIDUAL_FORCE_NUMERICAL_PLATEAU_EV_PER_A",
    "RESIDUAL_FORCE_SAFETY_FACTOR",
    "RESIDUAL_FORCE_SCHEMA_VERSION",
    "summarize_residual_force_panel",
    "summarize_residual_force_refinement",
]
