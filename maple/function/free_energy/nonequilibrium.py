"""Fail-closed analysis of bidirectional nonequilibrium switching work.

MAPLE delegates the exponential-average and Bennett estimators to upstream
PyMBAR.  This module only validates work records, preserves independent
replicate boundaries, converts units, and applies explicit numerical and
sampling-claim gates.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .mbar import (
    R_KCAL_PER_MOL_K,
    _capture_solver_warnings,
    _load_pymbar,
    _solver_diagnostics,
)

_WORK_UNITS = {"kcal/mol", "reduced"}


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not np.isfinite(value) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return float(value)


def _nonnegative_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not np.isfinite(value) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return float(value)


def _overlap_threshold(name: str, value: float) -> float:
    threshold = _positive_float(name, value)
    if threshold >= 1.0:
        raise ValueError(f"{name} must be finite and strictly between 0 and 1.")
    return threshold


def _work_replicates(
    values: Sequence[Sequence[float] | np.ndarray],
    *,
    name: str,
    beta: float,
    work_unit: str,
) -> list[np.ndarray]:
    if not values:
        raise ValueError(f"{name} must contain at least one replicate.")
    output: list[np.ndarray] = []
    for replicate_index, replicate in enumerate(values):
        array = np.asarray(replicate, dtype=np.float64)
        if array.ndim != 1 or len(array) < 2 or not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} replicate {replicate_index} must be a finite "
                "one-dimensional array with at least two work values."
            )
        output.append(array * beta if work_unit == "kcal/mol" else array)
    return output


def _weight_diagnostics(dimensionless_work: np.ndarray) -> dict[str, float | int]:
    log_weights = -np.asarray(dimensionless_work, dtype=np.float64)
    shifted = np.exp(log_weights - float(np.max(log_weights)))
    normalized = shifted / float(np.sum(shifted))
    effective = float(1.0 / np.dot(normalized, normalized))
    return {
        "sample_count": int(len(normalized)),
        "effective_sample_count": effective,
        "effective_sample_fraction": effective / len(normalized),
        "maximum_normalized_weight": float(np.max(normalized)),
    }


def _exp_result(pymbar, work: np.ndarray, *, sign: float, kbt: float) -> dict[str, Any]:
    result = pymbar.exp(work, compute_uncertainty=True)
    delta = sign * float(result["Delta_f"]) * kbt
    uncertainty = float(result["dDelta_f"]) * kbt
    if not np.all(np.isfinite((delta, uncertainty))):
        raise ValueError("PyMBAR EXP returned non-finite outputs.")
    return {
        "delta_g_kcal_mol": delta,
        "uncertainty_kcal_mol": uncertainty,
        **_weight_diagnostics(work),
    }


def _replicate_exp_results(
    pymbar,
    replicates: Sequence[np.ndarray],
    *,
    sign: float,
    kbt: float,
) -> list[dict[str, Any]]:
    return [
        {
            "replicate_index": index,
            **_exp_result(pymbar, work, sign=sign, kbt=kbt),
        }
        for index, work in enumerate(replicates)
    ]


def _range(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(array.max() - array.min()) if len(array) > 1 else 0.0


def _bar_result(
    pymbar,
    forward: np.ndarray,
    reverse: np.ndarray,
    *,
    kbt: float,
    maximum_iterations: int,
    relative_tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with _capture_solver_warnings() as solver_warnings:
        result = pymbar.bar(
            forward,
            reverse,
            compute_uncertainty=True,
            uncertainty_method="MBAR",
            maximum_iterations=maximum_iterations,
            relative_tolerance=relative_tolerance,
        )
        raw_overlap = complex(pymbar.bar_overlap(forward, reverse))
    diagnostics = _solver_diagnostics(solver_warnings)
    if abs(raw_overlap.imag) > 1.0e-12:
        raise ValueError("PyMBAR BAR overlap has a non-negligible imaginary part.")
    overlap = float(raw_overlap.real)
    delta = float(result["Delta_f"]) * kbt
    uncertainty = float(result["dDelta_f"]) * kbt
    if not np.all(np.isfinite((delta, uncertainty, overlap))):
        raise ValueError("PyMBAR BAR returned non-finite outputs.")
    return (
        {
            "delta_g_kcal_mol": delta,
            "uncertainty_kcal_mol": uncertainty,
            "overlap": overlap,
            "solver_diagnostics": diagnostics,
        },
        diagnostics,
    )


def _leave_one_replicate_out(
    pymbar,
    forward_replicates: Sequence[np.ndarray],
    reverse_replicates: Sequence[np.ndarray],
    *,
    pooled_delta: float,
    kbt: float,
    maximum_iterations: int,
    relative_tolerance: float,
) -> dict[str, Any]:
    estimates: list[dict[str, Any]] = []
    for direction, replicates in (
        ("forward", forward_replicates),
        ("reverse", reverse_replicates),
    ):
        if len(replicates) < 2:
            continue
        for omitted_index in range(len(replicates)):
            forward = np.concatenate(
                [
                    values
                    for index, values in enumerate(forward_replicates)
                    if direction != "forward" or index != omitted_index
                ]
            )
            reverse = np.concatenate(
                [
                    values
                    for index, values in enumerate(reverse_replicates)
                    if direction != "reverse" or index != omitted_index
                ]
            )
            result, diagnostics = _bar_result(
                pymbar,
                forward,
                reverse,
                kbt=kbt,
                maximum_iterations=maximum_iterations,
                relative_tolerance=relative_tolerance,
            )
            estimates.append(
                {
                    "omitted_direction": direction,
                    "omitted_replicate_index": omitted_index,
                    "delta_g_kcal_mol": result["delta_g_kcal_mol"],
                    "absolute_deviation_from_pooled_kcal_mol": abs(
                        result["delta_g_kcal_mol"] - pooled_delta
                    ),
                    "solver_diagnostics": diagnostics,
                }
            )
    maximum_deviation = max(
        (estimate["absolute_deviation_from_pooled_kcal_mol"] for estimate in estimates),
        default=float("inf"),
    )
    return {
        "estimates": estimates,
        "maximum_absolute_deviation_from_pooled_kcal_mol": maximum_deviation,
        "all_solver_convergence_established": bool(estimates)
        and all(
            estimate["solver_diagnostics"]["convergence_established"]
            for estimate in estimates
        ),
    }


def analyze_nonequilibrium_switching(
    forward_work_replicates: Sequence[Sequence[float] | np.ndarray],
    reverse_work_replicates: Sequence[Sequence[float] | np.ndarray],
    *,
    work_unit: str,
    temperature_kelvin: float,
    standard_state: str,
    endpoint_equilibrium_claim: bool,
    independent_work_values_claim: bool,
    minimum_work_values_per_direction: int = 50,
    minimum_independent_replicates_per_direction: int = 2,
    minimum_bar_overlap: float = 0.03,
    maximum_forward_reverse_disagreement_kcal_mol: float = 0.50,
    maximum_bar_uncertainty_kcal_mol: float = 0.50,
    maximum_replicate_exp_range_kcal_mol: float = 0.50,
    maximum_leave_one_replicate_out_deviation_kcal_mol: float = 0.50,
    maximum_negative_dissipation_kcal_mol: float = 0.10,
    maximum_iterations: int = 10000,
    relative_tolerance: float = 1.0e-7,
) -> dict[str, Any]:
    """Analyze forward and reverse switching work with upstream PyMBAR.

    Forward work is defined for the reference-to-target protocol.  Reverse
    work is defined in the target-to-reference direction.  Work may be
    supplied either in kcal/mol or already reduced by ``k_B T``.

    Replicate boundaries must identify independently initialized endpoint
    trajectories or chains.  This function does not decorrelate endpoint
    configurations after the fact and does not convert a short diagnostic
    into an equilibrium claim.
    """

    if work_unit not in _WORK_UNITS:
        allowed = ", ".join(sorted(_WORK_UNITS))
        raise ValueError(f"work_unit must be one of: {allowed}.")
    if (
        isinstance(temperature_kelvin, bool)
        or not np.isfinite(temperature_kelvin)
        or float(temperature_kelvin) <= 0.0
    ):
        raise ValueError("temperature_kelvin must be finite and positive.")
    if not isinstance(standard_state, str) or not standard_state.strip():
        raise ValueError("standard_state must be a non-empty declaration.")
    if not isinstance(endpoint_equilibrium_claim, bool):
        raise ValueError("endpoint_equilibrium_claim must be boolean.")
    if not isinstance(independent_work_values_claim, bool):
        raise ValueError("independent_work_values_claim must be boolean.")

    minimum_work_values_per_direction = _positive_integer(
        "minimum_work_values_per_direction",
        minimum_work_values_per_direction,
    )
    minimum_independent_replicates_per_direction = _positive_integer(
        "minimum_independent_replicates_per_direction",
        minimum_independent_replicates_per_direction,
    )
    maximum_iterations = _positive_integer("maximum_iterations", maximum_iterations)
    minimum_bar_overlap = _overlap_threshold(
        "minimum_bar_overlap",
        minimum_bar_overlap,
    )
    maximum_forward_reverse_disagreement_kcal_mol = _positive_float(
        "maximum_forward_reverse_disagreement_kcal_mol",
        maximum_forward_reverse_disagreement_kcal_mol,
    )
    maximum_bar_uncertainty_kcal_mol = _positive_float(
        "maximum_bar_uncertainty_kcal_mol",
        maximum_bar_uncertainty_kcal_mol,
    )
    maximum_replicate_exp_range_kcal_mol = _positive_float(
        "maximum_replicate_exp_range_kcal_mol",
        maximum_replicate_exp_range_kcal_mol,
    )
    maximum_leave_one_replicate_out_deviation_kcal_mol = _positive_float(
        "maximum_leave_one_replicate_out_deviation_kcal_mol",
        maximum_leave_one_replicate_out_deviation_kcal_mol,
    )
    maximum_negative_dissipation_kcal_mol = _nonnegative_float(
        "maximum_negative_dissipation_kcal_mol",
        maximum_negative_dissipation_kcal_mol,
    )
    relative_tolerance = _positive_float("relative_tolerance", relative_tolerance)

    kbt = R_KCAL_PER_MOL_K * float(temperature_kelvin)
    beta = 1.0 / kbt
    forward_replicates = _work_replicates(
        forward_work_replicates,
        name="forward_work_replicates",
        beta=beta,
        work_unit=work_unit,
    )
    reverse_replicates = _work_replicates(
        reverse_work_replicates,
        name="reverse_work_replicates",
        beta=beta,
        work_unit=work_unit,
    )

    pymbar, _ = _load_pymbar()
    forward = np.concatenate(forward_replicates)
    reverse = np.concatenate(reverse_replicates)
    forward_exp = _exp_result(pymbar, forward, sign=1.0, kbt=kbt)
    reverse_exp = _exp_result(pymbar, reverse, sign=-1.0, kbt=kbt)
    forward_by_replicate = _replicate_exp_results(
        pymbar,
        forward_replicates,
        sign=1.0,
        kbt=kbt,
    )
    reverse_by_replicate = _replicate_exp_results(
        pymbar,
        reverse_replicates,
        sign=-1.0,
        kbt=kbt,
    )
    bar, bar_solver = _bar_result(
        pymbar,
        forward,
        reverse,
        kbt=kbt,
        maximum_iterations=maximum_iterations,
        relative_tolerance=relative_tolerance,
    )
    leave_one_out = _leave_one_replicate_out(
        pymbar,
        forward_replicates,
        reverse_replicates,
        pooled_delta=bar["delta_g_kcal_mol"],
        kbt=kbt,
        maximum_iterations=maximum_iterations,
        relative_tolerance=relative_tolerance,
    )

    forward_reverse_disagreement = abs(
        forward_exp["delta_g_kcal_mol"] - reverse_exp["delta_g_kcal_mol"]
    )
    forward_replicate_range = _range(
        [record["delta_g_kcal_mol"] for record in forward_by_replicate]
    )
    reverse_replicate_range = _range(
        [record["delta_g_kcal_mol"] for record in reverse_by_replicate]
    )
    forward_mean_work = float(np.mean(forward) * kbt)
    reverse_mean_work = float(np.mean(reverse) * kbt)
    forward_dissipation = forward_mean_work - bar["delta_g_kcal_mol"]
    reverse_dissipation = reverse_mean_work + bar["delta_g_kcal_mol"]

    checks = {
        "endpoint_equilibrium_claim": endpoint_equilibrium_claim,
        "independent_work_values_claim": independent_work_values_claim,
        "minimum_work_values_per_direction": min(len(forward), len(reverse))
        >= minimum_work_values_per_direction,
        "minimum_independent_replicates_per_direction": min(
            len(forward_replicates),
            len(reverse_replicates),
        )
        >= minimum_independent_replicates_per_direction,
        "bar_solver_convergence": bar_solver["convergence_established"],
        "leave_one_replicate_out_solver_convergence": leave_one_out[
            "all_solver_convergence_established"
        ],
        "minimum_bar_overlap": bar["overlap"] >= minimum_bar_overlap,
        "maximum_forward_reverse_disagreement": forward_reverse_disagreement
        <= maximum_forward_reverse_disagreement_kcal_mol,
        "maximum_bar_uncertainty": bar["uncertainty_kcal_mol"]
        <= maximum_bar_uncertainty_kcal_mol,
        "maximum_replicate_exp_range": max(
            forward_replicate_range,
            reverse_replicate_range,
        )
        <= maximum_replicate_exp_range_kcal_mol,
        "maximum_leave_one_replicate_out_deviation": leave_one_out[
            "maximum_absolute_deviation_from_pooled_kcal_mol"
        ]
        <= maximum_leave_one_replicate_out_deviation_kcal_mol,
        "nonnegative_mean_dissipation": min(
            forward_dissipation,
            reverse_dissipation,
        )
        >= -maximum_negative_dissipation_kcal_mol,
    }
    sampling_checks = {
        "endpoint_equilibrium_claim",
        "independent_work_values_claim",
    }
    numerical_checks = {
        name: passed for name, passed in checks.items() if name not in sampling_checks
    }
    return {
        "schema_version": 1,
        "estimator": {
            "bidirectional": "PyMBAR BAR",
            "directional": "PyMBAR EXP",
            "pymbar_version": str(getattr(pymbar, "__version__", "unknown")),
            "handwritten_estimator": False,
        },
        "standard_state": standard_state.strip(),
        "temperature_kelvin": float(temperature_kelvin),
        "kbt_kcal_mol": float(kbt),
        "input": {
            "work_unit": work_unit,
            "forward_replicate_count": len(forward_replicates),
            "reverse_replicate_count": len(reverse_replicates),
            "forward_work_count": int(len(forward)),
            "reverse_work_count": int(len(reverse)),
        },
        "directional": {
            "forward_exp": forward_exp,
            "reverse_exp": reverse_exp,
            "absolute_forward_reverse_disagreement_kcal_mol": (
                forward_reverse_disagreement
            ),
            "forward_by_replicate": forward_by_replicate,
            "reverse_by_replicate": reverse_by_replicate,
            "forward_replicate_range_kcal_mol": forward_replicate_range,
            "reverse_replicate_range_kcal_mol": reverse_replicate_range,
        },
        "bar": bar,
        "leave_one_replicate_out": leave_one_out,
        "dissipation": {
            "forward_mean_work_kcal_mol": forward_mean_work,
            "reverse_mean_work_kcal_mol": reverse_mean_work,
            "forward_mean_kcal_mol": forward_dissipation,
            "reverse_mean_kcal_mol": reverse_dissipation,
            "mean_hysteresis_kcal_mol": forward_mean_work + reverse_mean_work,
        },
        "gates": {
            "requirements": {
                "endpoint_equilibrium_claim": True,
                "independent_work_values_claim": True,
                "minimum_work_values_per_direction": (
                    minimum_work_values_per_direction
                ),
                "minimum_independent_replicates_per_direction": (
                    minimum_independent_replicates_per_direction
                ),
                "minimum_bar_overlap": minimum_bar_overlap,
                "maximum_forward_reverse_disagreement_kcal_mol": (
                    maximum_forward_reverse_disagreement_kcal_mol
                ),
                "maximum_bar_uncertainty_kcal_mol": (maximum_bar_uncertainty_kcal_mol),
                "maximum_replicate_exp_range_kcal_mol": (
                    maximum_replicate_exp_range_kcal_mol
                ),
                "maximum_leave_one_replicate_out_deviation_kcal_mol": (
                    maximum_leave_one_replicate_out_deviation_kcal_mol
                ),
                "maximum_negative_dissipation_kcal_mol": (
                    maximum_negative_dissipation_kcal_mol
                ),
            },
            "checks": checks,
            "numerical_gates_passed": all(numerical_checks.values()),
            "scientific_gates_passed": all(checks.values()),
        },
        "claim_boundary": {
            "equilibrium_established_by_analysis": False,
            "independent_starting_configurations_established_by_analysis": False,
            "chemical_accuracy_established": False,
            "experimental_confirmation_established": False,
            "product_promotion_decided_here": False,
        },
    }
