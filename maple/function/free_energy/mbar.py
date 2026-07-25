"""Fail-closed MBAR analysis for reduced potentials sampled by MAPLE.

This module deliberately delegates the MBAR estimator and time-series
decorrelation to PyMBAR.  MAPLE only validates the data contract, arranges the
samples, converts the dimensionless result to the declared energy unit, and
applies explicit diagnostic gates.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
import logging
from typing import Any, Sequence

import numpy as np

R_KCAL_PER_MOL_K = 0.00198720425864083


class MBARDependencyError(ImportError):
    """Raised when the optional, upstream MBAR implementation is unavailable."""


class _SolverWarningHandler(logging.Handler):
    def __init__(self, messages: list[str]) -> None:
        super().__init__(level=logging.WARNING)
        self._messages = messages

    def emit(self, record: logging.LogRecord) -> None:
        self._messages.append(record.getMessage())


@contextmanager
def _capture_solver_warnings():
    messages: list[str] = []
    logger = logging.getLogger("pymbar.mbar_solvers")
    handler = _SolverWarningHandler(messages)
    previous_level = logger.level
    previous_disabled = logger.disabled
    logger.setLevel(logging.WARNING)
    logger.disabled = False
    logger.addHandler(handler)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.disabled = previous_disabled


def _solver_diagnostics(messages: Sequence[str]) -> dict[str, Any]:
    warnings = [str(message) for message in messages]
    final_failures = [
        message
        for message in warnings
        if (
            "no solution found" in message.casefold()
            and "tolerance" in message.casefold()
        )
        or "exercise caution with this solution" in message.casefold()
    ]
    return {
        "warning_count": len(warnings),
        "warning_messages": warnings,
        "final_nonconvergence_messages": final_failures,
        "convergence_established": not final_failures,
    }


def _load_pymbar():
    try:
        pymbar = import_module("pymbar")
        timeseries = import_module("pymbar.timeseries")
    except ImportError as exc:
        raise MBARDependencyError(
            "MBAR analysis requires the optional PyMBAR dependency. Install "
            "MAPLE with the 'implicit-free-energy' extra; MAPLE does not ship "
            "a handwritten MBAR substitute."
        ) from exc
    return pymbar, timeseries


def _validate_inputs(
    state_replicates: Sequence[Sequence[np.ndarray]],
    lambda_values: Sequence[float],
    temperature_kelvin: float,
) -> tuple[list[list[np.ndarray]], np.ndarray]:
    lambdas = np.asarray(lambda_values, dtype=np.float64)
    if (
        lambdas.ndim != 1
        or len(lambdas) < 2
        or not np.all(np.isfinite(lambdas))
        or not np.all(np.diff(lambdas) > 0)
        or not np.isclose(lambdas[0], 0.0)
        or not np.isclose(lambdas[-1], 1.0)
    ):
        raise ValueError(
            "lambda_values must be finite, strictly increasing, and span [0, 1]."
        )
    if (
        isinstance(temperature_kelvin, bool)
        or not np.isfinite(temperature_kelvin)
        or temperature_kelvin <= 0
    ):
        raise ValueError("temperature_kelvin must be finite and positive.")
    if len(state_replicates) != len(lambdas):
        raise ValueError("One replicate collection is required per lambda state.")

    normalized: list[list[np.ndarray]] = []
    state_count = len(lambdas)
    for state_index, replicates in enumerate(state_replicates):
        if not replicates:
            raise ValueError(f"Lambda state {state_index} has no sampled replicate.")
        state_arrays: list[np.ndarray] = []
        for replicate_index, values in enumerate(replicates):
            array = np.asarray(values, dtype=np.float64)
            if (
                array.ndim != 2
                or array.shape[0] < 2
                or array.shape[1] != state_count
                or not np.all(np.isfinite(array))
            ):
                raise ValueError(
                    "Each replicate must be a finite [sample, state] reduced-"
                    f"potential matrix; state={state_index}, "
                    f"replicate={replicate_index}."
                )
            state_arrays.append(array)
        normalized.append(state_arrays)
    return normalized, lambdas


def _decorrelate_replicate(
    values: np.ndarray,
    *,
    sampled_state: int,
    timeseries,
    detect_equilibration: bool,
    equilibration_nskip: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    diagnostic = values[:, sampled_state]
    if detect_equilibration:
        t0, statistical_inefficiency, estimated_effective = (
            timeseries.detect_equilibration(
                diagnostic,
                nskip=equilibration_nskip,
            )
        )
    else:
        t0 = 0
        statistical_inefficiency = timeseries.statistical_inefficiency(diagnostic)
        estimated_effective = len(diagnostic) / statistical_inefficiency

    t0 = int(t0)
    statistical_inefficiency = float(statistical_inefficiency)
    estimated_effective = float(estimated_effective)
    if (
        t0 < 0
        or t0 >= len(values)
        or not np.isfinite(statistical_inefficiency)
        or statistical_inefficiency < 1.0
        or not np.isfinite(estimated_effective)
        or estimated_effective <= 0
    ):
        raise ValueError("PyMBAR returned invalid equilibration diagnostics.")

    relative = np.asarray(
        timeseries.subsample_correlated_data(
            diagnostic[t0:],
            g=statistical_inefficiency,
        ),
        dtype=np.int64,
    )
    indices = relative + t0
    if (
        indices.ndim != 1
        or len(indices) == 0
        or np.any(indices < t0)
        or np.any(indices >= len(values))
    ):
        raise ValueError("PyMBAR returned invalid decorrelated sample indices.")
    selected = values[indices]
    return selected, {
        "input_sample_count": int(len(values)),
        "equilibration_start_index": t0,
        "statistical_inefficiency": statistical_inefficiency,
        "estimated_effective_sample_count": estimated_effective,
        "selected_sample_count": int(len(selected)),
        "selected_indices": indices.tolist(),
    }


def analyze_mbar(
    state_replicates: Sequence[Sequence[np.ndarray]],
    *,
    lambda_values: Sequence[float],
    temperature_kelvin: float,
    standard_state: str,
    equilibrium_claim: bool,
    detect_equilibration: bool = True,
    equilibration_nskip: int = 1,
    minimum_uncorrelated_samples_per_state: int = 50,
    minimum_effective_samples_per_state: float = 50.0,
    minimum_adjacent_overlap: float = 0.03,
    maximum_iterations: int = 10000,
    relative_tolerance: float = 1.0e-7,
) -> dict[str, Any]:
    """Analyze multiple lambda states with upstream PyMBAR.

    ``state_replicates[k][r]`` is a two-dimensional array containing samples
    generated at state ``k`` in repeat ``r``.  Each row stores the reduced
    potential of that configuration evaluated at every lambda state.

    The returned ``statistical_gates_passed`` value is deliberately narrower
    than a scientific or product-promotion decision.  It cannot establish
    chemical accuracy, adequate conformer mixing, or an experimental
    confirmation partition.
    """

    normalized, lambdas = _validate_inputs(
        state_replicates,
        lambda_values,
        temperature_kelvin,
    )
    if not isinstance(standard_state, str) or not standard_state.strip():
        raise ValueError("standard_state must be a non-empty declaration.")
    integer_options = {
        "equilibration_nskip": equilibration_nskip,
        "minimum_uncorrelated_samples_per_state": (
            minimum_uncorrelated_samples_per_state
        ),
        "maximum_iterations": maximum_iterations,
    }
    for name, value in integer_options.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    numeric_options = {
        "minimum_effective_samples_per_state": (minimum_effective_samples_per_state),
        "relative_tolerance": relative_tolerance,
    }
    for name, value in numeric_options.items():
        if isinstance(value, bool) or not np.isfinite(value) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if (
        isinstance(minimum_adjacent_overlap, bool)
        or not np.isfinite(minimum_adjacent_overlap)
        or not 0 < minimum_adjacent_overlap < 1
    ):
        raise ValueError("minimum_adjacent_overlap must be strictly between 0 and 1.")

    pymbar, timeseries = _load_pymbar()
    state_diagnostics: list[dict[str, Any]] = []
    selected_by_state: list[np.ndarray] = []
    counts: list[int] = []
    for state_index, replicates in enumerate(normalized):
        selected_replicates: list[np.ndarray] = []
        repeat_diagnostics: list[dict[str, Any]] = []
        for replicate_index, values in enumerate(replicates):
            selected, diagnostics = _decorrelate_replicate(
                values,
                sampled_state=state_index,
                timeseries=timeseries,
                detect_equilibration=detect_equilibration,
                equilibration_nskip=equilibration_nskip,
            )
            diagnostics["replicate_index"] = replicate_index
            selected_replicates.append(selected)
            repeat_diagnostics.append(diagnostics)
        state_samples = np.concatenate(selected_replicates, axis=0)
        selected_by_state.append(state_samples)
        counts.append(int(len(state_samples)))
        state_diagnostics.append(
            {
                "state_index": state_index,
                "lambda": float(lambdas[state_index]),
                "selected_sample_count": int(len(state_samples)),
                "replicates": repeat_diagnostics,
            }
        )

    sample_counts = np.asarray(counts, dtype=np.int64)
    reduced_potentials = np.concatenate(selected_by_state, axis=0).T
    with _capture_solver_warnings() as solver_warnings:
        mbar = pymbar.MBAR(
            reduced_potentials,
            sample_counts,
            maximum_iterations=maximum_iterations,
            relative_tolerance=relative_tolerance,
            solver_protocol="robust",
        )
        differences = mbar.compute_free_energy_differences()
        overlap = mbar.compute_overlap()
        effective_samples = np.asarray(
            mbar.compute_effective_sample_number(),
            dtype=np.float64,
        )
    solver_diagnostics = _solver_diagnostics(solver_warnings)
    delta_f = np.asarray(differences["Delta_f"], dtype=np.float64)
    delta_f_uncertainty = np.asarray(
        differences["dDelta_f"],
        dtype=np.float64,
    )
    overlap_matrix = np.asarray(overlap["matrix"], dtype=np.float64)
    state_count = len(lambdas)
    expected_shape = (state_count, state_count)
    if (
        delta_f.shape != expected_shape
        or delta_f_uncertainty.shape != expected_shape
        or overlap_matrix.shape != expected_shape
        or effective_samples.shape != (state_count,)
    ):
        raise ValueError("PyMBAR returned an unexpected result shape.")

    adjacent = [
        {
            "left_state": index,
            "right_state": index + 1,
            "left_to_right": float(overlap_matrix[index, index + 1]),
            "right_to_left": float(overlap_matrix[index + 1, index]),
            "minimum_directional_overlap": float(
                min(
                    overlap_matrix[index, index + 1],
                    overlap_matrix[index + 1, index],
                )
            ),
        }
        for index in range(state_count - 1)
    ]
    minimum_observed_overlap = min(
        item["minimum_directional_overlap"] for item in adjacent
    )
    finite_outputs = all(
        np.all(np.isfinite(value))
        for value in (
            delta_f,
            delta_f_uncertainty,
            overlap_matrix,
            effective_samples,
        )
    )
    checks = {
        "equilibrium_claim": equilibrium_claim is True,
        "finite_outputs": bool(finite_outputs),
        "solver_convergence": solver_diagnostics["convergence_established"],
        "minimum_uncorrelated_samples_per_state": (
            int(sample_counts.min()) >= minimum_uncorrelated_samples_per_state
        ),
        "minimum_effective_samples_per_state": (
            float(effective_samples.min()) >= minimum_effective_samples_per_state
        ),
        "adjacent_overlap": (minimum_observed_overlap >= minimum_adjacent_overlap),
    }
    kbt_kcal_mol = R_KCAL_PER_MOL_K * temperature_kelvin
    endpoint_kcal_mol = float(delta_f[0, -1] * kbt_kcal_mol)
    endpoint_uncertainty_kcal_mol = float(delta_f_uncertainty[0, -1] * kbt_kcal_mol)
    return {
        "schema_version": 1,
        "estimator": {
            "name": "PyMBAR MBAR",
            "pymbar_version": str(getattr(pymbar, "__version__", "unknown")),
            "handwritten_estimator": False,
        },
        "standard_state": standard_state.strip(),
        "temperature_kelvin": float(temperature_kelvin),
        "kbt_kcal_mol": float(kbt_kcal_mol),
        "lambda_values": lambdas.tolist(),
        "sample_counts": sample_counts.tolist(),
        "state_diagnostics": state_diagnostics,
        "free_energy_matrix_kcal_mol": (delta_f * kbt_kcal_mol).tolist(),
        "uncertainty_matrix_kcal_mol": (delta_f_uncertainty * kbt_kcal_mol).tolist(),
        "endpoint_delta_g_kcal_mol": endpoint_kcal_mol,
        "endpoint_uncertainty_kcal_mol": endpoint_uncertainty_kcal_mol,
        "solver_diagnostics": solver_diagnostics,
        "overlap": {
            "matrix": overlap_matrix.tolist(),
            "scalar": float(overlap["scalar"]),
            "eigenvalues": np.asarray(
                overlap["eigenvalues"],
                dtype=np.float64,
            ).tolist(),
            "adjacent_pairs": adjacent,
            "minimum_directional_adjacent_overlap": float(minimum_observed_overlap),
        },
        "effective_sample_count_by_state": effective_samples.tolist(),
        "gates": {
            "requirements": {
                "equilibrium_claim": True,
                "minimum_uncorrelated_samples_per_state": (
                    minimum_uncorrelated_samples_per_state
                ),
                "minimum_effective_samples_per_state": float(
                    minimum_effective_samples_per_state
                ),
                "minimum_adjacent_overlap": float(minimum_adjacent_overlap),
            },
            "checks": checks,
            "statistical_gates_passed": all(checks.values()),
        },
        "claim_boundary": {
            "chemical_accuracy_established": False,
            "conformer_mixing_established": False,
            "experimental_confirmation_established": False,
            "product_promotion_decided_here": False,
        },
    }
