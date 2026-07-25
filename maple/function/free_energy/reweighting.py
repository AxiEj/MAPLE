"""Fail-closed reference-to-target Hamiltonian reweighting.

The sampled states may use a cheap molecular-mechanics reference potential,
but the returned correction is always defined between the explicitly supplied
reference and target reduced potentials.  MAPLE delegates EXP, BAR, MBAR, and
time-series decorrelation to upstream PyMBAR.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .mbar import (
    R_KCAL_PER_MOL_K,
    _capture_solver_warnings,
    _load_pymbar,
    _solver_diagnostics,
    analyze_mbar,
)


def _validate_fraction(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not np.isfinite(value)
        or not 0.0 < float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be finite and in (0, 1].")
    return float(value)


def _validate_overlap(name: str, value: float) -> float:
    overlap = _validate_fraction(name, value)
    if overlap >= 1.0:
        raise ValueError(f"{name} must be finite and strictly between 0 and 1.")
    return overlap


def _validate_positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not np.isfinite(value) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return float(value)


def _selected_work(
    state_replicates: Sequence[Sequence[np.ndarray]],
    state_diagnostics: Sequence[dict[str, Any]],
    *,
    sampled_state: int,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    diagnostics = state_diagnostics[sampled_state]["replicates"]
    if len(diagnostics) != len(state_replicates[sampled_state]):
        raise ValueError("MBAR replicate diagnostics do not match the input.")
    for values, replicate in zip(
        state_replicates[sampled_state],
        diagnostics,
        strict=True,
    ):
        array = np.asarray(values, dtype=np.float64)
        indices = np.asarray(replicate["selected_indices"], dtype=np.int64)
        chosen = array[indices]
        if sampled_state == 0:
            selected.append(chosen[:, 1] - chosen[:, 0])
        else:
            selected.append(chosen[:, 0] - chosen[:, 1])
    return np.concatenate(selected)


def _weight_diagnostics(work: np.ndarray) -> dict[str, float | int]:
    log_weights = -np.asarray(work, dtype=np.float64)
    shifted = np.exp(log_weights - float(np.max(log_weights)))
    normalized = shifted / float(np.sum(shifted))
    effective = float(1.0 / np.dot(normalized, normalized))
    return {
        "sample_count": int(len(normalized)),
        "effective_sample_count": effective,
        "effective_sample_fraction": effective / len(normalized),
        "maximum_normalized_weight": float(np.max(normalized)),
    }


def analyze_bidirectional_reweighting(
    state_replicates: Sequence[Sequence[np.ndarray]],
    *,
    temperature_kelvin: float,
    standard_state: str,
    equilibrium_claim: bool,
    detect_equilibration: bool = True,
    equilibration_nskip: int = 1,
    minimum_uncorrelated_samples_per_state: int = 50,
    minimum_effective_samples_per_state: float = 50.0,
    minimum_bar_overlap: float = 0.03,
    minimum_directional_effective_fraction: float = 0.10,
    maximum_directional_disagreement_kcal_mol: float = 0.50,
    maximum_bar_uncertainty_kcal_mol: float = 0.50,
    maximum_bar_mbar_disagreement_kcal_mol: float = 0.005,
    maximum_iterations: int = 10000,
    relative_tolerance: float = 1.0e-7,
) -> dict[str, Any]:
    """Analyze a sampled reference and target state with upstream PyMBAR.

    ``state_replicates`` contains exactly two sampled states.  Every replicate
    is a ``[sample, 2]`` reduced-potential matrix evaluated under both the
    reference and target Hamiltonians.  State 0 is the cheap reference and
    state 1 is the target.

    The forward EXP result is the estimate available from reference samples
    alone.  Reverse EXP, BAR, and MBAR are validation diagnostics requiring
    target samples.  Passing this function does not by itself certify that a
    future reference-only calculation is converged.
    """

    if len(state_replicates) != 2:
        raise ValueError(
            "Bidirectional reweighting requires reference and target samples."
        )
    minimum_directional_effective_fraction = _validate_fraction(
        "minimum_directional_effective_fraction",
        minimum_directional_effective_fraction,
    )
    minimum_bar_overlap = _validate_overlap(
        "minimum_bar_overlap",
        minimum_bar_overlap,
    )
    maximum_directional_disagreement_kcal_mol = _validate_positive(
        "maximum_directional_disagreement_kcal_mol",
        maximum_directional_disagreement_kcal_mol,
    )
    maximum_bar_uncertainty_kcal_mol = _validate_positive(
        "maximum_bar_uncertainty_kcal_mol",
        maximum_bar_uncertainty_kcal_mol,
    )
    maximum_bar_mbar_disagreement_kcal_mol = _validate_positive(
        "maximum_bar_mbar_disagreement_kcal_mol",
        maximum_bar_mbar_disagreement_kcal_mol,
    )

    mbar = analyze_mbar(
        state_replicates,
        lambda_values=[0.0, 1.0],
        temperature_kelvin=temperature_kelvin,
        standard_state=standard_state,
        equilibrium_claim=equilibrium_claim,
        detect_equilibration=detect_equilibration,
        equilibration_nskip=equilibration_nskip,
        minimum_uncorrelated_samples_per_state=(minimum_uncorrelated_samples_per_state),
        minimum_effective_samples_per_state=(minimum_effective_samples_per_state),
        minimum_adjacent_overlap=minimum_bar_overlap,
        maximum_iterations=maximum_iterations,
        relative_tolerance=relative_tolerance,
    )
    pymbar, _ = _load_pymbar()
    forward_work = _selected_work(
        state_replicates,
        mbar["state_diagnostics"],
        sampled_state=0,
    )
    reverse_work = _selected_work(
        state_replicates,
        mbar["state_diagnostics"],
        sampled_state=1,
    )
    with _capture_solver_warnings() as bar_solver_warnings:
        forward = pymbar.exp(forward_work, compute_uncertainty=True)
        reverse = pymbar.exp(reverse_work, compute_uncertainty=True)
        bar = pymbar.bar(
            forward_work,
            reverse_work,
            compute_uncertainty=True,
            uncertainty_method="MBAR",
            maximum_iterations=maximum_iterations,
            relative_tolerance=relative_tolerance,
        )
        overlap = float(pymbar.bar_overlap(forward_work, reverse_work))
    bar_solver_diagnostics = _solver_diagnostics(bar_solver_warnings)
    kbt = R_KCAL_PER_MOL_K * float(temperature_kelvin)
    forward_delta = float(forward["Delta_f"] * kbt)
    reverse_delta = float(-reverse["Delta_f"] * kbt)
    bar_delta = float(bar["Delta_f"] * kbt)
    bar_uncertainty = float(bar["dDelta_f"] * kbt)
    mbar_delta = float(mbar["endpoint_delta_g_kcal_mol"])
    bar_mbar_disagreement = abs(bar_delta - mbar_delta)
    forward_weights = _weight_diagnostics(forward_work)
    reverse_weights = _weight_diagnostics(reverse_work)
    directional_disagreement = abs(forward_delta - reverse_delta)
    maximum_bar_deviation = max(
        abs(forward_delta - bar_delta),
        abs(reverse_delta - bar_delta),
    )
    checks = {
        "mbar_statistical_gates": bool(mbar["gates"]["statistical_gates_passed"]),
        "mbar_solver_convergence": mbar["gates"]["checks"]["solver_convergence"],
        "bar_solver_convergence": bar_solver_diagnostics["convergence_established"],
        "minimum_uncorrelated_samples_per_state": mbar["gates"]["checks"][
            "minimum_uncorrelated_samples_per_state"
        ],
        "minimum_effective_samples_per_state": mbar["gates"]["checks"][
            "minimum_effective_samples_per_state"
        ],
        "minimum_mbar_directional_overlap": mbar["gates"]["checks"]["adjacent_overlap"],
        "minimum_bar_overlap": overlap >= minimum_bar_overlap,
        "minimum_directional_effective_fraction": min(
            forward_weights["effective_sample_fraction"],
            reverse_weights["effective_sample_fraction"],
        )
        >= minimum_directional_effective_fraction,
        "maximum_directional_disagreement": (
            directional_disagreement <= maximum_directional_disagreement_kcal_mol
        ),
        "maximum_bar_uncertainty": (
            bar_uncertainty <= maximum_bar_uncertainty_kcal_mol
        ),
        "maximum_bar_mbar_disagreement": (
            bar_mbar_disagreement <= maximum_bar_mbar_disagreement_kcal_mol
        ),
    }
    return {
        "schema_version": 1,
        "estimator": {
            "bidirectional": "PyMBAR BAR",
            "directional": "PyMBAR EXP",
            "multistate": "PyMBAR MBAR",
            "pymbar_version": mbar["estimator"]["pymbar_version"],
            "handwritten_estimator": False,
        },
        "standard_state": mbar["standard_state"],
        "temperature_kelvin": mbar["temperature_kelvin"],
        "kbt_kcal_mol": mbar["kbt_kcal_mol"],
        "directional": {
            "reference_to_target_exp": {
                "delta_g_kcal_mol": forward_delta,
                "uncertainty_kcal_mol": float(forward["dDelta_f"] * kbt),
                **forward_weights,
            },
            "target_to_reference_exp": {
                "delta_g_kcal_mol": reverse_delta,
                "uncertainty_kcal_mol": float(reverse["dDelta_f"] * kbt),
                **reverse_weights,
            },
            "absolute_forward_reverse_disagreement_kcal_mol": (
                directional_disagreement
            ),
            "maximum_absolute_deviation_from_bar_kcal_mol": (maximum_bar_deviation),
        },
        "bar": {
            "delta_g_kcal_mol": bar_delta,
            "uncertainty_kcal_mol": bar_uncertainty,
            "overlap": overlap,
            "absolute_difference_from_mbar_kcal_mol": (bar_mbar_disagreement),
            "solver_diagnostics": bar_solver_diagnostics,
        },
        "mbar": mbar,
        "gates": {
            "requirements": {
                "minimum_bar_overlap": minimum_bar_overlap,
                "minimum_directional_effective_fraction": (
                    minimum_directional_effective_fraction
                ),
                "maximum_directional_disagreement_kcal_mol": (
                    maximum_directional_disagreement_kcal_mol
                ),
                "maximum_bar_uncertainty_kcal_mol": (maximum_bar_uncertainty_kcal_mol),
                "maximum_bar_mbar_disagreement_kcal_mol": (
                    maximum_bar_mbar_disagreement_kcal_mol
                ),
            },
            "checks": checks,
            "statistical_gates_passed": all(checks.values()),
        },
        "claim_boundary": {
            "reference_only_production_validated": False,
            "chemical_accuracy_established": False,
            "experimental_confirmation_established": False,
            "product_promotion_decided_here": False,
        },
    }
