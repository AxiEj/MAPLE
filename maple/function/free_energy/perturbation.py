"""Fail-closed one-sided thermodynamic perturbation analysis.

The reference ensemble is sampled under ``U_reference`` and each input value
is ``U_target - U_reference`` in kcal/mol.  MAPLE delegates exponential
averaging and time-series analysis to upstream PyMBAR.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .mbar import R_KCAL_PER_MOL_K, _load_pymbar


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not np.isfinite(value) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return float(value)


def _fraction(name: str, value: float) -> float:
    value = _positive_float(name, value)
    if value > 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1].")
    return value


def _replicates(
    values: Sequence[Sequence[float] | np.ndarray],
) -> list[np.ndarray]:
    if not values:
        raise ValueError("energy_difference_replicates_kcal_mol cannot be empty.")
    output: list[np.ndarray] = []
    for index, replicate in enumerate(values):
        array = np.asarray(replicate, dtype=np.float64)
        if array.ndim != 1 or len(array) < 2 or not np.all(np.isfinite(array)):
            raise ValueError(
                "Each energy-difference replicate must be a finite "
                f"one-dimensional array with at least two values; replicate={index}."
            )
        output.append(array)
    return output


def _weight_diagnostics(
    dimensionless_difference: np.ndarray,
) -> dict[str, float | int]:
    log_weights = -np.asarray(dimensionless_difference, dtype=np.float64)
    shifted = np.exp(log_weights - float(np.max(log_weights)))
    normalized = shifted / float(np.sum(shifted))
    effective = float(1.0 / np.dot(normalized, normalized))
    return {
        "sample_count": int(len(normalized)),
        "effective_sample_count": effective,
        "effective_sample_fraction": effective / len(normalized),
        "maximum_normalized_weight": float(np.max(normalized)),
    }


def _exp_result(
    pymbar,
    dimensionless_difference: np.ndarray,
    *,
    kbt_kcal_mol: float,
) -> dict[str, Any]:
    result = pymbar.exp(
        dimensionless_difference,
        compute_uncertainty=True,
    )
    delta_g = float(result["Delta_f"]) * kbt_kcal_mol
    uncertainty = float(result["dDelta_f"]) * kbt_kcal_mol
    if not np.all(np.isfinite((delta_g, uncertainty))):
        raise ValueError("PyMBAR EXP returned non-finite outputs.")
    return {
        "delta_g_kcal_mol": delta_g,
        "uncertainty_kcal_mol": uncertainty,
        **_weight_diagnostics(dimensionless_difference),
    }


def analyze_one_sided_perturbation(
    energy_difference_replicates_kcal_mol: Sequence[
        Sequence[float] | np.ndarray
    ],
    *,
    temperature_kelvin: float,
    reference_equilibrium_claim: bool,
    detect_equilibration: bool = True,
    equilibration_nskip: int = 1,
    minimum_independent_replicates: int = 2,
    minimum_uncorrelated_samples_per_replicate: int = 20,
    minimum_effective_sample_count: float = 20.0,
    minimum_effective_sample_fraction: float = 0.10,
    maximum_normalized_weight: float = 0.20,
    maximum_replicate_difference_kcal_mol: float = 0.50,
) -> dict[str, Any]:
    r"""Estimate a reference-to-target correction with the Zwanzig equation.

    .. math::

       \Delta F_{\mathrm{ref}\rightarrow\mathrm{target}}
       = -RT\ln\left\langle
       \exp[-\beta(U_{\mathrm{target}}-U_{\mathrm{ref}})]
       \right\rangle_{\mathrm{ref}}.

    Passing the numerical gates does not establish that the reference
    trajectories cover every target-relevant configuration.  Bidirectional
    sampling or intermediate Hamiltonians remain stronger validation when
    target forces are available.
    """

    replicates = _replicates(energy_difference_replicates_kcal_mol)
    if not isinstance(reference_equilibrium_claim, bool):
        raise ValueError("reference_equilibrium_claim must be boolean.")
    if not isinstance(detect_equilibration, bool):
        raise ValueError("detect_equilibration must be boolean.")
    temperature = _positive_float("temperature_kelvin", temperature_kelvin)
    equilibration_nskip = _positive_integer(
        "equilibration_nskip",
        equilibration_nskip,
    )
    minimum_independent_replicates = _positive_integer(
        "minimum_independent_replicates",
        minimum_independent_replicates,
    )
    minimum_uncorrelated_samples_per_replicate = _positive_integer(
        "minimum_uncorrelated_samples_per_replicate",
        minimum_uncorrelated_samples_per_replicate,
    )
    minimum_effective_sample_count = _positive_float(
        "minimum_effective_sample_count",
        minimum_effective_sample_count,
    )
    minimum_effective_sample_fraction = _fraction(
        "minimum_effective_sample_fraction",
        minimum_effective_sample_fraction,
    )
    maximum_normalized_weight = _fraction(
        "maximum_normalized_weight",
        maximum_normalized_weight,
    )
    maximum_replicate_difference_kcal_mol = _positive_float(
        "maximum_replicate_difference_kcal_mol",
        maximum_replicate_difference_kcal_mol,
    )

    pymbar, timeseries = _load_pymbar()
    beta = 1.0 / (R_KCAL_PER_MOL_K * temperature)
    kbt = 1.0 / beta
    selected_replicates: list[np.ndarray] = []
    replicate_diagnostics: list[dict[str, Any]] = []
    replicate_results: list[dict[str, Any]] = []
    for index, values in enumerate(replicates):
        dimensionless = values * beta
        if detect_equilibration:
            start, inefficiency, estimated_effective = (
                timeseries.detect_equilibration(
                    dimensionless,
                    nskip=equilibration_nskip,
                )
            )
        else:
            start = 0
            inefficiency = timeseries.statistical_inefficiency(dimensionless)
            estimated_effective = len(dimensionless) / inefficiency
        start = int(start)
        inefficiency = float(inefficiency)
        estimated_effective = float(estimated_effective)
        if (
            start < 0
            or start >= len(dimensionless)
            or not np.isfinite(inefficiency)
            or inefficiency < 1.0
            or not np.isfinite(estimated_effective)
            or estimated_effective <= 0.0
        ):
            raise ValueError("PyMBAR returned invalid equilibration diagnostics.")
        relative = np.asarray(
            timeseries.subsample_correlated_data(
                dimensionless[start:],
                g=inefficiency,
            ),
            dtype=np.int64,
        )
        indices = relative + start
        if (
            indices.ndim != 1
            or len(indices) < 2
            or np.any(indices < start)
            or np.any(indices >= len(dimensionless))
        ):
            raise ValueError(
                "PyMBAR returned fewer than two valid decorrelated samples."
            )
        selected = dimensionless[indices]
        selected_replicates.append(selected)
        result = _exp_result(
            pymbar,
            selected,
            kbt_kcal_mol=kbt,
        )
        replicate_results.append(
            {
                "replicate_index": index,
                **result,
            }
        )
        replicate_diagnostics.append(
            {
                "replicate_index": index,
                "input_sample_count": int(len(values)),
                "equilibration_start_index": start,
                "statistical_inefficiency": inefficiency,
                "estimated_effective_sample_count": estimated_effective,
                "selected_sample_count": int(len(selected)),
                "selected_indices": indices.tolist(),
            }
        )

    combined_values = np.concatenate(selected_replicates)
    combined = _exp_result(
        pymbar,
        combined_values,
        kbt_kcal_mol=kbt,
    )
    replicate_estimates = np.asarray(
        [result["delta_g_kcal_mol"] for result in replicate_results],
        dtype=np.float64,
    )
    replicate_range = (
        float(np.ptp(replicate_estimates)) if len(replicate_estimates) > 1 else 0.0
    )
    checks = {
        "reference_equilibrium_claim": reference_equilibrium_claim,
        "minimum_independent_replicates": (
            len(replicates) >= minimum_independent_replicates
        ),
        "minimum_uncorrelated_samples_per_replicate": (
            min(item["selected_sample_count"] for item in replicate_diagnostics)
            >= minimum_uncorrelated_samples_per_replicate
        ),
        "minimum_effective_sample_count": (
            combined["effective_sample_count"] >= minimum_effective_sample_count
        ),
        "minimum_effective_sample_fraction": (
            combined["effective_sample_fraction"]
            >= minimum_effective_sample_fraction
        ),
        "maximum_normalized_weight": (
            combined["maximum_normalized_weight"] <= maximum_normalized_weight
        ),
        "maximum_replicate_difference": (
            replicate_range <= maximum_replicate_difference_kcal_mol
        ),
    }
    numerical_checks = {
        key: value
        for key, value in checks.items()
        if key != "reference_equilibrium_claim"
    }
    return {
        "schema_version": 1,
        "estimator": {
            "name": "PyMBAR EXP / Zwanzig one-sided perturbation",
            "pymbar_version": str(getattr(pymbar, "__version__", "unknown")),
            "handwritten_estimator": False,
        },
        "formula": (
            "DeltaF=-RT*ln(<exp[-beta*(U_target-U_reference)]>_reference)"
        ),
        "temperature_kelvin": temperature,
        "kbt_kcal_mol": kbt,
        "combined": combined,
        "replicates": replicate_results,
        "replicate_diagnostics": replicate_diagnostics,
        "replicate_estimate_range_kcal_mol": replicate_range,
        "gates": {
            "requirements": {
                "reference_equilibrium_claim": True,
                "minimum_independent_replicates": minimum_independent_replicates,
                "minimum_uncorrelated_samples_per_replicate": (
                    minimum_uncorrelated_samples_per_replicate
                ),
                "minimum_effective_sample_count": minimum_effective_sample_count,
                "minimum_effective_sample_fraction": (
                    minimum_effective_sample_fraction
                ),
                "maximum_normalized_weight": maximum_normalized_weight,
                "maximum_replicate_difference_kcal_mol": (
                    maximum_replicate_difference_kcal_mol
                ),
            },
            "checks": checks,
            "numerical_gates_passed": all(numerical_checks.values()),
            "scientific_gates_passed": all(checks.values()),
        },
        "claim_boundary": {
            "target_ensemble_sampled": False,
            "bidirectional_validation_established": False,
            "phase_space_completeness_established": False,
            "chemical_accuracy_established": False,
            "product_promotion_decided_here": False,
        },
    }
