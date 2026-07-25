"""Label-free discrete-conformer analysis for the Route 1 additive potential.

This module evaluates a finite equal-measure state set. It is intentionally
not a replacement for equilibrated sampling, basin integration, or MBAR.
Mobley, Dill, and Chodera showed why conformational changes cannot generally
be omitted from small-molecule implicit-solvent free energies
(J. Phys. Chem. B 2008, DOI: 10.1021/jp0764384).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .mbar import R_KCAL_PER_MOL_K


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def _normalized_weights(logits: np.ndarray) -> np.ndarray:
    return np.exp(logits - _logsumexp(logits))


def _effective_count(weights: np.ndarray) -> float:
    return float(1.0 / np.square(weights).sum())


def _validate_fraction(name: str, value: float, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    value = float(value)
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not lower_ok or value > 1.0:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {interval}.")
    return value


def _validate_inputs(
    gas_energy_kcal_mol: Sequence[float],
    solvent_correction_kcal_mol: Sequence[float],
    temperature_kelvin: float,
    state_ids: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    gas = np.asarray(gas_energy_kcal_mol, dtype=np.float64)
    solvent = np.asarray(solvent_correction_kcal_mol, dtype=np.float64)
    if gas.ndim != 1 or solvent.ndim != 1 or len(gas) == 0 or len(solvent) == 0:
        raise ValueError("Energy arrays must be non-empty one-dimensional sequences.")
    if len(gas) != len(solvent):
        raise ValueError("Gas and solvent energy arrays must have equal length.")
    if not np.isfinite(gas).all() or not np.isfinite(solvent).all():
        raise ValueError("Gas and solvent energies must be finite.")
    if (
        isinstance(temperature_kelvin, bool)
        or not np.isfinite(temperature_kelvin)
        or float(temperature_kelvin) <= 0.0
    ):
        raise ValueError("temperature_kelvin must be finite and positive.")

    if state_ids is None:
        identifiers = [str(index) for index in range(len(gas))]
    else:
        identifiers = [str(value) for value in state_ids]
        if len(identifiers) != len(gas):
            raise ValueError("state_ids and energy arrays must have equal length.")
        if any(not value for value in identifiers):
            raise ValueError("state_ids must be non-empty.")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("state_ids must be unique.")
    return gas, solvent, identifiers, float(temperature_kelvin)


def analyze_discrete_conformer_ensemble(
    *,
    gas_energy_kcal_mol: Sequence[float],
    solvent_correction_kcal_mol: Sequence[float],
    temperature_kelvin: float,
    state_ids: Sequence[str] | None = None,
    minimum_effective_conformer_count: float = 2.0,
    maximum_dominant_weight: float = 0.95,
    minimum_distribution_overlap: float = 0.10,
) -> dict:
    """Evaluate an equal-measure discrete conformer partition-function ratio.

    The estimator is

    ``-RT log(sum_i exp[-beta(E_i + W_i)] / sum_i exp[-beta E_i])``,

    where ``E_i`` is a gas-phase MLIP energy and ``W_i`` is the fixed-charge
    Route 1 solvent correction on the same state. Only relative gas energies
    contribute. The returned diagnostics can reject a numerically concentrated
    state set, but cannot prove that the submitted conformers span the
    continuous gas and solution ensembles.
    """

    gas, solvent, identifiers, temperature = _validate_inputs(
        gas_energy_kcal_mol,
        solvent_correction_kcal_mol,
        temperature_kelvin,
        state_ids,
    )
    if (
        isinstance(minimum_effective_conformer_count, bool)
        or not np.isfinite(minimum_effective_conformer_count)
        or float(minimum_effective_conformer_count) < 1.0
    ):
        raise ValueError(
            "minimum_effective_conformer_count must be finite and at least 1."
        )
    minimum_effective_conformer_count = float(
        minimum_effective_conformer_count
    )
    maximum_dominant_weight = _validate_fraction(
        "maximum_dominant_weight",
        maximum_dominant_weight,
        allow_zero=False,
    )
    minimum_distribution_overlap = _validate_fraction(
        "minimum_distribution_overlap",
        minimum_distribution_overlap,
        allow_zero=True,
    )

    rt = R_KCAL_PER_MOL_K * temperature
    gas_zero = float(np.min(gas))
    relative_gas = gas - gas_zero
    gas_logits = -relative_gas / rt
    solution_logits = -(relative_gas + solvent) / rt
    gas_log_partition = _logsumexp(gas_logits)
    solution_log_partition = _logsumexp(solution_logits)
    delta_g = -rt * (solution_log_partition - gas_log_partition)
    gas_weights = _normalized_weights(gas_logits)
    solution_weights = _normalized_weights(solution_logits)

    gas_log_weights = gas_logits - gas_log_partition
    perturbation_delta_g = -rt * _logsumexp(
        gas_log_weights - solvent / rt
    )
    gas_effective = _effective_count(gas_weights)
    solution_effective = _effective_count(solution_weights)
    gas_maximum = float(np.max(gas_weights))
    solution_maximum = float(np.max(solution_weights))
    distribution_overlap = float(
        np.minimum(gas_weights, solution_weights).sum()
    )
    bhattacharyya = float(np.sqrt(gas_weights * solution_weights).sum())
    gas_dominant = int(np.argmax(gas_weights))
    solution_dominant = int(np.argmax(solution_weights))
    endpoint_bounds_satisfied = bool(
        float(np.min(solvent)) - 1.0e-12
        <= delta_g
        <= float(np.max(solvent)) + 1.0e-12
    )

    checks = {
        "minimum_state_count": len(gas) >= 2,
        "minimum_gas_effective_conformer_count": (
            gas_effective >= minimum_effective_conformer_count
        ),
        "minimum_solution_effective_conformer_count": (
            solution_effective >= minimum_effective_conformer_count
        ),
        "maximum_gas_weight": gas_maximum <= maximum_dominant_weight,
        "maximum_solution_weight": solution_maximum <= maximum_dominant_weight,
        "minimum_distribution_overlap": (
            distribution_overlap >= minimum_distribution_overlap
        ),
        "endpoint_bounds": endpoint_bounds_satisfied,
    }

    return {
        "schema_version": 1,
        "estimator": "equal-measure discrete-conformer partition ratio",
        "formula": (
            "-RT*ln(sum_i exp[-beta*(E_MLIP,gas_i+W_i)]/"
            "sum_i exp[-beta*E_MLIP,gas_i])"
        ),
        "temperature_kelvin": temperature,
        "rt_kcal_mol": rt,
        "state_count": int(len(gas)),
        "state_ids": identifiers,
        "gas_energy_zero_kcal_mol": gas_zero,
        "gas_relative_energy_kcal_mol": relative_gas.tolist(),
        "solvent_correction_kcal_mol": solvent.tolist(),
        "delta_g_discrete_kcal_mol": float(delta_g),
        "uncertainty_kcal_mol": None,
        "gas_weights": gas_weights.tolist(),
        "solution_weights": solution_weights.tolist(),
        "gas_effective_conformer_count": gas_effective,
        "solution_effective_conformer_count": solution_effective,
        "gas_maximum_weight": gas_maximum,
        "solution_maximum_weight": solution_maximum,
        "gas_dominant_state_index": gas_dominant,
        "solution_dominant_state_index": solution_dominant,
        "gas_dominant_state_id": identifiers[gas_dominant],
        "solution_dominant_state_id": identifiers[solution_dominant],
        "distribution_overlap": distribution_overlap,
        "total_variation_distance": 1.0 - distribution_overlap,
        "bhattacharyya_coefficient": bhattacharyya,
        "identities": {
            "endpoint_bounds_satisfied": endpoint_bounds_satisfied,
            "partition_ratio_closure_abs_kcal_mol": abs(
                float(delta_g - perturbation_delta_g)
            ),
        },
        "gates": {
            "requirements": {
                "minimum_state_count": 2,
                "minimum_effective_conformer_count": (
                    minimum_effective_conformer_count
                ),
                "maximum_dominant_weight": maximum_dominant_weight,
                "minimum_distribution_overlap": minimum_distribution_overlap,
            },
            "checks": checks,
            "ensemble_diagnostic_passed": all(checks.values()),
        },
        "route_contract": {
            "gas_phase_mm_energy": False,
            "hydration_label_residual": False,
            "mlip_retraining": False,
            "fixed_charge_solvent_correction": True,
        },
        "claim_boundary": {
            "discrete_conformer_diagnostic_only": True,
            "public_solvfe_eligible": False,
            "hydration_free_energy_claim": False,
            "equal_basin_measure_assumed": True,
            "basin_volumes_included": False,
            "vibrational_free_energies_included": False,
            "standard_state_correction_included": False,
            "alchemical_path_sampled": False,
            "experimental_labels_used": False,
            "reason": (
                "Weight diagnostics cannot prove conformer-set completeness or "
                "replace basin integration, equilibrated sampling, uncertainty, "
                "and overlap validation."
            ),
        },
        "literature": [
            {
                "citation": (
                    "Mobley, Dill, and Chodera, J. Phys. Chem. B 2008, "
                    "112, 938-946"
                ),
                "doi": "10.1021/jp0764384",
            }
        ],
    }
