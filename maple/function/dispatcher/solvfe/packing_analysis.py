from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase.units import kB

from .analysis import ReducedPotentialTable, diagnose_mbar
from .packing_contract import PackingSchedule
from .protocol import canonical_sha256


_R_KCAL_PER_MOL_K = 0.001987204258640831


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _require_pymbar() -> Any:
    try:
        import pymbar
    except ImportError as exc:
        raise RuntimeError(
            "Route A packing analysis requires the optional 'pymbar' "
            "dependency."
        ) from exc
    return pymbar


def soft_packing_result_hash(result: dict[str, Any]) -> str:
    """Recompute the canonical hash of a soft-packing result mapping."""

    return canonical_sha256(
        {
            "contract_id": "soft-packing-result-v3",
            "packing_analysis_input_sha256": result[
                "packing_analysis_input_sha256"
            ],
            "status": result["status"],
            "failure_codes": result["failure_codes"],
            "p0_from_free_energy": result["p0_from_free_energy"],
            "p0_from_free_energy_standard_error": result[
                "p0_from_free_energy_standard_error"
            ],
            "p0_from_reweighted_expectation": result[
                "p0_from_reweighted_expectation"
            ],
            "p0_reweighted_expectation_standard_error": result[
                "p0_reweighted_expectation_standard_error"
            ],
            "packing_free_energy_kcal_mol": result[
                "packing_free_energy_kcal_mol"
            ],
            "packing_free_energy_standard_error_kcal_mol": result[
                "packing_free_energy_standard_error_kcal_mol"
            ],
            "estimator_disagreement_kcal_mol": result[
                "estimator_disagreement_kcal_mol"
            ],
            "estimator_disagreement_allowed_kcal_mol": result[
                "estimator_disagreement_allowed_kcal_mol"
            ],
            "thresholds": result["thresholds"],
        }
    )


@dataclass(frozen=True)
class PackingAnalysisInput:
    """A reduced-potential table proven to implement one packing schedule."""

    table: ReducedPotentialTable
    schedule: PackingSchedule
    log_empty_weights: np.ndarray
    content_hash: str

    @classmethod
    def create(
        cls,
        *,
        table: ReducedPotentialTable,
        schedule: PackingSchedule,
        log_empty_weights: np.ndarray,
    ) -> "PackingAnalysisInput":
        if not isinstance(table, ReducedPotentialTable):
            raise ValueError("table must be a ReducedPotentialTable.")
        if not isinstance(schedule, PackingSchedule):
            raise ValueError("schedule must be a PackingSchedule.")
        if table.row_labels != schedule.row_labels:
            raise ValueError(
                "Packing table row labels do not match the schedule."
            )
        if table.boundary_conditions != schedule.boundary_conditions:
            raise ValueError(
                "Packing table boundary conditions do not match the schedule."
            )
        if table.measure_id != schedule.conditioning_measure_id:
            raise ValueError(
                "Packing table conditioning measure does not match the schedule."
            )
        expected_beta = 1.0 / (kB * float(schedule.temperature_k))
        if not math.isclose(
            float(table.beta),
            expected_beta,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Packing table beta does not match the schedule temperature."
            )

        log_weights = np.asarray(log_empty_weights, dtype=float)
        if (
            log_weights.shape != (table.u_kn.shape[1],)
            or not np.all(np.isfinite(log_weights))
            or np.any(log_weights > 0.0)
        ):
            raise ValueError(
                "log_empty_weights must contain one finite non-positive "
                "value per sampled frame."
            )
        full_bias = -log_weights
        target_row = table.u_kn[schedule.target_state_index]
        for row_index, state in enumerate(schedule.states):
            expected_row = (
                target_row + float(state.bias_scale) * full_bias
            )
            if not np.allclose(
                table.u_kn[row_index],
                expected_row,
                rtol=1.0e-12,
                atol=1.0e-10,
            ):
                raise ValueError(
                    "Packing table bias relation does not match the explicit "
                    f"schedule at row {row_index}."
                )

        immutable = np.array(
            log_weights,
            dtype=float,
            copy=True,
            order="C",
        )
        immutable.setflags(write=False)
        content_hash = canonical_sha256(
            {
                "contract_id": "packing-analysis-input-v3",
                "reduced_potential_table_hash": table.state_hash,
                "packing_schedule_hash": schedule.content_hash,
                "log_empty_weights_sha256": _array_sha256(immutable),
            }
        )
        return cls(
            table=table,
            schedule=schedule,
            log_empty_weights=immutable,
            content_hash=content_hash,
        )

    @property
    def target_state_index(self) -> int:
        return self.schedule.target_state_index

    @property
    def full_field_state_index(self) -> int:
        return self.schedule.full_field_state_index


def estimate_soft_packing(
    analysis_input: PackingAnalysisInput,
    *,
    overlap_min: float,
    effective_samples_min: float,
    bar_disagreement_kcal_max: float,
    packing_se_kcal_max: float,
    estimator_agreement_kcal_max: float,
    estimator_agreement_z_max: float,
) -> dict[str, Any]:
    """Estimate ``-RT ln p0`` from one complementary soft empty field."""

    if not isinstance(analysis_input, PackingAnalysisInput):
        raise ValueError(
            "analysis_input must be a PackingAnalysisInput."
        )
    for name, value, allow_zero in (
        ("overlap_min", overlap_min, True),
        ("effective_samples_min", effective_samples_min, False),
        (
            "bar_disagreement_kcal_max",
            bar_disagreement_kcal_max,
            True,
        ),
        ("packing_se_kcal_max", packing_se_kcal_max, False),
        (
            "estimator_agreement_kcal_max",
            estimator_agreement_kcal_max,
            True,
        ),
        ("estimator_agreement_z_max", estimator_agreement_z_max, False),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not math.isfinite(float(value))
            or (float(value) < 0.0 if allow_zero else float(value) <= 0.0)
        ):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}.")

    table = analysis_input.table
    schedule = analysis_input.schedule
    kcal_per_dimensionless = (
        _R_KCAL_PER_MOL_K * float(schedule.temperature_k)
    )
    diagnostics = diagnose_mbar(
        table,
        overlap_min=float(overlap_min),
        effective_samples_min=float(effective_samples_min),
        bar_disagreement_kcal_max=float(
            bar_disagreement_kcal_max
        ),
        kcal_per_dimensionless=float(kcal_per_dimensionless),
    )
    mbar_result = diagnostics["mbar"]
    target = schedule.target_state_index
    full = schedule.full_field_state_index
    delta_f = float(mbar_result["delta_f"][target, full])
    delta_f_se = float(
        mbar_result["delta_f_uncertainty"][target, full]
    )
    probability_from_free_energy = math.exp(-delta_f)

    pymbar = _require_pymbar()
    estimator = pymbar.MBAR(
        table.u_kn,
        np.asarray(table.N_k, dtype=int),
        verbose=False,
    )
    empty_weights = np.exp(analysis_input.log_empty_weights)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="divide by zero encountered in log",
            category=RuntimeWarning,
            module=r"pymbar\.mbar",
        )
        expectation = estimator.compute_expectations(
            empty_weights,
            output="averages",
            state_dependent=False,
            compute_uncertainty=True,
        )
    probability_from_expectation = float(expectation["mu"][target])
    probability_expectation_se = float(expectation["sigma"][target])
    if (
        not math.isfinite(delta_f)
        or not math.isfinite(delta_f_se)
        or delta_f_se < 0.0
        or not 0.0 < probability_from_free_energy <= 1.0
        or not math.isfinite(probability_from_expectation)
        or not 0.0 < probability_from_expectation <= 1.0
        or not math.isfinite(probability_expectation_se)
        or probability_expectation_se < 0.0
    ):
        raise RuntimeError(
            "PACKING_ESTIMATE_INVALID: soft-field MBAR returned a "
            "nonphysical p0 estimate."
        )

    packing_kcal = delta_f * float(kcal_per_dimensionless)
    packing_se_kcal = delta_f_se * float(kcal_per_dimensionless)
    expectation_packing_kcal = (
        -math.log(probability_from_expectation)
        * float(kcal_per_dimensionless)
    )
    expectation_packing_se_kcal = (
        probability_expectation_se
        / probability_from_expectation
        * float(kcal_per_dimensionless)
    )
    disagreement = abs(
        packing_kcal - expectation_packing_kcal
    )

    failure_codes = list(diagnostics["failure_codes"])
    if packing_se_kcal > float(packing_se_kcal_max):
        failure_codes.append("PACKING_STANDARD_ERROR_TOO_HIGH")
    combined_estimator_se = math.hypot(
        packing_se_kcal,
        expectation_packing_se_kcal,
    )
    estimator_agreement_allowed = max(
        float(estimator_agreement_kcal_max),
        float(estimator_agreement_z_max) * combined_estimator_se,
    )
    if disagreement > estimator_agreement_allowed:
        failure_codes.append("PACKING_ESTIMATOR_DISAGREEMENT")
    result = {
        "status": "passed" if not failure_codes else "failed",
        "failure_codes": failure_codes,
        "packing_analysis_input_sha256": analysis_input.content_hash,
        "packing_schedule_sha256": schedule.content_hash,
        "reduced_potential_table_sha256": table.state_hash,
        "target_state_index": target,
        "target_state_label": schedule.target_state.label,
        "full_field_state_index": full,
        "full_field_state_label": schedule.full_field_state.label,
        "delta_f_target_to_full": delta_f,
        "delta_f_standard_error": delta_f_se,
        "p0_from_free_energy": probability_from_free_energy,
        "p0_from_free_energy_standard_error": (
            probability_from_free_energy * delta_f_se
        ),
        "p0_from_reweighted_expectation": (
            probability_from_expectation
        ),
        "p0_reweighted_expectation_standard_error": (
            probability_expectation_se
        ),
        "packing_free_energy_kcal_mol": packing_kcal,
        "packing_free_energy_standard_error_kcal_mol": (
            packing_se_kcal
        ),
        "expectation_packing_free_energy_kcal_mol": (
            expectation_packing_kcal
        ),
        "expectation_packing_standard_error_kcal_mol": (
            expectation_packing_se_kcal
        ),
        "estimator_disagreement_kcal_mol": disagreement,
        "estimator_disagreement_standard_error_kcal_mol": (
            combined_estimator_se
        ),
        "estimator_disagreement_allowed_kcal_mol": (
            estimator_agreement_allowed
        ),
        "thresholds": {
            "packing_se_kcal_max": float(packing_se_kcal_max),
            "estimator_agreement_kcal_max": float(
                estimator_agreement_kcal_max
            ),
            "estimator_agreement_z_max": float(
                estimator_agreement_z_max
            ),
        },
        "mbar_diagnostics": diagnostics,
        "estimator": (
            "PyMBAR free-energy difference plus target-state "
            "reweighted soft-empty expectation"
        ),
    }
    result["packing_result_sha256"] = soft_packing_result_hash(result)
    return result


__all__ = [
    "PackingAnalysisInput",
    "estimate_soft_packing",
    "soft_packing_result_hash",
]
