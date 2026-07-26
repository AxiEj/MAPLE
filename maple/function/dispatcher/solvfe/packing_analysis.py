from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import InitVar, dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from ase.units import kB

from .analysis import ReducedPotentialTable, diagnose_mbar
from .packing_contract import PackingSchedule
from .protocol import canonical_sha256


_R_KCAL_PER_MOL_K = 0.001987204258640831
_PACKING_ESTIMATE_TOKEN = object()


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


def soft_packing_result_hash(result: Mapping[str, Any]) -> str:
    """Recompute the canonical hash of a soft-packing result mapping."""

    return canonical_sha256(
        {
            "contract_id": "soft-packing-result-v3",
            "packing_analysis_input_sha256": result[
                "packing_analysis_input_sha256"
            ],
            "packing_schedule_sha256": result["packing_schedule_sha256"],
            "reduced_potential_table_sha256": result[
                "reduced_potential_table_sha256"
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
            "estimator_disagreement_standard_error_kcal_mol": result[
                "estimator_disagreement_standard_error_kcal_mol"
            ],
            "p0_estimator_difference_standard_error": result[
                "p0_estimator_difference_standard_error"
            ],
            "paired_estimator_covariance_kcal2_mol2": result[
                "paired_estimator_covariance_kcal2_mol2"
            ],
            "estimator_agreement_bootstrap": result[
                "estimator_agreement_bootstrap"
            ],
            "thresholds": result["thresholds"],
        }
    )


@dataclass(frozen=True)
class SoftPackingEstimate(Mapping[str, Any]):
    """Factory-sealed soft-packing estimate and diagnostics."""

    _values: Mapping[str, Any]
    analysis_input_hash: str
    content_hash: str
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _PACKING_ESTIMATE_TOKEN:
            raise ValueError(
                "SoftPackingEstimate must be constructed by "
                "estimate_soft_packing."
            )
        if self.analysis_input_hash != self._values.get(
            "packing_analysis_input_sha256"
        ):
            raise ValueError(
                "Soft-packing estimate does not match its analysis input."
            )
        if (
            self.content_hash != self._values.get("packing_result_sha256")
            or self.content_hash != soft_packing_result_hash(self._values)
        ):
            raise ValueError(
                "Soft-packing estimate content hash does not match its data."
            )

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def _paired_estimator_bootstrap(
    analysis_input: "PackingAnalysisInput",
    *,
    replicates: int,
    seed: int,
    rt_kcal_mol: float,
) -> dict[str, Any]:
    """State-stratified bootstrap for two correlated p0 estimators."""

    table = analysis_input.table
    target = analysis_input.target_state_index
    full = analysis_input.full_field_state_index
    state_counts = np.asarray(table.N_k, dtype=int)
    offsets = np.concatenate(([0], np.cumsum(state_counts)))
    rng = np.random.default_rng(int(seed))
    pymbar = _require_pymbar()
    free_energy_values = np.empty(replicates, dtype=float)
    expectation_values = np.empty(replicates, dtype=float)
    probability_free_energy = np.empty(replicates, dtype=float)
    probability_expectation = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_indices: list[np.ndarray] = []
        for state_index, count in enumerate(state_counts):
            start = int(offsets[state_index])
            stop = int(offsets[state_index + 1])
            sampled_indices.append(
                rng.integers(start, stop, size=int(count))
            )
        indices = np.concatenate(sampled_indices)
        estimator = pymbar.MBAR(
            table.u_kn[:, indices],
            state_counts,
            verbose=False,
        )
        delta = estimator.compute_free_energy_differences(
            compute_uncertainty=False
        )
        delta_f = float(delta["Delta_f"][target, full])
        empty_weights = np.exp(
            analysis_input.log_empty_weights[indices]
        )
        expectation = estimator.compute_expectations(
            empty_weights,
            output="averages",
            state_dependent=False,
            compute_uncertainty=False,
        )
        p_free = math.exp(-delta_f)
        p_expectation = float(expectation["mu"][target])
        probability_free_energy[replicate] = p_free
        probability_expectation[replicate] = p_expectation
        free_energy_values[replicate] = delta_f * rt_kcal_mol
        expectation_values[replicate] = (
            -math.log(p_expectation) * rt_kcal_mol
        )
    paired = np.column_stack(
        (free_energy_values, expectation_values)
    )
    covariance = np.cov(paired, rowvar=False, ddof=1)
    differences = free_energy_values - expectation_values
    probability_differences = (
        probability_free_energy - probability_expectation
    )
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "paired_estimates_sha256": _array_sha256(paired),
        "covariance_kcal2_mol2": float(covariance[0, 1]),
        "difference_standard_error_kcal_mol": float(
            np.std(differences, ddof=1)
        ),
        "p0_difference_standard_error": float(
            np.std(probability_differences, ddof=1)
        ),
    }


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
    estimator_agreement_bootstrap_replicates: int = 64,
    estimator_agreement_bootstrap_seed: int = 20260726,
) -> SoftPackingEstimate:
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
    if (
        isinstance(estimator_agreement_bootstrap_replicates, (bool, np.bool_))
        or int(estimator_agreement_bootstrap_replicates)
        != estimator_agreement_bootstrap_replicates
        or int(estimator_agreement_bootstrap_replicates) < 32
    ):
        raise ValueError(
            "estimator_agreement_bootstrap_replicates must be an integer "
            "of at least 32."
        )
    if (
        isinstance(estimator_agreement_bootstrap_seed, (bool, np.bool_))
        or int(estimator_agreement_bootstrap_seed)
        != estimator_agreement_bootstrap_seed
        or int(estimator_agreement_bootstrap_seed) < 0
    ):
        raise ValueError(
            "estimator_agreement_bootstrap_seed must be a non-negative "
            "integer."
        )

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
    bootstrap = _paired_estimator_bootstrap(
        analysis_input,
        replicates=int(estimator_agreement_bootstrap_replicates),
        seed=int(estimator_agreement_bootstrap_seed),
        rt_kcal_mol=float(kcal_per_dimensionless),
    )

    failure_codes = list(diagnostics["failure_codes"])
    if packing_se_kcal > float(packing_se_kcal_max):
        failure_codes.append("PACKING_STANDARD_ERROR_TOO_HIGH")
    combined_estimator_se = float(
        bootstrap["difference_standard_error_kcal_mol"]
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
        "p0_estimator_difference_standard_error": float(
            bootstrap["p0_difference_standard_error"]
        ),
        "paired_estimator_covariance_kcal2_mol2": float(
            bootstrap["covariance_kcal2_mol2"]
        ),
        "estimator_agreement_bootstrap": bootstrap,
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
    return SoftPackingEstimate(
        _values=MappingProxyType(result),
        analysis_input_hash=analysis_input.content_hash,
        content_hash=result["packing_result_sha256"],
        _factory_token=_PACKING_ESTIMATE_TOKEN,
    )


__all__ = [
    "PackingAnalysisInput",
    "SoftPackingEstimate",
    "estimate_soft_packing",
    "soft_packing_result_hash",
]
