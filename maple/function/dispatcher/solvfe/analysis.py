from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from .protocol import canonical_sha256


def _require_pymbar() -> Any:
    try:
        import pymbar
    except ImportError as exc:
        raise RuntimeError(
            "Route A free-energy analysis requires the optional 'pymbar' "
            "dependency."
        ) from exc
    return pymbar


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ReducedPotentialTable:
    """Immutable, provenance-bound reduced potentials for MBAR/BAR.

    Columns are ordered by sampled state: the first ``N_k[0]`` columns were
    sampled from row 0, followed by the columns sampled from row 1, and so on.
    Every entry in ``u_kn`` is dimensionless (beta times potential energy).
    """

    u_kn: np.ndarray
    N_k: tuple[int, ...]
    row_labels: tuple[str, ...]
    frame_ids: tuple[str, ...]
    beta: float
    measure_id: str
    boundary_conditions: str
    units: str
    state_hashes: tuple[str, ...]
    state_hash: str

    @classmethod
    def create(
        cls,
        *,
        u_kn: np.ndarray,
        N_k: tuple[int, ...],
        row_labels: tuple[str, ...],
        frame_ids: tuple[str, ...],
        beta: float,
        measure_id: str,
        boundary_conditions: str,
        units: str = "dimensionless",
    ) -> "ReducedPotentialTable":
        if units != "dimensionless":
            raise ValueError(
                "Reduced potentials must use dimensionless units."
            )

        reduced = np.asarray(u_kn, dtype=float)
        if reduced.ndim != 2 or 0 in reduced.shape:
            raise ValueError("u_kn must be a non-empty two-dimensional array.")
        if not np.all(np.isfinite(reduced)):
            raise ValueError("u_kn contains non-finite reduced potentials.")

        state_count, frame_count = reduced.shape
        counts = tuple(int(value) for value in N_k)
        if (
            len(counts) != state_count
            or any(
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) <= 0
                for value in N_k
            )
            or sum(counts) != frame_count
        ):
            raise ValueError(
                "N_k must contain one positive integer per state and sum "
                "to the number of u_kn columns."
            )

        labels = tuple(row_labels)
        frames = tuple(frame_ids)
        cls._validate_identifiers(labels, state_count, "row_labels")
        cls._validate_identifiers(frames, frame_count, "frame_ids")

        if (
            isinstance(beta, (bool, np.bool_))
            or not math.isfinite(float(beta))
            or float(beta) <= 0.0
        ):
            raise ValueError("beta must be finite and positive.")
        if not isinstance(measure_id, str) or not measure_id.strip():
            raise ValueError("measure_id must be a non-empty string.")
        if (
            not isinstance(boundary_conditions, str)
            or not boundary_conditions.strip()
        ):
            raise ValueError(
                "boundary_conditions must be a non-empty string."
            )

        immutable = np.array(reduced, dtype=float, order="C", copy=True)
        immutable.setflags(write=False)
        common = {
            "N_k": list(counts),
            "frame_ids": list(frames),
            "beta": float(beta),
            "measure_id": measure_id,
            "boundary_conditions": boundary_conditions,
            "units": units,
        }
        state_hashes = tuple(
            canonical_sha256(
                {
                    **common,
                    "row_index": index,
                    "row_label": labels[index],
                    "row_sha256": _array_sha256(immutable[index]),
                }
            )
            for index in range(state_count)
        )
        state_hash = canonical_sha256(
            {
                **common,
                "row_labels": list(labels),
                "state_hashes": list(state_hashes),
                "u_kn_sha256": _array_sha256(immutable),
            }
        )
        return cls(
            u_kn=immutable,
            N_k=counts,
            row_labels=labels,
            frame_ids=frames,
            beta=float(beta),
            measure_id=measure_id,
            boundary_conditions=boundary_conditions,
            units=units,
            state_hashes=state_hashes,
            state_hash=state_hash,
        )

    @staticmethod
    def _validate_identifiers(
        values: tuple[str, ...],
        expected_count: int,
        name: str,
    ) -> None:
        if (
            len(values) != expected_count
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(
                f"{name} must contain {expected_count} unique non-empty strings."
            )


def estimate_mbar(table: ReducedPotentialTable) -> dict[str, Any]:
    """Estimate all-state dimensionless free-energy differences with MBAR."""

    pymbar = _require_pymbar()
    estimator = pymbar.MBAR(
        table.u_kn,
        np.asarray(table.N_k, dtype=int),
        verbose=False,
    )
    free_energies = estimator.compute_free_energy_differences()
    overlap = estimator.compute_overlap()
    effective_sample_numbers = estimator.compute_effective_sample_number()
    return {
        "method": "PyMBAR.MBAR",
        "reduced_potential_table_sha256": table.state_hash,
        "state_hashes": table.state_hashes,
        "row_labels": table.row_labels,
        "delta_f": np.asarray(free_energies["Delta_f"], dtype=float),
        "delta_f_uncertainty": np.asarray(
            free_energies["dDelta_f"],
            dtype=float,
        ),
        "overlap_matrix": np.asarray(overlap["matrix"], dtype=float),
        "overlap_scalar": float(overlap["scalar"]),
        "overlap_eigenvalues": np.asarray(
            overlap["eigenvalues"],
            dtype=float,
        ),
        "effective_sample_numbers": np.asarray(
            effective_sample_numbers,
            dtype=float,
        ),
    }


def estimate_adjacent_bar(
    table: ReducedPotentialTable,
) -> list[dict[str, Any]]:
    """Estimate each adjacent-state difference using bidirectional BAR."""

    pymbar = _require_pymbar()
    boundaries = np.concatenate(([0], np.cumsum(table.N_k)))
    estimates: list[dict[str, Any]] = []
    for index in range(len(table.N_k) - 1):
        forward_slice = slice(boundaries[index], boundaries[index + 1])
        reverse_slice = slice(boundaries[index + 1], boundaries[index + 2])
        forward = (
            table.u_kn[index + 1, forward_slice]
            - table.u_kn[index, forward_slice]
        )
        reverse = (
            table.u_kn[index, reverse_slice]
            - table.u_kn[index + 1, reverse_slice]
        )

        if (
            np.allclose(forward, forward[0], rtol=0.0, atol=1.0e-14)
            and np.allclose(reverse, reverse[0], rtol=0.0, atol=1.0e-14)
            and math.isclose(
                float(forward[0]),
                -float(reverse[0]),
                rel_tol=0.0,
                abs_tol=1.0e-14,
            )
        ):
            delta_f = 0.5 * (float(forward[0]) - float(reverse[0]))
            uncertainty = 0.0
            implementation = "analytic-degenerate-BAR"
        else:
            result = pymbar.other_estimators.bar(forward, reverse)
            delta_f = float(result["Delta_f"])
            uncertainty = float(result["dDelta_f"])
            implementation = "pymbar.other_estimators.bar"

        estimates.append(
            {
                "from": table.row_labels[index],
                "to": table.row_labels[index + 1],
                "from_state_sha256": table.state_hashes[index],
                "to_state_sha256": table.state_hashes[index + 1],
                "reduced_potential_table_sha256": table.state_hash,
                "delta_f": delta_f,
                "delta_f_uncertainty": uncertainty,
                "forward_sample_count": int(len(forward)),
                "reverse_sample_count": int(len(reverse)),
                "implementation": implementation,
            }
        )
    return estimates


def subsample_state_series(values: np.ndarray) -> dict[str, Any]:
    """Detect equilibration and return state-local uncorrelated indices."""

    pymbar = _require_pymbar()
    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or len(series) < 3:
        raise ValueError(
            "Timeseries subsampling requires a one-dimensional series with "
            "at least three observations."
        )
    if not np.all(np.isfinite(series)):
        raise ValueError("Timeseries contains non-finite observations.")

    start, statistical_inefficiency, estimated_count = (
        pymbar.timeseries.detect_equilibration(series)
    )
    start = int(start)
    statistical_inefficiency = float(statistical_inefficiency)
    local_indices = pymbar.timeseries.subsample_correlated_data(
        series[start:],
        g=statistical_inefficiency,
    )
    indices = sorted({start + int(index) for index in local_indices})
    if not indices:
        raise ValueError(
            "Timeseries analysis produced no equilibrated independent samples."
        )
    return {
        "equilibrated_start": start,
        "statistical_inefficiency": statistical_inefficiency,
        "effective_sample_count": len(indices),
        "estimated_effective_sample_count": float(estimated_count),
        "indices": indices,
    }


def diagnose_mbar(
    table: ReducedPotentialTable,
    *,
    overlap_min: float,
    effective_samples_min: float,
    bar_disagreement_kcal_max: float,
    kcal_per_dimensionless: float,
) -> dict[str, Any]:
    """Apply fail-closed overlap/ESS/BAR gates to one MBAR table."""

    for name, value, allow_zero in (
        ("overlap_min", overlap_min, True),
        ("effective_samples_min", effective_samples_min, False),
        (
            "bar_disagreement_kcal_max",
            bar_disagreement_kcal_max,
            True,
        ),
        ("kcal_per_dimensionless", kcal_per_dimensionless, False),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not math.isfinite(float(value))
            or (float(value) < 0.0 if allow_zero else float(value) <= 0.0)
        ):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}.")

    mbar = estimate_mbar(table)
    adjacent_bar = estimate_adjacent_bar(table)
    adjacent_overlaps = np.asarray(
        [
            min(
                mbar["overlap_matrix"][index, index + 1],
                mbar["overlap_matrix"][index + 1, index],
            )
            for index in range(len(table.N_k) - 1)
        ],
        dtype=float,
    )
    adjacent_disagreements = np.asarray(
        [
            abs(
                float(mbar["delta_f"][index, index + 1])
                - float(adjacent_bar[index]["delta_f"])
            )
            * float(kcal_per_dimensionless)
            for index in range(len(table.N_k) - 1)
        ],
        dtype=float,
    )
    minimum_overlap = (
        float(np.min(adjacent_overlaps)) if len(adjacent_overlaps) else 1.0
    )
    minimum_effective_samples = float(
        np.min(mbar["effective_sample_numbers"])
    )
    maximum_disagreement = (
        float(np.max(adjacent_disagreements))
        if len(adjacent_disagreements)
        else 0.0
    )

    failure_codes: list[str] = []
    if minimum_overlap < float(overlap_min):
        failure_codes.append("MBAR_OVERLAP_DISCONNECTED")
    if minimum_effective_samples < float(effective_samples_min):
        failure_codes.append("MBAR_ESS_TOO_LOW")
    if maximum_disagreement > float(bar_disagreement_kcal_max):
        failure_codes.append("BAR_MBAR_DISAGREEMENT")
    return {
        "status": "passed" if not failure_codes else "failed",
        "failure_codes": failure_codes,
        "minimum_adjacent_overlap": minimum_overlap,
        "adjacent_overlaps": adjacent_overlaps,
        "minimum_effective_sample_number": minimum_effective_samples,
        "effective_sample_numbers": mbar["effective_sample_numbers"],
        "maximum_adjacent_bar_mbar_disagreement_kcal_mol": (
            maximum_disagreement
        ),
        "adjacent_bar_mbar_disagreements_kcal_mol": (
            adjacent_disagreements
        ),
        "thresholds": {
            "overlap_min": float(overlap_min),
            "effective_samples_min": float(effective_samples_min),
            "bar_disagreement_kcal_max": float(
                bar_disagreement_kcal_max
            ),
        },
        "mbar": mbar,
        "adjacent_bar": adjacent_bar,
    }


def diagnose_independent_replicas(
    *,
    estimates_kcal_mol: tuple[float, ...],
    standard_errors_kcal_mol: tuple[float, ...],
    replica_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    per_replica_status: tuple[str, ...],
    minimum_replicas: int,
    pairwise_z_max: float,
) -> dict[str, Any]:
    """Apply fail-closed independent-replica agreement gates.

    Replica estimates are not pooled until every replica has passed its own
    diagnostics.  Pairwise agreement uses the standard error of a difference,
    ``sqrt(se_i**2 + se_j**2)``.
    """

    estimates = np.asarray(estimates_kcal_mol, dtype=float)
    errors = np.asarray(standard_errors_kcal_mol, dtype=float)
    count = len(estimates)
    if (
        errors.shape != (count,)
        or len(replica_ids) != count
        or len(seeds) != count
        or len(per_replica_status) != count
        or count == 0
    ):
        raise ValueError(
            "Replica diagnostics require equally sized, non-empty inputs."
        )
    if not np.all(np.isfinite(estimates)):
        raise ValueError("Replica estimates must be finite.")
    if not np.all(np.isfinite(errors)) or np.any(errors <= 0.0):
        raise ValueError("Replica standard errors must be finite and positive.")
    if (
        isinstance(minimum_replicas, (bool, np.bool_))
        or int(minimum_replicas) != minimum_replicas
        or int(minimum_replicas) < 2
    ):
        raise ValueError("minimum_replicas must be an integer of at least two.")
    if (
        isinstance(pairwise_z_max, (bool, np.bool_))
        or not math.isfinite(float(pairwise_z_max))
        or float(pairwise_z_max) <= 0.0
    ):
        raise ValueError("pairwise_z_max must be finite and positive.")
    if any(not isinstance(value, str) or not value for value in replica_ids):
        raise ValueError("replica_ids must be non-empty strings.")
    if any(value not in {"passed", "failed"} for value in per_replica_status):
        raise ValueError("per_replica_status values must be passed or failed.")
    if any(
        isinstance(seed, (bool, np.bool_)) or int(seed) != seed
        for seed in seeds
    ):
        raise ValueError("Replica seeds must be integers.")

    pairwise: list[dict[str, Any]] = []
    for left in range(count):
        for right in range(left + 1, count):
            difference = abs(float(estimates[left] - estimates[right]))
            combined_error = math.sqrt(
                float(errors[left] ** 2 + errors[right] ** 2)
            )
            pairwise.append(
                {
                    "left_replica_id": replica_ids[left],
                    "right_replica_id": replica_ids[right],
                    "absolute_difference_kcal_mol": difference,
                    "combined_standard_error_kcal_mol": combined_error,
                    "pairwise_z": difference / combined_error,
                }
            )
    maximum_z = max(
        (float(item["pairwise_z"]) for item in pairwise),
        default=0.0,
    )

    failure_codes: list[str] = []
    if count < int(minimum_replicas):
        failure_codes.append("REPLICA_COUNT_TOO_LOW")
    if (
        len(set(replica_ids)) != count
        or len(set(int(seed) for seed in seeds)) != count
    ):
        failure_codes.append("REPLICA_NOT_INDEPENDENT")
    if any(status != "passed" for status in per_replica_status):
        failure_codes.append("REPLICA_INTERNAL_GATE_FAILED")
    if maximum_z > float(pairwise_z_max):
        failure_codes.append("REPLICA_DISAGREEMENT")

    if failure_codes:
        pooled_estimate = None
        pooled_standard_error = None
    else:
        weights = 1.0 / errors**2
        pooled_estimate = float(np.sum(weights * estimates) / np.sum(weights))
        pooled_standard_error = float(math.sqrt(1.0 / np.sum(weights)))
    preimage = {
        "contract_id": "independent-replica-diagnostics-v1",
        "estimates_kcal_mol": estimates.tolist(),
        "standard_errors_kcal_mol": errors.tolist(),
        "replica_ids": list(replica_ids),
        "seeds": [int(seed) for seed in seeds],
        "per_replica_status": list(per_replica_status),
        "minimum_replicas": int(minimum_replicas),
        "pairwise_z_max": float(pairwise_z_max),
    }
    return {
        "status": "passed" if not failure_codes else "failed",
        "failure_codes": failure_codes,
        "replica_count": count,
        "minimum_replicas": int(minimum_replicas),
        "pairwise_z_max_threshold": float(pairwise_z_max),
        "maximum_pairwise_z": maximum_z,
        "pairwise": pairwise,
        "pooled_estimate_kcal_mol": pooled_estimate,
        "pooled_standard_error_kcal_mol": pooled_standard_error,
        "diagnostic_input_sha256": canonical_sha256(preimage),
    }


def estimate_packing_probability(
    table: ReducedPotentialTable,
    *,
    empty_indicator: np.ndarray,
    kcal_per_dimensionless: float,
    minimum_final_state_empty_samples: int,
    packing_se_kcal_max: float,
) -> dict[str, Any]:
    """Estimate ``p0`` in the unbiased state from staged-bias MBAR weights."""

    indicator = np.asarray(empty_indicator)
    if (
        indicator.shape != (table.u_kn.shape[1],)
        or indicator.dtype.kind != "b"
    ):
        raise ValueError(
            "empty_indicator must be one boolean value per sampled frame."
        )
    if (
        isinstance(kcal_per_dimensionless, (bool, np.bool_))
        or not math.isfinite(float(kcal_per_dimensionless))
        or float(kcal_per_dimensionless) <= 0.0
    ):
        raise ValueError("kcal_per_dimensionless must be finite and positive.")
    if (
        isinstance(minimum_final_state_empty_samples, (bool, np.bool_))
        or int(minimum_final_state_empty_samples)
        != minimum_final_state_empty_samples
        or int(minimum_final_state_empty_samples) < 1
    ):
        raise ValueError(
            "minimum_final_state_empty_samples must be a positive integer."
        )
    if (
        isinstance(packing_se_kcal_max, (bool, np.bool_))
        or not math.isfinite(float(packing_se_kcal_max))
        or float(packing_se_kcal_max) <= 0.0
    ):
        raise ValueError("packing_se_kcal_max must be finite and positive.")

    pymbar = _require_pymbar()
    estimator = pymbar.MBAR(
        table.u_kn,
        np.asarray(table.N_k, dtype=int),
        verbose=False,
    )
    with warnings.catch_warnings():
        # A zero-valued indicator is mathematically required for occupied
        # frames; PyMBAR internally takes log(A) and emits log(0) warnings.
        warnings.filterwarnings(
            "ignore",
            message="divide by zero encountered in log",
            category=RuntimeWarning,
            module=r"pymbar\.mbar",
        )
        expectation = estimator.compute_expectations(
            indicator.astype(float),
            output="averages",
            state_dependent=False,
            compute_uncertainty=True,
        )
    probability = float(expectation["mu"][0])
    probability_se = float(expectation["sigma"][0])
    if (
        not math.isfinite(probability)
        or not 0.0 < probability <= 1.0
        or not math.isfinite(probability_se)
        or probability_se < 0.0
    ):
        raise RuntimeError(
            "PACKING_ESTIMATE_INVALID: MBAR returned an invalid p0 estimate."
        )
    packing_dimensionless = -math.log(probability)
    packing_kcal = packing_dimensionless * float(kcal_per_dimensionless)
    packing_se_kcal = (
        probability_se / probability * float(kcal_per_dimensionless)
    )
    final_start = int(sum(table.N_k[:-1]))
    final_empty_count = int(np.count_nonzero(indicator[final_start:]))

    failure_codes: list[str] = []
    if final_empty_count < int(minimum_final_state_empty_samples):
        failure_codes.append("PACKING_EMPTY_SUPPORT_TOO_LOW")
    if packing_se_kcal > float(packing_se_kcal_max):
        failure_codes.append("PACKING_STANDARD_ERROR_TOO_HIGH")
    return {
        "status": "passed" if not failure_codes else "failed",
        "failure_codes": failure_codes,
        "p0": probability,
        "p0_standard_error": probability_se,
        "packing_free_energy_dimensionless": packing_dimensionless,
        "packing_free_energy_kcal_mol": packing_kcal,
        "packing_free_energy_standard_error_kcal_mol": packing_se_kcal,
        "final_state_empty_sample_count": final_empty_count,
        "minimum_final_state_empty_samples": int(
            minimum_final_state_empty_samples
        ),
        "packing_se_kcal_max": float(packing_se_kcal_max),
        "reduced_potential_table_sha256": table.state_hash,
        "empty_indicator_sha256": _array_sha256(
            indicator.astype(np.float64)
        ),
        "estimator": "PyMBAR.MBAR.compute_expectations",
        "target_state": table.row_labels[0],
    }
