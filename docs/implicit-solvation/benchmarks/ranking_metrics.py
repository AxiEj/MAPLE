"""Deterministic series-level ranking metrics for Route 1 diagnostics.

This module deliberately separates ranking *measurement* from any energy
correction.  It never alters a prediction, derives an offset, or chooses an
endpoint.  Callers provide fixed experimental/predicted values only in their
label-exposed scoring phase.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np


def _finite_vector(values: Iterable[float], *, name: str) -> np.ndarray:
    vector = np.asarray(list(values), dtype=np.float64)
    if vector.ndim != 1 or not vector.size:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains a non-finite value.")
    return vector


def _validated_vectors(
    experimental: Iterable[float], prediction: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    observed = _finite_vector(experimental, name="experimental")
    predicted = _finite_vector(prediction, name="prediction")
    if observed.shape != predicted.shape:
        raise ValueError("experimental and prediction vectors must have equal length.")
    return observed, predicted


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, deterministically breaking only sorting ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    centered_left = left - float(np.mean(left))
    centered_right = right - float(np.mean(right))
    denominator = math.sqrt(
        float(np.dot(centered_left, centered_left))
        * float(np.dot(centered_right, centered_right))
    )
    if denominator == 0.0:
        return None
    return float(np.dot(centered_left, centered_right) / denominator)


def spearman_rho(experimental: Iterable[float], prediction: Iterable[float]) -> float | None:
    """Compute Spearman's rho with average ranks and a fail-closed tie case."""
    observed, predicted = _validated_vectors(experimental, prediction)
    return _pearson(_average_ranks(observed), _average_ranks(predicted))


def kendall_tau_b(
    experimental: Iterable[float], prediction: Iterable[float]
) -> float | None:
    """Compute Kendall tau-b without treating pairs as independent samples."""
    observed, predicted = _validated_vectors(experimental, prediction)
    concordant = discordant = tied_observed = tied_predicted = 0
    for first in range(observed.size - 1):
        for second in range(first + 1, observed.size):
            observed_delta = observed[first] - observed[second]
            predicted_delta = predicted[first] - predicted[second]
            observed_tie = observed_delta == 0.0
            predicted_tie = predicted_delta == 0.0
            if observed_tie and predicted_tie:
                continue
            if observed_tie:
                tied_observed += 1
                continue
            if predicted_tie:
                tied_predicted += 1
                continue
            if (observed_delta > 0.0) == (predicted_delta > 0.0):
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tied_observed)
        * (concordant + discordant + tied_predicted)
    )
    if denominator == 0.0:
        return None
    return float((concordant - discordant) / denominator)


def _display_float(value: float) -> str:
    numeric = float(value)
    return f"{numeric:.1f}" if numeric.is_integer() else f"{numeric:.12g}"


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _pair_rows(
    experimental: np.ndarray,
    predicted: np.ndarray,
    uncertainty: np.ndarray,
    *,
    fixed_threshold: float | None,
    uncertainty_z: float | None,
) -> list[dict[str, float | int | bool]]:
    rows: list[dict[str, float | int | bool]] = []
    for first in range(experimental.size - 1):
        for second in range(first + 1, experimental.size):
            observed_delta = float(experimental[first] - experimental[second])
            if fixed_threshold is not None and abs(observed_delta) < fixed_threshold:
                continue
            if uncertainty_z is not None:
                distinguishable = uncertainty_z * math.sqrt(
                    float(uncertainty[first] ** 2 + uncertainty[second] ** 2)
                )
                if abs(observed_delta) <= distinguishable:
                    continue
            if observed_delta == 0.0:
                continue
            predicted_delta = float(predicted[first] - predicted[second])
            predicted_sign = _sign(predicted_delta)
            observed_sign = _sign(observed_delta)
            rows.append(
                {
                    "first": first,
                    "second": second,
                    "experimental_delta": observed_delta,
                    "predicted_delta": predicted_delta,
                    "absolute_delta_error": abs(predicted_delta - observed_delta),
                    "squared_delta_error": (predicted_delta - observed_delta) ** 2,
                    "predicted_margin": abs(predicted_delta),
                    "correct": predicted_sign == observed_sign,
                    "predicted_tie": predicted_sign == 0,
                }
            )
    return rows


def _summarize_pair_rows(rows: Sequence[Mapping[str, float | int | bool]]) -> dict[str, float | int | None]:
    count = len(rows)
    if not count:
        return {
            "pair_count": 0,
            "correct_count": 0,
            "wrong_count": 0,
            "predicted_tie_count": 0,
            "forced_sign_accuracy": None,
            "answered_sign_accuracy": None,
            "answered_coverage": None,
            "delta_delta_mae": None,
            "delta_delta_rmse": None,
        }
    correct = sum(bool(row["correct"]) for row in rows)
    predicted_ties = sum(bool(row["predicted_tie"]) for row in rows)
    wrong = count - correct - predicted_ties
    answered = correct + wrong
    absolute_errors = [float(row["absolute_delta_error"]) for row in rows]
    squared_errors = [float(row["squared_delta_error"]) for row in rows]
    return {
        "pair_count": count,
        "correct_count": correct,
        "wrong_count": wrong,
        "predicted_tie_count": predicted_ties,
        "forced_sign_accuracy": float(correct / count),
        "answered_sign_accuracy": float(correct / answered) if answered else None,
        "answered_coverage": float(answered / count),
        "delta_delta_mae": float(np.mean(absolute_errors)),
        "delta_delta_rmse": float(math.sqrt(float(np.mean(squared_errors)))),
    }


def _top_k_summary(
    compound_ids: Sequence[str],
    experimental: np.ndarray,
    predicted: np.ndarray,
    top_k: int,
) -> dict[str, float | int | str | None]:
    if top_k <= 0:
        raise ValueError("top-k values must be positive integers.")
    if top_k > experimental.size:
        return {
            "status": "insufficient_members",
            "k": top_k,
            "overlap_count": None,
            "overlap_fraction": None,
            "enrichment_vs_random": None,
            "best_of_predicted_top_k_regret": None,
        }

    def _top_indices(values: np.ndarray) -> list[int] | None:
        order = sorted(range(values.size), key=lambda index: (values[index], compound_ids[index]))
        if top_k < values.size and values[order[top_k - 1]] == values[order[top_k]]:
            return None
        return order[:top_k]

    observed_top = _top_indices(experimental)
    predicted_top = _top_indices(predicted)
    if observed_top is None or predicted_top is None:
        return {
            "status": "tied_boundary",
            "k": top_k,
            "overlap_count": None,
            "overlap_fraction": None,
            "enrichment_vs_random": None,
            "best_of_predicted_top_k_regret": None,
        }
    overlap = len(set(observed_top).intersection(predicted_top))
    overlap_fraction = overlap / top_k
    random_overlap_fraction = top_k / experimental.size
    best_predicted = min(float(experimental[index]) for index in predicted_top)
    best_observed = min(float(value) for value in experimental)
    return {
        "status": "evaluable",
        "k": top_k,
        "overlap_count": overlap,
        "overlap_fraction": float(overlap_fraction),
        "enrichment_vs_random": float(overlap_fraction / random_overlap_fraction),
        "best_of_predicted_top_k_regret": float(best_predicted - best_observed),
    }


def series_ranking_metrics(
    *,
    compound_ids: Sequence[str],
    experimental: Iterable[float],
    experimental_uncertainty: Iterable[float],
    predicted: Iterable[float],
    fixed_delta_thresholds: Iterable[float],
    uncertainty_z: float,
    top_ks: Iterable[int],
    confidence_margin_thresholds: Iterable[float],
) -> dict[str, object]:
    """Measure a single predeclared series without changing its predictions.

    Lower values are considered better throughout (the free-energy convention
    used by the Route 1 hydration diagnostics).
    """
    if len(compound_ids) < 2 or len(compound_ids) != len(set(compound_ids)):
        raise ValueError("A series requires at least two unique compound ids.")
    observed, predicted_values = _validated_vectors(experimental, predicted)
    uncertainty = _finite_vector(experimental_uncertainty, name="experimental_uncertainty")
    if uncertainty.shape != observed.shape or np.any(uncertainty < 0.0):
        raise ValueError(
            "experimental_uncertainty must be non-negative and align with experimental."
        )
    if not math.isfinite(float(uncertainty_z)) or uncertainty_z <= 0.0:
        raise ValueError("uncertainty_z must be finite and positive.")

    fixed_thresholds = sorted({float(value) for value in fixed_delta_thresholds})
    if any(not math.isfinite(value) or value <= 0.0 for value in fixed_thresholds):
        raise ValueError("fixed delta thresholds must be finite and positive.")
    margins = sorted({float(value) for value in confidence_margin_thresholds})
    if any(not math.isfinite(value) or value < 0.0 for value in margins):
        raise ValueError("confidence margins must be finite and non-negative.")

    all_pairs = _pair_rows(
        observed,
        predicted_values,
        uncertainty,
        fixed_threshold=0.0,
        uncertainty_z=None,
    )
    uncertainty_pairs = _pair_rows(
        observed,
        predicted_values,
        uncertainty,
        fixed_threshold=None,
        uncertainty_z=float(uncertainty_z),
    )
    pair_metrics: dict[str, dict[str, float | int | None]] = {
        "all_experimentally_distinct": _summarize_pair_rows(all_pairs),
        f"uncertainty_z_{_display_float(float(uncertainty_z))}": _summarize_pair_rows(
            uncertainty_pairs
        ),
    }
    for threshold in fixed_thresholds:
        pair_metrics[f"fixed_abs_delta_gte_{_display_float(threshold)}"] = _summarize_pair_rows(
            _pair_rows(
                observed,
                predicted_values,
                uncertainty,
                fixed_threshold=threshold,
                uncertainty_z=None,
            )
        )

    curve: dict[str, dict[str, float | int | None]] = {}
    eligible_count = len(uncertainty_pairs)
    for margin in margins:
        selected = [
            row
            for row in uncertainty_pairs
            if float(row["predicted_margin"]) >= margin
        ]
        summary = _summarize_pair_rows(selected)
        accuracy = summary["forced_sign_accuracy"]
        curve[_display_float(margin)] = {
            "eligible_pair_count": eligible_count,
            "answered_pair_count": len(selected),
            "coverage": float(len(selected) / eligible_count) if eligible_count else None,
            "forced_sign_accuracy": accuracy,
            "risk": (1.0 - float(accuracy)) if accuracy is not None else None,
        }

    return {
        "member_count": len(compound_ids),
        "kendall_tau_b": kendall_tau_b(observed, predicted_values),
        "spearman_rho": spearman_rho(observed, predicted_values),
        "pair_metrics": pair_metrics,
        "top_k": {
            str(int(top_k)): _top_k_summary(
                compound_ids, observed, predicted_values, int(top_k)
            )
            for top_k in sorted({int(value) for value in top_ks})
        },
        "coverage_risk_curve": curve,
    }


def series_macro_summary(
    per_series: Mapping[str, Mapping[str, object]]
) -> dict[str, dict[str, float | int | None]]:
    """Average ranking metrics over series, not over overlapping molecular pairs."""
    result: dict[str, dict[str, float | int | None]] = {}
    for metric in ("kendall_tau_b", "spearman_rho"):
        values = []
        for summary in [per_series[series_id] for series_id in sorted(per_series)]:
            raw_metric = summary.get(metric)
            if raw_metric is not None and isinstance(raw_metric, (int, float)):
                values.append(float(raw_metric))
        result[metric] = {
            "series_count": len(values),
            "value": float(np.mean(values)) if values else None,
        }
    return result


def paired_series_macro_bootstrap(
    baseline: Mapping[str, Mapping[str, object]],
    candidate: Mapping[str, Mapping[str, object]],
    *,
    metric: str,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int | list[float] | None]:
    """Bootstrap complete series to avoid pseudo-replication of pair metrics."""
    if resamples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Bootstrap resamples/confidence are invalid.")
    ids = [
        series_id
        for series_id in sorted(set(baseline).intersection(candidate))
        if baseline[series_id].get(metric) is not None
        and candidate[series_id].get(metric) is not None
    ]
    if not ids:
        return {
            "series_count": 0,
            "candidate_minus_baseline": None,
            "bootstrap_ci": None,
        }
    differences = []
    for series_id in ids:
        baseline_value = baseline[series_id].get(metric)
        candidate_value = candidate[series_id].get(metric)
        if not isinstance(baseline_value, (int, float)) or not isinstance(
            candidate_value, (int, float)
        ):
            raise RuntimeError("paired_series_macro_bootstrap invariant violated")
        differences.append(float(candidate_value) - float(baseline_value))
    differences = np.asarray(differences, dtype=np.float64)
    point = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    draws = np.mean(
        differences[rng.integers(0, differences.size, size=(resamples, differences.size))],
        axis=1,
    )
    alpha = (1.0 - confidence) / 2.0
    return {
        "series_count": len(ids),
        "candidate_minus_baseline": point,
        "bootstrap_ci": [
            float(np.quantile(draws, alpha)),
            float(np.quantile(draws, 1.0 - alpha)),
        ],
    }
