"""Preregistered selection algebra for one fixed radial-source experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RADIAL_EMBEDDING_SELECTION_CONTRACT_VERSION = (
    "route2-fixed-radial-embedding-selection-v1"
)


@dataclass(frozen=True, slots=True)
class RadialEmbeddingCandidateMetric:
    candidate_id: str
    compound_id: str
    split: str
    weighted_relative_l2: float
    weighted_rmse_hartree_per_e: float
    maximum_absolute_error_hartree_per_e: float
    relative_infinity_norm: float
    embedding_distance_from_original: float

    def __post_init__(self) -> None:
        for name in ("candidate_id", "compound_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty.")
        if self.split not in ("train", "heldout"):
            raise ValueError("split must be train or heldout.")
        values = np.asarray(
            (
                self.weighted_relative_l2,
                self.weighted_rmse_hartree_per_e,
                self.maximum_absolute_error_hartree_per_e,
                self.relative_infinity_norm,
                self.embedding_distance_from_original,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("candidate metrics must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class RadialEmbeddingSelection:
    selected_candidate_id: str
    selected_training_objective: float
    baseline_training_objective: float
    relative_training_improvement: float
    candidate_training_objectives: tuple[tuple[str, float], ...]
    contract_version: str = RADIAL_EMBEDDING_SELECTION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.selected_candidate_id, str)
            or not self.selected_candidate_id
        ):
            raise ValueError("selected_candidate_id must be non-empty.")
        if self.contract_version != RADIAL_EMBEDDING_SELECTION_CONTRACT_VERSION:
            raise ValueError("Unsupported radial embedding selection contract.")
        values = np.asarray(
            (
                self.selected_training_objective,
                self.baseline_training_objective,
                self.relative_training_improvement,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("selection summary values must be finite.")

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_training_objective": self.selected_training_objective,
            "baseline_training_objective": self.baseline_training_objective,
            "relative_training_improvement": self.relative_training_improvement,
            "candidate_training_objectives": dict(self.candidate_training_objectives),
        }


def select_fixed_radial_embedding(
    metrics: tuple[RadialEmbeddingCandidateMetric, ...],
    *,
    candidate_ids: tuple[str, ...],
    training_compound_ids: tuple[str, ...],
    baseline_candidate_id: str,
) -> RadialEmbeddingSelection:
    """Select only from preregistered training metrics.

    The primary objective is the unweighted mean across molecules of squared
    weighted relative MEP error.  This gives each preregistered chemistry one
    vote.  Exact ties prefer the candidate with the smaller declared distance
    from the original 1.5-A representation, then the lexical ID.
    """

    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_ids must be non-empty and unique.")
    if not training_compound_ids or len(set(training_compound_ids)) != len(
        training_compound_ids
    ):
        raise ValueError("training_compound_ids must be non-empty and unique.")
    if baseline_candidate_id not in candidate_ids:
        raise ValueError("baseline_candidate_id must be preregistered.")
    lookup: dict[tuple[str, str], RadialEmbeddingCandidateMetric] = {}
    for metric in metrics:
        if metric.split != "train":
            continue
        key = (metric.candidate_id, metric.compound_id)
        if key in lookup:
            raise ValueError("duplicate training metric.")
        lookup[key] = metric
    expected = {
        (candidate_id, compound_id)
        for candidate_id in candidate_ids
        for compound_id in training_compound_ids
    }
    if set(lookup) != expected:
        raise ValueError(
            "training metric matrix does not match preregistered identities."
        )
    scores: list[tuple[float, float, str]] = []
    for candidate_id in candidate_ids:
        records = [lookup[(candidate_id, item)] for item in training_compound_ids]
        objective = float(np.mean([item.weighted_relative_l2**2 for item in records]))
        distances = {item.embedding_distance_from_original for item in records}
        if len(distances) != 1:
            raise ValueError("candidate embedding distance changed between records.")
        scores.append((objective, distances.pop(), candidate_id))
    scores.sort()
    selected_objective, _, selected_id = scores[0]
    objective_by_id = {candidate_id: objective for objective, _, candidate_id in scores}
    baseline_objective = objective_by_id[baseline_candidate_id]
    relative_improvement = float(
        (baseline_objective - selected_objective)
        / max(baseline_objective, np.finfo(float).tiny)
    )
    return RadialEmbeddingSelection(
        selected_candidate_id=selected_id,
        selected_training_objective=selected_objective,
        baseline_training_objective=baseline_objective,
        relative_training_improvement=relative_improvement,
        candidate_training_objectives=tuple(
            (candidate_id, objective_by_id[candidate_id])
            for candidate_id in candidate_ids
        ),
    )


__all__ = [
    "RADIAL_EMBEDDING_SELECTION_CONTRACT_VERSION",
    "RadialEmbeddingCandidateMetric",
    "RadialEmbeddingSelection",
    "select_fixed_radial_embedding",
]
