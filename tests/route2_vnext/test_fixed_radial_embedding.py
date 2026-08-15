from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.gto_density import (
    gaussian_multipole_potential,
)
from maple.solvation.coupling.radial_embedding import FixedRadialSourceEmbedding
from maple.solvation.release.radial_embedding import (
    RadialEmbeddingCandidateMetric,
    select_fixed_radial_embedding,
)


def test_fixed_radial_embedding_preserves_source_and_adjoint() -> None:
    embedding = FixedRadialSourceEmbedding((0.5, 1.5), (0.25, 0.75))
    source = np.asarray(((0.3, 0.2, -0.1, 0.4), (-0.3, -0.2, 0.1, -0.4)))
    expanded = embedding.expand(source)

    np.testing.assert_allclose(np.sum(expanded, axis=1), source, atol=1.0e-15)
    rng = np.random.default_rng(20260815)
    cotangent = rng.normal(size=expanded.shape)
    np.testing.assert_allclose(
        np.vdot(expanded, cotangent),
        np.vdot(source, embedding.reduce_cotangent(cotangent)),
        atol=2.0e-15,
    )


def test_fixed_radial_surface_operator_matches_explicit_mixture() -> None:
    embedding = FixedRadialSourceEmbedding((0.5, 1.5), (0.25, 0.75))
    positions = np.asarray(((0.0, 0.0, 0.0), (1.2, -0.3, 0.4)))
    points = np.asarray(((4.0, 0.2, -0.7), (0.1, 5.0, 0.8)))
    source = np.asarray(((0.3, 0.2, -0.1, 0.4), (-0.3, -0.2, 0.1, -0.4)))
    expected = sum(
        weight
        * gaussian_multipole_potential(points, positions, source, sigma_angstrom=sigma)
        for sigma, weight in zip(
            embedding.sigmas_angstrom, embedding.weights, strict=True
        )
    )
    np.testing.assert_allclose(
        embedding.surface_potential(points, positions, source),
        expected,
        atol=2.0e-15,
    )


def test_fixed_radial_embedding_is_content_addressed_and_fail_closed() -> None:
    first = FixedRadialSourceEmbedding((0.5, 1.5), (0.25, 0.75))
    second = FixedRadialSourceEmbedding((0.5, 1.5), (0.5, 0.5))
    assert first.configuration_sha256 != second.configuration_sha256
    with pytest.raises(ValueError, match="sum to one"):
        FixedRadialSourceEmbedding((0.5, 1.5), (0.2, 0.7))
    with pytest.raises(ValueError, match="strictly positive"):
        FixedRadialSourceEmbedding((0.5, 1.5), (0.0, 1.0))


def _metric(candidate: str, compound: str, error: float, distance: float):
    return RadialEmbeddingCandidateMetric(
        candidate_id=candidate,
        compound_id=compound,
        split="train",
        weighted_relative_l2=error,
        weighted_rmse_hartree_per_e=0.001,
        maximum_absolute_error_hartree_per_e=0.002,
        relative_infinity_norm=error,
        embedding_distance_from_original=distance,
    )


def test_selection_uses_training_matrix_and_preregistered_tie_break() -> None:
    metrics = (
        _metric("baseline", "a", 0.4, 0.0),
        _metric("baseline", "b", 0.2, 0.0),
        _metric("candidate-z", "a", 0.2, 0.3),
        _metric("candidate-z", "b", 0.2, 0.3),
        _metric("candidate-a", "a", 0.2, 0.2),
        _metric("candidate-a", "b", 0.2, 0.2),
    )
    result = select_fixed_radial_embedding(
        metrics,
        candidate_ids=("baseline", "candidate-z", "candidate-a"),
        training_compound_ids=("a", "b"),
        baseline_candidate_id="baseline",
    )
    assert result.selected_candidate_id == "candidate-a"
    assert result.selected_training_objective == pytest.approx(0.04)
    assert result.relative_training_improvement == pytest.approx(0.6)


def test_selection_rejects_missing_or_heldout_substitution() -> None:
    metrics = (
        _metric("baseline", "a", 0.4, 0.0),
        _metric("candidate", "a", 0.2, 0.2),
    )
    with pytest.raises(ValueError, match="does not match"):
        select_fixed_radial_embedding(
            metrics,
            candidate_ids=("baseline", "candidate"),
            training_compound_ids=("a", "b"),
            baseline_candidate_id="baseline",
        )


def test_fixed_radial_embedding_preregistration_is_terminal_and_complete() -> None:
    root = Path(__file__).resolve().parents[2]
    path = (
        root / "docs/route2/preregistrations/"
        "mace-original-source-fixed-radial-embedding-v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    grid = payload["candidate_grid"]
    selection = payload["selection"]
    assert payload["status"] == "frozen-before-transfer-execution"
    assert grid["candidate_count"] == 1 + len(grid["narrow_sigmas_angstrom"]) * len(
        grid["narrow_weights"]
    )
    train = tuple(selection["training_compound_ids"])
    heldout = tuple(selection["heldout_from_selection_compound_ids"])
    assert len(train) == 8
    assert len(heldout) == 4
    assert not set(train).intersection(heldout)
    assert set(payload["capabilities"].values()) == {False}
    assert payload["terminal_decision"]["failure"].startswith(
        "Close this radial-embedding branch"
    )
