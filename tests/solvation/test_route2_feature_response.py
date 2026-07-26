from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_feature_response import (
    FeatureDrivenUnmixedDensityResidualLinearization,
)
from maple.function.calculator.extra_correction.implicit.route2_response import (
    project_neutral_density_tangent,
)


class _MatrixModelFeatureMap:
    def __init__(
        self,
        matrix: np.ndarray,
        atom_count: int,
        model_feature_count: int,
    ):
        self.matrix = np.asarray(matrix, dtype=float)
        self.atom_count = atom_count
        self.model_feature_count = model_feature_count
        self.reciprocal_energy_pairing = True

    def model_feature_jvp(self, density_direction: np.ndarray) -> np.ndarray:
        flat = np.asarray(density_direction, dtype=float).reshape(-1)
        return (self.matrix @ flat).reshape(
            self.atom_count,
            self.model_feature_count,
        )

    def model_feature_vjp(self, feature_cotangent: np.ndarray) -> np.ndarray:
        flat = np.asarray(feature_cotangent, dtype=float).reshape(-1)
        return (self.matrix.T @ flat).reshape(self.atom_count, 4)


class _MatrixFeatureDensityResponse:
    def __init__(
        self,
        matrix: np.ndarray,
        atom_count: int,
        model_feature_count: int,
    ):
        self.matrix = np.asarray(matrix, dtype=float)
        self.atom_count = atom_count
        self.model_feature_count = model_feature_count

    def jvp(self, feature_direction: np.ndarray) -> np.ndarray:
        flat = np.asarray(feature_direction, dtype=float).reshape(-1)
        return (self.matrix @ flat).reshape(self.atom_count, 4)

    def vjp(self, density_cotangent: np.ndarray) -> np.ndarray:
        flat = np.asarray(density_cotangent, dtype=float).reshape(-1)
        return (self.matrix.T @ flat).reshape(
            self.atom_count,
            self.model_feature_count,
        )


def _neutral_projector(atom_count: int) -> np.ndarray:
    dimension = 4 * atom_count
    projector = np.eye(dimension)
    charges = np.zeros(dimension)
    charges[::4] = 1.0
    return projector - np.outer(charges, charges) / atom_count


def test_feature_driven_residual_matches_dense_rectangular_composition():
    atom_count = 3
    feature_count = 8
    density_dimension = 4 * atom_count
    feature_dimension = feature_count * atom_count
    rng = np.random.default_rng(20260727)
    continuum_features = rng.normal(
        scale=0.05,
        size=(feature_dimension, density_dimension),
    )
    model_response = rng.normal(
        scale=0.04,
        size=(density_dimension, feature_dimension),
    )
    projector = _neutral_projector(atom_count)
    dense = (
        projector
        @ (np.eye(density_dimension) - model_response @ continuum_features)
        @ projector
    )
    linearization = FeatureDrivenUnmixedDensityResidualLinearization(
        atom_count=atom_count,
        model_feature_map=_MatrixModelFeatureMap(
            continuum_features,
            atom_count,
            feature_count,
        ),
        density_response=_MatrixFeatureDensityResponse(
            model_response,
            atom_count,
            feature_count,
        ),
    )
    direction = project_neutral_density_tangent(rng.normal(size=(atom_count, 4)))
    cotangent = project_neutral_density_tangent(rng.normal(size=(atom_count, 4)))

    jvp = linearization.jvp(direction)
    vjp = linearization.vjp(cotangent)

    np.testing.assert_allclose(
        jvp.reshape(-1),
        dense @ direction.reshape(-1),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        vjp.reshape(-1),
        dense.T @ cotangent.reshape(-1),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert np.vdot(cotangent, jvp) == pytest.approx(
        np.vdot(vjp, direction),
        abs=1.0e-12,
    )


def test_feature_driven_residual_rejects_incompatible_map_metadata():
    atom_count = 2
    feature_count = 3
    continuum_features = np.zeros(
        (atom_count * feature_count, atom_count * 4),
    )
    model_response = np.zeros(
        (atom_count * 4, atom_count * feature_count),
    )
    density_response = _MatrixFeatureDensityResponse(
        model_response,
        atom_count,
        feature_count,
    )

    with pytest.raises(ValueError, match="atom count"):
        FeatureDrivenUnmixedDensityResidualLinearization(
            atom_count=atom_count,
            model_feature_map=_MatrixModelFeatureMap(
                continuum_features,
                atom_count + 1,
                feature_count,
            ),
            density_response=density_response,
        )

    with pytest.raises(ValueError, match="count must be positive"):
        FeatureDrivenUnmixedDensityResidualLinearization(
            atom_count=atom_count,
            model_feature_map=_MatrixModelFeatureMap(
                continuum_features,
                atom_count,
                0,
            ),
            density_response=density_response,
        )
