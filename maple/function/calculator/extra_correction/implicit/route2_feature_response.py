"""Fixed-geometry Route-2 response algebra in model-native feature space.

The local-field and exact-GTO interfaces have different dimensional contracts.
This module owns the exact-GTO diagnostic composition, where the continuum
feature width need not equal the four ``l<=1`` density coefficients per atom.
It provides no coordinate derivative and is not a production force path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .route2_response import (
    NeutralDensityCoordinates,
    project_neutral_density_tangent,
)


class ModelFeatureLinearMap(Protocol):
    """Fixed-geometry density-to-model-feature map and its discrete adjoint."""

    atom_count: int
    model_feature_count: int
    reciprocal_energy_pairing: bool

    def model_feature_jvp(
        self,
        density_direction: np.ndarray,
    ) -> np.ndarray:
        """Map density directions to ``(n_atoms, n_features)``."""
        ...

    def model_feature_vjp(
        self,
        feature_cotangent: np.ndarray,
    ) -> np.ndarray:
        """Map model-feature cotangents back to density space."""
        ...


class FeatureDensityResponseLinearization(Protocol):
    """Learned density response to checkpoint-native model features."""

    def jvp(self, feature_direction: np.ndarray) -> np.ndarray:
        """Map model-feature directions to density directions."""
        ...

    def vjp(self, density_cotangent: np.ndarray) -> np.ndarray:
        """Map density cotangents back to model-feature space."""
        ...


def _validated_block(
    values: np.ndarray,
    *,
    expected_shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != expected_shape or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must be finite with shape {expected_shape}; "
            f"received {array.shape}."
        )
    return array


@dataclass(frozen=True)
class FeatureDrivenUnmixedDensityResidualLinearization:
    """Linearization of ``Pi0[c-M(z(c))]`` for native GTO features.

    The continuum map ``z`` and learned response ``M`` are rectangular and
    retain their own exact transposes. Numerical SCF mixing remains absent.
    This fixed-geometry object supplies the density-space residual algebra
    only; it does not provide any nuclear-coordinate derivative.
    """

    atom_count: int
    model_feature_map: ModelFeatureLinearMap
    density_response: FeatureDensityResponseLinearization
    neutral_tolerance: float = 1.0e-10
    _coordinates: NeutralDensityCoordinates = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.atom_count <= 0:
            raise ValueError("Route-2 residual linearization requires atoms.")
        if self.neutral_tolerance <= 0.0:
            raise ValueError("Neutral tangent tolerance must be positive.")
        if self.model_feature_map.atom_count != self.atom_count:
            raise ValueError(
                "Model-feature map atom count does not match the residual."
            )
        feature_count = self.model_feature_map.model_feature_count
        if not isinstance(feature_count, (int, np.integer)) or feature_count <= 0:
            raise ValueError("Model-feature count must be positive.")
        object.__setattr__(
            self,
            "_coordinates",
            NeutralDensityCoordinates(
                self.atom_count,
                neutral_tolerance=self.neutral_tolerance,
            ),
        )

    @property
    def model_feature_count(self) -> int:
        """Return the validated rectangular feature width."""

        return int(self.model_feature_map.model_feature_count)

    def _validated_neutral_density(
        self,
        values: np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        array = _validated_block(
            values,
            expected_shape=(self.atom_count, 4),
            name=name,
        )
        self._coordinates.reduce(array)
        return array

    def jvp(self, density_direction: np.ndarray) -> np.ndarray:
        """Apply ``Pi0 (I - J_M^z J_z)`` without dense Jacobians."""

        direction = self._validated_neutral_density(
            density_direction,
            name="density_direction",
        )
        feature_direction = _validated_block(
            self.model_feature_map.model_feature_jvp(direction),
            expected_shape=(self.atom_count, self.model_feature_count),
            name="continuum model-feature JVP",
        )
        response_direction = _validated_block(
            self.density_response.jvp(feature_direction),
            expected_shape=(self.atom_count, 4),
            name="MACE model-feature density-response JVP",
        )
        return project_neutral_density_tangent(direction - response_direction)

    def vjp(self, density_cotangent: np.ndarray) -> np.ndarray:
        """Apply ``Pi0 (I - J_z* J_M^z*)`` in the neutral subspace."""

        cotangent = self._validated_neutral_density(
            density_cotangent,
            name="density_cotangent",
        )
        feature_cotangent = _validated_block(
            self.density_response.vjp(cotangent),
            expected_shape=(self.atom_count, self.model_feature_count),
            name="MACE model-feature density-response VJP",
        )
        response_cotangent = _validated_block(
            self.model_feature_map.model_feature_vjp(feature_cotangent),
            expected_shape=(self.atom_count, 4),
            name="continuum model-feature VJP",
        )
        return project_neutral_density_tangent(cotangent - response_cotangent)


__all__ = [
    "FeatureDensityResponseLinearization",
    "FeatureDrivenUnmixedDensityResidualLinearization",
    "ModelFeatureLinearMap",
]
