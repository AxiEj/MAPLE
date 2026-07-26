"""Canonical discrete electrostatic pairing for Route 2.

This module owns the explicit order contracts between MACE-POLAR's raw
real-spherical ``l<=1`` density coefficients, MAPLE's Cartesian external
field convention, and graph-longrange's local-field feature input. It
deliberately performs no unit conversion: callers must provide conjugate
quantities in one consistent unit system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ElectrostaticPairing:
    """One immutable density/field ordering and bilinear pairing contract."""

    name: str
    field_to_density_indices: tuple[int, int, int, int]
    density_to_field_indices: tuple[int, int, int, int]

    @staticmethod
    def _validated_block(values: np.ndarray, *, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if (
            array.ndim != 2
            or array.shape[1] != 4
            or not np.all(np.isfinite(array))
        ):
            raise ValueError(
                f"{name} must be finite with shape (n_atoms, 4)."
            )
        return array

    def field_to_density_order(self, values: np.ndarray) -> np.ndarray:
        """Map ``[V,gx,gy,gz]`` to the raw density-dual component order."""

        field = self._validated_block(values, name="External node field")
        return field[:, self.field_to_density_indices].copy()

    def density_to_field_order(self, values: np.ndarray) -> np.ndarray:
        """Map raw ``[q,l1_0,l1_1,l1_2]`` to Cartesian dual order."""

        density = self._validated_block(
            values,
            name="Raw density-dual values",
        )
        return density[:, self.density_to_field_indices].copy()

    def pair(
        self,
        density_coefficients: np.ndarray,
        external_field: np.ndarray,
    ) -> float:
        """Return ``<density, field>`` without an implicit half factor."""

        density = self._validated_block(
            density_coefficients,
            name="Density coefficients",
        )
        field = self._validated_block(
            external_field,
            name="External node field",
        )
        if density.shape[0] != field.shape[0]:
            raise ValueError(
                "Density and external-field atom counts must match."
            )
        return float(
            np.vdot(
                density,
                field[:, self.field_to_density_indices],
            )
        )


MACE_POLAR_L1_PAIRING = ElectrostaticPairing(
    name="mace-polar-real-spherical-l1-v1",
    field_to_density_indices=(0, 2, 3, 1),
    density_to_field_indices=(0, 3, 1, 2),
)

# graph-longrange 0.4.0's GTOInternalFieldtoFeaturesBlock consumes the
# Cartesian node field through this e3nn component order. Keep local-jet and
# exact-GTO projection paths on this single explicit contract.
MACE_POLAR_MODEL_FEATURE_FIELD_INDICES = (0, 3, 1, 2)


__all__ = [
    "ElectrostaticPairing",
    "MACE_POLAR_L1_PAIRING",
    "MACE_POLAR_MODEL_FEATURE_FIELD_INDICES",
]
