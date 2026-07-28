"""Fixed full-density MEP source bound to one continuum surface.

The continuum consumes the molecular electrostatic potential directly.  It
must not recover atom-centred charges or multipoles from this object: doing so
would discard the charge-penetration and higher-angular information that a
SALTED or QM density is intended to preserve.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ..continuum_response import SurfaceChargeState


FROZEN_DENSITY_CHARGE_TOLERANCE_E = 1.0e-6


def _immutable_surface_points(values: np.ndarray) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(
            "Frozen-density surface points must be finite with shape (n, 3)."
        )
    result = np.array(points, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _immutable_surface_potential(
    values: np.ndarray,
    *,
    surface_size: int,
) -> np.ndarray:
    potential = np.asarray(values, dtype=float)
    if (
        potential.shape != (surface_size,)
        or not np.all(np.isfinite(potential))
    ):
        raise ValueError(
            "Frozen-density surface MEP must be finite with shape "
            f"({surface_size},); received {potential.shape}."
        )
    result = np.array(potential, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FrozenDensityMEPSource:
    """One immutable frozen density represented by its exact cavity MEP."""

    surface_points_bohr: np.ndarray
    surface_potential_hartree_per_e: np.ndarray
    declared_total_charge_e: float
    observed_total_charge_e: float
    source_model: str
    density_representation: str
    charge_tolerance_e: float = FROZEN_DENSITY_CHARGE_TOLERANCE_E

    def __post_init__(self) -> None:
        points = _immutable_surface_points(self.surface_points_bohr)
        potential = _immutable_surface_potential(
            self.surface_potential_hartree_per_e,
            surface_size=points.shape[0],
        )
        declared = float(self.declared_total_charge_e)
        observed = float(self.observed_total_charge_e)
        tolerance = float(self.charge_tolerance_e)
        if not math.isfinite(declared) or not math.isfinite(observed):
            raise ValueError("Frozen-density molecular charges must be finite.")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Frozen-density charge tolerance must be positive.")
        residual = abs(observed - declared)
        if residual > tolerance:
            raise ValueError(
                "Frozen density does not conserve its declared molecular "
                f"charge (observed={observed:.12g} e, "
                f"declared={declared:.12g} e, residual={residual:.3e} e)."
            )
        if not self.source_model:
            raise ValueError("Frozen-density source requires a model identity.")
        if not self.density_representation:
            raise ValueError(
                "Frozen-density source requires a representation identity."
            )
        object.__setattr__(self, "surface_points_bohr", points)
        object.__setattr__(
            self,
            "surface_potential_hartree_per_e",
            potential,
        )
        object.__setattr__(self, "declared_total_charge_e", declared)
        object.__setattr__(self, "observed_total_charge_e", observed)
        object.__setattr__(self, "charge_tolerance_e", tolerance)

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "solute_source": "frozen-density-surface-mep",
            "source_model": self.source_model,
            "density_representation": self.density_representation,
            "polarization_response": "fixed",
            "declared_total_charge_e": self.declared_total_charge_e,
            "observed_total_charge_e": self.observed_total_charge_e,
            "absolute_charge_residual_e": abs(
                self.observed_total_charge_e - self.declared_total_charge_e
            ),
        }


@dataclass(frozen=True)
class FrozenDensityMEPContinuumState:
    """Continuum result retaining the density-source identity."""

    source: FrozenDensityMEPSource
    response_state: SurfaceChargeState

    def __post_init__(self) -> None:
        if not np.array_equal(
            self.source.surface_potential_hartree_per_e,
            self.response_state.surface_potential_hartree_per_e,
        ):
            raise ValueError(
                "Continuum response state does not belong to the frozen "
                "density surface MEP."
            )


def solve_frozen_density_mep_continuum(
    response: Any,
    source: FrozenDensityMEPSource,
) -> FrozenDensityMEPContinuumState:
    """Solve one fixed-density continuum state on the source-bound surface."""

    try:
        response_points = np.asarray(response.surface_points_bohr, dtype=float)
    except AttributeError as exc:
        raise TypeError(
            "Frozen-density continuum response must expose surface_points_bohr."
        ) from exc
    if not np.array_equal(response_points, source.surface_points_bohr):
        raise ValueError(
            "Frozen density MEP can only be solved on its exact cavity surface."
        )
    state = response.solve(source.surface_potential_hartree_per_e)
    if not isinstance(state, SurfaceChargeState):
        raise TypeError(
            "Frozen-density continuum response must return SurfaceChargeState."
        )
    return FrozenDensityMEPContinuumState(
        source=source,
        response_state=state,
    )


__all__ = [
    "FROZEN_DENSITY_CHARGE_TOLERANCE_E",
    "FrozenDensityMEPContinuumState",
    "FrozenDensityMEPSource",
    "solve_frozen_density_mep_continuum",
]
