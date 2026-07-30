"""Exact zero external potential for the Route-2 V0 pure-liquid scalar.

A pure-liquid coexistence or planar-interface calculation must not reuse a
solute--solvent MACE/Pauli external potential merely because a field is zero.
Its molecular configuration scalar is exactly

``u_pure(Gamma) = 0``.

This module names and validates that distinct source contract.  It binds one
regular Cartesian grid, one rigid solvent reference, and one declared
configuration list, then rejects every nonzero entry bit-for-bit.  The result
is a source-bound finite-grid control only: it supplies no physical-liquid
admission, surface tension, force, PES, solvation free energy, or accuracy
claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .route2_v0_molecular_external_potential import (
    Route2V0MolecularConfigurations,
    Route2V0MolecularSolventReference,
)
from .route2_v0_molecular_external_potential_contract import (
    Route2V0MolecularExternalPotentialContract,
)
from .route2_v0_structured_solvent import RegularCartesianGrid

V0_MOLECULAR_PURE_LIQUID_EXTERNAL_POTENTIAL_CONSTRUCTION = (
    "route2-v0-molecular-pure-liquid-zero-external-potential-v1"
)
V0_MOLECULAR_PURE_LIQUID_EXTERNAL_POTENTIAL_SCOPE = (
    "exact-zero-external-potential-pure-liquid-control-only-v1"
)


def _zero_immutable_vector(values: np.ndarray, *, count: int) -> np.ndarray:
    """Return one immutable exact-zero configuration-energy vector."""

    vector = np.asarray(values, dtype=float)
    if vector.shape != (count,) or not np.all(np.isfinite(vector)):
        raise ValueError(
            "Pure-liquid external potential must be finite with one entry for "
            "every configuration."
        )
    if np.any(vector != 0.0):
        raise ValueError(
            "Pure-liquid external potential must be exactly zero for every "
            "configuration."
        )
    result = np.array(vector, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Route2V0MolecularPureLiquidExternalPotential(
    Route2V0MolecularExternalPotentialContract
):
    """The exact zero external scalar required by a pure-liquid HNC control."""

    integration_grid: RegularCartesianGrid
    solvent: Route2V0MolecularSolventReference
    configurations: Route2V0MolecularConfigurations
    external_potential_hartree: np.ndarray
    construction: str = V0_MOLECULAR_PURE_LIQUID_EXTERNAL_POTENTIAL_CONSTRUCTION
    scope: str = V0_MOLECULAR_PURE_LIQUID_EXTERNAL_POTENTIAL_SCOPE

    def __post_init__(self) -> None:
        if not isinstance(self.integration_grid, RegularCartesianGrid):
            raise TypeError(
                "Pure-liquid external potential requires a regular Cartesian grid."
            )
        if not isinstance(self.solvent, Route2V0MolecularSolventReference):
            raise TypeError(
                "Pure-liquid external potential requires a molecular solvent reference."
            )
        if not isinstance(self.configurations, Route2V0MolecularConfigurations):
            raise TypeError(
                "Pure-liquid external potential requires molecular configurations."
            )
        if self.construction != V0_MOLECULAR_PURE_LIQUID_EXTERNAL_POTENTIAL_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 pure-liquid external potential.")
        if self.scope != V0_MOLECULAR_PURE_LIQUID_EXTERNAL_POTENTIAL_SCOPE:
            raise ValueError("Unsupported Route-2 V0 pure-liquid external scope.")
        object.__setattr__(
            self,
            "external_potential_hartree",
            _zero_immutable_vector(
                self.external_potential_hartree,
                count=self.configurations.configuration_count,
            ),
        )



def build_route2_v0_molecular_pure_liquid_external_potential(
    *,
    integration_grid: RegularCartesianGrid,
    solvent: Route2V0MolecularSolventReference,
    configurations: Route2V0MolecularConfigurations,
) -> Route2V0MolecularPureLiquidExternalPotential:
    """Build the exact-zero pure-liquid source on one declared grid and rule."""

    if not isinstance(integration_grid, RegularCartesianGrid):
        raise TypeError(
            "Pure-liquid external potential requires a regular Cartesian grid."
        )
    if not isinstance(solvent, Route2V0MolecularSolventReference):
        raise TypeError(
            "Pure-liquid external potential requires a molecular solvent reference."
        )
    if not isinstance(configurations, Route2V0MolecularConfigurations):
        raise TypeError(
            "Pure-liquid external potential requires molecular configurations."
        )
    return Route2V0MolecularPureLiquidExternalPotential(
        integration_grid=integration_grid,
        solvent=solvent,
        configurations=configurations,
        external_potential_hartree=np.zeros(
            configurations.configuration_count,
            dtype=float,
        ),
    )


__all__ = [
    "V0_MOLECULAR_PURE_LIQUID_EXTERNAL_POTENTIAL_CONSTRUCTION",
    "V0_MOLECULAR_PURE_LIQUID_EXTERNAL_POTENTIAL_SCOPE",
    "Route2V0MolecularPureLiquidExternalPotential",
    "build_route2_v0_molecular_pure_liquid_external_potential",
]
