from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np


_ROUTE2_LEAF_COMPONENT_NAMES = (
    "solute_polarization",
    "pcm_polarization",
    "cds",
    "standard_state",
)
_ROUTE2_DERIVED_TOTAL_NAMES = (
    "electrostatic",
    "delta_g_solv",
)
_ROUTE2_ENERGY_LEDGER_RELATIVE_TOLERANCE = 1.0e-12


def _frozen_finite_mapping(
    values: Mapping[str, float],
    *,
    name: str,
) -> Mapping[str, float]:
    normalized: dict[str, float] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key)
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"{name}[{key!r}] must be finite.")
        normalized[key] = value
    return MappingProxyType(normalized)


def _require_matching_total(
    *,
    name: str,
    observed: float,
    expected: float,
) -> None:
    scale = max(1.0, abs(observed), abs(expected))
    if abs(observed - expected) > _ROUTE2_ENERGY_LEDGER_RELATIVE_TOLERANCE * scale:
        raise ValueError(
            f"Route 2 energy ledger {name} does not close: "
            f"observed={observed:.16g}, expected={expected:.16g}."
        )


@dataclass(frozen=True)
class Route2EnergyLedger:
    """Immutable Route-2 energy leaves and their checked derived totals."""

    leaf_components_hartree: Mapping[str, float]
    derived_totals_hartree: Mapping[str, float]

    def __post_init__(self) -> None:
        leaves = _frozen_finite_mapping(
            self.leaf_components_hartree,
            name="leaf_components_hartree",
        )
        totals = _frozen_finite_mapping(
            self.derived_totals_hartree,
            name="derived_totals_hartree",
        )
        if set(leaves) != set(_ROUTE2_LEAF_COMPONENT_NAMES):
            raise ValueError(
                "Route 2 energy ledger leaves must be "
                f"{_ROUTE2_LEAF_COMPONENT_NAMES!r}."
            )
        if set(totals) != set(_ROUTE2_DERIVED_TOTAL_NAMES):
            raise ValueError(
                "Route 2 energy ledger totals must be "
                f"{_ROUTE2_DERIVED_TOTAL_NAMES!r}."
            )
        _require_matching_total(
            name="electrostatic",
            observed=totals["electrostatic"],
            expected=(
                leaves["solute_polarization"] + leaves["pcm_polarization"]
            ),
        )
        _require_matching_total(
            name="delta_g_solv",
            observed=totals["delta_g_solv"],
            expected=(
                totals["electrostatic"]
                + leaves["cds"]
                + leaves["standard_state"]
            ),
        )
        object.__setattr__(
            self,
            "leaf_components_hartree",
            MappingProxyType(
                {key: leaves[key] for key in _ROUTE2_LEAF_COMPONENT_NAMES}
            ),
        )
        object.__setattr__(
            self,
            "derived_totals_hartree",
            MappingProxyType(
                {key: totals[key] for key in _ROUTE2_DERIVED_TOTAL_NAMES}
            ),
        )

    @classmethod
    def from_components(
        cls,
        components_hartree: Mapping[str, float],
    ) -> "Route2EnergyLedger":
        components = _frozen_finite_mapping(
            components_hartree,
            name="components_hartree",
        )
        allowed = set(_ROUTE2_LEAF_COMPONENT_NAMES).union(
            _ROUTE2_DERIVED_TOTAL_NAMES
        )
        unknown = sorted(set(components).difference(allowed))
        missing = [
            key for key in _ROUTE2_LEAF_COMPONENT_NAMES if key not in components
        ]
        if unknown or missing:
            details = []
            if unknown:
                details.append("unknown=" + ", ".join(unknown))
            if missing:
                details.append("missing=" + ", ".join(missing))
            raise ValueError(
                "Route 2 energy components must contain only the declared "
                "leaf terms and optional checked totals (" + "; ".join(details) + ")."
            )

        leaves = {
            key: components[key] for key in _ROUTE2_LEAF_COMPONENT_NAMES
        }
        totals = {
            "electrostatic": (
                leaves["solute_polarization"] + leaves["pcm_polarization"]
            ),
            "delta_g_solv": 0.0,
        }
        totals["delta_g_solv"] = (
            totals["electrostatic"] + leaves["cds"] + leaves["standard_state"]
        )
        for key in _ROUTE2_DERIVED_TOTAL_NAMES:
            if key in components:
                _require_matching_total(
                    name=key,
                    observed=components[key],
                    expected=totals[key],
                )
        return cls(leaves, totals)

    @property
    def components_hartree(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                **self.leaf_components_hartree,
                **self.derived_totals_hartree,
            }
        )


@dataclass(frozen=True)
class SolvationResult:
    energy_hartree: float
    forces_hartree_per_angstrom: np.ndarray | None = None
    components_hartree: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    leaf_components_hartree: Mapping[str, float] = field(init=False)
    derived_totals_hartree: Mapping[str, float] = field(init=False)

    def __post_init__(self) -> None:
        energy = float(self.energy_hartree)
        if not np.isfinite(energy):
            raise ValueError("Solvation energy must be finite.")
        object.__setattr__(self, "energy_hartree", energy)

        if self.forces_hartree_per_angstrom is not None:
            forces = np.array(
                self.forces_hartree_per_angstrom,
                dtype=float,
                copy=True,
            )
            if forces.ndim != 2 or forces.shape[1:] != (3,):
                raise ValueError(
                    "Solvation forces must have shape (n_atoms, 3)."
                )
            if not np.all(np.isfinite(forces)):
                raise ValueError("Solvation forces must be finite.")
            forces.setflags(write=False)
            object.__setattr__(self, "forces_hartree_per_angstrom", forces)

        if self.components_hartree:
            ledger = Route2EnergyLedger.from_components(self.components_hartree)
            _require_matching_total(
                name="energy_hartree",
                observed=energy,
                expected=ledger.derived_totals_hartree["delta_g_solv"],
            )
            object.__setattr__(
                self,
                "components_hartree",
                ledger.components_hartree,
            )
            object.__setattr__(
                self,
                "leaf_components_hartree",
                ledger.leaf_components_hartree,
            )
            object.__setattr__(
                self,
                "derived_totals_hartree",
                ledger.derived_totals_hartree,
            )
        else:
            empty = MappingProxyType({})
            object.__setattr__(self, "components_hartree", empty)
            object.__setattr__(self, "leaf_components_hartree", empty)
            object.__setattr__(self, "derived_totals_hartree", empty)
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )


@dataclass(frozen=True)
class SinglePointDerivativeEvidence:
    """Research-only derivative evidence that is not an ASE/PES result.

    Route 2's pyddx derivative expression is retained for falsification and
    finite-difference checks, but it has not passed the solution-phase PES
    admission gate. Keeping it separate from :class:`SolvationResult`
    prevents an ASE caller from interpreting it as an available force field.
    """

    energy_hartree: float
    forces_hartree_per_angstrom: np.ndarray
    components_hartree: Mapping[str, float]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        forces = np.array(self.forces_hartree_per_angstrom, dtype=float, copy=True)
        if forces.ndim != 2 or forces.shape[1:] != (3,):
            raise ValueError(
                "Single-point derivative evidence requires an (n_atoms, 3) force array."
            )
        forces.setflags(write=False)
        object.__setattr__(self, "forces_hartree_per_angstrom", forces)
        object.__setattr__(
            self,
            "components_hartree",
            MappingProxyType(
                {
                    str(name): float(value)
                    for name, value in self.components_hartree.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )
