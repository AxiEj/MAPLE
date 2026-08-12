"""Evidence-admitted Route-2 capability declarations.

Possessing a callable method never grants a capability.  All flags therefore
default to false and stronger tiers require their prerequisite tiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CapabilityTier(str, Enum):
    """Stable capability labels from the Route-2 release contract."""

    ENERGY = "E"
    CONSERVATIVE_FORCE = "F"
    HESSIAN = "H"
    VARIATIONAL = "V"
    MOLECULAR_DYNAMICS = "M"


@dataclass(frozen=True, slots=True)
class CapabilityStatus:
    """Immutable, fail-closed capability status for one scalar/profile."""

    energy: bool = False
    conservative_force: bool = False
    hessian: bool = False
    variational_functional: bool = False
    molecular_dynamics: bool = False

    def __post_init__(self) -> None:
        for name in (
            "energy",
            "conservative_force",
            "hessian",
            "variational_functional",
            "molecular_dynamics",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"Capability {name!r} must be a bool.")
        if self.conservative_force and not self.energy:
            raise ValueError("Conservative force capability requires energy capability.")
        if self.hessian and not self.conservative_force:
            raise ValueError("Hessian capability requires conservative force capability.")
        if self.variational_functional and not self.energy:
            raise ValueError("Variational capability requires energy capability.")
        if self.molecular_dynamics and not self.conservative_force:
            raise ValueError("MD capability requires conservative force capability.")

    def admits(self, tier: CapabilityTier) -> bool:
        """Return whether *tier* has explicit evidence admission."""

        if not isinstance(tier, CapabilityTier):
            raise TypeError("tier must be a CapabilityTier.")
        return {
            CapabilityTier.ENERGY: self.energy,
            CapabilityTier.CONSERVATIVE_FORCE: self.conservative_force,
            CapabilityTier.HESSIAN: self.hessian,
            CapabilityTier.VARIATIONAL: self.variational_functional,
            CapabilityTier.MOLECULAR_DYNAMICS: self.molecular_dynamics,
        }[tier]

    @property
    def enabled_tiers(self) -> tuple[CapabilityTier, ...]:
        return tuple(tier for tier in CapabilityTier if self.admits(tier))
