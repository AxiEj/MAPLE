"""Route-2 execution availability and evidence-admitted capability declarations.

An experimentally callable operation is deliberately distinct from a workflow
or release admission.  Both surfaces default to false and stronger operations
or tiers require their prerequisites.
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


class ExecutionCapability(str, Enum):
    """Stable operation labels for an exact experimental execution surface."""

    ENERGY = "energy"
    FORCE = "force"
    MOLECULAR_VIRIAL = "molecular_virial"
    HESSIAN_VECTOR_PRODUCT = "hessian_vector_product"
    HESSIAN = "hessian"
    PERIODIC_STRESS = "periodic_stress"


@dataclass(frozen=True, slots=True)
class ExecutionStatus:
    """Callable operations without implying workflow or release admission.

    A true flag means the exact registered implementation exposes that
    operation under its declared experimental policy.  It does not claim an
    independent physical reference, production workflow validation, or release
    admission.
    """

    energy: bool = False
    force: bool = False
    molecular_virial: bool = False
    hessian_vector_product: bool = False
    hessian: bool = False
    periodic_stress: bool = False

    def __post_init__(self) -> None:
        for name in (
            "energy",
            "force",
            "molecular_virial",
            "hessian_vector_product",
            "hessian",
            "periodic_stress",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"Execution operation {name!r} must be a bool.")
        if self.force and not self.energy:
            raise ValueError("Force execution requires energy execution.")
        if self.molecular_virial and not self.force:
            raise ValueError("Molecular-virial execution requires force execution.")
        if self.hessian_vector_product and not self.force:
            raise ValueError("HVP execution requires force execution.")
        if self.hessian and not self.hessian_vector_product:
            raise ValueError("Hessian execution requires HVP execution.")
        if self.periodic_stress and not self.energy:
            raise ValueError("Periodic-stress execution requires energy execution.")

    def supports(self, operation: ExecutionCapability) -> bool:
        """Return whether *operation* is exposed by the execution surface."""

        if not isinstance(operation, ExecutionCapability):
            raise TypeError("operation must be an ExecutionCapability.")
        return {
            ExecutionCapability.ENERGY: self.energy,
            ExecutionCapability.FORCE: self.force,
            ExecutionCapability.MOLECULAR_VIRIAL: self.molecular_virial,
            ExecutionCapability.HESSIAN_VECTOR_PRODUCT: (self.hessian_vector_product),
            ExecutionCapability.HESSIAN: self.hessian,
            ExecutionCapability.PERIODIC_STRESS: self.periodic_stress,
        }[operation]

    @property
    def available_operations(self) -> tuple[ExecutionCapability, ...]:
        return tuple(
            operation for operation in ExecutionCapability if self.supports(operation)
        )

    def as_dict(self) -> dict[str, bool]:
        """Return a JSON-serializable execution declaration."""

        return {
            "energy": self.energy,
            "force": self.force,
            "molecular_virial": self.molecular_virial,
            "hessian_vector_product": self.hessian_vector_product,
            "hessian": self.hessian,
            "periodic_stress": self.periodic_stress,
        }


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
            raise ValueError(
                "Conservative force capability requires energy capability."
            )
        if self.hessian and not self.conservative_force:
            raise ValueError(
                "Hessian capability requires conservative force capability."
            )
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
