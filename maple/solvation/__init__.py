"""Contract-first public surface for MAPLE's conservative Route-2 rebuild.

This package is intentionally independent of the legacy calculator path.  Phase 1
defines identities and validation contracts only; importing it cannot enable a
solvation capability or change an existing calculator.
"""

from .api import (
    ASE_PUBLIC_UNITS,
    PROFILE_REGISTRY,
    SCALAR_REGISTRY,
    STATE_REGISTRY,
    CapabilityStatus,
    CapabilityTier,
    Route2Result,
    ScalarDefinition,
    StateEquationDefinition,
)

__all__ = [
    "ASE_PUBLIC_UNITS",
    "PROFILE_REGISTRY",
    "SCALAR_REGISTRY",
    "STATE_REGISTRY",
    "CapabilityStatus",
    "CapabilityTier",
    "Route2Result",
    "ScalarDefinition",
    "StateEquationDefinition",
]
