"""Stable identifiers for Route-2 electronic-model interoperability.

This module intentionally contains only dependency-free identifiers.  It is
safe to import from calculator, profile-registry, and implicit-solvation
layers without introducing a calculator/continuum import cycle.
"""

from __future__ import annotations

import re
from types import MappingProxyType

ROUTE2_ELECTRONIC_MODEL_ADAPTER_VERSION = 1
ROUTE2_FIELD_CONDITIONED_OPERATIONAL_ENERGY = (
    "field-conditioned-operational-energy-not-density-variational-v1"
)
ROUTE2_COMMON_VARIATIONAL_ELECTRONIC_FUNCTIONAL = (
    "common-variational-electronic-functional-v1"
)

# Route-2 currently standardises atom-centred net monopoles plus real-spherical
# l=1 dipoles at its public MLIP/continuum boundary.  A model may use any native
# representation internally, but its adapter must map to this exact source
# space before it enters the legacy fixed-point engine.
ROUTE2_ATOMIC_L1_SOURCE_SPACE = "maple-atomic-net-monopole-real-spherical-l1-v1"

# The existing released profiles remain identity-bound to the audited
# MACE-POLAR checkpoint.  Other models receive separate, versioned profiles;
# an adapter never silently substitutes a model under an existing profile.
ROUTE2_MACE_POLAR_MODEL_FAMILY = "mace-polar-1"
ROUTE2_MACE_POLAR_PROFILE_BINDING = (
    "official-mace-polar-1-m/float64/local-gto-reaction-field/v1"
)

# Input-model names are mapped to scientific families in one dependency-free
# registry.  Parser/factory validation uses the profile's expected family;
# continuum providers never inspect the input spelling.
_ROUTE2_INPUT_MODEL_FAMILIES = MappingProxyType(
    {
        "macepolm": ROUTE2_MACE_POLAR_MODEL_FAMILY,
    }
)
_ROUTE2_MODEL_FAMILY_LABELS = MappingProxyType(
    {
        ROUTE2_MACE_POLAR_MODEL_FAMILY: "official MACE-POLAR-1-M",
    }
)


def normalize_route2_input_model_name(value: object) -> str:
    """Return the compact model selector used by Route-2 profile matching."""

    return re.sub(r"[-_ ()]", "", str(value).strip().lower())


def route2_input_model_family(value: object) -> str | None:
    """Return the registered Route-2 family for one input model selector."""

    return _ROUTE2_INPUT_MODEL_FAMILIES.get(normalize_route2_input_model_name(value))


def route2_model_family_label(model_family: str) -> str:
    """Return a stable human-readable label without weakening identity."""

    family = str(model_family).strip()
    return _ROUTE2_MODEL_FAMILY_LABELS.get(family, family)


def validate_route2_input_model_family(
    value: object,
    *,
    expected_model_family: str,
) -> str:
    """Fail closed unless an input model belongs to the profile's family."""

    expected = str(expected_model_family).strip()
    if not expected:
        raise ValueError("Route-2 expected model family must be non-empty.")
    actual = route2_input_model_family(value)
    if actual != expected:
        label = route2_model_family_label(expected)
        raise ValueError(
            "Route-2 profile requires electronic model family "
            f"{expected!r} ({label}); model={value!r} is "
            + (
                "not registered for Route 2."
                if actual is None
                else f"registered as family {actual!r}."
            )
        )
    return actual


__all__ = [
    "ROUTE2_ATOMIC_L1_SOURCE_SPACE",
    "ROUTE2_ELECTRONIC_MODEL_ADAPTER_VERSION",
    "ROUTE2_FIELD_CONDITIONED_OPERATIONAL_ENERGY",
    "ROUTE2_COMMON_VARIATIONAL_ELECTRONIC_FUNCTIONAL",
    "ROUTE2_MACE_POLAR_MODEL_FAMILY",
    "ROUTE2_MACE_POLAR_PROFILE_BINDING",
    "normalize_route2_input_model_name",
    "route2_input_model_family",
    "route2_model_family_label",
    "validate_route2_input_model_family",
]
