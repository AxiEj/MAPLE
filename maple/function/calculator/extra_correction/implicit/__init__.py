"""Evidence-gated implicit-solvation providers.

The independent OBC-II surface is importable without loading any legacy
OpenMM adapter or compatibility module. Compatibility exports are resolved
only when explicitly requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .independent_obc2 import IndependentOBC2
from .obc2_parameters import OBC2Parameters, build_obc2_parameters
from .obc2_profile import (
    OBC2_FORMULA_MANIFEST_ID,
    load_obc2_profile_license,
    load_obc2_profile_manifest,
    verify_obc2_profile_manifest,
)
from .obc2_topology import OBC2Atom, OBC2Bond, OBC2Topology
from .radii import IndependentMbondi2RadiusProvider, RadiusResult
from .result import SolvationDirectionalResult, SolvationResult


_LAZY_EXPORTS = {
    "APBSSASANonpolarProvider": (".nonpolar", "APBSSASANonpolarProvider"),
    "AmberToolsChaGB": (".amber_chagb", "AmberToolsChaGB"),
    "ChargeResult": (".charges", "ChargeResult"),
    "ImplicitSolvationCorrection": (".correction", "ImplicitSolvationCorrection"),
    "OpenMMAmberGBRadiusProvider": (".radii", "OpenMMAmberGBRadiusProvider"),
    "OpenMMMbondi2RadiusProvider": (".radii", "OpenMMMbondi2RadiusProvider"),
    "OpenMMNonpolarProvider": (".nonpolar", "OpenMMNonpolarProvider"),
    "prepare_charges": (".charges", "prepare_charges"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "APBSSASANonpolarProvider",
    "AmberToolsChaGB",
    "ChargeResult",
    "ImplicitSolvationCorrection",
    "IndependentMbondi2RadiusProvider",
    "IndependentOBC2",
    "OBC2Atom",
    "OBC2Bond",
    "OBC2Topology",
    "OBC2_FORMULA_MANIFEST_ID",
    "OpenMMAmberGBRadiusProvider",
    "OpenMMMbondi2RadiusProvider",
    "OpenMMNonpolarProvider",
    "OBC2Parameters",
    "RadiusResult",
    "SolvationResult",
    "SolvationDirectionalResult",
    "build_obc2_parameters",
    "load_obc2_profile_license",
    "load_obc2_profile_manifest",
    "prepare_charges",
    "verify_obc2_profile_manifest",
]
