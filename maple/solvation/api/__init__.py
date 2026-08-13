"""Dependency-light public contracts for Route 2."""

from .capabilities import CapabilityStatus, CapabilityTier
from .profiles import (
    DIAGNOSTIC_FIXED_BOX40_CPCM_590_RADIAL_GTO_PROFILE_V1,
    DIAGNOSTIC_FIXED_BOX48_CPCM_1202_RADIAL_GTO_PROFILE_V1,
    DIAGNOSTIC_FIXED_BOX_CPCM_590_RADIAL_GTO_PROFILE_IDS,
    DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_PROFILE_V1,
    DIAGNOSTIC_PAIR_FRAME_CPCM_RADIAL_GTO_PROFILE_V1,
    DIAGNOSTIC_RADIAL_GTO_CPCM_ELECTROSTATIC_PROFILE_V1,
    PROFILE_REGISTRY,
    OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1,
    OPERATIONAL_CPCM_RADIAL_GTO_ELECTROSTATIC_PROFILE_V1,
    OPERATIONAL_CPCM_SMDCDS_PROFILE_V1,
    VARIATIONAL_COMMON_FUNCTIONAL_PROFILE_V1,
    SolvationProfile,
    get_solvation_profile,
    profile_registry_manifest,
)
from .provenance import ProvenanceBundle, ProvenanceRecord, RuntimeProvenance
from .result import EnergyComponent, ForceComponent, Route2Result
from .scalar_registry import (
    DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1,
    SCALAR_REGISTRY,
    ScalarDefinition,
    get_scalar_definition,
    scalar_registry_manifest,
)
from .state_registry import STATE_REGISTRY, StateEquationDefinition, get_state_equation
from .units import ASE_PUBLIC_UNITS, UnitContract

__all__ = [
    "ASE_PUBLIC_UNITS",
    "PROFILE_REGISTRY",
    "SCALAR_REGISTRY",
    "STATE_REGISTRY",
    "CapabilityStatus",
    "CapabilityTier",
    "DIAGNOSTIC_FIXED_BOX40_CPCM_590_RADIAL_GTO_PROFILE_V1",
    "DIAGNOSTIC_FIXED_BOX48_CPCM_1202_RADIAL_GTO_PROFILE_V1",
    "DIAGNOSTIC_FIXED_BOX_CPCM_590_RADIAL_GTO_PROFILE_IDS",
    "DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_PROFILE_V1",
    "DIAGNOSTIC_PAIR_FRAME_CPCM_RADIAL_GTO_PROFILE_V1",
    "DIAGNOSTIC_RADIAL_GTO_CPCM_ELECTROSTATIC_PROFILE_V1",
    "DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1",
    "EnergyComponent",
    "ForceComponent",
    "ProvenanceBundle",
    "ProvenanceRecord",
    "Route2Result",
    "RuntimeProvenance",
    "ScalarDefinition",
    "SolvationProfile",
    "StateEquationDefinition",
    "UnitContract",
    "get_scalar_definition",
    "get_solvation_profile",
    "get_state_equation",
    "scalar_registry_manifest",
    "profile_registry_manifest",
    "OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1",
    "OPERATIONAL_CPCM_RADIAL_GTO_ELECTROSTATIC_PROFILE_V1",
    "OPERATIONAL_CPCM_SMDCDS_PROFILE_V1",
    "VARIATIONAL_COMMON_FUNCTIONAL_PROFILE_V1",
]
