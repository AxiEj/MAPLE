"""Authoritative, immutable Route-2 profile and admission registry."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .capabilities import CapabilityStatus
from .scalar_registry import (
    DIAGNOSTIC_DDX_DDPCM_RADIAL_GTO_ELECTROSTATIC_V1,
    DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1,
    OPERATIONAL_CPCM_ELECTROSTATIC_V1,
    OPERATIONAL_CPCM_SMDCDS_V1,
    SCALAR_REGISTRY,
    VARIATIONAL_COMMON_FUNCTIONAL_V1,
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_V1,
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_V1,
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
)
from .state_registry import (
    OPERATIONAL_STATE_EQUATION_ID,
    STATE_REGISTRY,
    VARIATIONAL_STATE_EQUATION_ID,
)

OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1 = (
    "route2-profile-operational-cpcm-fixedtopology-electrostatic-v1"
)
OPERATIONAL_CPCM_RADIAL_GTO_ELECTROSTATIC_PROFILE_V1 = (
    "route2-profile-operational-cpcm-fixedtopology-radialgto-electrostatic-v1"
)
DIAGNOSTIC_FIXED_BOX_CPCM_590_RADIAL_GTO_PROFILE_IDS = MappingProxyType(
    {
        box_length: (
            f"route2-profile-diagnostic-fixedbox{box_length}-cpcm590-"
            "radialgto-electrostatic-v1"
        )
        for box_length in (32, 40, 48, 56)
    }
)
DIAGNOSTIC_FIXED_BOX40_CPCM_590_RADIAL_GTO_PROFILE_V1 = (
    DIAGNOSTIC_FIXED_BOX_CPCM_590_RADIAL_GTO_PROFILE_IDS[40]
)
DIAGNOSTIC_FIXED_BOX48_CPCM_1202_RADIAL_GTO_PROFILE_V1 = (
    "route2-profile-diagnostic-fixedbox48-cpcm1202-" "radialgto-electrostatic-v1"
)
DIAGNOSTIC_PAIR_FRAME_CPCM_RADIAL_GTO_PROFILE_V1 = (
    "route2-profile-diagnostic-pairframe-cpcm110-radialgto-electrostatic-v1"
)
DIAGNOSTIC_DDX_DDPCM_194_RADIAL_GTO_PROFILE_V1 = (
    "route2-profile-diagnostic-ddx-ddpcm194-radialgto-electrostatic-v1"
)
DIAGNOSTIC_RADIAL_GTO_CPCM_ELECTROSTATIC_PROFILE_V1 = (
    "route2-profile-diagnostic-cpcm-injectedgrid-radialgto-electrostatic-v1"
)
DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_PROFILE_V1 = (
    "route2-profile-diagnostic-localjet-cpcm-fixedtopology-electrostatic-v1"
)
OPERATIONAL_CPCM_SMDCDS_PROFILE_V1 = (
    "route2-profile-operational-cpcm-fixedtopology-smdcds-v1"
)
VARIATIONAL_COMMON_FUNCTIONAL_PROFILE_V1 = (
    "route2-profile-variational-common-functional-v1"
)
VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_PROFILE_V1 = (
    "route2-profile-variational-macepolar-energygradient-fixedcavity-cpcm-v1"
)
VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_PROFILE_V1 = (
    "route2-profile-variational-macepolar-energygradient-fixedcavity-"
    "harmonicgalerkin-cpcm-v1"
)
VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1 = (
    "route2-profile-variational-macepolar-energygradient-"
    "smoothharmonicgalerkin-cpcm-v1"
)
MACE_POLAR_VARIATIONAL_EFFECTIVE_SOURCE_MODEL_PROFILE_ID = (
    "mace-polar-route2-variational-effective-source-v1"
)
EXACT_GTO_COUPLING_CANDIDATE_ID = (
    "maple.route2.coupling.single-width-same-basis-gto-candidate.v1"
)
LOCAL_JET_DIAGNOSTIC_COUPLING_ID = (
    "maple.route2.coupling.exterior-local-l1-jet-diagnostic.v1"
)
MACE_POLAR_RADIAL_GTO_COUPLING_ID = "route2-coupling-mace-polar-native-radial-gto-v1"
ATOMIC_L1_SOURCE_SPACE_ID = "maple.route2.atomic-l1-source-space.v1"
ATOMIC_L1_FIELD_DUAL_SPACE_ID = "maple.route2.atomic-l1-field-dual-space.v1"
ATOMIC_L1_PAIRING_ID = "maple.route2.atomic-l1-pairing.v1"
MACE_POLAR_RADIAL_GTO_SOURCE_SPACE_ID = (
    "maple.route2.mace-polar-radial-gto-source-space.v1"
)
MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE_ID = (
    "maple.route2.mace-polar-radial-gto-field-dual-space.v1"
)
MACE_POLAR_RADIAL_GTO_PAIRING_ID = "maple.route2.mace-polar-radial-gto-pairing.v1"
LEGACY_UNBOUND_COORDINATE_CONTRACT_ID = (
    "maple.route2.legacy-profile-coordinate-scales-unbound.v1"
)
MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT_ID = (
    "maple.route2.mace-polar-radial-gto-linear-charge-coordinates.v1"
)
UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID = (
    "maple.route2.continuum-configuration.unbound-diagnostic.v1"
)
WATER_CPCM_194_CONFIGURATION_CONTRACT_ID = (
    "maple.route2.continuum-configuration.water-eps78p39-smd-radii-" "lebedev194.v1"
)
WATER_CPCM_590_CONFIGURATION_CONTRACT_ID = (
    "maple.route2.continuum-configuration.water-eps78p39-smd-radii-" "lebedev590.v1"
)
WATER_CPCM_1202_CONFIGURATION_CONTRACT_ID = (
    "maple.route2.continuum-configuration.water-eps78p39-smd-radii-" "lebedev1202.v1"
)
PAIR_FRAME_WATER_CPCM_110_CONFIGURATION_CONTRACT_ID = (
    "maple.route2.continuum-configuration.water-eps78p39-smd-radii-"
    "pairframe-lebedev110.v1"
)
DDX_WATER_DDPCM_194_CONFIGURATION_CONTRACT_ID = (
    "maple.route2.continuum-configuration.water-eps78p39-smd-radii-"
    "ddx-ddpcm-l8-lebedev194.v1"
)
FIXED_HARMONIC_GALERKIN_CPCM_CONTINUUM_PROFILE_ID = (
    "fixed-harmonic-galerkin-cpcm-candidate-v1"
)
FIXED_EXTERNAL_HARMONIC_CAVITY_PROFILE_ID = (
    "fixed-external-harmonic-coefficient-cavity-v1"
)
FIXED_HARMONIC_GALERKIN_CONFIGURATION_CONTRACT_ID = (
    "maple.route2.continuum-configuration.fixed-harmonic-galerkin-unbound.v1"
)
SMOOTH_HARMONIC_GALERKIN_CPCM_CONTINUUM_PROFILE_ID = (
    "smooth-weighted-harmonic-galerkin-cpcm-candidate-v1"
)
SMOOTH_HARMONIC_CAVITY_PROFILE_ID = (
    "smooth-weighted-overlap-harmonic-cavity-candidate-v1"
)
SMOOTH_HARMONIC_GALERKIN_CONFIGURATION_CONTRACT_ID = (
    "maple.route2.continuum-configuration.smooth-weighted-harmonic-galerkin.v1"
)
MACE_POLAR_MODEL_PROFILE_ID = "mace-polar-route2-source-field-contract-v1"
MACE_POLAR_FIXED_BOX_MODEL_PROFILE_IDS = MappingProxyType(
    {
        box_length: f"mace-polar-route2-source-field-fixed-box{box_length}-contract-v1"
        for box_length in (32, 40, 48, 56)
    }
)
MACE_POLAR_FIXED_BOX40_MODEL_PROFILE_ID = MACE_POLAR_FIXED_BOX_MODEL_PROFILE_IDS[40]
MACE_POLAR_MOLECULAR_REALSPACE_EVALUATOR_ID = "graph-longrange-molecular-realspace-v1"
MACE_POLAR_FORCED_RECIPROCAL_FIXED_BOX_EVALUATOR_IDS = MappingProxyType(
    {
        box_length: f"graph-longrange-forced-periodic-fixed-box{box_length}-v1"
        for box_length in (32, 40, 48, 56)
    }
)
MACE_POLAR_FORCED_RECIPROCAL_FIXED_BOX40_EVALUATOR_ID = (
    MACE_POLAR_FORCED_RECIPROCAL_FIXED_BOX_EVALUATOR_IDS[40]
)


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


@dataclass(frozen=True, slots=True)
class SolvationProfile:
    """One immutable scalar/state/provider identity and its evidence admission."""

    profile_id: str
    scalar_id: str
    state_equation_id: str
    model_profile: str
    continuum_profile: str
    cavity_profile: str
    nonpolar_profile: str
    coupling_id: str
    source_space_id: str
    field_space_id: str
    pairing_id: str
    coordinate_contract_id: str
    continuum_configuration_contract_id: str
    capabilities: CapabilityStatus = CapabilityStatus()
    evidence_artifact_ids: tuple[str, ...] = ()
    enabled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "scalar_id",
            "state_equation_id",
            "model_profile",
            "continuum_profile",
            "cavity_profile",
            "nonpolar_profile",
            "coupling_id",
            "source_space_id",
            "field_space_id",
            "pairing_id",
            "coordinate_contract_id",
            "continuum_configuration_contract_id",
        ):
            object.__setattr__(self, name, _nonempty_text(getattr(self, name), name))
        if not isinstance(self.capabilities, CapabilityStatus):
            raise TypeError("capabilities must be a CapabilityStatus.")
        evidence = tuple(
            _nonempty_text(item, "evidence artifact ID")
            for item in self.evidence_artifact_ids
        )
        if len(set(evidence)) != len(evidence):
            raise ValueError("evidence_artifact_ids entries must be unique.")
        object.__setattr__(self, "evidence_artifact_ids", evidence)
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a bool.")
        admitted = bool(self.capabilities.enabled_tiers)
        if admitted and not self.enabled:
            raise ValueError("An admitted profile must be enabled.")
        if admitted and not evidence:
            raise ValueError("An admitted profile requires evidence artifact IDs.")
        if self.enabled and not self.capabilities.energy:
            raise ValueError("An enabled profile must admit scalar energy capability.")
        if evidence and not admitted:
            raise ValueError(
                "Admission evidence cannot be attached without an admitted capability."
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "scalar_id": self.scalar_id,
            "state_equation_id": self.state_equation_id,
            "model_profile": self.model_profile,
            "continuum_profile": self.continuum_profile,
            "cavity_profile": self.cavity_profile,
            "nonpolar_profile": self.nonpolar_profile,
            "coupling_id": self.coupling_id,
            "source_space_id": self.source_space_id,
            "field_space_id": self.field_space_id,
            "pairing_id": self.pairing_id,
            "coordinate_contract_id": self.coordinate_contract_id,
            "continuum_configuration_contract_id": (
                self.continuum_configuration_contract_id
            ),
            "capabilities": {
                "E": self.capabilities.energy,
                "F": self.capabilities.conservative_force,
                "H": self.capabilities.hessian,
                "V": self.capabilities.variational_functional,
                "M": self.capabilities.molecular_dynamics,
            },
            "evidence_artifact_ids": list(self.evidence_artifact_ids),
            "enabled": self.enabled,
        }


_COMMON = dict(
    model_profile=MACE_POLAR_MODEL_PROFILE_ID,
    continuum_profile="fixed-topology-linear-reciprocal-cpcm-v1",
    cavity_profile="fixed-topology-amplitude-swig-v1",
    coupling_id=EXACT_GTO_COUPLING_CANDIDATE_ID,
    source_space_id=ATOMIC_L1_SOURCE_SPACE_ID,
    field_space_id=ATOMIC_L1_FIELD_DUAL_SPACE_ID,
    pairing_id=ATOMIC_L1_PAIRING_ID,
    coordinate_contract_id=LEGACY_UNBOUND_COORDINATE_CONTRACT_ID,
    continuum_configuration_contract_id=(UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID),
    capabilities=CapabilityStatus(),
    evidence_artifact_ids=(),
    enabled=False,
)

_PROFILE_ENTRIES = (
    SolvationProfile(
        profile_id=OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1,
        scalar_id=OPERATIONAL_CPCM_ELECTROSTATIC_V1,
        state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
        nonpolar_profile="none",
        **_COMMON,
    ),
    SolvationProfile(
        profile_id=OPERATIONAL_CPCM_RADIAL_GTO_ELECTROSTATIC_PROFILE_V1,
        scalar_id=OPERATIONAL_CPCM_ELECTROSTATIC_V1,
        state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
        model_profile=MACE_POLAR_MODEL_PROFILE_ID,
        continuum_profile="fixed-topology-linear-reciprocal-cpcm-v1",
        cavity_profile="fixed-topology-amplitude-swig-v1",
        nonpolar_profile="none",
        coupling_id=MACE_POLAR_RADIAL_GTO_COUPLING_ID,
        source_space_id=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE_ID,
        field_space_id=MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE_ID,
        pairing_id=MACE_POLAR_RADIAL_GTO_PAIRING_ID,
        coordinate_contract_id=MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT_ID,
        continuum_configuration_contract_id=(WATER_CPCM_194_CONFIGURATION_CONTRACT_ID),
        capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    *(
        SolvationProfile(
            profile_id=DIAGNOSTIC_FIXED_BOX_CPCM_590_RADIAL_GTO_PROFILE_IDS[box_length],
            scalar_id=OPERATIONAL_CPCM_ELECTROSTATIC_V1,
            state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
            model_profile=MACE_POLAR_FIXED_BOX_MODEL_PROFILE_IDS[box_length],
            continuum_profile="fixed-topology-linear-reciprocal-cpcm-v1",
            cavity_profile="fixed-topology-amplitude-swig-v1",
            nonpolar_profile="none",
            coupling_id=MACE_POLAR_RADIAL_GTO_COUPLING_ID,
            source_space_id=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE_ID,
            field_space_id=MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE_ID,
            pairing_id=MACE_POLAR_RADIAL_GTO_PAIRING_ID,
            coordinate_contract_id=MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT_ID,
            continuum_configuration_contract_id=(
                WATER_CPCM_590_CONFIGURATION_CONTRACT_ID
            ),
            capabilities=CapabilityStatus(),
            evidence_artifact_ids=(),
            enabled=False,
        )
        for box_length in (32, 40, 48, 56)
    ),
    SolvationProfile(
        profile_id=DIAGNOSTIC_FIXED_BOX48_CPCM_1202_RADIAL_GTO_PROFILE_V1,
        scalar_id=OPERATIONAL_CPCM_ELECTROSTATIC_V1,
        state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
        model_profile=MACE_POLAR_FIXED_BOX_MODEL_PROFILE_IDS[48],
        continuum_profile="fixed-topology-linear-reciprocal-cpcm-v1",
        cavity_profile="fixed-topology-amplitude-swig-v1",
        nonpolar_profile="none",
        coupling_id=MACE_POLAR_RADIAL_GTO_COUPLING_ID,
        source_space_id=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE_ID,
        field_space_id=MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE_ID,
        pairing_id=MACE_POLAR_RADIAL_GTO_PAIRING_ID,
        coordinate_contract_id=MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT_ID,
        continuum_configuration_contract_id=(WATER_CPCM_1202_CONFIGURATION_CONTRACT_ID),
        capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    SolvationProfile(
        profile_id=DIAGNOSTIC_PAIR_FRAME_CPCM_RADIAL_GTO_PROFILE_V1,
        scalar_id=OPERATIONAL_CPCM_ELECTROSTATIC_V1,
        state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
        model_profile=MACE_POLAR_FIXED_BOX40_MODEL_PROFILE_ID,
        continuum_profile="fixed-topology-linear-reciprocal-cpcm-v1",
        cavity_profile="fixed-topology-amplitude-swig-v1",
        nonpolar_profile="none",
        coupling_id=MACE_POLAR_RADIAL_GTO_COUPLING_ID,
        source_space_id=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE_ID,
        field_space_id=MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE_ID,
        pairing_id=MACE_POLAR_RADIAL_GTO_PAIRING_ID,
        coordinate_contract_id=MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT_ID,
        continuum_configuration_contract_id=(
            PAIR_FRAME_WATER_CPCM_110_CONFIGURATION_CONTRACT_ID
        ),
        capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    SolvationProfile(
        profile_id=DIAGNOSTIC_DDX_DDPCM_194_RADIAL_GTO_PROFILE_V1,
        scalar_id=DIAGNOSTIC_DDX_DDPCM_RADIAL_GTO_ELECTROSTATIC_V1,
        state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
        model_profile=MACE_POLAR_MODEL_PROFILE_ID,
        continuum_profile="ddx-ddpcm-radial-gto-v1",
        cavity_profile="ddx-union-of-spheres-exposed-lebedev-v0p8p0",
        nonpolar_profile="none",
        coupling_id=MACE_POLAR_RADIAL_GTO_COUPLING_ID,
        source_space_id=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE_ID,
        field_space_id=MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE_ID,
        pairing_id=MACE_POLAR_RADIAL_GTO_PAIRING_ID,
        coordinate_contract_id=MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT_ID,
        continuum_configuration_contract_id=(
            DDX_WATER_DDPCM_194_CONFIGURATION_CONTRACT_ID
        ),
        capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    SolvationProfile(
        profile_id=DIAGNOSTIC_RADIAL_GTO_CPCM_ELECTROSTATIC_PROFILE_V1,
        scalar_id=OPERATIONAL_CPCM_ELECTROSTATIC_V1,
        state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
        model_profile=MACE_POLAR_MODEL_PROFILE_ID,
        continuum_profile="fixed-topology-linear-reciprocal-cpcm-v1",
        cavity_profile="fixed-topology-amplitude-swig-v1",
        nonpolar_profile="none",
        coupling_id=MACE_POLAR_RADIAL_GTO_COUPLING_ID,
        source_space_id=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE_ID,
        field_space_id=MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE_ID,
        pairing_id=MACE_POLAR_RADIAL_GTO_PAIRING_ID,
        coordinate_contract_id=MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT_ID,
        continuum_configuration_contract_id=(
            UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID
        ),
        capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    SolvationProfile(
        profile_id=DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_PROFILE_V1,
        scalar_id=DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1,
        state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
        model_profile=MACE_POLAR_MODEL_PROFILE_ID,
        continuum_profile="fixed-topology-linear-reciprocal-cpcm-v1",
        cavity_profile="fixed-topology-amplitude-swig-v1",
        nonpolar_profile="none",
        coupling_id=LOCAL_JET_DIAGNOSTIC_COUPLING_ID,
        source_space_id=ATOMIC_L1_SOURCE_SPACE_ID,
        field_space_id=ATOMIC_L1_FIELD_DUAL_SPACE_ID,
        pairing_id=ATOMIC_L1_PAIRING_ID,
        coordinate_contract_id=LEGACY_UNBOUND_COORDINATE_CONTRACT_ID,
        continuum_configuration_contract_id=(
            UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID
        ),
        capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    SolvationProfile(
        profile_id=OPERATIONAL_CPCM_SMDCDS_PROFILE_V1,
        scalar_id=OPERATIONAL_CPCM_SMDCDS_V1,
        state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
        nonpolar_profile="fixed-topology-smd-derived-cds-v1",
        **_COMMON,
    ),
    SolvationProfile(
        profile_id=VARIATIONAL_COMMON_FUNCTIONAL_PROFILE_V1,
        scalar_id=VARIATIONAL_COMMON_FUNCTIONAL_V1,
        state_equation_id=VARIATIONAL_STATE_EQUATION_ID,
        nonpolar_profile="profile-defined; electrostatic profile uses none",
        **_COMMON,
    ),
    SolvationProfile(
        profile_id=(VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_PROFILE_V1),
        scalar_id=(VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_V1),
        state_equation_id=VARIATIONAL_STATE_EQUATION_ID,
        model_profile=MACE_POLAR_VARIATIONAL_EFFECTIVE_SOURCE_MODEL_PROFILE_ID,
        continuum_profile="fixed-topology-linear-reciprocal-cpcm-v1",
        cavity_profile="fixed-topology-amplitude-swig-v1",
        nonpolar_profile="none",
        coupling_id=MACE_POLAR_RADIAL_GTO_COUPLING_ID,
        source_space_id=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE_ID,
        field_space_id=MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE_ID,
        pairing_id=MACE_POLAR_RADIAL_GTO_PAIRING_ID,
        coordinate_contract_id=MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT_ID,
        continuum_configuration_contract_id=WATER_CPCM_194_CONFIGURATION_CONTRACT_ID,
        capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    SolvationProfile(
        profile_id=(
            VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_PROFILE_V1
        ),
        scalar_id=(
            VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_V1
        ),
        state_equation_id=VARIATIONAL_STATE_EQUATION_ID,
        model_profile=MACE_POLAR_VARIATIONAL_EFFECTIVE_SOURCE_MODEL_PROFILE_ID,
        continuum_profile=FIXED_HARMONIC_GALERKIN_CPCM_CONTINUUM_PROFILE_ID,
        cavity_profile=FIXED_EXTERNAL_HARMONIC_CAVITY_PROFILE_ID,
        nonpolar_profile="none",
        coupling_id=MACE_POLAR_RADIAL_GTO_COUPLING_ID,
        source_space_id=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE_ID,
        field_space_id=MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE_ID,
        pairing_id=MACE_POLAR_RADIAL_GTO_PAIRING_ID,
        coordinate_contract_id=MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT_ID,
        continuum_configuration_contract_id=(
            FIXED_HARMONIC_GALERKIN_CONFIGURATION_CONTRACT_ID
        ),
        capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    SolvationProfile(
        profile_id=(
            VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1
        ),
        scalar_id=(
            VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
        ),
        state_equation_id=VARIATIONAL_STATE_EQUATION_ID,
        model_profile=MACE_POLAR_VARIATIONAL_EFFECTIVE_SOURCE_MODEL_PROFILE_ID,
        continuum_profile=SMOOTH_HARMONIC_GALERKIN_CPCM_CONTINUUM_PROFILE_ID,
        cavity_profile=SMOOTH_HARMONIC_CAVITY_PROFILE_ID,
        nonpolar_profile="none",
        coupling_id=MACE_POLAR_RADIAL_GTO_COUPLING_ID,
        source_space_id=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE_ID,
        field_space_id=MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE_ID,
        pairing_id=MACE_POLAR_RADIAL_GTO_PAIRING_ID,
        coordinate_contract_id=MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT_ID,
        continuum_configuration_contract_id=(
            SMOOTH_HARMONIC_GALERKIN_CONFIGURATION_CONTRACT_ID
        ),
        capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
)


def _validated_registry(
    entries: tuple[SolvationProfile, ...],
) -> Mapping[str, SolvationProfile]:
    registry = {entry.profile_id: entry for entry in entries}
    if len(registry) != len(entries):
        raise RuntimeError("Duplicate Route-2 profile IDs.")
    for entry in entries:
        scalar = SCALAR_REGISTRY.get(entry.scalar_id)
        if scalar is None:
            raise RuntimeError(
                f"Profile {entry.profile_id!r} names an unregistered scalar."
            )
        if entry.state_equation_id not in STATE_REGISTRY:
            raise RuntimeError(
                f"Profile {entry.profile_id!r} names an unregistered state equation."
            )
        if entry.state_equation_id != scalar.state_equation_id:
            raise RuntimeError(
                f"Profile {entry.profile_id!r} does not match its scalar state equation."
            )
        for name in ("continuum_profile", "cavity_profile", "nonpolar_profile"):
            if getattr(entry, name) != getattr(scalar, name):
                raise RuntimeError(
                    f"Profile {entry.profile_id!r} does not match scalar field {name!r}."
                )
        undeclared = set(entry.capabilities.enabled_tiers) - set(
            scalar.admitted_capabilities.enabled_tiers
        )
        if undeclared:
            raise RuntimeError(
                f"Profile {entry.profile_id!r} admits scalar-undeclared tiers: {sorted(undeclared)}"
            )
        unknown_evidence = set(entry.evidence_artifact_ids) - set(
            scalar.evidence_artifact_ids
        )
        if unknown_evidence:
            raise RuntimeError(
                f"Profile {entry.profile_id!r} names scalar-undeclared evidence: {sorted(unknown_evidence)}"
            )
        if entry.enabled and not scalar.enabled:
            raise RuntimeError(
                f"Profile {entry.profile_id!r} cannot enable a disabled scalar."
            )
        if entry.enabled and not STATE_REGISTRY[entry.state_equation_id].enabled:
            raise RuntimeError(
                f"Profile {entry.profile_id!r} cannot enable a disabled state equation."
            )
    return MappingProxyType(registry)


PROFILE_REGISTRY: Mapping[str, SolvationProfile] = _validated_registry(_PROFILE_ENTRIES)


def get_solvation_profile(profile_id: str) -> SolvationProfile:
    try:
        return PROFILE_REGISTRY[profile_id]
    except KeyError as exc:
        raise KeyError(f"Unregistered Route-2 profile: {profile_id!r}.") from exc


def profile_registry_manifest() -> dict[str, dict[str, object]]:
    return {
        profile_id: entry.as_dict() for profile_id, entry in PROFILE_REGISTRY.items()
    }


__all__ = [
    "DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_PROFILE_V1",
    "DIAGNOSTIC_FIXED_BOX40_CPCM_590_RADIAL_GTO_PROFILE_V1",
    "DIAGNOSTIC_FIXED_BOX48_CPCM_1202_RADIAL_GTO_PROFILE_V1",
    "DIAGNOSTIC_FIXED_BOX_CPCM_590_RADIAL_GTO_PROFILE_IDS",
    "DIAGNOSTIC_PAIR_FRAME_CPCM_RADIAL_GTO_PROFILE_V1",
    "DIAGNOSTIC_DDX_DDPCM_194_RADIAL_GTO_PROFILE_V1",
    "DIAGNOSTIC_RADIAL_GTO_CPCM_ELECTROSTATIC_PROFILE_V1",
    "OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1",
    "OPERATIONAL_CPCM_RADIAL_GTO_ELECTROSTATIC_PROFILE_V1",
    "OPERATIONAL_CPCM_SMDCDS_PROFILE_V1",
    "PROFILE_REGISTRY",
    "VARIATIONAL_COMMON_FUNCTIONAL_PROFILE_V1",
    "VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_PROFILE_V1",
    "VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_PROFILE_V1",
    "VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1",
    "MACE_POLAR_VARIATIONAL_EFFECTIVE_SOURCE_MODEL_PROFILE_ID",
    "SolvationProfile",
    "EXACT_GTO_COUPLING_CANDIDATE_ID",
    "LOCAL_JET_DIAGNOSTIC_COUPLING_ID",
    "MACE_POLAR_RADIAL_GTO_COUPLING_ID",
    "ATOMIC_L1_SOURCE_SPACE_ID",
    "ATOMIC_L1_FIELD_DUAL_SPACE_ID",
    "ATOMIC_L1_PAIRING_ID",
    "MACE_POLAR_RADIAL_GTO_SOURCE_SPACE_ID",
    "MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE_ID",
    "MACE_POLAR_RADIAL_GTO_PAIRING_ID",
    "MACE_POLAR_MODEL_PROFILE_ID",
    "MACE_POLAR_FIXED_BOX40_MODEL_PROFILE_ID",
    "MACE_POLAR_FIXED_BOX_MODEL_PROFILE_IDS",
    "MACE_POLAR_MOLECULAR_REALSPACE_EVALUATOR_ID",
    "MACE_POLAR_FORCED_RECIPROCAL_FIXED_BOX40_EVALUATOR_ID",
    "MACE_POLAR_FORCED_RECIPROCAL_FIXED_BOX_EVALUATOR_IDS",
    "LEGACY_UNBOUND_COORDINATE_CONTRACT_ID",
    "MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT_ID",
    "UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID",
    "WATER_CPCM_194_CONFIGURATION_CONTRACT_ID",
    "WATER_CPCM_590_CONFIGURATION_CONTRACT_ID",
    "WATER_CPCM_1202_CONFIGURATION_CONTRACT_ID",
    "PAIR_FRAME_WATER_CPCM_110_CONFIGURATION_CONTRACT_ID",
    "DDX_WATER_DDPCM_194_CONFIGURATION_CONTRACT_ID",
    "FIXED_EXTERNAL_HARMONIC_CAVITY_PROFILE_ID",
    "FIXED_HARMONIC_GALERKIN_CPCM_CONTINUUM_PROFILE_ID",
    "FIXED_HARMONIC_GALERKIN_CONFIGURATION_CONTRACT_ID",
    "SMOOTH_HARMONIC_CAVITY_PROFILE_ID",
    "SMOOTH_HARMONIC_GALERKIN_CPCM_CONTINUUM_PROFILE_ID",
    "SMOOTH_HARMONIC_GALERKIN_CONFIGURATION_CONTRACT_ID",
    "get_solvation_profile",
    "profile_registry_manifest",
]
