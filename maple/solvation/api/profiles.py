"""Authoritative, immutable Route-2 profile and admission registry."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .capabilities import CapabilityStatus, ExecutionCapability, ExecutionStatus
from .scalar_registry import (
    DIAGNOSTIC_DDX_DDPCM_RADIAL_GTO_ELECTROSTATIC_V1,
    DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1,
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_V1,
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1,
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_V1,
    MACE_MDP_POLAR_HYBRID_HARMONIC_FORCE_ADMISSION_EVIDENCE_ID,
    OPERATIONAL_CPCM_ELECTROSTATIC_V1,
    OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    OPERATIONAL_CPCM_SMDCDS_V1,
    PURE_MACEPOLAR_POINT_L1_MNSOL505_DEVELOPMENT_EVIDENCE_ID,
    SCALAR_REGISTRY,
    VARIATIONAL_COMMON_FUNCTIONAL_V1,
    VARIATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_V1,
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_V1,
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
)
from .state_registry import (
    MACE_MDP_POLAR_HYBRID_HARMONIC_STATE_EQUATION_ID,
    MACE_MDP_POLAR_HYBRID_STATE_EQUATION_ID,
    OPERATIONAL_STATE_EQUATION_ID,
    PURE_MACEPOLAR_FROZEN_SOURCE_STATE_EQUATION_ID,
    STATE_REGISTRY,
    VARIATIONAL_STATE_EQUATION_ID,
)

EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_PROFILE_V1 = (
    "route2-profile-experimental-pure-macepolar-frozen-point-l1-ddpcm-smd-v1"
)

OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1 = (
    "route2-profile-operational-cpcm-fixedtopology-electrostatic-v1"
)
OPERATIONAL_CPCM_RADIAL_GTO_ELECTROSTATIC_PROFILE_V1 = (
    "route2-profile-operational-cpcm-fixedtopology-radialgto-electrostatic-v1"
)
OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1 = (
    "route2-profile-operational-macepolar-analytic-gaussian-multipole-"
    "smoothharmonicgalerkin-cpcm-v1"
)
EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_PROFILE_V1 = (
    "route2-profile-experimental-macemdppoint-macepolarinduced-"
    "pcmsolver-electrostatic-v1"
)
EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_PROFILE_V1 = (
    "route2-profile-experimental-macemdppoint-macepolarinduced-"
    "smoothharmonicgalerkin-electrostatic-v1"
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
VARIATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1 = (
    "route2-profile-variational-macepolar-analytic-gaussian-multipole-"
    "energygradient-smoothharmonicgalerkin-cpcm-v1"
)
MACE_POLAR_VARIATIONAL_EFFECTIVE_SOURCE_MODEL_PROFILE_ID = (
    "mace-polar-route2-variational-effective-source-v1"
)
MACE_POLAR_VARIATIONAL_ANALYTIC_GAUSSIAN_MULTIPOLE_MODEL_PROFILE_ID = (
    "mace-polar-route2-variational-effective-source-" "analytic-gaussian-multipole-v1"
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
MACE_MDP_POLAR_HYBRID_COUPLING_ID = (
    "route2-coupling-macemdppoint-macepolarinduced-pcmsolver-radialreceiver-v1"
)
MACE_MDP_POLAR_HYBRID_HARMONIC_COUPLING_ID = (
    "route2-coupling-macemdppoint-macepolarinduced-"
    "smoothharmonicgalerkin-radialreceiver-v1"
)
MACE_MDP_POLAR_HYBRID_MODEL_PROFILE_ID = (
    "route2-research-mace-mdp-point-permanent-macepolar-gto-induced-v1"
)
MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE_ID = (
    "maple.route2.mace-polar-native-radial-field-space.v1"
)
MACE_MDP_POLAR_HYBRID_NO_DUALITY_ID = (
    "maple.route2.hybrid-distinct-source-kernels-no-source-field-duality.v1"
)
MACE_MDP_POLAR_HYBRID_NUMERICAL_FORCE_COORDINATE_CONTRACT_ID = (
    "maple.route2.hybrid-full-scalar-richardson-coordinate-gradient.v1"
)
PCMSOLVER_CONTENT_ADDRESSED_CONFIGURATION_CONTRACT_ID = (
    "maple.route2.continuum-configuration.explicit-pcmsolver-input-sha256.v1"
)
LEGACY_UNBOUND_COORDINATE_CONTRACT_ID = (
    "maple.route2.legacy-profile-coordinate-scales-unbound.v1"
)
MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT_ID = (
    "maple.route2.mace-polar-radial-gto-linear-charge-coordinates.v1"
)
MACE_POLAR_POINT_L1_DDX_COORDINATE_CONTRACT_ID = (
    "maple.route2.mace-polar-point-l1-source-embedding-coordinate-gradient.v1"
)
MACE_POLAR_POINT_L1_SMD_DDPCM_CONFIGURATION_CONTRACT_ID = (
    "maple.route2.continuum-configuration."
    "smd-solvent-radii-ddpcm-l15-lebedev1202-tol1e-12-nproc1-"
    "pyscf-smd-cds.v1"
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
MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_MODEL_PROFILE_ID = (
    "mace-polar-route2-analytic-gaussian-multipole-realspace-contract-v1"
)
MACE_POLAR_FIXED_BOX_MODEL_PROFILE_IDS = MappingProxyType(
    {
        box_length: f"mace-polar-route2-source-field-fixed-box{box_length}-contract-v1"
        for box_length in (32, 40, 48, 56)
    }
)
MACE_POLAR_FIXED_BOX40_MODEL_PROFILE_ID = MACE_POLAR_FIXED_BOX_MODEL_PROFILE_IDS[40]
MACE_POLAR_MOLECULAR_REALSPACE_EVALUATOR_ID = "graph-longrange-molecular-realspace-v1"
MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID = (
    "graph-longrange-analytic-gaussian-multipole-realspace-v1"
)
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
    experimental_execution: ExecutionStatus = ExecutionStatus()
    experimental_operation_policies: tuple[tuple[ExecutionCapability, str], ...] = ()
    experimental_evidence_artifact_ids: tuple[str, ...] = ()
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
        if not isinstance(self.experimental_execution, ExecutionStatus):
            raise TypeError("experimental_execution must be an ExecutionStatus.")
        policies: list[tuple[ExecutionCapability, str]] = []
        for item in self.experimental_operation_policies:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "experimental_operation_policies entries must be "
                    "(ExecutionCapability, policy_id) pairs."
                )
            operation, policy_id = item
            if not isinstance(operation, ExecutionCapability):
                raise TypeError(
                    "experimental operation policy keys must be "
                    "ExecutionCapability values."
                )
            if not isinstance(policy_id, str) or not policy_id.strip():
                raise ValueError("experimental operation policy IDs must be non-empty.")
            if not self.experimental_execution.supports(operation):
                raise ValueError(
                    f"Policy declared for unavailable operation {operation.value!r}."
                )
            policies.append((operation, policy_id.strip()))
        if len({operation for operation, _ in policies}) != len(policies):
            raise ValueError("Experimental operation policy keys must be unique.")
        object.__setattr__(self, "experimental_operation_policies", tuple(policies))
        experimental_evidence = tuple(
            _nonempty_text(item, "experimental evidence artifact ID")
            for item in self.experimental_evidence_artifact_ids
        )
        if len(set(experimental_evidence)) != len(experimental_evidence):
            raise ValueError(
                "experimental_evidence_artifact_ids entries must be unique."
            )
        object.__setattr__(
            self,
            "experimental_evidence_artifact_ids",
            experimental_evidence,
        )
        experimental = bool(self.experimental_execution.available_operations)
        if experimental and not experimental_evidence:
            raise ValueError(
                "An experimental execution surface requires evidence artifact IDs."
            )
        if experimental_evidence and not experimental:
            raise ValueError(
                "Experimental evidence cannot be attached without an execution surface."
            )
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
            "experimental_execution": self.experimental_execution.as_dict(),
            "experimental_operation_policies": {
                operation.value: policy_id
                for operation, policy_id in self.experimental_operation_policies
            },
            "experimental_evidence_artifact_ids": list(
                self.experimental_evidence_artifact_ids
            ),
            "experimental_enabled": bool(
                self.experimental_execution.available_operations
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
        profile_id=EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_PROFILE_V1,
        scalar_id=EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_V1,
        state_equation_id=PURE_MACEPOLAR_FROZEN_SOURCE_STATE_EQUATION_ID,
        model_profile=MACE_POLAR_MODEL_PROFILE_ID,
        continuum_profile="ddx-ddpcm-macepolar-point-l1-embedding-v1",
        cavity_profile="ddx-union-of-spheres-exposed-lebedev-v0p8p0",
        nonpolar_profile="pyscf-2.13.1-smd-cds-legacy-v1",
        coupling_id=MACE_POLAR_RADIAL_GTO_COUPLING_ID,
        source_space_id=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE_ID,
        field_space_id=MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE_ID,
        pairing_id=MACE_POLAR_RADIAL_GTO_PAIRING_ID,
        coordinate_contract_id=MACE_POLAR_POINT_L1_DDX_COORDINATE_CONTRACT_ID,
        continuum_configuration_contract_id=(
            MACE_POLAR_POINT_L1_SMD_DDPCM_CONFIGURATION_CONTRACT_ID
        ),
        experimental_execution=ExecutionStatus(
            energy=True,
            force=True,
            molecular_virial=True,
            hessian_vector_product=True,
            hessian=True,
        ),
        experimental_operation_policies=(
            (
                ExecutionCapability.HESSIAN_VECTOR_PRODUCT,
                "observed-components-only-experimental-v1",
            ),
            (
                ExecutionCapability.HESSIAN,
                "observed-components-only-experimental-v1",
            ),
        ),
        experimental_evidence_artifact_ids=(
            PURE_MACEPOLAR_POINT_L1_MNSOL505_DEVELOPMENT_EVIDENCE_ID,
        ),
        capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
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
    SolvationProfile(
        profile_id=(
            OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1
        ),
        scalar_id=(
            OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
        ),
        state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
        model_profile=MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_MODEL_PROFILE_ID,
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
    SolvationProfile(
        profile_id=(
            EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_PROFILE_V1
        ),
        scalar_id=(EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_V1),
        state_equation_id=MACE_MDP_POLAR_HYBRID_STATE_EQUATION_ID,
        model_profile=MACE_MDP_POLAR_HYBRID_MODEL_PROFILE_ID,
        continuum_profile="pcmsolver-symmetric-external-mep-electrostatic-v1",
        cavity_profile="pcmsolver-input-defined-gepol-cavity-v1",
        nonpolar_profile="none",
        coupling_id=MACE_MDP_POLAR_HYBRID_COUPLING_ID,
        source_space_id=ATOMIC_L1_SOURCE_SPACE_ID,
        field_space_id=MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE_ID,
        pairing_id=MACE_MDP_POLAR_HYBRID_NO_DUALITY_ID,
        coordinate_contract_id=(
            MACE_MDP_POLAR_HYBRID_NUMERICAL_FORCE_COORDINATE_CONTRACT_ID
        ),
        continuum_configuration_contract_id=(
            PCMSOLVER_CONTENT_ADDRESSED_CONFIGURATION_CONTRACT_ID
        ),
        capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    SolvationProfile(
        profile_id=(
            EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_PROFILE_V1
        ),
        scalar_id=(
            EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1
        ),
        state_equation_id=MACE_MDP_POLAR_HYBRID_HARMONIC_STATE_EQUATION_ID,
        model_profile=MACE_MDP_POLAR_HYBRID_MODEL_PROFILE_ID,
        continuum_profile=SMOOTH_HARMONIC_GALERKIN_CPCM_CONTINUUM_PROFILE_ID,
        cavity_profile=SMOOTH_HARMONIC_CAVITY_PROFILE_ID,
        nonpolar_profile="none",
        coupling_id=MACE_MDP_POLAR_HYBRID_HARMONIC_COUPLING_ID,
        source_space_id=ATOMIC_L1_SOURCE_SPACE_ID,
        field_space_id=MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE_ID,
        pairing_id=MACE_MDP_POLAR_HYBRID_NO_DUALITY_ID,
        coordinate_contract_id=(
            MACE_MDP_POLAR_HYBRID_NUMERICAL_FORCE_COORDINATE_CONTRACT_ID
        ),
        continuum_configuration_contract_id=(
            SMOOTH_HARMONIC_GALERKIN_CONFIGURATION_CONTRACT_ID
        ),
        capabilities=CapabilityStatus(energy=True, conservative_force=True),
        evidence_artifact_ids=(
            MACE_MDP_POLAR_HYBRID_HARMONIC_FORCE_ADMISSION_EVIDENCE_ID,
        ),
        enabled=True,
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
    SolvationProfile(
        profile_id=(
            VARIATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1
        ),
        scalar_id=(
            VARIATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
        ),
        state_equation_id=VARIATIONAL_STATE_EQUATION_ID,
        model_profile=(
            MACE_POLAR_VARIATIONAL_ANALYTIC_GAUSSIAN_MULTIPOLE_MODEL_PROFILE_ID
        ),
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
        undeclared_execution = set(
            entry.experimental_execution.available_operations
        ) - set(scalar.experimental_execution.available_operations)
        if undeclared_execution:
            raise RuntimeError(
                f"Profile {entry.profile_id!r} exposes scalar-undeclared experimental "
                f"operations: {sorted(undeclared_execution)}"
            )
        undeclared_policies = set(entry.experimental_operation_policies) - set(
            scalar.experimental_operation_policies
        )
        if undeclared_policies:
            raise RuntimeError(
                f"Profile {entry.profile_id!r} exposes scalar-undeclared experimental "
                f"operation policies: {sorted(undeclared_policies)}"
            )
        unknown_experimental_evidence = set(
            entry.experimental_evidence_artifact_ids
        ) - set(scalar.experimental_evidence_artifact_ids)
        if unknown_experimental_evidence:
            raise RuntimeError(
                f"Profile {entry.profile_id!r} names scalar-undeclared experimental "
                f"evidence: {sorted(unknown_experimental_evidence)}"
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
    "EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_PROFILE_V1",
    "EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_PROFILE_V1",
    "EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_PROFILE_V1",
    "OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1",
    "OPERATIONAL_CPCM_RADIAL_GTO_ELECTROSTATIC_PROFILE_V1",
    "OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1",
    "OPERATIONAL_CPCM_SMDCDS_PROFILE_V1",
    "PROFILE_REGISTRY",
    "VARIATIONAL_COMMON_FUNCTIONAL_PROFILE_V1",
    "VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_PROFILE_V1",
    "VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_PROFILE_V1",
    "VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1",
    "VARIATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1",
    "MACE_POLAR_VARIATIONAL_EFFECTIVE_SOURCE_MODEL_PROFILE_ID",
    "MACE_POLAR_VARIATIONAL_ANALYTIC_GAUSSIAN_MULTIPOLE_MODEL_PROFILE_ID",
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
    "MACE_MDP_POLAR_HYBRID_COUPLING_ID",
    "MACE_MDP_POLAR_HYBRID_HARMONIC_COUPLING_ID",
    "MACE_MDP_POLAR_HYBRID_MODEL_PROFILE_ID",
    "MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE_ID",
    "MACE_MDP_POLAR_HYBRID_NO_DUALITY_ID",
    "MACE_MDP_POLAR_HYBRID_NUMERICAL_FORCE_COORDINATE_CONTRACT_ID",
    "PCMSOLVER_CONTENT_ADDRESSED_CONFIGURATION_CONTRACT_ID",
    "MACE_POLAR_MODEL_PROFILE_ID",
    "MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_MODEL_PROFILE_ID",
    "MACE_POLAR_FIXED_BOX40_MODEL_PROFILE_ID",
    "MACE_POLAR_FIXED_BOX_MODEL_PROFILE_IDS",
    "MACE_POLAR_MOLECULAR_REALSPACE_EVALUATOR_ID",
    "MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID",
    "MACE_POLAR_FORCED_RECIPROCAL_FIXED_BOX40_EVALUATOR_ID",
    "MACE_POLAR_FORCED_RECIPROCAL_FIXED_BOX_EVALUATOR_IDS",
    "LEGACY_UNBOUND_COORDINATE_CONTRACT_ID",
    "MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT_ID",
    "MACE_POLAR_POINT_L1_DDX_COORDINATE_CONTRACT_ID",
    "MACE_POLAR_POINT_L1_SMD_DDPCM_CONFIGURATION_CONTRACT_ID",
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
