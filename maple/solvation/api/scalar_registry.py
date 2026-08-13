"""Machine-readable registry of versioned Route-2 scalar definitions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .capabilities import CapabilityStatus
from .state_registry import OPERATIONAL_STATE_EQUATION_ID, VARIATIONAL_STATE_EQUATION_ID

OPERATIONAL_CPCM_ELECTROSTATIC_V1 = (
    "route2-operational-cpcm-fixedtopology-electrostatic-v1"
)
DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1 = (
    "route2-diagnostic-localjet-cpcm-fixedtopology-electrostatic-v1"
)
DIAGNOSTIC_DDX_DDPCM_RADIAL_GTO_ELECTROSTATIC_V1 = (
    "route2-diagnostic-ddx-ddpcm-radialgto-electrostatic-v1"
)
DIAGNOSTIC_DDX_DDCOSMO_RADIAL_GTO_ELECTROSTATIC_V1 = (
    "route2-diagnostic-ddx-ddcosmo-radialgto-electrostatic-v1"
)
OPERATIONAL_CPCM_SMDCDS_V1 = "route2-operational-cpcm-fixedtopology-smdcds-v1"
VARIATIONAL_COMMON_FUNCTIONAL_V1 = "route2-variational-common-functional-v1"
VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_V1 = (
    "route2-variational-macepolar-energygradient-fixedcavity-cpcm-v1"
)


@dataclass(frozen=True, slots=True)
class ScalarDefinition:
    scalar_id: str
    exact_formula: str
    implementation_entry_point: str
    included_components: tuple[str, ...]
    excluded_components: tuple[str, ...]
    source_representation: str
    field_convention: str
    continuum_profile: str
    cavity_profile: str
    nonpolar_profile: str
    state_equation_id: str
    derivative_route: str
    admitted_capabilities: CapabilityStatus = CapabilityStatus()
    evidence_artifact_ids: tuple[str, ...] = ()
    enabled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "scalar_id",
            "exact_formula",
            "implementation_entry_point",
            "source_representation",
            "field_convention",
            "continuum_profile",
            "cavity_profile",
            "nonpolar_profile",
            "state_equation_id",
            "derivative_route",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        for name in (
            "included_components",
            "excluded_components",
            "evidence_artifact_ids",
        ):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{name} must contain only non-empty strings.")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} entries must be unique.")
            object.__setattr__(self, name, values)
        overlap = set(self.included_components) & set(self.excluded_components)
        if overlap:
            raise ValueError(
                f"Components cannot be both included and excluded: {sorted(overlap)}"
            )
        if not isinstance(self.admitted_capabilities, CapabilityStatus):
            raise TypeError("admitted_capabilities must be a CapabilityStatus.")
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a bool.")
        admitted = bool(self.admitted_capabilities.enabled_tiers)
        if admitted and not self.enabled:
            raise ValueError("An admitted scalar must be enabled.")
        if admitted and not self.evidence_artifact_ids:
            raise ValueError("An admitted scalar requires evidence artifact IDs.")
        if self.enabled and not self.admitted_capabilities.energy:
            raise ValueError("An enabled scalar must admit energy capability.")
        if self.evidence_artifact_ids and not admitted:
            raise ValueError(
                "Admission evidence cannot be attached without an admitted capability."
            )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable registry record."""

        return {
            "scalar_id": self.scalar_id,
            "exact_formula": self.exact_formula,
            "implementation_entry_point": self.implementation_entry_point,
            "included_components": list(self.included_components),
            "excluded_components": list(self.excluded_components),
            "source_representation": self.source_representation,
            "field_convention": self.field_convention,
            "continuum_profile": self.continuum_profile,
            "cavity_profile": self.cavity_profile,
            "nonpolar_profile": self.nonpolar_profile,
            "state_equation_id": self.state_equation_id,
            "derivative_route": self.derivative_route,
            "admitted_capabilities": {
                "E": self.admitted_capabilities.energy,
                "F": self.admitted_capabilities.conservative_force,
                "H": self.admitted_capabilities.hessian,
                "V": self.admitted_capabilities.variational_functional,
                "M": self.admitted_capabilities.molecular_dynamics,
            },
            "evidence_artifact_ids": list(self.evidence_artifact_ids),
            "enabled": self.enabled,
        }


_COMMON = dict(
    source_representation=(
        "profile-bound atom-centred l<=1 electrostatic source; exact component, "
        "radial-basis, unit, and charge-functional identity is registered by profile"
    ),
    field_convention=(
        "profile-bound positive energy-dual field with pairing c^T Q(R) u"
    ),
    continuum_profile="fixed-topology-linear-reciprocal-cpcm-v1",
    cavity_profile="fixed-topology-amplitude-swig-v1",
    admitted_capabilities=CapabilityStatus(),
    evidence_artifact_ids=(),
    enabled=False,
)

_SCALAR_ENTRIES = (
    ScalarDefinition(
        scalar_id=OPERATIONAL_CPCM_ELECTROSTATIC_V1,
        exact_formula=(
            "E_op(R)=Phi_op(R,y*(R)); Phi_op=E_vac(R)+"
            "1/2<c_ref(R)+T(R)y,P_R(c_ref(R)+T(R)y)>_Q; G_np=0"
        ),
        included_components=("vacuum_energy", "cpcm_half_coupling_electrostatic"),
        excluded_components=(
            "field_conditioned_model_energy_difference",
            "nonpolar_smd_cds",
        ),
        nonpolar_profile="none",
        state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
        implementation_entry_point=(
            "maple.solvation.coupling.energy:OperationalElectrostaticScalar"
        ),
        derivative_route="implicit adjoint total derivative of this scalar along y*(R)",
        **_COMMON,
    ),
    ScalarDefinition(
        scalar_id=DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1,
        exact_formula=(
            "E_diag(R)=Phi_diag(R,y*(R)); Phi_diag=E_vac(R)+"
            "1/2<c_ref(R)+T(R)y,P_R^local-jet(c_ref(R)+T(R)y)>_Q; G_np=0"
        ),
        included_components=(
            "vacuum_energy",
            "local_jet_cpcm_half_coupling_electrostatic",
        ),
        excluded_components=(
            "checkpoint_native_exact_gto_coupling",
            "field_conditioned_model_energy_difference",
            "nonpolar_smd_cds",
        ),
        nonpolar_profile="none",
        state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
        implementation_entry_point=(
            "maple.solvation.coupling.energy:OperationalElectrostaticScalar"
        ),
        derivative_route=(
            "implicit-adjoint diagnostic only; public E/F/H/V/M disabled because "
            "the local-jet coupling is not the checkpoint-native exact-GTO mainline"
        ),
        **_COMMON,
    ),
    ScalarDefinition(
        scalar_id=DIAGNOSTIC_DDX_DDPCM_RADIAL_GTO_ELECTROSTATIC_V1,
        exact_formula=(
            "E_diag(R)=Phi_diag(R,y*(R)); Phi_diag=E_vac(R)+"
            "1/2<c_ref(R)+T(R)y,P_R^ddPCM(c_ref(R)+T(R)y)>_Q; G_np=0"
        ),
        implementation_entry_point=(
            "maple.solvation.coupling.energy:OperationalElectrostaticScalar"
        ),
        included_components=(
            "vacuum_energy",
            "ddx_ddpcm_half_coupling_electrostatic",
        ),
        excluded_components=(
            "field_conditioned_model_energy_difference",
            "nonpolar_smd_cds",
            "rotation_admission",
        ),
        source_representation=(
            "complete two-width atom-centred radial-GTO l<=1 source mapped "
            "jointly to ddX psi and exposed-node phi"
        ),
        field_convention=(
            "positive energy-dual radial-GTO field from the derivative of the "
            "same ddPCM half-coupling scalar"
        ),
        continuum_profile="ddx-ddpcm-radial-gto-v1",
        cavity_profile="ddx-union-of-spheres-exposed-lebedev-v0p8p0",
        nonpolar_profile="none",
        state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
        derivative_route=(
            "implicit-adjoint diagnostic only; finite laboratory-frame quadrature "
            "is not structurally rotation equivariant"
        ),
        admitted_capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    ScalarDefinition(
        scalar_id=DIAGNOSTIC_DDX_DDCOSMO_RADIAL_GTO_ELECTROSTATIC_V1,
        exact_formula=(
            "E_diag(R)=Phi_diag(R,y*(R)); Phi_diag=E_vac(R)+"
            "1/2<c_ref(R)+T(R)y,P_R^ddCOSMO(c_ref(R)+T(R)y)>_Q; G_np=0"
        ),
        implementation_entry_point=(
            "maple.solvation.coupling.energy:OperationalElectrostaticScalar"
        ),
        included_components=(
            "vacuum_energy",
            "ddx_ddcosmo_half_coupling_electrostatic",
        ),
        excluded_components=(
            "field_conditioned_model_energy_difference",
            "nonpolar_smd_cds",
            "rotation_admission",
            "registered_release_profile",
        ),
        source_representation=(
            "complete two-width atom-centred radial-GTO l<=1 source mapped "
            "jointly to ddX psi and exposed-node phi"
        ),
        field_convention=(
            "positive energy-dual radial-GTO field from the derivative of the "
            "same dielectric-scaled ddCOSMO half-coupling scalar"
        ),
        continuum_profile="ddx-ddcosmo-radial-gto-v1",
        cavity_profile="ddx-union-of-spheres-exposed-lebedev-v0p8p0",
        nonpolar_profile="none",
        state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
        derivative_route=(
            "unregistered diagnostic only; finite laboratory-frame quadrature "
            "is not structurally rotation equivariant"
        ),
        admitted_capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    ScalarDefinition(
        scalar_id=OPERATIONAL_CPCM_SMDCDS_V1,
        exact_formula=(
            "E_op(R)=Phi_op(R,y*(R)); Phi_op=E_vac(R)+"
            "1/2<c_ref(R)+T(R)y,P_R(c_ref(R)+T(R)y)>_Q+G_np^SMD-CDS(R,c)"
        ),
        included_components=(
            "vacuum_energy",
            "cpcm_half_coupling_electrostatic",
            "fixed_topology_smd_cds",
        ),
        excluded_components=("field_conditioned_model_energy_difference",),
        nonpolar_profile="fixed-topology-smd-derived-cds-v1",
        state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
        implementation_entry_point="disabled:not-implemented-fixedtopology-smdcds-v1",
        derivative_route="implicit adjoint total derivative; disabled pending same-scalar CDS force gate",
        **_COMMON,
    ),
    ScalarDefinition(
        scalar_id=VARIATIONAL_COMMON_FUNCTIONAL_V1,
        exact_formula=(
            "F_var(R,c)=Gamma_theta(R,c)+G_pcm(R,c)+G_np(R,c); "
            "Gamma_theta=stat_u[E_theta(R,u)-<c,u>_Q]"
        ),
        included_components=(
            "electronic_legendre_functional",
            "continuum_energy",
            "declared_nonpolar_energy",
        ),
        excluded_components=("independently_assembled_response_correction",),
        nonpolar_profile="profile-defined; electrostatic profile uses none",
        state_equation_id=VARIATIONAL_STATE_EQUATION_ID,
        implementation_entry_point="disabled:not-implemented-common-functional-v1",
        derivative_route="stationary envelope derivative; disabled until every strict-variational gate passes",
        **_COMMON,
    ),
    ScalarDefinition(
        scalar_id=VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_V1,
        exact_formula=(
            "candidate equations: M_var=Q^-T dE_anc/du and "
            "u=grad_Q G_cpcm_fixed(c); G_cpcm_fixed=1/2<c,P_R c>_Q; "
            "G_np=0; the combined stationary scalar "
            "F_var=Gamma_anc+G_cpcm_fixed remains unimplemented and disabled"
        ),
        included_components=(
            "macepolar_anchored_field_energy_candidate",
            "fixed_cavity_reciprocal_cpcm_target",
        ),
        excluded_components=(
            "original_four_channel_density_as_variational_source",
            "source_dependent_cavity",
            "nonpolar_smd_cds",
        ),
        source_representation=(
            "complete eight-channel MACE-POLAR field-energy-conjugate effective source"
        ),
        field_convention=(
            "positive energy-dual reduced radial-GTO field; fixed charge and "
            "constant-potential gauge separated"
        ),
        continuum_profile="fixed-topology-linear-reciprocal-cpcm-v1",
        cavity_profile="fixed-topology-amplitude-swig-v1",
        nonpolar_profile="none",
        state_equation_id=VARIATIONAL_STATE_EQUATION_ID,
        implementation_entry_point=(
            "disabled:model and continuum scalar-first candidates only; "
            "combined stationary scalar not implemented"
        ),
        derivative_route=(
            "stationary envelope derivative; disabled pending sign, gauge, "
            "passivity, root, continuum, coordinate, rotation, and release gates"
        ),
        admitted_capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
)

SCALAR_REGISTRY: Mapping[str, ScalarDefinition] = MappingProxyType(
    {entry.scalar_id: entry for entry in _SCALAR_ENTRIES}
)
if len(SCALAR_REGISTRY) != len(
    _SCALAR_ENTRIES
):  # pragma: no cover - import-time invariant
    raise RuntimeError("Duplicate Route-2 scalar IDs.")


def get_scalar_definition(scalar_id: str) -> ScalarDefinition:
    try:
        return SCALAR_REGISTRY[scalar_id]
    except KeyError as exc:
        raise KeyError(f"Unregistered Route-2 scalar: {scalar_id!r}.") from exc


def scalar_registry_manifest() -> dict[str, dict[str, object]]:
    """Return a detached JSON-serializable snapshot keyed by scalar ID."""

    return {scalar_id: entry.as_dict() for scalar_id, entry in SCALAR_REGISTRY.items()}


__all__ = [
    "DIAGNOSTIC_DDX_DDCOSMO_RADIAL_GTO_ELECTROSTATIC_V1",
    "DIAGNOSTIC_DDX_DDPCM_RADIAL_GTO_ELECTROSTATIC_V1",
    "DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1",
    "OPERATIONAL_CPCM_ELECTROSTATIC_V1",
    "OPERATIONAL_CPCM_SMDCDS_V1",
    "SCALAR_REGISTRY",
    "VARIATIONAL_COMMON_FUNCTIONAL_V1",
    "VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_V1",
    "ScalarDefinition",
    "get_scalar_definition",
    "scalar_registry_manifest",
]
