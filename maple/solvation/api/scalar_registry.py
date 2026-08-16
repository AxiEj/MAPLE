"""Machine-readable registry of versioned Route-2 scalar definitions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .capabilities import CapabilityStatus
from .state_registry import (
    MACE_MDP_POLAR_HYBRID_HARMONIC_STATE_EQUATION_ID,
    MACE_MDP_POLAR_HYBRID_STATE_EQUATION_ID,
    OPERATIONAL_STATE_EQUATION_ID,
    SEPARATED_OPERATIONAL_STATE_EQUATION_ID,
    VARIATIONAL_STATE_EQUATION_ID,
)

OPERATIONAL_CPCM_ELECTROSTATIC_V1 = (
    "route2-operational-cpcm-fixedtopology-electrostatic-v1"
)
OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_V1 = (
    "route2-operational-macepolar-analytic-gaussian-multipole-"
    "smoothharmonicgalerkin-cpcm-v1"
)
OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_GALERKIN_CPCM_V1 = (
    "route2-operational-macepolar-source4-nativefield8-"
    "smoothharmonicgalerkin-cpcm-phi0-v1"
)
OPERATIONAL_MACEPOLAR_SEPARATED_PHI1_SMOOTH_HARMONIC_GALERKIN_CPCM_V1 = (
    "route2-operational-macepolar-source4-nativefield8-"
    "smoothharmonicgalerkin-cpcm-conditioneddelta-phi1-v1"
)
EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_V1 = (
    "route2-experimental-macemdppoint-macepolarinduced-" "pcmsolver-electrostatic-v1"
)
EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1 = (
    "route2-experimental-macemdppoint-macepolarinduced-"
    "smoothharmonicgalerkin-electrostatic-v1"
)
MACE_MDP_POLAR_HYBRID_HARMONIC_FORCE_ADMISSION_EVIDENCE_ID = (
    "route2-mace-mdp-polar-hybrid-harmonic-force-admission-replicated-v1"
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
VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_V1 = (
    "route2-variational-macepolar-energygradient-fixedcavity-"
    "harmonicgalerkin-cpcm-v1"
)
VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1 = (
    "route2-variational-macepolar-energygradient-" "smoothharmonicgalerkin-cpcm-v1"
)
VARIATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1 = (
    "route2-variational-macepolar-analytic-gaussian-multipole-"
    "energygradient-smoothharmonicgalerkin-cpcm-v1"
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
        scalar_id=(
            OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
        ),
        exact_formula=(
            "E_op(R)=E_vac^analytic(R)+G_harm(R,c*(R)); "
            "c*=M_orig^analytic(R,u*), u*=grad_Q G_harm(R,c*); "
            "G_harm=-1/2 (S(R)c)^T A(R)^-1 (S(R)c)="
            "1/2<c,grad_Q G_harm(R,c)>_Q; G_np=0"
        ),
        implementation_entry_point=(
            "maple.solvation.coupling.energy:OperationalElectrostaticScalar"
        ),
        included_components=(
            "macepolar_analytic_gaussian_multipole_vacuum_energy",
            "smooth_weighted_harmonic_galerkin_half_coupling_electrostatic",
        ),
        excluded_components=(
            "field_conditioned_model_energy_difference",
            "energy_gradient_variational_effective_source",
            "sharp_union_of_spheres_identity",
            "nonpolar_smd_cds",
        ),
        source_representation=(
            "original four-channel MACE-POLAR density head embedded in the "
            "sigma=1.5 radial-GTO block; sigma=3.0 source block is zero before "
            "the fixed-total-charge affine projection"
        ),
        field_convention=(
            "positive energy-dual eight-channel radial-GTO field with pairing "
            "c^T Q u; constant-potential gauge is separated from the reduced root"
        ),
        continuum_profile="smooth-weighted-harmonic-galerkin-cpcm-candidate-v1",
        cavity_profile="smooth-weighted-overlap-harmonic-cavity-candidate-v1",
        nonpolar_profile="none",
        state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
        derivative_route=(
            "implicit adjoint total derivative exists internally; disabled and "
            "scientifically stopped because the unchanged original source failed "
            "the frozen four-case matched QM/PCMSolver physical gate; the later "
            "far-field pass authorized one separately identified radial-source "
            "research profile, which then failed all four matched PCMSolver "
            "energy cases and is terminally closed"
        ),
        admitted_capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    ScalarDefinition(
        scalar_id=(
            OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
        ),
        exact_formula=(
            "Phi0(R,y*)=E_vac(R)-1/2 (B_R c*)^T A_R^-1(B_R c*); "
            "c*=c_ref+Ty*, A_R sigma*=B_R c*, u*=L_R sigma*, "
            "c*=Pi_q M_orig(R,u*)"
        ),
        implementation_entry_point=(
            "maple.solvation.coupling.separated_ledgers:FrozenVacuumContinuumLedger"
        ),
        included_components=(
            "macepolar_vacuum_energy",
            "smooth_harmonic_continuum_stationary_energy",
        ),
        excluded_components=(
            "field_conditioned_intrinsic_energy",
            "solute_internal_polarization_cost",
            "nonpolar_smd_cds",
            "strict_common_functional_claim",
        ),
        source_representation=(
            "original four-channel MACE-POLAR source; no artificial second-radial "
            "source coefficients"
        ),
        field_convention=(
            "separate eight-channel native radial receiver u=L sigma; no L=B* assertion"
        ),
        continuum_profile="smooth-weighted-harmonic-galerkin-cpcm-candidate-v1",
        cavity_profile="smooth-weighted-overlap-harmonic-cavity-candidate-v1",
        nonpolar_profile="none",
        state_equation_id=SEPARATED_OPERATIONAL_STATE_EQUATION_ID,
        derivative_route=(
            "implicit adjoint of this frozen ledger only; unchanged-source route "
            "closed before ledger/force admission by the four-case matched "
            "QM/PCMSolver source-MEP gate"
        ),
        admitted_capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    ScalarDefinition(
        scalar_id=(
            OPERATIONAL_MACEPOLAR_SEPARATED_PHI1_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
        ),
        exact_formula=(
            "Phi1Delta(R,y*)=E_vac(R)+[E_conditioned_raw(R,L_R sigma*)-"
            "E_conditioned_raw(R,0)]+1/2 sigma*^T A_R sigma*; "
            "c*=c_ref+Ty*, A_R sigma*=B_R c*, "
            "c*=Pi_q M_orig(R,L_R sigma*)"
        ),
        implementation_entry_point=(
            "maple.solvation.coupling.separated_ledgers:NormalizedPhi1DeltaLedger"
        ),
        included_components=(
            "macepolar_vacuum_energy",
            "macepolar_conditioned_raw_energy_difference",
            "continuum_polarization_self_energy",
        ),
        excluded_components=(
            "explicit_checkpoint_uniform_field_work_term",
            "complete_external_enthalpy_claim_before_field_semantics_gate",
            "nonpolar_smd_cds",
            "strict_common_functional_claim",
            "original_source_energy_conjugacy_claim",
        ),
        source_representation=(
            "original four-channel MACE-POLAR source; no artificial second-radial "
            "source coefficients"
        ),
        field_convention=(
            "separate eight-channel native radial receiver u=L sigma; ledger sign "
            "is frozen to A sigma=B c and requires a separate external-enthalpy audit"
        ),
        continuum_profile="smooth-weighted-harmonic-galerkin-cpcm-candidate-v1",
        cavity_profile="smooth-weighted-overlap-harmonic-cavity-candidate-v1",
        nonpolar_profile="none",
        state_equation_id=SEPARATED_OPERATIONAL_STATE_EQUATION_ID,
        derivative_route=(
            "implicit adjoint of this explicitly selected operational ledger; "
            "vacuum-normalized and field-semantics-manifest-bound; closed because "
            "native injection omits upstream explicit work and the unchanged "
            "source fails the matched QM/PCMSolver physical gate"
        ),
        admitted_capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    ScalarDefinition(
        scalar_id=(EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_V1),
        exact_formula=(
            "E_exp(R)=E_vac^MACEPOLAR(R)+G_PCM(R,u*(R)); "
            "G_PCM=1/2 v(u*)^T q(u*)*HartreeToEV; "
            "v=B_point c_MDP+B_GTO1p5[M_POLAR(R,u*)-M_POLAR(R,0)]; "
            "q=Q_PCMSolver v; u*=L_radial q; G_np=0"
        ),
        implementation_entry_point=(
            "maple.solvation.experimental.mace_mdp_polar_pcmsolver:"
            "MACE_MDPPolarHybridPCMSolverPES"
        ),
        included_components=(
            "macepolar_zero_field_vacuum_energy",
            "hybrid_pcmsolver_half_coupling_electrostatic",
        ),
        excluded_components=(
            "macepolar_field_conditioned_raw_energy_difference",
            "mace_mdp_polarizability_response",
            "nonpolar_smd_cds",
            "thermal_and_standard_state_terms",
            "complete_solvation_free_energy_claim",
            "analytic_coordinate_derivative",
            "strict_common_functional_claim",
        ),
        source_representation=(
            "MACE-MDP permanent atom-centred q/p through the exterior point kernel "
            "plus only the MACE-POLAR field-induced q/p increment through the "
            "normalized sigma=1.5-A Gaussian kernel"
        ),
        field_convention=(
            "MACE-POLAR native eight-channel 1.5/3.0-A radial receiver field; "
            "source and receiver are deliberately not declared one dual space"
        ),
        continuum_profile="pcmsolver-symmetric-external-mep-electrostatic-v1",
        cavity_profile="pcmsolver-input-defined-gepol-cavity-v1",
        nonpolar_profile="none",
        state_equation_id=MACE_MDP_POLAR_HYBRID_STATE_EQUATION_ID,
        derivative_route=(
            "fourth-order Richardson central derivative of the exact registered "
            "scalar; every displacement rebuilds PCMSolver and resolves both root "
            "starts; topology changes and excessive stencil error fail closed; "
            "this is numerical, not an analytic adjoint"
        ),
        admitted_capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    ScalarDefinition(
        scalar_id=(
            EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1
        ),
        exact_formula=(
            "E_exp(R)=E_vac^MACEPOLAR(R)-1/2 b(u*)^T A_harm(R)^-1 b(u*); "
            "b=B_point^harm(R)c_MDP+B_GTO1p5^harm(R)"
            "[M_POLAR(R,u*)-M_POLAR(R,0)]; "
            "u*=-S8_harm(R)^T A_harm(R)^-1 b(u*); G_np=0"
        ),
        implementation_entry_point=(
            "maple.solvation.experimental.mace_mdp_polar_harmonic:"
            "MACE_MDPPolarHybridSmoothHarmonicPES"
        ),
        included_components=(
            "macepolar_zero_field_vacuum_energy",
            "hybrid_smooth_harmonic_half_coupling_electrostatic",
        ),
        excluded_components=(
            "macepolar_field_conditioned_raw_energy_difference",
            "mace_mdp_polarizability_response",
            "nonpolar_smd_cds",
            "thermal_and_standard_state_terms",
            "complete_solvation_free_energy_claim",
            "analytic_coordinate_derivative",
            "strict_common_functional_claim",
        ),
        source_representation=(
            "MACE-MDP permanent atom-centred q/p through the invariant exterior "
            "point harmonic kernel plus only the MACE-POLAR field-induced q/p "
            "increment through the normalized sigma=1.5-A Gaussian harmonic kernel"
        ),
        field_convention=(
            "MACE-POLAR native eight-channel 1.5/3.0-A radial receiver assembled "
            "as minus the exact transpose of the harmonic Gaussian source operator; "
            "the permanent point source remains categorically distinct"
        ),
        continuum_profile="smooth-weighted-harmonic-galerkin-cpcm-candidate-v1",
        cavity_profile="smooth-weighted-overlap-harmonic-cavity-candidate-v1",
        nonpolar_profile="none",
        state_equation_id=MACE_MDP_POLAR_HYBRID_HARMONIC_STATE_EQUATION_ID,
        derivative_route=(
            "fourth-order Richardson central derivative of this exact registered "
            "scalar; every displacement rebuilds the fixed-dimensional smooth "
            "harmonic assembly and resolves both root starts; excessive stencil "
            "error fails closed; this is numerical, not an analytic adjoint"
        ),
        admitted_capabilities=CapabilityStatus(
            energy=True,
            conservative_force=True,
        ),
        evidence_artifact_ids=(
            MACE_MDP_POLAR_HYBRID_HARMONIC_FORCE_ADMISSION_EVIDENCE_ID,
        ),
        enabled=True,
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
            "G_np=0; L_s(R,c,u)=E_anc(R,u)-s<c,u>_Q+sG_cpcm_fixed(R,c); "
            "the eliminated common-stationarity state is implemented but disabled"
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
            "maple.solvation.coupling.variational_state:" "VariationalCommonFunctional"
        ),
        derivative_route=(
            "stationary envelope derivative; disabled pending sign, gauge, "
            "passivity, root, continuum, coordinate, rotation, and release gates"
        ),
        admitted_capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    ScalarDefinition(
        scalar_id=(
            VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_V1
        ),
        exact_formula=(
            "candidate equations: M_var=Q^-T dE_anc/du and "
            "u=grad_Q G_harm(c); G_harm=-1/2 (S c)^T A^-1 (S c), "
            "with complete per-atom harmonic irreps and S^dagger as receiver; "
            "L_s(R,c,u)=E_anc(R,u)-s<c,u>_Q+sG_harm(R,c); the eliminated "
            "common-stationarity state is implemented but disabled"
        ),
        included_components=(
            "macepolar_anchored_field_energy_candidate",
            "fixed_external_harmonic_galerkin_stationary_electrostatic_target",
        ),
        excluded_components=(
            "original_four_channel_density_as_variational_source",
            "geometry_intertwiner_assembly",
            "moving_cavity_coordinate_derivative",
            "source_dependent_cavity",
            "nonpolar_smd_cds",
        ),
        source_representation=(
            "complete eight-channel MACE-POLAR field-energy-conjugate effective "
            "source coupled to complete per-atom harmonic coefficient blocks"
        ),
        field_convention=(
            "positive energy-dual reduced radial-GTO field; fixed charge and "
            "constant-potential gauge separated"
        ),
        continuum_profile="fixed-harmonic-galerkin-cpcm-candidate-v1",
        cavity_profile="fixed-external-harmonic-coefficient-cavity-v1",
        nonpolar_profile="none",
        state_equation_id=VARIATIONAL_STATE_EQUATION_ID,
        implementation_entry_point=(
            "maple.solvation.coupling.variational_state:" "VariationalCommonFunctional"
        ),
        derivative_route=(
            "same-scalar state and stationary-envelope diagnostic; disabled "
            "pending sign, gauge, passivity, root uniqueness, analytic coordinate "
            "assembly, rotation, and release gates"
        ),
        admitted_capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    ScalarDefinition(
        scalar_id=(
            VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
        ),
        exact_formula=(
            "candidate equations: M_var=Q^-T dE_anc/du and "
            "u=grad_Q G_harm(R,c); G_harm=-1/2 (S(R)c)^T "
            "A(R)^-1 (S(R)c), A=E(R)^T K(R) E(R), S=E(R)^T V(R); "
            "L_s(R,c,u)=E_anc(R,u)-s<c,u>_Q+sG_harm(R,c); the "
            "eliminated common-stationarity state is implemented but disabled"
        ),
        included_components=(
            "macepolar_anchored_field_energy_candidate",
            "smooth_weighted_harmonic_galerkin_stationary_electrostatic_target",
            "same_scalar_moving_geometry_coordinate_derivative",
        ),
        excluded_components=(
            "original_four_channel_density_as_variational_source",
            "sharp_union_of_spheres_identity",
            "nonpolar_smd_cds",
            "coordinate_hessian",
        ),
        source_representation=(
            "complete eight-channel MACE-POLAR field-energy-conjugate effective "
            "source coupled to complete per-atom harmonic coefficient blocks"
        ),
        field_convention=(
            "positive energy-dual reduced radial-GTO field; fixed charge and "
            "constant-potential gauge separated"
        ),
        continuum_profile="smooth-weighted-harmonic-galerkin-cpcm-candidate-v1",
        cavity_profile="smooth-weighted-overlap-harmonic-cavity-candidate-v1",
        nonpolar_profile="none",
        state_equation_id=VARIATIONAL_STATE_EQUATION_ID,
        implementation_entry_point=(
            "maple.solvation.coupling.variational_state:" "VariationalCommonFunctional"
        ),
        derivative_route=(
            "same-scalar state and stationary-envelope diagnostic; continuum "
            "coordinate partial and mixed pullback come from the same Torch graph; "
            "disabled pending real-checkpoint root, rotation, stability, physical, "
            "and release gates"
        ),
        admitted_capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    ScalarDefinition(
        scalar_id=(
            VARIATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
        ),
        exact_formula=(
            "candidate equations: M_var=Q^-T dE_anc^analytic/du and "
            "u=grad_Q G_harm(R,c); E_anc^analytic uses the separately "
            "identified analytic isotropic Gaussian l<=1 molecular "
            "real-space checkpoint evaluator; G_harm=-1/2 (S(R)c)^T "
            "A(R)^-1 (S(R)c), A=E(R)^T K(R) E(R), S=E(R)^T V(R); "
            "the eliminated common-stationarity state is disabled"
        ),
        included_components=(
            "macepolar_analytic_gaussian_multipole_anchored_field_energy_candidate",
            "smooth_weighted_harmonic_galerkin_stationary_electrostatic_target",
            "same_scalar_moving_geometry_coordinate_derivative",
        ),
        excluded_components=(
            "upstream_fixed_axis_finite_difference_realspace_operator",
            "original_four_channel_density_as_variational_source",
            "sharp_union_of_spheres_identity",
            "nonpolar_smd_cds",
            "coordinate_hessian",
        ),
        source_representation=(
            "complete eight-channel analytic-evaluator MACE-POLAR "
            "field-energy-conjugate effective source coupled to complete "
            "per-atom harmonic coefficient blocks"
        ),
        field_convention=(
            "positive energy-dual reduced radial-GTO field; fixed charge and "
            "constant-potential gauge separated"
        ),
        continuum_profile="smooth-weighted-harmonic-galerkin-cpcm-candidate-v1",
        cavity_profile="smooth-weighted-overlap-harmonic-cavity-candidate-v1",
        nonpolar_profile="none",
        state_equation_id=VARIATIONAL_STATE_EQUATION_ID,
        implementation_entry_point=(
            "maple.solvation.coupling.variational_state:" "VariationalCommonFunctional"
        ),
        derivative_route=(
            "same-scalar state and stationary-envelope diagnostic; the "
            "analytic long-range operator is structurally SO(3) but the "
            "complete scalar remains disabled pending numerical full-model "
            "rotation, passivity, root uniqueness, combined-Hessian, physical, "
            "and release gates"
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
    "EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_V1",
    "EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1",
    "MACE_MDP_POLAR_HYBRID_HARMONIC_FORCE_ADMISSION_EVIDENCE_ID",
    "OPERATIONAL_CPCM_ELECTROSTATIC_V1",
    "OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_V1",
    "OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_GALERKIN_CPCM_V1",
    "OPERATIONAL_MACEPOLAR_SEPARATED_PHI1_SMOOTH_HARMONIC_GALERKIN_CPCM_V1",
    "OPERATIONAL_CPCM_SMDCDS_V1",
    "SCALAR_REGISTRY",
    "VARIATIONAL_COMMON_FUNCTIONAL_V1",
    "VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_V1",
    "VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_V1",
    "VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1",
    "VARIATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1",
    "ScalarDefinition",
    "get_scalar_definition",
    "scalar_registry_manifest",
]
