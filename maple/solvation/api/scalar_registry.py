"""Machine-readable registry of versioned Route-2 scalar definitions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .capabilities import CapabilityStatus
from .state_registry import (
    GEOMETRY_MEDIATED_SOURCE_MAP_ID,
    OPERATIONAL_STATE_EQUATION_ID,
    VARIATIONAL_STATE_EQUATION_ID,
)

DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_DDX_DDPCM_ELECTROSTATIC_V1 = (
    "route2-diagnostic-aimnet2-geometry-mediated-ddx-ddpcm-electrostatic-v1"
)
DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_ELECTROSTATIC_V1 = (
    "route2-diagnostic-aimnet2-geometry-mediated-"
    "smoothharmonicgalerkin-cpcm-electrostatic-v1"
)
DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1 = (
    "route2-diagnostic-aimnet2-geometry-mediated-"
    "smoothharmonicgalerkin-ddpcm-electrostatic-v1"
)
CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1 = (
    "route2-candidate-aimnet2-frozen-charge-water-"
    "smoothharmonicgalerkin-ddpcm-electrostatic-v1"
)
CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_SMDCDS_V1 = (
    "route2-candidate-aimnet2-frozen-charge-water-"
    "smoothharmonicgalerkin-ddpcm-pyscf-smdcds-v1"
)

OPERATIONAL_CPCM_ELECTROSTATIC_V1 = (
    "route2-operational-cpcm-fixedtopology-electrostatic-v1"
)
OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_V1 = (
    "route2-operational-macepolar-analytic-gaussian-multipole-"
    "smoothharmonicgalerkin-cpcm-v1"
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
        scalar_id=(DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_DDX_DDPCM_ELECTROSTATIC_V1),
        exact_formula=(
            "E_gm(R)=E_AIMNet2(R)+1/2<c_AIMNet2(R),"
            "P_R(c_AIMNet2(R))>_Q; "
            "c_AIMNet2(R)=[q_NQE(R),0,0,0]; G_np=0"
        ),
        implementation_entry_point=(
            "maple.solvation.coupling.geometry_mediated:"
            "GeometryMediatedElectrostaticScalar"
        ),
        included_components=(
            "aimnet2_vacuum_energy",
            "geometry_dependent_aimnet2_nqe_point_charge_source",
            "ddx_ddpcm_half_coupling_electrostatic",
            "complete_first_derivative_charge_chain_rule",
        ),
        excluded_components=(
            "fixed_geometry_electronic_mutual_polarization",
            "field_conditioned_model_energy_difference",
            "nonpolar_smd_cds",
            "standard_state_correction",
            "hessian_frequency_ts_md",
        ),
        source_representation=(
            "AIMNet2 neural-charge-equilibration atom-centred monopoles embedded "
            "as [q,0,0,0] in the atomic l<=1 source space"
        ),
        field_convention=(
            "positive energy-dual atom-centred potential/Cartesian-gradient field "
            "with pairing c^T Q u"
        ),
        continuum_profile="ddx-ddpcm-atomic-l1-v1",
        cavity_profile="ddx-union-of-spheres-exposed-lebedev-v0p8p0",
        nonpolar_profile="none",
        state_equation_id=GEOMETRY_MEDIATED_SOURCE_MAP_ID,
        derivative_route=(
            "direct same-scalar chain rule: AIMNet2 intrinsic gradient plus fixed-"
            "source ddPCM coordinate partial plus (D_R c)^T grad_c G_pcm; "
            "diagnostic only"
        ),
        admitted_capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    ScalarDefinition(
        scalar_id=(
            DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_ELECTROSTATIC_V1
        ),
        exact_formula=(
            "E_gm_harm(R)=E_AIMNet2(R)-1/2(S_R c_AIMNet2(R))^T "
            "A_R^-1(S_R c_AIMNet2(R)); "
            "c_AIMNet2(R)=[q_NQE(R),0,0,0]; "
            "A_R=E_R^T K_R E_R; S_R=E_R^T V_point,R; G_np=0"
        ),
        implementation_entry_point=(
            "maple.solvation.coupling.geometry_mediated:"
            "GeometryMediatedElectrostaticScalar"
        ),
        included_components=(
            "aimnet2_vacuum_energy",
            "geometry_dependent_aimnet2_nqe_point_charge_source",
            "smooth_weighted_harmonic_conductor_electrostatic_scalar",
            "complete_first_derivative_charge_chain_rule",
            "structural_so3_coefficient_intertwiners",
        ),
        excluded_components=(
            "fixed_geometry_electronic_mutual_polarization",
            "field_conditioned_model_energy_difference",
            "finite_dielectric_solvent_parameterization",
            "nonpolar_smd_cds",
            "standard_state_correction",
            "hessian_frequency_ts_md",
        ),
        source_representation=(
            "AIMNet2 neural-charge-equilibration atom-centred point monopoles "
            "mapped analytically to complete real-harmonic boundary data"
        ),
        field_convention=(
            "positive energy-dual atom-centred potential/Cartesian-gradient field "
            "with pairing c^T Q u; l=1 source components are identically zero"
        ),
        continuum_profile="smooth-weighted-harmonic-galerkin-cpcm-candidate-v1",
        cavity_profile="smooth-weighted-overlap-harmonic-cavity-candidate-v1",
        nonpolar_profile="none",
        state_equation_id=GEOMETRY_MEDIATED_SOURCE_MAP_ID,
        derivative_route=(
            "direct same-scalar chain rule through an analytic point-source map, "
            "fixed-dimensional harmonic Galerkin solve, and AIMNet2 charge VJP; "
            "disabled conductor-reference diagnostic only"
        ),
        admitted_capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    ScalarDefinition(
        scalar_id=(
            DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1
        ),
        exact_formula=(
            "E_gm_ddpcm(R)=E_AIMNet2(R)-1/2 b_R^T x_R; "
            "b_R=S_R c_AIMNet2(R); M_R f_R=b_R; "
            "[2*pi*(eps+1)/(eps-1) M_R-D_R] phi_eps,R="
            "[2*pi M_R-D_R] f_R; A_R x_R=M_R phi_eps,R; "
            "A_R=E_R^T K_R E_R; M_R=E_R^T E_R; "
            "D_R=E_R^T D0_R E_R; S_R=E_R^T V_point,R; G_np=0"
        ),
        implementation_entry_point=(
            "maple.solvation.coupling.geometry_mediated:"
            "GeometryMediatedElectrostaticScalar"
        ),
        included_components=(
            "aimnet2_vacuum_energy",
            "geometry_dependent_aimnet2_nqe_point_charge_source",
            "smooth_weighted_harmonic_single_layer_electrostatics",
            "finite_dielectric_ddpcm_double_layer_response",
            "complete_first_derivative_charge_chain_rule",
            "structural_so3_coefficient_intertwiners",
        ),
        excluded_components=(
            "fixed_geometry_electronic_mutual_polarization",
            "field_conditioned_model_energy_difference",
            "uniform_cosmo_dielectric_energy_scaling",
            "nonpolar_smd_cds",
            "standard_state_correction",
            "hessian_frequency_ts_md",
        ),
        source_representation=(
            "AIMNet2 neural-charge-equilibration atom-centred point monopoles "
            "mapped analytically to complete real-harmonic boundary data"
        ),
        field_convention=(
            "full energy-dual derivative of the finite-dielectric ddPCM scalar "
            "in the registered atomic l<=1 pairing; l=1 source components are zero"
        ),
        continuum_profile="smooth-weighted-harmonic-galerkin-ddpcm-candidate-v1",
        cavity_profile="smooth-weighted-overlap-harmonic-cavity-candidate-v1",
        nonpolar_profile="none",
        state_equation_id=GEOMETRY_MEDIATED_SOURCE_MAP_ID,
        derivative_route=(
            "sealed same-scalar autograd through weighted Galerkin projection, "
            "finite-dielectric double-layer solve, single-layer solve, and "
            "AIMNet2 charge VJP; disabled diagnostic only"
        ),
        admitted_capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    ScalarDefinition(
        scalar_id=(
            CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1
        ),
        exact_formula=(
            "E_AIMNet2-frozen-water(R)=E_AIMNet2(R)-1/2 b_R^T x_R; "
            "b_R=S_R c_AIMNet2(R); M_R f_R=b_R; "
            "[2*pi*(eps_water+1)/(eps_water-1) M_R-D_R] phi_R="
            "[2*pi M_R-D_R] f_R; A_R x_R=M_R phi_R; "
            "eps_water=78.355; c_AIMNet2(R)=[q_NQE(R),0,0,0]; G_np=0"
        ),
        implementation_entry_point=(
            "maple.solvation.coupling.geometry_mediated:"
            "GeometryMediatedElectrostaticScalar"
        ),
        included_components=(
            "aimnet2_wb97m_d3_vacuum_energy",
            "one_shot_geometry_dependent_aimnet2_nqe_point_charge_source",
            "water_bound_smooth_harmonic_finite_dielectric_ddpcm_electrostatics",
            "complete_first_derivative_charge_chain_rule",
            "structural_so3_coefficient_intertwiners",
        ),
        excluded_components=(
            "continuum_field_input_to_aimnet2",
            "fixed_geometry_electronic_mutual_polarization",
            "electronic_scf_iteration",
            "nonpolar_smd_cds",
            "standard_state_correction",
            "public_hessian_frequency_ts_irc_md",
        ),
        source_representation=(
            "AIMNet2 neural-charge-equilibration atom-centred point monopoles, "
            "evaluated once per geometry and embedded as [q,0,0,0]"
        ),
        field_convention=(
            "full energy-dual derivative of the water-bound finite-dielectric "
            "ddPCM scalar; never supplied to AIMNet2"
        ),
        continuum_profile=(
            "smooth-weighted-harmonic-galerkin-water-ddpcm-frozen-charge-candidate-v1"
        ),
        cavity_profile=(
            "smooth-weighted-overlap-harmonic-water-smd-coulomb-cavity-candidate-v1"
        ),
        nonpolar_profile="none",
        state_equation_id=GEOMETRY_MEDIATED_SOURCE_MAP_ID,
        derivative_route=(
            "direct same-scalar chain rule through the water-bound harmonic ddPCM "
            "primal/transpose solves and AIMNet2 charge-position VJP; no electronic SCF"
        ),
        admitted_capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
    ScalarDefinition(
        scalar_id=(
            CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_SMDCDS_V1
        ),
        exact_formula=(
            "E_AIMNet2-frozen-water-CDS(R)=E_AIMNet2(R)+G_ddPCM,harm(R,"
            "q_NQE(R))+G_CDS,PySCF-SMD-water(R); AIMNet2 is evaluated once "
            "per geometry, receives no continuum field, and has no electronic SCF"
        ),
        implementation_entry_point=(
            "maple.solvation.coupling.geometry_mediated_smd:"
            "GeometryMediatedSMDTotalScalar"
        ),
        included_components=(
            "aimnet2_wb97m_d3_vacuum_energy",
            "one_shot_geometry_dependent_aimnet2_nqe_point_charge_source",
            "water_bound_smooth_harmonic_finite_dielectric_ddpcm_electrostatics",
            "pyscf_2p13p1_water_smd_cds_energy",
            "complete_first_derivative_charge_chain_rule",
            "pyscf_water_smd_cds_analytic_coordinate_gradient",
            "structural_so3_coefficient_intertwiners",
        ),
        excluded_components=(
            "continuum_field_input_to_aimnet2",
            "fixed_geometry_electronic_mutual_polarization",
            "electronic_scf_iteration",
            "standard_state_correction",
            "strict_original_smd_electrostatic_equivalence",
            "public_hessian_frequency_ts_irc_md",
        ),
        source_representation=(
            "AIMNet2 neural-charge-equilibration atom-centred point monopoles, "
            "evaluated once per geometry and embedded as [q,0,0,0]"
        ),
        field_convention=(
            "full energy-dual derivative of the water-bound finite-dielectric "
            "ddPCM scalar; never supplied to AIMNet2"
        ),
        continuum_profile=(
            "smooth-weighted-harmonic-galerkin-water-ddpcm-frozen-charge-candidate-v1"
        ),
        cavity_profile=(
            "smooth-weighted-overlap-harmonic-water-smd-coulomb-cavity-candidate-v1"
        ),
        nonpolar_profile="pyscf-2.13.1-water-smd-cds-analytic-gradient-v1",
        state_equation_id=GEOMETRY_MEDIATED_SOURCE_MAP_ID,
        derivative_route=(
            "sum of the sealed AIMNet2/harmonic-ddPCM same-scalar chain rule "
            "and the analytic gradient returned with the exact PySCF SMD-CDS "
            "energy; disabled pending total-scalar domain, accuracy, HVP, and "
            "public workflow gates"
        ),
        admitted_capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
    ),
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
            "implicit adjoint total derivative of this operational scalar; "
            "disabled pending real-checkpoint root, force, rotation, physical, "
            "and release gates"
        ),
        admitted_capabilities=CapabilityStatus(),
        evidence_artifact_ids=(),
        enabled=False,
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
    "CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1",
    "CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_SMDCDS_V1",
    "DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_DDX_DDPCM_ELECTROSTATIC_V1",
    "DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_ELECTROSTATIC_V1",
    "DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1",
    "DIAGNOSTIC_DDX_DDCOSMO_RADIAL_GTO_ELECTROSTATIC_V1",
    "DIAGNOSTIC_DDX_DDPCM_RADIAL_GTO_ELECTROSTATIC_V1",
    "DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1",
    "OPERATIONAL_CPCM_ELECTROSTATIC_V1",
    "OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_V1",
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
