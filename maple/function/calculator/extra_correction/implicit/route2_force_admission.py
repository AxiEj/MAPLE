"""Fail-closed admission evidence for a Route-2 force profile.

This module intentionally does *not* expose an ASE force.  It turns the
separate numerical prerequisites for a conservative Route-2 fixed-point force
into one machine-readable record: a local residual condition screen, primal
and adjoint residuals, continuum smoothness evidence, root-repeat evidence,
and an explicit distinction between an operational scalar force and an
unproven common electronic free-energy functional.

In particular, an iterative estimate of ``||J_M J_P||`` is not a proof of a
lower singular-value bound.  Small systems use an explicit neutral-space SVD;
larger systems are recorded as an insufficient matrix-free screen and remain
fail-closed until a validated bound is supplied.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

from ....route2_energy_ledger import (
    LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
    PCM_HALF_COUPLING_ONLY_V1,
    Route2ElectrostaticEnergyLedger,
    validate_route2_electrostatic_energy_ledger,
)
from .route2_response import FixedChargeCoordinates
from .route2_thermodynamic_diagnostics import (
    FixedPointFeedbackSpectrumDiagnostic,
    fixed_point_feedback_spectrum_diagnostic,
)

FORCE_ADMISSION_CONTRACT_VERSION = 3
RHODROP_CAVITY_FORCE_ADMISSION_CONTRACT_VERSION = 2


@dataclass(frozen=True)
class RhoDropCavityForceGateEvidence:
    """Cavity-specific prerequisites layered below the generic force gate.

    This record is intentionally separate from the existing Route-2 force
    certificate v3, whose archived release evidence must remain stable.  A
    source-dependent cavity cannot enter that generic gate until every local
    DROP/source condition, the field and reference-anchor coordinate pieces,
    and the end-to-end Gate B/C checks below are independently evidenced.
    """

    profile_kind: str
    all_projection_points_converged: bool
    surface_residual_gate_passed: bool
    minimum_gradient_gate_passed: bool
    reconstructed_density_nonnegative_gate_passed: bool
    electron_count_gate_passed: bool
    cpcm_linear_residual_gate_passed: bool
    half_coupling_identity_gate_passed: bool
    reaction_map_jvp_vjp_gate_passed: bool
    cold_replay_gate_passed: bool
    branch_stability_gate_passed: bool
    efficient_jvp_gate_passed: bool
    field_coordinate_vjp_gate_passed: bool
    anchor_coordinate_vjp_gate_passed: bool
    gate_b_component_finite_difference_passed: bool
    gate_c_end_to_end_force_passed: bool
    evidence: str
    provider_contract_version: int | None = None
    provider_configuration_sha256: str | None = None
    runtime_sha256: str | None = None
    drop_parameter_sha256: str | None = None
    cpcm_parameter_sha256: str | None = None
    geometry_sha256: str | None = None
    source_sha256: str | None = None
    level_set_state_sha256: str | None = None
    surface_snapshot_sha256: str | None = None
    operator_sha256: str | None = None
    forward_state_sha256: str | None = None
    evidence_artifact_path: str | None = None
    evidence_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.profile_kind.strip():
            raise ValueError("A rho-DROP force-gate profile kind is required.")
        if not self.evidence.strip():
            raise ValueError("rho-DROP force-gate evidence is required.")
        for name in self._gate_names():
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"rho-DROP force gate {name} must be bool.")
        if self.provider_contract_version is not None and (
            not isinstance(self.provider_contract_version, int)
            or isinstance(self.provider_contract_version, bool)
            or self.provider_contract_version <= 0
        ):
            raise ValueError("rho-DROP provider contract version must be positive.")
        for name in self._binding_hash_names():
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"rho-DROP binding {name} must be a lowercase SHA256.")
        path = self.evidence_artifact_path
        artifact_hash = self.evidence_artifact_sha256
        if (path is None) != (artifact_hash is None):
            raise ValueError(
                "rho-DROP evidence artifact path and SHA256 must be supplied together."
            )
        if path is not None:
            if not isinstance(path, str) or not path.strip():
                raise ValueError("rho-DROP evidence artifact path must be nonempty.")
            artifact = Path(path).expanduser().resolve()
            if not artifact.is_file():
                raise ValueError(f"rho-DROP evidence artifact does not exist: {artifact}")
            observed_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if observed_hash != artifact_hash:
                raise ValueError("rho-DROP evidence artifact SHA256 does not match.")

    @staticmethod
    def _gate_names() -> tuple[str, ...]:
        return (
            "all_projection_points_converged",
            "surface_residual_gate_passed",
            "minimum_gradient_gate_passed",
            "reconstructed_density_nonnegative_gate_passed",
            "electron_count_gate_passed",
            "cpcm_linear_residual_gate_passed",
            "half_coupling_identity_gate_passed",
            "reaction_map_jvp_vjp_gate_passed",
            "cold_replay_gate_passed",
            "branch_stability_gate_passed",
            "efficient_jvp_gate_passed",
            "field_coordinate_vjp_gate_passed",
            "anchor_coordinate_vjp_gate_passed",
            "gate_b_component_finite_difference_passed",
            "gate_c_end_to_end_force_passed",
        )

    @staticmethod
    def _binding_hash_names() -> tuple[str, ...]:
        return (
            "provider_configuration_sha256",
            "runtime_sha256",
            "drop_parameter_sha256",
            "cpcm_parameter_sha256",
            "geometry_sha256",
            "source_sha256",
            "level_set_state_sha256",
            "surface_snapshot_sha256",
            "operator_sha256",
            "forward_state_sha256",
            "evidence_artifact_sha256",
        )

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        labels = {
            "all_projection_points_converged": "drop-projection-convergence-unverified",
            "surface_residual_gate_passed": "drop-surface-residual-gate-unverified",
            "minimum_gradient_gate_passed": "drop-minimum-gradient-gate-unverified",
            "reconstructed_density_nonnegative_gate_passed": (
                "reconstructed-density-nonnegativity-unverified"
            ),
            "electron_count_gate_passed": "reconstructed-electron-count-unverified",
            "cpcm_linear_residual_gate_passed": "cpcm-linear-residual-unverified",
            "half_coupling_identity_gate_passed": "pcm-half-coupling-identity-unverified",
            "reaction_map_jvp_vjp_gate_passed": "reaction-map-jvp-vjp-unverified",
            "cold_replay_gate_passed": "source-dependent-cavity-cold-replay-unverified",
            "branch_stability_gate_passed": "drop-branch-stability-unverified",
            "efficient_jvp_gate_passed": "efficient-drop-forward-jvp-unavailable",
            "field_coordinate_vjp_gate_passed": "drop-field-coordinate-vjp-unverified",
            "anchor_coordinate_vjp_gate_passed": "drop-anchor-coordinate-vjp-unverified",
            "gate_b_component_finite_difference_passed": (
                "rho-drop-gate-b-component-finite-difference-unverified"
            ),
            "gate_c_end_to_end_force_passed": (
                "rho-drop-gate-c-end-to-end-force-unverified"
            ),
        }
        gate_failures = tuple(
            labels[name] for name in self._gate_names() if not getattr(self, name)
        )
        binding_failures = []
        if self.provider_contract_version is None:
            binding_failures.append("rho-drop-provider-contract-unbound")
        for name in self._binding_hash_names():
            if getattr(self, name) is None:
                binding_failures.append(f"rho-drop-{name.replace('_', '-')}-unbound")
        if self.evidence_artifact_path is None:
            binding_failures.append("rho-drop-evidence-artifact-path-unbound")
        return gate_failures + tuple(binding_failures)

    @property
    def force_admitted(self) -> bool:
        return not self.failure_reasons

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": RHODROP_CAVITY_FORCE_ADMISSION_CONTRACT_VERSION,
            "profile_kind": self.profile_kind,
            "provider_contract_version": self.provider_contract_version,
            **{name: getattr(self, name) for name in self._gate_names()},
            **{name: getattr(self, name) for name in self._binding_hash_names()},
            "force_admitted": self.force_admitted,
            "failure_reasons": list(self.failure_reasons),
            "evidence": self.evidence,
            "evidence_artifact_path": self.evidence_artifact_path,
        }


UNSPECIFIED_RHODROP_CAVITY_FORCE_GATE_EVIDENCE = (
    RhoDropCavityForceGateEvidence(
        profile_kind="route2-rhodrop-cpcm-operational-v1",
        all_projection_points_converged=False,
        surface_residual_gate_passed=False,
        minimum_gradient_gate_passed=False,
        reconstructed_density_nonnegative_gate_passed=False,
        electron_count_gate_passed=False,
        cpcm_linear_residual_gate_passed=False,
        half_coupling_identity_gate_passed=False,
        reaction_map_jvp_vjp_gate_passed=False,
        cold_replay_gate_passed=False,
        branch_stability_gate_passed=False,
        efficient_jvp_gate_passed=False,
        field_coordinate_vjp_gate_passed=False,
        anchor_coordinate_vjp_gate_passed=False,
        gate_b_component_finite_difference_passed=False,
        gate_c_end_to_end_force_passed=False,
        evidence=(
            "No production rho-DROP coordinate-force admission artifact has "
            "been supplied. Mathematical source/field tests do not substitute "
            "for the missing anchor term or Gate B/C."
        ),
    )
)


def require_rhodrop_cavity_force_admission(
    evidence: RhoDropCavityForceGateEvidence,
    provider_audit: Mapping[str, object],
) -> None:
    """Fail closed unless every rho-DROP-specific force prerequisite passed."""

    if not isinstance(evidence, RhoDropCavityForceGateEvidence):
        raise TypeError(
            "Source-dependent rho-DROP forces require "
            "RhoDropCavityForceGateEvidence."
        )
    if evidence.failure_reasons:
        raise RuntimeError(
            "rho-DROP source-dependent cavity forces are not admitted: "
            + ", ".join(evidence.failure_reasons)
        )
    if not isinstance(provider_audit, Mapping):
        raise TypeError("rho-DROP force admission requires a provider audit mapping.")
    expected = {
        "profile_kind": evidence.profile_kind,
        "contract_version": evidence.provider_contract_version,
        **{
            name: getattr(evidence, name)
            for name in evidence._binding_hash_names()
            if name != "evidence_artifact_sha256"
        },
    }
    mismatches = tuple(
        name for name, value in expected.items() if provider_audit.get(name) != value
    )
    if mismatches:
        raise RuntimeError(
            "rho-DROP force evidence does not bind the current provider state: "
            + ", ".join(mismatches)
        )


@dataclass(frozen=True)
class ForceEnergySemanticsContract:
    """State exactly which scalar a force differentiates.

    A conservative force only requires one declared scalar energy, a smooth
    unique root, and the derivative of *that same scalar*.  It does not imply
    that the MACE-POLAR density fixed point is stationary for a common
    MACE--PCM electronic free-energy functional.  Keeping these claims apart
    prevents the direct PCM half-coupling ledger from being rejected for the
    wrong mathematical reason, while also preventing an operational force
    certificate from being misreported as a variational electronic theory.
    """

    electrostatic_energy_ledger: Route2ElectrostaticEnergyLedger
    operational_scalar_is_explicit: bool
    derivative_matches_declared_operational_scalar: bool
    excludes_unproven_mace_field_energy_cross_term: bool
    common_variational_electronic_free_energy_proven: bool
    evidence: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "electrostatic_energy_ledger",
            validate_route2_electrostatic_energy_ledger(
                self.electrostatic_energy_ledger
            ),
        )
        if not self.evidence.strip():
            raise ValueError("Force energy-semantics evidence is required.")

    def as_dict(self) -> dict[str, object]:
        return {
            "electrostatic_energy_ledger": self.electrostatic_energy_ledger,
            "operational_scalar_is_explicit": self.operational_scalar_is_explicit,
            "derivative_matches_declared_operational_scalar": (
                self.derivative_matches_declared_operational_scalar
            ),
            "excludes_unproven_mace_field_energy_cross_term": (
                self.excludes_unproven_mace_field_energy_cross_term
            ),
            "common_variational_electronic_free_energy_proven": (
                self.common_variational_electronic_free_energy_proven
            ),
            "evidence": self.evidence,
        }


DIRECT_PCM_HALF_COUPLING_FORCE_ENERGY_SEMANTICS = ForceEnergySemanticsContract(
    electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    operational_scalar_is_explicit=True,
    derivative_matches_declared_operational_scalar=True,
    excludes_unproven_mace_field_energy_cross_term=True,
    common_variational_electronic_free_energy_proven=False,
    evidence=(
        "The declared scalar is 0.5*<c_MACE-POLAR, f_reac_PCM> + G_CDS. "
        "The ledger-specific outer adjoint differentiates that scalar and "
        "does not insert E_MACE[V_reac]-E_MACE[gas].  The current MACE "
        "fixed point is nevertheless not proven stationary for a common "
        "electronic free-energy functional."
    ),
)


LEGACY_MACE_FIELD_PLUS_PCM_FORCE_ENERGY_SEMANTICS = ForceEnergySemanticsContract(
    electrostatic_energy_ledger=LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
    operational_scalar_is_explicit=True,
    derivative_matches_declared_operational_scalar=True,
    excludes_unproven_mace_field_energy_cross_term=False,
    common_variational_electronic_free_energy_proven=False,
    evidence=(
        "The legacy operational scalar includes the field-conditioned MACE "
        "energy difference, whose relation to the returned MACE density is "
        "not established.  It is retained as a diagnostic control, not a "
        "release candidate for a direct PCM force profile."
    ),
)


def force_energy_semantics_contract(
    ledger: str,
) -> ForceEnergySemanticsContract:
    """Return the immutable force-semantics record for one registered ledger."""

    selected = validate_route2_electrostatic_energy_ledger(ledger)
    if selected == PCM_HALF_COUPLING_ONLY_V1:
        return DIRECT_PCM_HALF_COUPLING_FORCE_ENERGY_SEMANTICS
    if selected == LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1:
        return LEGACY_MACE_FIELD_PLUS_PCM_FORCE_ENERGY_SEMANTICS
    raise AssertionError(f"Unhandled Route-2 force ledger: {selected!r}.")


@dataclass(frozen=True)
class ForcePESValidationContract:
    """Profile-level evidence beyond one local derivative calculation.

    A same-scalar coordinate VJP is necessary but it does not demonstrate a
    usable conservative PES.  This record holds the independent end-to-end
    gates that a public force profile must satisfy.  They are deliberately
    separate from the continuum algebra, so a unit-tested surface operator
    cannot accidentally promote itself to an optimizer/MD interface.
    """

    profile_kind: str
    component_resolved_finite_difference_verified: bool
    rigid_translation_verified: bool
    rigid_rotation_covariance_verified: bool
    coordinate_path_smoothness_verified: bool
    closed_loop_work_verified: bool
    short_nve_verified: bool
    evidence: str
    evidence_artifact_path: str | None = None

    def __post_init__(self) -> None:
        if not self.profile_kind.strip():
            raise ValueError("A force/PES validation profile kind is required.")
        if not self.evidence.strip():
            raise ValueError("Force/PES validation evidence is required.")
        if self.evidence_artifact_path is not None and not (
            isinstance(self.evidence_artifact_path, str)
            and self.evidence_artifact_path.strip()
        ):
            raise ValueError(
                "A supplied force/PES evidence-artifact path must be nonempty."
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "profile_kind": self.profile_kind,
            "component_resolved_finite_difference_verified": (
                self.component_resolved_finite_difference_verified
            ),
            "rigid_translation_verified": self.rigid_translation_verified,
            "rigid_rotation_covariance_verified": (
                self.rigid_rotation_covariance_verified
            ),
            "coordinate_path_smoothness_verified": (
                self.coordinate_path_smoothness_verified
            ),
            "closed_loop_work_verified": self.closed_loop_work_verified,
            "short_nve_verified": self.short_nve_verified,
            "evidence": self.evidence,
            "evidence_artifact_path": self.evidence_artifact_path,
        }


UNSPECIFIED_FORCE_PES_VALIDATION_CONTRACT = ForcePESValidationContract(
    profile_kind="unspecified-route2-force-pes-v1",
    component_resolved_finite_difference_verified=False,
    rigid_translation_verified=False,
    rigid_rotation_covariance_verified=False,
    coordinate_path_smoothness_verified=False,
    closed_loop_work_verified=False,
    short_nve_verified=False,
    evidence=(
        "No profile-level same-scalar finite-difference, symmetry, path, "
        "closed-work, and NVE evidence has been supplied."
    ),
)


FC_ASWIG_JGP94_D2_DIRECT_PCM_FORCE_PES_VALIDATION_CONTRACT = (
    ForcePESValidationContract(
        profile_kind="fc-aswig-jgp94-d2-direct-cpcm-aqueous-force-v3",
        component_resolved_finite_difference_verified=True,
        rigid_translation_verified=True,
        rigid_rotation_covariance_verified=True,
        coordinate_path_smoothness_verified=True,
        closed_loop_work_verified=True,
        short_nve_verified=True,
        evidence=(
            "Clean-tree evidence at e724cf5a923e9751af153ef1e5f7f1a6e1b4bffd "
            "passed the pre-registered acetone component finite-difference, "
            "translation, rotation, Cartesian-path, closed-loop, and three-step "
            "short-NVE gates. The independent 20-atom 2-acetoxyethyl-acetate "
            "torsion and two-coordinate closed loop also passed with fixed "
            "cardinality. This admits only the bounded neutral, nondegenerate, "
            "local-jet, water FC-aSWIG profile; it does not prove a universal "
            "MACE--PCM free-energy functional or all-geometry force domain."
        ),
        evidence_artifact_path=(
            "docs/implicit-solvation/benchmarks/"
            "route2-fc-aswig-force-v3-release-evidence-v1.json"
        ),
    )
)


@dataclass(frozen=True)
class NonpolarSmoothnessContract:
    """Same-scalar smoothness evidence for the selected nonpolar term.

    A continuum surface may be fixed-topology while a separately evaluated
    CDS/SASA term still removes nodes or counts exposed points discontinuously.
    A Route-2 total force must therefore certify the two terms independently.
    """

    profile_kind: str
    same_energy_coordinate_derivative: bool
    fixed_node_topology: bool
    geometry_path_smoothness_verified: bool
    evidence: str

    def __post_init__(self) -> None:
        if not self.profile_kind.strip():
            raise ValueError("A nonpolar smoothness profile kind is required.")
        if not self.evidence.strip():
            raise ValueError("Nonpolar smoothness evidence is required.")

    def as_dict(self) -> dict[str, object]:
        return {
            "profile_kind": self.profile_kind,
            "same_energy_coordinate_derivative": (
                self.same_energy_coordinate_derivative
            ),
            "fixed_node_topology": self.fixed_node_topology,
            "geometry_path_smoothness_verified": (
                self.geometry_path_smoothness_verified
            ),
            "evidence": self.evidence,
        }


UNSPECIFIED_NONPOLAR_SMOOTHNESS_CONTRACT = NonpolarSmoothnessContract(
    profile_kind="unspecified-route2-nonpolar-v1",
    same_energy_coordinate_derivative=False,
    fixed_node_topology=False,
    geometry_path_smoothness_verified=False,
    evidence=(
        "No profile-specific same-energy, fixed-topology, or geometry-path "
        "smoothness evidence was supplied for the nonpolar term."
    ),
)

PYSCF_SMD_CDS_VARIABLE_SURFACE_SMOOTHNESS_CONTRACT = NonpolarSmoothnessContract(
    profile_kind="pyscf-smd-cds-variable-surface-v1",
    same_energy_coordinate_derivative=True,
    fixed_node_topology=False,
    geometry_path_smoothness_verified=False,
    evidence=(
        "The upstream PySCF SMD-CDS energy/gradient pair is same-energy at a "
        "single geometry, but it is not the fixed-topology surface paired "
        "with a Route-2 continuum force profile."
    ),
)

FIXED_TOPOLOGY_AQUEOUS_SMD_CDS_SMOOTHNESS_CONTRACT = NonpolarSmoothnessContract(
    profile_kind="aqueous-smd-cds-fixed-topology-c3-area-v1-experimental",
    same_energy_coordinate_derivative=True,
    fixed_node_topology=True,
    geometry_path_smoothness_verified=True,
    evidence=(
        "All atom/angular candidates remain allocated; the published aqueous "
        "SMD tension functions and base-area*g**2 term share one analytic "
        "coordinate derivative.  Unit crossing and finite-difference tests "
        "cover the discrete construction."
    ),
)


@dataclass(frozen=True)
class ContinuumSmoothnessContract:
    """Evidence supplied by one continuum implementation/profile.

    ``fixed_node_topology`` means the discrete degree of freedom count and its
    ownership map remain fixed under the admitted coordinate neighbourhood.
    A smooth switching weight alone is insufficient when upstream code drops
    grid points or changes their parent atom.
    """

    profile_kind: str
    same_energy_coordinate_derivative: bool
    fixed_node_topology: bool
    geometry_path_smoothness_verified: bool
    evidence: str
    nonpolar: NonpolarSmoothnessContract = (
        UNSPECIFIED_NONPOLAR_SMOOTHNESS_CONTRACT
    )

    def __post_init__(self) -> None:
        if not self.profile_kind.strip():
            raise ValueError("A continuum smoothness profile kind is required.")
        if not self.evidence.strip():
            raise ValueError("Continuum smoothness evidence is required.")

    def as_dict(self) -> dict[str, object]:
        return {
            "profile_kind": self.profile_kind,
            "same_energy_coordinate_derivative": (
                self.same_energy_coordinate_derivative
            ),
            "fixed_node_topology": self.fixed_node_topology,
            "geometry_path_smoothness_verified": (
                self.geometry_path_smoothness_verified
            ),
            "evidence": self.evidence,
            "nonpolar": self.nonpolar.as_dict(),
        }


PYDDX_HARD_ACTIVE_SET_SMOOTHNESS_CONTRACT = ContinuumSmoothnessContract(
    profile_kind="pyddx-ddpcm-active-set-v1",
    same_energy_coordinate_derivative=True,
    fixed_node_topology=False,
    geometry_path_smoothness_verified=False,
    evidence=(
        "The same-energy coordinate VJP exists at one fixed active set, but "
        "Lebedev/sphere ownership changes under admitted geometry scans."
    ),
    nonpolar=PYSCF_SMD_CDS_VARIABLE_SURFACE_SMOOTHNESS_CONTRACT,
)

PYSCF_SWIG_VARIABLE_SURFACE_SMOOTHNESS_CONTRACT = ContinuumSmoothnessContract(
    profile_kind="pyscf-swig-variable-surface-v1",
    same_energy_coordinate_derivative=True,
    fixed_node_topology=False,
    geometry_path_smoothness_verified=False,
    evidence=(
        "PySCF SWIG supplies a same-energy operator gradient, but current "
        "surface construction changes retained grid-point counts/parents "
        "across geometry/orientation canaries."
    ),
    nonpolar=PYSCF_SMD_CDS_VARIABLE_SURFACE_SMOOTHNESS_CONTRACT,
)

FIXED_TOPOLOGY_ASWIG_CPCM_AQUEOUS_SMD_SMOOTHNESS_CONTRACT = (
    ContinuumSmoothnessContract(
        profile_kind="cpcm-fc-aswig-aqueous-smd-cds-v1-experimental",
        same_energy_coordinate_derivative=True,
        fixed_node_topology=True,
        geometry_path_smoothness_verified=True,
        evidence=(
            "The amplitude-CPCM scalar, reaction map, and full continuum "
            "coordinate VJP share all fixed candidates.  Unit crossing and "
            "finite-difference tests cover exposure, Gaussian-kernel, "
            "source, and receiver terms."
        ),
        nonpolar=FIXED_TOPOLOGY_AQUEOUS_SMD_CDS_SMOOTHNESS_CONTRACT,
    )
)

JGP94_FIXED_TOPOLOGY_ASWIG_CPCM_AQUEOUS_SMD_SMOOTHNESS_CONTRACT = (
    ContinuumSmoothnessContract(
        profile_kind="jgp94-cpcm-fc-aswig-aqueous-smd-cds-v1-experimental",
        same_energy_coordinate_derivative=True,
        fixed_node_topology=True,
        geometry_path_smoothness_verified=True,
        evidence=(
            "The JGP94 nondegenerate principal-axis wrapper applies the same "
            "fixed-topology amplitude-CPCM scalar and complete body-to-lab "
            "coordinate VJP.  Unit tests cover rigid rotation, translation, "
            "and a rebuilt-coordinate finite difference; the paired aqueous "
            "CDS uses the same JGP94 frame."
        ),
        nonpolar=FIXED_TOPOLOGY_AQUEOUS_SMD_CDS_SMOOTHNESS_CONTRACT,
    )
)

UNSPECIFIED_CONTINUUM_SMOOTHNESS_CONTRACT = ContinuumSmoothnessContract(
    profile_kind="unspecified-route2-continuum-v1",
    same_energy_coordinate_derivative=False,
    fixed_node_topology=False,
    geometry_path_smoothness_verified=False,
    evidence=(
        "No profile-specific same-energy, fixed-topology, or geometry-path "
        "smoothness evidence was supplied."
    ),
)


@dataclass(frozen=True)
class ForceAdmissionPolicy:
    """Predeclared numerical requirements for a small-system force gate."""

    maximum_primal_monopole_residual_e: float = 1.0e-9
    maximum_primal_dipole_residual_e_angstrom: float = 1.0e-9
    maximum_adjoint_relative_residual: float = 1.0e-8
    maximum_continuum_identity_error_ev: float = 1.0e-10
    minimum_residual_singular_value: float = 1.0e-5
    maximum_residual_condition_number_2: float = 1.0e5
    maximum_feedback_singular_value: float = 0.95
    maximum_dense_dimension: int = 128

    def __post_init__(self) -> None:
        positive = (
            self.maximum_primal_monopole_residual_e,
            self.maximum_primal_dipole_residual_e_angstrom,
            self.maximum_adjoint_relative_residual,
            self.maximum_continuum_identity_error_ev,
            self.minimum_residual_singular_value,
            self.maximum_residual_condition_number_2,
            self.maximum_feedback_singular_value,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("Route-2 force-admission thresholds must be positive.")
        if self.maximum_feedback_singular_value >= 1.0:
            raise ValueError(
                "The feedback singular-value gate must be strictly below one."
            )
        if self.maximum_dense_dimension <= 0:
            raise ValueError("The dense condition-screen dimension must be positive.")

    def as_dict(self) -> dict[str, float | int]:
        return {
            "maximum_primal_monopole_residual_e": (
                self.maximum_primal_monopole_residual_e
            ),
            "maximum_primal_dipole_residual_e_angstrom": (
                self.maximum_primal_dipole_residual_e_angstrom
            ),
            "maximum_adjoint_relative_residual": (
                self.maximum_adjoint_relative_residual
            ),
            "maximum_continuum_identity_error_ev": (
                self.maximum_continuum_identity_error_ev
            ),
            "minimum_residual_singular_value": (self.minimum_residual_singular_value),
            "maximum_residual_condition_number_2": (
                self.maximum_residual_condition_number_2
            ),
            "maximum_feedback_singular_value": (self.maximum_feedback_singular_value),
            "maximum_dense_dimension": self.maximum_dense_dimension,
        }


@dataclass(frozen=True)
class FixedPointConditionScreen:
    """Local conditioning evidence for ``A = I - J_M J_P``.

    ``exact_small_system`` is deliberately narrow: it means all reduced
    columns were evaluated and a dense floating-point SVD was performed.  A
    matrix-free Ritz value is useful for triage, but cannot certify a force
    admission lower bound by itself.
    """

    dimension: int
    method: str
    exact_small_system: bool
    feedback_largest_singular_value: float | None
    residual_smallest_singular_value: float | None
    residual_condition_number_2: float | None
    residual_is_numerically_singular: bool | None
    gate_passed: bool
    reason: str | None

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("Fixed-point condition-screen dimension must be positive.")
        for value in (
            self.feedback_largest_singular_value,
            self.residual_smallest_singular_value,
            self.residual_condition_number_2,
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError("Fixed-point condition metrics must be finite.")
        if self.gate_passed and not self.exact_small_system:
            raise ValueError(
                "A matrix-free condition screen cannot certify force admission."
            )
        if self.gate_passed and self.reason is not None:
            raise ValueError(
                "A passing condition screen must not carry a failure reason."
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "method": self.method,
            "exact_small_system": self.exact_small_system,
            "feedback_largest_singular_value": self.feedback_largest_singular_value,
            "residual_smallest_singular_value": self.residual_smallest_singular_value,
            "residual_condition_number_2": self.residual_condition_number_2,
            "residual_is_numerically_singular": (self.residual_is_numerically_singular),
            "gate_passed": self.gate_passed,
            "reason": self.reason,
        }


def _condition_screen_from_dense_diagnostic(
    diagnostic: FixedPointFeedbackSpectrumDiagnostic,
    policy: ForceAdmissionPolicy,
) -> FixedPointConditionScreen:
    condition = diagnostic.residual_operator_i_minus_feedback
    feedback = diagnostic.largest_singular_value
    smallest = condition.smallest_singular_value
    condition_number = condition.condition_number_2
    failures = []
    if condition.numerically_singular:
        failures.append("residual-operator-numerically-singular")
    if feedback > policy.maximum_feedback_singular_value:
        failures.append("feedback-gain-above-predeclared-gate")
    if smallest < policy.minimum_residual_singular_value:
        failures.append("residual-smallest-singular-value-below-gate")
    if (
        condition_number is None
        or condition_number > policy.maximum_residual_condition_number_2
    ):
        failures.append("residual-condition-number-above-gate")
    return FixedPointConditionScreen(
        dimension=diagnostic.dimension,
        method="dense-neutral-space-svd-v1",
        exact_small_system=True,
        feedback_largest_singular_value=feedback,
        residual_smallest_singular_value=smallest,
        residual_condition_number_2=condition_number,
        residual_is_numerically_singular=condition.numerically_singular,
        gate_passed=not failures,
        reason=None if not failures else ",".join(failures),
    )


def fixed_point_condition_screen(
    residual_linearization,
    *,
    policy: ForceAdmissionPolicy = ForceAdmissionPolicy(),
) -> FixedPointConditionScreen:
    """Return exact bounded-dimensional conditioning or an explicit non-pass."""

    atom_count = int(getattr(residual_linearization, "atom_count", 0))
    if atom_count <= 0:
        raise ValueError("Fixed-point condition screen requires a positive atom count.")
    dimension = FixedChargeCoordinates(atom_count).dimension
    if dimension <= policy.maximum_dense_dimension:
        diagnostic = fixed_point_feedback_spectrum_diagnostic(
            residual_linearization,
            maximum_dimension=policy.maximum_dense_dimension,
        )
        return _condition_screen_from_dense_diagnostic(diagnostic, policy)

    # The standalone matrix-free gain diagnostic is intentionally not used by
    # a public/admission calculation path.  Its Ritz value is a useful
    # diagnostic estimate, but it is not a certified upper gain bound or a
    # lower bound on sigma_min(I - J_M J_P).  Do not turn a non-pass screen
    # into a potentially misleading expensive calculation.
    return FixedPointConditionScreen(
        dimension=dimension,
        method="dense-fixed-charge-condition-screen-required-v1",
        exact_small_system=False,
        feedback_largest_singular_value=None,
        residual_smallest_singular_value=None,
        residual_condition_number_2=None,
        residual_is_numerically_singular=None,
        gate_passed=False,
        reason=(
            "full-fixed-charge-condition-screen-exceeds-declared-dense-"
            "dimension-bound"
        ),
    )


@dataclass(frozen=True)
class ForceAdmissionCertificate:
    """One fail-closed Route-2 force-release decision record."""

    contract_version: int
    policy: ForceAdmissionPolicy
    nominal_root: bool
    primal_monopole_residual_e: float
    primal_dipole_residual_e_angstrom: float
    adjoint_relative_residual: float
    continuum_identity_error_ev: float
    condition: FixedPointConditionScreen
    continuum: ContinuumSmoothnessContract
    multi_start_root_agreement: bool | None
    energy_semantics: ForceEnergySemanticsContract
    pes_validation: ForcePESValidationContract
    release_admitted: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.contract_version != FORCE_ADMISSION_CONTRACT_VERSION:
            raise ValueError("Unsupported Route-2 force-admission contract version.")
        for value in (
            self.primal_monopole_residual_e,
            self.primal_dipole_residual_e_angstrom,
            self.adjoint_relative_residual,
            self.continuum_identity_error_ev,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "Force-admission residuals must be finite and nonnegative."
                )
        if self.release_admitted != (not self.failure_reasons):
            raise ValueError("Force-admission decision must match its failure reasons.")

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "policy": self.policy.as_dict(),
            "nominal_root": self.nominal_root,
            "primal_monopole_residual_e": self.primal_monopole_residual_e,
            "primal_dipole_residual_e_angstrom": (
                self.primal_dipole_residual_e_angstrom
            ),
            "adjoint_relative_residual": self.adjoint_relative_residual,
            "continuum_identity_error_ev": self.continuum_identity_error_ev,
            "condition": self.condition.as_dict(),
            "continuum": self.continuum.as_dict(),
            "multi_start_root_agreement": self.multi_start_root_agreement,
            "energy_semantics": self.energy_semantics.as_dict(),
            "pes_validation": self.pes_validation.as_dict(),
            "release_admitted": self.release_admitted,
            "failure_reasons": list(self.failure_reasons),
        }


def evaluate_force_admission(
    residual_linearization,
    *,
    nominal_root: bool,
    primal_monopole_residual_e: float,
    primal_dipole_residual_e_angstrom: float,
    adjoint_relative_residual: float,
    continuum_identity_error_ev: float,
    continuum: ContinuumSmoothnessContract,
    multi_start_root_agreement: bool | None,
    energy_semantics: ForceEnergySemanticsContract,
    pes_validation: ForcePESValidationContract = (
        UNSPECIFIED_FORCE_PES_VALIDATION_CONTRACT
    ),
    policy: ForceAdmissionPolicy = ForceAdmissionPolicy(),
) -> ForceAdmissionCertificate:
    """Assess every prerequisite without changing the public force API."""

    values = {
        "primal_monopole_residual_e": primal_monopole_residual_e,
        "primal_dipole_residual_e_angstrom": (primal_dipole_residual_e_angstrom),
        "adjoint_relative_residual": adjoint_relative_residual,
        "continuum_identity_error_ev": continuum_identity_error_ev,
    }
    normalized = {}
    for name, raw_value in values.items():
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative.")
        normalized[name] = value

    condition = fixed_point_condition_screen(
        residual_linearization,
        policy=policy,
    )
    failures = []
    if not nominal_root:
        failures.append("fixed-point-root-is-not-nominal")
    if (
        normalized["primal_monopole_residual_e"]
        > policy.maximum_primal_monopole_residual_e
    ):
        failures.append("primal-monopole-residual-above-gate")
    if (
        normalized["primal_dipole_residual_e_angstrom"]
        > policy.maximum_primal_dipole_residual_e_angstrom
    ):
        failures.append("primal-dipole-residual-above-gate")
    if (
        normalized["adjoint_relative_residual"]
        > policy.maximum_adjoint_relative_residual
    ):
        failures.append("adjoint-residual-above-gate")
    if (
        normalized["continuum_identity_error_ev"]
        > policy.maximum_continuum_identity_error_ev
    ):
        failures.append("continuum-energy-identity-above-gate")
    if not condition.gate_passed:
        failures.append("fixed-point-condition-screen-not-admitted")
    if not continuum.same_energy_coordinate_derivative:
        failures.append("continuum-coordinate-derivative-does-not-match-energy")
    if not continuum.fixed_node_topology:
        failures.append("continuum-node-topology-is-not-fixed")
    if not continuum.geometry_path_smoothness_verified:
        failures.append("continuum-geometry-path-smoothness-unverified")
    nonpolar = continuum.nonpolar
    if not nonpolar.same_energy_coordinate_derivative:
        failures.append("nonpolar-coordinate-derivative-does-not-match-energy")
    if not nonpolar.fixed_node_topology:
        failures.append("nonpolar-node-topology-is-not-fixed")
    if not nonpolar.geometry_path_smoothness_verified:
        failures.append("nonpolar-geometry-path-smoothness-unverified")
    if multi_start_root_agreement is not True:
        failures.append("multi-start-root-uniqueness-unverified")
    if not energy_semantics.operational_scalar_is_explicit:
        failures.append("force-ledger-operational-scalar-unresolved")
    if not energy_semantics.derivative_matches_declared_operational_scalar:
        failures.append("force-ledger-derivative-not-matched-to-scalar")
    if not energy_semantics.excludes_unproven_mace_field_energy_cross_term:
        failures.append("unproven-mace-field-energy-cross-term-included")
    if not pes_validation.component_resolved_finite_difference_verified:
        failures.append("component-resolved-force-finite-difference-unverified")
    if not pes_validation.rigid_translation_verified:
        failures.append("rigid-translation-force-gate-unverified")
    if not pes_validation.rigid_rotation_covariance_verified:
        failures.append("rigid-rotation-force-gate-unverified")
    if not pes_validation.coordinate_path_smoothness_verified:
        failures.append("coordinate-path-smoothness-force-gate-unverified")
    if not pes_validation.closed_loop_work_verified:
        failures.append("closed-loop-work-force-gate-unverified")
    if not pes_validation.short_nve_verified:
        failures.append("short-nve-force-gate-unverified")

    return ForceAdmissionCertificate(
        contract_version=FORCE_ADMISSION_CONTRACT_VERSION,
        policy=policy,
        nominal_root=bool(nominal_root),
        primal_monopole_residual_e=(normalized["primal_monopole_residual_e"]),
        primal_dipole_residual_e_angstrom=(
            normalized["primal_dipole_residual_e_angstrom"]
        ),
        adjoint_relative_residual=normalized["adjoint_relative_residual"],
        continuum_identity_error_ev=normalized["continuum_identity_error_ev"],
        condition=condition,
        continuum=continuum,
        multi_start_root_agreement=multi_start_root_agreement,
        energy_semantics=energy_semantics,
        pes_validation=pes_validation,
        release_admitted=not failures,
        failure_reasons=tuple(failures),
    )


__all__ = [
    "FORCE_ADMISSION_CONTRACT_VERSION",
    "RHODROP_CAVITY_FORCE_ADMISSION_CONTRACT_VERSION",
    "ContinuumSmoothnessContract",
    "DIRECT_PCM_HALF_COUPLING_FORCE_ENERGY_SEMANTICS",
    "FC_ASWIG_JGP94_D2_DIRECT_PCM_FORCE_PES_VALIDATION_CONTRACT",
    "FIXED_TOPOLOGY_AQUEOUS_SMD_CDS_SMOOTHNESS_CONTRACT",
    "FIXED_TOPOLOGY_ASWIG_CPCM_AQUEOUS_SMD_SMOOTHNESS_CONTRACT",
    "JGP94_FIXED_TOPOLOGY_ASWIG_CPCM_AQUEOUS_SMD_SMOOTHNESS_CONTRACT",
    "FixedPointConditionScreen",
    "ForceEnergySemanticsContract",
    "ForceAdmissionCertificate",
    "ForceAdmissionPolicy",
    "ForcePESValidationContract",
    "LEGACY_MACE_FIELD_PLUS_PCM_FORCE_ENERGY_SEMANTICS",
    "NonpolarSmoothnessContract",
    "PYDDX_HARD_ACTIVE_SET_SMOOTHNESS_CONTRACT",
    "PYSCF_SMD_CDS_VARIABLE_SURFACE_SMOOTHNESS_CONTRACT",
    "PYSCF_SWIG_VARIABLE_SURFACE_SMOOTHNESS_CONTRACT",
    "RhoDropCavityForceGateEvidence",
    "UNSPECIFIED_NONPOLAR_SMOOTHNESS_CONTRACT",
    "UNSPECIFIED_CONTINUUM_SMOOTHNESS_CONTRACT",
    "UNSPECIFIED_FORCE_PES_VALIDATION_CONTRACT",
    "UNSPECIFIED_RHODROP_CAVITY_FORCE_GATE_EVIDENCE",
    "evaluate_force_admission",
    "force_energy_semantics_contract",
    "fixed_point_condition_screen",
    "require_rhodrop_cavity_force_admission",
]
