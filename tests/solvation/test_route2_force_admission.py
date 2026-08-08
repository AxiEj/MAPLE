from __future__ import annotations

import hashlib

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_force_admission import (
    ContinuumSmoothnessContract,
    DIRECT_PCM_HALF_COUPLING_FORCE_ENERGY_SEMANTICS,
    FIXED_TOPOLOGY_ASWIG_CPCM_AQUEOUS_SMD_SMOOTHNESS_CONTRACT,
    ForceAdmissionPolicy,
    ForcePESValidationContract,
    LEGACY_MACE_FIELD_PLUS_PCM_FORCE_ENERGY_SEMANTICS,
    NonpolarSmoothnessContract,
    PYDDX_HARD_ACTIVE_SET_SMOOTHNESS_CONTRACT,
    RhoDropCavityForceGateEvidence,
    UNSPECIFIED_RHODROP_CAVITY_FORCE_GATE_EVIDENCE,
    evaluate_force_admission,
    fixed_point_condition_screen,
    require_rhodrop_cavity_force_admission,
)
from maple.function.calculator.extra_correction.implicit.route2_response import (
    FixedChargeCoordinates,
)


class _ReducedFeedbackResidual:
    """Synthetic residual with a known reduced fixed-point feedback matrix."""

    def __init__(self, feedback: np.ndarray, *, atom_count: int = 2):
        self.atom_count = int(atom_count)
        self.coordinates = FixedChargeCoordinates(self.atom_count)
        matrix = np.asarray(feedback, dtype=float)
        expected_shape = (
            self.coordinates.dimension,
            self.coordinates.dimension,
        )
        if matrix.shape != expected_shape:
            raise ValueError(f"feedback must have shape {expected_shape}.")
        self.feedback = matrix

    def jvp(self, direction: np.ndarray) -> np.ndarray:
        reduced = self.coordinates.reduce(direction)
        return self.coordinates.expand(reduced - self.feedback @ reduced)

    def vjp(self, cotangent: np.ndarray) -> np.ndarray:
        reduced = self.coordinates.reduce(cotangent)
        return self.coordinates.expand(reduced - self.feedback.T @ reduced)


_SMOOTH_SYNTHETIC_CONTINUUM = ContinuumSmoothnessContract(
    profile_kind="synthetic-fixed-node-galerkin-v1",
    same_energy_coordinate_derivative=True,
    fixed_node_topology=True,
    geometry_path_smoothness_verified=True,
    evidence="Synthetic linear control with one fixed set of degrees of freedom.",
    nonpolar=NonpolarSmoothnessContract(
        profile_kind="synthetic-fixed-node-nonpolar-v1",
        same_energy_coordinate_derivative=True,
        fixed_node_topology=True,
        geometry_path_smoothness_verified=True,
        evidence="Synthetic scalar nonpolar control.",
    ),
)

_PES_VALIDATED_SYNTHETIC = ForcePESValidationContract(
    profile_kind="synthetic-fully-validated-force-pes-v1",
    component_resolved_finite_difference_verified=True,
    rigid_translation_verified=True,
    rigid_rotation_covariance_verified=True,
    coordinate_path_smoothness_verified=True,
    closed_loop_work_verified=True,
    short_nve_verified=True,
    evidence="Synthetic control where every same-scalar PES gate is supplied.",
)


def _zero_feedback_residual(*, atom_count: int = 2) -> _ReducedFeedbackResidual:
    dimension = FixedChargeCoordinates(atom_count).dimension
    return _ReducedFeedbackResidual(
        np.zeros((dimension, dimension)),
        atom_count=atom_count,
    )


def test_rhodrop_cavity_force_gate_preserves_missing_anchor_and_gate_bc_blockers():
    evidence = UNSPECIFIED_RHODROP_CAVITY_FORCE_GATE_EVIDENCE
    assert not evidence.force_admitted
    failures = set(evidence.failure_reasons)
    assert "drop-anchor-coordinate-vjp-unverified" in failures
    assert "efficient-drop-forward-jvp-unavailable" in failures
    assert "rho-drop-gate-b-component-finite-difference-unverified" in failures
    assert "rho-drop-gate-c-end-to-end-force-unverified" in failures
    assert evidence.as_dict()["force_admitted"] is False


def test_rhodrop_cavity_force_gate_requires_every_independent_boolean(
    tmp_path,
) -> None:
    gate_names = RhoDropCavityForceGateEvidence._gate_names()
    values = {name: True for name in gate_names}
    artifact = tmp_path / "rho-drop-force-evidence.json"
    artifact.write_text('{"status":"synthetic-complete"}\n', encoding="utf-8")
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    binding_hashes = {
        name: f"{index + 1:064x}"
        for index, name in enumerate(
            RhoDropCavityForceGateEvidence._binding_hash_names()
        )
        if name != "evidence_artifact_sha256"
    }
    bindings = {
        "provider_contract_version": 2,
        **binding_hashes,
        "evidence_artifact_path": str(artifact),
        "evidence_artifact_sha256": artifact_hash,
    }
    admitted = RhoDropCavityForceGateEvidence(
        profile_kind="synthetic-rhodrop-complete-v1",
        evidence="Synthetic control with every independent gate supplied.",
        **values,
        **bindings,
    )
    assert admitted.force_admitted
    assert admitted.failure_reasons == ()

    provider_audit = {
        "profile_kind": admitted.profile_kind,
        "contract_version": admitted.provider_contract_version,
        **binding_hashes,
    }
    require_rhodrop_cavity_force_admission(admitted, provider_audit)
    stale_audit = dict(provider_audit)
    stale_audit["forward_state_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="forward_state_sha256"):
        require_rhodrop_cavity_force_admission(admitted, stale_audit)
    wrong_profile_audit = dict(provider_audit)
    wrong_profile_audit["profile_kind"] = "another-source-dependent-provider"
    with pytest.raises(RuntimeError, match="profile_kind"):
        require_rhodrop_cavity_force_admission(admitted, wrong_profile_audit)

    values["anchor_coordinate_vjp_gate_passed"] = False
    blocked = RhoDropCavityForceGateEvidence(
        profile_kind="synthetic-rhodrop-missing-anchor-v1",
        evidence="Synthetic control missing only the anchor coordinate term.",
        **values,
        **bindings,
    )
    assert not blocked.force_admitted
    assert blocked.failure_reasons == ("drop-anchor-coordinate-vjp-unverified",)


def test_dense_condition_screen_records_exact_small_system_svd_gate():
    screen = fixed_point_condition_screen(_zero_feedback_residual())

    assert screen.exact_small_system is True
    assert screen.method == "dense-neutral-space-svd-v1"
    assert screen.feedback_largest_singular_value == pytest.approx(0.0)
    assert screen.residual_smallest_singular_value == pytest.approx(1.0)
    assert screen.residual_condition_number_2 == pytest.approx(1.0)
    assert screen.gate_passed is True
    assert screen.reason is None


def test_matrix_free_gain_estimate_never_becomes_an_admission_certificate():
    screen = fixed_point_condition_screen(
        _zero_feedback_residual(),
        policy=ForceAdmissionPolicy(maximum_dense_dimension=1),
    )

    assert screen.exact_small_system is False
    assert screen.feedback_largest_singular_value is None
    assert screen.gate_passed is False
    assert screen.reason == (
        "full-fixed-charge-condition-screen-exceeds-declared-dense-" "dimension-bound"
    )


def test_force_admission_requires_all_numeric_structural_and_semantic_gates():
    certificate = evaluate_force_admission(
        _zero_feedback_residual(),
        nominal_root=True,
        primal_monopole_residual_e=1.0e-12,
        primal_dipole_residual_e_angstrom=1.0e-12,
        adjoint_relative_residual=1.0e-12,
        continuum_identity_error_ev=1.0e-12,
        continuum=_SMOOTH_SYNTHETIC_CONTINUUM,
        multi_start_root_agreement=True,
        energy_semantics=DIRECT_PCM_HALF_COUPLING_FORCE_ENERGY_SEMANTICS,
        pes_validation=_PES_VALIDATED_SYNTHETIC,
    )

    assert certificate.release_admitted is True
    assert certificate.failure_reasons == ()
    assert certificate.condition.exact_small_system is True


def test_known_pyddx_active_set_profile_stays_fail_closed_despite_good_residuals():
    certificate = evaluate_force_admission(
        _zero_feedback_residual(),
        nominal_root=True,
        primal_monopole_residual_e=1.0e-12,
        primal_dipole_residual_e_angstrom=1.0e-12,
        adjoint_relative_residual=1.0e-12,
        continuum_identity_error_ev=1.0e-12,
        continuum=PYDDX_HARD_ACTIVE_SET_SMOOTHNESS_CONTRACT,
        multi_start_root_agreement=True,
        energy_semantics=LEGACY_MACE_FIELD_PLUS_PCM_FORCE_ENERGY_SEMANTICS,
    )

    assert certificate.release_admitted is False
    assert set(certificate.failure_reasons) == {
        "continuum-node-topology-is-not-fixed",
        "continuum-geometry-path-smoothness-unverified",
        "nonpolar-node-topology-is-not-fixed",
        "nonpolar-geometry-path-smoothness-unverified",
        "unproven-mace-field-energy-cross-term-included",
        "component-resolved-force-finite-difference-unverified",
        "rigid-translation-force-gate-unverified",
        "rigid-rotation-force-gate-unverified",
        "coordinate-path-smoothness-force-gate-unverified",
        "closed-loop-work-force-gate-unverified",
        "short-nve-force-gate-unverified",
    }


def test_known_pyscf_swig_variable_surface_profile_stays_fail_closed():
    from maple.function.calculator.extra_correction.implicit.route2_force_admission import (
        PYSCF_SWIG_VARIABLE_SURFACE_SMOOTHNESS_CONTRACT,
    )

    certificate = evaluate_force_admission(
        _zero_feedback_residual(),
        nominal_root=True,
        primal_monopole_residual_e=1.0e-12,
        primal_dipole_residual_e_angstrom=1.0e-12,
        adjoint_relative_residual=1.0e-12,
        continuum_identity_error_ev=1.0e-12,
        continuum=PYSCF_SWIG_VARIABLE_SURFACE_SMOOTHNESS_CONTRACT,
        multi_start_root_agreement=True,
        energy_semantics=DIRECT_PCM_HALF_COUPLING_FORCE_ENERGY_SEMANTICS,
        pes_validation=_PES_VALIDATED_SYNTHETIC,
    )

    assert certificate.release_admitted is False
    assert set(certificate.failure_reasons) == {
        "continuum-node-topology-is-not-fixed",
        "continuum-geometry-path-smoothness-unverified",
        "nonpolar-node-topology-is-not-fixed",
        "nonpolar-geometry-path-smoothness-unverified",
    }


def test_fixed_topology_cpcm_and_smooth_aqueous_cds_supply_both_geometry_contracts():
    contract = FIXED_TOPOLOGY_ASWIG_CPCM_AQUEOUS_SMD_SMOOTHNESS_CONTRACT
    assert contract.same_energy_coordinate_derivative is True
    assert contract.fixed_node_topology is True
    assert contract.geometry_path_smoothness_verified is True
    assert contract.nonpolar.same_energy_coordinate_derivative is True
    assert contract.nonpolar.fixed_node_topology is True
    assert contract.nonpolar.geometry_path_smoothness_verified is True


def test_fixed_topology_profile_stays_closed_until_full_pes_evidence_exists():
    certificate = evaluate_force_admission(
        _zero_feedback_residual(),
        nominal_root=True,
        primal_monopole_residual_e=1.0e-12,
        primal_dipole_residual_e_angstrom=1.0e-12,
        adjoint_relative_residual=1.0e-12,
        continuum_identity_error_ev=1.0e-12,
        continuum=FIXED_TOPOLOGY_ASWIG_CPCM_AQUEOUS_SMD_SMOOTHNESS_CONTRACT,
        multi_start_root_agreement=True,
        energy_semantics=DIRECT_PCM_HALF_COUPLING_FORCE_ENERGY_SEMANTICS,
    )

    assert certificate.release_admitted is False
    assert set(certificate.failure_reasons) == {
        "component-resolved-force-finite-difference-unverified",
        "rigid-translation-force-gate-unverified",
        "rigid-rotation-force-gate-unverified",
        "coordinate-path-smoothness-force-gate-unverified",
        "closed-loop-work-force-gate-unverified",
        "short-nve-force-gate-unverified",
    }
