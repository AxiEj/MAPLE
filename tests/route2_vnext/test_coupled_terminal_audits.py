from __future__ import annotations

import numpy as np
import pytest

from maple.solvation.release.coupled_conjugacy import (
    analyze_coupled_conjugacy,
)
from maple.solvation.release.root_well_posedness import (
    certify_root_well_posedness,
    dense_state_map_jacobian,
)
from maple.solvation.release.source_mep import (
    compare_source_mep_to_reference,
    source_electrostatic_observables,
)


def test_coupled_conjugacy_passes_only_after_actual_B_and_L_projection():
    source_to_boundary = np.eye(2)
    boundary_to_field = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.2, -0.1]])
    source = np.asarray([0.3, -0.2])
    gradient = np.asarray([-0.3, 0.2, 0.0])
    source_jacobian = np.asarray([[0.2, 0.0, 0.0], [0.0, 0.1, 0.0]])
    result = analyze_coupled_conjugacy(
        source_to_boundary=source_to_boundary,
        boundary_to_native_field=boundary_to_field,
        original_source=source,
        intrinsic_energy_native_field_gradient=gradient,
        source_native_field_jacobian=source_jacobian,
    )
    assert result.direct_identity_passed is True
    assert result.coupled_reciprocity_passed is True
    assert result.terminal_no_go_witness_detected is False
    assert "finite pass is not a global proof" in result.as_dict()["claim_boundary"]

    broken_gradient = gradient.copy()
    broken_gradient[0] += 1.0e-3
    broken = analyze_coupled_conjugacy(
        source_to_boundary=source_to_boundary,
        boundary_to_native_field=boundary_to_field,
        original_source=source,
        intrinsic_energy_native_field_gradient=broken_gradient,
        source_native_field_jacobian=source_jacobian,
        energy_semantics_verified=True,
    )
    assert broken.terminal_no_go_witness_detected is True
    assert broken.residual_relative > broken.residual_tolerance

    unverified = analyze_coupled_conjugacy(
        source_to_boundary=source_to_boundary,
        boundary_to_native_field=boundary_to_field,
        original_source=source,
        intrinsic_energy_native_field_gradient=broken_gradient,
        source_native_field_jacobian=source_jacobian,
    )
    assert unverified.material_direct_defect_detected is True
    assert unverified.direct_identity_interpretable is False
    assert unverified.terminal_no_go_witness_detected is False


def test_coupled_curl_uses_unsymmetrized_original_response():
    base = dict(
        source_to_boundary=np.eye(2),
        boundary_to_native_field=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        original_source=np.asarray([0.2, -0.1]),
        intrinsic_energy_native_field_gradient=np.asarray([-0.2, 0.1, 0.0]),
    )
    reciprocal = analyze_coupled_conjugacy(
        **base,
        source_native_field_jacobian=np.asarray([[0.2, 0.0, 0.0], [0.0, 0.1, 0.0]]),
    )
    broken = analyze_coupled_conjugacy(
        **base,
        source_native_field_jacobian=np.asarray([[0.2, 0.3, 0.0], [0.0, 0.1, 0.0]]),
    )
    assert reciprocal.coupled_reciprocity_passed is True
    assert broken.coupled_reciprocity_passed is False
    assert broken.coupled_curl_absolute_frobenius > 0.0


def test_root_certificate_does_not_promote_single_state_spectrum_to_global():
    jacobian = np.diag([0.2, 0.4, -0.1])
    local = certify_root_well_posedness(jacobian)
    assert local.local_implicit_branch_certified is True
    assert local.contraction_at_state is True
    assert local.global_unique_root_certified is False
    assert local.residual_error_bound(6.0e-7) is None

    domain = certify_root_well_posedness(
        jacobian,
        domain_invariance_certified=True,
        domain_uniform_bound_certified=True,
    )
    assert domain.global_unique_root_certified is True
    assert domain.residual_error_bound(6.0e-7) == pytest.approx(1.0e-6)

    noncontractive = certify_root_well_posedness(np.asarray([[1.2, 2.0], [0.0, 1.2]]))
    assert noncontractive.contraction_at_state is False
    assert noncontractive.strong_monotonicity_at_state is False
    assert noncontractive.global_unique_root_certified is False
    assert noncontractive.residual_error_bound(1.0e-6) is None


def test_dense_state_map_jacobian_is_bounded_and_column_correct():
    matrix = np.asarray([[0.2, -0.1], [0.3, 0.4]])
    dense = dense_state_map_jacobian(lambda direction: matrix @ direction, dimension=2)
    np.testing.assert_array_equal(dense, matrix)
    with pytest.raises(ValueError, match="bounded audit"):
        dense_state_map_jacobian(
            lambda direction: direction, dimension=3, maximum_dimension=2
        )


def test_source_mep_diagnostic_is_reference_bound_and_component_matched():
    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    source = np.asarray([[0.3, 0.02, -0.01, 0.04], [-0.3, -0.01, 0.02, -0.03]])
    points_bohr = np.asarray([[5.0, 0.0, 0.0], [0.0, 5.0, 1.0]])
    observables = source_electrostatic_observables(
        positions_angstrom=positions,
        source4=source,
        evaluation_points_bohr=points_bohr,
    )
    assert observables.total_charge_e == pytest.approx(0.0, abs=1.0e-15)
    result = compare_source_mep_to_reference(
        observables,
        observables,
        predicted_fixed_source_pcm_energy_eV=-0.123,
        reference_fixed_source_pcm_energy_eV=-0.123,
        reference_identity="test-same-cavity-qm-reference-v1",
    )
    assert result.total_charge_absolute_error_e == 0.0
    assert result.sampled_mep_relative_l2_error == 0.0
    assert result.fixed_source_pcm_energy_absolute_error_eV == 0.0
    assert result.as_dict()["capability_admitted"] is False
