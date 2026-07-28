from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.gto_density import (
    density_to_external_field_order,
    external_field_to_density_order,
)
from maple.function.calculator.extra_correction.implicit.route2_thermodynamic_diagnostics import (
    energy_density_conjugacy_diagnostic,
    field_loop_work_diagnostic,
    fixed_point_feedback_spectrum_diagnostic,
    intrinsic_feature_conjugacy_diagnostic,
    matrix_free_fixed_point_feedback_gain_diagnostic,
    response_reciprocity_diagnostic,
    response_stability_diagnostic,
)
from maple.function.calculator.extra_correction.implicit.route2_response import (
    NeutralDensityCoordinates,
)


class _LinearDensityResponse:
    """Linear response expressed in external Cartesian field order."""

    def __init__(self, external_response_matrix):
        matrix = np.asarray(external_response_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Response matrix must be square.")
        self.matrix = matrix
        self.atom_count = matrix.shape[0] // 4

    def jvp(self, field_direction):
        field = np.asarray(field_direction, dtype=float)
        external_density = (self.matrix @ field.reshape(-1)).reshape(
            self.atom_count,
            4,
        )
        return external_field_to_density_order(external_density)

    def vjp(self, density_cotangent):
        external_cotangent = density_to_external_field_order(
            np.asarray(density_cotangent, dtype=float)
        )
        return (self.matrix.T @ external_cotangent.reshape(-1)).reshape(
            self.atom_count,
            4,
        )


class _LinearFeatureDensityResponse:
    """Rectangular native-feature-to-density response."""

    def __init__(self, matrix, *, atom_count, feature_count):
        values = np.asarray(matrix, dtype=float)
        expected_shape = (
            4 * int(atom_count),
            int(atom_count) * int(feature_count),
        )
        if values.shape != expected_shape:
            raise ValueError(
                f"Feature response must have shape {expected_shape}."
            )
        self.matrix = values
        self.atom_count = int(atom_count)
        self.feature_count = int(feature_count)

    def vjp(self, density_cotangent):
        cotangent = np.asarray(density_cotangent, dtype=float)
        return (self.matrix.T @ cotangent.reshape(-1)).reshape(
            self.atom_count,
            self.feature_count,
        )


class _ReducedFeedbackResidual:
    """Synthetic ``R(c)=c-F(c)`` in neutral orthonormal coordinates."""

    def __init__(self, feedback_matrix, *, atom_count=2):
        self.atom_count = int(atom_count)
        self.coordinates = NeutralDensityCoordinates(self.atom_count)
        matrix = np.asarray(feedback_matrix, dtype=float)
        expected_shape = (
            self.coordinates.dimension,
            self.coordinates.dimension,
        )
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Feedback matrix must have shape {expected_shape}."
            )
        self.feedback_matrix = matrix

    def jvp(self, density_direction):
        direction = np.asarray(density_direction, dtype=float)
        reduced = self.coordinates.reduce(direction)
        feedback = self.coordinates.expand(
            self.feedback_matrix @ reduced
        )
        return direction - feedback

    def vjp(self, density_cotangent):
        cotangent = np.asarray(density_cotangent, dtype=float)
        reduced = self.coordinates.reduce(cotangent)
        feedback = self.coordinates.expand(
            self.feedback_matrix.T @ reduced
        )
        return cotangent - feedback


def _field():
    return np.asarray(
        [
            [0.20, -0.04, 0.03, 0.02],
            [-0.11, 0.01, -0.06, 0.05],
        ],
        dtype=float,
    )


def _stable_response():
    susceptibility = np.diag(
        [0.11, 0.07, 0.08, 0.09, 0.13, 0.05, 0.06, 0.04]
    )
    return _LinearDensityResponse(-susceptibility)


def _density(response, field):
    return response.jvp(field)


def test_intrinsic_energy_identity_is_distinguished_from_coupled_energy():
    response = _stable_response()
    field = _field()
    density = _density(response, field)
    intrinsic_gradient = -density_to_external_field_order(density)

    diagnostic = energy_density_conjugacy_diagnostic(
        reaction_field_values_ev=field,
        response_density_coefficients=density,
        intrinsic_energy_field_gradient=intrinsic_gradient,
        density_response=response,
    )

    assert diagnostic.lower_defect_candidate == "intrinsic"
    assert (
        diagnostic.intrinsic_candidate.maximum_relative_block_error
        == pytest.approx(0.0, abs=1.0e-15)
    )
    assert diagnostic.coupled_candidate.maximum_relative_block_error > 0.9


def test_coupled_energy_identity_is_distinguished_from_intrinsic_energy():
    response = _stable_response()
    field = _field()
    density = _density(response, field)
    coupled_gradient = density_to_external_field_order(density)

    diagnostic = energy_density_conjugacy_diagnostic(
        reaction_field_values_ev=field,
        response_density_coefficients=density,
        intrinsic_energy_field_gradient=coupled_gradient,
        density_response=response,
    )

    assert diagnostic.lower_defect_candidate == "coupled"
    assert (
        diagnostic.coupled_candidate.maximum_relative_block_error
        == pytest.approx(0.0, abs=1.0e-15)
    )
    assert diagnostic.intrinsic_candidate.maximum_relative_block_error > 0.9


def test_native_feature_intrinsic_conjugacy_identity_closes_exactly():
    atom_count = 2
    feature_count = 3
    matrix = np.arange(
        1,
        4 * atom_count * atom_count * feature_count + 1,
        dtype=float,
    ).reshape(4 * atom_count, atom_count * feature_count)
    matrix *= 1.0e-3
    response = _LinearFeatureDensityResponse(
        matrix,
        atom_count=atom_count,
        feature_count=feature_count,
    )
    field = _field()
    density_dual = external_field_to_density_order(field)
    expected_gradient = -response.vjp(density_dual)

    diagnostic = intrinsic_feature_conjugacy_diagnostic(
        reaction_field_values_ev=field,
        intrinsic_energy_feature_gradient=expected_gradient,
        density_response=response,
        l0_feature_count=1,
    )

    assert diagnostic.feature_count == feature_count
    assert diagnostic.l0_feature_count == 1
    assert diagnostic.coupled_candidate is None
    assert diagnostic.all_features.relative_l2 == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert diagnostic.l0_radial_features.relative_l2 == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert (
        diagnostic.l1_radial_cartesian_features.relative_l2
        == pytest.approx(0.0, abs=1.0e-15)
    )


def test_native_feature_conjugacy_separates_l0_and_l1_defects():
    atom_count = 2
    feature_count = 3
    matrix = np.eye(4 * atom_count, atom_count * feature_count)
    response = _LinearFeatureDensityResponse(
        matrix,
        atom_count=atom_count,
        feature_count=feature_count,
    )
    field = _field()
    expected_gradient = -response.vjp(
        external_field_to_density_order(field)
    )
    perturbed = expected_gradient.copy()
    perturbed[:, 1:] *= -1.0

    diagnostic = intrinsic_feature_conjugacy_diagnostic(
        reaction_field_values_ev=field,
        intrinsic_energy_feature_gradient=perturbed,
        density_response=response,
        l0_feature_count=1,
    )

    assert diagnostic.l0_radial_features.relative_l2 == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert (
        diagnostic.l1_radial_cartesian_features.relative_l2
        == pytest.approx(1.0)
    )


def test_native_feature_conjugacy_rejects_an_invalid_l0_split():
    response = _LinearFeatureDensityResponse(
        np.eye(8, 6),
        atom_count=2,
        feature_count=3,
    )

    with pytest.raises(ValueError, match="split"):
        intrinsic_feature_conjugacy_diagnostic(
            reaction_field_values_ev=_field(),
            intrinsic_energy_feature_gradient=np.zeros((2, 3)),
            density_response=response,
            l0_feature_count=3,
        )


def test_response_reciprocity_uses_the_electrostatic_pairing():
    response = _stable_response()
    first = np.zeros((2, 4), dtype=float)
    second = np.zeros((2, 4), dtype=float)
    first[0, 0] = 0.2
    first[1, 2] = -0.1
    second[0, 3] = 0.3
    second[1, 1] = 0.4

    diagnostic = response_reciprocity_diagnostic(
        response,
        first,
        second,
    )

    assert diagnostic.absolute_error_ev == pytest.approx(0.0, abs=1.0e-15)
    assert diagnostic.relative_error == pytest.approx(0.0, abs=1.0e-15)


def test_response_reciprocity_detects_a_nonsymmetric_response():
    matrix = np.zeros((8, 8), dtype=float)
    matrix[0, 1] = 0.4
    response = _LinearDensityResponse(matrix)
    first = np.zeros((2, 4), dtype=float)
    second = np.zeros((2, 4), dtype=float)
    first.reshape(-1)[0] = 1.0
    second.reshape(-1)[1] = 1.0

    diagnostic = response_reciprocity_diagnostic(
        response,
        first,
        second,
    )

    assert diagnostic.forward_pairing_ev == pytest.approx(0.4)
    assert diagnostic.reverse_pairing_ev == pytest.approx(0.0)
    assert diagnostic.absolute_error_ev == pytest.approx(0.4)
    assert diagnostic.relative_error == pytest.approx(1.0)


def test_response_stability_uses_the_nonpositive_susceptibility_sign():
    diagnostic = response_stability_diagnostic(
        _stable_response(),
        atom_count=2,
    )

    assert diagnostic.dimension == 8
    assert diagnostic.maximum_eigenvalue_ev < 0.0
    assert diagnostic.positive_eigenvalue_count == 0
    assert diagnostic.negative_eigenvalue_count == 8
    assert diagnostic.near_zero_eigenvalue_count == 0
    assert diagnostic.passivity_violation_ev == 0.0
    assert diagnostic.antisymmetric_frobenius_norm_ev == pytest.approx(
        0.0,
        abs=1.0e-15,
    )


def test_response_stability_detects_a_positive_susceptibility_mode():
    matrix = -np.diag(np.linspace(0.04, 0.11, 8))
    matrix[3, 3] = 0.025

    diagnostic = response_stability_diagnostic(
        _LinearDensityResponse(matrix),
        atom_count=2,
    )

    assert diagnostic.positive_eigenvalue_count == 1
    assert diagnostic.passivity_violation_ev == pytest.approx(0.025)


def test_response_stability_fails_closed_above_the_small_system_limit():
    response = _LinearDensityResponse(-np.eye(8))

    with pytest.raises(ValueError, match="small-system limit"):
        response_stability_diagnostic(
            response,
            atom_count=2,
            maximum_dimension=7,
        )


def test_fixed_point_feedback_spectrum_recovers_a_known_diagonal_map():
    dimension = NeutralDensityCoordinates(2).dimension
    eigenvalues = np.asarray(
        [0.2, -0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
        dtype=float,
    )
    assert eigenvalues.size == dimension
    residual = _ReducedFeedbackResidual(np.diag(eigenvalues))

    diagnostic = fixed_point_feedback_spectrum_diagnostic(
        residual,
        mixing_values=(0.5, 1.0),
    )

    assert diagnostic.dimension == dimension
    assert diagnostic.spectral_radius == pytest.approx(0.5)
    assert diagnostic.maximum_real_eigenvalue == pytest.approx(0.2)
    assert diagnostic.minimum_real_eigenvalue == pytest.approx(-0.5)
    assert diagnostic.maximum_absolute_imaginary_eigenvalue == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert diagnostic.largest_singular_value == pytest.approx(0.5)
    assert diagnostic.antisymmetric_frobenius_norm == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    condition = diagnostic.residual_operator_i_minus_feedback
    assert condition.numerically_singular is False
    assert condition.largest_singular_value == pytest.approx(1.5)
    assert condition.smallest_singular_value == pytest.approx(0.8)
    assert condition.condition_number_2 == pytest.approx(1.875)
    assert condition.inverse_norm_2 == pytest.approx(1.25)
    by_mixing = {
        item.mixing: item
        for item in diagnostic.mixing_iteration_matrices
    }
    assert by_mixing[0.5].spectral_radius == pytest.approx(0.6)
    assert by_mixing[1.0].spectral_radius == pytest.approx(0.5)
    assert all(
        item.locally_contracting_by_eigenvalues
        for item in by_mixing.values()
    )


def test_fixed_point_feedback_spectrum_reports_nonnormality():
    dimension = NeutralDensityCoordinates(2).dimension
    feedback = np.zeros((dimension, dimension), dtype=float)
    feedback[0, 1] = 0.4

    diagnostic = fixed_point_feedback_spectrum_diagnostic(
        _ReducedFeedbackResidual(feedback),
        mixing_values=(1.0,),
    )

    assert diagnostic.spectral_radius == pytest.approx(0.0, abs=1.0e-15)
    assert diagnostic.largest_singular_value == pytest.approx(0.4)
    assert diagnostic.antisymmetric_frobenius_norm == pytest.approx(
        np.sqrt(2.0) * 0.2
    )
    assert (
        diagnostic.antisymmetric_to_symmetric_frobenius_ratio
        == pytest.approx(1.0)
    )


def test_fixed_point_feedback_spectrum_does_not_hide_pure_antisymmetry():
    dimension = NeutralDensityCoordinates(2).dimension
    feedback = np.zeros((dimension, dimension), dtype=float)
    feedback[0, 1] = 0.2
    feedback[1, 0] = -0.2

    diagnostic = fixed_point_feedback_spectrum_diagnostic(
        _ReducedFeedbackResidual(feedback),
        mixing_values=(1.0,),
    )

    assert diagnostic.symmetric_frobenius_norm == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert diagnostic.antisymmetric_frobenius_norm > 0.0
    assert diagnostic.antisymmetric_to_symmetric_frobenius_ratio is None


def test_fixed_point_feedback_spectrum_validates_budget_and_mixing():
    dimension = NeutralDensityCoordinates(2).dimension
    residual = _ReducedFeedbackResidual(np.zeros((dimension, dimension)))

    with pytest.raises(ValueError, match="small-system limit"):
        fixed_point_feedback_spectrum_diagnostic(
            residual,
            maximum_dimension=dimension - 1,
        )
    with pytest.raises(ValueError, match="unique"):
        fixed_point_feedback_spectrum_diagnostic(
            residual,
            mixing_values=(0.5, 0.5),
        )


@pytest.mark.parametrize(
    "feedback",
    (
        np.diag([0.5, -0.3, 0.2, 0.1, 0.05, -0.02, 0.01]),
        np.asarray(
            [
                [0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01],
            ]
        ),
    ),
)
def test_matrix_free_feedback_gain_matches_the_complete_small_system_oracle(
    feedback,
):
    residual = _ReducedFeedbackResidual(feedback)
    complete = fixed_point_feedback_spectrum_diagnostic(
        residual,
        mixing_values=(1.0,),
    )

    estimated = matrix_free_fixed_point_feedback_gain_diagnostic(
        residual,
        solver_tolerance=1.0e-12,
        maximum_iterations=100,
        random_seed=20260728,
    )

    assert estimated.dimension == feedback.shape[0]
    assert estimated.largest_singular_value_estimate == pytest.approx(
        complete.largest_singular_value,
        rel=1.0e-10,
        abs=1.0e-12,
    )
    assert estimated.relative_singular_triplet_residual < 1.0e-10
    assert estimated.solver_converged is True
    assert estimated.feedback_jvp_applications > 0
    assert estimated.feedback_vjp_applications > 0


def test_matrix_free_feedback_gain_validates_solver_controls():
    dimension = NeutralDensityCoordinates(2).dimension
    residual = _ReducedFeedbackResidual(np.zeros((dimension, dimension)))

    with pytest.raises(ValueError, match="tolerance"):
        matrix_free_fixed_point_feedback_gain_diagnostic(
            residual,
            solver_tolerance=0.0,
        )
    with pytest.raises(ValueError, match="iterations"):
        matrix_free_fixed_point_feedback_gain_diagnostic(
            residual,
            maximum_iterations=0,
        )
    with pytest.raises(ValueError, match="Krylov"):
        matrix_free_fixed_point_feedback_gain_diagnostic(
            residual,
            krylov_subspace_dimension=1,
        )


def test_matrix_free_feedback_gain_handles_the_zero_operator():
    dimension = NeutralDensityCoordinates(2).dimension
    diagnostic = matrix_free_fixed_point_feedback_gain_diagnostic(
        _ReducedFeedbackResidual(np.zeros((dimension, dimension))),
        solver_tolerance=1.0e-12,
    )

    assert diagnostic.largest_singular_value_estimate == pytest.approx(0.0)
    assert diagnostic.relative_singular_triplet_residual == pytest.approx(0.0)
    assert diagnostic.estimated_euclidean_contraction is True
    assert diagnostic.solver == "zero-feedback-random-probe-v1"


def test_field_loop_work_vanishes_for_a_conservative_linear_response():
    response = _stable_response()
    base = _field()
    first = np.zeros((2, 4), dtype=float)
    second = np.zeros((2, 4), dtype=float)
    first.reshape(-1)[0] = 0.03
    second.reshape(-1)[5] = -0.04

    diagnostic = field_loop_work_diagnostic(
        response.jvp,
        base_field=base,
        first_field_step=first,
        second_field_step=second,
    )

    assert diagnostic.closed_loop_work_ev == pytest.approx(0.0, abs=1.0e-15)
    assert diagnostic.relative_closed_loop_work == pytest.approx(
        0.0,
        abs=1.0e-15,
    )


def test_field_loop_work_detects_a_nonconservative_linear_response():
    matrix = np.zeros((8, 8), dtype=float)
    matrix[0, 1] = 0.4
    response = _LinearDensityResponse(matrix)
    base = np.zeros((2, 4), dtype=float)
    first = np.zeros((2, 4), dtype=float)
    second = np.zeros((2, 4), dtype=float)
    first.reshape(-1)[0] = 0.2
    second.reshape(-1)[1] = 0.3

    diagnostic = field_loop_work_diagnostic(
        response.jvp,
        base_field=base,
        first_field_step=first,
        second_field_step=second,
    )

    assert abs(diagnostic.closed_loop_work_ev) == pytest.approx(0.024)
    assert diagnostic.relative_closed_loop_work == pytest.approx(1.0)
