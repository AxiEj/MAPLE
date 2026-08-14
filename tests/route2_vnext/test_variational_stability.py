from __future__ import annotations

import numpy as np
import pytest

from maple.solvation.release.variational_stability import (
    VariationalStabilityThresholds,
    dense_matrix_from_action,
    variational_stability_diagnostic,
)


def test_dense_matrix_from_action_materializes_columns_and_fails_closed():
    matrix = np.asarray([[-2.0, 0.3], [0.1, -1.5]])
    actual = dense_matrix_from_action(
        lambda direction: matrix @ direction,
        2,
        name="synthetic action",
    )
    np.testing.assert_array_equal(actual, matrix)
    assert actual.flags.writeable is False

    with pytest.raises(ValueError, match="column 0"):
        dense_matrix_from_action(
            lambda _direction: np.asarray([np.nan, 0.0]),
            2,
            name="bad action",
        )
    with pytest.raises(ValueError, match="positive integer"):
        dense_matrix_from_action(lambda direction: direction, 0, name="bad")


def test_strictly_passive_contractive_common_scalar_passes_local_diagnostic():
    model = -np.diag([2.0, 3.0])
    continuum = -np.diag([0.1, 0.2])
    result = variational_stability_diagnostic(model, continuum)

    assert result["admitted"] is False
    assert result["decisions"] == {
        "model_reciprocity_gate_passed": True,
        "model_passivity_gate_passed": True,
        "model_local_invertibility_gate_passed": True,
        "continuum_reciprocity_gate_passed": True,
        "continuum_nonpositive_curvature_gate_passed": True,
        "feedback_real_spectrum_gate_passed": True,
        "feedback_nonnegative_spectrum_gate_passed": True,
        "feedback_contraction_gate_passed": True,
        "residual_local_nonsingularity_gate_passed": True,
        "combined_hessian_positive_gate_passed": True,
        "all_local_stability_gates_passed": True,
    }
    assert result["feedback"]["spectral_radius"] == pytest.approx(0.6)
    assert result["residual_jacobian"][
        "minimum_to_maximum_singular_ratio"
    ] == pytest.approx(0.5)
    combined = np.asarray(result["combined_hessian"]["matrix"])
    np.testing.assert_allclose(combined, np.diag([0.4, 2.0 / 15.0]))


def test_positive_or_nonreciprocal_model_response_is_rejected():
    positive = variational_stability_diagnostic(
        np.diag([0.1, -2.0]),
        -np.diag([0.2, 0.3]),
    )
    assert positive["decisions"]["model_passivity_gate_passed"] is False
    assert positive["decisions"]["all_local_stability_gates_passed"] is False

    nonreciprocal = variational_stability_diagnostic(
        np.asarray([[-2.0, 0.5], [0.0, -3.0]]),
        -np.diag([0.1, 0.2]),
    )
    assert nonreciprocal["decisions"]["model_reciprocity_gate_passed"] is False
    assert nonreciprocal["decisions"]["all_local_stability_gates_passed"] is False


def test_singular_model_response_has_no_legendre_combined_hessian():
    result = variational_stability_diagnostic(
        -np.diag([2.0, 0.0]),
        -np.diag([0.1, 0.2]),
    )
    assert result["combined_hessian"] is None
    assert result["decisions"]["model_local_invertibility_gate_passed"] is False
    assert result["decisions"]["combined_hessian_positive_gate_passed"] is False
    assert result["decisions"]["all_local_stability_gates_passed"] is False


def test_threshold_contract_rejects_loose_or_nonfinite_values():
    with pytest.raises(ValueError, match="below one"):
        VariationalStabilityThresholds(feedback_spectral_radius_maximum=1.0)
    with pytest.raises(ValueError, match="positive finite"):
        VariationalStabilityThresholds(symmetry_relative=float("nan"))
    with pytest.raises(TypeError, match="positive finite"):
        VariationalStabilityThresholds(absolute_eigenvalue=True)
