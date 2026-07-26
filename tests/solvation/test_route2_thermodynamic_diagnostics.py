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
    response_reciprocity_diagnostic,
    response_stability_diagnostic,
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
