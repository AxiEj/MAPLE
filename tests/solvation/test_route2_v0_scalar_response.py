from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.gto_density import (
    density_to_external_field_order,
    external_field_to_density_order,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_scalar_response import (
    evaluate_anchored_scalar_response_v0,
)


class _LinearDensityResponse:
    """Linear field-to-density response in external Cartesian order."""

    def __init__(self, response_matrix: np.ndarray, *, atom_count: int):
        matrix = np.asarray(response_matrix, dtype=float)
        expected_shape = (4 * atom_count, 4 * atom_count)
        if matrix.shape != expected_shape:
            raise ValueError(f"Response matrix must have shape {expected_shape}.")
        self.matrix = matrix
        self.atom_count = int(atom_count)

    def vjp(self, density_cotangent: np.ndarray) -> np.ndarray:
        external_cotangent = density_to_external_field_order(
            np.asarray(density_cotangent, dtype=float)
        )
        return (self.matrix.T @ external_cotangent.reshape(-1)).reshape(
            self.atom_count,
            4,
        )


def _evaluate_linear_candidate(
    field: np.ndarray,
    *,
    gas_density_external: np.ndarray,
    response_matrix: np.ndarray,
    energy_hessian: np.ndarray,
    zero_field_energy_gradient: np.ndarray,
):
    atom_count = field.shape[0]
    field_vector = field.reshape(-1)
    gas_vector = gas_density_external.reshape(-1)
    zero_gradient_vector = zero_field_energy_gradient.reshape(-1)
    density_external = gas_vector + response_matrix @ field_vector
    energy_gradient = (
        zero_gradient_vector + energy_hessian @ field_vector
    ).reshape(atom_count, 4)
    energy_change = float(
        np.dot(zero_gradient_vector, field_vector)
        + 0.5 * field_vector @ energy_hessian @ field_vector
    )
    return evaluate_anchored_scalar_response_v0(
        field_conditioned_model_energy_ev=3.0 + energy_change,
        zero_field_model_energy_ev=3.0,
        reaction_field_values_ev=field,
        model_density_coefficients=external_field_to_density_order(
            density_external.reshape(atom_count, 4)
        ),
        field_conditioned_energy_field_gradient=energy_gradient,
        zero_field_energy_field_gradient=zero_field_energy_gradient,
        density_response=_LinearDensityResponse(
            response_matrix,
            atom_count=atom_count,
        ),
    )


def test_zero_field_anchor_reproduces_the_frozen_model_density():
    gas_density_external = np.asarray(
        [
            [0.3, 0.2, -0.1, 0.4],
            [-0.3, -0.5, 0.6, -0.2],
        ]
    )
    zero_gradient = np.asarray(
        [
            [0.9, -0.8, 0.7, -0.6],
            [0.5, -0.4, 0.3, -0.2],
        ]
    )

    state = _evaluate_linear_candidate(
        np.zeros((2, 4)),
        gas_density_external=gas_density_external,
        response_matrix=np.diag(np.linspace(-0.2, -0.05, 8)),
        energy_hessian=np.diag(np.linspace(0.1, 0.3, 8)),
        zero_field_energy_gradient=zero_gradient,
    )

    np.testing.assert_allclose(
        state.response_density_coefficients,
        external_field_to_density_order(gas_density_external),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert state.anchored_field_scalar_change_ev == pytest.approx(0.0)
    assert state.electrostatic_energy_ev == pytest.approx(0.0)


def test_scalar_response_matches_the_exact_gradient_of_the_anchored_scalar():
    atom_count = 2
    dimension = 4 * atom_count
    rng = np.random.default_rng(20260729)
    raw_response = rng.normal(scale=0.03, size=(dimension, dimension))
    energy_factor = rng.normal(scale=0.02, size=(dimension, dimension))
    energy_hessian = 0.5 * (energy_factor + energy_factor.T)
    gas_density_external = np.asarray(
        [
            [0.2, 0.1, -0.3, 0.4],
            [-0.2, 0.5, -0.6, -0.1],
        ]
    )
    zero_gradient = rng.normal(scale=0.2, size=(atom_count, 4))
    field = rng.normal(scale=0.1, size=(atom_count, 4))

    state = _evaluate_linear_candidate(
        field,
        gas_density_external=gas_density_external,
        response_matrix=raw_response,
        energy_hessian=energy_hessian,
        zero_field_energy_gradient=zero_gradient,
    )

    field_vector = field.reshape(-1)
    expected_external = (
        gas_density_external.reshape(-1)
        + (energy_hessian + raw_response + raw_response.T) @ field_vector
    ).reshape(atom_count, 4)
    np.testing.assert_allclose(
        density_to_external_field_order(state.response_density_coefficients),
        expected_external,
        rtol=0.0,
        atol=2.0e-15,
    )


def test_finite_difference_v0_response_is_reciprocal_by_construction():
    atom_count = 2
    dimension = 4 * atom_count
    rng = np.random.default_rng(9173)
    raw_response = rng.normal(scale=0.04, size=(dimension, dimension))
    energy_factor = rng.normal(scale=0.03, size=(dimension, dimension))
    energy_hessian = 0.5 * (energy_factor + energy_factor.T)
    gas_density_external = np.zeros((atom_count, 4))
    zero_gradient = rng.normal(scale=0.1, size=(atom_count, 4))
    field = rng.normal(scale=0.05, size=(atom_count, 4))
    step = 1.0e-6
    jacobian = np.empty((dimension, dimension), dtype=float)

    for column in range(dimension):
        direction = np.zeros(dimension, dtype=float)
        direction[column] = step
        plus = _evaluate_linear_candidate(
            field + direction.reshape(atom_count, 4),
            gas_density_external=gas_density_external,
            response_matrix=raw_response,
            energy_hessian=energy_hessian,
            zero_field_energy_gradient=zero_gradient,
        )
        minus = _evaluate_linear_candidate(
            field - direction.reshape(atom_count, 4),
            gas_density_external=gas_density_external,
            response_matrix=raw_response,
            energy_hessian=energy_hessian,
            zero_field_energy_gradient=zero_gradient,
        )
        plus_external = density_to_external_field_order(
            plus.response_density_coefficients
        ).reshape(-1)
        minus_external = density_to_external_field_order(
            minus.response_density_coefficients
        ).reshape(-1)
        jacobian[:, column] = (plus_external - minus_external) / (2.0 * step)

    np.testing.assert_allclose(jacobian, jacobian.T, rtol=0.0, atol=1.0e-10)


def test_variational_limit_reduces_to_the_existing_half_coupling_ledger():
    atom_count = 2
    susceptibility = np.diag(np.linspace(-0.12, -0.03, 8))
    gas_density_external = np.asarray(
        [
            [0.1, 0.2, -0.3, 0.4],
            [-0.1, -0.5, 0.6, -0.2],
        ]
    )
    zero_gradient = np.zeros((atom_count, 4))
    field = np.asarray(
        [
            [0.03, -0.02, 0.01, 0.04],
            [-0.01, 0.05, -0.03, 0.02],
        ]
    )

    state = _evaluate_linear_candidate(
        field,
        gas_density_external=gas_density_external,
        response_matrix=susceptibility,
        energy_hessian=-susceptibility,
        zero_field_energy_gradient=zero_gradient,
    )

    np.testing.assert_allclose(
        state.response_density_coefficients,
        state.model_density_coefficients,
        rtol=0.0,
        atol=1.0e-15,
    )
    existing_ledger = (
        state.field_conditioned_model_energy_ev
        - state.zero_field_model_energy_ev
        + 0.5 * state.model_density_field_coupling_ev
    )
    assert state.electrostatic_energy_ev == pytest.approx(
        existing_ledger,
        abs=1.0e-15,
    )


def test_v0_reports_charge_and_rejects_non_local_field_shapes():
    state = _evaluate_linear_candidate(
        np.zeros((2, 4)),
        gas_density_external=np.asarray(
            [
                [0.15, 0.0, 0.0, 0.0],
                [-0.15, 0.0, 0.0, 0.0],
            ]
        ),
        response_matrix=np.zeros((8, 8)),
        energy_hessian=np.zeros((8, 8)),
        zero_field_energy_gradient=np.zeros((2, 4)),
    )
    assert state.response_total_charge_e == pytest.approx(0.0, abs=1.0e-15)

    with pytest.raises(ValueError, match=r"shape \(n_atoms, 4\)"):
        evaluate_anchored_scalar_response_v0(
            field_conditioned_model_energy_ev=0.0,
            zero_field_model_energy_ev=0.0,
            reaction_field_values_ev=np.zeros((2, 8)),
            model_density_coefficients=np.zeros((2, 4)),
            field_conditioned_energy_field_gradient=np.zeros((2, 4)),
            zero_field_energy_field_gradient=np.zeros((2, 4)),
            density_response=_LinearDensityResponse(
                np.zeros((8, 8)),
                atom_count=2,
            ),
        )
