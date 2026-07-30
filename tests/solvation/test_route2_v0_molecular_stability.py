from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_stability import (
    V0_MOLECULAR_HESSIAN_STABILITY_CONSTRUCTION,
    certify_route2_v0_molecular_hessian_stability,
)


class _SyntheticMolecularHessian:
    """A weighted finite-dimensional Hessian action for certificate tests."""

    def __init__(self, weighted_matrix: np.ndarray, weights: np.ndarray) -> None:
        self._weighted_matrix = np.asarray(weighted_matrix, dtype=float)
        self._weights = np.asarray(weights, dtype=float)
        self.projection = SimpleNamespace(
            quadrature=SimpleNamespace(phase_space_weights_bohr3=self._weights)
        )

    def dimensionless_hessian_matvec(
        self,
        configuration_density_bohr3: np.ndarray,
        direction_bohr3: np.ndarray,
    ) -> np.ndarray:
        del configuration_density_bohr3
        sqrt_weights = np.sqrt(self._weights)
        return (
            self._weighted_matrix @ (sqrt_weights * np.asarray(direction_bohr3))
        ) / sqrt_weights


def test_dense_molecular_hessian_certificate_uses_the_declared_quadrature_pairing():
    weights = np.array([2.0, 8.0])
    expected_weighted_matrix = np.array([[2.0, 0.25], [0.25, 3.0]])
    functional = _SyntheticMolecularHessian(expected_weighted_matrix, weights)
    density = np.array([0.2, 0.4])

    certificate = certify_route2_v0_molecular_hessian_stability(
        functional,
        density,
        maximum_dimension=2,
    )
    direction = np.array([0.3, -0.1])
    action = functional.dimensionless_hessian_matvec(density, direction)
    transformed_direction = np.sqrt(weights) * direction

    assert certificate.construction == V0_MOLECULAR_HESSIAN_STABILITY_CONSTRUCTION
    assert certificate.classification == "positive-definite"
    assert certificate.is_positive_definite
    assert (
        certificate.reciprocity_relative_frobenius_residual
        <= certificate.reciprocity_relative_tolerance
    )
    np.testing.assert_allclose(
        certificate.weighted_hessian_matrix,
        expected_weighted_matrix,
        rtol=0.0,
        atol=3.0e-15,
    )
    assert transformed_direction @ certificate.weighted_hessian_matrix @ (
        transformed_direction
    ) == pytest.approx(
        np.sum(weights * direction * action),
        rel=2.0e-14,
        abs=2.0e-15,
    )
    with pytest.raises(ValueError):
        certificate.weighted_hessian_matrix[0, 0] = 0.0
    with pytest.raises(ValueError, match="operator_two_norm_bohr3"):
        replace(certificate, operator_two_norm_bohr3=0.0)


def test_dense_molecular_hessian_certificate_reports_negative_modes_without_clipping():
    functional = _SyntheticMolecularHessian(
        np.array([[2.0, 0.0], [0.0, -0.5]]),
        np.array([1.0, 3.0]),
    )

    certificate = certify_route2_v0_molecular_hessian_stability(
        functional,
        np.array([0.2, 0.4]),
    )

    assert certificate.classification == "negative-mode"
    assert not certificate.is_positive_definite
    assert certificate.minimum_eigenvalue_bohr3 < (
        -certificate.numerical_eigenvalue_tolerance_bohr3
    )


def test_dense_molecular_hessian_certificate_reports_numerical_singularity_without_repair():
    functional = _SyntheticMolecularHessian(np.zeros((2, 2)), np.array([1.0, 2.0]))

    certificate = certify_route2_v0_molecular_hessian_stability(
        functional,
        np.array([0.2, 0.4]),
    )

    assert certificate.classification == "numerically-singular"
    assert not certificate.is_positive_definite
    assert abs(certificate.minimum_eigenvalue_bohr3) <= (
        certificate.numerical_eigenvalue_tolerance_bohr3
    )


def test_dense_molecular_hessian_certificate_rejects_nonreciprocity_instead_of_hiding_it():
    functional = _SyntheticMolecularHessian(
        np.array([[1.0, 0.5], [0.0, 1.0]]),
        np.ones(2),
    )

    with pytest.raises(RuntimeError, match="refusing to symmetrise"):
        certify_route2_v0_molecular_hessian_stability(
            functional,
            np.array([0.2, 0.4]),
        )


def test_dense_molecular_hessian_certificate_rejects_an_oversized_grid():
    functional = _SyntheticMolecularHessian(np.eye(2), np.ones(2))

    with pytest.raises(ValueError, match="dense diagnostic limit"):
        certify_route2_v0_molecular_hessian_stability(
            functional,
            np.array([0.2, 0.4]),
            maximum_dimension=1,
        )
