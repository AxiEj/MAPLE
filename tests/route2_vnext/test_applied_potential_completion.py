from __future__ import annotations

import hashlib

import numpy as np
import pytest

from maple.solvation.coupling.separated_operators import (
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
    SEPARATED_MACE_POLAR_HARMONIC_COUPLING_ID,
)
from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE
from maple.solvation.models.applied_potential import (
    APPLIED_POTENTIAL_WORK_COMPLETION_ID,
    MACE_POLAR_ORIGINAL_SOURCE_APPLIED_WORK_BRIDGE,
    AppliedPotentialEnergyCompletion,
)


class _NonlinearResponse:
    provider_id = "test.route2.nonlinear-response.v1"
    model_profile_id = "test-route2-nonlinear-response-v1"
    coupling_id = SEPARATED_MACE_POLAR_HARMONIC_COUPLING_ID
    provenance_sha256 = hashlib.sha256(b"nonlinear-response").hexdigest()
    source_space = ATOMIC_L1_SOURCE_SPACE
    receiver_space = MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE

    _source_offset = np.asarray([0.3, -0.05, 0.07, 0.02])
    _position_source = np.asarray([0.12, -0.04, 0.03, 0.08])
    _source_jacobian = np.asarray(
        [
            [0.11, -0.02, 0.03, 0.04, 0.01, -0.03, 0.02, 0.05],
            [0.02, 0.07, -0.01, 0.03, -0.05, 0.04, 0.01, -0.02],
            [-0.03, 0.01, 0.09, -0.02, 0.02, 0.03, -0.04, 0.01],
            [0.04, -0.02, 0.01, 0.08, -0.01, 0.02, 0.05, -0.03],
        ]
    )
    _raw_field_hessian = np.diag(np.linspace(0.15, 0.36, 8))

    def configuration_sha256(self) -> str:
        return hashlib.sha256(b"nonlinear-response-configuration-v1").hexdigest()

    def evaluate_source(self, geometry, field):
        positions = np.asarray(geometry, dtype=float)
        values = np.asarray(field, dtype=float)
        linear = values @ self._source_jacobian.T
        nonlinear = 0.03 * values[:, :4] ** 2
        return (
            self._source_offset
            + linear
            + nonlinear
            + positions[:, :1] * self._position_source
        )

    def field_vjp(self, geometry, field, source_cotangent):
        del geometry
        values = np.asarray(field, dtype=float)
        cotangent = np.asarray(source_cotangent, dtype=float)
        jacobian = np.broadcast_to(
            self._source_jacobian, (values.shape[0], 4, 8)
        ).copy()
        diagonal = np.arange(4)
        jacobian[:, diagonal, diagonal] += 0.06 * values[:, :4]
        return np.einsum("ns,nsr->nr", cotangent, jacobian)

    def coordinate_vjp(self, geometry, field, source_cotangent):
        del field
        result = np.zeros_like(np.asarray(geometry, dtype=float))
        result[:, 0] = np.asarray(source_cotangent) @ self._position_source
        return result

    def conditioned_raw_energy_ev(self, geometry, field):
        positions = np.asarray(geometry, dtype=float)
        values = np.asarray(field, dtype=float)
        return float(
            0.5 * np.vdot(positions, positions)
            + 0.5 * np.einsum("ni,ij,nj->", values, self._raw_field_hessian, values)
        )

    def conditioned_raw_energy_field_gradient(self, geometry, field):
        del geometry
        return np.asarray(field, dtype=float) @ self._raw_field_hessian

    def conditioned_raw_energy_fixed_field_coordinate_gradient(self, geometry, field):
        del field
        return np.asarray(geometry, dtype=float).copy()


def _geometry() -> np.ndarray:
    return np.asarray([[0.1, -0.2, 0.3], [1.2, 0.4, -0.1]])


def _field() -> np.ndarray:
    return np.linspace(-0.04, 0.05, 16).reshape(2, 8)


def _completion() -> AppliedPotentialEnergyCompletion:
    return AppliedPotentialEnergyCompletion(
        _NonlinearResponse(), MACE_POLAR_ORIGINAL_SOURCE_APPLIED_WORK_BRIDGE
    )


def test_source4_field8_bridge_is_the_exact_registered_work_pairing():
    bridge = MACE_POLAR_ORIGINAL_SOURCE_APPLIED_WORK_BRIDGE
    source = np.asarray([[1.0, 2.0, 3.0, 4.0]])
    field = np.asarray([[11.0, 13.0, 17.0, 19.0, 23.0, 29.0, 31.0, 37.0]])
    embedded = bridge.embed_source(source)
    np.testing.assert_array_equal(
        embedded, np.asarray([[1.0, 0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0]])
    )
    assert bridge.work_ev(source, field) == pytest.approx(
        1.0 * 11.0 + 2.0 * 17.0 + 3.0 * 19.0 + 4.0 * 23.0
    )
    np.testing.assert_array_equal(
        bridge.source_cotangent(field), np.asarray([[11.0, 17.0, 19.0, 23.0]])
    )


def test_completed_scalar_components_are_explicit_and_capabilities_stay_closed():
    completion = _completion()
    geometry = _geometry()
    field = _field()
    response = completion.response
    source = response.evaluate_source(geometry, field)
    expected_raw = response.conditioned_raw_energy_ev(geometry, field)
    expected_work = completion.bridge.work_ev(source, field)
    components = completion.components_ev(geometry, field)
    assert components == pytest.approx(
        {
            "conditioned_raw_energy_ev": expected_raw,
            "explicit_applied_work_ev": expected_work,
            "complete_energy_ev": expected_raw + expected_work,
        }
    )
    assert completion.completion_id == APPLIED_POTENTIAL_WORK_COMPLETION_ID
    assert completion.capabilities == ()
    assert completion.variational_functional_admitted is False
    assert completion.metadata()["capabilities"] == {tier: False for tier in "EFHVM"}
    assert "not a common MACE-continuum" in completion.metadata()["claim_boundary"]


def test_completed_scalar_field_gradient_matches_all_coordinate_finite_differences():
    completion = _completion()
    geometry = _geometry()
    field = _field()
    analytic = completion.field_gradient(geometry, field)
    finite = np.zeros_like(field)
    step = 2.0e-6
    for index in np.ndindex(field.shape):
        plus = field.copy()
        minus = field.copy()
        plus[index] += step
        minus[index] -= step
        finite[index] = (
            completion.evaluate_energy_ev(geometry, plus)
            - completion.evaluate_energy_ev(geometry, minus)
        ) / (2.0 * step)
    np.testing.assert_allclose(analytic, finite, rtol=2.0e-9, atol=2.0e-10)


def test_completed_scalar_fixed_field_coordinate_gradient_matches_finite_difference():
    completion = _completion()
    geometry = _geometry()
    field = _field()
    analytic = completion.fixed_field_coordinate_gradient(geometry, field)
    finite = np.zeros_like(geometry)
    step = 2.0e-6
    for index in np.ndindex(geometry.shape):
        plus = geometry.copy()
        minus = geometry.copy()
        plus[index] += step
        minus[index] -= step
        finite[index] = (
            completion.evaluate_energy_ev(plus, field)
            - completion.evaluate_energy_ev(minus, field)
        ) / (2.0 * step)
    np.testing.assert_allclose(analytic, finite, rtol=2.0e-9, atol=2.0e-10)


def test_completion_rejects_mutation_and_missing_total_derivative_hook():
    completion = _completion()
    with pytest.raises(AttributeError, match="immutable"):
        completion.provider_id = "changed"

    response = _NonlinearResponse()
    response.conditioned_raw_energy_fixed_field_coordinate_gradient = None
    with pytest.raises(TypeError, match="fixed_field_coordinate_gradient"):
        AppliedPotentialEnergyCompletion(
            response, MACE_POLAR_ORIGINAL_SOURCE_APPLIED_WORK_BRIDGE
        )
