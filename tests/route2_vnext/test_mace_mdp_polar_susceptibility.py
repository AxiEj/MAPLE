from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from maple.solvation.coupling.separated_operators import (
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
)
from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE
from maple.solvation.models.mace_mdp_polar_hybrid import (
    PermanentAnchoredInducedSourceModel,
)
from maple.solvation.models.mace_mdp_polar_susceptibility import (
    MDP_POLAR_ARITHMETIC_TANGENT_MODEL_PROFILE_ID,
    MDP_POLAR_ARITHMETIC_TANGENT_RESPONSE_PROVIDER_ID,
    MDPPolarArithmeticTangentResponse,
)
from maple.solvation.release.uniform_response_manifold import (
    affine_uniform_native_field,
    molecular_dipole_eangstrom,
)


def _geometry() -> Atoms:
    atoms = Atoms(
        "CON",
        positions=[
            [-0.8, 0.1, 0.2],
            [0.5, -0.4, 0.1],
            [1.1, 0.6, -0.3],
        ],
    )
    atoms.info["charge"] = 0
    atoms.info["multiplicity"] = 1
    return atoms


def _arithmetic_chart(atom_count: int) -> np.ndarray:
    result = np.zeros((3, atom_count * 8), dtype=float)
    for atom in range(atom_count):
        offset = atom * 8
        for channel in (4, 7):
            result[0, offset + channel] = 1.0 / (2.0 * atom_count)
        for channel in (2, 5):
            result[1, offset + channel] = 1.0 / (2.0 * atom_count)
        for channel in (3, 6):
            result[2, offset + channel] = 1.0 / (2.0 * atom_count)
    return result


class _MDP:
    provider_id = "test.mdp.v1"
    model_profile_id = "test.mdp-profile.v1"

    def __init__(self, alpha: np.ndarray) -> None:
        self.digest = "1" * 64
        self.alpha = np.asarray(alpha, dtype=float)
        self.calls = 0

    def configuration_sha256(self) -> str:
        return self.digest

    def evaluate(self, _geometry: object) -> object:
        self.calls += 1
        return SimpleNamespace(
            public_polarizability_eangstrom2_per_volt=self.alpha.copy()
        )


class _NonlinearResponse:
    provider_id = "test.polar.v1"
    model_profile_id = "test.polar-profile.v1"
    provenance_sha256 = "2" * 64
    source_space = ATOMIC_L1_SOURCE_SPACE
    receiver_space = MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE
    long_range_evaluator_profile = (
        "graph-longrange-analytic-gaussian-multipole-realspace-v1"
    )
    field_energy_pairing_sha256 = "3" * 64
    coordinate_derivative_available = True

    def __init__(self, geometry: Atoms) -> None:
        self.digest = "4" * 64
        self.zero = np.array(
            [
                [-0.10, 0.02, -0.01, 0.03],
                [0.04, -0.03, 0.02, -0.01],
                [0.06, 0.01, -0.02, -0.02],
            ],
            dtype=float,
        )
        positions = np.asarray(geometry.positions, dtype=float)
        count = len(positions)
        basis = np.stack(
            [
                affine_uniform_native_field(positions, np.eye(3)[axis])
                for axis in range(3)
            ],
            axis=-1,
        )
        rng = np.random.default_rng(20260818)
        source_uniform = rng.normal(scale=0.04, size=(count, 4, 3))
        source_uniform[:, 0, :] -= np.mean(
            source_uniform[:, 0, :], axis=0, keepdims=True
        )
        molecular = np.column_stack(
            [
                molecular_dipole_eangstrom(positions, source_uniform[:, :, axis])
                for axis in range(3)
            ]
        )
        desired = -np.array([[0.8, 0.04, 0.01], [0.01, 1.0, -0.02], [0.03, 0.02, 1.2]])
        source_uniform = np.einsum(
            "nsc,cd->nsd", source_uniform, np.linalg.solve(molecular, desired)
        )
        chart = _arithmetic_chart(count)
        flat_basis = basis.reshape(count * 8, 3)
        assert np.allclose(chart @ flat_basis, np.eye(3), atol=2.0e-14)
        self.jacobian = source_uniform.reshape(count * 4, 3) @ chart

        kernel = rng.normal(size=count * 8)
        kernel -= flat_basis @ (chart @ kernel)
        kernel /= np.linalg.norm(kernel)
        nonlinear_source = rng.normal(scale=0.02, size=(count, 4))
        nonlinear_source[:, 0] -= np.mean(nonlinear_source[:, 0])
        self.kernel = kernel
        self.nonlinear_source = nonlinear_source.reshape(-1)

    def configuration_sha256(self) -> str:
        return self.digest

    def _source(self, field: object) -> np.ndarray:
        values = np.asarray(field, dtype=float).reshape(-1)
        nonlinear_coordinate = float(np.vdot(self.kernel, values))
        return (
            self.zero.reshape(-1)
            + self.jacobian @ values
            + 0.5 * self.nonlinear_source * nonlinear_coordinate**2
        ).reshape(self.zero.shape)

    def evaluate_source(self, _geometry: object, field: object) -> np.ndarray:
        return self._source(field)

    def field_jvp(
        self, _geometry: object, field: object, direction: object
    ) -> np.ndarray:
        values = np.asarray(field, dtype=float).reshape(-1)
        tangent = np.asarray(direction, dtype=float).reshape(-1)
        result = self.jacobian @ tangent
        result += (
            self.nonlinear_source
            * float(np.vdot(self.kernel, values))
            * float(np.vdot(self.kernel, tangent))
        )
        return result.reshape(self.zero.shape)

    def field_vjp(
        self, _geometry: object, field: object, cotangent: object
    ) -> np.ndarray:
        values = np.asarray(field, dtype=float).reshape(-1)
        source_bar = np.asarray(cotangent, dtype=float).reshape(-1)
        result = self.jacobian.T @ source_bar
        result += (
            self.kernel
            * float(np.vdot(self.nonlinear_source, source_bar))
            * float(np.vdot(self.kernel, values))
        )
        return result.reshape(len(self.zero), 8)

    def dense_source_jacobian(self, geometry: object, field: object) -> np.ndarray:
        count = len(geometry)
        columns = []
        for index in range(count * 8):
            direction = np.zeros((count, 8))
            direction.reshape(-1)[index] = 1.0
            columns.append(self.field_jvp(geometry, field, direction).reshape(-1))
        return np.column_stack(columns)

    def vacuum_energy_ev(self, _geometry: object) -> float:
        return -7.5

    def vacuum_forces_ev_per_angstrom(self, geometry: object) -> np.ndarray:
        return np.zeros((len(geometry), 3))

    def coordinate_vjp(
        self, geometry: object, _field: object, _cotangent: object
    ) -> np.ndarray:
        return np.zeros((len(geometry), 3))

    def conditioned_raw_energy_ev(self, _geometry: object, field: object) -> float:
        return 0.5 * float(np.vdot(field, field))


class _Permanent:
    provider_id = "test.permanent.v1"
    model_profile_id = "test.permanent-profile.v1"
    source_space = ATOMIC_L1_SOURCE_SPACE

    def configuration_sha256(self) -> str:
        return "5" * 64

    def evaluate_source(self, geometry: object) -> np.ndarray:
        return np.zeros((len(geometry), 4))

    def source_position_vjp(
        self, geometry: object, _source_cotangent: object
    ) -> np.ndarray:
        return np.zeros((len(geometry), 3))


def _problem():
    geometry = _geometry()
    alpha = np.array([[1.15, 0.06, 0.02], [0.06, 1.35, -0.03], [0.02, -0.03, 1.55]])
    mdp = _MDP(alpha)
    base = _NonlinearResponse(geometry)
    corrected = MDPPolarArithmeticTangentResponse(mdp=mdp, base=base)
    return geometry, alpha, mdp, base, corrected


def test_zero_field_identity_and_exact_mdp_molecular_susceptibility() -> None:
    geometry, alpha, mdp, base, corrected = _problem()
    count = len(geometry)
    zero = np.zeros((count, 8))
    np.testing.assert_array_equal(
        corrected.evaluate_source(geometry, zero), base.evaluate_source(geometry, zero)
    )

    positions = np.asarray(geometry.positions)
    basis = np.stack(
        [affine_uniform_native_field(positions, np.eye(3)[axis]) for axis in range(3)],
        axis=-1,
    )
    source_jacobian = np.stack(
        [corrected.field_jvp(geometry, zero, basis[:, :, axis]) for axis in range(3)],
        axis=-1,
    )
    molecular = np.column_stack(
        [
            molecular_dipole_eangstrom(positions, source_jacobian[:, :, axis])
            for axis in range(3)
        ]
    )
    np.testing.assert_allclose(molecular, -alpha, atol=3.0e-13, rtol=0.0)
    assert np.max(np.abs(np.sum(source_jacobian[:, 0, :], axis=0))) < 3.0e-14
    assert mdp.calls == 1
    corrected.evaluate_source(geometry, zero)
    assert mdp.calls == 1


def test_jvp_vjp_dense_jacobian_and_finite_difference_close() -> None:
    geometry, _alpha, _mdp, _base, corrected = _problem()
    rng = np.random.default_rng(7)
    field = rng.normal(scale=0.02, size=(len(geometry), 8))
    direction = rng.normal(size=field.shape)
    cotangent = rng.normal(size=(len(geometry), 4))
    jvp = corrected.field_jvp(geometry, field, direction)
    vjp = corrected.field_vjp(geometry, field, cotangent)
    assert float(np.vdot(cotangent, jvp)) == pytest.approx(
        float(np.vdot(vjp, direction)), abs=3.0e-13
    )
    dense = corrected.dense_source_jacobian(geometry, field)
    np.testing.assert_allclose(
        dense @ direction.reshape(-1), jvp.reshape(-1), atol=3.0e-13, rtol=0.0
    )
    step = 2.0e-6
    finite = (
        corrected.evaluate_source(geometry, field + step * direction)
        - corrected.evaluate_source(geometry, field - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(finite, jvp, atol=2.0e-10, rtol=2.0e-10)


def test_correction_is_linear_and_leaves_arithmetic_chart_kernel_unchanged() -> None:
    geometry, _alpha, _mdp, base, corrected = _problem()
    rng = np.random.default_rng(19)
    chart = corrected.chart_for_geometry(geometry)
    field = rng.normal(size=(len(geometry), 8))
    field -= np.einsum(
        "nfc,c->nf",
        chart.uniform_native_basis,
        chart.uniform_coordinates(field),
    )
    np.testing.assert_allclose(chart.uniform_coordinates(field), 0.0, atol=3.0e-14)
    np.testing.assert_allclose(
        corrected.evaluate_source(geometry, field),
        base.evaluate_source(geometry, field),
        atol=3.0e-14,
        rtol=0.0,
    )

    direction = rng.normal(size=field.shape)
    first = corrected.field_jvp(geometry, field, direction) - base.field_jvp(
        geometry, field, direction
    )
    second_field = rng.normal(size=field.shape)
    second = corrected.field_jvp(geometry, second_field, direction) - base.field_jvp(
        geometry, second_field, direction
    )
    np.testing.assert_allclose(first, second, atol=3.0e-14, rtol=0.0)


def test_identity_drift_geometry_cache_and_coordinate_derivative_fail_closed() -> None:
    geometry, _alpha, mdp, base, corrected = _problem()
    assert corrected.provider_id == MDP_POLAR_ARITHMETIC_TANGENT_RESPONSE_PROVIDER_ID
    assert corrected.model_profile_id == MDP_POLAR_ARITHMETIC_TANGENT_MODEL_PROFILE_ID
    assert corrected.capabilities == ()
    assert corrected.variational_functional_admitted is False
    assert corrected.coordinate_derivative_available is False
    with pytest.raises(NotImplementedError, match="complete coordinate VJP"):
        corrected.coordinate_vjp(
            geometry, np.zeros((len(geometry), 8)), np.zeros((len(geometry), 4))
        )
    with pytest.raises(AttributeError, match="immutable"):
        corrected.provider_id = "tampered"

    corrected.chart_for_geometry(geometry)
    moved = geometry.copy()
    moved.positions[0, 0] += 1.0e-3
    corrected.chart_for_geometry(moved)
    assert mdp.calls == 2
    base.digest = "6" * 64
    with pytest.raises(RuntimeError, match="configuration drifted"):
        corrected.configuration_sha256()


def test_hybrid_propagates_missing_response_coordinate_derivative() -> None:
    geometry, _alpha, _mdp, _base, corrected = _problem()
    hybrid = PermanentAnchoredInducedSourceModel(
        _Permanent(),
        corrected,
        provider_id="test.corrected-hybrid.v1",
        model_profile_id="test.corrected-hybrid-profile.v1",
    )
    anchor = hybrid.prepare(geometry)
    assert hybrid.coordinate_derivative_available is False
    with pytest.raises(NotImplementedError, match="no complete coordinate derivative"):
        hybrid.induced_source_position_vjp(
            geometry,
            anchor,
            np.zeros((len(geometry), 8)),
            np.ones((len(geometry), 4)),
        )


def test_rank_deficient_original_uniform_response_fails_closed() -> None:
    geometry = _geometry()
    alpha = np.diag([1.1, 1.2, 1.3])
    mdp = _MDP(alpha)
    base = _NonlinearResponse(geometry)
    count = len(geometry)
    basis = np.stack(
        [
            affine_uniform_native_field(geometry.positions, np.eye(3)[axis])
            for axis in range(3)
        ],
        axis=-1,
    )
    source_uniform = base.jacobian @ basis.reshape(count * 8, 3)
    source_uniform[:, 2] = 0.0
    base.jacobian = source_uniform @ _arithmetic_chart(count)
    corrected = MDPPolarArithmeticTangentResponse(mdp=mdp, base=base)

    with pytest.raises(ValueError, match="rank three"):
        corrected.chart_for_geometry(geometry)
