from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
    GaussianMixtureAtom,
)
from maple.solvation.coupling.adt_radial_shape import (
    load_repository_adt_radial_shape_registry,
)
from maple.solvation.coupling.separated_operators import (
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
)
from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE
from maple.solvation.models.mace_mdp_polar_adt import (
    MACE_POLAR_ZERO_POINT_PERMANENT_MODEL_PROFILE_ID,
    MACE_POLAR_ZERO_POINT_PERMANENT_PROVIDER_ID,
    MACEPolarZeroFieldPointPermanentSource,
    MDP_POLAR_CANONICAL_ADT_MODEL_PROFILE_ID,
    MDP_POLAR_CANONICAL_ADT_RESPONSE_PROVIDER_ID,
    MDP_POLAR_ROLE_SEPARATED_ADT_CONTRACT,
    MDP_POLAR_ROLE_SEPARATED_ADT_MODEL_PROFILE_ID,
    MDP_POLAR_ROLE_SEPARATED_ADT_RESPONSE_PROVIDER_ID,
    MDPPolarCanonicalADTResponse,
)
from maple.solvation.release.uniform_response_manifold import (
    affine_uniform_native_field,
    molecular_dipole_eangstrom,
)
from maple.solvation.release.uniform_susceptibility_replacement import (
    arithmetic_uniform_gradient_left_inverse,
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


def _mixtures() -> dict[int, GaussianMixtureAtom]:
    return {
        6: GaussianMixtureAtom(np.array([4.0, 2.0]), np.array([0.8, 2.1])),
        7: GaussianMixtureAtom(np.array([5.0, 2.0]), np.array([0.9, 2.4])),
        8: GaussianMixtureAtom(np.array([6.0, 2.0]), np.array([1.0, 2.7])),
    }


class _MDPState:
    def __init__(self, alpha: np.ndarray) -> None:
        self.public_polarizability_eangstrom2_per_volt = np.asarray(alpha, dtype=float)

    @property
    def atomic_polarizabilities_eangstrom2_per_volt(self) -> np.ndarray:
        raise AssertionError("latent atomwise MDP alpha must not be read")

    @property
    def atomic_dipole_weights(self) -> np.ndarray:
        raise AssertionError("latent MDP dipole weights must not be read")


class _MDP:
    provider_id = "test.mdp.v1"
    model_profile_id = "test.mdp-profile.v1"

    def __init__(self, alpha: np.ndarray) -> None:
        self.alpha = np.asarray(alpha, dtype=float)
        self.digest = "1" * 64
        self.calls = 0

    def configuration_sha256(self) -> str:
        return self.digest

    def evaluate(self, _geometry: object) -> _MDPState:
        self.calls += 1
        return _MDPState(self.alpha.copy())


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

    def __init__(self, geometry: Atoms) -> None:
        self.digest = "4" * 64
        self.zero = np.array(
            [
                [-0.10, 0.02, -0.01, 0.03],
                [0.04, -0.03, 0.02, -0.01],
                [0.06, 0.01, -0.02, -0.02],
            ]
        )
        count = len(geometry)
        positions = np.asarray(geometry.positions)
        basis = np.stack(
            [
                affine_uniform_native_field(positions, np.eye(3)[axis])
                for axis in range(3)
            ],
            axis=-1,
        )
        left = arithmetic_uniform_gradient_left_inverse(count)
        rng = np.random.default_rng(20260818)
        uniform_tangent = rng.normal(scale=0.04, size=(count, 4, 3))
        uniform_tangent[:, 0, :] -= np.mean(
            uniform_tangent[:, 0, :], axis=0, keepdims=True
        )
        self.jacobian = uniform_tangent.reshape(count * 4, 3) @ left

        flat_basis = basis.reshape(count * 8, 3)
        kernel = rng.normal(size=count * 8)
        kernel -= flat_basis @ (left @ kernel)
        kernel /= np.linalg.norm(kernel)
        nonlinear_source = rng.normal(scale=0.02, size=(count, 4))
        nonlinear_source[:, 0] -= np.mean(nonlinear_source[:, 0])
        self.kernel = kernel
        self.nonlinear_source = nonlinear_source.reshape(-1)

    def configuration_sha256(self) -> str:
        return self.digest

    def _source(self, field: object) -> np.ndarray:
        values = np.asarray(field, dtype=float).reshape(-1)
        coordinate = float(np.vdot(self.kernel, values))
        return (
            self.zero.reshape(-1)
            + self.jacobian @ values
            + 0.5 * self.nonlinear_source * coordinate**2
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

    def vacuum_energy_ev(self, _geometry: object) -> float:
        return -8.0

    def vacuum_forces_ev_per_angstrom(self, geometry: object) -> np.ndarray:
        return np.zeros((len(geometry), 3))


def _problem():
    geometry = _geometry()
    alpha = np.array([[1.15, 0.06, 0.02], [0.06, 1.35, -0.03], [0.02, -0.03, 1.55]])
    mdp = _MDP(alpha)
    base = _NonlinearResponse(geometry)
    response = MDPPolarCanonicalADTResponse(
        mdp=mdp,
        base=base,
        mixtures_by_atomic_number=_mixtures(),
        source_asset_sha256="9" * 64,
    )
    return geometry, alpha, mdp, base, response


def _uniform_basis(geometry: Atoms) -> np.ndarray:
    return np.stack(
        [
            affine_uniform_native_field(geometry.positions, np.eye(3)[axis])
            for axis in range(3)
        ],
        axis=-1,
    )


def test_zero_field_identity_and_exact_molecular_uniform_response() -> None:
    geometry, alpha, mdp, _base, response = _problem()
    zero = np.zeros((len(geometry), 8))
    components = response.evaluate_components(geometry, zero)
    np.testing.assert_array_equal(components.radial_residual_source4, 0.0)
    np.testing.assert_array_equal(components.adt_atomic_dipoles_eangstrom, 0.0)

    basis = _uniform_basis(geometry)
    radial = []
    adt = []
    for axis in range(3):
        result = response.field_jvp_components(geometry, zero, basis[:, :, axis])
        radial.append(
            molecular_dipole_eangstrom(
                geometry.positions, result.radial_residual_source4
            )
        )
        adt.append(np.sum(result.adt_atomic_dipoles_eangstrom, axis=0))
    np.testing.assert_allclose(np.column_stack(radial), 0.0, atol=3.0e-13)
    np.testing.assert_allclose(np.column_stack(adt), -alpha, atol=3.0e-13)
    assert mdp.calls == 1
    response.evaluate_components(geometry, zero)
    assert mdp.calls == 1


def test_nonuniform_chart_kernel_preserves_original_polar_increment() -> None:
    geometry, _alpha, _mdp, _base, response = _problem()
    rng = np.random.default_rng(17)
    chart = response.chart_for_geometry(geometry)
    field = rng.normal(size=(len(geometry), 8))
    field -= np.einsum(
        "nfc,c->nf",
        chart.uniform_native_basis,
        chart.uniform_coordinates(field),
    )
    np.testing.assert_allclose(chart.uniform_coordinates(field), 0.0, atol=3.0e-14)
    components = response.evaluate_components(geometry, field)
    np.testing.assert_allclose(
        components.radial_residual_source4,
        response.original_induced_source(geometry, field),
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        components.adt_atomic_dipoles_eangstrom, 0.0, atol=3.0e-14
    )


def test_direct_sum_jvp_vjp_dense_and_finite_difference() -> None:
    geometry, _alpha, _mdp, _base, response = _problem()
    rng = np.random.default_rng(23)
    field = rng.normal(scale=0.03, size=(len(geometry), 8))
    direction = rng.normal(size=field.shape)
    radial_bar = rng.normal(size=(len(geometry), 4))
    adt_bar = rng.normal(size=(len(geometry), 3))
    jvp = response.field_jvp_components(geometry, field, direction)
    vjp = response.field_vjp_components(
        geometry,
        field,
        radial_source_cotangent=radial_bar,
        adt_atomic_dipole_cotangent=adt_bar,
    )
    left = float(np.vdot(radial_bar, jvp.radial_residual_source4)) + float(
        np.vdot(adt_bar, jvp.adt_atomic_dipoles_eangstrom)
    )
    assert left == pytest.approx(float(np.vdot(vjp, direction)), abs=4.0e-13)

    radial_jacobian, adt_jacobian = response.dense_component_jacobians(geometry, field)
    np.testing.assert_allclose(
        radial_jacobian @ direction.reshape(-1),
        jvp.radial_residual_source4.reshape(-1),
        atol=4.0e-13,
    )
    np.testing.assert_allclose(
        adt_jacobian @ direction.reshape(-1),
        jvp.adt_atomic_dipoles_eangstrom.reshape(-1),
        atol=4.0e-13,
    )

    step = 2.0e-6
    plus = response.evaluate_components(geometry, field + step * direction)
    minus = response.evaluate_components(geometry, field - step * direction)
    np.testing.assert_allclose(
        (plus.radial_residual_source4 - minus.radial_residual_source4) / (2.0 * step),
        jvp.radial_residual_source4,
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    np.testing.assert_allclose(
        (plus.adt_atomic_dipoles_eangstrom - minus.adt_atomic_dipoles_eangstrom)
        / (2.0 * step),
        jvp.adt_atomic_dipoles_eangstrom,
        atol=2.0e-10,
        rtol=2.0e-10,
    )


def test_chart_uses_canonical_density_metric_not_mdp_atomic_gauge() -> None:
    geometry, _alpha, _mdp, _base, response = _problem()
    chart = response.chart_for_geometry(geometry)
    inverse = 1.0 / chart.adt_lift.self_work_hartree_per_ebohr2
    expected = inverse / np.sum(inverse)
    np.testing.assert_allclose(
        chart.adt_lift.atomic_dipole_weights, expected, atol=0.0, rtol=2.0e-14
    )
    assert np.ptp(expected) > 0.0


def test_identity_cache_drift_and_coordinate_derivative_fail_closed() -> None:
    geometry, _alpha, mdp, base, response = _problem()
    assert response.provider_id == MDP_POLAR_CANONICAL_ADT_RESPONSE_PROVIDER_ID
    assert response.model_profile_id == MDP_POLAR_CANONICAL_ADT_MODEL_PROFILE_ID
    assert response.capabilities == ()
    assert response.coordinate_derivative_available is False
    assert response.variational_functional_admitted is False
    with pytest.raises(NotImplementedError, match="complete coordinate VJP"):
        response.coordinate_vjp(geometry, np.zeros((len(geometry), 8)))
    with pytest.raises(AttributeError, match="immutable"):
        response.provider_id = "tampered"

    response.chart_for_geometry(geometry)
    moved = geometry.copy()
    moved.positions[0, 0] += 1.0e-3
    response.chart_for_geometry(moved)
    assert mdp.calls == 2
    base.digest = "6" * 64
    with pytest.raises(RuntimeError, match="configuration drifted"):
        response.configuration_sha256()


def test_polar_zero_field_point_permanent_is_exact_response_chart_view() -> None:
    geometry, _alpha, _mdp, _base, response = _problem()
    permanent = MACEPolarZeroFieldPointPermanentSource(response)
    chart = response.chart_for_geometry(geometry)

    assert permanent.provider_id == MACE_POLAR_ZERO_POINT_PERMANENT_PROVIDER_ID
    assert (
        permanent.model_profile_id == MACE_POLAR_ZERO_POINT_PERMANENT_MODEL_PROFILE_ID
    )
    assert permanent.response_configuration_sha256 == response.configuration_sha256()
    assert permanent.source_space == response.source_space
    np.testing.assert_array_equal(
        permanent.evaluate_source(geometry), chart.polar_zero_source4
    )
    assert not permanent.evaluate_source(geometry).flags.writeable
    with pytest.raises(NotImplementedError, match="coordinate VJP"):
        permanent.coordinate_vjp(geometry)
    with pytest.raises(AttributeError, match="immutable"):
        permanent.provider_id = "tampered"


def test_unsupported_element_and_bad_alpha_fail_closed() -> None:
    geometry, _alpha, mdp, base, _response = _problem()
    unsupported = Atoms("Si", positions=[[0.0, 0.0, 0.0]])
    response = MDPPolarCanonicalADTResponse(
        mdp=mdp,
        base=base,
        mixtures_by_atomic_number=_mixtures(),
        source_asset_sha256="9" * 64,
    )
    with pytest.raises(ValueError, match="no elements"):
        response.chart_for_geometry(unsupported)

    mdp.alpha = np.diag([1.0, 1.0, -0.1])
    moved = geometry.copy()
    moved.positions[0, 1] += 1.0e-3
    with pytest.raises(RuntimeError, match="positive definite"):
        response.chart_for_geometry(moved)


def test_role_separated_v2_accepts_iodine_without_changing_v1_semantics() -> None:
    geometry = Atoms(
        "COI",
        positions=[[-0.8, 0.1, 0.2], [0.5, -0.4, 0.1], [1.9, 0.6, -0.3]],
    )
    geometry.info["charge"] = 0
    geometry.info["multiplicity"] = 1
    alpha = np.asarray([[1.15, 0.06, 0.02], [0.06, 1.35, -0.03], [0.02, -0.03, 1.55]])
    mdp = _MDP(alpha)
    base = _NonlinearResponse(geometry)

    legacy = MDPPolarCanonicalADTResponse(
        mdp=mdp,
        base=base,
        mixtures_by_atomic_number=_mixtures(),
        source_asset_sha256="9" * 64,
    )
    assert legacy.provider_id == MDP_POLAR_CANONICAL_ADT_RESPONSE_PROVIDER_ID
    assert legacy.model_profile_id == MDP_POLAR_CANONICAL_ADT_MODEL_PROFILE_ID
    with pytest.raises(ValueError, match="no elements"):
        legacy.chart_for_geometry(geometry)

    role_separated = MDPPolarCanonicalADTResponse.from_role_separated_repository_assets(
        mdp=mdp,
        base=base,
        source_root=".",
    )
    chart = role_separated.chart_for_geometry(geometry)
    assert role_separated.contract_id == MDP_POLAR_ROLE_SEPARATED_ADT_CONTRACT
    assert (
        role_separated.provider_id == MDP_POLAR_ROLE_SEPARATED_ADT_RESPONSE_PROVIDER_ID
    )
    assert (
        role_separated.model_profile_id == MDP_POLAR_ROLE_SEPARATED_ADT_MODEL_PROFILE_ID
    )
    assert chart.adt_lift.effective_electron_counts_by_atomic_number[53] == 25.0
    assert chart.state_sha256 != legacy.configuration_sha256()


def test_role_separated_and_legacy_assets_cannot_be_mixed() -> None:
    geometry, alpha, mdp, base, _response = _problem()
    with pytest.raises(ValueError, match="cannot be combined"):
        MDPPolarCanonicalADTResponse(
            mdp=_MDP(alpha),
            base=_NonlinearResponse(geometry),
            mixtures_by_atomic_number=_mixtures(),
            source_asset_sha256="9" * 64,
            radial_shape_registry=load_repository_adt_radial_shape_registry(
                source_root="."
            ),
        )
