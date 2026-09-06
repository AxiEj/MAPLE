from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

from ase import Atoms
import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
    GaussianMixtureAtom,
)
from maple.solvation.continuum import SeparatedSourceDDXBackend
from maple.solvation.coupling.separated_operators import (
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
)
from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE
from maple.solvation.experimental.mace_mdp_polar_adt_ddx import (
    CANONICAL_ADT_HYBRID_DDX_PROFILE_ID,
    CANONICAL_ADT_HYBRID_DDX_PROVIDER_ID,
    ROOT_TOLERANCE_EV,
    ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID,
    ROLE_SEPARATED_ADT_HYBRID_DDX_PROVIDER_ID,
    POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID,
    POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROVIDER_ID,
    CanonicalADTHybridDDXState,
    MACE_MDPPolarCanonicalADTDDXEnergy,
)
from maple.solvation.models.mace_mdp_polar_adt import (
    MACEPolarZeroFieldPointPermanentSource,
    MDP_POLAR_ROLE_SEPARATED_ADT_CONTRACT,
    MDPPolarCanonicalADTResponse,
)
from maple.solvation.release.uniform_response_manifold import (
    affine_uniform_native_field,
)
from maple.solvation.release.uniform_susceptibility_replacement import (
    arithmetic_uniform_gradient_left_inverse,
)


def _geometry() -> Atoms:
    atoms = Atoms(
        "HHH",
        positions=[
            [0.0, 0.0, 0.0],
            [1.43, 0.10, -0.20],
            [0.20, 1.10, 0.30],
        ],
    )
    atoms.info["charge"] = 0
    atoms.info["multiplicity"] = 1
    return atoms


class _MDPState:
    def __init__(self, alpha: np.ndarray) -> None:
        self.public_polarizability_eangstrom2_per_volt = np.asarray(alpha, dtype=float)

    @property
    def atomic_polarizabilities_eangstrom2_per_volt(self) -> np.ndarray:
        raise AssertionError("latent atomwise MDP alpha must not be read")


class _MDP:
    provider_id = "test.adt-ddx.mdp.v1"
    model_profile_id = "test.adt-ddx.mdp-profile.v1"

    def __init__(self) -> None:
        self.alpha = np.asarray(
            [
                [1.15, 0.06, 0.02],
                [0.06, 1.35, -0.03],
                [0.02, -0.03, 1.55],
            ]
        )
        self.digest = "1" * 64

    def configuration_sha256(self) -> str:
        return self.digest

    def evaluate(self, _geometry: object) -> _MDPState:
        return _MDPState(self.alpha.copy())


class _NonlinearPOLARResponse:
    provider_id = "test.adt-ddx.polar.v1"
    model_profile_id = "test.adt-ddx.polar-profile.v1"
    provenance_sha256 = "2" * 64
    source_space = ATOMIC_L1_SOURCE_SPACE
    receiver_space = MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE
    long_range_evaluator_profile = (
        "graph-longrange-analytic-gaussian-multipole-realspace-v1"
    )
    field_energy_pairing_sha256 = "3" * 64

    def __init__(self, geometry: Atoms) -> None:
        self.digest = "4" * 64
        self.zero = np.asarray(
            [
                [-0.10, 0.02, -0.01, 0.03],
                [0.04, -0.03, 0.02, -0.01],
                [0.06, 0.01, -0.02, -0.02],
            ]
        )
        count = len(geometry)
        positions = np.asarray(geometry.positions)
        uniform_basis = np.stack(
            [
                affine_uniform_native_field(positions, np.eye(3)[axis])
                for axis in range(3)
            ],
            axis=-1,
        )
        left_inverse = arithmetic_uniform_gradient_left_inverse(count)
        rng = np.random.default_rng(20260818)

        uniform_tangent = rng.normal(scale=0.04, size=(count, 4, 3))
        uniform_tangent[:, 0, :] -= np.mean(
            uniform_tangent[:, 0, :], axis=0, keepdims=True
        )
        self.jacobian = uniform_tangent.reshape(count * 4, 3) @ left_inverse

        flat_uniform_basis = uniform_basis.reshape(count * 8, 3)
        kernel = rng.normal(size=count * 8)
        kernel -= flat_uniform_basis @ (left_inverse @ kernel)
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


class _Permanent:
    provider_id = "test.adt-ddx.permanent.v1"
    model_profile_id = "test.adt-ddx.permanent-profile.v1"
    source_space = ATOMIC_L1_SOURCE_SPACE

    def __init__(self, mdp_digest: str, *, charge_shift: float = 0.0) -> None:
        self.mdp_configuration_sha256 = mdp_digest
        self.charge_shift = float(charge_shift)

    def configuration_sha256(self) -> str:
        return "5" * 64

    def evaluate_source(self, _geometry: object) -> np.ndarray:
        result = np.asarray(
            [
                [-0.12, 0.02, -0.01, 0.03],
                [0.05, -0.03, 0.02, -0.01],
                [0.07, 0.01, -0.01, -0.02],
            ]
        ).copy()
        result[0, 0] += self.charge_shift
        return result


def _response(geometry: Atoms, mdp: _MDP) -> MDPPolarCanonicalADTResponse:
    mixture = GaussianMixtureAtom(np.asarray([1.0]), np.asarray([1.2]))
    return MDPPolarCanonicalADTResponse(
        mdp=mdp,
        base=_NonlinearPOLARResponse(geometry),
        mixtures_by_atomic_number={1: mixture},
        source_asset_sha256="9" * 64,
    )


def _continuum() -> SeparatedSourceDDXBackend:
    return SeparatedSourceDDXBackend(
        ("H", "H", "H"),
        (1.5, 1.6, 1.2),
        continuum_model="pcm",
        dielectric=78.39,
        lmax=6,
        n_lebedev=110,
        solver_tolerance=1.0e-12,
    )


def _evaluator() -> tuple[Atoms, MACE_MDPPolarCanonicalADTDDXEnergy]:
    geometry = _geometry()
    mdp = _MDP()
    evaluator = MACE_MDPPolarCanonicalADTDDXEnergy(
        geometry,
        permanent=_Permanent(mdp.configuration_sha256()),
        response=_response(geometry, mdp),
        continuum=_continuum(),
    )
    return geometry, evaluator


@pytest.fixture(scope="module")
def solved_problem() -> tuple[
    Atoms,
    MACE_MDPPolarCanonicalADTDDXEnergy,
    CanonicalADTHybridDDXState,
]:
    if importlib.util.find_spec("pyddx") is None:
        pytest.skip("optional pyddx==0.8.0 runtime is unavailable")
    geometry, evaluator = _evaluator()
    return geometry, evaluator, evaluator.solve(geometry)


def test_import_is_dependency_light_without_pyddx() -> None:
    code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'pyddx' or name.startswith('pyddx.'):
        raise AssertionError('pyddx imported eagerly')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from maple.solvation.experimental.mace_mdp_polar_adt_ddx import MACE_MDPPolarCanonicalADTDDXEnergy
print(MACE_MDPPolarCanonicalADTDDXEnergy.__name__)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "MACE_MDPPolarCanonicalADTDDXEnergy"


def test_five_start_root_replay_and_operational_scalar_close(
    solved_problem: tuple[
        Atoms,
        MACE_MDPPolarCanonicalADTDDXEnergy,
        CanonicalADTHybridDDXState,
    ],
) -> None:
    geometry, evaluator, state = solved_problem
    assert evaluator.provider_id == CANONICAL_ADT_HYBRID_DDX_PROVIDER_ID
    assert evaluator.profile_id == CANONICAL_ADT_HYBRID_DDX_PROFILE_ID
    assert evaluator.capabilities.enabled_tiers == ()
    assert evaluator.force_available is False
    assert evaluator.variational_functional_admitted is False
    assert state.primal_residual_ev < ROOT_TOLERANCE_EV
    assert len(state.root_starts) == 5
    assert {item.label for item in state.root_starts} == {
        "zero",
        "permanent",
        "twice-permanent",
        "negative-permanent",
        "seeded-transverse",
    }
    assert max(item.residual_norm_ev for item in state.root_starts) < ROOT_TOLERANCE_EV
    assert (
        max(item.polarization_energy_ev for item in state.root_starts)
        - min(item.polarization_energy_ev for item in state.root_starts)
        < 2.0e-10
    )
    assert state.total_energy_ev == pytest.approx(
        state.vacuum_energy_ev + state.polarization_energy_ev, abs=0.0
    )
    target, components, continuum_state = evaluator.state_map(
        geometry, state.native_field8
    )
    assert np.linalg.norm(target - state.native_field8) < ROOT_TOLERANCE_EV
    np.testing.assert_array_equal(
        components.radial_residual_source4, state.radial_residual_source4
    )
    np.testing.assert_array_equal(
        components.adt_atomic_dipoles_eangstrom,
        state.adt_atomic_dipoles_eangstrom,
    )
    assert continuum_state.state_sha256 == state.continuum_state_sha256
    assert evaluator.solve(geometry).root_sha256 == state.root_sha256
    for values in (
        state.permanent_source4,
        state.native_field8,
        state.radial_residual_source4,
        state.adt_atomic_dipoles_eangstrom,
    ):
        assert not values.flags.writeable

    screen = evaluator.solve_zero_start_screen(geometry)
    assert screen.screen_sha256
    assert screen.iterations == state.root_starts[0].iterations
    assert screen.primal_residual_ev == state.root_starts[0].residual_norm_ev
    assert screen.polarization_energy_ev == pytest.approx(
        state.polarization_energy_ev, abs=2.0e-10
    )
    np.testing.assert_array_equal(screen.native_field8, state.native_field8)
    assert not screen.native_field8.flags.writeable


def test_state_map_jvp_vjp_dot_identity_and_finite_field(
    solved_problem: tuple[
        Atoms,
        MACE_MDPPolarCanonicalADTDDXEnergy,
        CanonicalADTHybridDDXState,
    ],
) -> None:
    geometry, evaluator, state = solved_problem
    rng = np.random.default_rng(77)
    direction = rng.normal(size=state.native_field8.shape)
    cotangent = rng.normal(size=state.native_field8.shape)
    jvp = evaluator.state_map_jvp(geometry, state.native_field8, direction)
    vjp = evaluator.state_map_vjp(geometry, state.native_field8, cotangent)
    assert float(np.vdot(cotangent, jvp)) == pytest.approx(
        float(np.vdot(vjp, direction)), abs=5.0e-12
    )

    errors = []
    for step in (4.0e-3, 2.0e-3, 1.0e-3):
        plus = evaluator.state_map(geometry, state.native_field8 + step * direction)[0]
        minus = evaluator.state_map(geometry, state.native_field8 - step * direction)[0]
        errors.append(float(np.max(np.abs((plus - minus) / (2.0 * step) - jvp))))
    # The synthetic source is quadratic, so the centred derivative is exact;
    # the remaining plateau is the independently repeated ddX linear solve.
    assert max(errors) < 2.0e-10


def test_uniform_passivity_and_local_root_certificate(
    solved_problem: tuple[
        Atoms,
        MACE_MDPPolarCanonicalADTDDXEnergy,
        CanonicalADTHybridDDXState,
    ],
) -> None:
    geometry, evaluator, state = solved_problem
    eigenvalues = evaluator.molecular_response_passivity_eigenvalues(geometry)
    assert eigenvalues.shape == (3,)
    assert np.min(eigenvalues) > 0.0
    assert not eigenvalues.flags.writeable

    certificate = evaluator.local_root_certificate(geometry, state)
    assert certificate.dimension == 24
    assert certificate.local_implicit_branch_certified is True
    assert certificate.contraction_at_state is True
    assert certificate.strong_monotonicity_at_state is True
    assert certificate.global_unique_root_certified is False
    assert certificate.domain_uniform_bound_certified is False
    assert certificate.residual_jacobian_sigma_min > 0.3
    assert certificate.weighted_contraction_norm < 0.8


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_identity_geometry_and_coordinate_derivative_fail_closed() -> None:
    geometry = _geometry()
    mdp = _MDP()
    response = _response(geometry, mdp)
    continuum = _continuum()
    with pytest.raises(ValueError, match="same frozen MDP"):
        MACE_MDPPolarCanonicalADTDDXEnergy(
            geometry,
            permanent=_Permanent("6" * 64),
            response=response,
            continuum=continuum,
        )
    with pytest.raises(ValueError, match="neutral sources"):
        MACE_MDPPolarCanonicalADTDDXEnergy(
            geometry,
            permanent=_Permanent(mdp.configuration_sha256(), charge_shift=0.1),
            response=response,
            continuum=continuum,
        )

    evaluator = MACE_MDPPolarCanonicalADTDDXEnergy(
        geometry,
        permanent=_Permanent(mdp.configuration_sha256()),
        response=response,
        continuum=continuum,
    )
    moved = geometry.copy()
    moved.positions[0, 0] += 1.0e-4
    with pytest.raises(ValueError, match="different geometry"):
        evaluator.solve(moved)
    with pytest.raises(NotImplementedError, match="complete coordinate VJP"):
        evaluator.coordinate_vjp(geometry)
    with pytest.raises(AttributeError, match="immutable"):
        evaluator._permanent_source4 = np.zeros((3, 4))


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_role_separated_response_gets_distinct_ddx_profile_identity() -> None:
    geometry = _geometry()
    mdp = _MDP()
    response = MDPPolarCanonicalADTResponse.from_role_separated_repository_assets(
        mdp=mdp,
        base=_NonlinearPOLARResponse(geometry),
        source_root=".",
    )
    evaluator = MACE_MDPPolarCanonicalADTDDXEnergy(
        geometry,
        permanent=_Permanent(mdp.configuration_sha256()),
        response=response,
        continuum=_continuum(),
    )
    assert response.contract_id == MDP_POLAR_ROLE_SEPARATED_ADT_CONTRACT
    assert evaluator.provider_id == ROLE_SEPARATED_ADT_HYBRID_DDX_PROVIDER_ID
    assert evaluator.profile_id == ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID
    assert evaluator.solve(geometry).primal_residual_ev < ROOT_TOLERANCE_EV


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_polar_zero_point_permanent_gets_distinct_v3_profile() -> None:
    geometry = _geometry()
    mdp = _MDP()
    response = MDPPolarCanonicalADTResponse.from_role_separated_repository_assets(
        mdp=mdp,
        base=_NonlinearPOLARResponse(geometry),
        source_root=".",
    )
    permanent = MACEPolarZeroFieldPointPermanentSource(response)
    evaluator = MACE_MDPPolarCanonicalADTDDXEnergy(
        geometry,
        permanent=permanent,
        response=response,
        continuum=_continuum(),
    )

    assert evaluator.provider_id == POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROVIDER_ID
    assert evaluator.profile_id == POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID
    np.testing.assert_array_equal(
        evaluator.permanent_source4,
        response.chart_for_geometry(geometry).polar_zero_source4,
    )
    assert evaluator.solve(geometry).primal_residual_ev < ROOT_TOLERANCE_EV


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_polar_zero_point_permanent_rejects_different_response_identity() -> None:
    geometry = _geometry()
    mdp = _MDP()
    first = MDPPolarCanonicalADTResponse.from_role_separated_repository_assets(
        mdp=mdp,
        base=_NonlinearPOLARResponse(geometry),
        source_root=".",
    )
    changed_base = _NonlinearPOLARResponse(geometry)
    changed_base.digest = "7" * 64
    second = MDPPolarCanonicalADTResponse.from_role_separated_repository_assets(
        mdp=mdp,
        base=changed_base,
        source_root=".",
    )
    with pytest.raises(ValueError, match="same frozen response"):
        MACE_MDPPolarCanonicalADTDDXEnergy(
            geometry,
            permanent=MACEPolarZeroFieldPointPermanentSource(first),
            response=second,
            continuum=_continuum(),
        )
