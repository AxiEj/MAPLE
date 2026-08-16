from __future__ import annotations

import importlib.util

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.continuum import (
    SeparatedSourceDDXBackend,
    embed_atomic_l1_in_first_radial_channel,
)
from maple.solvation.coupling.separated_operators import (
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
)
from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE
from maple.solvation.experimental.mace_mdp_polar_ddx import (
    HYBRID_DDX_PES_PROVIDER_ID,
    HYBRID_DDX_SCALAR_PROVIDER_ID,
    HybridDDXEnergyState,
    MACE_MDPPolarHybridDDXEnergy,
    MACE_MDPPolarHybridDDXPES,
    ROOT_TOLERANCE_EV,
)
from maple.solvation.derivatives import (
    RichardsonScalarForce,
    RichardsonScalarHessian,
)
from maple.solvation.models.mace_mdp_polar_hybrid import (
    PermanentAnchoredInducedSourceModel,
)


class _Permanent:
    provider_id = "test.ddx-hybrid.permanent.v1"
    model_profile_id = "test.ddx-hybrid.permanent-profile.v1"
    source_space = ATOMIC_L1_SOURCE_SPACE

    def configuration_sha256(self) -> str:
        return "1" * 64

    def evaluate_source(self, _geometry: object) -> np.ndarray:
        return np.asarray([[-0.15, 0.01, -0.02, 0.03], [0.15, -0.02, 0.01, -0.01]])

    def source_position_vjp(
        self, geometry: object, _source_cotangent: object
    ) -> np.ndarray:
        return np.zeros((len(geometry), 3), dtype=float)


class _Responsive:
    provider_id = "test.ddx-hybrid.response.v1"
    model_profile_id = "test.ddx-hybrid.response-profile.v1"
    provenance_sha256 = "2" * 64
    source_space = ATOMIC_L1_SOURCE_SPACE
    receiver_space = MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE
    long_range_evaluator_profile = (
        "graph-longrange-analytic-gaussian-multipole-realspace-v1"
    )

    def __init__(self) -> None:
        rng = np.random.default_rng(44)
        self.jacobian = rng.normal(scale=2.0e-7, size=(8, 16))
        self.jacobian[4] = -self.jacobian[0]
        self.zero = np.asarray([[-0.08, 0.01, 0.02, -0.01], [0.08, -0.01, 0.01, 0.02]])

    def configuration_sha256(self) -> str:
        return "3" * 64

    def vacuum_energy_ev(self, _geometry: object) -> float:
        return -123.456

    def vacuum_forces_ev_per_angstrom(self, geometry: object) -> np.ndarray:
        return np.zeros((len(geometry), 3), dtype=float)

    def evaluate_source(self, _geometry: object, field: object) -> np.ndarray:
        return self.zero + (self.jacobian @ np.asarray(field).reshape(-1)).reshape(2, 4)

    def field_jvp(
        self, _geometry: object, _field: object, direction: object
    ) -> np.ndarray:
        return (self.jacobian @ np.asarray(direction).reshape(-1)).reshape(2, 4)

    def field_vjp(
        self, _geometry: object, _field: object, cotangent: object
    ) -> np.ndarray:
        return (self.jacobian.T @ np.asarray(cotangent).reshape(-1)).reshape(2, 8)

    def coordinate_vjp(
        self, geometry: object, _field: object, _source_cotangent: object
    ) -> np.ndarray:
        return np.zeros((len(geometry), 3), dtype=float)


def _atoms() -> Atoms:
    atoms = Atoms("CO", positions=[[-0.5, 0.1, 0.0], [0.7, -0.1, 0.2]])
    atoms.info["charge"] = 0
    atoms.info["multiplicity"] = 1
    return atoms


def _hybrid() -> PermanentAnchoredInducedSourceModel:
    return PermanentAnchoredInducedSourceModel(_Permanent(), _Responsive())


def _continuum() -> SeparatedSourceDDXBackend:
    return SeparatedSourceDDXBackend(
        ("C", "O"),
        (1.8, 1.7),
        continuum_model="pcm",
        dielectric=78.39,
        lmax=8,
        n_lebedev=194,
        solver_tolerance=1.0e-12,
    )


def _pes(
    *,
    cavity_radii_angstrom: tuple[float, float] = (1.8, 1.7),
) -> MACE_MDPPolarHybridDDXPES:
    return MACE_MDPPolarHybridDDXPES(
        hybrid=_hybrid(),
        symbols=("C", "O"),
        cavity_radii_angstrom=cavity_radii_angstrom,
        continuum_model="pcm",
        dielectric=78.39,
        lmax=8,
        n_lebedev=194,
        solver_tolerance=1.0e-12,
        force_backend=RichardsonScalarForce(
            coarse_step_angstrom=2.0e-3,
            maximum_error_eV_per_A=2.0e-3,
        ),
        hessian_backend=RichardsonScalarHessian(
            coarse_step_angstrom=2.0e-3,
            maximum_error_eV_per_A2=1.0e-4,
            maximum_antisymmetry_eV_per_A2=1.0e-7,
        ),
    )


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_hybrid_ddx_root_and_operational_scalar_close():
    atoms = _atoms()
    hybrid = _hybrid()
    continuum = _continuum()
    evaluator = MACE_MDPPolarHybridDDXEnergy(atoms, hybrid=hybrid, continuum=continuum)
    state = evaluator.solve(atoms)
    assert evaluator.provider_id == HYBRID_DDX_SCALAR_PROVIDER_ID
    assert isinstance(state, HybridDDXEnergyState)
    assert state.primal_residual_ev < ROOT_TOLERANCE_EV
    assert state.replay_field_max_abs_difference_ev < 2.0e-9
    assert state.replay_energy_abs_difference_ev < 1.0e-10
    assert np.sum(state.induced_source4[:, 0]) == pytest.approx(0.0, abs=1.0e-12)
    assert np.sum(state.total_source4[:, 0]) == pytest.approx(0.0, abs=1.0e-12)
    assert state.total_energy_ev == pytest.approx(
        state.vacuum_energy_ev + state.polarization_energy_ev, abs=0.0
    )
    assert not state.native_field_ev.flags.writeable
    assert state.root_sha256 == evaluator.solve(atoms).root_sha256

    direct = continuum.prepare(atoms, evaluator.anchor.permanent_source4).solve(
        embed_atomic_l1_in_first_radial_channel(state.induced_source4)
    )
    assert direct.polarization_energy_ev == pytest.approx(
        state.polarization_energy_ev, abs=2.0e-12
    )


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_hybrid_ddx_evaluator_is_geometry_bound_and_immutable():
    atoms = _atoms()
    evaluator = MACE_MDPPolarHybridDDXEnergy(
        atoms, hybrid=_hybrid(), continuum=_continuum()
    )
    moved = atoms.copy()
    moved.positions[0, 0] += 1.0e-4
    with pytest.raises(ValueError, match="different geometry"):
        evaluator.solve(moved)
    with pytest.raises(AttributeError, match="immutable"):
        evaluator._prepared = object()


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_hybrid_ddx_pes_force_component_is_same_scalar_richardson_gradient():
    atoms = _atoms()
    pes = _pes()
    center = pes.sample(atoms)
    component = pes.numerical_force_component(
        atoms,
        atom_index=0,
        axis_index=0,
    )

    assert pes.provider_id == HYBRID_DDX_PES_PROVIDER_ID
    assert np.isfinite(center.energy_eV)
    assert np.isfinite(component.force_eV_per_A)
    assert component.error_estimate_eV_per_A < 2.0e-3
    assert len(set(component.displaced_state_sha256)) == 4
    with pytest.raises(AttributeError, match="immutable"):
        pes._symbols = ("O", "C")


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_hybrid_ddx_block_adjoint_force_matches_same_scalar_richardson():
    atoms = _atoms()
    pes = _pes()
    state = pes.solve(atoms)
    analytic = pes.evaluate_forces(atoms, central_state=state)
    numerical = pes.numerical_force(atoms)
    np.testing.assert_allclose(
        analytic.total_forces_ev_per_angstrom,
        numerical.forces_eV_per_A,
        atol=3.0e-7,
        rtol=0.0,
    )
    assert analytic.adjoint_residual_ev < 1.0e-9
    assert not analytic.total_forces_ev_per_angstrom.flags.writeable


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_hybrid_ddx_molecular_virial_is_same_scalar_deformation_derivative():
    atoms = _atoms()
    pes = _pes()
    force = pes.evaluate_forces(atoms)
    virial = pes.molecular_virial(atoms, force_evaluation=force)
    strain = np.asarray([[0.21, -0.07, 0.03], [0.02, -0.12, 0.05], [-0.01, 0.04, 0.08]])
    step = 1.0e-5
    energies = []
    for sign in (1.0, -1.0):
        displaced = atoms.copy()
        transform = np.eye(3) + sign * step * strain
        displaced.positions = (
            virial.origin_angstrom
            + (atoms.positions - virial.origin_angstrom) @ transform.T
        )
        energies.append(pes.get_potential_energy(displaced))
    finite_difference = (energies[0] - energies[1]) / (2.0 * step)
    assert float(np.vdot(virial.strain_gradient_eV, strain)) == pytest.approx(
        finite_difference,
        abs=2.0e-6,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_hybrid_ddx_molecular_virial_rejects_tampered_force_cache():
    atoms = _atoms()
    pes = _pes()
    force = pes.evaluate_forces(atoms)
    tampered_vacuum = np.array(force.vacuum_forces_ev_per_angstrom, copy=True)
    tampered_vacuum[0, 0] += 0.4
    tampered = type(force)(
        central_state=force.central_state,
        vacuum_forces_ev_per_angstrom=tampered_vacuum,
        continuum_fixed_source_forces_ev_per_angstrom=(
            force.continuum_fixed_source_forces_ev_per_angstrom
        ),
        permanent_source_forces_ev_per_angstrom=(
            force.permanent_source_forces_ev_per_angstrom
        ),
        induced_source_forces_ev_per_angstrom=(
            force.induced_source_forces_ev_per_angstrom
        ),
        model_field_geometry_forces_ev_per_angstrom=(
            force.model_field_geometry_forces_ev_per_angstrom
        ),
        adjoint_field_ev=force.adjoint_field_ev,
        adjoint_residual_ev=force.adjoint_residual_ev,
        adjoint_iterations=force.adjoint_iterations,
    )

    with pytest.raises(ValueError, match="did not replay"):
        pes.molecular_virial(atoms, force_evaluation=tampered)


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_hybrid_ddx_hvp_and_hessian_follow_the_same_block_adjoint_force():
    atoms = _atoms()
    pes = _pes()
    center = pes.evaluate_forces(atoms)
    direction = np.linspace(-0.3, 0.4, 6).reshape(2, 3)

    force_sample = pes.force_sample(atoms)
    assert force_sample.evaluation_sha256 == center.evaluation_sha256
    assert force_sample.energy_sample.state_sha256 == center.central_state.root_sha256

    hvp = pes.hessian_vector_product(
        atoms,
        direction,
        central_force=center,
    )
    step = 2.5e-4
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions += step * direction
    minus.positions -= step * direction
    finite_difference = -(pes.get_forces(plus) - pes.get_forces(minus)) / (2.0 * step)
    np.testing.assert_allclose(
        hvp.hvp_eV_per_A2,
        finite_difference,
        atol=2.0e-4,
        rtol=0.0,
    )

    hessian = pes.evaluate_hessian(atoms, central_force=center)
    np.testing.assert_allclose(
        hessian.hessian_eV_per_A2,
        hessian.hessian_eV_per_A2.T,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        hessian.hessian_eV_per_A2 @ direction.reshape(-1),
        hvp.hvp_eV_per_A2.reshape(-1),
        atol=3.0e-4,
        rtol=0.0,
    )
    assert hvp.maximum_error_estimate_eV_per_A2 < 1.0e-4
    assert hessian.maximum_error_estimate_eV_per_A2 < 1.0e-4
    assert hessian.maximum_antisymmetry_eV_per_A2 < 1.0e-7
    assert hessian.topology_observation_coverage == "complete"
    assert hessian.topology_guard_status == "complete"
    assert hessian.derivative_policy_sha256 == pes.hessian_backend.policy_sha256()

    other_pes = _pes(cavity_radii_angstrom=(1.9, 1.7))
    with pytest.raises(ValueError, match="different evaluator"):
        other_pes.hessian_vector_product(
            atoms,
            direction,
            central_force=center,
        )
