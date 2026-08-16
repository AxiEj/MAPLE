from __future__ import annotations

import hashlib

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.derivatives import (
    OBSERVED_COMPONENTS_ONLY_EXPERIMENTAL_V1,
    REQUIRE_COMPLETE_TOPOLOGY_OBSERVATION_V1,
    FiniteDifferenceTopologyObservationError,
    RichardsonScalarForce,
    RichardsonScalarHessian,
    ScalarEnergySample,
    ScalarForceSample,
)


class _PolynomialScalar:
    provider_id = "test.scalar-polynomial.v1"

    def __init__(
        self,
        *,
        topology_switch: bool = False,
        topology_threshold_angstrom: float = 0.0,
        topology_observation_coverage: str = "complete",
        unobservable_topology_components: tuple[str, ...] = (),
        observation_metadata_switch: bool = False,
        scale: float = 1.0,
    ) -> None:
        self.topology_switch = topology_switch
        self.topology_threshold_angstrom = float(topology_threshold_angstrom)
        self.topology_observation_coverage = topology_observation_coverage
        self.unobservable_topology_components = unobservable_topology_components
        self.observation_metadata_switch = observation_metadata_switch
        self.scale = float(scale)
        self.sample_count = 0

    def configuration_sha256(self) -> str:
        return hashlib.sha256(
            (
                f"{self.topology_switch}:"
                f"{self.topology_threshold_angstrom:.17g}"
                f":{self.topology_observation_coverage}:"
                f"{self.unobservable_topology_components}:"
                f"{self.observation_metadata_switch}:"
                f"{self.scale:.17g}"
            ).encode()
        ).hexdigest()

    def energy(self, positions: np.ndarray) -> float:
        weights = np.arange(1, positions.size + 1, dtype=float).reshape(positions.shape)
        return self.scale * float(np.sum(weights * (positions**2 + 0.2 * positions**6)))

    def sample(self, geometry: object) -> ScalarEnergySample:
        self.sample_count += 1
        positions = np.asarray(geometry.positions, dtype=float)
        state = hashlib.sha256(positions.tobytes()).hexdigest()
        topology = (
            "positive-x"
            if self.topology_switch
            and positions[0, 0] > self.topology_threshold_angstrom
            else "fixed"
        )
        coverage = self.topology_observation_coverage
        components = self.unobservable_topology_components
        if self.observation_metadata_switch and positions[0, 0] > 0.0:
            coverage = "partial"
            components = ("switched-hidden-surface",)
        return ScalarEnergySample(
            energy_eV=self.energy(positions),
            state_sha256=state,
            topology_id=topology,
            topology_observation_coverage=coverage,
            unobservable_topology_components=components,
        )

    def force_sample(self, geometry: object) -> ScalarForceSample:
        positions = np.asarray(geometry.positions, dtype=float)
        weights = np.arange(1, positions.size + 1, dtype=float).reshape(positions.shape)
        forces = -self.scale * weights * (2.0 * positions + 1.2 * positions**5)
        energy_sample = self.sample(geometry)
        evaluation = hashlib.sha256(
            energy_sample.state_sha256.encode() + forces.tobytes()
        ).hexdigest()
        return ScalarForceSample(energy_sample, forces, evaluation)


def _atoms() -> Atoms:
    return Atoms(
        "HeNe",
        positions=np.asarray([[-0.4, 0.2, 0.1], [0.8, -0.3, 0.5]], dtype=float),
    )


def test_richardson_force_is_fourth_order_scalar_gradient():
    atoms = _atoms()
    provider = _PolynomialScalar()
    backend = RichardsonScalarForce(
        coarse_step_angstrom=1.0e-2,
        maximum_error_eV_per_A=1.0e-3,
    )
    result = backend.evaluate(provider, atoms)

    positions = np.asarray(atoms.positions)
    weights = np.arange(1, positions.size + 1, dtype=float).reshape(positions.shape)
    expected = -weights * (2.0 * positions + 1.2 * positions**5)
    assert result.forces_eV_per_A == pytest.approx(expected, abs=2.0e-8)
    assert result.maximum_error_estimate_eV_per_A < 1.0e-3
    assert len(result.displaced_state_sha256) == 4 * positions.size
    assert result.evaluation_sha256
    with pytest.raises(ValueError):
        result.forces_eV_per_A.setflags(write=True)


def test_richardson_force_fails_closed_on_topology_change():
    atoms = _atoms()
    atoms.positions[0, 0] = 0.0
    with pytest.raises(RuntimeError, match="changed the continuum topology"):
        RichardsonScalarForce().evaluate(_PolynomialScalar(topology_switch=True), atoms)


def test_focused_component_uses_the_same_stencil_contract():
    atoms = _atoms()
    backend = RichardsonScalarForce(
        coarse_step_angstrom=1.0e-2,
        maximum_error_eV_per_A=1.0e-3,
    )
    component = backend.evaluate_component(
        _PolynomialScalar(), atoms, atom_index=1, axis_index=2
    )
    position = atoms.positions[1, 2]
    expected = -6.0 * (2.0 * position + 1.2 * position**5)
    assert component.force_eV_per_A == pytest.approx(expected, abs=2.0e-8)
    assert component.error_estimate_eV_per_A < 1.0e-3
    assert len(component.displaced_state_sha256) == 4
    assert component.derivative_policy_sha256 == backend.policy_sha256()
    assert component.evaluation_sha256

    looser = RichardsonScalarForce(
        coarse_step_angstrom=1.0e-2,
        maximum_error_eV_per_A=2.0e-3,
    ).evaluate_component(_PolynomialScalar(), atoms, atom_index=1, axis_index=2)
    assert looser.force_eV_per_A == component.force_eV_per_A
    assert looser.derivative_policy_sha256 != component.derivative_policy_sha256
    assert looser.evaluation_sha256 != component.evaluation_sha256


def test_richardson_force_fails_closed_when_error_budget_is_too_small():
    with pytest.raises(RuntimeError, match="error estimate exceeds"):
        RichardsonScalarForce(
            coarse_step_angstrom=0.1,
            maximum_error_eV_per_A=1.0e-12,
        ).evaluate(_PolynomialScalar(), _atoms())


def test_richardson_hvp_and_hessian_differentiate_the_same_scalar_force():
    atoms = _atoms()
    provider = _PolynomialScalar()
    backend = RichardsonScalarHessian(
        coarse_step_angstrom=1.0e-2,
        maximum_error_eV_per_A2=1.0e-3,
        maximum_antisymmetry_eV_per_A2=1.0e-10,
    )
    positions = np.asarray(atoms.positions)
    weights = np.arange(1, positions.size + 1, dtype=float).reshape(positions.shape)
    expected_diagonal = (weights * (2.0 + 6.0 * positions**4)).reshape(-1)
    direction = np.linspace(-0.4, 0.6, positions.size).reshape(positions.shape)
    hvp = backend.evaluate_hvp(provider, atoms, direction)
    np.testing.assert_allclose(
        hvp.hvp_eV_per_A2,
        expected_diagonal.reshape(positions.shape) * direction,
        atol=3.0e-8,
        rtol=0.0,
    )
    assert hvp.maximum_error_estimate_eV_per_A2 < 1.0e-3

    evaluated = backend.evaluate(provider, atoms)
    expected = np.diag(expected_diagonal)
    np.testing.assert_allclose(evaluated.raw_hessian_eV_per_A2, expected, atol=3.0e-8)
    np.testing.assert_allclose(evaluated.hessian_eV_per_A2, expected, atol=3.0e-8)
    assert evaluated.maximum_antisymmetry_eV_per_A2 < 1.0e-12
    assert evaluated.maximum_error_estimate_eV_per_A2 < 1.0e-3
    assert evaluated.evaluation_sha256
    with pytest.raises(ValueError):
        evaluated.hessian_eV_per_A2.setflags(write=True)


def test_richardson_hessian_fails_closed_on_topology_change():
    atoms = _atoms()
    atoms.positions[0, 0] = 0.0
    direction = np.zeros((2, 3))
    direction[0, 0] = 1.0
    with pytest.raises(RuntimeError, match="changed the continuum topology"):
        RichardsonScalarHessian().evaluate_hvp(
            _PolynomialScalar(topology_switch=True), atoms, direction
        )


def test_richardson_hessian_reduces_only_until_one_topology_is_shared():
    atoms = _atoms()
    atoms.positions[0, 0] = 0.0
    direction = np.zeros((2, 3))
    direction[0, 0] = 1.0
    provider = _PolynomialScalar(
        topology_switch=True,
        topology_threshold_angstrom=7.5e-4,
    )
    adaptive = RichardsonScalarHessian(
        coarse_step_angstrom=1.0e-3,
        maximum_error_eV_per_A2=1.0e-3,
        maximum_antisymmetry_eV_per_A2=1.0e-10,
        maximum_topology_step_reductions=2,
    )
    evaluated = adaptive.evaluate_hvp(provider, atoms, direction)
    assert evaluated.coarse_step_angstrom == pytest.approx(5.0e-4)
    assert evaluated.fine_step_angstrom == pytest.approx(2.5e-4)
    assert evaluated.topology_step_reductions_used == 1
    assert evaluated.topology_guard_status == "complete"

    hessian = adaptive.evaluate(provider, atoms)
    assert hessian.coarse_step_angstrom == pytest.approx(5.0e-4)
    assert hessian.fine_step_angstrom == pytest.approx(2.5e-4)
    assert hessian.topology_step_reductions_used == 1
    assert hessian.topology_guard_status == "complete"


def test_strict_topology_policy_rejects_partial_observation_before_stencil():
    atoms = _atoms()
    provider = _PolynomialScalar(
        topology_observation_coverage="partial",
        unobservable_topology_components=("hidden-solvent-surface",),
    )
    with pytest.raises(
        FiniteDifferenceTopologyObservationError,
        match="requires complete topology observation",
    ):
        RichardsonScalarForce().evaluate(provider, atoms)
    assert provider.sample_count == 1

    direction = np.ones((2, 3))
    with pytest.raises(
        FiniteDifferenceTopologyObservationError,
        match="requires complete topology observation",
    ):
        RichardsonScalarHessian().evaluate_hvp(provider, atoms, direction)
    assert provider.sample_count == 2


def test_experimental_partial_topology_policy_records_its_scientific_boundary():
    atoms = _atoms()
    provider = _PolynomialScalar(
        topology_observation_coverage="partial",
        unobservable_topology_components=("hidden-solvent-surface",),
    )
    force_backend = RichardsonScalarForce(
        topology_guard_policy=OBSERVED_COMPONENTS_ONLY_EXPERIMENTAL_V1,
    )
    force = force_backend.evaluate(provider, atoms)
    assert force.topology_observation_coverage == "partial"
    assert force.unobservable_topology_components == ("hidden-solvent-surface",)
    assert force.topology_guard_status == "partial-experimental"
    assert force.derivative_policy_sha256 == force_backend.policy_sha256()

    hessian_backend = RichardsonScalarHessian(
        coarse_step_angstrom=1.0e-2,
        maximum_error_eV_per_A2=1.0e-3,
        maximum_antisymmetry_eV_per_A2=1.0e-10,
        topology_guard_policy=OBSERVED_COMPONENTS_ONLY_EXPERIMENTAL_V1,
    )
    hvp = hessian_backend.evaluate_hvp(provider, atoms, np.ones((2, 3)))
    assert hvp.topology_observation_coverage == "partial"
    assert hvp.unobservable_topology_components == ("hidden-solvent-surface",)
    assert hvp.topology_guard_status == "partial-experimental"
    assert hvp.derivative_policy_sha256 == hessian_backend.policy_sha256()


def test_legacy_unspecified_topology_is_unobservable_and_requires_opt_in():
    sample = ScalarEnergySample(
        energy_eV=0.0,
        state_sha256="0" * 64,
        topology_id="legacy-sentinel",
    )
    assert sample.topology_observation_coverage == "unobservable"
    assert sample.unobservable_topology_components == (
        "legacy-unspecified-topology-observation",
    )

    atoms = _atoms()
    provider = _PolynomialScalar(
        topology_observation_coverage="unobservable",
        unobservable_topology_components=("hidden-surface",),
    )
    with pytest.raises(FiniteDifferenceTopologyObservationError):
        RichardsonScalarHessian().evaluate_hvp(provider, atoms, np.ones((2, 3)))
    experimental = RichardsonScalarHessian(
        coarse_step_angstrom=1.0e-2,
        maximum_error_eV_per_A2=1.0e-3,
        topology_guard_policy=OBSERVED_COMPONENTS_ONLY_EXPERIMENTAL_V1,
    ).evaluate_hvp(provider, atoms, np.ones((2, 3)))
    assert experimental.topology_guard_status == "unobservable-experimental"


def test_topology_observation_metadata_drift_is_not_an_adaptive_retry():
    atoms = _atoms()
    atoms.positions[0, 0] = 0.0
    provider = _PolynomialScalar(observation_metadata_switch=True)
    adaptive = RichardsonScalarHessian(maximum_topology_step_reductions=3)
    with pytest.raises(RuntimeError, match="observation metadata drifted"):
        adaptive.evaluate_hvp(provider, atoms, np.ones((2, 3)))
    assert provider.sample_count == 2


def test_derivative_policy_hash_binds_requested_step_thresholds_and_retry_history():
    atoms = _atoms()
    atoms.positions[0, 0] = 0.0
    direction = np.zeros((2, 3))
    direction[0, 0] = 1.0
    provider = _PolynomialScalar(
        topology_switch=True,
        topology_threshold_angstrom=7.5e-4,
    )
    adaptive = RichardsonScalarHessian(
        coarse_step_angstrom=1.0e-3,
        maximum_error_eV_per_A2=1.0e-3,
        maximum_antisymmetry_eV_per_A2=1.0e-10,
        maximum_topology_step_reductions=1,
    )
    direct = RichardsonScalarHessian(
        coarse_step_angstrom=5.0e-4,
        maximum_error_eV_per_A2=1.0e-3,
        maximum_antisymmetry_eV_per_A2=1.0e-10,
    )
    adaptive_result = adaptive.evaluate_hvp(provider, atoms, direction)
    direct_result = direct.evaluate_hvp(provider, atoms, direction)
    assert adaptive_result.coarse_step_angstrom == direct_result.coarse_step_angstrom
    np.testing.assert_array_equal(
        adaptive_result.hvp_eV_per_A2, direct_result.hvp_eV_per_A2
    )
    assert (
        adaptive_result.derivative_policy_sha256
        != direct_result.derivative_policy_sha256
    )
    assert adaptive_result.evaluation_sha256 != direct_result.evaluation_sha256

    looser = RichardsonScalarHessian(
        coarse_step_angstrom=5.0e-4,
        maximum_error_eV_per_A2=2.0e-3,
        maximum_antisymmetry_eV_per_A2=1.0e-10,
    )
    looser_result = looser.evaluate_hvp(provider, atoms, direction)
    assert (
        looser_result.derivative_policy_sha256 != direct_result.derivative_policy_sha256
    )
    assert looser_result.evaluation_sha256 != direct_result.evaluation_sha256


def test_topology_policy_names_are_versioned_and_validated():
    assert REQUIRE_COMPLETE_TOPOLOGY_OBSERVATION_V1.endswith("-v1")
    assert OBSERVED_COMPONENTS_ONLY_EXPERIMENTAL_V1.endswith("-v1")
    with pytest.raises(ValueError, match="topology_guard_policy"):
        RichardsonScalarForce(topology_guard_policy="allow")
    with pytest.raises(ValueError, match="topology_guard_policy"):
        RichardsonScalarHessian(topology_guard_policy="allow")


def test_cached_generic_center_must_replay_for_current_provider_and_geometry():
    atoms = _atoms()
    provider_a = _PolynomialScalar(scale=1.0)
    provider_b = _PolynomialScalar(scale=3.0)
    stale_energy = provider_a.sample(atoms)
    stale_force = provider_a.force_sample(atoms)

    force_backend = RichardsonScalarForce(
        coarse_step_angstrom=1.0e-2,
        maximum_error_eV_per_A=1.0e-2,
    )
    with pytest.raises(ValueError, match="central_sample did not replay"):
        force_backend.evaluate(
            provider_b,
            atoms,
            central_sample=stale_energy,
        )
    with pytest.raises(ValueError, match="central_sample did not replay"):
        force_backend.evaluate_component(
            provider_b,
            atoms,
            atom_index=0,
            axis_index=0,
            central_sample=stale_energy,
        )

    hessian_backend = RichardsonScalarHessian(
        coarse_step_angstrom=1.0e-2,
        maximum_error_eV_per_A2=1.0e-2,
    )
    with pytest.raises(ValueError, match="central_sample did not replay"):
        hessian_backend.evaluate_hvp(
            provider_b,
            atoms,
            np.ones((2, 3)),
            central_sample=stale_force,
        )
    with pytest.raises(ValueError, match="central_sample did not replay"):
        hessian_backend.evaluate(
            provider_b,
            atoms,
            central_sample=stale_force,
        )

    displaced = atoms.copy()
    displaced.positions[0, 0] += 0.1
    with pytest.raises(ValueError, match="central_sample did not replay"):
        force_backend.evaluate(
            provider_a,
            displaced,
            central_sample=stale_energy,
        )


@pytest.mark.parametrize("reductions", [-1, 21, True, 1.0])
def test_richardson_hessian_rejects_invalid_topology_reduction_budget(reductions):
    with pytest.raises(ValueError, match="maximum_topology_step_reductions"):
        RichardsonScalarHessian(maximum_topology_step_reductions=reductions)
