from __future__ import annotations

import hashlib

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.derivatives.scalar_finite_difference import (
    BOUNDED_TOPOLOGY_NOISE_EXPERIMENTAL_V1,
    FiniteDifferenceTopologyChangeError,
    RichardsonScalarForce,
    RichardsonScalarHessian,
    ScalarEnergySample,
    ScalarForceSample,
)


class _NoisyQuadratic:
    provider_id = "test.noisy-quadratic.v1"

    def __init__(
        self,
        *,
        even_energy_step_eV: float = 0.0,
        odd_energy_step_eV: float = 0.0,
        displaced_force_jump_eV_per_A: float = 0.0,
        metadata_drift: bool = False,
        configuration_drift: bool = False,
        nonfinite_force: bool = False,
        cross_force_coupling_eV_per_A: float = 0.0,
    ) -> None:
        self.even_energy_step_eV = float(even_energy_step_eV)
        self.odd_energy_step_eV = float(odd_energy_step_eV)
        self.displaced_force_jump_eV_per_A = float(displaced_force_jump_eV_per_A)
        self.metadata_drift = metadata_drift
        self.configuration_drift = configuration_drift
        self.nonfinite_force = nonfinite_force
        self.cross_force_coupling_eV_per_A = float(cross_force_coupling_eV_per_A)
        self.force_sample_count = 0
        self.last_force_sample_x: float | None = None

    def configuration_sha256(self) -> str:
        epoch = self.force_sample_count if self.configuration_drift else 0
        return hashlib.sha256(f"noisy-quadratic:{epoch}".encode()).hexdigest()

    def _values(self, geometry: object) -> tuple[float, np.ndarray]:
        positions = np.asarray(geometry.positions, dtype=float)
        x = float(positions[0, 0])
        energy = 0.5 * float(np.sum(positions**2))
        if x != 0.0:
            energy += self.even_energy_step_eV
            energy += np.sign(x) * self.odd_energy_step_eV
        forces = -positions.copy()
        if x != 0.0:
            forces[0, 0] += self.displaced_force_jump_eV_per_A
            forces[0, 1] += self.cross_force_coupling_eV_per_A * np.sign(x)
        if self.nonfinite_force and x != 0.0:
            forces[0, 0] = np.nan
        return energy, forces

    def sample(self, geometry: object) -> ScalarEnergySample:
        positions = np.asarray(geometry.positions, dtype=float)
        x = float(positions[0, 0])
        energy, _ = self._values(geometry)
        coverage = "partial" if self.metadata_drift and x > 0.0 else "complete"
        components = ("drifted-surface",) if coverage == "partial" else ()
        return ScalarEnergySample(
            energy_eV=energy,
            state_sha256=hashlib.sha256(positions.tobytes()).hexdigest(),
            topology_id="center" if x == 0.0 else ("plus" if x > 0.0 else "minus"),
            topology_observation_coverage=coverage,
            unobservable_topology_components=components,
        )

    def force_sample(self, geometry: object) -> ScalarForceSample:
        self.force_sample_count += 1
        self.last_force_sample_x = float(np.asarray(geometry.positions)[0, 0])
        energy_sample = self.sample(geometry)
        _, forces = self._values(geometry)
        evaluation = hashlib.sha256(
            energy_sample.state_sha256.encode() + forces.tobytes()
        ).hexdigest()
        return ScalarForceSample(energy_sample, forces, evaluation)


class _StencilFailure(RuntimeError):
    pass


class _RestoreFailure(RuntimeError):
    pass


class _Python310StyleStencilFailure(RuntimeError):
    # Simulate Python 3.10, where BaseException.add_note is unavailable.
    add_note = None


class _FailingNoisyQuadratic(_NoisyQuadratic):
    def __init__(
        self,
        *,
        fail_at_displaced_call: int,
        failure_type: type[BaseException] = _StencilFailure,
        fail_center_recovery: bool = False,
    ) -> None:
        super().__init__()
        self.fail_at_displaced_call = fail_at_displaced_call
        self.failure_type = failure_type
        self.fail_center_recovery = fail_center_recovery
        self.displaced_call_count = 0
        self.failure: BaseException | None = None
        self.failed = False
        self.cache_valid = False
        self.invalidation_count = 0
        self.center_call_count = 0

    def force_sample(self, geometry: object) -> ScalarForceSample:
        x = float(np.asarray(geometry.positions)[0, 0])
        self.last_force_sample_x = x
        self.cache_valid = True
        if x != 0.0:
            self.displaced_call_count += 1
            if self.displaced_call_count == self.fail_at_displaced_call:
                self.failed = True
                self.failure = self.failure_type(
                    f"stencil failure {self.fail_at_displaced_call}"
                )
                raise self.failure
        else:
            self.center_call_count += 1
            if self.failed and self.fail_center_recovery:
                raise _RestoreFailure("central recovery failed")
        return super().force_sample(geometry)

    def invalidate_cached_state(self) -> None:
        self.invalidation_count += 1
        self.cache_valid = False
        self.last_force_sample_x = None


class _TopologyRecoveryQuadratic(_NoisyQuadratic):
    def __init__(self, recovery_failure_type: type[BaseException]) -> None:
        super().__init__()
        self.recovery_failure_type = recovery_failure_type
        self.displaced_call_count = 0
        self.center_call_count = 0

    def force_sample(self, geometry: object) -> ScalarForceSample:
        x = float(np.asarray(geometry.positions)[0, 0])
        if x == 0.0:
            self.center_call_count += 1
            if self.displaced_call_count:
                raise self.recovery_failure_type("central recovery failed")
        else:
            self.displaced_call_count += 1
        return super().force_sample(geometry)


def _atoms() -> Atoms:
    return Atoms("He", positions=np.zeros((1, 3), dtype=float))


def _bounded(**overrides: object) -> RichardsonScalarHessian:
    kwargs: dict[str, object] = {
        "coarse_step_angstrom": 4.0e-3,
        "maximum_error_eV_per_A2": 5.0e-2,
        "maximum_antisymmetry_eV_per_A2": 5.0e-2,
        "topology_guard_policy": BOUNDED_TOPOLOGY_NOISE_EXPERIMENTAL_V1,
    }
    kwargs.update(overrides)
    return RichardsonScalarHessian(**kwargs)


def test_workflow_diagnostics_preserve_every_ordered_accepted_stencil():
    from maple.solvation.derivatives.molecular_modes import (
        hessian_numerical_diagnostics,
        hessian_numerical_summary,
    )

    evaluated = _bounded().evaluate(_NoisyQuadratic(), _atoms())
    record = hessian_numerical_diagnostics(evaluated)
    assert len(record["displaced_topology_ids"]) == 12
    assert record["displaced_topology_ids"] == list(evaluated.displaced_topology_ids)
    assert record["displaced_topology_changed"] == list(
        evaluated.displaced_topology_changed
    )
    assert record["energy_force_discrepancies_eV_per_A"] == list(
        evaluated.energy_force_discrepancies_eV_per_A
    )
    assert len(record["displaced_force_sha256"]) == 12
    assert record["stencil_order_per_column"] == ["+h", "-h", "+h/2", "-h/2"]
    summary = hessian_numerical_summary(evaluated)
    assert evaluated.evaluation_sha256 in summary
    assert "Richardson" in summary and "raw antisymmetry" in summary
    assert "changed samples" in summary and "endpoint work" in summary


def test_bounded_policy_accepts_and_hashes_small_changing_topology_noise():
    provider = _NoisyQuadratic(even_energy_step_eV=2.0e-6)
    direction = np.asarray([[1.0, 0.0, 0.0]])
    result = _bounded().evaluate_hvp(provider, _atoms(), direction)

    assert result.topology_guard_status == "bounded-topology-noise-experimental"
    assert result.displaced_topology_ids == ("plus", "minus", "plus", "minus")
    assert result.displaced_topology_changed == (True, True, True, True)
    assert len(result.energy_force_discrepancies_eV_per_A) == 4
    assert max(result.energy_force_discrepancies_eV_per_A) <= 3.0e-3
    assert result.evaluation_sha256
    changed_diagnostics = _bounded().evaluate_hvp(
        _NoisyQuadratic(even_energy_step_eV=3.0e-6), _atoms(), direction
    )
    assert changed_diagnostics.hvp_eV_per_A2 == pytest.approx(result.hvp_eV_per_A2)
    assert changed_diagnostics.evaluation_sha256 != result.evaluation_sha256
    with pytest.raises(AttributeError):
        result.displaced_topology_ids += ("tamper",)

    full = _bounded().evaluate(_NoisyQuadratic(even_energy_step_eV=2.0e-6), _atoms())
    assert full.raw_hessian_eV_per_A2.shape == (3, 3)
    assert len(full.displaced_force_sha256) == 12
    assert len(full.displaced_topology_ids) == 12
    assert len(full.displaced_topology_changed) == 12
    assert len(full.energy_force_discrepancies_eV_per_A) == 12


def test_strict_policy_still_rejects_the_same_topology_changes() -> None:
    provider = _NoisyQuadratic()
    with pytest.raises(FiniteDifferenceTopologyChangeError):
        RichardsonScalarHessian().evaluate_hvp(
            provider, _atoms(), np.asarray([[1.0, 0.0, 0.0]])
        )
    assert provider.last_force_sample_x == 0.0


@pytest.mark.parametrize("fail_at_displaced_call", [1, 2, 3, 4])
def test_stencil_failure_restores_center_without_replacing_original_traceback(
    fail_at_displaced_call: int,
) -> None:
    provider = _FailingNoisyQuadratic(fail_at_displaced_call=fail_at_displaced_call)
    with pytest.raises(_StencilFailure) as caught:
        _bounded().evaluate_hvp(provider, _atoms(), np.asarray([[1.0, 0.0, 0.0]]))

    assert caught.value is provider.failure
    assert str(caught.value) == f"stencil failure {fail_at_displaced_call}"
    traceback_names = []
    traceback = caught.value.__traceback__
    while traceback is not None:
        traceback_names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert "force_sample" in traceback_names
    assert provider.last_force_sample_x == 0.0
    assert provider.center_call_count == 2
    assert provider.cache_valid is True


def test_center_recovery_failure_is_only_noted_on_original_stencil_failure() -> None:
    provider = _FailingNoisyQuadratic(
        fail_at_displaced_call=2,
        fail_center_recovery=True,
    )
    with pytest.raises(_StencilFailure) as caught:
        _bounded().evaluate_hvp(provider, _atoms(), np.asarray([[1.0, 0.0, 0.0]]))

    assert caught.value is provider.failure
    assert caught.value.__notes__ == [
        "central force-provider recovery also failed: "
        "_RestoreFailure: central recovery failed"
    ]


def test_recovery_note_supports_python310_style_exceptions() -> None:
    provider = _FailingNoisyQuadratic(
        fail_at_displaced_call=1,
        failure_type=_Python310StyleStencilFailure,
        fail_center_recovery=True,
    )
    with pytest.raises(_Python310StyleStencilFailure) as caught:
        _bounded().evaluate_hvp(provider, _atoms(), np.asarray([[1.0, 0.0, 0.0]]))

    assert caught.value is provider.failure
    assert caught.value.__notes__ == [
        "central force-provider recovery also failed: "
        "_RestoreFailure: central recovery failed"
    ]


@pytest.mark.parametrize("evaluation", ["hvp", "hessian"])
def test_adaptive_topology_does_not_retry_after_center_recovery_failure(
    evaluation: str,
) -> None:
    provider = _TopologyRecoveryQuadratic(_RestoreFailure)
    backend = RichardsonScalarHessian(maximum_topology_step_reductions=3)
    with pytest.raises(FiniteDifferenceTopologyChangeError) as caught:
        if evaluation == "hvp":
            backend.evaluate_hvp(provider, _atoms(), np.asarray([[1.0, 0.0, 0.0]]))
        else:
            backend.evaluate(provider, _atoms())

    assert provider.displaced_call_count == 1
    assert provider.center_call_count == 2
    assert caught.value.__notes__ == [
        "central force-provider recovery also failed: "
        "_RestoreFailure: central recovery failed"
    ]


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_process_control_failure_during_recovery_propagates_immediately(
    failure_type: type[BaseException],
) -> None:
    provider = _TopologyRecoveryQuadratic(failure_type)
    backend = RichardsonScalarHessian(maximum_topology_step_reductions=3)
    with pytest.raises(failure_type) as caught:
        backend.evaluate_hvp(provider, _atoms(), np.asarray([[1.0, 0.0, 0.0]]))

    assert isinstance(caught.value.__context__, FiniteDifferenceTopologyChangeError)
    assert provider.displaced_call_count == 1
    assert provider.center_call_count == 2


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_process_control_failure_invalidates_without_expensive_center_replay(
    failure_type: type[BaseException],
) -> None:
    provider = _FailingNoisyQuadratic(
        fail_at_displaced_call=3,
        failure_type=failure_type,
    )
    with pytest.raises(failure_type) as caught:
        _bounded().evaluate_hvp(provider, _atoms(), np.asarray([[1.0, 0.0, 0.0]]))

    assert caught.value is provider.failure
    assert provider.center_call_count == 1
    assert provider.invalidation_count == 1
    assert provider.cache_valid is False
    assert provider.last_force_sample_x is None


@pytest.mark.parametrize(
    "provider",
    [
        _NoisyQuadratic(even_energy_step_eV=2.0e-5),
        _NoisyQuadratic(odd_energy_step_eV=2.0e-5),
        _NoisyQuadratic(displaced_force_jump_eV_per_A=2.0e-2),
    ],
)
def test_bounded_policy_rejects_inconsistent_energy_force_pairs(provider):
    with pytest.raises(RuntimeError, match="energy-force discrepancy"):
        _bounded().evaluate_hvp(provider, _atoms(), np.asarray([[1.0, 0.0, 0.0]]))
    assert provider.last_force_sample_x == 0.0


def test_bounded_policy_rejects_metadata_configuration_and_nonfinite_drift():
    direction = np.asarray([[1.0, 0.0, 0.0]])
    with pytest.raises(RuntimeError, match="observation metadata drifted"):
        _bounded().evaluate_hvp(
            _NoisyQuadratic(metadata_drift=True), _atoms(), direction
        )
    configuration_drift = _NoisyQuadratic(configuration_drift=True)
    with pytest.raises(RuntimeError, match="configuration drifted"):
        _bounded().evaluate_hvp(configuration_drift, _atoms(), direction)
    assert configuration_drift.last_force_sample_x == 0.0
    with pytest.raises(ValueError, match="forces_eV_per_A must be finite"):
        _bounded().evaluate_hvp(
            _NoisyQuadratic(nonfinite_force=True), _atoms(), direction
        )


def test_bounded_policy_keeps_raw_antisymmetry_as_a_hard_limit():
    with pytest.raises(RuntimeError, match="antisymmetry exceeds"):
        _bounded(maximum_antisymmetry_eV_per_A2=1.0e-3).evaluate(
            _NoisyQuadratic(cross_force_coupling_eV_per_A=2.0e-5), _atoms()
        )


def test_legacy_policy_and_evaluation_hashes_are_unchanged():
    # Regression anchors captured before the bounded-noise extension.
    from test_scalar_finite_difference_force import (
        _PolynomialScalar,
        _atoms as old_atoms,
    )

    backend = RichardsonScalarHessian(
        coarse_step_angstrom=1.0e-2,
        maximum_error_eV_per_A2=1.0e-3,
        maximum_antisymmetry_eV_per_A2=1.0e-10,
    )
    assert backend.policy_sha256() == (
        "703b8c33b0c0937c1175ef45fae4111e7921eba2eb12a998571041fff82658c7"
    )
    hvp = backend.evaluate_hvp(_PolynomialScalar(), old_atoms(), np.ones((2, 3)))
    assert hvp.evaluation_sha256 == (
        "38ddd1be1a6348d901869ad2e95fe6498b91903a3b3682aa4387aeeb5555c780"
    )
    hessian = backend.evaluate(_PolynomialScalar(), old_atoms())
    assert hessian.evaluation_sha256 == (
        "b284bf48e5ee2e9ad837f6ee42f895e43258366e7bcb70fad46f02261c0e58b1"
    )


def test_bounded_policy_default_and_validation_are_policy_local():
    default = _bounded()
    assert default.maximum_energy_force_discrepancy_eV_per_A == 3.0e-3
    assert (
        default.policy_sha256()
        != _bounded(maximum_energy_force_discrepancy_eV_per_A=2.0e-3).policy_sha256()
    )
    assert RichardsonScalarHessian().maximum_energy_force_discrepancy_eV_per_A is None
    with pytest.raises(ValueError, match="only valid with"):
        RichardsonScalarHessian(maximum_energy_force_discrepancy_eV_per_A=3.0e-3)
    with pytest.raises(ValueError, match="positive and finite"):
        _bounded(maximum_energy_force_discrepancy_eV_per_A=np.nan)

    with pytest.raises(ValueError, match="topology_guard_policy"):
        RichardsonScalarForce(
            topology_guard_policy=BOUNDED_TOPOLOGY_NOISE_EXPERIMENTAL_V1
        )


def test_bounded_policy_keeps_richardson_error_and_stale_center_hard_failures():
    direction = np.asarray([[1.0, 0.0, 0.0]])
    with pytest.raises(RuntimeError, match="Richardson HVP error estimate exceeds"):
        _bounded(maximum_error_eV_per_A2=1.0e-12).evaluate_hvp(
            _NoisyQuadratic(cross_force_coupling_eV_per_A=1.0e-6),
            _atoms(),
            direction,
        )

    atoms = _atoms()
    provider = _NoisyQuadratic()
    stale = provider.force_sample(atoms)
    atoms.positions[0, 0] = 0.1
    with pytest.raises(ValueError, match="central_sample did not replay"):
        _bounded().evaluate_hvp(provider, atoms, direction, central_sample=stale)
