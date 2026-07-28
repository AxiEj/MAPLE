from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest
from ase import Atoms
from ase.units import Hartree

import maple.function.calculator.extra_correction.implicit.route2_engine as route2_engine_module
from maple.function.calculator.extra_correction.implicit.electrostatic_pairing import (
    MACE_POLAR_L1_PAIRING,
)
from maple.function.calculator.extra_correction.implicit.route2_engine import (
    Route2ContinuumEngine,
    Route2EngineSettings,
    Route2FiniteResolutionPolicy,
    Route2SCFConvergenceError,
    Route2SCFIterationState,
)
from maple.function.calculator.extra_correction.implicit.route2_fixed_point import (
    DAMPED_PICARD_SOLVER,
    SAFEGUARDED_ANDERSON_SOLVER,
    FixedPointStep,
)
from maple.function.calculator.extra_correction.implicit.route2_field_state import (
    ReactionFieldDrive,
)


@dataclass
class _State:
    energy_ev: float
    density_coefficients: np.ndarray


@dataclass
class _CDS:
    energy_hartree: float = 0.0


class _ExactFeatureReactionMap:
    atom_count = 2

    def __init__(self):
        self.last_density = None
        self.last_drive = None

    def apply_scf_drive(self, density):
        self.last_density = np.asarray(density, dtype=float).copy()
        dual_field = 0.25 * MACE_POLAR_L1_PAIRING.density_to_field_order(density)
        features = np.arange(16, dtype=float).reshape(2, 8) / 10.0
        self.last_drive = ReactionFieldDrive(
            density_dual_field_ev=dual_field,
            model_local_field_ev=None,
            model_field_features=features,
            projector="exact-gto-v1",
            model_field_gauge="atomic-center-mean-zero-v1",
            model_field_gauge_reference_ev=0.25,
        )
        return self.last_drive

    def scf_polarization_energy_hartree(self, density):
        assert self.last_drive is not None
        return (
            0.5
            * MACE_POLAR_L1_PAIRING.pair(
                density,
                self.last_drive.density_dual_field_ev,
            )
            / Hartree
        )


class _FeatureAwareCalculator:
    def __init__(self, state):
        self.state = state
        self.calls = []

    def polar_state(self, _atoms, **kwargs):
        self.calls.append(kwargs)
        assert "model_field_features" in kwargs
        assert "node_potential_ev" not in kwargs
        assert "node_gradient_ev_per_angstrom" not in kwargs
        np.testing.assert_allclose(
            kwargs["model_field_features"],
            np.arange(16, dtype=float).reshape(2, 8) / 10.0,
        )
        return self.state, {}


class _CenteredLocalReactionMap(_ExactFeatureReactionMap):
    def apply_scf_drive(self, density):
        self.last_density = np.asarray(density, dtype=float).copy()
        dual_field = 0.25 * MACE_POLAR_L1_PAIRING.density_to_field_order(density)
        gauge_reference = float(np.mean(dual_field[:, 0]))
        self.last_drive = ReactionFieldDrive.local_jet(
            dual_field,
            model_field_gauge="atomic-center-mean-zero-v1",
            model_field_gauge_reference_ev=gauge_reference,
        )
        return self.last_drive


class _LocalFieldAwareCalculator:
    def __init__(self, state, reaction_map):
        self.state = state
        self.reaction_map = reaction_map
        self.calls = []

    def polar_state(self, _atoms, **kwargs):
        self.calls.append(kwargs)
        assert "model_field_features" not in kwargs
        np.testing.assert_allclose(
            kwargs["node_potential_ev"],
            self.reaction_map.last_drive.model_local_field_ev[:, 0],
        )
        np.testing.assert_allclose(
            kwargs["node_gradient_ev_per_angstrom"],
            self.reaction_map.last_drive.model_local_field_ev[:, 1:],
        )
        return self.state, {}


class _IdentityReactionMap:
    atom_count = 2

    def __init__(self):
        self.last_density = None
        self.last_drive = None

    def apply_scf_drive(self, density):
        self.last_density = np.asarray(density, dtype=float).copy()
        dual_field = np.asarray(density, dtype=float).copy()
        self.last_drive = ReactionFieldDrive.local_jet(dual_field)
        return self.last_drive

    def scf_polarization_energy_hartree(self, density):
        assert self.last_drive is not None
        return (
            0.5
            * MACE_POLAR_L1_PAIRING.pair(
                density,
                self.last_drive.density_dual_field_ev,
            )
            / Hartree
        )


class _LinearContractiveCalculator:
    def __init__(self, fixed_point: np.ndarray, contraction: float):
        self.fixed_point = np.asarray(fixed_point, dtype=float).copy()
        self.contraction = float(contraction)

    def polar_state(self, _atoms, **kwargs):
        density = np.column_stack(
            (
                np.asarray(kwargs["node_potential_ev"], dtype=float),
                np.asarray(kwargs["node_gradient_ev_per_angstrom"], dtype=float),
            )
        )
        response = self.fixed_point + self.contraction * (density - self.fixed_point)
        energy_ev = float(np.sum(response * response))
        return _State(energy_ev=energy_ev, density_coefficients=response), {}


class _ChargeNoisyCalculator:
    def __init__(self, fixed_point: np.ndarray, per_atom_offset_e: float):
        self.fixed_point = np.asarray(fixed_point, dtype=float).copy()
        self.per_atom_offset_e = float(per_atom_offset_e)

    def polar_state(self, _atoms, **_kwargs):
        response = self.fixed_point.copy()
        response[:, 0] += self.per_atom_offset_e
        return _State(energy_ev=0.0, density_coefficients=response), {}


class _FiniteResolutionCalculator:
    def __init__(
        self,
        residuals: list[np.ndarray],
        *,
        energies_ev: list[float] | None = None,
    ):
        self.residuals = [
            np.asarray(residual, dtype=float).copy() for residual in residuals
        ]
        self.energies_ev = (
            [-9.9] * len(self.residuals)
            if energies_ev is None
            else [float(value) for value in energies_ev]
        )
        if len(self.energies_ev) != len(self.residuals):
            raise ValueError("Energy and residual schedules must have equal length.")
        self.calls = 0

    def polar_state(self, _atoms, **kwargs):
        density = np.column_stack(
            (
                np.asarray(kwargs["node_potential_ev"], dtype=float),
                np.asarray(
                    kwargs["node_gradient_ev_per_angstrom"],
                    dtype=float,
                ),
            )
        )
        if self.calls >= len(self.residuals):
            raise RuntimeError("Finite-resolution response schedule exhausted.")
        response = density + self.residuals[self.calls]
        energy_ev = self.energies_ev[self.calls]
        self.calls += 1
        return _State(energy_ev=energy_ev, density_coefficients=response), {}


def _settings() -> Route2EngineSettings:
    return Route2EngineSettings(
        continuum_label="synthetic exact GTO",
        scf_mixing=0.5,
        scf_density_tolerance=1.0e-12,
        scf_energy_tolerance_ev=1.0e-12,
        scf_max_iterations=3,
        adjoint_relative_tolerance=1.0e-10,
        adjoint_absolute_tolerance=1.0e-13,
        adjoint_max_iterations=20,
        energy_identity_tolerance_ev=1.0e-12,
        force_state_energy_tolerance_ev=1.0e-12,
        neutral_density_tolerance=1.0e-12,
        scf_solver=DAMPED_PICARD_SOLVER,
    )


def _finite_resolution_policy() -> Route2FiniteResolutionPolicy:
    return Route2FiniteResolutionPolicy(
        version="finite-resolution-stagnation-v1",
        history_length=7,
        map_replay_count=3,
        monopole_residual_ceiling_e=1.0e-10,
        dipole_residual_ceiling_e_angstrom=1.0e-10,
        potential_span_tolerance_ev=1.0e-10,
        gradient_span_tolerance_ev_per_angstrom=1.0e-10,
        ledger_span_tolerance_ev=1.0e-10,
    )


def _finite_resolution_case(
    monkeypatch,
    *,
    residuals: list[np.ndarray],
    energies_ev: list[float] | None = None,
    maximum_iterations: int = 8,
    policy: Route2FiniteResolutionPolicy | None,
    runtime_identity: dict[str, object] | None,
):
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    gas_density = np.asarray(
        [[-0.2, 0.1, -0.3, 0.4], [0.2, -0.5, 0.6, -0.7]],
        dtype=float,
    )
    gas_state = _State(energy_ev=-10.0, density_coefficients=gas_density)
    calculator = _FiniteResolutionCalculator(
        residuals,
        energies_ev=energies_ev,
    )
    reaction_maps: list[_IdentityReactionMap] = []

    def reaction_field_factory(_atoms):
        reaction_map = _IdentityReactionMap()
        reaction_maps.append(reaction_map)
        return reaction_map

    roots = []
    for step in range(1, maximum_iterations):
        delta = step * 1.0e-13
        root = gas_density.copy()
        root[:, 0] += np.asarray([delta, -delta])
        root[:, 1:] += delta
        roots.append(root)
    proposed_roots = iter(roots)

    def controlled_step(samples, **_kwargs):
        return FixedPointStep(
            density=next(proposed_roots),
            method=SAFEGUARDED_ANDERSON_SOLVER,
            history_size=len(samples),
        )

    monkeypatch.setattr(
        route2_engine_module,
        "next_fixed_point_density",
        controlled_step,
    )
    settings = replace(
        _settings(),
        continuum_label="finite-resolution synthetic ddPCM",
        scf_mixing=1.0,
        scf_density_tolerance=2.0e-12,
        scf_energy_tolerance_ev=1.0e-10,
        scf_max_iterations=maximum_iterations,
        scf_solver=SAFEGUARDED_ANDERSON_SOLVER,
        scf_finite_resolution_policy=policy,
    )
    engine = Route2ContinuumEngine(
        reaction_field_factory=reaction_field_factory,
        cds_evaluator=lambda _atoms: _CDS(),
        settings=settings,
    )
    return (
        engine,
        atoms,
        calculator,
        gas_state,
        reaction_maps,
        runtime_identity,
    )


def test_engine_keeps_model_features_separate_from_energy_dual_field():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    density = np.asarray([[-0.2, 0.1, -0.3, 0.4], [0.2, -0.5, 0.6, -0.7]])
    gas_state = _State(energy_ev=-10.0, density_coefficients=density)
    solvent_state = _State(energy_ev=-9.9, density_coefficients=density)
    reaction_map = _ExactFeatureReactionMap()
    calculator = _FeatureAwareCalculator(solvent_state)
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: reaction_map,
        cds_evaluator=lambda _atoms: _CDS(),
        settings=_settings(),
    )

    coupled = engine.solve_coupled_state(
        atoms,
        calculator,
        gas_state,
        provider_cache_signature=("synthetic",),
    )

    assert len(calculator.calls) == 1
    np.testing.assert_allclose(
        coupled.reaction_field_values_ev,
        reaction_map.last_drive.density_dual_field_ev,
    )
    np.testing.assert_allclose(
        coupled.model_field_features,
        reaction_map.last_drive.model_field_features,
    )
    assert coupled.reaction_field_projector == "exact-gto-v1"
    assert coupled.model_field_gauge == "atomic-center-mean-zero-v1"
    assert coupled.model_field_gauge_reference_ev == pytest.approx(0.25)
    assert coupled.energy_identity_error_ev == pytest.approx(0.0, abs=1.0e-15)


def test_exact_model_feature_state_fails_closed_in_legacy_force_path():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    density = np.asarray([[-0.2, 0.1, -0.3, 0.4], [0.2, -0.5, 0.6, -0.7]])
    gas_state = _State(energy_ev=-10.0, density_coefficients=density)
    solvent_state = _State(energy_ev=-9.9, density_coefficients=density)
    reaction_map = _ExactFeatureReactionMap()
    calculator = _FeatureAwareCalculator(solvent_state)
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: reaction_map,
        cds_evaluator=lambda _atoms: _CDS(),
        settings=_settings(),
    )
    coupled = engine.solve_coupled_state(
        atoms,
        calculator,
        gas_state,
        provider_cache_signature=("synthetic",),
    )

    with pytest.raises(
        NotImplementedError,
        match="non-default model-field profile.*energy-only",
    ):
        engine.solvent_correction_force(
            atoms,
            calculator,
            gas_state,
            coupled,
        )


def test_engine_keeps_centered_local_model_field_out_of_energy_pairing():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    density = np.asarray([[-0.2, 0.1, -0.3, 0.4], [0.2, -0.5, 0.6, -0.7]])
    gas_state = _State(energy_ev=-10.0, density_coefficients=density)
    solvent_state = _State(energy_ev=-9.9, density_coefficients=density)
    reaction_map = _CenteredLocalReactionMap()
    calculator = _LocalFieldAwareCalculator(solvent_state, reaction_map)
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: reaction_map,
        cds_evaluator=lambda _atoms: _CDS(),
        settings=_settings(),
    )

    coupled = engine.solve_coupled_state(
        atoms,
        calculator,
        gas_state,
        provider_cache_signature=("synthetic-centered-local",),
    )

    np.testing.assert_allclose(
        coupled.reaction_field_values_ev,
        reaction_map.last_drive.density_dual_field_ev,
    )
    np.testing.assert_allclose(
        coupled.model_local_field_values_ev,
        reaction_map.last_drive.model_local_field_ev,
    )
    assert coupled.model_field_features is None
    assert coupled.model_field_gauge == "atomic-center-mean-zero-v1"
    assert coupled.energy_identity_error_ev == pytest.approx(0.0, abs=1.0e-15)
    with pytest.raises(
        NotImplementedError,
        match="non-default model-field profile.*energy-only",
    ):
        engine.solvent_correction_force(
            atoms,
            calculator,
            gas_state,
            coupled,
        )


def test_engine_uses_anderson_acceleration_for_unit_mixing_fixed_point_iterations():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    fixed_point = np.asarray([[-0.4, 0.2, -0.1, 0.3], [0.4, -0.2, 0.1, -0.3]])
    gas_density = np.zeros_like(fixed_point)
    gas_state = _State(energy_ev=0.0, density_coefficients=gas_density)
    calculator = _LinearContractiveCalculator(fixed_point, contraction=0.95)
    reaction_map = _IdentityReactionMap()
    settings = replace(
        _settings(),
        continuum_label="synthetic ddPCM",
        scf_mixing=1.0,
        scf_density_tolerance=1.0e-8,
        scf_energy_tolerance_ev=1.0e-8,
        scf_max_iterations=5,
        scf_solver=SAFEGUARDED_ANDERSON_SOLVER,
    )
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: reaction_map,
        cds_evaluator=lambda _atoms: _CDS(),
        settings=settings,
    )

    coupled = engine.solve_coupled_state(
        atoms,
        calculator,
        gas_state,
        provider_cache_signature=("synthetic-ddpcm",),
    )

    np.testing.assert_allclose(
        coupled.density_coefficients,
        fixed_point,
        atol=1.0e-12,
    )
    assert coupled.history[0]["next_density_update"] == DAMPED_PICARD_SOLVER
    assert coupled.history[1]["next_density_update"] == SAFEGUARDED_ANDERSON_SOLVER
    assert coupled.history[2]["next_density_update"] == SAFEGUARDED_ANDERSON_SOLVER
    assert coupled.history[-1]["next_density_update"] == "converged"


def test_engine_converges_with_the_neutral_tangent_unmixed_residual():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    fixed_point = np.asarray(
        [[-0.4, 0.2, -0.1, 0.3], [0.4, -0.2, 0.1, -0.3]]
    )
    gas_state = _State(energy_ev=0.0, density_coefficients=fixed_point)
    calculator = _ChargeNoisyCalculator(
        fixed_point,
        per_atom_offset_e=5.0e-12,
    )
    reaction_map = _IdentityReactionMap()
    settings = replace(
        _settings(),
        continuum_label="synthetic constrained ddPCM",
        scf_mixing=1.0,
        scf_density_tolerance=1.0e-12,
        scf_energy_tolerance_ev=1.0e-12,
        scf_max_iterations=2,
        neutral_density_tolerance=1.0e-8,
    )
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: reaction_map,
        cds_evaluator=lambda _atoms: _CDS(),
        settings=settings,
    )

    coupled = engine.solve_coupled_state(
        atoms,
        calculator,
        gas_state,
        provider_cache_signature=("synthetic-constrained-ddpcm",),
    )

    assert coupled.density_residual_inf == pytest.approx(0.0, abs=1.0e-18)
    assert coupled.history[0]["raw_response_total_charge_e"] == pytest.approx(
        1.0e-11
    )
    assert coupled.history[0][
        "response_charge_projection_max_e"
    ] == pytest.approx(5.0e-12)
    assert coupled.history[0]["next_density_update"] == "converged"


def test_engine_resets_anderson_history_after_observed_residual_growth(
    monkeypatch,
):
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    fixed_point = np.asarray(
        [[-0.4, 0.2, -0.1, 0.3], [0.4, -0.2, 0.1, -0.3]]
    )
    gas_state = _State(
        energy_ev=0.0,
        density_coefficients=np.zeros_like(fixed_point),
    )
    calculator = _LinearContractiveCalculator(
        fixed_point,
        contraction=0.5,
    )
    reaction_map = _IdentityReactionMap()
    proposed_densities = iter(
        (
            -10.0 * fixed_point,
            fixed_point,
            fixed_point,
        )
    )

    def _controlled_step(samples, **_kwargs):
        density = next(proposed_densities)
        method = (
            SAFEGUARDED_ANDERSON_SOLVER
            if len(samples) == 1 and np.allclose(samples[-1].density, 0.0)
            else DAMPED_PICARD_SOLVER
        )
        return FixedPointStep(
            density=density,
            method=method,
            history_size=len(samples),
        )

    monkeypatch.setattr(
        route2_engine_module,
        "next_fixed_point_density",
        _controlled_step,
    )
    settings = replace(
        _settings(),
        continuum_label="synthetic reset ddPCM",
        scf_mixing=1.0,
        scf_density_tolerance=1.0e-12,
        scf_energy_tolerance_ev=1.0e-12,
        scf_max_iterations=4,
        scf_solver=SAFEGUARDED_ANDERSON_SOLVER,
    )
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: reaction_map,
        cds_evaluator=lambda _atoms: _CDS(),
        settings=settings,
    )

    coupled = engine.solve_coupled_state(
        atoms,
        calculator,
        gas_state,
        provider_cache_signature=("synthetic-reset-ddpcm",),
    )

    assert coupled.history[1]["arrived_by"] == SAFEGUARDED_ANDERSON_SOLVER
    assert coupled.history[1]["anderson_history_reset"] is True
    assert coupled.history[1]["fixed_point_history_size"] == 1
    assert coupled.history[-1]["next_density_update"] == "converged"


def test_engine_nonconvergence_exposes_the_complete_numerical_history():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    fixed_point = np.asarray(
        [[-0.4, 0.2, -0.1, 0.3], [0.4, -0.2, 0.1, -0.3]]
    )
    gas_state = _State(
        energy_ev=0.0,
        density_coefficients=np.zeros_like(fixed_point),
    )
    calculator = _LinearContractiveCalculator(
        fixed_point,
        contraction=0.95,
    )
    reaction_map = _IdentityReactionMap()
    settings = replace(
        _settings(),
        continuum_label="synthetic failing ddPCM",
        scf_mixing=1.0,
        scf_density_tolerance=1.0e-12,
        scf_energy_tolerance_ev=1.0e-12,
        scf_max_iterations=2,
        scf_solver=DAMPED_PICARD_SOLVER,
    )
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: reaction_map,
        cds_evaluator=lambda _atoms: _CDS(),
        settings=settings,
    )

    with pytest.raises(Route2SCFConvergenceError) as caught:
        engine.solve_coupled_state(
            atoms,
            calculator,
            gas_state,
            provider_cache_signature=("synthetic-failing-ddpcm",),
        )

    assert len(caught.value.history) == 2
    assert caught.value.history[-1]["density_residual_e"] > 1.0e-12
    assert "minimum density residual=" in str(caught.value)
    assert isinstance(caught.value.best_state, Route2SCFIterationState)
    assert caught.value.best_state.iteration == 2
    assert caught.value.best_state.density_residual_e == pytest.approx(
        min(record["density_residual_e"] for record in caught.value.history)
    )
    assert caught.value.best_state.density_coefficients.flags.writeable is False
    assert (
        caught.value.best_state.response_density_coefficients.flags.writeable
        is False
    )


def test_engine_accepts_the_earliest_online_window_satisfying_all_predicates(
    monkeypatch,
):
    warm_residuals = []
    for step in range(8):
        delta = step * 1.0e-13
        warm_residuals.append(
            np.asarray(
                [
                    [5.0e-11 + delta, 4.0e-11 + delta, 0.0, 0.0],
                    [-5.0e-11 - delta, -4.0e-11 - delta, 0.0, 0.0],
                ]
            )
        )
    replay_residual = warm_residuals[-1]
    runtime_identity = {
        "profile": "synthetic-frozen-profile",
        "device": "cpu",
        "dtype": "torch.float64",
        "torch_threads": 1,
    }
    case = _finite_resolution_case(
        monkeypatch,
        residuals=warm_residuals + [replay_residual] * 3,
        policy=_finite_resolution_policy(),
        runtime_identity=runtime_identity,
    )
    engine, atoms, calculator, gas_state, reaction_maps, identity = case

    coupled = engine.solve_coupled_state(
        atoms,
        calculator,
        gas_state,
        provider_cache_signature=("finite-resolution",),
        finite_resolution_runtime_identity=identity,
    )

    assert len(coupled.history) == 8
    assert coupled.history[-1]["next_density_update"] == (
        "converged-finite-resolution"
    )
    assert len(reaction_maps) == 4
    assert calculator.calls == 11
    assert coupled.scf_convergence["reason"] == (
        "finite-resolution-stagnation-v1"
    )
    assert coupled.scf_convergence["online_candidate_iteration"] == 8
    assert coupled.scf_convergence["final_monopole_residual_e"] <= 1.0e-10
    assert (
        coupled.scf_convergence["final_dipole_residual_e_angstrom"]
        <= 1.0e-10
    )
    assert coupled.scf_convergence["runtime_identity"] == runtime_identity
    window = coupled.scf_convergence["history_window"]
    assert window["start_iteration"] == 2
    assert window["end_iteration"] == 8
    assert window["root_monopole_span_e"] <= 2.0e-12
    assert window["root_dipole_span_e_angstrom"] <= 2.0e-12
    assert window["residual_monopole_span_e"] <= 2.0e-12
    assert window["residual_dipole_span_e_angstrom"] <= 2.0e-12
    assert window["potential_span_ev"] <= 1.0e-10
    assert window["gradient_span_ev_per_angstrom"] <= 1.0e-10
    replay = coupled.scf_convergence["fresh_map_replay"]
    assert replay["replay_count"] == 3
    assert replay["evaluation_count"] == 4
    assert replay["includes_online_candidate"] is True
    assert replay["all_field_arrays_identical"] is True
    assert replay["all_response_arrays_identical"] is True
    assert replay["maximum_monopole_residual_e"] <= 1.0e-10
    assert replay["maximum_dipole_residual_e_angstrom"] <= 1.0e-10
    assert replay["electrostatic_ledger_span_ev"] <= 1.0e-10
    assert replay["maximum_polarization_identity_error_ev"] <= 1.0e-12
    assert len(replay["field_sha256"]) == 64
    assert len(replay["response_sha256"]) == 64
    np.testing.assert_allclose(
        coupled.density_coefficients,
        reaction_maps[0].last_density,
    )


@pytest.mark.parametrize(
    ("policy", "runtime_identity"),
    (
        (None, {"profile": "synthetic-frozen-profile"}),
        (_finite_resolution_policy(), None),
    ),
)
def test_engine_finite_resolution_branch_is_both_configured_and_runtime_armed(
    monkeypatch,
    policy,
    runtime_identity,
):
    residual = np.asarray(
        [
            [5.0e-11, 4.0e-11, 0.0, 0.0],
            [-5.0e-11, -4.0e-11, 0.0, 0.0],
        ]
    )
    case = _finite_resolution_case(
        monkeypatch,
        residuals=[residual] * 8,
        policy=policy,
        runtime_identity=runtime_identity,
    )
    engine, atoms, calculator, gas_state, _, identity = case

    with pytest.raises(Route2SCFConvergenceError):
        engine.solve_coupled_state(
            atoms,
            calculator,
            gas_state,
            provider_cache_signature=("finite-resolution-disabled",),
            finite_resolution_runtime_identity=identity,
        )


def test_engine_rejects_a_stable_residual_norm_with_changing_directions(
    monkeypatch,
):
    residuals = []
    for step in range(8):
        sign = -1.0 if step % 2 else 1.0
        residuals.append(
            np.asarray(
                [
                    [sign * 5.0e-11, sign * 4.0e-11, 0.0, 0.0],
                    [-sign * 5.0e-11, -sign * 4.0e-11, 0.0, 0.0],
                ]
            )
        )
    case = _finite_resolution_case(
        monkeypatch,
        residuals=residuals,
        policy=_finite_resolution_policy(),
        runtime_identity={"profile": "synthetic-frozen-profile"},
    )
    engine, atoms, calculator, gas_state, _, identity = case

    with pytest.raises(Route2SCFConvergenceError):
        engine.solve_coupled_state(
            atoms,
            calculator,
            gas_state,
            provider_cache_signature=("finite-resolution-cycle",),
            finite_resolution_runtime_identity=identity,
        )


def test_engine_rejects_direction_reversal_in_either_physical_channel(
    monkeypatch,
):
    residuals = []
    for step in range(8):
        dipole_sign = -1.0 if step % 2 else 1.0
        residuals.append(
            np.asarray(
                [
                    [5.0e-11, dipole_sign * 5.0e-13, 0.0, 0.0],
                    [-5.0e-11, -dipole_sign * 5.0e-13, 0.0, 0.0],
                ]
            )
        )
    case = _finite_resolution_case(
        monkeypatch,
        residuals=residuals,
        policy=_finite_resolution_policy(),
        runtime_identity={"profile": "synthetic-frozen-profile"},
    )
    engine, atoms, calculator, gas_state, _, identity = case

    with pytest.raises(Route2SCFConvergenceError):
        engine.solve_coupled_state(
            atoms,
            calculator,
            gas_state,
            provider_cache_signature=("finite-resolution-channel-cycle",),
            finite_resolution_runtime_identity=identity,
        )


def test_engine_rejects_nonrepeatable_fresh_map_replay(
    monkeypatch,
):
    warm_residual = np.asarray(
        [
            [5.0e-11, 4.0e-11, 0.0, 0.0],
            [-5.0e-11, -4.0e-11, 0.0, 0.0],
        ]
    )
    replay_residuals = [
        warm_residual,
        warm_residual + np.asarray(
            [[0.0, 3.0e-12, 0.0, 0.0], [0.0, -3.0e-12, 0.0, 0.0]]
        ),
        warm_residual,
    ]
    case = _finite_resolution_case(
        monkeypatch,
        residuals=[warm_residual] * 8 + replay_residuals,
        policy=_finite_resolution_policy(),
        runtime_identity={"profile": "synthetic-frozen-profile"},
    )
    engine, atoms, calculator, gas_state, _, identity = case

    with pytest.raises(Route2SCFConvergenceError):
        engine.solve_coupled_state(
            atoms,
            calculator,
            gas_state,
            provider_cache_signature=("finite-resolution-replay-mismatch",),
            finite_resolution_runtime_identity=identity,
        )


def test_engine_rejects_online_to_map_replay_ledger_drift(monkeypatch):
    residual = np.asarray(
        [
            [5.0e-11, 4.0e-11, 0.0, 0.0],
            [-5.0e-11, -4.0e-11, 0.0, 0.0],
        ]
    )
    case = _finite_resolution_case(
        monkeypatch,
        residuals=[residual] * 11,
        energies_ev=[-9.9] * 8 + [-8.9] * 3,
        policy=_finite_resolution_policy(),
        runtime_identity={"profile": "synthetic-frozen-profile"},
    )
    engine, atoms, calculator, gas_state, _, identity = case

    with pytest.raises(
        Route2SCFConvergenceError,
        match="online/map-replay ledger span",
    ):
        engine.solve_coupled_state(
            atoms,
            calculator,
            gas_state,
            provider_cache_signature=("finite-resolution-ledger-drift",),
            finite_resolution_runtime_identity=identity,
        )


def test_engine_rejects_nonfinite_map_replay_energy(monkeypatch):
    residual = np.asarray(
        [
            [5.0e-11, 4.0e-11, 0.0, 0.0],
            [-5.0e-11, -4.0e-11, 0.0, 0.0],
        ]
    )
    case = _finite_resolution_case(
        monkeypatch,
        residuals=[residual] * 11,
        energies_ev=[-9.9] * 8 + [float("nan")] * 3,
        policy=_finite_resolution_policy(),
        runtime_identity={"profile": "synthetic-frozen-profile"},
    )
    engine, atoms, calculator, gas_state, _, identity = case

    with pytest.raises(
        Route2SCFConvergenceError,
        match="non-finite energy scalar",
    ):
        engine.solve_coupled_state(
            atoms,
            calculator,
            gas_state,
            provider_cache_signature=("finite-resolution-nonfinite-replay",),
            finite_resolution_runtime_identity=identity,
        )


def test_engine_rejects_forces_from_finite_resolution_candidate(monkeypatch):
    residual = np.asarray(
        [
            [5.0e-11, 4.0e-11, 0.0, 0.0],
            [-5.0e-11, -4.0e-11, 0.0, 0.0],
        ]
    )
    case = _finite_resolution_case(
        monkeypatch,
        residuals=[residual] * 11,
        policy=_finite_resolution_policy(),
        runtime_identity={"profile": "synthetic-frozen-profile"},
    )
    engine, atoms, calculator, gas_state, _, identity = case
    coupled = engine.solve_coupled_state(
        atoms,
        calculator,
        gas_state,
        provider_cache_signature=("finite-resolution-energy-only",),
        finite_resolution_runtime_identity=identity,
    )

    calls_before_force = calculator.calls
    with pytest.raises(RuntimeError, match="energy-only"):
        engine.solvent_correction_force(
            atoms,
            calculator,
            gas_state,
            coupled,
        )
    assert calculator.calls == calls_before_force


def test_engine_reports_nominal_channel_specific_convergence():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    gas_density = np.asarray(
        [[-0.2, 0.1, -0.3, 0.4], [0.2, -0.5, 0.6, -0.7]],
        dtype=float,
    )
    residual = np.asarray(
        [[5.0e-13, 1.5e-12, 0.0, 0.0], [-5.0e-13, -1.5e-12, 0.0, 0.0]]
    )
    gas_state = _State(energy_ev=-10.0, density_coefficients=gas_density)
    calculator = _FiniteResolutionCalculator([residual])
    reaction_map = _IdentityReactionMap()
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: reaction_map,
        cds_evaluator=lambda _atoms: _CDS(),
        settings=replace(
            _settings(),
            scf_density_tolerance=1.0e-12,
            scf_dipole_tolerance_e_angstrom=2.0e-12,
            scf_max_iterations=1,
        ),
    )

    coupled = engine.solve_coupled_state(
        atoms,
        calculator,
        gas_state,
        provider_cache_signature=("nominal-channel-specific",),
    )

    assert coupled.history[-1]["next_density_update"] == "converged"
    assert coupled.history[-1]["monopole_residual_e"] == pytest.approx(5.0e-13)
    assert coupled.history[-1]["dipole_residual_e_angstrom"] == pytest.approx(
        1.5e-12
    )
    assert coupled.scf_convergence == {
        "reason": "nominal-density-and-energy-v1",
        "online_candidate_iteration": 1,
        "final_monopole_residual_e": pytest.approx(5.0e-13),
        "final_dipole_residual_e_angstrom": pytest.approx(1.5e-12),
        "runtime_identity": None,
        "history_window": None,
        "fresh_map_replay": None,
    }


def test_engine_rejects_component_drift_hidden_by_a_constant_peak_norm(
    monkeypatch,
):
    residuals = []
    for step in range(8):
        first = 5.0e-11 if step % 2 == 0 else 4.0e-11
        second = 4.0e-11 if step % 2 == 0 else 5.0e-11
        residuals.append(
            np.asarray(
                [
                    [5.0e-11, first, 0.0, 0.0],
                    [-5.0e-11, -second, 0.0, 0.0],
                ]
            )
        )
    case = _finite_resolution_case(
        monkeypatch,
        residuals=residuals,
        policy=_finite_resolution_policy(),
        runtime_identity={"profile": "synthetic-frozen-profile"},
    )
    engine, atoms, calculator, gas_state, _, identity = case

    with pytest.raises(Route2SCFConvergenceError):
        engine.solve_coupled_state(
            atoms,
            calculator,
            gas_state,
            provider_cache_signature=("finite-resolution-component-drift",),
            finite_resolution_runtime_identity=identity,
        )


def test_finite_resolution_energy_span_uses_only_the_triggering_window(
    monkeypatch,
):
    residual = np.asarray(
        [
            [5.0e-11, 4.0e-11, 0.0, 0.0],
            [-5.0e-11, -4.0e-11, 0.0, 0.0],
        ]
    )
    case = _finite_resolution_case(
        monkeypatch,
        residuals=[residual] * 12,
        energies_ev=[-9.0] + [-9.9] * 11,
        maximum_iterations=9,
        policy=_finite_resolution_policy(),
        runtime_identity={"profile": "synthetic-frozen-profile"},
    )
    engine, atoms, calculator, gas_state, _, identity = case

    coupled = engine.solve_coupled_state(
        atoms,
        calculator,
        gas_state,
        provider_cache_signature=("finite-resolution-window-energy",),
        finite_resolution_runtime_identity=identity,
    )

    assert coupled.scf_convergence["online_candidate_iteration"] == 9
    assert (
        coupled.scf_convergence["history_window"]["intrinsic_energy_span_ev"]
        == pytest.approx(0.0)
    )
