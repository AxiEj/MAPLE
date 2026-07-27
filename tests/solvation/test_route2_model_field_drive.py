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
    NEAR_ROOT_NEWTON_CORRECTOR,
    Route2ContinuumEngine,
    Route2EngineSettings,
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
        self.last_drive = None

    def apply_scf_drive(self, density):
        dual_field = np.asarray(density, dtype=float).copy()
        self.last_drive = ReactionFieldDrive.local_jet(dual_field)
        return self.last_drive

    def apply(self, density_direction):
        return np.asarray(density_direction, dtype=float).copy()

    def adjoint(self, field_cotangent):
        return np.asarray(field_cotangent, dtype=float).copy()

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
    def __init__(
        self,
        fixed_point: np.ndarray,
        contraction: float,
        *,
        linearization_contraction: float | None = None,
    ):
        self.fixed_point = np.asarray(fixed_point, dtype=float).copy()
        self.contraction = float(contraction)
        self.linearization_contraction = float(
            contraction
            if linearization_contraction is None
            else linearization_contraction
        )

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

    def linearize_density_response(self, _atoms, **_kwargs):
        return _ScaledDensityResponse(self.linearization_contraction)


class _ScaledDensityResponse:
    def __init__(self, scale: float):
        self.scale = float(scale)

    def jvp(self, field_direction):
        return self.scale * np.asarray(field_direction, dtype=float)

    def vjp(self, density_cotangent):
        return self.scale * np.asarray(density_cotangent, dtype=float)


class _LineSearchFailingCalculator(_LinearContractiveCalculator):
    def __init__(self, fixed_point: np.ndarray, contraction: float):
        super().__init__(fixed_point, contraction)
        self.polar_calls = 0

    def polar_state(self, atoms, **kwargs):
        self.polar_calls += 1
        if 2 <= self.polar_calls <= 4:
            raise RuntimeError("synthetic trial-map failure")
        return super().polar_state(atoms, **kwargs)


class _ChargeNoisyCalculator:
    def __init__(self, fixed_point: np.ndarray, per_atom_offset_e: float):
        self.fixed_point = np.asarray(fixed_point, dtype=float).copy()
        self.per_atom_offset_e = float(per_atom_offset_e)

    def polar_state(self, _atoms, **_kwargs):
        response = self.fixed_point.copy()
        response[:, 0] += self.per_atom_offset_e
        return _State(energy_ev=0.0, density_coefficients=response), {}


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


@pytest.mark.parametrize("model_drive", ["exact-gto", "gauge-transformed-local"])
def test_near_root_newton_fails_closed_for_unsupported_model_drives(
    model_drive,
):
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    fixed_point = np.asarray(
        [[-0.4, 0.2, -0.1, 0.3], [0.4, -0.2, 0.1, -0.3]]
    )
    gas_state = _State(
        energy_ev=0.0,
        density_coefficients=np.zeros_like(fixed_point),
    )
    solvent_state = _State(
        energy_ev=-9.9,
        density_coefficients=fixed_point,
    )
    if model_drive == "exact-gto":
        reaction_map = _ExactFeatureReactionMap()
        calculator = _FeatureAwareCalculator(solvent_state)
    else:
        reaction_map = _CenteredLocalReactionMap()
        calculator = _LocalFieldAwareCalculator(
            solvent_state,
            reaction_map,
        )
    settings = replace(
        _settings(),
        continuum_label="synthetic unsupported Newton drive",
        scf_mixing=1.0,
        scf_density_tolerance=1.0e-1,
        scf_energy_tolerance_ev=1.0,
        scf_max_iterations=2,
        scf_newton_trigger_factor=5.0,
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
        provider_cache_signature=("synthetic-unsupported-newton-drive",),
    )

    first = coupled.history[0]
    assert first["newton_attempted"] is True
    assert first["newton_accepted"] is False
    assert first["newton_fallback_reason"] == (
        "unsupported-nonlocal-or-gauge-transformed-model-field"
    )
    assert first["next_density_update"] == DAMPED_PICARD_SOLVER
    assert coupled.history[-1]["next_density_update"] == "converged"


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


def test_engine_accepts_fresh_map_near_root_newton_correction():
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
        continuum_label="synthetic Newton ddPCM",
        scf_mixing=1.0,
        scf_density_tolerance=1.0e-3,
        scf_energy_tolerance_ev=1.0,
        scf_max_iterations=3,
        scf_newton_trigger_factor=100.0,
        scf_newton_step_ratio_limit=25.0,
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
        provider_cache_signature=("synthetic-newton-ddpcm",),
    )

    first = coupled.history[0]
    assert first["newton_attempted"] is True
    assert first["newton_accepted"] is True
    assert first["newton_accepted_alpha"] == pytest.approx(1.0)
    assert first["newton_trial_residual_e"] is not None
    assert first["newton_trial_residual_e"] < first["density_residual_e"]
    assert first["next_density_update"] == NEAR_ROOT_NEWTON_CORRECTOR
    assert coupled.history[1]["arrived_by"] == NEAR_ROOT_NEWTON_CORRECTOR
    assert coupled.history[1]["next_density_update"] == "converged"
    np.testing.assert_allclose(
        coupled.density_coefficients,
        fixed_point,
        atol=1.0e-12,
    )


def test_engine_backtracks_newton_until_fresh_map_residual_decreases():
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
        linearization_contraction=0.75,
    )
    reaction_map = _IdentityReactionMap()
    settings = replace(
        _settings(),
        continuum_label="synthetic backtracked Newton ddPCM",
        scf_mixing=1.0,
        scf_density_tolerance=1.0e-1,
        scf_energy_tolerance_ev=1.0,
        scf_max_iterations=3,
        scf_newton_trigger_factor=3.0,
        scf_newton_step_ratio_limit=10.0,
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
        provider_cache_signature=("synthetic-backtracked-newton-ddpcm",),
    )

    first = coupled.history[0]
    assert first["newton_attempted"] is True
    assert first["newton_accepted"] is True
    assert first["newton_accepted_alpha"] == pytest.approx(0.5)
    assert first["newton_trial_residual_e"] == pytest.approx(0.0)
    assert first["next_density_update"] == NEAR_ROOT_NEWTON_CORRECTOR
    assert coupled.history[-1]["next_density_update"] == "converged"


def test_engine_rejects_bad_newton_linearization_and_keeps_picard_path():
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
        linearization_contraction=2.0,
    )
    reaction_map = _IdentityReactionMap()
    settings = replace(
        _settings(),
        continuum_label="synthetic safeguarded Newton ddPCM",
        scf_mixing=1.0,
        scf_density_tolerance=1.0e-1,
        scf_energy_tolerance_ev=1.0,
        scf_max_iterations=3,
        scf_newton_trigger_factor=3.0,
        scf_newton_step_ratio_limit=10.0,
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
        provider_cache_signature=("synthetic-safeguarded-newton-ddpcm",),
    )

    first = coupled.history[0]
    assert first["newton_attempted"] is True
    assert first["newton_accepted"] is False
    assert first["newton_fallback_reason"] == (
        "fresh-map-line-search-rejected"
    )
    assert first["newton_trial_residual_e"] is not None
    assert first["newton_trial_residual_e"] > first["density_residual_e"]
    assert first["next_density_update"] == DAMPED_PICARD_SOLVER
    assert coupled.history[-1]["next_density_update"] == "converged"


def test_engine_preserves_failed_newton_trial_map_evidence():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    fixed_point = np.asarray(
        [[-0.4, 0.2, -0.1, 0.3], [0.4, -0.2, 0.1, -0.3]]
    )
    gas_state = _State(
        energy_ev=0.0,
        density_coefficients=np.zeros_like(fixed_point),
    )
    calculator = _LineSearchFailingCalculator(
        fixed_point,
        contraction=0.5,
    )
    reaction_map = _IdentityReactionMap()
    settings = replace(
        _settings(),
        continuum_label="synthetic trial-failure Newton ddPCM",
        scf_mixing=1.0,
        scf_density_tolerance=1.0e-1,
        scf_energy_tolerance_ev=1.0,
        scf_max_iterations=3,
        scf_newton_trigger_factor=3.0,
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
        provider_cache_signature=("synthetic-trial-failure-newton-ddpcm",),
    )

    first = coupled.history[0]
    assert first["newton_attempted"] is True
    assert first["newton_accepted"] is False
    fallback = first["newton_fallback_reason"]
    assert fallback is not None
    assert fallback.startswith("fresh-map-line-search-failed:")
    assert "alpha=1:RuntimeError:synthetic trial-map failure" in fallback
    assert "alpha=0.5:RuntimeError:synthetic trial-map failure" in fallback
    assert "alpha=0.25:RuntimeError:synthetic trial-map failure" in fallback
    assert first["next_density_update"] == DAMPED_PICARD_SOLVER
    assert coupled.history[-1]["next_density_update"] == "converged"


def test_near_root_newton_settings_reject_invalid_trigger_factor():
    with pytest.raises(ValueError, match="trigger factor"):
        replace(
            _settings(),
            scf_newton_trigger_factor=1.0,
        )


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
