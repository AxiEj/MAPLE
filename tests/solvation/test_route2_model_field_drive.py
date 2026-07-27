from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest
from ase import Atoms
from ase.units import Hartree

from maple.function.calculator.extra_correction.implicit.electrostatic_pairing import (
    MACE_POLAR_L1_PAIRING,
)
from maple.function.calculator.extra_correction.implicit.route2_engine import (
    Route2ContinuumEngine,
    Route2EngineSettings,
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
        scf_max_iterations=4,
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
    assert coupled.history[0]["next_density_update"] == "linear-mixing"
    assert coupled.history[1]["next_density_update"] == "anderson-accelerated"
    assert coupled.history[2]["next_density_update"] == "anderson-accelerated"
    assert coupled.history[3]["next_density_update"] == "converged"
