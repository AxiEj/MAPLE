from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
import json

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
    Route2SCFHistoryRecord,
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
from maple.function.route2_energy_ledger import PCM_HALF_COUPLING_ONLY_V1


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


class _ScheduledDensityCalculator:
    def __init__(
        self,
        responses: list[np.ndarray],
        *,
        energies_ev: list[float] | None = None,
    ):
        self.responses = [
            np.asarray(response, dtype=float).copy() for response in responses
        ]
        self.energies_ev = (
            [0.0] * len(self.responses)
            if energies_ev is None
            else [float(value) for value in energies_ev]
        )
        if len(self.energies_ev) != len(self.responses):
            raise ValueError("Energy and response schedules must align.")
        self.calls = 0
        self.polar_calls: list[np.ndarray] = []

    def polar_state(self, _atoms, **kwargs):
        if self.calls >= len(self.responses):
            raise RuntimeError("Scheduled response set exhausted.")
        if "model_field_features" in kwargs:
            raise RuntimeError(
                "Scheduled calculator only supports node-potential calls."
            )
        density = np.column_stack(
            (
                np.asarray(kwargs["node_potential_ev"], dtype=float),
                np.asarray(
                    kwargs["node_gradient_ev_per_angstrom"],
                    dtype=float,
                ),
            )
        )
        self.polar_calls.append(np.array(density, copy=True))
        response = self.responses[self.calls]
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
        version="finite-resolution-stagnation-v2",
        history_length=7,
        map_replay_count=3,
        monopole_residual_ceiling_e=1.0e-10,
        dipole_residual_ceiling_e_angstrom=1.0e-10,
        potential_span_tolerance_ev=1.0e-10,
        gradient_span_tolerance_ev_per_angstrom=1.0e-10,
        ledger_span_tolerance_ev=1.0e-10,
    )


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (0.0, -0.0, 0),
        (0.0, np.nextafter(0.0, np.inf), 1),
        (0.0, np.nextafter(0.0, -np.inf), 1),
        (-1.0, np.nextafter(-1.0, np.inf), 1),
        (1.0, 1.0, 0),
        (
            -np.finfo(np.float64).max,
            np.finfo(np.float64).max,
            0xFFDFFFFFFFFFFFFE,
        ),
    ],
)
def test_maximum_ulp_distance_uses_contiguous_signed_float_order(
    left,
    right,
    expected,
):
    assert Route2ContinuumEngine._maximum_ulp_distance(
        np.asarray([left]),
        np.asarray([right]),
    ) == expected


def test_array_digest_preserves_signed_zero_for_byte_identity():
    positive_zero = np.asarray([0.0])
    negative_zero = np.asarray([-0.0])

    assert Route2ContinuumEngine._array_sha256(positive_zero) != (
        Route2ContinuumEngine._array_sha256(negative_zero)
    )
    assert (
        Route2ContinuumEngine._maximum_ulp_distance(
            positive_zero,
            negative_zero,
        )
        == 0
    )


class _OffsetReplayFieldMap:
    atom_count = 2

    def __init__(self, offset: float = 0.0):
        self.offset = float(offset)
        self.last_density = None
        self.last_drive = None

    def apply_scf_drive(self, density):
        self.last_density = np.asarray(density, dtype=float).copy()
        dual_field = self.last_density.copy()
        if self.offset != 0.0:
            dual_field = dual_field.copy()
            dual_field[:, 0] += np.asarray([self.offset, -self.offset])
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


def _finite_resolution_case(
    monkeypatch,
    *,
    residuals: list[np.ndarray],
    energies_ev: list[float] | None = None,
    maximum_iterations: int = 8,
    policy: Route2FiniteResolutionPolicy | None,
    runtime_identity: dict[str, object] | None,
    reaction_field_factory: Callable[[Atoms], object] | None = None,
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
    reaction_maps: list[object] = []

    if reaction_field_factory is None:
        def tracked_reaction_field_factory(_atoms) -> _IdentityReactionMap:
            reaction_map = _IdentityReactionMap()
            reaction_maps.append(reaction_map)
            return reaction_map
    else:
        supplied_reaction_field_factory = reaction_field_factory

        def tracked_reaction_field_factory(
            _atoms,
        ) -> object:
            reaction_map = supplied_reaction_field_factory(_atoms)
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
        reaction_field_factory=tracked_reaction_field_factory,
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


def test_engine_does_not_reject_zero_over_zero_actual_residual_growth():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    gas_density = np.zeros((2, 4), dtype=float)
    first_residual = np.asarray(
        [[1.0e-2, 0.0, 0.0, 0.0], [-1.0e-2, 0.0, 0.0, 0.0]]
    )
    zero_residual = np.zeros_like(first_residual)
    calculator = _FiniteResolutionCalculator(
        [first_residual, zero_residual, zero_residual],
        energies_ev=[0.0, 1.0, 1.0],
    )
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: _IdentityReactionMap(),
        cds_evaluator=lambda _atoms: _CDS(),
        settings=replace(
            _settings(),
            continuum_label="synthetic zero-baseline ddPCM",
            scf_density_tolerance=1.0e-12,
            scf_energy_tolerance_ev=1.0e-12,
            scf_max_iterations=3,
            scf_solver=SAFEGUARDED_ANDERSON_SOLVER,
        ),
    )

    coupled = engine.solve_coupled_state(
        atoms,
        calculator,
        _State(energy_ev=0.0, density_coefficients=gas_density),
        provider_cache_signature=("zero-over-zero-growth",),
    )

    final = coupled.history[-1]
    assert final["arrived_by"] == SAFEGUARDED_ANDERSON_SOLVER
    assert final["actual_residual_objective"] == 0.0
    assert final["actual_residual_growth_baseline_objective"] == 0.0
    assert final["actual_residual_growth_ratio"] is None
    assert final["attempt_status"] == "accepted"
    assert final["accepted"] is True
    assert final["rejected"] is False
    assert final["next_density_update"] == "converged"
    json.dumps(coupled.history, allow_nan=False)


def test_engine_rejects_positive_actual_residual_growth_from_zero_baseline():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    gas_density = np.zeros((2, 4), dtype=float)
    first_residual = np.asarray(
        [[1.0e-2, 0.0, 0.0, 0.0], [-1.0e-2, 0.0, 0.0, 0.0]]
    )
    zero_residual = np.zeros_like(first_residual)
    positive_residual = np.asarray(
        [[1.0e-15, 0.0, 0.0, 0.0], [-1.0e-15, 0.0, 0.0, 0.0]]
    )
    calculator = _FiniteResolutionCalculator(
        [first_residual, zero_residual, positive_residual],
        energies_ev=[0.0, 1.0, 1.0],
    )
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: _IdentityReactionMap(),
        cds_evaluator=lambda _atoms: _CDS(),
        settings=replace(
            _settings(),
            continuum_label="synthetic positive-over-zero ddPCM",
            scf_density_tolerance=1.0e-16,
            scf_energy_tolerance_ev=1.0e-12,
            scf_max_iterations=3,
            scf_solver=SAFEGUARDED_ANDERSON_SOLVER,
        ),
    )

    with pytest.raises(Route2SCFConvergenceError) as caught:
        engine.solve_coupled_state(
            atoms,
            calculator,
            _State(energy_ev=0.0, density_coefficients=gas_density),
            provider_cache_signature=("positive-over-zero-growth",),
        )

    rejected = caught.value.history[-1]
    assert rejected["actual_residual_growth_baseline_objective"] == 0.0
    assert rejected["actual_residual_growth_ratio"] is None
    assert rejected["accepted"] is False
    assert rejected["rejected"] is True
    assert rejected["attempt_status"] == (
        "rejected-anderson-actual-residual-growth"
    )
    json.dumps(caught.value.history, allow_nan=False)


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


def test_direct_pcm_scf_tracks_pcm_half_coupling_not_mace_energy_drift():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    density = np.asarray(
        [[-0.2, 0.1, -0.3, 0.4], [0.2, -0.5, 0.6, -0.7]],
        dtype=float,
    )
    gas_state = _State(energy_ev=-10.0, density_coefficients=density)
    calculator = _ScheduledDensityCalculator(
        [density, density],
        energies_ev=[-9.9, 50.0],
    )
    reaction_map = _IdentityReactionMap()
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: reaction_map,
        cds_evaluator=lambda _atoms: _CDS(),
        settings=replace(
            _settings(),
            scf_require_two_energy_samples=True,
            scf_max_iterations=2,
        ),
    )

    coupled = engine.solve_coupled_state(
        atoms,
        calculator,
        gas_state,
        provider_cache_signature=("direct-pcm-energy-monitor",),
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    )

    assert coupled.history[-1]["next_density_update"] == "converged"
    assert coupled.history[-1]["intrinsic_energy_ev"] == pytest.approx(50.0)
    assert coupled.history[-1]["energy_residual_ev"] == pytest.approx(0.0)
    assert coupled.history[-1]["energy_residual_source"] == (
        "pcm-half-coupling-v1"
    )
    assert coupled.history[-1]["ledger_energy_ev"] == pytest.approx(
        reaction_map.scf_polarization_energy_hartree(density) * Hartree
    )


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

    def _controlled_step(samples, **kwargs):
        density = next(proposed_densities)
        if kwargs.get("solver") == DAMPED_PICARD_SOLVER:
            method = DAMPED_PICARD_SOLVER
        else:
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
    assert coupled.history[1]["fixed_point_history_size"] == 1
    assert coupled.history[1]["accepted"] is False
    assert coupled.history[1]["rejected"] is True
    assert coupled.history[1]["attempt_status"] == (
        "rejected-anderson-actual-residual-growth"
    )
    assert coupled.history[1]["anderson_history_reset"] is True
    assert coupled.history[1]["accepted_parent_attempt"] == 1
    assert coupled.history[1]["rollback_anchor_attempt"] == 1
    assert coupled.history[1]["next_density_update"] == DAMPED_PICARD_SOLVER
    assert coupled.history[1]["anderson_fallback_reason"] == (
        "anderson-actual-residual-growth-rejected"
    )
    assert coupled.history[2]["anderson_history_reset"] is False
    assert coupled.history[2]["rejected"] is False
    assert coupled.history[2]["accepted"] is True
    assert coupled.history[2]["accepted_parent_attempt"] == 1
    assert coupled.history[2]["rollback_anchor_attempt"] is None
    assert coupled.history[-1]["next_density_update"] == "converged"


def test_engine_rolls_back_anderson_attempt_to_anchor_picard_step(
    monkeypatch,
):
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    fixed_point = np.asarray(
        [[-0.4, 0.2, -0.1, 0.3], [0.4, -0.2, 0.1, -0.3]]
    )
    anchor_candidate = 2.0 * fixed_point
    anchor_step = 0.25 * fixed_point

    response_1 = fixed_point.copy()
    response_2 = 0.5 * fixed_point
    response_3 = 100.0 * fixed_point
    response_4 = anchor_candidate + (response_2 - anchor_step)
    calculator = _ScheduledDensityCalculator(
        responses=[response_1, response_2, response_3, response_4],
    )

    reaction_map = _IdentityReactionMap()
    proposed = iter((anchor_step, anchor_candidate, response_4))

    def _controlled_step(samples, **kwargs):
        density = np.array(next(proposed), copy=True)
        if kwargs.get("solver") == DAMPED_PICARD_SOLVER:
            method = DAMPED_PICARD_SOLVER
        else:
            method = (
                SAFEGUARDED_ANDERSON_SOLVER
                if len(samples) == 1
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
        continuum_label="synthetic rollback ddPCM",
        scf_mixing=1.0,
        scf_density_tolerance=1.0e-12,
        scf_energy_tolerance_ev=1.0e-12,
        scf_max_iterations=8,
        scf_solver=SAFEGUARDED_ANDERSON_SOLVER,
    )
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: reaction_map,
        cds_evaluator=lambda _atoms: _CDS(),
        settings=settings,
    )

    gas_state = _State(energy_ev=0.0, density_coefficients=np.zeros_like(fixed_point))
    coupled = engine.solve_coupled_state(
        atoms,
        calculator,
        gas_state,
        provider_cache_signature=("synthetic-anchor-rollback",),
    )

    rollback_density = anchor_candidate + (response_2 - anchor_step)
    assert len(calculator.polar_calls) >= 4
    np.testing.assert_allclose(
        calculator.polar_calls[3],
        rollback_density,
    )
    np.testing.assert_allclose(
        coupled.density_coefficients,
        rollback_density,
    )
    assert all(
        event["accepted"] is False
        for event in coupled.history
        if event["rejected"]
    )
    assert any(
        event["accepted"] and event["rollback_anchor_attempt"] is None
        for event in coupled.history
    )


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
    assert coupled.scf_convergence["reason"] == "finite-resolution-stagnation-v2"
    assert coupled.scf_convergence["online_candidate_iteration"] == 8
    assert coupled.scf_convergence["final_monopole_residual_e"] <= 1.0e-10
    assert (
        coupled.scf_convergence["final_dipole_residual_e_angstrom"]
        <= 1.0e-10
    )
    assert coupled.history[0]["reaction_potential_change_ev"] is None
    assert (
        coupled.history[-1]["reaction_potential_change_ev"]
        == coupled.scf_convergence["final_reaction_potential_change_ev"]
    )
    assert (
        coupled.history[-1]["reaction_gradient_change_ev_per_angstrom"]
        == coupled.scf_convergence[
            "final_reaction_gradient_change_ev_per_angstrom"
        ]
    )
    assert coupled.scf_convergence["final_energy_residual_ev"] == pytest.approx(
        coupled.history[-1]["energy_residual_ev"]
    )
    assert all(
        abs(record["root_total_charge_e"]) <= 1.0e-14
        for record in coupled.history
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
    assert replay["cold_replay_field_arrays_identical"] is True
    assert replay["cold_replay_response_arrays_identical"] is True
    assert replay["maximum_monopole_residual_e"] <= 1.0e-10
    assert replay["maximum_dipole_residual_e_angstrom"] <= 1.0e-10
    assert replay["electrostatic_ledger_span_ev"] <= 1.0e-10
    assert replay["maximum_polarization_identity_error_ev"] <= 1.0e-12
    assert len(replay["field_sha256_by_evaluation"]) == 4
    assert all(len(item) == 64 for item in replay["field_sha256_by_evaluation"])
    assert len(replay["response_sha256_by_evaluation"]) == 4
    assert all(
        len(item) == 64
        for item in replay["response_sha256_by_evaluation"]
    )
    assert replay["maximum_online_to_replay_potential_delta_ev_per_e"] <= 1.0e-10
    assert (
        replay["maximum_online_to_replay_gradient_delta_ev_per_e_angstrom"]
        <= 1.0e-10
    )
    assert (
        replay["maximum_online_to_replay_monopole_response_delta_e"]
        <= 1.0e-12
    )
    assert (
        replay["maximum_online_to_replay_dipole_response_delta_e_angstrom"]
        <= 1.0e-12
    )
    for field in (
        "maximum_online_to_replay_potential_ulp",
        "maximum_online_to_replay_gradient_ulp",
        "maximum_online_to_replay_monopole_ulp",
        "maximum_online_to_replay_dipole_ulp",
    ):
        assert field in replay
        assert isinstance(replay[field], int)
        assert replay[field] >= 0
    assert "all_field_arrays_identical" not in replay
    assert "all_response_arrays_identical" not in replay
    assert "field_sha256" not in replay
    assert "response_sha256" not in replay
    np.testing.assert_allclose(
        coupled.density_coefficients,
        reaction_maps[0].last_density,
    )


def test_engine_rejects_warm_v_cold_map_replay_field_delta_beyond_tolerance(
    monkeypatch,
):
    warm_residual = np.asarray(
        [
            [5.0e-11, 4.0e-11, 0.0, 0.0],
            [-5.0e-11, -4.0e-11, 0.0, 0.0],
        ]
    )
    map_offsets = iter([0.0, 1.1e-10, 1.1e-10, 1.1e-10])

    case = _finite_resolution_case(
        monkeypatch,
        residuals=[warm_residual] * 11,
        policy=_finite_resolution_policy(),
        runtime_identity={"profile": "synthetic-frozen-profile"},
        reaction_field_factory=lambda _atoms: _OffsetReplayFieldMap(
            offset=next(map_offsets),
        ),
    )
    engine, atoms, calculator, gas_state, _, identity = case

    with pytest.raises(
        Route2SCFConvergenceError,
        match="map-replay field potential delta exceeds",
    ):
        engine.solve_coupled_state(
            atoms,
            calculator,
            gas_state,
            provider_cache_signature=("finite-resolution-field-delta",),
            finite_resolution_runtime_identity=identity,
        )


def test_engine_accepts_bounded_warm_v_cold_field_delta_with_repeatable_cold_maps(
    monkeypatch,
):
    warm_residual = np.asarray(
        [
            [5.0e-11, 4.0e-11, 0.0, 0.0],
            [-5.0e-11, -4.0e-11, 0.0, 0.0],
        ]
    )
    potential_offset = 5.0e-11
    offset_array = np.zeros_like(warm_residual)
    offset_array[:, 0] = np.asarray([potential_offset, -potential_offset])
    replay_residual = warm_residual - offset_array
    map_offsets = iter([0.0, potential_offset, potential_offset, potential_offset])

    case = _finite_resolution_case(
        monkeypatch,
        residuals=[warm_residual] * 8 + [replay_residual] * 3,
        policy=_finite_resolution_policy(),
        runtime_identity={"profile": "synthetic-frozen-profile"},
        reaction_field_factory=lambda _atoms: _OffsetReplayFieldMap(
            offset=next(map_offsets),
        ),
    )
    engine, atoms, calculator, gas_state, _, identity = case

    coupled = engine.solve_coupled_state(
        atoms,
        calculator,
        gas_state,
        provider_cache_signature=("finite-resolution-bounded-field-delta",),
        finite_resolution_runtime_identity=identity,
    )

    replay = coupled.scf_convergence["fresh_map_replay"]
    assert replay["cold_replay_field_arrays_identical"] is True
    assert replay["cold_replay_response_arrays_identical"] is True
    assert replay["maximum_online_to_replay_potential_delta_ev_per_e"] == (
        pytest.approx(potential_offset)
    )
    assert replay["maximum_online_to_replay_monopole_response_delta_e"] == (
        pytest.approx(0.0)
    )
    assert len(set(replay["field_sha256_by_evaluation"][1:])) == 1
    assert (
        replay["field_sha256_by_evaluation"][0]
        != replay["field_sha256_by_evaluation"][1]
    )
    assert len(set(replay["response_sha256_by_evaluation"])) == 1


def test_engine_rejects_map_replay_response_delta_beyond_nominal_tolerance(
    monkeypatch,
):
    warm_residual = np.asarray(
        [
            [5.0e-11, 4.0e-11, 0.0, 0.0],
            [-5.0e-11, -4.0e-11, 0.0, 0.0],
        ]
    )
    replay_residual = np.asarray(
        [
            [5.0e-11, 7.0e-12, 0.0, 0.0],
            [-5.0e-11, -7.0e-12, 0.0, 0.0],
        ]
    )

    case = _finite_resolution_case(
        monkeypatch,
        residuals=[warm_residual] * 8 + [replay_residual] * 3,
        policy=_finite_resolution_policy(),
        runtime_identity={"profile": "synthetic-frozen-profile"},
    )
    engine, atoms, calculator, gas_state, _, identity = case

    with pytest.raises(
        Route2SCFConvergenceError,
        match="response dipole delta exceeds nominal dipole tolerance",
    ):
        engine.solve_coupled_state(
            atoms,
            calculator,
            gas_state,
            provider_cache_signature=("finite-resolution-response-delta",),
            finite_resolution_runtime_identity=identity,
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
            [[0.0, 5.0e-13, 0.0, 0.0], [0.0, -5.0e-13, 0.0, 0.0]]
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

    with pytest.raises(
        Route2SCFConvergenceError,
        match="response arrays are not byte-identical",
    ):
        engine.solve_coupled_state(
            atoms,
            calculator,
            gas_state,
            provider_cache_signature=("finite-resolution-replay-mismatch",),
            finite_resolution_runtime_identity=identity,
        )


def test_engine_rejects_nonrepeatable_fresh_map_replay_fields(
    monkeypatch,
):
    warm_residual = np.asarray(
        [
            [5.0e-11, 4.0e-11, 0.0, 0.0],
            [-5.0e-11, -4.0e-11, 0.0, 0.0],
        ]
    )
    cold_offsets = (5.0e-11, 4.0e-11, 5.0e-11)
    replay_residuals = []
    for offset in cold_offsets:
        offset_array = np.zeros_like(warm_residual)
        offset_array[:, 0] = np.asarray([offset, -offset])
        replay_residuals.append(warm_residual - offset_array)
    map_offsets = iter((0.0, *cold_offsets))
    case = _finite_resolution_case(
        monkeypatch,
        residuals=[warm_residual] * 8 + replay_residuals,
        policy=_finite_resolution_policy(),
        runtime_identity={"profile": "synthetic-frozen-profile"},
        reaction_field_factory=lambda _atoms: _OffsetReplayFieldMap(
            offset=next(map_offsets),
        ),
    )
    engine, atoms, calculator, gas_state, _, identity = case

    with pytest.raises(
        Route2SCFConvergenceError,
        match="field arrays are not byte-identical",
    ):
        engine.solve_coupled_state(
            atoms,
            calculator,
            gas_state,
            provider_cache_signature=("finite-resolution-field-mismatch",),
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
        "final_reaction_potential_change_ev": None,
        "final_reaction_gradient_change_ev_per_angstrom": None,
        "final_energy_residual_ev": None,
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


def test_finite_resolution_window_rejects_nonaccepted_or_inconsistent_parentage():
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: _IdentityReactionMap(),
        cds_evaluator=lambda _atoms: _CDS(),
        settings=replace(
            _settings(),
            scf_density_tolerance=1.0,
            scf_dipole_tolerance_e_angstrom=1.0,
            scf_energy_tolerance_ev=1.0,
        ),
    )
    policy = replace(
        _finite_resolution_policy(),
        history_length=3,
        map_replay_count=1,
    )
    density_records = [
        np.asarray([[-0.4, 0.2, -0.1, 0.3], [0.4, -0.2, 0.1, -0.3]]),
        np.asarray(
            [
                [-0.40000000005, 0.20000000005, -0.1, 0.30000000005],
                [0.4, -0.2, 0.10000000005, -0.30000000005],
            ]
        ),
        np.asarray(
            [
                [-0.40000000009, 0.20000000009, -0.10000000009, 0.30000000009],
                [0.40000000009, -0.20000000009, 0.10000000009, -0.30000000009],
            ]
        ),
    ]
    residual_records = [
        np.asarray(
            [
                [1.0e-11, 5.0e-12, -3.0e-12, 4.0e-12],
                [-1.0e-11, -5.0e-12, 3.0e-12, -4.0e-12],
            ]
        ),
        np.asarray(
            [
                [8.0e-12, 4.0e-12, -2.0e-12, 3.0e-12],
                [-8.0e-12, -4.0e-12, 2.0e-12, -3.0e-12],
            ]
        ),
        np.asarray(
            [
                [5.0e-12, 2.0e-12, -1.0e-12, 2.0e-12],
                [-5.0e-12, -2.0e-12, 1.0e-12, -2.0e-12],
            ]
        ),
    ]
    field_records = [
        np.asarray(
            [
                [1.0e-12, 1.0e-12, 1.0e-12, 1.0e-12],
                [1.0e-12, 1.0e-12, 1.0e-12, 1.0e-12],
            ]
        ),
        np.asarray(
            [
                [1.2e-12, 1.0e-12, 1.0e-12, 1.0e-12],
                [1.0e-12, 1.1e-12, 1.0e-12, 1.0e-12],
            ]
        ),
        np.asarray(
            [
                [1.1e-12, 1.2e-12, 1.0e-12, 1.0e-12],
                [1.0e-12, 1.0e-12, 1.1e-12, 1.0e-12],
            ]
        ),
    ]
    energies = [-10.0, -10.0, -10.0]
    base_records: list[Route2SCFHistoryRecord] = []
    for iteration, (density, residual, field, energy) in enumerate(
        zip(density_records, residual_records, field_records, energies),
        start=1,
    ):
        base_records.append(
            {
                "iteration": iteration,
                "density_residual_e": 0.1,
                "monopole_residual_e": 0.1,
                "dipole_residual_e_angstrom": 0.02,
                "energy_residual_ev": 0.0,
                "intrinsic_energy_ev": energy,
                "root_total_charge_e": 0.0,
                "raw_response_total_charge_e": 0.0,
                "response_charge_projection_max_e": 0.0,
                "arrived_by": SAFEGUARDED_ANDERSON_SOLVER,
                "anderson_history_reset": False,
                "attempt_status": "accepted",
                "accepted": True,
                "rejected": False,
                "actual_residual_objective": 0.0,
                "actual_residual_growth_baseline_objective": 0.0,
                "actual_residual_growth_ratio": None,
                "accepted_parent_attempt": 41 if iteration == 1 else iteration - 1,
                "accepted_state_index": iteration - 1,
                "solver_epoch": 0,
                "rollback_anchor_attempt": None,
                "next_density_update": SAFEGUARDED_ANDERSON_SOLVER,
                "fixed_point_history_size": 2,
                "anderson_predicted_residual_l2": None,
                "anderson_coefficient_l1": None,
                "anderson_step_ratio_to_picard": None,
                "anderson_fallback_reason": None,
                "density_sha256": "ignored",
                "response_sha256": "ignored",
                "field_sha256": "ignored",
            }
        )
    assert (
        engine._finite_resolution_candidate_window(
            policy=policy,
            iteration_history=base_records,
            density_history=density_records,
            residual_history=residual_records,
            field_history=field_records,
            intrinsic_energy_history=energies,
        )
        is not None
    )

    inconsistent_records = deepcopy(base_records)
    inconsistent_records[1]["accepted"] = False
    assert (
        engine._finite_resolution_candidate_window(
            policy=policy,
            iteration_history=inconsistent_records,
            density_history=density_records,
            residual_history=residual_records,
            field_history=field_records,
            intrinsic_energy_history=energies,
        )
        is None
    )

    inconsistent_parentage = deepcopy(base_records)
    inconsistent_parentage[2]["accepted_parent_attempt"] = 1
    assert (
        engine._finite_resolution_candidate_window(
            policy=policy,
            iteration_history=inconsistent_parentage,
            density_history=density_records,
            residual_history=residual_records,
            field_history=field_records,
            intrinsic_energy_history=energies,
        )
        is None
    )

    inconsistent_epoch = deepcopy(base_records)
    inconsistent_epoch[1]["solver_epoch"] = 1
    assert (
        engine._finite_resolution_candidate_window(
            policy=policy,
            iteration_history=inconsistent_epoch,
            density_history=density_records,
            residual_history=residual_records,
            field_history=field_records,
            intrinsic_energy_history=energies,
        )
        is None
    )
