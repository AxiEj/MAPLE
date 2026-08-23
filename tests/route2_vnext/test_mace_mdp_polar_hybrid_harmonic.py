from __future__ import annotations

from dataclasses import replace

from ase import Atoms
import numpy as np
import pytest
import torch

from maple.solvation.api.profiles import (
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_PROFILE_V1,
    PROFILE_REGISTRY,
)
from maple.solvation.continuum import (
    SmoothWeightedHarmonicGalerkinFunctionalCandidate,
)
from maple.solvation.coupling.separated_operators import (
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
    mace_polar_native_field_convention_contract,
    mace_polar_native_field_convention_contract_sha256,
)
from maple.solvation.coupling.metrics import (
    atomic_l1_source_convention_contract,
    atomic_l1_source_convention_contract_sha256,
)
from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE
from maple.solvation.derivatives import RichardsonScalarForce
from maple.solvation.experimental.mace_mdp_polar_harmonic import (
    MACE_MDPPolarHybridSmoothHarmonicEnergy,
    MACE_MDPPolarHybridSmoothHarmonicPES,
    ROOT_TOLERANCE_EV,
    SCALAR_ID,
)
from maple.solvation.models.mace_mdp_polar_hybrid import (
    PermanentAnchoredInducedSourceModel,
)


class _Permanent:
    provider_id = "test.harmonic-hybrid.permanent.v1"
    model_profile_id = "test.harmonic-hybrid.permanent-profile.v1"
    source_space = ATOMIC_L1_SOURCE_SPACE

    def configuration_sha256(self) -> str:
        return "1" * 64

    def evaluate_source(self, _geometry: object) -> np.ndarray:
        return np.asarray([[-0.15, 0.01, -0.02, 0.03], [0.15, -0.02, 0.01, -0.01]])

    def source_position_vjp(
        self, geometry: object, _source_cotangent: object
    ) -> np.ndarray:
        return np.zeros((len(geometry), 3))


class _Responsive:
    provider_id = "test.harmonic-hybrid.response.v1"
    model_profile_id = "test.harmonic-hybrid.response-profile.v1"
    provenance_sha256 = "2" * 64
    source_space = ATOMIC_L1_SOURCE_SPACE
    receiver_space = MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE
    long_range_evaluator_profile = (
        "graph-longrange-analytic-gaussian-multipole-realspace-v1"
    )

    def __init__(self) -> None:
        rng = np.random.default_rng(41)
        self.jacobian = rng.normal(scale=2.0e-7, size=(8, 16))
        self.jacobian[4] = -self.jacobian[0]
        self.zero = np.asarray([[-0.08, 0.01, 0.02, -0.01], [0.08, -0.01, 0.01, 0.02]])

    def configuration_sha256(self) -> str:
        return "3" * 64

    def vacuum_energy_ev(self, _geometry: object) -> float:
        return -123.456

    def vacuum_forces_ev_per_angstrom(self, geometry: object) -> np.ndarray:
        return np.zeros((len(geometry), 3))

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
        return np.zeros((len(geometry), 3))


def _atoms() -> Atoms:
    atoms = Atoms("CO", positions=[[-0.5, 0.1, 0.0], [0.7, -0.1, 0.2]])
    atoms.info["charge"] = 0
    atoms.info["multiplicity"] = 1
    return atoms


def _hybrid() -> PermanentAnchoredInducedSourceModel:
    return PermanentAnchoredInducedSourceModel(_Permanent(), _Responsive())


def _continuum() -> SmoothWeightedHarmonicGalerkinFunctionalCandidate:
    return SmoothWeightedHarmonicGalerkinFunctionalCandidate(
        atomic_numbers=(6, 8),
        radii_angstrom=(1.8, 1.7),
        transition_width_angstrom2=0.18,
        surface_lmax=1,
        exposure_lmax=2,
        exposure_radial_quadrature_order=40,
        source_radial_quadrature_order=48,
        green_radial_quadrature_order=48,
        dtype=torch.float64,
        device="cpu",
        scalar_id=SCALAR_ID,
    )


def test_harmonic_hybrid_root_scalar_and_fixed_topology_close() -> None:
    atoms = _atoms()
    evaluator = MACE_MDPPolarHybridSmoothHarmonicEnergy(
        atoms, hybrid=_hybrid(), continuum=_continuum()
    )
    state = evaluator.solve(atoms)
    assert state.primal_residual_ev < ROOT_TOLERANCE_EV
    assert state.replay_field_max_abs_difference_ev < 2.0e-9
    assert np.sum(state.induced_source4[:, 0]) == pytest.approx(0.0, abs=1.0e-12)
    assert np.sum(state.total_source4[:, 0]) == pytest.approx(0.0, abs=1.0e-12)
    assert state.polarization_energy_ev == pytest.approx(
        -0.5 * float(np.vdot(state.boundary_rhs, state.boundary_state)),
        abs=2.0e-13,
    )
    moved = atoms.copy()
    moved.positions[0, 0] += 2.5e-4
    moved_state = MACE_MDPPolarHybridSmoothHarmonicEnergy(
        moved, hybrid=_hybrid(), continuum=_continuum()
    ).solve(moved)
    assert moved_state.coefficient_topology_id == state.coefficient_topology_id


def test_solve_with_diagnostics_preserves_legacy_state_and_raw_algebra() -> None:
    atoms = _atoms()
    evaluator = MACE_MDPPolarHybridSmoothHarmonicEnergy(
        atoms, hybrid=_hybrid(), continuum=_continuum()
    )

    legacy = evaluator.solve(atoms)
    state, diagnostics = evaluator.solve_with_diagnostics(atoms)

    assert state.root_sha256 == legacy.root_sha256
    assert state.geometry_sha256 == legacy.geometry_sha256
    assert state.evaluator_configuration_sha256 == legacy.evaluator_configuration_sha256
    assert state.anchor_state_sha256 == legacy.anchor_state_sha256
    assert state.continuum_state_sha256 == legacy.continuum_state_sha256
    assert state.vacuum_energy_ev == legacy.vacuum_energy_ev
    assert state.polarization_energy_ev == legacy.polarization_energy_ev
    assert state.primal_residual_ev == legacy.primal_residual_ev
    assert state.cold_iterations == legacy.cold_iterations
    assert state.wide_iterations == legacy.wide_iterations
    assert np.array_equal(state.native_field_ev, legacy.native_field_ev)
    assert np.array_equal(state.induced_source4, legacy.induced_source4)
    assert np.array_equal(state.total_source4, legacy.total_source4)
    assert np.array_equal(state.boundary_rhs, legacy.boundary_rhs)
    assert np.array_equal(state.boundary_state, legacy.boundary_state)

    cold = diagnostics.cold_start
    wide = diagnostics.wide_start
    assert cold.converged is True
    assert wide.converged is True
    assert cold.iterations == state.cold_iterations
    assert wide.iterations == state.wide_iterations
    assert cold.final_residual_norm_ev == np.linalg.norm(cold.final_residual_ev)
    assert wide.final_residual_norm_ev == np.linalg.norm(wide.final_residual_ev)
    assert cold.final_residual_norm_ev == state.primal_residual_ev
    assert cold.final_polarization_energy_ev == state.polarization_energy_ev
    assert wide.final_polarization_energy_ev == pytest.approx(
        state.polarization_energy_ev, abs=0.0
    )
    assert np.array_equal(cold.final_native_field_ev, state.native_field_ev)
    assert (
        np.max(np.abs(cold.final_native_field_ev - wide.final_native_field_ev)) < 2e-9
    )
    assert cold.initial_state_sha256 != wide.initial_state_sha256

    assert np.array_equal(
        diagnostics.audit_coefficient_sum4,
        diagnostics.permanent_source4 + diagnostics.induced_source4,
    )
    assert np.array_equal(
        diagnostics.response_final_source4,
        diagnostics.response_zero_source4 + diagnostics.induced_source4,
    )
    assert np.array_equal(diagnostics.induced_source4, state.induced_source4)
    assert np.array_equal(diagnostics.audit_coefficient_sum4, state.total_source4)
    assert np.array_equal(
        diagnostics.permanent_source4, evaluator.anchor.permanent_source4
    )
    assert np.array_equal(
        diagnostics.response_zero_source4, evaluator.anchor.response_zero_source4
    )
    assert diagnostics.target_charge_e == pytest.approx(
        np.sum(diagnostics.permanent_source4[:, 0]), abs=0.0
    )
    assert np.sum(diagnostics.audit_coefficient_sum4[:, 0]) == pytest.approx(
        diagnostics.target_charge_e, abs=1.0e-12
    )
    assert diagnostics.source_basis_id == "maple.route2.atomic-l1-source-space.v1"
    assert diagnostics.source_space_contract_sha256 == (
        atomic_l1_source_convention_contract_sha256()
    )
    assert atomic_l1_source_convention_contract()["pairing_metric"][
        "field_to_source_indices"
    ] == [0, 2, 3, 1]
    assert diagnostics.source_component_order == (
        "net_monopole",
        "real_l1_m0",
        "real_l1_m1",
        "real_l1_m_minus1",
    )
    with pytest.raises(ValueError, match="component order drifted"):
        replace(diagnostics, source_component_order=("wrong",) * 4)
    with pytest.raises(AttributeError):
        diagnostics.source_component_order = ("wrong",) * 4
    assert diagnostics.receiver_basis_id == (
        "maple.route2.mace-polar-native-radial-field-space.v1"
    )
    assert diagnostics.receiver_space_contract_sha256 == (
        mace_polar_native_field_convention_contract_sha256()
    )
    assert dict(diagnostics.source_role_identities)["response_zero_source4"] == (
        "zero-field responsive subtraction reference"
    )
    with pytest.raises(ValueError):
        diagnostics.audit_coefficient_sum4[0, 0] = 1.0


def test_diagnostics_require_final_minus_zero_not_inverse_addition() -> None:
    atoms = _atoms()
    _, diagnostics = MACE_MDPPolarHybridSmoothHarmonicEnergy(
        atoms, hybrid=_hybrid(), continuum=_continuum()
    ).solve_with_diagnostics(atoms)
    zero = diagnostics.response_zero_source4.copy()
    final = diagnostics.response_final_source4.copy()
    induced = diagnostics.induced_source4.copy()
    value_zero = 1.799707382720902
    value_induced = 1.1441658720372287
    zero[0, 1] = value_zero
    final[0, 1] = value_zero + value_induced
    induced[0, 1] = value_induced
    audit = diagnostics.permanent_source4 + induced
    assert final[0, 1] - zero[0, 1] != induced[0, 1]
    with pytest.raises(ValueError, match="frozen subtraction"):
        replace(
            diagnostics,
            response_zero_source4=zero,
            response_final_source4=final,
            induced_source4=induced,
            audit_coefficient_sum4=audit,
        )


def test_representation_contract_factories_are_nested_mutation_safe() -> None:
    source_before = atomic_l1_source_convention_contract_sha256()
    source = atomic_l1_source_convention_contract()
    source["pairing_metric"]["field_to_source_indices"][0] = 3
    assert atomic_l1_source_convention_contract_sha256() == source_before
    assert atomic_l1_source_convention_contract()["pairing_metric"][
        "field_to_source_indices"
    ] == [0, 2, 3, 1]

    receiver_before = mace_polar_native_field_convention_contract_sha256()
    receiver = mace_polar_native_field_convention_contract()
    receiver["receiver_space"]["components"][0] = "tampered"
    assert mace_polar_native_field_convention_contract_sha256() == receiver_before
    assert mace_polar_native_field_convention_contract()["receiver_space"][
        "components"
    ][0] == "potential_sigma_1p5"


def test_harmonic_hybrid_numerical_force_converges_without_topology_jump() -> None:
    atoms = _atoms()
    pes = MACE_MDPPolarHybridSmoothHarmonicPES(
        hybrid=_hybrid(),
        atomic_numbers=atoms.numbers,
        cavity_radii_angstrom=(1.8, 1.7),
        dtype=torch.float64,
        device="cpu",
        exposure_radial_quadrature_order=40,
        source_radial_quadrature_order=48,
        green_radial_quadrature_order=48,
        force_backend=RichardsonScalarForce(
            coarse_step_angstrom=5.0e-4,
            maximum_error_eV_per_A=2.0e-4,
        ),
    )
    assert pes.prepared_atomic_numbers == (6, 8)
    assert pes.prepared_cavity_radii_angstrom == (1.8, 1.7)
    assert pes.prepared_charge == 0
    assert pes.prepared_multiplicity == 1
    assert pes.prepared_profile_id == pes.profile_id
    assert pes.prepared_scalar_id == SCALAR_ID
    assert pes.prepared_provider_id == pes.provider_id
    assert pes.prepared_configuration_sha256 == pes.configuration_sha256()
    state = pes.solve(atoms)
    component = pes.numerical_force_component(
        atoms, atom_index=0, axis_index=0, central_state=state
    )
    assert component.error_estimate_eV_per_A < 2.0e-4
    assert pes.sample(atoms).topology_id == state.coefficient_topology_id
    translated = atoms.copy()
    translated.positions += np.asarray([3.1, -2.7, 0.8])
    assert pes.sample(translated).energy_eV == pytest.approx(
        state.total_energy_ev, rel=2.0e-12, abs=2.0e-10
    )


def test_harmonic_hybrid_public_result_fails_closed_while_profile_is_disabled() -> None:
    atoms = _atoms()
    profile = PROFILE_REGISTRY[
        EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_PROFILE_V1
    ]
    assert profile.enabled is False
    assert profile.capabilities.enabled_tiers == ()
    assert profile.evidence_artifact_ids == ()
    pes = MACE_MDPPolarHybridSmoothHarmonicPES(
        hybrid=_hybrid(),
        atomic_numbers=atoms.numbers,
        cavity_radii_angstrom=(1.8, 1.7),
        dtype=torch.float64,
        device="cpu",
        exposure_radial_quadrature_order=40,
        source_radial_quadrature_order=48,
        green_radial_quadrature_order=48,
    )
    with pytest.raises(RuntimeError, match="has not passed admission"):
        pes.evaluate(atoms, need_forces=True)
