from __future__ import annotations

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


def test_harmonic_hybrid_public_capabilities_remain_closed_before_admission() -> None:
    atoms = _atoms()
    profile = PROFILE_REGISTRY[
        EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_PROFILE_V1
    ]
    assert profile.enabled is False
    assert profile.capabilities.enabled_tiers == ()
    pes = MACE_MDPPolarHybridSmoothHarmonicPES(
        hybrid=_hybrid(),
        atomic_numbers=atoms.numbers,
        cavity_radii_angstrom=(1.8, 1.7),
        dtype=torch.float64,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="has not passed admission"):
        pes.evaluate(atoms, need_forces=True)
