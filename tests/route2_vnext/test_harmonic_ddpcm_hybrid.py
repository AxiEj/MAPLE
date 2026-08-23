from __future__ import annotations

import hashlib

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.continuum.harmonic_ddpcm_functional import (
    SmoothPartitionHarmonicDDPCMFunctionalCandidate,
)
from maple.solvation.continuum.harmonic_ddpcm_hybrid import (
    HarmonicDDPCMHybridSnapshot,
    build_harmonic_ddpcm_hybrid_snapshot,
)
from maple.solvation.continuum.separated_source_ddx import (
    SeparatedSourceDDXBackend,
    embed_atomic_l1_in_first_radial_channel,
)
from maple.solvation.coupling.fixed_point import FixedPointOptions
from maple.solvation.coupling.permanent_induced_ledgers import (
    HybridHarmonicDDPCMPhi0Ledger,
)
from maple.solvation.coupling.permanent_induced_state import (
    PermanentInducedContinuumProvider,
    PermanentInducedOperationalStateEquation,
)
from maple.solvation.coupling.separated_fixed_point import solve_separated_fixed_point
from maple.solvation.coupling.separated_operators import (
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
)
from maple.solvation.coupling.spaces import (
    ATOMIC_L1_SOURCE_SPACE,
    AffineChargeCoordinates,
)
from maple.solvation.coupling.state_equation import provider_behavior_sha256
from maple.solvation.experimental.harmonic_ddpcm_operational import (
    MACE_MDPPolarGeneralSourceHarmonicDDPCMBuilder,
)
from maple.solvation.models.mace_mdp_polar_hybrid import (
    PermanentAnchoredInducedSourceModel,
)


POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.927297, 0.0]]
)
# The SMD Coulomb radii leave all three overlapping sphere charts exposed.
# This is the geometry on which the old full-sphere charge reconstruction
# failed and therefore guards the general-source primal/adjoint formulation.
RADII = (1.52, 1.2, 1.2)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _atoms() -> Atoms:
    atoms = Atoms("OH2", positions=POSITIONS)
    atoms.info.update(charge=0, multiplicity=1)
    return atoms


class _Permanent:
    provider_id = "test.hybrid-ddpcm.permanent.v1"
    model_profile_id = "test.hybrid-ddpcm.permanent-profile.v1"
    source_space = ATOMIC_L1_SOURCE_SPACE

    def __init__(self) -> None:
        generator = np.random.default_rng(20260820)
        jacobian = generator.normal(scale=2.0e-3, size=(12, 9))
        jacobian[[0, 4, 8]] -= np.mean(jacobian[[0, 4, 8]], axis=0)
        self.jacobian = jacobian
        self.base = np.asarray(
            [
                [-0.7, 0.04, -0.02, 0.03],
                [0.35, 0.0, 0.01, -0.02],
                [0.35, -0.01, 0.0, 0.02],
            ]
        )

    def configuration_sha256(self) -> str:
        return _digest(repr((self.jacobian.tolist(), self.base.tolist())))

    def evaluate_source(self, geometry: object) -> np.ndarray:
        positions = np.asarray(geometry.positions, dtype=float).reshape(-1)
        return self.base + (self.jacobian @ positions).reshape(3, 4)

    def source_position_vjp(
        self, geometry: object, source_cotangent: object
    ) -> np.ndarray:
        del geometry
        return (self.jacobian.T @ np.asarray(source_cotangent).reshape(-1)).reshape(
            3, 3
        )


class _Responsive:
    provider_id = "test.hybrid-ddpcm.response.v1"
    model_profile_id = "test.hybrid-ddpcm.response-profile.v1"
    provenance_sha256 = _digest("test-hybrid-ddpcm-response-provenance")
    source_space = ATOMIC_L1_SOURCE_SPACE
    receiver_space = MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE
    long_range_evaluator_profile = "test-analytic-gaussian-multipole-v1"

    def __init__(self) -> None:
        generator = np.random.default_rng(20260821)
        self.jacobian = generator.normal(scale=2.0e-3, size=(12, 24))
        self.jacobian[[0, 4, 8]] -= np.mean(
            self.jacobian[[0, 4, 8]], axis=0
        )
        self.zero = np.asarray(
            [[-0.2, 0.01, 0.02, 0.0], [0.1, 0.0, -0.01, 0.0], [0.1, 0.0, 0.0, 0.01]]
        )

    def configuration_sha256(self) -> str:
        return _digest(repr((self.jacobian.tolist(), self.zero.tolist())))

    def vacuum_energy_ev(self, geometry: object) -> float:
        positions = np.asarray(geometry.positions, dtype=float)
        return 1.5 + 0.015 * float(np.vdot(positions, positions))

    def vacuum_forces_ev_per_angstrom(self, geometry: object) -> np.ndarray:
        return -0.03 * np.asarray(geometry.positions, dtype=float)

    def evaluate_source(self, geometry: object, field: object) -> np.ndarray:
        del geometry
        return self.zero + (
            self.jacobian @ np.asarray(field, dtype=float).reshape(-1)
        ).reshape(3, 4)

    def field_jvp(
        self, geometry: object, field: object, direction: object
    ) -> np.ndarray:
        del geometry, field
        return (
            self.jacobian @ np.asarray(direction, dtype=float).reshape(-1)
        ).reshape(3, 4)

    def field_vjp(
        self, geometry: object, field: object, cotangent: object
    ) -> np.ndarray:
        del geometry, field
        return (
            self.jacobian.T @ np.asarray(cotangent, dtype=float).reshape(-1)
        ).reshape(3, 8)

    def coordinate_vjp(
        self, geometry: object, field: object, source_cotangent: object
    ) -> np.ndarray:
        del field, source_cotangent
        return np.zeros((len(geometry), 3))


def _functional(
    *, lmax: int = 2, quadrature_order: int = 80
) -> SmoothPartitionHarmonicDDPCMFunctionalCandidate:
    torch = pytest.importorskip("torch")
    return SmoothPartitionHarmonicDDPCMFunctionalCandidate(
        atomic_numbers=(8, 1, 1),
        radii_angstrom=RADII,
        transition_width_angstrom2=0.08,
        surface_lmax=lmax,
        partition_lmax=2 * lmax,
        partition_radial_quadrature_order=quadrature_order,
        source_radial_quadrature_order=quadrature_order,
        double_layer_radial_quadrature_order=quadrature_order,
        dielectric=80.0,
        dtype=torch.float64,
        device="cpu",
    )


def _hybrid() -> PermanentAnchoredInducedSourceModel:
    return PermanentAnchoredInducedSourceModel(_Permanent(), _Responsive())


def test_hybrid_builder_behavior_identity_is_stable_after_real_build():
    builder = MACE_MDPPolarGeneralSourceHarmonicDDPCMBuilder(
        hybrid=_hybrid(),
        continuum=_functional(lmax=1, quadrature_order=24),
        receiver_radial_quadrature_order=24,
    )
    before = provider_behavior_sha256(
        builder,
        ("configuration_sha256", "build"),
        label="hybrid_operational_builder_regression",
    )
    bundle = builder.build(_atoms())
    assert bundle.scalar_id == builder.scalar_id
    after = provider_behavior_sha256(
        builder,
        ("configuration_sha256", "build"),
        label="hybrid_operational_builder_regression",
    )
    assert after == before


def test_hybrid_general_source_actions_and_coordinate_vjps():
    atoms = _atoms()
    functional = _functional()
    snapshot = build_harmonic_ddpcm_hybrid_snapshot(
        functional, atoms, receiver_radial_quadrature_order=80
    )
    assert isinstance(snapshot, HarmonicDDPCMHybridSnapshot)
    assert isinstance(snapshot, PermanentInducedContinuumProvider)
    assert np.isfinite(snapshot.general_source_raw_asymmetry_relative_defect)
    assert snapshot.boundary_dimension == 2 * snapshot.forward_boundary_dimension
    np.testing.assert_array_equal(
        snapshot.permanent_field_map, snapshot.induced_field_map
    )

    hybrid = _hybrid()
    permanent = hybrid.prepare(atoms).permanent_source4
    generator = np.random.default_rng(20260822)
    induced = generator.normal(scale=4.0e-3, size=(3, 4))
    induced[:, 0] -= np.mean(induced[:, 0])
    direction = generator.normal(size=(3, 4))
    direction[:, 0] -= np.mean(direction[:, 0])
    field_cotangent = generator.normal(size=(3, 8))

    _permanent_vjp, induced_vjp = snapshot.model_field_vjps(field_cotangent)
    assert np.vdot(field_cotangent, snapshot.induced_field_jvp(direction)) == (
        pytest.approx(np.vdot(induced_vjp, direction), abs=2.0e-12)
    )
    permanent_gradient, induced_gradient = snapshot.continuum_source_gradients(
        permanent, induced
    )
    step = 2.0e-6
    induced_fd = (
        snapshot.continuum_energy_eV(permanent, induced + step * direction)
        - snapshot.continuum_energy_eV(permanent, induced - step * direction)
    ) / (2.0 * step)
    assert np.vdot(induced_gradient, direction) == pytest.approx(
        induced_fd, abs=2.0e-10
    )
    permanent_direction = generator.normal(size=(3, 4))
    permanent_fd = (
        snapshot.continuum_energy_eV(
            permanent + step * permanent_direction, induced
        )
        - snapshot.continuum_energy_eV(
            permanent - step * permanent_direction, induced
        )
    ) / (2.0 * step)
    assert np.vdot(permanent_gradient, permanent_direction) == pytest.approx(
        permanent_fd, abs=2.0e-10
    )

    coordinate_direction = generator.normal(size=(3, 3))
    coordinate_direction -= np.mean(coordinate_direction, axis=0, keepdims=True)
    coordinate_direction /= np.linalg.norm(coordinate_direction)
    energy_position = snapshot.continuum_energy_position_gradient(
        permanent, induced
    )
    field_position = snapshot.model_field_position_vjp(
        permanent, induced, field_cotangent
    )
    coordinate_step = 2.0e-5
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions += coordinate_step * coordinate_direction
    minus.positions -= coordinate_step * coordinate_direction
    plus_snapshot = build_harmonic_ddpcm_hybrid_snapshot(
        functional, plus, receiver_radial_quadrature_order=80
    )
    minus_snapshot = build_harmonic_ddpcm_hybrid_snapshot(
        functional, minus, receiver_radial_quadrature_order=80
    )
    energy_fd = (
        plus_snapshot.continuum_energy_eV(permanent, induced)
        - minus_snapshot.continuum_energy_eV(permanent, induced)
    ) / (2.0 * coordinate_step)
    field_fd = np.vdot(
        field_cotangent,
        (
            plus_snapshot.native_field(permanent, induced)
            - minus_snapshot.native_field(permanent, induced)
        )
        / (2.0 * coordinate_step),
    )
    assert np.vdot(energy_position, coordinate_direction) == pytest.approx(
        energy_fd, abs=3.0e-8
    )
    assert np.vdot(field_position, coordinate_direction) == pytest.approx(
        field_fd, abs=4.0e-8
    )


def test_overlapping_smd_cavity_converges_to_pyddx_general_source_reference():
    pytest.importorskip("pyddx")
    atoms = _atoms()
    hybrid = _hybrid()
    permanent = hybrid.prepare(atoms).permanent_source4
    induced = np.asarray(
        [
            [0.008, 0.002, -0.001, 0.001],
            [-0.004, -0.001, 0.0005, 0.0],
            [-0.004, 0.0005, 0.0005, -0.001],
        ]
    )
    snapshot = build_harmonic_ddpcm_hybrid_snapshot(
        _functional(lmax=3, quadrature_order=96),
        atoms,
        receiver_radial_quadrature_order=96,
    )
    reference_backend = SeparatedSourceDDXBackend(
        atoms.get_chemical_symbols(),
        RADII,
        continuum_model="pcm",
        dielectric=80.0,
        lmax=15,
        n_lebedev=1202,
        solver_tolerance=1.0e-12,
    )
    reference = reference_backend.prepare(atoms, permanent).solve(
        embed_atomic_l1_in_first_radial_channel(induced)
    )
    harmonic_energy = snapshot.continuum_energy_eV(permanent, induced)
    harmonic_field = snapshot.native_field(permanent, induced)

    # The two smooth cavity partitions are not pointwise identical, so this
    # is a continuum-limit fidelity check rather than a bitwise parity test.
    assert harmonic_energy == pytest.approx(
        reference.polarization_energy_ev, abs=3.0e-3
    )
    assert np.linalg.norm(harmonic_field - reference.model_field) / max(
        np.linalg.norm(reference.model_field), 1.0e-15
    ) < 2.0e-2


def test_hybrid_complete_implicit_phi0_gradient_matches_resolved_energy_fd():
    atoms = _atoms()
    functional = _functional()
    hybrid = _hybrid()
    coordinates = AffineChargeCoordinates(
        atom_count=3,
        total_charge=0.0,
        source_space=ATOMIC_L1_SOURCE_SPACE,
    )
    options = FixedPointOptions(
        method="anderson",
        tolerance=1.0e-12,
        max_iterations=80,
        damping=0.8,
        history=6,
    )

    def solve(geometry: Atoms, *, context: str):
        anchor = hybrid.prepare(geometry)
        continuum = build_harmonic_ddpcm_hybrid_snapshot(
            functional, geometry, receiver_radial_quadrature_order=80
        )
        equation = PermanentInducedOperationalStateEquation(
            coordinates, hybrid, anchor, continuum
        )
        state = solve_separated_fixed_point(
            equation,
            geometry,
            root_context_id=context,
            options=options,
        )
        ledger = HybridHarmonicDDPCMPhi0Ledger(equation)
        energy = ledger.evaluate_root(
            geometry,
            state.y_array(),
            root_tolerance=options.tolerance,
        ).total_energy_eV
        return ledger, state, energy

    ledger, state, _energy = solve(atoms, context="hybrid-ddpcm-gradient-base")
    gradient = ledger.implicit_gradient(atoms, state)
    assert gradient.adjoint.converged is True
    generator = np.random.default_rng(20260823)
    direction = generator.normal(size=(3, 3))
    direction -= np.mean(direction, axis=0, keepdims=True)
    direction /= np.linalg.norm(direction)
    step = 2.0e-5
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions += step * direction
    minus.positions -= step * direction
    _plus_ledger, _plus_state, plus_energy = solve(
        plus, context="hybrid-ddpcm-gradient-plus"
    )
    _minus_ledger, _minus_state, minus_energy = solve(
        minus, context="hybrid-ddpcm-gradient-minus"
    )
    finite_difference = (plus_energy - minus_energy) / (2.0 * step)
    analytic = float(np.vdot(gradient.gradient_array(), direction))
    assert analytic == pytest.approx(finite_difference, abs=8.0e-8)
