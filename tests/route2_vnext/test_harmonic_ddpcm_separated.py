from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest
from ase import Atoms

from maple.solvation.api.scalar_registry import (
    OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_DDPCM_V1,
)
from maple.solvation.continuum.harmonic_ddpcm_functional import (
    SmoothPartitionHarmonicDDPCMFunctionalCandidate,
)
from maple.solvation.continuum.harmonic_ddpcm_separated import (
    HarmonicDDPCMSeparatedSnapshot,
    build_mace_polar_point_harmonic_ddpcm_separated_snapshot,
)
from maple.solvation.continuum.separated_source_ddx import (
    SeparatedSourceDDXBackend,
)
from maple.solvation.coupling.separated_operators import (
    DifferentiableSeparatedContinuumProvider,
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
    SeparatedContinuumProvider,
)
from maple.solvation.coupling.fixed_point import FixedPointOptions
from maple.solvation.coupling.adjoint import solve_reduced_adjoint
from maple.solvation.coupling.linearization import (
    ReducedLinearizationOperator,
    SeparatedReducedLinearization,
)
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING
from maple.solvation.coupling.separated_ledgers import (
    HarmonicDDPCMFrozenVacuumLedger,
)
from maple.solvation.coupling.separated_fixed_point import (
    separated_roots_numerically_equivalent,
    solve_separated_fixed_point,
)
from maple.solvation.coupling.separated_state import (
    SeparatedOperationalStateEquation,
)
from maple.solvation.coupling.spaces import (
    ATOMIC_L1_SOURCE_SPACE,
    AffineChargeCoordinates,
)
from maple.solvation.release.field_semantics import FieldSemanticsManifest


WATER_POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.927297, 0.0]]
)
WATER_RADII = (2.294, 1.2, 1.2)
WATER_SOURCE = np.asarray(
    [
        [-0.7, 0.04, -0.02, 0.03],
        [0.35, 0.0, 0.01, -0.02],
        [0.35, -0.01, 0.0, 0.02],
    ]
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _water() -> Atoms:
    return Atoms("OH2", positions=WATER_POSITIONS)


def _candidate(*, lmax: int) -> SmoothPartitionHarmonicDDPCMFunctionalCandidate:
    torch = pytest.importorskip("torch")
    return SmoothPartitionHarmonicDDPCMFunctionalCandidate(
        atomic_numbers=(8, 1, 1),
        radii_angstrom=WATER_RADII,
        transition_width_angstrom2=0.08,
        surface_lmax=lmax,
        partition_lmax=2 * lmax,
        partition_radial_quadrature_order=96,
        source_radial_quadrature_order=128,
        double_layer_radial_quadrature_order=128,
        dielectric=80.0,
        dtype=torch.float64,
        device="cpu",
    )


class _LinearElectronicResponse:
    provider_id = "test.linear-separated-electronic.v1"
    model_profile_id = "test.linear-separated-model.v1"
    provenance_sha256 = _digest("linear-separated-electronic-provenance")
    checkpoint_sha256 = _digest("linear-separated-checkpoint")
    field_energy_pairing_sha256 = MACE_POLAR_RADIAL_GTO_PAIRING.metadata_hash()
    source_space = ATOMIC_L1_SOURCE_SPACE
    receiver_space = MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE

    def __init__(self, coupling_id: str) -> None:
        self.coupling_id = coupling_id
        generator = np.random.default_rng(20260816)
        self._jacobian = generator.normal(size=(12, 24)) * 2.0e-3

    def configuration_sha256(self) -> str:
        return _digest(repr(self._jacobian.tolist()))

    def evaluate_source(self, geometry, field):
        del geometry
        value = WATER_SOURCE.reshape(-1) + self._jacobian @ np.asarray(field).reshape(-1)
        return value.reshape(3, 4)

    def evaluate_energy(self, geometry):
        del geometry
        return 1.25

    def coordinate_gradient(self, geometry):
        positions = np.asarray(geometry.positions, dtype=float)
        return np.zeros_like(positions)

    def field_jvp(self, geometry, field, field_direction):
        del geometry, field
        return (self._jacobian @ np.asarray(field_direction).reshape(-1)).reshape(3, 4)

    def field_vjp(self, geometry, field, source_cotangent):
        del geometry, field
        return (self._jacobian.T @ np.asarray(source_cotangent).reshape(-1)).reshape(3, 8)

    def coordinate_vjp(self, geometry, field, source_cotangent):
        del geometry, field, source_cotangent
        return np.zeros((3, 3))


def _field_semantics(electronic: _LinearElectronicResponse) -> FieldSemanticsManifest:
    return FieldSemanticsManifest(
        checkpoint_sha256=electronic.checkpoint_sha256,
        adapter_configuration_sha256=electronic.configuration_sha256(),
        adapter_provenance_sha256=electronic.provenance_sha256,
        model_provider_id=electronic.provider_id,
        model_profile_id=electronic.model_profile_id,
        source_space_sha256=electronic.source_space.metadata_hash(),
        native_field_space_sha256=electronic.receiver_space.metadata_hash(),
        pairing_metric_sha256=MACE_POLAR_RADIAL_GTO_PAIRING.metadata_hash(),
        field_channel_order=electronic.receiver_space.components,
        field_radial_widths_angstrom=(1.5, 3.0),
        real_ylm_convention="test raw real-l1 convention",
        cartesian_spherical_l1_transform="test l1 transform",
        field_units=electronic.receiver_space.units,
        energy_unit="eV",
        external_potential_sign=None,
        uniform_field_sign=None,
        spin_channel_factor=None,
        native_injection_explicit_work_included=None,
        upstream_uniform_explicit_work_included=None,
        origin_convention="test fixed origin",
        evidence_measurement_sha256s=(),
    )


def test_separated_snapshot_preserves_scalar_and_rectangular_adjoint_contract():
    atoms = _water()
    functional = _candidate(lmax=2)
    snapshot = build_mace_polar_point_harmonic_ddpcm_separated_snapshot(
        functional, atoms
    )
    assert isinstance(snapshot, HarmonicDDPCMSeparatedSnapshot)
    assert isinstance(snapshot, SeparatedContinuumProvider)
    assert snapshot.source_dimension == 12
    assert snapshot.receiver_dimension == 24
    assert snapshot.source_to_native_field.shape == (24, 12)
    assert snapshot.point_receiver_reconstruction_max_abs_error < 3.0e-11
    assert (
        snapshot.scalar_id
        == OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_DDPCM_V1
    )
    assert snapshot.continuum_energy_eV(WATER_SOURCE) == pytest.approx(
        functional.energy_eV(atoms, WATER_SOURCE), rel=0.0, abs=4.0e-13
    )

    generator = np.random.default_rng(18)
    direction = generator.normal(size=(3, 4))
    cotangent = generator.normal(size=(3, 8))
    jvp = snapshot.source_field_jvp(direction)
    vjp = snapshot.source_field_vjp(cotangent)
    assert float(np.vdot(cotangent, jvp)) == pytest.approx(
        float(np.vdot(vjp, direction)), rel=0.0, abs=3.0e-13
    )

    step = 2.0e-6
    finite_difference = (
        snapshot.native_field(WATER_SOURCE + step * direction)
        - snapshot.native_field(WATER_SOURCE - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(jvp, finite_difference, rtol=0.0, atol=5.0e-10)

    coordinates = AffineChargeCoordinates(
        atom_count=3,
        total_charge=0.0,
        source_space=ATOMIC_L1_SOURCE_SPACE,
    )
    electronic = _LinearElectronicResponse(snapshot.coupling_id)
    equation = SeparatedOperationalStateEquation(
        coordinates, electronic, snapshot
    )
    y = coordinates.reduce(WATER_SOURCE)
    reduced_direction = generator.normal(size=equation.reduced_dimension)
    reduced_cotangent = generator.normal(size=equation.reduced_dimension)
    residual_jvp = equation.residual_jvp(atoms, y, reduced_direction)
    residual_vjp = equation.residual_vjp(atoms, y, reduced_cotangent)
    assert float(np.vdot(reduced_cotangent, residual_jvp)) == pytest.approx(
        float(np.vdot(residual_vjp, reduced_direction)),
        rel=0.0,
        abs=2.0e-12,
    )

    options = FixedPointOptions(
        method="anderson",
        tolerance=1.0e-12,
        max_iterations=80,
        damping=0.8,
        history=6,
    )
    cold = solve_separated_fixed_point(
        equation,
        atoms,
        root_context_id="harmonic-ddpcm-separated-unit-root",
        options=options,
    )
    warm = solve_separated_fixed_point(
        equation,
        atoms,
        root_context_id="harmonic-ddpcm-separated-unit-root",
        initial_y=cold.y_array() + 0.02 * generator.normal(size=cold.y_array().shape),
        options=options,
    )
    assert cold.converged is warm.converged is True
    assert cold.source_array().shape == (3, 4)
    assert cold.field_array().shape == (3, 8)
    assert cold.boundary_state_array().shape == (snapshot.boundary_dimension,)
    assert cold.root_hash != warm.root_hash
    assert separated_roots_numerically_equivalent(cold, warm)
    forged_field = cold.field_array().copy()
    forged_field[0, 0] += 1.0e-6
    with pytest.raises(ValueError, match="root_hash"):
        replace(
            cold,
            field=tuple(tuple(float(value) for value in row) for row in forged_field),
        )

    assert len(snapshot.configuration_sha256()) == 64
    assert len(snapshot.provenance_sha256) == 64
    assert snapshot.capabilities == ()
    with pytest.raises(AttributeError, match="immutable"):
        snapshot.scalar_id = "forged"


def test_ddpcm_phi0_ledger_gradient_and_separated_implicit_adjoint():
    atoms = _water()
    snapshot = build_mace_polar_point_harmonic_ddpcm_separated_snapshot(
        _candidate(lmax=2), atoms
    )
    electronic = _LinearElectronicResponse(snapshot.coupling_id)
    coordinates = AffineChargeCoordinates(
        atom_count=3,
        total_charge=0.0,
        source_space=ATOMIC_L1_SOURCE_SPACE,
    )
    equation = SeparatedOperationalStateEquation(
        coordinates, electronic, snapshot
    )
    options = FixedPointOptions(
        method="anderson",
        tolerance=1.0e-12,
        max_iterations=80,
        damping=0.8,
        history=6,
    )
    root = solve_separated_fixed_point(
        equation,
        atoms,
        root_context_id="harmonic-ddpcm-phi0-adjoint-unit-root",
        options=options,
    )
    ledger = HarmonicDDPCMFrozenVacuumLedger(
        equation=equation,
        vacuum=electronic,
        field_semantics_manifest=_field_semantics(electronic),
    )
    evaluation = ledger.evaluate_root(
        atoms, root.y_array(), root_tolerance=options.tolerance
    )
    assert dict(evaluation.components_eV) == pytest.approx(
        {
            "macepolar_vacuum_energy": 1.25,
            "smooth_harmonic_ddpcm_stationary_energy": (
                snapshot.continuum_energy_eV(root.source_array())
            ),
        },
        rel=0.0,
        abs=2.0e-13,
    )

    reduced_gradient = ledger.reduced_gradient(atoms, root.y_array())
    generator = np.random.default_rng(20260817)
    direction = generator.normal(size=equation.reduced_dimension)
    step = 1.0e-6
    finite_difference = (
        snapshot.continuum_energy_eV(
            equation.source(root.y_array() + step * direction)
        )
        - snapshot.continuum_energy_eV(
            equation.source(root.y_array() - step * direction)
        )
    ) / (2.0 * step)
    assert float(np.vdot(reduced_gradient, direction)) == pytest.approx(
        finite_difference, rel=0.0, abs=2.0e-10
    )

    linearization = SeparatedReducedLinearization.at(
        equation, atoms, root.y_array()
    )
    assert isinstance(linearization, ReducedLinearizationOperator)
    adjoint = solve_reduced_adjoint(linearization, reduced_gradient)
    assert adjoint.converged is True
    dense_reference = np.linalg.solve(
        linearization.dense_jacobian().T, reduced_gradient
    )
    np.testing.assert_allclose(
        adjoint.solution_array(), dense_reference, rtol=0.0, atol=2.0e-13
    )


def test_checkpoint_native_receiver_matches_real_pyddx_ddpcm():
    pytest.importorskip("pyddx")
    atoms = _water()
    snapshot = build_mace_polar_point_harmonic_ddpcm_separated_snapshot(
        _candidate(lmax=5), atoms
    )
    reference = SeparatedSourceDDXBackend(
        atoms.get_chemical_symbols(),
        WATER_RADII,
        continuum_model="pcm",
        dielectric=80.0,
        lmax=15,
        n_lebedev=1202,
        solver_tolerance=1.0e-12,
        eta=0.1,
    ).prepare(atoms, WATER_SOURCE).solve(np.zeros((len(atoms), 8)))
    field = snapshot.native_field(WATER_SOURCE)
    relative_error = float(
        np.linalg.norm(field - reference.model_field)
        / np.linalg.norm(reference.model_field)
    )
    assert relative_error < 4.0e-7
    assert np.max(np.abs(field - reference.model_field)) < 2.0e-7


def test_ddpcm_energy_and_native_field_coordinate_pullbacks_match_fd():
    atoms = _water()
    functional = _candidate(lmax=2)
    snapshot = build_mace_polar_point_harmonic_ddpcm_separated_snapshot(
        functional, atoms
    )
    assert isinstance(snapshot, DifferentiableSeparatedContinuumProvider)
    generator = np.random.default_rng(20260818)
    coordinate_direction = generator.normal(size=(3, 3))
    source = WATER_SOURCE + 0.01 * generator.normal(size=WATER_SOURCE.shape)
    field_cotangent = generator.normal(size=(3, 8))

    energy_gradient = snapshot.continuum_energy_position_gradient(source)
    field_gradient = snapshot.source_field_position_vjp(
        source, field_cotangent
    )
    np.testing.assert_allclose(
        energy_gradient,
        functional.coordinate_partial(atoms, source),
        rtol=0.0,
        atol=3.0e-11,
    )

    step = 2.0e-5
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions += step * coordinate_direction
    minus.positions -= step * coordinate_direction
    plus_snapshot = build_mace_polar_point_harmonic_ddpcm_separated_snapshot(
        functional, plus
    )
    minus_snapshot = build_mace_polar_point_harmonic_ddpcm_separated_snapshot(
        functional, minus
    )
    energy_fd = (
        plus_snapshot.continuum_energy_eV(source)
        - minus_snapshot.continuum_energy_eV(source)
    ) / (2.0 * step)
    field_fd = (
        float(np.vdot(field_cotangent, plus_snapshot.native_field(source)))
        - float(np.vdot(field_cotangent, minus_snapshot.native_field(source)))
    ) / (2.0 * step)
    assert float(np.vdot(energy_gradient, coordinate_direction)) == pytest.approx(
        energy_fd, rel=0.0, abs=2.0e-8
    )
    assert float(np.vdot(field_gradient, coordinate_direction)) == pytest.approx(
        field_fd, rel=0.0, abs=3.0e-8
    )
    np.testing.assert_allclose(
        np.sum(energy_gradient, axis=0), np.zeros(3), rtol=0.0, atol=2.0e-11
    )
    np.testing.assert_allclose(
        np.sum(field_gradient, axis=0), np.zeros(3), rtol=0.0, atol=2.0e-10
    )


def test_ddpcm_phi0_complete_implicit_gradient_matches_resolved_energy_fd():
    atoms = _water()
    functional = _candidate(lmax=2)
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

    def solve_ledger(geometry: Atoms, *, context: str):
        snapshot = build_mace_polar_point_harmonic_ddpcm_separated_snapshot(
            functional, geometry
        )
        electronic = _LinearElectronicResponse(snapshot.coupling_id)
        equation = SeparatedOperationalStateEquation(
            coordinates, electronic, snapshot
        )
        state = solve_separated_fixed_point(
            equation,
            geometry,
            root_context_id=context,
            options=options,
        )
        ledger = HarmonicDDPCMFrozenVacuumLedger(
            equation=equation,
            vacuum=electronic,
            field_semantics_manifest=_field_semantics(electronic),
        )
        evaluation = ledger.evaluate_root(
            geometry,
            state.y_array(),
            root_tolerance=options.tolerance,
        )
        return ledger, state, evaluation.total_energy_eV

    ledger, state, _energy = solve_ledger(
        atoms, context="harmonic-ddpcm-complete-gradient-base"
    )
    gradient = ledger.implicit_gradient(atoms, state)
    assert gradient.adjoint.converged is True
    np.testing.assert_allclose(
        gradient.forces_array(),
        -gradient.gradient_array(),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.sum(gradient.gradient_array(), axis=0),
        np.zeros(3),
        rtol=0.0,
        atol=4.0e-10,
    )

    generator = np.random.default_rng(20260819)
    direction = generator.normal(size=(3, 3))
    direction -= np.mean(direction, axis=0, keepdims=True)
    direction /= np.linalg.norm(direction)
    step = 2.0e-5
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions += step * direction
    minus.positions -= step * direction
    _plus_ledger, _plus_state, plus_energy = solve_ledger(
        plus, context="harmonic-ddpcm-complete-gradient-plus"
    )
    _minus_ledger, _minus_state, minus_energy = solve_ledger(
        minus, context="harmonic-ddpcm-complete-gradient-minus"
    )
    finite_difference = (plus_energy - minus_energy) / (2.0 * step)
    analytic = float(np.vdot(gradient.gradient_array(), direction))
    assert analytic == pytest.approx(finite_difference, rel=0.0, abs=5.0e-8)
