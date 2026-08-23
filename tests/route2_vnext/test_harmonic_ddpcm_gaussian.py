from __future__ import annotations

import hashlib

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.api.scalar_registry import (
    OPERATIONAL_MACEPOLAR_GTO1P5_NATIVEFIELD8_SMOOTH_HARMONIC_DDPCM_PHI0_V2,
)
from maple.solvation.continuum.harmonic_ddpcm_functional import (
    SmoothPartitionHarmonicDDPCMFunctionalCandidate,
)
from maple.solvation.continuum.harmonic_ddpcm_gaussian import (
    HarmonicDDPCMPureGaussianSnapshot,
    build_mace_polar_gaussian_harmonic_ddpcm_snapshot,
)
from maple.solvation.continuum.harmonic_ddpcm_hybrid import (
    build_harmonic_ddpcm_hybrid_snapshot,
)
from maple.solvation.coupling.fixed_point import FixedPointOptions
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING
from maple.solvation.coupling.separated_fixed_point import solve_separated_fixed_point
from maple.solvation.coupling.separated_ledgers import (
    HarmonicDDPCMFrozenVacuumLedger,
)
from maple.solvation.coupling.separated_operators import (
    DifferentiableSeparatedContinuumProvider,
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
    SeparatedContinuumProvider,
)
from maple.solvation.coupling.separated_state import (
    SeparatedOperationalStateEquation,
)
from maple.solvation.coupling.spaces import (
    ATOMIC_L1_SOURCE_SPACE,
    AffineChargeCoordinates,
)
from maple.solvation.release.field_semantics import FieldSemanticsManifest


POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.927297, 0.0]]
)
RADII = (1.52, 1.2, 1.2)
SOURCE = np.asarray(
    [
        [-0.7, 0.04, -0.02, 0.03],
        [0.35, 0.0, 0.01, -0.02],
        [0.35, -0.01, 0.0, 0.02],
    ]
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _atoms() -> Atoms:
    atoms = Atoms("OH2", positions=POSITIONS)
    atoms.info.update(charge=0, multiplicity=1)
    return atoms


def _functional() -> SmoothPartitionHarmonicDDPCMFunctionalCandidate:
    torch = pytest.importorskip("torch")
    return SmoothPartitionHarmonicDDPCMFunctionalCandidate(
        atomic_numbers=(8, 1, 1),
        radii_angstrom=RADII,
        transition_width_angstrom2=0.08,
        surface_lmax=2,
        partition_lmax=4,
        partition_radial_quadrature_order=64,
        source_radial_quadrature_order=80,
        double_layer_radial_quadrature_order=80,
        dielectric=80.0,
        dtype=torch.float64,
        device="cpu",
    )


class _Electronic:
    provider_id = "test.pure-gaussian-ddpcm.electronic.v1"
    model_profile_id = "test.pure-gaussian-ddpcm.model.v1"
    provenance_sha256 = _digest("pure-gaussian-ddpcm-electronic-provenance")
    checkpoint_sha256 = _digest("pure-gaussian-ddpcm-checkpoint")
    field_energy_pairing_sha256 = MACE_POLAR_RADIAL_GTO_PAIRING.metadata_hash()
    source_space = ATOMIC_L1_SOURCE_SPACE
    receiver_space = MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE

    def __init__(self, coupling_id: str) -> None:
        self.coupling_id = coupling_id
        generator = np.random.default_rng(20260830)
        self._jacobian = generator.normal(scale=1.0e-3, size=(12, 24))

    def configuration_sha256(self) -> str:
        return _digest(repr(self._jacobian.tolist()))

    def evaluate_source(self, geometry, field):
        del geometry
        return (
            SOURCE.reshape(-1) + self._jacobian @ np.asarray(field).reshape(-1)
        ).reshape(3, 4)

    def evaluate_energy(self, geometry):
        positions = np.asarray(geometry.positions, dtype=float)
        return 1.2 + 0.01 * float(np.vdot(positions, positions))

    def coordinate_gradient(self, geometry):
        return 0.02 * np.asarray(geometry.positions, dtype=float)

    def field_jvp(self, geometry, field, field_direction):
        del geometry, field
        return (
            self._jacobian @ np.asarray(field_direction).reshape(-1)
        ).reshape(3, 4)

    def field_vjp(self, geometry, field, source_cotangent):
        del geometry, field
        return (
            self._jacobian.T @ np.asarray(source_cotangent).reshape(-1)
        ).reshape(3, 8)

    def coordinate_vjp(self, geometry, field, source_cotangent):
        del field, source_cotangent
        return np.zeros((len(geometry), 3))


def _manifest(electronic: _Electronic) -> FieldSemanticsManifest:
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


def test_pure_gaussian_view_is_exact_general_source_restriction():
    atoms = _atoms()
    functional = _functional()
    general = build_harmonic_ddpcm_hybrid_snapshot(
        functional, atoms, receiver_radial_quadrature_order=80
    )
    pure = build_mace_polar_gaussian_harmonic_ddpcm_snapshot(
        functional, atoms, receiver_radial_quadrature_order=80
    )
    zero = np.zeros_like(SOURCE)
    assert isinstance(pure, HarmonicDDPCMPureGaussianSnapshot)
    assert isinstance(pure, SeparatedContinuumProvider)
    assert isinstance(pure, DifferentiableSeparatedContinuumProvider)
    assert (
        pure.scalar_id
        == OPERATIONAL_MACEPOLAR_GTO1P5_NATIVEFIELD8_SMOOTH_HARMONIC_DDPCM_PHI0_V2
    )
    np.testing.assert_allclose(
        pure.solve_boundary(SOURCE), general.solve_boundary(zero, SOURCE), atol=0.0
    )
    np.testing.assert_allclose(
        pure.native_field(SOURCE), general.native_field(zero, SOURCE), atol=0.0
    )
    assert pure.continuum_energy_eV(SOURCE) == pytest.approx(
        general.continuum_energy_eV(zero, SOURCE), abs=0.0
    )
    _permanent_gradient, induced_gradient = general.continuum_source_gradients(
        zero, SOURCE
    )
    np.testing.assert_allclose(
        pure.continuum_source_gradient(SOURCE), induced_gradient, atol=0.0
    )

    generator = np.random.default_rng(20260831)
    direction = generator.normal(size=SOURCE.shape)
    cotangent = generator.normal(size=(3, 8))
    assert np.vdot(cotangent, pure.source_field_jvp(direction)) == pytest.approx(
        np.vdot(pure.source_field_vjp(cotangent), direction), abs=2.0e-12
    )
    step = 2.0e-6
    energy_fd = (
        pure.continuum_energy_eV(SOURCE + step * direction)
        - pure.continuum_energy_eV(SOURCE - step * direction)
    ) / (2.0 * step)
    assert np.vdot(pure.continuum_source_gradient(SOURCE), direction) == (
        pytest.approx(energy_fd, abs=2.0e-10)
    )


def test_pure_gaussian_phi0_implicit_force_matches_resolved_scalar_fd():
    atoms = _atoms()
    functional = _functional()
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

    def resolved(geometry: Atoms, *, gradient: bool):
        continuum = build_mace_polar_gaussian_harmonic_ddpcm_snapshot(
            functional, geometry, receiver_radial_quadrature_order=80
        )
        electronic = _Electronic(continuum.coupling_id)
        equation = SeparatedOperationalStateEquation(
            coordinates, electronic, continuum
        )
        root = solve_separated_fixed_point(
            equation,
            geometry,
            root_context_id="pure-gaussian-ddpcm-unit-root",
            options=options,
        )
        ledger = HarmonicDDPCMFrozenVacuumLedger(
            equation=equation,
            vacuum=electronic,
            field_semantics_manifest=_manifest(electronic),
        )
        if gradient:
            return ledger.implicit_gradient(geometry, root)
        return ledger.evaluate_root(
            geometry, root.y_array(), root_tolerance=options.tolerance
        )

    evaluated = resolved(atoms, gradient=True)
    assert evaluated.ledger.scalar_id == (
        OPERATIONAL_MACEPOLAR_GTO1P5_NATIVEFIELD8_SMOOTH_HARMONIC_DDPCM_PHI0_V2
    )
    generator = np.random.default_rng(20260901)
    direction = generator.normal(size=(3, 3))
    direction -= np.mean(direction, axis=0, keepdims=True)
    direction /= np.linalg.norm(direction)
    analytic = float(np.vdot(evaluated.gradient_array(), direction))
    step = 2.0e-5
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions += step * direction
    minus.positions -= step * direction
    finite_difference = (
        resolved(plus, gradient=False).total_energy_eV
        - resolved(minus, gradient=False).total_energy_eV
    ) / (2.0 * step)
    assert analytic == pytest.approx(finite_difference, abs=8.0e-8)
