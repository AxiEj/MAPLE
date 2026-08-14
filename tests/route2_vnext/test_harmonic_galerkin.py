from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.linalg import expm

from maple.solvation.api import (
    PROFILE_REGISTRY,
    SCALAR_REGISTRY,
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_PROFILE_V1,
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_V1,
)
from maple.solvation.continuum import (
    FixedHarmonicGalerkinCPCMCandidate,
    FixedHarmonicGalerkinSnapshot,
    PerAtomHarmonicSpace,
    radial_gto_source_rotation_matrix,
    real_wigner_generators,
    real_wigner_matrix,
)
from maple.solvation.release.symmetry_panel import rotate_radial_gto_blocks

ROOT = Path(__file__).resolve().parents[2]
POSITIONS = np.asarray([[0.0, 0.0, 0.0], [2.3, -0.4, 0.2]])
SOURCE = np.asarray(
    [
        [0.31, -0.04, 0.02, -0.03, 0.01, 0.005, -0.008, 0.012],
        [-0.29, 0.02, -0.01, 0.04, -0.02, -0.006, 0.009, -0.004],
    ]
)


def _rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    q = q @ np.diag(np.where(np.diag(r) < 0.0, -1.0, 1.0))
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return q


def _snapshot(*, lmax: int = 2) -> FixedHarmonicGalerkinSnapshot:
    space = PerAtomHarmonicSpace(atom_count=2, lmax=lmax)
    rng = np.random.default_rng(20260814 + lmax)
    raw = rng.normal(scale=0.07, size=(space.dimension, space.dimension))
    surface_operator = raw.T @ raw + 1.5 * np.eye(space.dimension)
    source_operator = rng.normal(scale=0.06, size=(space.dimension, 2 * 8))
    return FixedHarmonicGalerkinSnapshot(
        atomic_numbers=(1, 8),
        coefficient_space=space,
        surface_operator=surface_operator,
        source_operator=source_operator,
        cavity_descriptor_sha256="a" * 64,
        assembly_contract_id="test-external-coefficient-intertwiner-v1",
    )


def _functional():
    torch = pytest.importorskip("torch")
    return FixedHarmonicGalerkinCPCMCandidate(
        _snapshot(), dtype=torch.float64, device="cpu"
    )


def test_harmonic_contract_imports_without_torch():
    script = r"""
import sys
sys.modules['torch'] = None
from maple.solvation.continuum.harmonic_galerkin import (
    FixedHarmonicGalerkinSnapshot,
    PerAtomHarmonicSpace,
    real_wigner_matrix,
)
assert PerAtomHarmonicSpace(atom_count=1, lmax=2).dimension == 9
assert callable(real_wigner_matrix)
assert FixedHarmonicGalerkinSnapshot.__name__ == 'FixedHarmonicGalerkinSnapshot'
"""
    result = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_complete_irrep_blocks_have_fixed_dimension_and_order():
    space = PerAtomHarmonicSpace(atom_count=2, lmax=3)
    assert space.dimension == 2 * (3 + 1) ** 2
    assert space.single_atom_dimension == 16
    assert space.labels[:6] == (
        (0, 0, 0),
        (0, 1, -1),
        (0, 1, 0),
        (0, 1, 1),
        (0, 2, -2),
        (0, 2, -1),
    )
    assert space.labels[16] == (1, 0, 0)
    assert len(space.metadata_sha256()) == 64


@pytest.mark.parametrize("lmax", (0, 1, 2, 3, 4))
def test_real_wigner_matrices_are_an_so3_representation(lmax):
    first = _rotation(17)
    second = _rotation(23)
    d_first = real_wigner_matrix(first, lmax=lmax)
    d_second = real_wigner_matrix(second, lmax=lmax)
    d_product = real_wigner_matrix(first @ second, lmax=lmax)
    identity = np.eye((lmax + 1) ** 2)
    np.testing.assert_allclose(d_first @ d_first.T, identity, atol=3e-13, rtol=0.0)
    np.testing.assert_allclose(d_first @ d_second, d_product, atol=6e-13, rtol=0.0)


def test_l1_harmonic_block_matches_the_independent_cartesian_vector_action():
    rotation = _rotation(29)
    l1_block = real_wigner_matrix(rotation, lmax=1)[1:4, 1:4]
    # This real-harmonic convention orders l=1 as (m=-1,0,1)=(y,z,x).
    cartesian_to_harmonic = np.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    )
    expected = cartesian_to_harmonic @ rotation @ cartesian_to_harmonic.T
    np.testing.assert_allclose(l1_block, expected, atol=2e-15, rtol=0.0)


@pytest.mark.parametrize("lmax", (0, 1, 2, 4, 6))
def test_real_wigner_generators_exponentiate_to_the_rotation_action(lmax):
    generators = real_wigner_generators(lmax=lmax)
    angle = 0.137
    rotations = (
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ]
        ),
        np.asarray(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        ),
        np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    dimension = (lmax + 1) ** 2
    for generator, rotation in zip(generators, rotations, strict=True):
        np.testing.assert_allclose(
            generator + generator.T,
            np.zeros((dimension, dimension)),
            atol=2e-15,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            expm(angle * generator),
            real_wigner_matrix(rotation, lmax=lmax),
            atol=2e-13,
            rtol=0.0,
        )
    np.testing.assert_allclose(
        generators[0] @ generators[1] - generators[1] @ generators[0],
        generators[2],
        atol=3e-14,
        rtol=0.0,
    )


def test_radial_source_rotation_matches_authoritative_raw_l1_convention():
    rotation = _rotation(31)
    operator = radial_gto_source_rotation_matrix(rotation, atom_count=2)
    expected = rotate_radial_gto_blocks(SOURCE, rotation)
    actual = (operator @ SOURCE.reshape(-1)).reshape(SOURCE.shape)
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0.0)
    np.testing.assert_allclose(
        operator @ operator.T, np.eye(SOURCE.size), atol=2e-15, rtol=0.0
    )


def test_snapshot_is_immutable_content_addressed_and_spd():
    snapshot = _snapshot()
    assert snapshot.coefficient_space.dimension == 18
    assert snapshot.source_dimension == 16
    assert snapshot.minimum_eigenvalue > 1.0
    assert len(snapshot.configuration_sha256) == 64
    assert len(snapshot.provenance_sha256) == 64
    assert len(snapshot.state_sha256) == 64
    original = snapshot.surface_operator
    assert original.flags.writeable is False
    writable = np.array(original, copy=True)
    writable[0, 0] += 100.0
    assert snapshot.surface_operator[0, 0] != writable[0, 0]
    with pytest.raises(AttributeError):
        snapshot.cavity_descriptor_sha256 = "b" * 64

    tampered = _snapshot()
    changed = list(tampered._surface_values)
    changed[0] += 1.0e-3
    object.__setattr__(tampered, "_surface_values", tuple(changed))
    with pytest.raises(RuntimeError, match="drifted"):
        tampered.validate()


def test_one_stationary_scalar_generates_drive_hvp_jvp_and_vjp():
    functional = _functional()
    snapshot = functional.snapshot
    source_vector = SOURCE.reshape(-1)
    rhs = snapshot.source_operator @ source_vector
    sigma = np.linalg.solve(snapshot.surface_operator, rhs)
    expected_energy = -0.5 * rhs @ sigma
    expected_drive = (-snapshot.source_operator.T @ sigma).reshape(SOURCE.shape)
    rng = np.random.default_rng(91)
    direction = rng.normal(scale=0.04, size=SOURCE.shape)
    cotangent = rng.normal(scale=0.04, size=SOURCE.shape)

    assert functional.energy_eV(POSITIONS, SOURCE) == pytest.approx(
        expected_energy, abs=2e-14
    )
    np.testing.assert_allclose(
        functional.drive(POSITIONS, SOURCE), expected_drive, atol=2e-14, rtol=0.0
    )
    jvp = functional.source_jvp(POSITIONS, SOURCE, direction)
    vjp = functional.source_vjp(POSITIONS, SOURCE, cotangent)
    assert np.vdot(cotangent, jvp) == pytest.approx(np.vdot(vjp, direction), abs=3e-14)
    np.testing.assert_allclose(
        functional.coordinate_partial(POSITIONS, SOURCE), np.zeros_like(POSITIONS)
    )

    step = 1.0e-6
    energy_fd = (
        functional.energy_eV(POSITIONS, SOURCE + step * direction)
        - functional.energy_eV(POSITIONS, SOURCE - step * direction)
    ) / (2.0 * step)
    assert np.vdot(functional.drive(POSITIONS, SOURCE), direction) == pytest.approx(
        energy_fd, abs=2e-11
    )


def test_explicit_coefficient_conjugation_preserves_energy_and_drive_covariance():
    functional = _functional()
    rotation = _rotation(101)
    rotated = functional.rotated(rotation)
    cold_replay = functional.rotated(rotation)
    assert rotated.snapshot.state_sha256 == cold_replay.snapshot.state_sha256
    surface_rotation = functional.snapshot.coefficient_space.representation_matrix(
        rotation
    )
    source_rotation = radial_gto_source_rotation_matrix(rotation, atom_count=2)
    expected_a = (
        surface_rotation @ functional.snapshot.surface_operator @ surface_rotation.T
    )
    expected_s = (
        surface_rotation @ functional.snapshot.source_operator @ source_rotation.T
    )
    np.testing.assert_allclose(
        rotated.snapshot.surface_operator, expected_a, atol=8e-13, rtol=0.0
    )
    np.testing.assert_allclose(
        rotated.snapshot.source_operator, expected_s, atol=8e-13, rtol=0.0
    )

    positions_rotated = POSITIONS @ rotation.T
    source_rotated = (source_rotation @ SOURCE.reshape(-1)).reshape(SOURCE.shape)
    base_energy = functional.energy_eV(POSITIONS, SOURCE)
    rotated_energy = rotated.energy_eV(positions_rotated, source_rotated)
    assert rotated_energy == pytest.approx(base_energy, abs=2e-13)
    base_drive = functional.drive(POSITIONS, SOURCE).reshape(-1)
    expected_drive = (source_rotation @ base_drive).reshape(SOURCE.shape)
    np.testing.assert_allclose(
        rotated.drive(positions_rotated, source_rotated),
        expected_drive,
        atol=3e-13,
        rtol=0.0,
    )


def test_candidate_is_explicitly_not_a_geometry_assembler_or_tier_v_admission():
    functional = _functional()
    assert functional.coefficient_action_is_so3_representation is True
    assert functional.full_geometry_intertwiner_assembly_available is False
    assert functional.moving_cavity_coordinate_derivative_available is False
    assert functional.tier_v_rotation_admitted is False
    assert functional.capabilities.enabled_tiers == ()
    assert functional.source_dependent_geometry is False
    assert functional.include_nonpolar is False
    assert functional.scalar_first is True
    provenance = dict(functional.runtime_provenance())
    assert provenance["geometry_assembly"] == "external-coefficient-snapshot-only"
    assert provenance["tier_v_admission"] == "disabled"


def test_registry_keeps_harmonic_candidate_distinct_and_disabled():
    scalar = SCALAR_REGISTRY[
        VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_V1
    ]
    profile = PROFILE_REGISTRY[
        VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_PROFILE_V1
    ]
    assert profile.scalar_id == scalar.scalar_id
    assert profile.continuum_profile == "fixed-harmonic-galerkin-cpcm-candidate-v1"
    assert profile.cavity_profile == "fixed-external-harmonic-coefficient-cavity-v1"
    assert profile.enabled is False
    assert scalar.enabled is False
    assert profile.capabilities.enabled_tiers == ()
    assert scalar.admitted_capabilities.enabled_tiers == ()
    assert "geometry_intertwiner_assembly" in scalar.excluded_components
    assert scalar.implementation_entry_point == (
        "maple.solvation.coupling.variational_state:VariationalCommonFunctional"
    )


@pytest.mark.parametrize(
    "surface_operator, source_operator, message",
    (
        (np.eye(7), np.zeros((8, 16)), "surface_operator"),
        (np.diag([1.0] * 7 + [-1.0]), np.zeros((8, 16)), "positive definite"),
        (np.eye(8), np.zeros((7, 16)), "source_operator"),
    ),
)
def test_snapshot_rejects_incomplete_or_nonstationary_matrices(
    surface_operator, source_operator, message
):
    space = PerAtomHarmonicSpace(atom_count=2, lmax=1)
    with pytest.raises(ValueError, match=message):
        FixedHarmonicGalerkinSnapshot(
            atomic_numbers=(1, 8),
            coefficient_space=space,
            surface_operator=surface_operator,
            source_operator=source_operator,
            cavity_descriptor_sha256="a" * 64,
            assembly_contract_id="test-invalid-v1",
        )


def test_invalid_rotation_and_space_inputs_fail_closed():
    with pytest.raises((TypeError, ValueError)):
        PerAtomHarmonicSpace(atom_count=True, lmax=2)
    with pytest.raises((TypeError, ValueError)):
        PerAtomHarmonicSpace(atom_count=2, lmax=-1)
    with pytest.raises(ValueError, match="bounded harmonic"):
        PerAtomHarmonicSpace(atom_count=2, lmax=17)
    with pytest.raises(ValueError, match="bounded harmonic"):
        real_wigner_matrix(np.eye(3), lmax=17)
    improper = np.diag((1.0, 1.0, -1.0))
    with pytest.raises(ValueError, match="proper rotation"):
        real_wigner_matrix(improper, lmax=2)
    with pytest.raises(ValueError, match="atom_count"):
        radial_gto_source_rotation_matrix(np.eye(3), atom_count=0)
