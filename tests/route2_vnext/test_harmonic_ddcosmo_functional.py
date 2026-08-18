from __future__ import annotations

import math

import numpy as np
import pytest
from ase import Atoms

from maple.solvation.continuum.harmonic_ddcosmo_functional import (
    SMOOTH_PARTITION_HARMONIC_DDCOSMO_SCALAR_ID,
    SmoothPartitionHarmonicDDCOSMOFunctionalCandidate,
)
from maple.solvation.continuum.harmonic_schwarz_primitives import (
    _assemble_smooth_partition,
    _regular_solid_harmonic_design,
)
from maple.solvation.continuum.harmonic_single_layer import (
    COULOMB_EV_ANGSTROM_PER_E2,
)
from maple.solvation.continuum.harmonic_torch_primitives import (
    _torch_real_harmonic_design,
)
from maple.solvation.continuum.atomic_l1_pyddx import AtomicL1PyDDXPCMBackend
from maple.function.calculator.extra_correction.implicit.pyddx_pcm_response import (
    PyDDXCOSMOReactionFieldLinearMap,
)

WATER_POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.927297, 0.0]]
)
WATER_RADII = (2.294, 1.2, 1.2)
WATER_SOURCE = np.asarray(
    [[-0.7, 0.04, -0.02, 0.03], [0.35, 0.0, 0.01, -0.02], [0.35, -0.01, 0.0, 0.02]]
)
PROPANOL_POSITIONS = np.asarray(
    [
        [-1.449486, 1.212993, 0.0],
        [0.0, 0.743231, 0.0],
        [0.099896, -0.767026, 0.0],
        [0.526471, 1.124432, 0.876596],
        [0.526471, 1.124432, -0.876596],
        [1.471457, -1.120843, 0.0],
        [1.555848, -2.075074, 0.0],
        [-1.51257, 2.299803, 0.0],
        [-1.980187, 0.849077, 0.880964],
        [-1.980187, 0.849077, -0.880964],
        [-0.404981, -1.170095, -0.885617],
        [-0.404981, -1.170095, 0.885617],
    ]
)
PROPANOL_RADII = (1.85, 1.85, 1.85, 1.2, 1.2, 2.294, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2)


def _water() -> Atoms:
    return Atoms("OH2", positions=WATER_POSITIONS)


def _candidate(atoms: Atoms, radii: tuple[float, ...], *, lmax: int = 3):
    torch = pytest.importorskip("torch")
    return SmoothPartitionHarmonicDDCOSMOFunctionalCandidate(
        atomic_numbers=tuple(int(value) for value in atoms.numbers),
        radii_angstrom=radii,
        transition_width_angstrom2=0.08,
        surface_lmax=lmax,
        partition_lmax=2 * lmax,
        partition_radial_quadrature_order=96,
        source_radial_quadrature_order=128,
        dielectric=80.0,
        dtype=torch.float64,
        device="cpu",
    )


def _rotate_source(source: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.array(source, copy=True)
    cartesian = source[:, (3, 1, 2)]
    rotated = cartesian @ rotation.T
    result[:, 1] = rotated[:, 1]
    result[:, 2] = rotated[:, 2]
    result[:, 3] = rotated[:, 0]
    return result


def test_regular_solid_harmonics_match_unit_sphere_and_are_defined_at_origin():
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(20260816)
    vectors = torch.randn((7, 3), generator=generator, dtype=torch.float64)
    unit = vectors / torch.linalg.vector_norm(vectors, dim=1)[:, None]
    assert torch.allclose(
        _regular_solid_harmonic_design(unit, lmax=5),
        _torch_real_harmonic_design(unit, lmax=5),
        rtol=0.0,
        atol=2.0e-13,
    )
    origin = _regular_solid_harmonic_design(
        torch.zeros((1, 3), dtype=torch.float64), lmax=5
    )[0]
    assert origin[0] == pytest.approx(1.0 / math.sqrt(4.0 * math.pi), abs=1.0e-15)
    assert np.count_nonzero(np.asarray(origin[1:])) == 0


def test_polynomial_partition_closes_and_handles_complete_burial():
    torch = pytest.importorskip("torch")
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.7, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    partition = _assemble_smooth_partition(
        positions,
        radii=(2.0, 0.4, 1.2),
        transition_width=0.08,
        partition_lmax=4,
        radial_order=96,
    )
    one = positions.new_zeros((25,))
    one[0] = math.sqrt(4.0 * math.pi)
    for exposed, overlap in partition:
        closure = exposed + sum(overlap.values(), start=torch.zeros_like(exposed))
        assert torch.allclose(closure, one, rtol=0.0, atol=2.0e-13)
    # Sphere 1 is wholly contained in sphere 0.  Its exposed trace vanishes,
    # but its Schwarz overlap equation remains present.
    assert torch.count_nonzero(partition[1][0]) == 0
    assert 0 in partition[1][1]


def test_single_sphere_matches_exact_conductor_monopole_and_dipole_energy():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    radius = 2.0
    candidate = _candidate(atoms, (radius,), lmax=1)
    source = np.asarray([[0.8, -0.03, 0.02, 0.04]])
    coulomb = COULOMB_EV_ANGSTROM_PER_E2
    charge = source[0, 0]
    dipole_squared = float(source[0, 1:] @ source[0, 1:])
    expected_conductor = (
        -0.5 * coulomb * (charge * charge / radius + dipole_squared / radius**3)
    )
    expected = ((80.0 - 1.0) / 80.0) * expected_conductor
    assert candidate.energy_eV(atoms, source) == pytest.approx(
        expected, rel=0.0, abs=2.0e-13
    )


def test_complete_burial_keeps_the_fixed_schwarz_system_full_rank():
    atoms = Atoms("C3H8O", positions=PROPANOL_POSITIONS)
    candidate = _candidate(atoms, PROPANOL_RADII, lmax=1)
    matrices = candidate.debug_geometry_matrices(atoms)
    schwarz = matrices["schwarz_operator"]
    assert schwarz.shape == (48, 48)
    assert np.linalg.matrix_rank(schwarz) == 48
    assert np.linalg.cond(schwarz) < 10.0
    # Atom 6 is the O-H hydrogen wholly buried by atom 5's oxygen sphere.
    assert np.count_nonzero(matrices["exposed_coefficients"][6]) == 0


def test_water_converges_to_the_real_pyddx_ddcosmo_scalar():
    pytest.importorskip("pyddx")
    atoms = _water()
    values = []
    for lmax in (2, 3, 4, 5):
        values.append(
            _candidate(atoms, WATER_RADII, lmax=lmax).energy_eV(atoms, WATER_SOURCE)
        )
    backend = AtomicL1PyDDXPCMBackend(
        atoms,
        WATER_RADII,
        dielectric=80.0,
        lmax=15,
        n_lebedev=1202,
        solver_tolerance=1.0e-12,
        eta=0.1,
        _map_factory=PyDDXCOSMOReactionFieldLinearMap,
    )
    reference = backend.energy(atoms, WATER_SOURCE)
    errors = np.abs(np.asarray(values) - reference)
    assert np.all(errors[1:] < errors[:-1])
    assert errors[-1] < 5.0e-5


def test_energy_force_rotation_and_translation_covariance_are_roundoff_limited():
    atoms = _water()
    candidate = _candidate(atoms, WATER_RADII, lmax=3)
    axis = np.asarray([0.2, -0.7, 0.4])
    axis /= np.linalg.norm(axis)
    angle = 0.83
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = (
        np.eye(3) * math.cos(angle)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * cross
    )
    translated = np.asarray([1.3, -0.4, 0.8])
    rotated_atoms = atoms.copy()
    rotated_atoms.positions = atoms.positions @ rotation.T + translated
    rotated_source = _rotate_source(WATER_SOURCE, rotation)
    energy = candidate.energy_eV(atoms, WATER_SOURCE)
    rotated_energy = candidate.energy_eV(rotated_atoms, rotated_source)
    gradient = candidate.coordinate_partial(atoms, WATER_SOURCE)
    rotated_gradient = candidate.coordinate_partial(rotated_atoms, rotated_source)
    assert rotated_energy == pytest.approx(energy, rel=0.0, abs=3.0e-13)
    assert np.allclose(rotated_gradient, gradient @ rotation.T, rtol=0.0, atol=2.0e-12)
    assert np.allclose(np.sum(gradient, axis=0), 0.0, rtol=0.0, atol=2.0e-13)


def test_source_jvp_vjp_coordinate_gradient_and_hvp_share_the_scalar():
    atoms = _water()
    candidate = _candidate(atoms, WATER_RADII, lmax=3)
    generator = np.random.default_rng(20260816)
    direction = generator.normal(size=WATER_SOURCE.shape)
    cotangent = generator.normal(size=WATER_SOURCE.shape)
    jvp = candidate.source_jvp(atoms, WATER_SOURCE, direction)
    vjp = candidate.source_vjp(atoms, WATER_SOURCE, cotangent)
    assert float(cotangent.reshape(-1) @ jvp.reshape(-1)) == pytest.approx(
        float(direction.reshape(-1) @ vjp.reshape(-1)), rel=0.0, abs=3.0e-12
    )

    coordinate_direction = generator.normal(size=(len(atoms), 3))
    coordinate_direction -= coordinate_direction.mean(axis=0)
    analytic = float(
        candidate.coordinate_partial(atoms, WATER_SOURCE).reshape(-1)
        @ coordinate_direction.reshape(-1)
    )
    finite = []
    for step in (2.0e-4, 1.0e-4, 5.0e-5):
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions += step * coordinate_direction
        minus.positions -= step * coordinate_direction
        finite.append(
            (
                candidate.energy_eV(plus, WATER_SOURCE)
                - candidate.energy_eV(minus, WATER_SOURCE)
            )
            / (2.0 * step)
        )
    errors = np.abs(np.asarray(finite) - analytic)
    assert errors[1] < 0.35 * errors[0]
    assert errors[2] < 0.35 * errors[1]
    assert errors[-1] < 2.0e-8

    hvp = candidate.coordinate_hvp(atoms, WATER_SOURCE, coordinate_direction)
    plus = atoms.copy()
    minus = atoms.copy()
    step = 2.0e-5
    plus.positions += step * coordinate_direction
    minus.positions -= step * coordinate_direction
    finite_hvp = (
        candidate.coordinate_partial(plus, WATER_SOURCE)
        - candidate.coordinate_partial(minus, WATER_SOURCE)
    ) / (2.0 * step)
    assert np.allclose(hvp, finite_hvp, rtol=2.0e-6, atol=3.0e-8)


def test_candidate_is_immutable_content_addressed_and_not_admitted():
    candidate = _candidate(_water(), WATER_RADII, lmax=2)
    assert candidate.scalar_id == SMOOTH_PARTITION_HARMONIC_DDCOSMO_SCALAR_ID
    assert candidate.capabilities.enabled_tiers == ()
    assert len(candidate.configuration_sha256()) == 64
    assert len(candidate.provenance_sha256) == 64
    with pytest.raises(AttributeError):
        candidate._dielectric = 2.0
