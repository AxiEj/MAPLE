from __future__ import annotations

import math

import numpy as np
import pytest
from ase import Atoms

from maple.solvation.continuum.harmonic_ddpcm_functional import (
    SMOOTH_PARTITION_HARMONIC_DDPCM_SCALAR_ID,
    SmoothPartitionHarmonicDDPCMFunctionalCandidate,
)
from maple.solvation.continuum.harmonic_single_layer import (
    COULOMB_EV_ANGSTROM_PER_E2,
)
from maple.solvation.continuum.atomic_l1_pyddx import AtomicL1PyDDXPCMBackend

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


def _water() -> Atoms:
    return Atoms("OH2", positions=WATER_POSITIONS)


def _candidate(
    atoms: Atoms,
    radii: tuple[float, ...],
    *,
    lmax: int = 3,
    dielectric: float = 80.0,
):
    torch = pytest.importorskip("torch")
    return SmoothPartitionHarmonicDDPCMFunctionalCandidate(
        atomic_numbers=tuple(int(value) for value in atoms.numbers),
        radii_angstrom=radii,
        transition_width_angstrom2=0.08,
        surface_lmax=lmax,
        partition_lmax=2 * lmax,
        partition_radial_quadrature_order=96,
        source_radial_quadrature_order=128,
        double_layer_radial_quadrature_order=128,
        dielectric=dielectric,
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


def test_single_sphere_matches_the_finite_dielectric_monopole_dipole_limit():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    radius = 2.0
    dielectric = 7.0
    source = np.asarray([[0.4, 0.2, -0.1, 0.3]])
    candidate = _candidate(atoms, (radius,), lmax=1, dielectric=dielectric)
    dipole_squared = float(source[0, 1:] @ source[0, 1:])
    expected = (
        -0.5
        * COULOMB_EV_ANGSTROM_PER_E2
        * (dielectric - 1.0)
        / dielectric
        * source[0, 0] ** 2
        / radius
        - COULOMB_EV_ANGSTROM_PER_E2
        * (dielectric - 1.0)
        / (2.0 * dielectric + 1.0)
        * dipole_squared
        / radius**3
    )
    assert candidate.energy_eV(atoms, source) == pytest.approx(
        expected, rel=0.0, abs=3.0e-15
    )


def test_water_converges_to_the_real_pyddx_ddpcm_scalar():
    pytest.importorskip("pyddx")
    atoms = _water()
    values = [
        _candidate(atoms, WATER_RADII, lmax=lmax).energy_eV(atoms, WATER_SOURCE)
        for lmax in (2, 3, 4, 5)
    ]
    backend = AtomicL1PyDDXPCMBackend(
        atoms,
        WATER_RADII,
        dielectric=80.0,
        lmax=15,
        n_lebedev=1202,
        solver_tolerance=1.0e-12,
        eta=0.1,
    )
    reference = backend.energy(atoms, WATER_SOURCE)
    errors = np.abs(np.asarray(values) - reference)
    assert np.all(errors[1:] < errors[:-1])
    assert errors[-1] < 1.3e-5


@pytest.mark.parametrize("separation", (0.2, 1.8))
def test_centered_double_layer_is_differentiable_at_sphere_tangencies(separation):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [separation, 0.0, 0.0]])
    source = np.asarray([[0.4, 0.05, -0.02, 0.03], [-0.4, -0.01, 0.04, -0.02]])
    candidate = _candidate(atoms, (1.0, 0.8), lmax=2, dielectric=20.0)
    gradient = candidate.coordinate_partial(atoms, source)
    direction = np.zeros((2, 3))
    direction[1, 0] = 1.0
    hvp = candidate.coordinate_hvp(atoms, source, direction)
    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(hvp))

    errors = []
    derivative_jumps = []
    analytic = float(gradient[1, 0])
    for step in (3.0e-4, 1.0e-4, 3.0e-5):
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions[1, 0] += step
        minus.positions[1, 0] -= step
        errors.append(
            abs(
                (candidate.energy_eV(plus, source) - candidate.energy_eV(minus, source))
                / (2.0 * step)
                - analytic
            )
        )
        derivative_jumps.append(
            abs(
                candidate.coordinate_partial(plus, source)[1, 0]
                - candidate.coordinate_partial(minus, source)[1, 0]
            )
        )
    assert errors[1] < 0.2 * errors[0]
    assert errors[2] < 0.2 * errors[1]
    assert errors[-1] < 8.0e-8
    assert derivative_jumps[1] < 0.5 * derivative_jumps[0]
    assert derivative_jumps[2] < 0.5 * derivative_jumps[1]


def test_rotation_source_adjoint_coordinate_gradient_and_hessian_share_scalar():
    atoms = _water()
    candidate = _candidate(atoms, WATER_RADII, lmax=2)
    axis = np.asarray([0.2, -0.7, 0.4])
    axis /= np.linalg.norm(axis)
    angle = 0.83
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        np.eye(3) * math.cos(angle)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * cross
    )
    rotated_atoms = atoms.copy()
    rotated_atoms.positions = atoms.positions @ rotation.T + np.asarray(
        [1.3, -0.4, 0.8]
    )
    rotated_source = _rotate_source(WATER_SOURCE, rotation)
    energy = candidate.energy_eV(atoms, WATER_SOURCE)
    rotated_energy = candidate.energy_eV(rotated_atoms, rotated_source)
    gradient = candidate.coordinate_partial(atoms, WATER_SOURCE)
    rotated_gradient = candidate.coordinate_partial(rotated_atoms, rotated_source)
    assert rotated_energy == pytest.approx(energy, rel=0.0, abs=5.0e-13)
    assert np.allclose(rotated_gradient, gradient @ rotation.T, rtol=0.0, atol=3.0e-12)
    assert np.allclose(np.sum(gradient, axis=0), 0.0, rtol=0.0, atol=3.0e-13)

    generator = np.random.default_rng(20260816)
    source_direction = generator.normal(size=WATER_SOURCE.shape)
    cotangent = generator.normal(size=WATER_SOURCE.shape)
    jvp = candidate.source_jvp(atoms, WATER_SOURCE, source_direction)
    vjp = candidate.source_vjp(atoms, WATER_SOURCE, cotangent)
    assert float(cotangent.reshape(-1) @ jvp.reshape(-1)) == pytest.approx(
        float(source_direction.reshape(-1) @ vjp.reshape(-1)),
        rel=0.0,
        abs=4.0e-12,
    )

    first_direction = generator.normal(size=(len(atoms), 3))
    second_direction = generator.normal(size=(len(atoms), 3))
    first_hvp = candidate.coordinate_hvp(atoms, WATER_SOURCE, first_direction)
    second_hvp = candidate.coordinate_hvp(atoms, WATER_SOURCE, second_direction)
    assert float(second_direction.reshape(-1) @ first_hvp.reshape(-1)) == pytest.approx(
        float(first_direction.reshape(-1) @ second_hvp.reshape(-1)),
        rel=0.0,
        abs=4.0e-12,
    )


def test_candidate_is_immutable_content_addressed_and_not_admitted():
    candidate = _candidate(_water(), WATER_RADII, lmax=2)
    assert candidate.scalar_id == SMOOTH_PARTITION_HARMONIC_DDPCM_SCALAR_ID
    assert candidate.capabilities.enabled_tiers == ()
    assert len(candidate.configuration_sha256()) == 64
    assert len(candidate.provenance_sha256) == 64
    with pytest.raises(AttributeError):
        candidate._dielectric = 2.0


def test_exact_source_response_operator_matches_the_sealed_scalar_hessian():
    atoms = _water()
    candidate = _candidate(atoms, WATER_RADII, lmax=2)
    direction = np.random.default_rng(20260818).normal(size=WATER_SOURCE.shape)
    operator = candidate.source_covector_response_operator(atoms)
    expected = candidate.source_hvp(atoms, WATER_SOURCE, direction)
    actual = (operator @ direction.reshape(-1)).reshape(direction.shape)

    assert np.allclose(operator, operator.T, rtol=0.0, atol=2.0e-13)
    assert np.allclose(actual, expected, rtol=0.0, atol=3.0e-12)
