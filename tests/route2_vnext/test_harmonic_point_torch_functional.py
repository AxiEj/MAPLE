from __future__ import annotations

import numpy as np
import pytest

from maple.solvation.continuum.harmonic_coefficients import PerAtomHarmonicSpace
from maple.solvation.continuum.harmonic_point_source import (
    point_harmonic_source_operator,
)
from maple.solvation.continuum.harmonic_torch_functional import (
    POINT_L1_HARMONIC_RESEARCH_SCALAR_ID,
    POINT_L1_SOURCE_EMBEDDING,
    SmoothWeightedHarmonicGalerkinFunctionalCandidate,
)

POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [1.72, -0.31, 0.21], [-0.39, 1.65, -0.17]],
    dtype=float,
)
NUMBERS = (8, 1, 6)
RADII = (1.43, 1.18, 1.57)
SOURCE = np.asarray(
    [
        [0.20, -0.03, 0.02, -0.01],
        [-0.14, 0.02, -0.03, 0.02],
        [-0.06, 0.01, 0.01, -0.01],
    ],
    dtype=float,
)


def _functional(*, dielectric: float | None = 78.39):
    torch = pytest.importorskip("torch")
    return SmoothWeightedHarmonicGalerkinFunctionalCandidate(
        atomic_numbers=NUMBERS,
        radii_angstrom=RADII,
        transition_width_angstrom2=0.18,
        surface_lmax=1,
        exposure_lmax=2,
        exposure_radial_quadrature_order=32,
        source_radial_quadrature_order=64,
        green_radial_quadrature_order=32,
        dtype=torch.float64,
        device="cpu",
        source_embedding=POINT_L1_SOURCE_EMBEDDING,
        dielectric=dielectric,
        scalar_id=POINT_L1_HARMONIC_RESEARCH_SCALAR_ID,
    )


def _rotation(seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(3, 3))
    rotation, triangular = np.linalg.qr(matrix)
    rotation = rotation @ np.diag(np.where(np.diag(triangular) < 0.0, -1.0, 1.0))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return rotation


def _source_rotation(rotation: np.ndarray) -> np.ndarray:
    cartesian_from_raw = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    raw_from_cartesian = cartesian_from_raw.T
    block = np.zeros((4, 4))
    block[0, 0] = 1.0
    block += raw_from_cartesian @ rotation @ cartesian_from_raw
    return np.kron(np.eye(len(POSITIONS)), block)


def test_torch_point_source_matches_independent_numpy_operator() -> None:
    functional = _functional()
    matrices = functional.debug_geometry_matrices(POSITIONS)
    expected = point_harmonic_source_operator(
        positions_angstrom=POSITIONS,
        radii_angstrom=RADII,
        surface_lmax=functional.physical_lmax,
        radial_quadrature_order=functional.source_radial_quadrature_order,
    )
    np.testing.assert_allclose(matrices["raw_source"], expected, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(
        matrices["source_operator"],
        matrices["weighted_basis"].T @ expected,
        rtol=0.0,
        atol=3.0e-12,
    )
    assert functional.source_space.component_count == 4
    assert functional.source_embedding == POINT_L1_SOURCE_EMBEDDING
    assert functional.registered_scalar is False


def test_cpcm_dielectric_factor_scales_the_same_stationary_scalar() -> None:
    conductor = _functional(dielectric=None)
    finite = _functional(dielectric=78.39)
    factor = (78.39 - 1.0) / 78.39
    assert finite.cpcm_screening_factor == pytest.approx(factor, abs=0.0)
    assert finite.energy_eV(POSITIONS, SOURCE) == pytest.approx(
        factor * conductor.energy_eV(POSITIONS, SOURCE), abs=2.0e-14
    )
    assert finite.configuration_sha256() != conductor.configuration_sha256()


def test_point_scalar_generates_source_coordinate_and_second_derivatives() -> None:
    functional = _functional()
    rng = np.random.default_rng(20260816)
    source_direction = rng.normal(scale=0.03, size=SOURCE.shape)
    source_cotangent = rng.normal(scale=0.02, size=SOURCE.shape)
    coordinate_direction = rng.normal(size=POSITIONS.shape)
    coordinate_direction -= np.mean(coordinate_direction, axis=0)
    coordinate_direction /= np.linalg.norm(coordinate_direction)

    step = 2.0e-6
    finite_source = (
        functional.energy_eV(POSITIONS, SOURCE + step * source_direction)
        - functional.energy_eV(POSITIONS, SOURCE - step * source_direction)
    ) / (2.0 * step)
    assert functional.pairing.pair(
        source_direction,
        functional.drive(POSITIONS, SOURCE),
    ) == pytest.approx(finite_source, abs=3.0e-11)

    jvp = functional.source_jvp(POSITIONS, SOURCE, source_direction)
    vjp = functional.source_vjp(POSITIONS, SOURCE, source_cotangent)
    assert np.vdot(source_cotangent, jvp) == pytest.approx(
        np.vdot(vjp, source_direction), abs=5.0e-13
    )

    gradient = functional.coordinate_partial(POSITIONS, SOURCE)
    analytic = float(np.vdot(gradient, coordinate_direction))
    coordinate_step = 1.0e-4
    finite_coordinate_coarse = (
        functional.energy_eV(POSITIONS + coordinate_step * coordinate_direction, SOURCE)
        - functional.energy_eV(
            POSITIONS - coordinate_step * coordinate_direction, SOURCE
        )
    ) / (2.0 * coordinate_step)
    fine_coordinate_step = coordinate_step / 2.0
    finite_coordinate_fine = (
        functional.energy_eV(
            POSITIONS + fine_coordinate_step * coordinate_direction, SOURCE
        )
        - functional.energy_eV(
            POSITIONS - fine_coordinate_step * coordinate_direction, SOURCE
        )
    ) / (2.0 * fine_coordinate_step)
    finite_coordinate = (4.0 * finite_coordinate_fine - finite_coordinate_coarse) / 3.0
    assert analytic == pytest.approx(finite_coordinate, abs=8.0e-10)

    hvp = functional.coordinate_hvp(POSITIONS, SOURCE, coordinate_direction)
    hessian = functional.coordinate_hessian(POSITIONS, SOURCE)
    np.testing.assert_allclose(hessian, hessian.T, rtol=0.0, atol=2.0e-11)
    np.testing.assert_allclose(
        hvp.reshape(-1),
        hessian @ coordinate_direction.reshape(-1),
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    hvp_step = 2.0e-5
    finite_hvp = (
        functional.coordinate_partial(
            POSITIONS + hvp_step * coordinate_direction, SOURCE
        )
        - functional.coordinate_partial(
            POSITIONS - hvp_step * coordinate_direction, SOURCE
        )
    ) / (2.0 * hvp_step)
    np.testing.assert_allclose(hvp, finite_hvp, rtol=0.0, atol=2.0e-7)


def test_point_scalar_is_jointly_rotation_covariant() -> None:
    functional = _functional()
    rotation = _rotation(1701)
    source_rotation = _source_rotation(rotation)
    rotated_positions = POSITIONS @ rotation.T
    rotated_source = (source_rotation @ SOURCE.reshape(-1)).reshape(SOURCE.shape)
    energy = functional.energy_eV(POSITIONS, SOURCE)
    gradient = functional.coordinate_partial(POSITIONS, SOURCE)
    assert functional.energy_eV(rotated_positions, rotated_source) == pytest.approx(
        energy, abs=2.0e-13
    )
    np.testing.assert_allclose(
        functional.coordinate_partial(rotated_positions, rotated_source),
        gradient @ rotation.T,
        rtol=0.0,
        atol=4.0e-12,
    )

    raw = functional.debug_geometry_matrices(POSITIONS)["raw_source"]
    rotated_raw = functional.debug_geometry_matrices(rotated_positions)["raw_source"]
    harmonic_rotation = PerAtomHarmonicSpace(
        atom_count=len(POSITIONS), lmax=functional.physical_lmax
    ).representation_matrix(rotation)
    np.testing.assert_allclose(
        rotated_raw @ source_rotation,
        harmonic_rotation @ raw,
        rtol=0.0,
        atol=7.0e-10,
    )


def test_point_binding_rejects_gaussian_scalar_and_singular_geometry() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="scalar/source-embedding"):
        SmoothWeightedHarmonicGalerkinFunctionalCandidate(
            atomic_numbers=NUMBERS,
            radii_angstrom=RADII,
            transition_width_angstrom2=0.18,
            surface_lmax=1,
            exposure_lmax=2,
            dtype=torch.float64,
            device="cpu",
            source_embedding=POINT_L1_SOURCE_EMBEDDING,
        )
    functional = _functional()
    singular = POSITIONS.copy()
    singular[1] = np.asarray([RADII[0], 0.0, 0.0])
    with pytest.raises(ValueError, match="lies on a target sphere"):
        functional.energy_eV(singular, SOURCE)
