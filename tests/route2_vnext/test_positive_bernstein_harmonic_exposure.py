from __future__ import annotations

import numpy as np
import pytest

from maple.solvation.continuum.harmonic_cds_area import (
    PositiveBernsteinHarmonicExposureArea,
)
from maple.solvation.continuum.harmonic_coefficients import real_wigner_matrix
from maple.solvation.continuum.harmonic_positive_exposure import (
    POSITIVE_BERNSTEIN_MAXIMUM_INTEGRAND_DEGREE,
    POSITIVE_BERNSTEIN_MAXIMUM_TRANSITION_FACTORS,
    POSITIVE_BERNSTEIN_PAIR_DEGREE,
    assemble_positive_bernstein_exposure,
)


def _rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(3, 3))
    rotation, triangular = np.linalg.qr(matrix)
    rotation = rotation @ np.diag(np.where(np.diag(triangular) < 0.0, -1.0, 1.0))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return rotation


def _fibonacci_shell(count: int, *, distance: float) -> np.ndarray:
    index = np.arange(count)
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    z = 1.0 - 2.0 * (index + 0.5) / count
    radial = np.sqrt(1.0 - z * z)
    angle = 2.0 * np.pi * index / golden
    directions = np.column_stack((radial * np.cos(angle), radial * np.sin(angle), z))
    return np.vstack((np.zeros(3), distance * directions))


def _area(
    *,
    atomic_numbers: tuple[int, ...] = (6, 8, 7),
    radii: tuple[float, ...] = (1.4, 1.2, 1.3),
):
    torch = pytest.importorskip("torch")
    return PositiveBernsteinHarmonicExposureArea(
        atomic_numbers=atomic_numbers,
        radii_angstrom=radii,
        transition_width_angstrom2=0.18,
        surface_lmax=2,
        dtype=torch.float64,
        device="cpu",
    )


def test_isolated_parent_has_exact_area_identity_block_and_zero_vjp() -> None:
    radius = 1.37
    area = _area(atomic_numbers=(8,), radii=(radius,))
    positions = np.asarray([[0.3, -0.2, 0.7]])
    np.testing.assert_allclose(
        area.atom_areas_angstrom2(positions),
        [4.0 * np.pi * radius * radius],
        atol=2.0e-14,
        rtol=0.0,
    )
    np.testing.assert_array_equal(area.multiplication_blocks(positions)[0], np.eye(9))
    np.testing.assert_allclose(
        area.position_vjp(positions, [2.3]), 0.0, atol=0.0, rtol=0.0
    )


def test_many_factor_nearly_buried_parent_remains_positive_without_clipping() -> None:
    count = 26
    positions = _fibonacci_shell(count, distance=1.5)
    radii = (2.1,) + (1.5,) * count
    area = _area(atomic_numbers=(6,) + (1,) * count, radii=radii)
    values = area.atom_areas_angstrom2(positions)
    full = 4.0 * np.pi * np.square(radii)
    assert values[0] > 0.0
    assert values[0] < 1.0e-2
    assert np.all(values >= 0.0)
    assert np.all(values <= full)

    blocks = area.multiplication_blocks(positions)
    for block in blocks:
        assert np.linalg.eigvalsh(block)[0] >= -2.0e-13
        assert np.linalg.eigvalsh(np.eye(9) - block)[0] >= -2.0e-13
    diagnostics = area.parent_diagnostics(positions)
    assert diagnostics["maximum_transition_factor_count"] == count
    assert diagnostics["maximum_integrand_degree"] == (
        POSITIVE_BERNSTEIN_PAIR_DEGREE * count + 4
    )


def test_parent_moments_and_blocks_are_rotation_translation_covariant() -> None:
    torch = pytest.importorskip("torch")
    positions = np.asarray([[0.0, 0.0, 0.0], [1.7, 0.2, 0.0], [-0.3, 1.8, 0.1]])
    radii = (1.4, 1.2, 1.3)
    rotation = _rotation(20260817)
    translation = np.asarray([1.2, -0.7, 0.3])

    def build(values: np.ndarray):
        return assemble_positive_bernstein_exposure(
            torch.tensor(values, dtype=torch.float64),
            radii=radii,
            transition_width=0.18,
            surface_lmax=2,
        )

    base = build(positions)
    transformed = build(positions @ rotation.T + translation)
    moment_rotation = real_wigner_matrix(rotation, lmax=4)
    surface_rotation = real_wigner_matrix(rotation, lmax=2)
    for atom in range(len(radii)):
        np.testing.assert_allclose(
            transformed.moments[atom].detach().numpy(),
            moment_rotation @ base.moments[atom].detach().numpy(),
            atol=2.0e-11,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            transformed.multiplication_blocks[atom].detach().numpy(),
            surface_rotation
            @ base.multiplication_blocks[atom].detach().numpy()
            @ surface_rotation.T,
            atol=2.0e-11,
            rtol=0.0,
        )


def test_area_vjp_matches_multistep_directional_difference() -> None:
    positions = np.asarray([[0.0, 0.0, 0.0], [1.7, 0.2, 0.0], [-0.3, 1.8, 0.1]])
    area = _area()
    rng = np.random.default_rng(4701)
    cotangent = rng.normal(size=3)
    direction = rng.normal(size=positions.shape)
    direction -= np.mean(direction, axis=0)
    direction /= np.linalg.norm(direction)
    analytic = float(np.vdot(area.position_vjp(positions, cotangent), direction))
    errors = []
    for step in (4.0e-5, 2.0e-5, 1.0e-5):
        finite = (
            np.vdot(
                cotangent,
                area.atom_areas_angstrom2(positions + step * direction),
            )
            - np.vdot(
                cotangent,
                area.atom_areas_angstrom2(positions - step * direction),
            )
        ) / (2.0 * step)
        errors.append(abs(finite - analytic))
    assert errors[-1] < 3.0e-7
    assert errors[-1] < 0.3 * errors[0]
    np.testing.assert_allclose(
        np.sum(area.position_vjp(positions, cotangent), axis=0),
        0.0,
        atol=3.0e-12,
        rtol=0.0,
    )


def test_transition_factor_and_algebraic_degree_caps_fail_closed() -> None:
    torch = pytest.importorskip("torch")
    count = POSITIVE_BERNSTEIN_MAXIMUM_TRANSITION_FACTORS + 1
    positions = _fibonacci_shell(count, distance=1.5)
    with pytest.raises(ValueError, match="transition-factor cap"):
        assemble_positive_bernstein_exposure(
            torch.tensor(positions, dtype=torch.float64),
            radii=(2.1,) + (1.5,) * count,
            transition_width=0.18,
            surface_lmax=2,
        )
    assert POSITIVE_BERNSTEIN_PAIR_DEGREE * count + 4 <= (
        POSITIVE_BERNSTEIN_MAXIMUM_INTEGRAND_DEGREE
    )


def test_identity_is_separate_from_legacy_projected_exposure() -> None:
    area = _area()
    assert area.positive_parent_pair_degree == 4
    assert area.positive_parent_maximum_transition_factors == 30
    assert area.reconstructed_low_band_used_as_mask is False
    assert "positive-bernstein" in area.cavity_profile_id
    assert "positive-bernstein" in area.exposure_contract_id
    assert area.capabilities.enabled_tiers == ()
