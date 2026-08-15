from __future__ import annotations

import numpy as np

from maple.function.calculator.extra_correction.implicit.gto_density import (
    point_multipole_potential,
)
from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.continuum.harmonic_coefficients import (
    PerAtomHarmonicSpace,
    _real_harmonic_design,
)
from maple.solvation.continuum.harmonic_gaussian_source import (
    gaussian_harmonic_source_operator,
)
from maple.solvation.continuum.harmonic_point_source import (
    harmonic_point_source_implementation_sha256,
    point_harmonic_source_operator,
)

POSITIONS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [2.18, -0.37, 0.29],
        [-0.42, 2.06, -0.33],
    ],
    dtype=float,
)
RADII = np.asarray([1.52, 1.21, 1.70], dtype=float)


def test_point_harmonic_implementation_digest_is_content_addressed() -> None:
    digest = harmonic_point_source_implementation_sha256()
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def _rotation(seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(3, 3))
    rotation, triangular = np.linalg.qr(matrix)
    rotation = rotation @ np.diag(np.where(np.diag(triangular) < 0.0, -1.0, 1.0))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return rotation


def _point_source_rotation(rotation: np.ndarray) -> np.ndarray:
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


def _sphere_rule(order: int = 112) -> tuple[np.ndarray, np.ndarray]:
    cosine, polar_weights = np.polynomial.legendre.leggauss(order)
    phi_count = 2 * order + 1
    phi = 2.0 * np.pi * np.arange(phi_count, dtype=float) / phi_count
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
    directions = np.stack(
        (
            (sine[:, None] * np.cos(phi)[None, :]).reshape(-1),
            (sine[:, None] * np.sin(phi)[None, :]).reshape(-1),
            np.repeat(cosine, phi_count),
        ),
        axis=1,
    )
    weights = np.repeat(polar_weights, phi_count) * (2.0 * np.pi / phi_count)
    return directions, weights


def _direct_projection(lmax: int) -> np.ndarray:
    directions, weights = _sphere_rule()
    harmonics = _real_harmonic_design(directions, lmax=lmax)
    block_dimension = (lmax + 1) ** 2
    operator = np.empty((len(POSITIONS) * block_dimension, len(POSITIONS) * 4))
    for target, (center, radius) in enumerate(zip(POSITIONS, RADII, strict=True)):
        points_angstrom = center[None, :] + radius * directions
        row = slice(target * block_dimension, (target + 1) * block_dimension)
        for source in range(len(POSITIONS)):
            for component in range(4):
                unit = np.zeros((1, 4), dtype=float)
                unit[0, component] = 1.0
                potential = point_multipole_potential(
                    points_angstrom / 0.5291772105638411,
                    POSITIONS[source : source + 1],
                    unit,
                )
                operator[row, source * 4 + component] = (
                    HARTREE_TO_EV * harmonics.T @ (weights * potential)
                )
    return operator


def test_point_harmonic_operator_matches_independent_surface_projection() -> None:
    actual = point_harmonic_source_operator(
        positions_angstrom=POSITIONS,
        radii_angstrom=RADII,
        surface_lmax=3,
        radial_quadrature_order=160,
    )
    expected = _direct_projection(3)
    np.testing.assert_allclose(actual, expected, rtol=3.0e-11, atol=2.0e-10)


def test_point_harmonic_operator_is_translation_invariant_and_not_gaussian() -> None:
    point = point_harmonic_source_operator(
        positions_angstrom=POSITIONS,
        radii_angstrom=RADII,
        surface_lmax=3,
        radial_quadrature_order=160,
    )
    translated = point_harmonic_source_operator(
        positions_angstrom=POSITIONS + np.asarray([8.2, -4.1, 2.7]),
        radii_angstrom=RADII,
        surface_lmax=3,
        radial_quadrature_order=160,
    )
    gaussian = gaussian_harmonic_source_operator(
        positions_angstrom=POSITIONS,
        radii_angstrom=RADII,
        surface_lmax=3,
        radial_quadrature_order=160,
    )[:, np.asarray([0, 2, 3, 4, 8, 10, 11, 12, 16, 18, 19, 20])]
    np.testing.assert_allclose(translated, point, rtol=2.0e-13, atol=2.0e-12)
    assert np.linalg.norm(point - gaussian) > 1.0e-2


def test_point_harmonic_operator_is_exactly_so3_covariant() -> None:
    rotation = _rotation(912)
    base = point_harmonic_source_operator(
        positions_angstrom=POSITIONS,
        radii_angstrom=RADII,
        surface_lmax=3,
        radial_quadrature_order=160,
    )
    rotated = point_harmonic_source_operator(
        positions_angstrom=POSITIONS @ rotation.T,
        radii_angstrom=RADII,
        surface_lmax=3,
        radial_quadrature_order=160,
    )
    harmonic_rotation = PerAtomHarmonicSpace(
        atom_count=len(POSITIONS), lmax=3
    ).representation_matrix(rotation)
    source_rotation = _point_source_rotation(rotation)
    np.testing.assert_allclose(
        rotated @ source_rotation,
        harmonic_rotation @ base,
        rtol=0.0,
        atol=4.0e-10,
    )


def test_point_harmonic_operator_fails_at_distinct_center_on_surface() -> None:
    positions = np.asarray([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    radii = np.asarray([1.5, 1.2])
    try:
        point_harmonic_source_operator(
            positions_angstrom=positions,
            radii_angstrom=radii,
            surface_lmax=1,
        )
    except ValueError as exc:
        assert "point source lies on a target sphere" in str(exc)
    else:  # pragma: no cover - fail-closed contract
        raise AssertionError("singular point-source geometry was accepted")
