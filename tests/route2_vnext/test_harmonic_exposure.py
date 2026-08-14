from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from maple.solvation.continuum import (
    SmoothHarmonicExposureSnapshot,
    build_smooth_harmonic_exposure,
    harmonic_multiplication_matrix,
    project_harmonic_product,
    real_wigner_matrix,
    smooth_flat_step,
    smooth_pair_exposure_coefficients,
)

ROOT = Path(__file__).resolve().parents[2]


def _rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    q = q @ np.diag(np.where(np.diag(r) < 0.0, -1.0, 1.0))
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return q


def _pair_coefficients(displacement: np.ndarray, *, lmax: int = 4) -> np.ndarray:
    return smooth_pair_exposure_coefficients(
        center_i=np.zeros(3),
        radius_i_angstrom=1.42,
        center_j=np.asarray(displacement, dtype=float),
        radius_j_angstrom=1.31,
        transition_width_angstrom2=0.18,
        lmax=lmax,
        radial_quadrature_order=128,
    )


def _snapshot(positions: np.ndarray | None = None) -> SmoothHarmonicExposureSnapshot:
    if positions is None:
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.15, -0.35, 0.22],
                [-0.48, 2.08, -0.31],
            ]
        )
    return build_smooth_harmonic_exposure(
        atomic_numbers=(8, 1, 6),
        positions_angstrom=positions,
        radii_angstrom=(1.52, 1.21, 1.70),
        transition_width_angstrom2=0.20,
        surface_lmax=2,
        exposure_lmax=4,
        radial_quadrature_order=128,
    )


def test_harmonic_exposure_imports_without_torch():
    script = r"""
import sys
sys.modules['torch'] = None
from maple.solvation.continuum.harmonic_exposure import (
    SmoothHarmonicExposureSnapshot,
    smooth_flat_step,
)
assert SmoothHarmonicExposureSnapshot.__name__ == 'SmoothHarmonicExposureSnapshot'
assert smooth_flat_step(0.0) == 0.5
"""
    result = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_flat_step_has_exact_compact_plateaus_and_symmetric_transition():
    points = np.asarray([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 4.0])
    values = smooth_flat_step(points)
    np.testing.assert_array_equal(values[[0, 1]], np.zeros(2))
    np.testing.assert_array_equal(values[[-2, -1]], np.ones(2))
    assert values[3] == pytest.approx(0.5, abs=0.0)
    np.testing.assert_allclose(values + smooth_flat_step(-points), 1.0, atol=0.0)
    assert smooth_flat_step(0.0) == 0.5


def test_pair_factor_screens_exactly_exposed_and_buried_limits():
    constant = smooth_pair_exposure_coefficients(
        center_i=np.zeros(3),
        radius_i_angstrom=1.0,
        center_j=np.asarray([5.0, 0.0, 0.0]),
        radius_j_angstrom=1.0,
        transition_width_angstrom2=0.1,
        lmax=4,
    )
    expected = np.zeros(25)
    expected[0] = np.sqrt(4.0 * np.pi)
    np.testing.assert_array_equal(constant, expected)

    buried = smooth_pair_exposure_coefficients(
        center_i=np.zeros(3),
        radius_i_angstrom=0.5,
        center_j=np.asarray([0.5, 0.0, 0.0]),
        radius_j_angstrom=2.0,
        transition_width_angstrom2=0.1,
        lmax=4,
    )
    np.testing.assert_array_equal(buried, np.zeros(25))


@pytest.mark.parametrize("lmax", (0, 1, 2, 4, 6))
def test_pair_coefficients_transform_as_complete_irreps(lmax):
    displacement = np.asarray([1.86, -0.52, 0.73])
    rotation = _rotation(41 + lmax)
    base = _pair_coefficients(displacement, lmax=lmax)
    rotated = _pair_coefficients(rotation @ displacement, lmax=lmax)
    expected = real_wigner_matrix(rotation, lmax=lmax) @ base
    np.testing.assert_allclose(rotated, expected, atol=2.5e-12, rtol=0.0)


def test_exact_finite_band_product_is_commutative_and_covariant():
    lmax = 4
    rotation = _rotation(53)
    first = _pair_coefficients(np.asarray([1.94, 0.24, -0.61]), lmax=lmax)
    second = _pair_coefficients(np.asarray([-0.28, 2.01, 0.43]), lmax=lmax)
    base = project_harmonic_product((first, second), lmax=lmax)
    reverse = project_harmonic_product((second, first), lmax=lmax)
    representation = real_wigner_matrix(rotation, lmax=lmax)
    rotated = project_harmonic_product(
        (representation @ first, representation @ second), lmax=lmax
    )
    np.testing.assert_allclose(reverse, base, atol=3e-14, rtol=0.0)
    np.testing.assert_allclose(rotated, representation @ base, atol=3.5e-12, rtol=0.0)


def test_constant_factor_is_identity_for_product_and_multiplication():
    lmax = 4
    coefficients = _pair_coefficients(np.asarray([1.86, -0.52, 0.73]), lmax=lmax)
    constant = np.zeros((lmax + 1) ** 2)
    constant[0] = np.sqrt(4.0 * np.pi)
    np.testing.assert_allclose(
        project_harmonic_product((coefficients, constant), lmax=lmax),
        coefficients,
        atol=8e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        harmonic_multiplication_matrix(constant, exposure_lmax=lmax, surface_lmax=2),
        np.eye(9),
        atol=8e-14,
        rtol=0.0,
    )


def test_multiplication_matrix_is_symmetric_and_covariant():
    surface_lmax = 2
    exposure_lmax = 4
    rotation = _rotation(61)
    coefficients = project_harmonic_product(
        (
            _pair_coefficients(np.asarray([1.92, -0.21, 0.55])),
            _pair_coefficients(np.asarray([-0.44, 2.05, -0.36])),
        ),
        lmax=exposure_lmax,
    )
    base = harmonic_multiplication_matrix(
        coefficients,
        exposure_lmax=exposure_lmax,
        surface_lmax=surface_lmax,
    )
    exposure_rotation = real_wigner_matrix(rotation, lmax=exposure_lmax)
    surface_rotation = real_wigner_matrix(rotation, lmax=surface_lmax)
    rotated = harmonic_multiplication_matrix(
        exposure_rotation @ coefficients,
        exposure_lmax=exposure_lmax,
        surface_lmax=surface_lmax,
    )
    np.testing.assert_allclose(base, base.T, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        rotated,
        surface_rotation @ base @ surface_rotation.T,
        atol=4e-12,
        rtol=0.0,
    )


def test_snapshot_is_fixed_dimensional_content_addressed_and_cold_replayable():
    snapshot = _snapshot()
    replay = _snapshot()
    assert snapshot.state_sha256 == replay.state_sha256
    assert snapshot.configuration_sha256 == replay.configuration_sha256
    assert snapshot.topology_sha256 == replay.topology_sha256
    assert snapshot.surface_space.dimension == 3 * 9
    assert snapshot.exposure_coefficients.shape == (3, 25)
    assert snapshot.multiplication_matrices.shape == (3, 9, 9)
    assert snapshot.global_multiplication_operator.shape == (27, 27)
    assert snapshot.exposure_coefficients.flags.writeable is False
    assert snapshot.multiplication_matrices.flags.writeable is False
    snapshot.validate()

    changed = list(snapshot._exposure_values)
    changed[0] += 1.0e-4
    object.__setattr__(snapshot, "_exposure_values", tuple(changed))
    with pytest.raises(RuntimeError, match="drifted"):
        snapshot.validate()


def test_snapshot_rotation_translation_and_permutation_covariance():
    base = _snapshot()
    rotation = _rotation(71)
    translation = np.asarray([1.3, -0.7, 0.4])
    rotated = _snapshot(base.positions_angstrom @ rotation.T)
    translated = _snapshot(base.positions_angstrom + translation)
    exposure_rotation = real_wigner_matrix(rotation, lmax=base.exposure_lmax)
    surface_rotation = real_wigner_matrix(rotation, lmax=base.surface_lmax)
    for atom in range(base.atom_count):
        np.testing.assert_allclose(
            rotated.exposure_coefficients[atom],
            exposure_rotation @ base.exposure_coefficients[atom],
            atol=6e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            rotated.multiplication_matrices[atom],
            surface_rotation @ base.multiplication_matrices[atom] @ surface_rotation.T,
            atol=7e-12,
            rtol=0.0,
        )
    np.testing.assert_allclose(
        translated.exposure_coefficients,
        base.exposure_coefficients,
        atol=3e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        translated.multiplication_matrices,
        base.multiplication_matrices,
        atol=3e-14,
        rtol=0.0,
    )
    assert rotated.topology_sha256 == base.topology_sha256
    assert translated.topology_sha256 == base.topology_sha256
    assert rotated.configuration_sha256 == base.configuration_sha256
    assert translated.configuration_sha256 == base.configuration_sha256

    permutation = np.asarray([2, 0, 1])
    permuted = build_smooth_harmonic_exposure(
        atomic_numbers=tuple(np.asarray(base.atomic_numbers)[permutation]),
        positions_angstrom=base.positions_angstrom[permutation],
        radii_angstrom=tuple(np.asarray(base.radii_angstrom)[permutation]),
        transition_width_angstrom2=base.transition_width_angstrom2,
        surface_lmax=base.surface_lmax,
        exposure_lmax=base.exposure_lmax,
        radial_quadrature_order=base.radial_quadrature_order,
    )
    np.testing.assert_allclose(
        permuted.exposure_coefficients,
        base.exposure_coefficients[permutation],
        atol=5e-13,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        permuted.multiplication_matrices,
        base.multiplication_matrices[permutation],
        atol=6e-13,
        rtol=0.0,
    )


def test_smooth_pair_tangency_has_convergent_coordinate_derivative():
    def coefficient(distance: float) -> float:
        values = smooth_pair_exposure_coefficients(
            center_i=np.zeros(3),
            radius_i_angstrom=1.0,
            center_j=np.asarray([distance, 0.0, 0.0]),
            radius_j_angstrom=1.0,
            transition_width_angstrom2=0.20,
            lmax=4,
            radial_quadrature_order=192,
        )
        return float(values[0])

    center = 2.0
    derivatives = []
    for step in (2.0e-3, 1.0e-3, 5.0e-4):
        derivatives.append(
            (coefficient(center + step) - coefficient(center - step)) / (2.0 * step)
        )
    assert np.isfinite(derivatives).all()
    assert abs(derivatives[2] - derivatives[1]) < 0.35 * abs(
        derivatives[1] - derivatives[0]
    )


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"transition_width_angstrom2": 0.0}, "transition_width_angstrom2"),
        ({"surface_lmax": 3, "exposure_lmax": 4}, "at least twice"),
        ({"surface_lmax": 9, "exposure_lmax": 18}, "bounded harmonic"),
        ({"radial_quadrature_order": 4097}, "bounded contract"),
    ),
)
def test_snapshot_invalid_contracts_fail_closed(kwargs, message):
    arguments = dict(
        atomic_numbers=(8, 1),
        positions_angstrom=np.asarray([[0.0, 0.0, 0.0], [1.8, 0.2, -0.1]]),
        radii_angstrom=(1.5, 1.2),
        transition_width_angstrom2=0.2,
        surface_lmax=2,
        exposure_lmax=4,
        radial_quadrature_order=96,
    )
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        build_smooth_harmonic_exposure(**arguments)


def test_pair_and_product_invalid_inputs_fail_closed():
    with pytest.raises(ValueError, match="distinct"):
        smooth_pair_exposure_coefficients(
            center_i=np.zeros(3),
            radius_i_angstrom=1.0,
            center_j=np.zeros(3),
            radius_j_angstrom=1.0,
            transition_width_angstrom2=0.2,
            lmax=2,
        )
    with pytest.raises(ValueError, match="shape"):
        project_harmonic_product((np.zeros(8),), lmax=2)
    with pytest.raises(ValueError, match="at least twice"):
        harmonic_multiplication_matrix(np.zeros(9), exposure_lmax=2, surface_lmax=2)
