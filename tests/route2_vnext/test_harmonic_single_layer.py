from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.continuum.harmonic_coefficients import real_wigner_matrix
from maple.solvation.continuum.harmonic_single_layer import (
    canonical_harmonic_cross_block,
    harmonic_sphere_pair_topology,
    harmonic_single_layer_operator,
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


def _l0_cross(radius_i: float, radius_j: float, distance: float) -> float:
    """Independent Newton-shell result in inverse-Angstrom units."""

    if distance >= radius_i + radius_j:
        return 4.0 * np.pi / distance
    if distance + radius_i <= radius_j:
        return 4.0 * np.pi / radius_j
    if distance + radius_j <= radius_i:
        return 4.0 * np.pi / radius_i
    breakpoint = (radius_j**2 - distance**2 - radius_i**2) / (2.0 * distance * radius_i)
    return (
        2.0
        * np.pi
        * (
            (breakpoint + 1.0) / radius_j
            + (distance + radius_i - radius_j) / (distance * radius_i)
        )
    )


def test_imports_without_torch_or_legacy_continuum_adapters():
    script = r"""
import sys
sys.modules['torch'] = None
from maple.solvation.continuum.harmonic_single_layer import (
    canonical_harmonic_cross_block,
    harmonic_single_layer_operator,
)
assert callable(canonical_harmonic_cross_block)
assert callable(harmonic_single_layer_operator)
assert 'maple.solvation.continuum.fixed_topology_cpcm' not in sys.modules
assert 'maple.solvation.coupling.exact_gto' not in sys.modules
"""
    result = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("lmax", (0, 1, 2, 4))
def test_one_sphere_has_the_exact_single_layer_spectrum(lmax):
    from ase.units import Bohr

    radius = 1.37
    actual = harmonic_single_layer_operator(
        positions_angstrom=np.zeros((1, 3)),
        radii_angstrom=(radius,),
        lmax=lmax,
        radial_quadrature_order=48,
    )
    expected = np.zeros_like(actual)
    coulomb = HARTREE_TO_EV * Bohr
    for ell in range(lmax + 1):
        section = slice(ell * ell, (ell + 1) * (ell + 1))
        expected[section, section] = np.eye(2 * ell + 1) * (
            coulomb * 4.0 * np.pi / (radius * (2 * ell + 1))
        )
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0.0)


@pytest.mark.parametrize(
    "radius_i,radius_j,distance",
    (
        (1.0, 0.8, 0.1),
        (1.0, 0.8, 0.4),
        (1.0, 0.8, 1.0),
        (1.0, 0.8, 1.8),
        (1.0, 0.8, 2.4),
    ),
)
def test_l0_cross_block_matches_independent_newton_shell_formula(
    radius_i, radius_j, distance
):
    from ase.units import Bohr

    actual = canonical_harmonic_cross_block(
        target_radius_angstrom=radius_i,
        source_radius_angstrom=radius_j,
        distance_angstrom=distance,
        lmax=0,
        radial_quadrature_order=120,
    )
    expected = HARTREE_TO_EV * Bohr * _l0_cross(radius_i, radius_j, distance)
    assert actual[0, 0] == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize("lmax", (0, 1, 2, 3))
@pytest.mark.parametrize("distance", (0.4, 1.0, 1.8, 2.4))
def test_cross_block_reciprocity_holds_across_nesting_intersection_and_separation(
    lmax, distance
):
    forward = canonical_harmonic_cross_block(
        target_radius_angstrom=1.0,
        source_radius_angstrom=0.8,
        distance_angstrom=distance,
        lmax=lmax,
        radial_quadrature_order=180,
    )
    reverse_along_its_positive_axis = canonical_harmonic_cross_block(
        target_radius_angstrom=0.8,
        source_radius_angstrom=1.0,
        distance_angstrom=distance,
        lmax=lmax,
        radial_quadrature_order=180,
    )
    reverse_axis = np.diag([1.0, -1.0, -1.0])
    direction_change = real_wigner_matrix(reverse_axis, lmax=lmax)
    reverse_in_forward_frame = (
        direction_change @ reverse_along_its_positive_axis @ direction_change.T
    )
    np.testing.assert_allclose(
        forward, reverse_in_forward_frame.T, atol=7e-11, rtol=0.0
    )


def test_pair_axis_block_commutes_with_so2_and_has_no_transverse_frame_gauge():
    lmax = 4
    canonical = canonical_harmonic_cross_block(
        target_radius_angstrom=1.07,
        source_radius_angstrom=0.91,
        distance_angstrom=1.34,
        lmax=lmax,
        radial_quadrature_order=144,
    )
    angle = 0.731
    stabilizer = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    stabilizer_representation = real_wigner_matrix(stabilizer, lmax=lmax)
    np.testing.assert_allclose(
        stabilizer_representation @ canonical,
        canonical @ stabilizer_representation,
        atol=2e-11,
        rtol=0.0,
    )

    section = _rotation(83)
    first_representation = real_wigner_matrix(section, lmax=lmax)
    second_representation = real_wigner_matrix(section @ stabilizer, lmax=lmax)
    first = first_representation @ canonical @ first_representation.T
    second = second_representation @ canonical @ second_representation.T
    np.testing.assert_allclose(first, second, atol=3e-11, rtol=0.0)


@pytest.mark.parametrize("boundary", (0.2, 1.8))
def test_exact_shell_l0_block_is_c1_but_does_not_claim_c2_at_tangency(boundary):
    from maple.solvation.continuum.harmonic_single_layer import (
        COULOMB_EV_ANGSTROM_PER_E2,
    )

    radius_i = 1.0
    radius_j = 0.8

    def value(distance: float) -> float:
        return float(
            canonical_harmonic_cross_block(
                target_radius_angstrom=radius_i,
                source_radius_angstrom=radius_j,
                distance_angstrom=distance,
                lmax=0,
                radial_quadrature_order=256,
            )[0, 0]
            / COULOMB_EV_ANGSTROM_PER_E2
        )

    step = 2.0e-5
    left_first = (value(boundary) - value(boundary - step)) / step
    right_first = (value(boundary + step) - value(boundary)) / step
    assert abs(left_first - right_first) < 5.0e-4

    left_second = (
        value(boundary) - 2.0 * value(boundary - step) + value(boundary - 2.0 * step)
    ) / step**2
    right_second = (
        value(boundary + 2.0 * step) - 2.0 * value(boundary + step) + value(boundary)
    ) / step**2
    assert abs(left_second - right_second) > 1.0


def test_sphere_pair_topology_is_rigid_invariant_and_classifies_all_strata():
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [1.6, 0.0, 0.0],
            [3.2, 0.0, 0.0],
        ]
    )
    radii = (1.0, 0.2, 0.8, 0.3)
    base = harmonic_sphere_pair_topology(positions, radii)
    rotation = _rotation(20260815)
    rotated = harmonic_sphere_pair_topology(positions @ rotation.T, radii)
    translated = harmonic_sphere_pair_topology(
        positions + np.asarray([0.8, -0.4, 0.2]), radii
    )

    assert rotated.topology_sha256 == base.topology_sha256
    assert translated.topology_sha256 == base.topology_sha256
    assert (0, 1, "nested") in base.relations
    assert (0, 2, "intersecting") in base.relations
    assert (0, 3, "separated") in base.relations
    assert base.minimum_tangency_margin_angstrom == pytest.approx(0.2)


@pytest.mark.parametrize("distance", (0.2, 1.8))
def test_sphere_pair_topology_fails_closed_at_internal_and_external_tangency(
    distance,
):
    with pytest.raises(ValueError, match="tangency event surface"):
        harmonic_sphere_pair_topology(
            np.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
            (1.0, 0.8),
        )


def test_one_sphere_topology_has_no_pair_margin():
    topology = harmonic_sphere_pair_topology(np.zeros((1, 3)), (1.0,))
    assert topology.relations == ()
    assert topology.minimum_tangency_margin_angstrom is None


def test_geometry_assembled_operator_is_spd_and_so3_covariant_for_intersecting_spheres():
    positions = np.asarray([[0.0, 0.0, 0.0], [1.20, 0.31, -0.19], [-0.42, 1.11, 0.13]])
    radii = (1.10, 0.92, 1.03)
    lmax = 3
    base = harmonic_single_layer_operator(
        positions_angstrom=positions,
        radii_angstrom=radii,
        lmax=lmax,
        radial_quadrature_order=144,
    )
    assert np.linalg.eigvalsh(base)[0] > 0.0
    rotation = _rotation(20260814)
    rotated = harmonic_single_layer_operator(
        positions_angstrom=positions @ rotation.T,
        radii_angstrom=radii,
        lmax=lmax,
        radial_quadrature_order=144,
    )
    representation = np.kron(
        np.eye(len(radii)), real_wigner_matrix(rotation, lmax=lmax)
    )
    np.testing.assert_allclose(
        rotated,
        representation @ base @ representation.T,
        atol=3e-11,
        rtol=0.0,
    )


def test_translation_and_label_permutation_change_only_the_declared_coefficient_order():
    positions = np.asarray([[0.1, -0.2, 0.3], [1.35, 0.25, -0.15], [-0.31, 1.26, 0.18]])
    radii = np.asarray((1.08, 0.88, 1.01))
    lmax = 2
    base = harmonic_single_layer_operator(
        positions_angstrom=positions,
        radii_angstrom=radii,
        lmax=lmax,
        radial_quadrature_order=128,
    )
    translated = harmonic_single_layer_operator(
        positions_angstrom=positions + np.asarray([1.2, -0.7, 0.4]),
        radii_angstrom=radii,
        lmax=lmax,
        radial_quadrature_order=128,
    )
    np.testing.assert_allclose(translated, base, atol=4e-13, rtol=0.0)

    permutation = np.asarray([2, 0, 1])
    permuted = harmonic_single_layer_operator(
        positions_angstrom=positions[permutation],
        radii_angstrom=radii[permutation],
        lmax=lmax,
        radial_quadrature_order=128,
    )
    block = (lmax + 1) ** 2
    coefficient_permutation = np.zeros((len(radii) * block,) * 2)
    for new_atom, old_atom in enumerate(permutation):
        coefficient_permutation[
            new_atom * block : (new_atom + 1) * block,
            old_atom * block : (old_atom + 1) * block,
        ] = np.eye(block)
    np.testing.assert_allclose(
        permuted,
        coefficient_permutation @ base @ coefficient_permutation.T,
        atol=3e-11,
        rtol=0.0,
    )


def test_invariant_radial_quadrature_converges_without_changing_rotation_covariance():
    positions = np.asarray([[0.0, 0.0, 0.0], [1.37, -0.28, 0.16]])
    radii = (1.05, 0.91)
    low = harmonic_single_layer_operator(
        positions_angstrom=positions,
        radii_angstrom=radii,
        lmax=5,
        radial_quadrature_order=8,
    )
    medium = harmonic_single_layer_operator(
        positions_angstrom=positions,
        radii_angstrom=radii,
        lmax=5,
        radial_quadrature_order=16,
    )
    high = harmonic_single_layer_operator(
        positions_angstrom=positions,
        radii_angstrom=radii,
        lmax=5,
        radial_quadrature_order=64,
    )
    assert np.linalg.norm(medium - high) < np.linalg.norm(low - high)

    rotation = _rotation(91)
    rotated_low = harmonic_single_layer_operator(
        positions_angstrom=positions @ rotation.T,
        radii_angstrom=radii,
        lmax=5,
        radial_quadrature_order=8,
    )
    representation = np.kron(np.eye(2), real_wigner_matrix(rotation, lmax=5))
    np.testing.assert_allclose(
        rotated_low,
        representation @ low @ representation.T,
        atol=2e-11,
        rtol=0.0,
    )


def test_coincident_centres_and_nonphysical_inputs_fail_closed():
    with pytest.raises(ValueError, match="distinct"):
        harmonic_single_layer_operator(
            positions_angstrom=np.zeros((2, 3)),
            radii_angstrom=(1.0, 0.8),
            lmax=2,
        )
    with pytest.raises(ValueError, match="radii_angstrom"):
        harmonic_single_layer_operator(
            positions_angstrom=np.zeros((1, 3)),
            radii_angstrom=(0.0,),
            lmax=2,
        )
    with pytest.raises(ValueError, match="distance_angstrom"):
        canonical_harmonic_cross_block(
            target_radius_angstrom=1.0,
            source_radius_angstrom=0.8,
            distance_angstrom=0.0,
            lmax=2,
        )
