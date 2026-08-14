from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from maple.solvation.continuum import (
    SmoothWeightedHarmonicGalerkinFunctionalCandidate,
    build_smooth_harmonic_exposure,
    build_smooth_weighted_harmonic_galerkin,
    radial_gto_source_rotation_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
POSITIONS = np.asarray([[0.0, 0.0, 0.0], [1.72, -0.31, 0.21], [-0.39, 1.65, -0.17]])
NUMBERS = (8, 1, 6)
RADII = (1.43, 1.18, 1.57)
SOURCE = np.asarray(
    [
        [0.20, -0.03, 0.02, -0.01, 0.01, 0.004, -0.006, 0.007],
        [-0.14, 0.02, -0.03, 0.02, -0.01, 0.005, 0.003, -0.004],
        [-0.06, 0.01, 0.01, -0.01, 0.02, -0.003, 0.002, -0.003],
    ]
)


def _rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(3, 3))
    rotation, triangular = np.linalg.qr(matrix)
    rotation = rotation @ np.diag(np.where(np.diag(triangular) < 0.0, -1.0, 1.0))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return rotation


@pytest.fixture(scope="module")
def functional():
    torch = pytest.importorskip("torch")
    return SmoothWeightedHarmonicGalerkinFunctionalCandidate(
        atomic_numbers=NUMBERS,
        radii_angstrom=RADII,
        transition_width_angstrom2=0.18,
        surface_lmax=1,
        exposure_lmax=2,
        exposure_radial_quadrature_order=32,
        source_radial_quadrature_order=32,
        green_radial_quadrature_order=32,
        dtype=torch.float64,
        device="cpu",
    )


def _numpy_assembly(positions: np.ndarray = POSITIONS):
    exposure = build_smooth_harmonic_exposure(
        atomic_numbers=NUMBERS,
        positions_angstrom=positions,
        radii_angstrom=RADII,
        transition_width_angstrom2=0.18,
        surface_lmax=1,
        exposure_lmax=2,
        radial_quadrature_order=32,
    )
    return build_smooth_weighted_harmonic_galerkin(
        exposure,
        source_radial_quadrature_order=32,
        green_radial_quadrature_order=32,
    )


def test_candidate_imports_without_eager_torch_or_legacy_continuum():
    script = r"""
import sys
sys.modules['torch'] = None
from maple.solvation.continuum.harmonic_torch_functional import (
    SmoothWeightedHarmonicGalerkinFunctionalCandidate,
)
assert SmoothWeightedHarmonicGalerkinFunctionalCandidate.capabilities.enabled_tiers == ()
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


def test_torch_geometry_matrices_match_independent_numpy_assembly(functional):
    actual = functional.debug_geometry_matrices(POSITIONS)
    expected = _numpy_assembly()
    references = {
        "weighted_basis": expected.weighted_basis_operator,
        "raw_single_layer": expected.raw_single_layer_operator,
        "raw_source": expected.source.raw_source_operator,
        "surface_operator": expected.surface_operator,
        "source_operator": expected.source_operator,
    }
    tolerances = {
        "weighted_basis": 2.0e-14,
        "raw_single_layer": 2.0e-12,
        "raw_source": 2.0e-13,
        "surface_operator": 3.0e-12,
        "source_operator": 3.0e-13,
    }
    for name, reference in references.items():
        np.testing.assert_allclose(
            actual[name], reference, atol=tolerances[name], rtol=0.0
        )

    right_hand_side = expected.source_operator @ SOURCE.reshape(-1)
    expected_energy = (
        -0.5
        * right_hand_side
        @ np.linalg.solve(expected.surface_operator, right_hand_side)
    )
    assert functional.energy_eV(POSITIONS, SOURCE) == pytest.approx(
        expected_energy, abs=2.0e-14
    )


def test_one_scalar_generates_source_and_coordinate_derivatives(functional):
    rng = np.random.default_rng(311)
    source_direction = rng.normal(scale=0.02, size=SOURCE.shape)
    field_cotangent = rng.normal(scale=0.03, size=SOURCE.shape)
    coordinate_direction = rng.normal(size=POSITIONS.shape)
    coordinate_direction -= np.mean(coordinate_direction, axis=0)
    coordinate_direction /= np.linalg.norm(coordinate_direction)

    source_step = 2.0e-6
    source_difference = (
        functional.energy_eV(POSITIONS, SOURCE + source_step * source_direction)
        - functional.energy_eV(POSITIONS, SOURCE - source_step * source_direction)
    ) / (2.0 * source_step)
    assert np.vdot(
        functional.drive(POSITIONS, SOURCE), source_direction
    ) == pytest.approx(source_difference, abs=2.0e-11)

    source_jvp = functional.source_jvp(POSITIONS, SOURCE, source_direction)
    source_vjp = functional.source_vjp(POSITIONS, SOURCE, field_cotangent)
    assert np.vdot(field_cotangent, source_jvp) == pytest.approx(
        np.vdot(source_vjp, source_direction), abs=3.0e-13
    )

    coordinate_gradient = functional.coordinate_partial(POSITIONS, SOURCE)
    analytic_coordinate = float(np.vdot(coordinate_gradient, coordinate_direction))
    coordinate_errors = []
    for step in (1.0e-3, 3.0e-4, 1.0e-4):
        finite_difference = (
            functional.energy_eV(POSITIONS + step * coordinate_direction, SOURCE)
            - functional.energy_eV(POSITIONS - step * coordinate_direction, SOURCE)
        ) / (2.0 * step)
        coordinate_errors.append(abs(analytic_coordinate - finite_difference))
    assert coordinate_errors[-1] < 3.0e-10
    assert coordinate_errors[-1] < coordinate_errors[0] / 20.0

    mixed = functional.mixed_coordinate_source_vjp(POSITIONS, SOURCE, field_cotangent)
    analytic_mixed = float(np.vdot(mixed, coordinate_direction))
    mixed_step = 3.0e-5
    finite_mixed = (
        np.vdot(
            field_cotangent,
            functional.drive(POSITIONS + mixed_step * coordinate_direction, SOURCE),
        )
        - np.vdot(
            field_cotangent,
            functional.drive(POSITIONS - mixed_step * coordinate_direction, SOURCE),
        )
    ) / (2.0 * mixed_step)
    assert analytic_mixed == pytest.approx(finite_mixed, abs=2.0e-9)


def test_energy_gradient_are_rotation_translation_and_permutation_covariant(
    functional,
):
    torch = pytest.importorskip("torch")
    rotation = _rotation(907)
    source_rotation = radial_gto_source_rotation_matrix(rotation, atom_count=3)
    rotated_positions = POSITIONS @ rotation.T
    rotated_source = (source_rotation @ SOURCE.reshape(-1)).reshape(SOURCE.shape)

    energy = functional.energy_eV(POSITIONS, SOURCE)
    gradient = functional.coordinate_partial(POSITIONS, SOURCE)
    rotated_energy = functional.energy_eV(rotated_positions, rotated_source)
    rotated_gradient = functional.coordinate_partial(rotated_positions, rotated_source)
    assert rotated_energy == pytest.approx(energy, abs=5.0e-14)
    np.testing.assert_allclose(
        rotated_gradient, gradient @ rotation.T, atol=8.0e-14, rtol=0.0
    )

    shift = np.asarray([0.73, -0.42, 0.19])
    assert functional.energy_eV(POSITIONS + shift, SOURCE) == pytest.approx(
        energy, abs=3.0e-14
    )
    np.testing.assert_allclose(
        functional.coordinate_partial(POSITIONS + shift, SOURCE),
        gradient,
        atol=8.0e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(np.sum(gradient, axis=0), 0.0, atol=3.0e-14, rtol=0.0)
    # A coordinate-only torque need not vanish while vector source
    # coefficients are held fixed in the laboratory frame.  The joint
    # geometry/source rotation above is the relevant covariance identity.

    permutation = np.asarray([2, 0, 1])
    permuted = SmoothWeightedHarmonicGalerkinFunctionalCandidate(
        atomic_numbers=tuple(NUMBERS[index] for index in permutation),
        radii_angstrom=tuple(RADII[index] for index in permutation),
        transition_width_angstrom2=0.18,
        surface_lmax=1,
        exposure_lmax=2,
        exposure_radial_quadrature_order=32,
        source_radial_quadrature_order=32,
        green_radial_quadrature_order=32,
        dtype=torch.float64,
        device="cpu",
    )
    assert permuted.energy_eV(
        POSITIONS[permutation], SOURCE[permutation]
    ) == pytest.approx(energy, abs=5.0e-14)
    np.testing.assert_allclose(
        permuted.coordinate_partial(POSITIONS[permutation], SOURCE[permutation]),
        gradient[permutation],
        atol=8.0e-14,
        rtol=0.0,
    )


def test_pair_axis_section_retains_the_transverse_derivative_at_both_poles():
    torch = pytest.importorskip("torch")
    positions = np.asarray([[0.0, 0.0, -0.9], [0.0, 0.0, 0.9]])
    source = np.asarray(
        [
            [0.1, 0.02, -0.01, 0.03, 0.01, -0.005, 0.002, 0.004],
            [-0.1, -0.01, 0.02, -0.02, -0.01, 0.003, -0.004, 0.002],
        ]
    )
    candidate = SmoothWeightedHarmonicGalerkinFunctionalCandidate(
        atomic_numbers=(1, 8),
        radii_angstrom=(1.2, 1.4),
        transition_width_angstrom2=0.18,
        surface_lmax=1,
        exposure_lmax=2,
        exposure_radial_quadrature_order=24,
        source_radial_quadrature_order=24,
        green_radial_quadrature_order=24,
        dtype=torch.float64,
        device="cpu",
    )
    direction = np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]) / np.sqrt(2.0)
    gradient = candidate.coordinate_partial(positions, source)
    analytic = float(np.vdot(gradient, direction))
    step = 1.0e-4
    finite_difference = (
        candidate.energy_eV(positions + step * direction, source)
        - candidate.energy_eV(positions - step * direction, source)
    ) / (2.0 * step)
    assert abs(analytic) > 1.0e-6
    assert analytic == pytest.approx(finite_difference, abs=2.0e-12)

    rotation = _rotation(919)
    source_rotation = radial_gto_source_rotation_matrix(rotation, atom_count=2)
    rotated_source = (source_rotation @ source.reshape(-1)).reshape(source.shape)
    rotated_gradient = candidate.coordinate_partial(
        positions @ rotation.T, rotated_source
    )
    np.testing.assert_allclose(
        rotated_gradient, gradient @ rotation.T, atol=3.0e-14, rtol=0.0
    )


def test_candidate_is_content_addressed_and_fail_closed(functional):
    assert len(functional.configuration_sha256()) == 64
    assert len(functional.provenance_sha256) == 64
    assert functional.capabilities.enabled_tiers == ()
    assert functional.full_geometry_intertwiner_assembly_available is True
    assert functional.moving_cavity_coordinate_derivative_available is True
    assert functional.derivatives_generated_from_same_scalar is True
    assert functional.laboratory_fixed_surface_grid is False
    assert functional.registered_scalar is False
    assert functional.tier_v_rotation_admitted is False
    assert functional.tier_v_admitted is False
    provenance = dict(functional.runtime_provenance())
    assert provenance["geometry_assembly"] == "same-scalar-E-K-V"
    assert provenance["tier_v_admission"] == "disabled"
    assert provenance["capabilities"] == "none"
    with pytest.raises(AttributeError, match="immutable"):
        functional._radii_angstrom = (2.0, 2.0, 2.0)
    with pytest.raises(ValueError, match="distinct"):
        functional.energy_eV(
            np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            SOURCE,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"radii_angstrom": (1.0, 1.0)}, "one value per atom"),
        ({"surface_lmax": 2, "exposure_lmax": 2}, "at least twice"),
        ({"transition_width_angstrom2": 0.0}, "positive"),
    ),
)
def test_invalid_configuration_fails_closed(kwargs, message):
    torch = pytest.importorskip("torch")
    options = dict(
        atomic_numbers=NUMBERS,
        radii_angstrom=RADII,
        transition_width_angstrom2=0.18,
        surface_lmax=1,
        exposure_lmax=2,
        exposure_radial_quadrature_order=16,
        source_radial_quadrature_order=16,
        green_radial_quadrature_order=16,
        dtype=torch.float64,
        device="cpu",
    )
    options.update(kwargs)
    with pytest.raises((ValueError, TypeError), match=message):
        SmoothWeightedHarmonicGalerkinFunctionalCandidate(**options)
