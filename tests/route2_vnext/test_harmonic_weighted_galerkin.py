from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import erf

from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.continuum import (
    FixedHarmonicGalerkinCPCMCandidate,
    SmoothWeightedHarmonicGalerkinAssembly,
    build_smooth_harmonic_exposure,
    build_smooth_weighted_harmonic_galerkin,
    gaussian_harmonic_source_operator,
    radial_gto_source_rotation_matrix,
)
from maple.solvation.continuum.harmonic_single_layer import (
    COULOMB_EV_ANGSTROM_PER_E2,
    harmonic_single_layer_operator,
)

ROOT = Path(__file__).resolve().parents[2]
POSITIONS = np.asarray([[0.0, 0.0, 0.0], [1.72, -0.31, 0.21], [-0.39, 1.65, -0.17]])
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
    q, r = np.linalg.qr(matrix)
    q = q @ np.diag(np.where(np.diag(r) < 0.0, -1.0, 1.0))
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return q


def _exposure(positions: np.ndarray = POSITIONS):
    return build_smooth_harmonic_exposure(
        atomic_numbers=(8, 1, 6),
        positions_angstrom=positions,
        radii_angstrom=RADII,
        transition_width_angstrom2=0.18,
        surface_lmax=1,
        exposure_lmax=2,
        radial_quadrature_order=96,
    )


def _assembly(positions: np.ndarray = POSITIONS):
    return build_smooth_weighted_harmonic_galerkin(
        _exposure(positions),
        source_radial_quadrature_order=112,
        green_radial_quadrature_order=112,
    )


def test_imports_without_torch_or_legacy_continuum_adapters():
    script = r"""
import sys
sys.modules['torch'] = None
from maple.solvation.continuum.harmonic_weighted_galerkin import (
    SmoothWeightedHarmonicGalerkinAssembly,
    build_smooth_weighted_harmonic_galerkin,
)
assert SmoothWeightedHarmonicGalerkinAssembly.__name__.startswith('SmoothWeighted')
assert callable(build_smooth_weighted_harmonic_galerkin)
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


def test_geometry_assembly_is_the_rectangular_weighted_coulomb_galerkin_form():
    assembly = _assembly()
    expected_green = harmonic_single_layer_operator(
        positions_angstrom=POSITIONS,
        radii_angstrom=RADII,
        lmax=assembly.physical_charge_space.lmax,
        radial_quadrature_order=assembly.green_radial_quadrature_order,
    )
    expected_surface = (
        assembly.weighted_basis_operator.T
        @ expected_green
        @ assembly.weighted_basis_operator
    )
    np.testing.assert_allclose(
        expected_surface, expected_surface.T, atol=2.0e-13, rtol=0.0
    )
    expected_surface = 0.5 * (expected_surface + expected_surface.T)
    np.testing.assert_allclose(
        assembly.raw_single_layer_operator, expected_green, atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        assembly.surface_operator, expected_surface, atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        assembly.source_operator, assembly.source.source_operator, atol=0.0, rtol=0.0
    )
    assert assembly.minimum_basis_singular_value > 0.0
    assert assembly.minimum_raw_eigenvalue > 0.0
    assert assembly.minimum_surface_eigenvalue > 0.0
    assert assembly.fixed_snapshot.minimum_eigenvalue == pytest.approx(
        assembly.minimum_surface_eigenvalue, abs=1e-14
    )


def test_square_exposure_sandwich_is_rejected_by_the_physics_canary():
    exposure = _exposure()
    square = exposure.global_multiplication_operator
    low_green = harmonic_single_layer_operator(
        positions_angstrom=POSITIONS,
        radii_angstrom=RADII,
        lmax=exposure.surface_lmax,
        radial_quadrature_order=112,
    )
    low_source = gaussian_harmonic_source_operator(
        positions_angstrom=POSITIONS,
        radii_angstrom=RADII,
        surface_lmax=exposure.surface_lmax,
        radial_quadrature_order=112,
    )
    source_vector = SOURCE.reshape(-1)
    unweighted_rhs = low_source @ source_vector
    square_rhs = square @ unweighted_rhs
    unweighted_energy = (
        -0.5 * unweighted_rhs @ np.linalg.solve(low_green, unweighted_rhs)
    )
    square_energy = (
        -0.5 * square_rhs @ np.linalg.solve(square @ low_green @ square, square_rhs)
    )
    assert square_energy == pytest.approx(unweighted_energy, abs=4e-14)

    assembly = _assembly()
    rectangular_rhs = assembly.source_operator @ source_vector
    rectangular_energy = (
        -0.5
        * rectangular_rhs
        @ np.linalg.solve(assembly.surface_operator, rectangular_rhs)
    )
    assert abs(rectangular_energy - unweighted_energy) > 1.0e-5


def test_one_sphere_matches_the_analytic_gaussian_conductor_energy():
    from ase.units import Bohr

    radius = 1.6
    exposure = build_smooth_harmonic_exposure(
        atomic_numbers=(1,),
        positions_angstrom=np.zeros((1, 3)),
        radii_angstrom=(radius,),
        transition_width_angstrom2=0.2,
        surface_lmax=1,
        exposure_lmax=2,
        radial_quadrature_order=64,
    )
    assembly = build_smooth_weighted_harmonic_galerkin(
        exposure,
        source_radial_quadrature_order=96,
        green_radial_quadrature_order=64,
    )
    source = np.zeros((1, 8))
    source[0, 0] = 1.0
    actual = FixedHarmonicGalerkinCPCMCandidate(
        assembly.fixed_snapshot,
        dtype=pytest.importorskip("torch").float64,
        device="cpu",
    ).energy_eV(np.zeros((1, 3)), source)
    sigma = 1.5
    potential = (
        HARTREE_TO_EV
        * erf((radius / Bohr) / (np.sqrt(2.0) * (sigma / Bohr)))
        / (radius / Bohr)
    )
    rhs_l0 = np.sqrt(4.0 * np.pi) * potential
    surface_l0 = COULOMB_EV_ANGSTROM_PER_E2 * 4.0 * np.pi / radius
    expected = -0.5 * rhs_l0**2 / surface_l0
    assert actual == pytest.approx(expected, abs=2e-13)


def test_stationary_scalar_drive_and_exact_receiver_come_from_one_snapshot():
    torch = pytest.importorskip("torch")
    assembly = _assembly()
    functional = FixedHarmonicGalerkinCPCMCandidate(
        assembly.fixed_snapshot, dtype=torch.float64, device="cpu"
    )
    direction = np.linspace(-0.02, 0.03, SOURCE.size).reshape(SOURCE.shape)
    step = 1.0e-6
    finite_difference = (
        functional.energy_eV(POSITIONS, SOURCE + step * direction)
        - functional.energy_eV(POSITIONS, SOURCE - step * direction)
    ) / (2.0 * step)
    assert np.vdot(functional.drive(POSITIONS, SOURCE), direction) == pytest.approx(
        finite_difference, abs=3e-11
    )
    cotangent = np.linspace(0.03, -0.01, SOURCE.size).reshape(SOURCE.shape)
    jvp = functional.source_jvp(POSITIONS, SOURCE, direction)
    vjp = functional.source_vjp(POSITIONS, SOURCE, cotangent)
    assert np.vdot(cotangent, jvp) == pytest.approx(np.vdot(vjp, direction), abs=3e-13)


def test_full_geometry_assembly_is_rotation_and_translation_covariant():
    torch = pytest.importorskip("torch")
    base = _assembly()
    rotation = _rotation(121)
    rotated = _assembly(POSITIONS @ rotation.T)
    translated = _assembly(POSITIONS + np.asarray([1.2, -0.7, 0.4]))
    basis_rotation = base.basis_space.representation_matrix(rotation)
    physical_rotation = base.physical_charge_space.representation_matrix(rotation)
    source_rotation = radial_gto_source_rotation_matrix(rotation, atom_count=len(RADII))
    np.testing.assert_allclose(
        rotated.raw_single_layer_operator,
        physical_rotation @ base.raw_single_layer_operator @ physical_rotation.T,
        atol=8e-11,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        rotated.weighted_basis_operator @ basis_rotation,
        physical_rotation @ base.weighted_basis_operator,
        atol=7e-11,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        rotated.surface_operator,
        basis_rotation @ base.surface_operator @ basis_rotation.T,
        atol=2e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        rotated.source_operator @ source_rotation,
        basis_rotation @ base.source_operator,
        atol=8e-11,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        translated.surface_operator, base.surface_operator, atol=5e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        translated.source_operator, base.source_operator, atol=5e-12, rtol=0.0
    )
    base_functional = FixedHarmonicGalerkinCPCMCandidate(
        base.fixed_snapshot, dtype=torch.float64, device="cpu"
    )
    rotated_functional = FixedHarmonicGalerkinCPCMCandidate(
        rotated.fixed_snapshot, dtype=torch.float64, device="cpu"
    )
    rotated_source = (source_rotation @ SOURCE.reshape(-1)).reshape(SOURCE.shape)
    assert rotated_functional.energy_eV(
        POSITIONS @ rotation.T, rotated_source
    ) == pytest.approx(base_functional.energy_eV(POSITIONS, SOURCE), abs=3.0e-13)


def test_external_tangency_sweep_keeps_fixed_dimension_and_a_c1_scalar():
    torch = pytest.importorskip("torch")
    radii = (1.0, 0.8)
    boundary = sum(radii)
    source = np.zeros((2, 8))
    source[:, 0] = (0.20, -0.20)

    def energy(distance: float) -> tuple[float, int, float]:
        positions = np.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
        exposure = build_smooth_harmonic_exposure(
            atomic_numbers=(1, 1),
            positions_angstrom=positions,
            radii_angstrom=radii,
            transition_width_angstrom2=0.12,
            surface_lmax=1,
            exposure_lmax=2,
            radial_quadrature_order=96,
        )
        assembly = build_smooth_weighted_harmonic_galerkin(
            exposure,
            source_radial_quadrature_order=112,
            green_radial_quadrature_order=160,
        )
        functional = FixedHarmonicGalerkinCPCMCandidate(
            assembly.fixed_snapshot, dtype=torch.float64, device="cpu"
        )
        return (
            functional.energy_eV(positions, source),
            assembly.basis_space.dimension,
            assembly.minimum_surface_eigenvalue,
        )

    def derivative_mismatch(step: float) -> tuple[float, tuple[int, ...]]:
        values = [energy(boundary + offset * step) for offset in (-1, 0, 1)]
        left = (values[1][0] - values[0][0]) / step
        right = (values[2][0] - values[1][0]) / step
        return abs(left - right), tuple(value[1] for value in values)

    coarse, coarse_dimensions = derivative_mismatch(2.0e-3)
    fine, fine_dimensions = derivative_mismatch(1.0e-3)
    assert coarse_dimensions == fine_dimensions == (8, 8, 8)
    assert fine < 0.55 * coarse
    assert min(energy(boundary + offset)[2] for offset in (-0.4, 0.0, 0.4)) > 0.0


def test_fully_buried_chart_and_tampered_state_fail_closed():
    buried = build_smooth_harmonic_exposure(
        atomic_numbers=(1, 8),
        positions_angstrom=np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        radii_angstrom=(0.4, 2.0),
        transition_width_angstrom2=0.1,
        surface_lmax=1,
        exposure_lmax=2,
        radial_quadrature_order=96,
    )
    with pytest.raises(ValueError, match="weighted basis"):
        build_smooth_weighted_harmonic_galerkin(
            buried,
            source_radial_quadrature_order=96,
            green_radial_quadrature_order=96,
        )

    assembly = _assembly()
    replay = _assembly()
    assert assembly.state_sha256 == replay.state_sha256
    assert assembly.coordinate_derivative_available is False
    assert assembly.tier_v_admitted is False
    changed = list(assembly._raw_single_layer_values)
    changed[0] += 1.0e-4
    object.__setattr__(assembly, "_raw_single_layer_values", tuple(changed))
    with pytest.raises(RuntimeError, match="drifted"):
        assembly.validate()


def test_assembly_requires_the_exact_snapshot_type():
    with pytest.raises(TypeError, match="SmoothHarmonicExposureSnapshot"):
        SmoothWeightedHarmonicGalerkinAssembly(object())
