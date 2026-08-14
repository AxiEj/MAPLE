from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.gto_density import (
    gaussian_multipole_potential,
)
from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.continuum import (
    build_harmonic_gaussian_source,
    build_smooth_harmonic_exposure,
    gaussian_harmonic_source_operator,
    radial_gto_source_rotation_matrix,
)
from maple.solvation.continuum.harmonic_coefficients import _real_harmonic_design

ROOT = Path(__file__).resolve().parents[2]
POSITIONS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [2.18, -0.37, 0.29],
        [-0.42, 2.06, -0.33],
    ]
)
RADII = (1.52, 1.21, 1.70)
SOURCE = np.asarray(
    [
        [0.21, -0.03, 0.04, -0.02, 0.01, 0.006, -0.004, 0.009],
        [-0.16, 0.02, -0.03, 0.01, 0.02, -0.008, 0.005, -0.006],
        [-0.05, 0.01, 0.02, 0.03, -0.01, 0.002, -0.007, 0.004],
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
        transition_width_angstrom2=0.20,
        surface_lmax=3,
        exposure_lmax=6,
        radial_quadrature_order=144,
    )


def _snapshot(positions: np.ndarray = POSITIONS):
    return build_harmonic_gaussian_source(_exposure(positions))


def _sphere_rule(order: int = 96) -> tuple[np.ndarray, np.ndarray]:
    cosine, polar_weights = np.polynomial.legendre.leggauss(order)
    phi_count = 2 * order + 1
    phi = 2.0 * np.pi * np.arange(phi_count) / phi_count
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
    directions = np.stack(
        (
            np.repeat(sine, phi_count) * np.tile(np.cos(phi), order),
            np.repeat(sine, phi_count) * np.tile(np.sin(phi), order),
            np.repeat(cosine, phi_count),
        ),
        axis=1,
    )
    weights = np.repeat(polar_weights * (2.0 * np.pi / phi_count), phi_count)
    return directions, weights


def _direct_projected_potential(source: np.ndarray, *, lmax: int) -> np.ndarray:
    from ase.units import Bohr

    directions, weights = _sphere_rule()
    design = _real_harmonic_design(directions, lmax=lmax)
    result = np.empty((len(POSITIONS), (lmax + 1) ** 2))
    for target, (center, radius) in enumerate(zip(POSITIONS, RADII, strict=True)):
        points_angstrom = center + radius * directions
        points_bohr = points_angstrom / Bohr
        potential = np.zeros(len(directions))
        for sigma, columns in (
            (1.5, (0, 2, 3, 4)),
            (3.0, (1, 5, 6, 7)),
        ):
            potential += gaussian_multipole_potential(
                points_bohr,
                POSITIONS,
                source[:, columns],
                sigma_angstrom=sigma,
            )
        result[target] = design.T @ (weights * potential * HARTREE_TO_EV)
    return result


def test_harmonic_gaussian_source_imports_without_torch_or_legacy_adapter():
    script = r"""
import sys
sys.modules['torch'] = None
from maple.solvation.continuum.harmonic_gaussian_source import (
    HarmonicGaussianSourceSnapshot,
    gaussian_harmonic_source_operator,
)
assert HarmonicGaussianSourceSnapshot.__name__ == 'HarmonicGaussianSourceSnapshot'
assert callable(gaussian_harmonic_source_operator)
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


def test_raw_operator_matches_independent_point_kernel_projection():
    snapshot = _snapshot()
    actual = (snapshot.raw_source_operator @ SOURCE.reshape(-1)).reshape(
        len(POSITIONS), -1
    )
    expected = _direct_projected_potential(SOURCE, lmax=snapshot.physical_charge_lmax)
    np.testing.assert_allclose(actual, expected, atol=2.5e-11, rtol=0.0)


def test_dipole_columns_are_analytic_source_center_derivatives():
    surface_lmax = 4
    base = gaussian_harmonic_source_operator(
        positions_angstrom=POSITIONS,
        radii_angstrom=RADII,
        surface_lmax=surface_lmax,
        radial_quadrature_order=160,
    )
    block = slice(0, (surface_lmax + 1) ** 2)
    source_atom = 1
    monopole_column = source_atom * 8
    raw_dipole_columns = (source_atom * 8 + 4, source_atom * 8 + 2, source_atom * 8 + 3)
    step = 2.0e-5
    for axis, dipole_column in enumerate(raw_dipole_columns):
        plus_positions = POSITIONS.copy()
        minus_positions = POSITIONS.copy()
        plus_positions[source_atom, axis] += step
        minus_positions[source_atom, axis] -= step
        plus = gaussian_harmonic_source_operator(
            positions_angstrom=plus_positions,
            radii_angstrom=RADII,
            surface_lmax=surface_lmax,
            radial_quadrature_order=160,
        )
        minus = gaussian_harmonic_source_operator(
            positions_angstrom=minus_positions,
            radii_angstrom=RADII,
            surface_lmax=surface_lmax,
            radial_quadrature_order=160,
        )
        finite_difference = (
            plus[block, monopole_column] - minus[block, monopole_column]
        ) / (2.0 * step)
        np.testing.assert_allclose(
            base[block, dipole_column],
            finite_difference,
            atol=1.5e-8,
            rtol=2e-8,
        )


def test_weighted_source_map_is_one_exact_adjoint_pair():
    snapshot = _snapshot()
    expected = snapshot.weighted_basis_operator.T @ snapshot.raw_source_operator
    np.testing.assert_allclose(snapshot.source_operator, expected, atol=0.0, rtol=0.0)
    assert snapshot.weighted_basis_operator.shape == (
        snapshot.physical_charge_space.dimension,
        snapshot.surface_space.dimension,
    )
    assert snapshot.physical_charge_lmax == (
        snapshot.exposure_lmax + snapshot.surface_lmax
    )
    surface_cotangent = np.linspace(-0.03, 0.04, snapshot.surface_space.dimension)
    source_direction = np.linspace(-0.02, 0.01, SOURCE.size).reshape(SOURCE.shape)
    source_image = snapshot.apply_source(source_direction)
    source_cotangent = snapshot.apply_adjoint(surface_cotangent)
    assert np.vdot(surface_cotangent, source_image) == pytest.approx(
        np.vdot(source_cotangent, source_direction), abs=2e-14
    )
    np.testing.assert_array_equal(snapshot.source_jvp(source_direction), source_image)
    np.testing.assert_array_equal(
        snapshot.source_vjp(surface_cotangent), source_cotangent
    )


def test_geometry_assembled_source_map_is_so3_covariant():
    base = _snapshot()
    rotation = _rotation(101)
    rotated = _snapshot(POSITIONS @ rotation.T)
    surface_rotation = base.surface_space.representation_matrix(rotation)
    physical_rotation = base.physical_charge_space.representation_matrix(rotation)
    source_rotation = radial_gto_source_rotation_matrix(
        rotation, atom_count=len(POSITIONS)
    )
    np.testing.assert_allclose(
        rotated.raw_source_operator @ source_rotation,
        physical_rotation @ base.raw_source_operator,
        atol=5e-11,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        rotated.weighted_basis_operator @ surface_rotation,
        physical_rotation @ base.weighted_basis_operator,
        atol=5e-11,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        rotated.source_operator @ source_rotation,
        surface_rotation @ base.source_operator,
        atol=3e-11,
        rtol=0.0,
    )
    rotated_source = (source_rotation @ SOURCE.reshape(-1)).reshape(SOURCE.shape)
    np.testing.assert_allclose(
        rotated.apply_source(rotated_source),
        surface_rotation @ base.apply_source(SOURCE),
        atol=3e-11,
        rtol=0.0,
    )


def test_source_snapshot_is_translation_invariant_immutable_and_bound():
    base = _snapshot()
    translated = _snapshot(POSITIONS + np.asarray([1.2, -0.8, 0.35]))
    np.testing.assert_allclose(
        translated.raw_source_operator,
        base.raw_source_operator,
        atol=2e-13,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        translated.source_operator,
        base.source_operator,
        atol=2e-13,
        rtol=0.0,
    )
    assert translated.configuration_sha256 == base.configuration_sha256
    assert translated.state_sha256 != base.state_sha256
    assert base.source_operator.flags.writeable is False
    assert base.raw_source_operator.flags.writeable is False
    assert base.weighted_basis_operator.flags.writeable is False
    assert base.coordinate_derivative_available is False
    assert base.tier_v_admitted is False
    base.validate()

    changed = list(base._source_values)
    changed[0] += 1.0e-5
    object.__setattr__(base, "_source_values", tuple(changed))
    with pytest.raises(RuntimeError, match="drifted"):
        base.validate()


def test_invalid_source_geometry_and_shape_fail_closed():
    with pytest.raises(ValueError, match="radii_angstrom"):
        gaussian_harmonic_source_operator(
            positions_angstrom=POSITIONS,
            radii_angstrom=(1.0, -1.0, 1.0),
            surface_lmax=2,
        )
    with pytest.raises(ValueError, match="shape"):
        gaussian_harmonic_source_operator(
            positions_angstrom=np.zeros((2, 2)),
            radii_angstrom=(1.0, 1.0),
            surface_lmax=2,
        )
    snapshot = _snapshot()
    with pytest.raises(ValueError, match="source"):
        snapshot.apply_source(np.zeros((3, 7)))
    with pytest.raises(ValueError, match="surface_cotangent"):
        snapshot.apply_adjoint(np.zeros(snapshot.surface_space.dimension - 1))
