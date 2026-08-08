from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
    load_atomic_reference_density_asset,
)
from maple.function.calculator.extra_correction.implicit.route2_density_levelset import (
    DensityBoundaryDiagnostics,
    LSFAdjointWeights,
    ReconstructedMacePolarDensityLevelSet,
)

ROOT = Path(__file__).resolve().parents[2]
TABLE = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-rhodrop-atomic-reference-gaussian-mixture-v1.npz"
)
MANIFEST = TABLE.with_suffix(".json")
POSITIONS_ANGSTROM = np.asarray([[-0.45, 0.10, 0.20], [0.62, -0.30, 0.15]])
SOURCE = np.asarray([[0.16, 0.05, -0.03, 0.02], [-0.07, -0.04, 0.06, -0.01]])


def _asset():
    return load_atomic_reference_density_asset(
        table_path=TABLE,
        manifest_path=MANIFEST,
    )


def _level_set(
    *,
    positions=None,
    source=None,
    minimum_gradient_norm_bohr=1.0e-10,
    expected_total_charge_e=0.09,
):
    if positions is None:
        positions = POSITIONS_ANGSTROM
    if source is None:
        source = SOURCE
    return ReconstructedMacePolarDensityLevelSet(
        _asset(),
        atomic_numbers=np.asarray([1, 8]),
        atom_positions_angstrom=np.asarray(positions),
        density_coefficients=np.asarray(source),
        n_iso_e_per_bohr3=0.02,
        expected_total_charge_e=expected_total_charge_e,
        sigma_angstrom=1.5,
        minimum_density_e_per_bohr3=0.0,
        minimum_gradient_norm_bohr=minimum_gradient_norm_bohr,
        surface_tolerance=1.0e-9,
    )


def _assert_jet_allclose(actual, expected, *, rtol, atol):
    for name in ("value", "gradient", "hessian", "third"):
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=rtol,
            atol=atol,
        )


def _contract(weights, jet):
    return float(
        np.vdot(weights.value, jet.value)
        + np.vdot(weights.gradient, jet.gradient)
        + np.vdot(weights.hessian, jet.hessian)
    )


def test_reconstructed_density_electron_count_is_nuclear_charge_minus_net_monopole():
    level_set = _level_set()
    changed_dipoles = SOURCE.copy()
    changed_dipoles[:, 1:] += np.asarray([[1.2, -0.7, 0.3], [-0.4, 0.9, -1.1]])
    same_monopoles = _level_set(source=changed_dipoles)

    assert level_set.atom_count == 2
    assert level_set.expected_electron_count == pytest.approx(8.91, abs=1.0e-14)
    assert level_set.integrated_electron_count == pytest.approx(8.91, abs=1.0e-14)
    assert same_monopoles.integrated_electron_count == pytest.approx(8.91, abs=1.0e-14)


@pytest.mark.parametrize(
    ("raw_component", "cartesian_axis"),
    [(1, 1), (2, 2), (3, 0)],
)
def test_residual_density_uses_raw_mace_l1_order_and_angstrom_to_bohr_units(
    raw_component,
    cartesian_axis,
):
    position = np.zeros((1, 3))
    point = np.asarray([[0.41, -0.27, 0.19]])
    zero = np.zeros((1, 4))
    source = zero.copy()
    source[0, raw_component] = 1.0
    base = ReconstructedMacePolarDensityLevelSet(
        _asset(),
        np.asarray([1]),
        position,
        zero,
        0.02,
        expected_total_charge_e=0.0,
        sigma_angstrom=1.5,
    )
    displaced = ReconstructedMacePolarDensityLevelSet(
        _asset(),
        np.asarray([1]),
        position,
        source,
        0.02,
        expected_total_charge_e=0.0,
        sigma_angstrom=1.5,
    )

    density_change = (
        displaced.evaluate_reconstructed_density(point).value[0]
        - base.evaluate_reconstructed_density(point).value[0]
    )
    sigma_bohr = 1.5 / Bohr
    dipole_bohr = 1.0 / Bohr
    gaussian = (2.0 * np.pi * sigma_bohr**2) ** -1.5 * np.exp(
        -np.dot(point[0], point[0]) / (2.0 * sigma_bohr**2)
    )
    expected_residual = (
        gaussian * dipole_bohr * point[0, cartesian_axis] / sigma_bohr**2
    )
    assert density_change == pytest.approx(-expected_residual, rel=2.0e-14, abs=2.0e-16)


def test_level_set_is_one_minus_reconstructed_density_over_isovalue():
    level_set = _level_set()
    points = np.asarray([[0.1, -0.2, 0.4], [1.3, 0.2, -0.5]])
    density = level_set.evaluate_reconstructed_density(points)
    level = level_set.evaluate_spatial(points)

    np.testing.assert_allclose(level.value, 1.0 - density.value / 0.02)
    for name in ("gradient", "hessian", "third"):
        np.testing.assert_allclose(
            getattr(level, name),
            -getattr(density, name) / 0.02,
            rtol=0.0,
            atol=2.0e-14,
        )


def test_level_set_source_jvp_matches_central_difference_for_the_full_spatial_jet():
    level_set = _level_set()
    points = np.asarray([[0.1, -0.2, 0.4], [1.3, 0.2, -0.5], [-0.8, 0.7, 0.3]])
    direction = np.asarray([[0.13, -0.07, 0.04, 0.09], [-0.13, 0.08, -0.11, 0.03]])
    analytic = level_set.source_jvp(points, direction)
    # The reconstructed level set is exactly linear in the source.  A larger
    # symmetric step therefore has no truncation error and avoids cancellation
    # when the small directional response is subtracted from the O(1) level set.
    step = 1.0e-3
    plus = _level_set(source=SOURCE + step * direction).evaluate_spatial(points)
    minus = _level_set(source=SOURCE - step * direction).evaluate_spatial(points)

    for name, rtol, atol in (
        ("value", 3.0e-9, 3.0e-10),
        ("gradient", 3.0e-8, 3.0e-9),
        ("hessian", 3.0e-7, 3.0e-8),
        ("third", 3.0e-6, 3.0e-7),
    ):
        finite_difference = (getattr(plus, name) - getattr(minus, name)) / (2.0 * step)
        np.testing.assert_allclose(
            getattr(analytic, name), finite_difference, rtol=rtol, atol=atol
        )


def test_level_set_source_vjp_satisfies_the_euclidean_dot_product_identity():
    rng = np.random.default_rng(20260807)
    level_set = _level_set()
    points = rng.normal(size=(7, 3))
    direction = rng.normal(size=(2, 4))
    raw_hessian = rng.normal(size=(7, 3, 3))
    weights = LSFAdjointWeights(
        value=rng.normal(size=7),
        gradient=rng.normal(size=(7, 3)),
        hessian=0.5 * (raw_hessian + raw_hessian.swapaxes(-1, -2)),
    )

    jvp = level_set.source_jvp(points, direction)
    vjp = level_set.source_vjp(points, weights)

    assert _contract(weights, jvp) == pytest.approx(
        float(np.vdot(vjp, direction)), rel=3.0e-13, abs=3.0e-13
    )


def test_level_set_nuclear_vjp_matches_central_difference_per_angstrom():
    rng = np.random.default_rng(20260808)
    level_set = _level_set()
    points = rng.normal(size=(6, 3))
    raw_hessian = rng.normal(size=(6, 3, 3))
    weights = LSFAdjointWeights(
        value=rng.normal(size=6),
        gradient=rng.normal(size=(6, 3)),
        hessian=0.5 * (raw_hessian + raw_hessian.swapaxes(-1, -2)),
    )
    analytic = level_set.nuclear_vjp(points, weights)
    positions = POSITIONS_ANGSTROM
    finite_difference = np.empty_like(positions)
    step_angstrom = 2.0e-6

    for atom_index in range(positions.shape[0]):
        for coordinate in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom_index, coordinate] += step_angstrom
            minus[atom_index, coordinate] -= step_angstrom
            plus_value = _contract(
                weights,
                _level_set(positions=plus, source=SOURCE).evaluate_spatial(points),
            )
            minus_value = _contract(
                weights,
                _level_set(positions=minus, source=SOURCE).evaluate_spatial(points),
            )
            finite_difference[atom_index, coordinate] = (plus_value - minus_value) / (
                2.0 * step_angstrom
            )

    np.testing.assert_allclose(analytic, finite_difference, rtol=3.0e-6, atol=3.0e-7)


def test_reconstructed_level_set_is_covariant_under_rigid_translation():
    level_set = _level_set()
    points = np.asarray([[0.1, -0.2, 0.4], [1.3, 0.2, -0.5]])
    shift_angstrom = np.asarray([0.23, -0.11, 0.37])
    translated = _level_set(
        positions=POSITIONS_ANGSTROM + shift_angstrom,
        source=SOURCE,
    )

    original_jet = level_set.evaluate_spatial(points)
    translated_jet = translated.evaluate_spatial(points + shift_angstrom / Bohr)

    _assert_jet_allclose(translated_jet, original_jet, rtol=0.0, atol=4.0e-13)


def test_boundary_diagnostics_fail_for_negative_reconstructed_density():
    negative = ReconstructedMacePolarDensityLevelSet(
        _asset(),
        np.asarray([1]),
        np.zeros((1, 3)),
        np.asarray([[20.0, 0.0, 0.0, 0.0]]),
        0.02,
        expected_total_charge_e=20.0,
        sigma_angstrom=0.5,
    )

    diagnostics = negative.diagnose(np.zeros((1, 3)), require_surface=False)

    assert isinstance(diagnostics, DensityBoundaryDiagnostics)
    assert diagnostics.passed is False
    assert any("negative" in reason.lower() for reason in diagnostics.reasons)


def test_boundary_validation_fails_closed_for_an_off_surface_point():
    level_set = _level_set()
    off_surface = np.zeros((1, 3))

    diagnostics = level_set.diagnose(off_surface, require_surface=True)

    assert diagnostics.passed is False
    with pytest.raises(ValueError):
        level_set.require_valid_boundary(off_surface)


def test_boundary_diagnostics_fail_for_a_vanishing_level_set_gradient():
    spherical = ReconstructedMacePolarDensityLevelSet(
        _asset(),
        np.asarray([1]),
        np.zeros((1, 3)),
        np.zeros((1, 4)),
        0.02,
        expected_total_charge_e=0.0,
        minimum_gradient_norm_bohr=1.0e-10,
    )

    diagnostics = spherical.diagnose(np.zeros((1, 3)), require_surface=False)

    assert diagnostics.passed is False
    assert any("gradient" in reason.lower() for reason in diagnostics.reasons)


def test_level_set_requires_an_independently_declared_total_charge():
    with pytest.raises(TypeError, match="expected_total_charge_e"):
        ReconstructedMacePolarDensityLevelSet(
            _asset(),
            np.asarray([1, 8]),
            POSITIONS_ANGSTROM,
            SOURCE,
            0.02,
        )


def test_level_set_rejects_source_charge_drift_from_declared_total_charge():
    drifted = SOURCE.copy()
    drifted[0, 0] += 2.0e-4

    with pytest.raises(ValueError, match="independently declared molecular charge"):
        _level_set(source=drifted)
