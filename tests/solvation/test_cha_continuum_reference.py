from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[2] / "docs/implicit-solvation/benchmarks"
)
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from cha_continuum_reference import (  # pyright: ignore[reportMissingImports]
    UnsupportedR6Geometry,
    cha_continuum_reference,
    r6_inverse_born_reference,
)


def _rotation() -> np.ndarray:
    axis = np.array([1.0, -2.0, 0.7])
    axis /= np.linalg.norm(axis)
    angle = 0.61
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3) * math.cos(angle)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * cross
    )


def _two_site_geometry() -> tuple[np.ndarray, np.ndarray]:
    return np.array([[0.0, 0.0, 0.0], [2.35, 0.0, 0.0]]), np.array([1.55, 1.25])


def _water_reference():
    return cha_continuum_reference(
        [[0.011, 0.404, 0.0], [0.777, -0.223, 0.0], [-0.788, -0.181, 0.0]],
        [-0.784666666667, 0.392333333333, 0.392333333333],
        [1.88, 1.04, 1.04],
        [1.82, 0.3019, 0.3019],
        [0.093, 0.0047, 0.0047],
    )


def test_single_sphere_r6_is_reciprocal_intrinsic_radius():
    result = r6_inverse_born_reference([[2.0, -1.0, 0.5]], [1.7])

    assert result.inverse_cube_per_angstrom3 == pytest.approx([1.7**-3], rel=2e-12)
    assert result.inverse_born_per_angstrom == pytest.approx([1.7**-1], rel=2e-12)


def test_single_sphere_r6_has_closed_gauss_surface():
    result = r6_inverse_born_reference([[2.0, -1.0, 0.5]], [1.7])

    assert result.gauss_closure_vector_angstrom2 == pytest.approx(
        np.zeros(3), abs=2e-12
    )


def test_two_sphere_r6_is_symmetric_for_equal_sites():
    result = r6_inverse_born_reference([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [1.4, 1.4])

    assert result.inverse_born_per_angstrom[0] == pytest.approx(
        result.inverse_born_per_angstrom[1], rel=2e-12
    )


def test_two_sphere_r6_matches_axisymmetric_reference_values():
    positions, radii = _two_site_geometry()
    result = r6_inverse_born_reference(positions, radii)

    # Independently evaluated axisymmetric contact-cap plus torus flux values.
    assert result.inverse_born_per_angstrom == pytest.approx(
        [0.6264203253468668, 0.7588206930087461], abs=2e-12
    )


def test_near_boundary_inside_target_uses_stable_complement_flux():
    # The O center is only ~0.05 A inside the H intrinsic sphere.  Directly
    # integrating the exposed cap complement must retain the small flux rather
    # than subtracting two separately huge full-sphere/cap integrals.
    result = r6_inverse_born_reference(
        [[0.011, 0.404, 0.0], [0.777, -0.223, 0.0]], [1.88, 1.04]
    )

    assert result.inverse_born_per_angstrom == pytest.approx(
        [0.5311515703873297, 0.7132863232466183], abs=2e-12
    )


def test_two_sphere_r6_surface_satisfies_gauss_closure():
    positions, radii = _two_site_geometry()
    result = r6_inverse_born_reference(positions, radii)

    assert np.linalg.norm(result.gauss_closure_vector_angstrom2) < 2e-8


def test_r6_is_rigid_motion_invariant():
    positions, radii = _two_site_geometry()
    rotation = _rotation()
    reference = r6_inverse_born_reference(positions, radii)
    moved = r6_inverse_born_reference(positions @ rotation.T + [7.3, -5.1, 2.7], radii)

    assert moved.inverse_born_per_angstrom == pytest.approx(
        reference.inverse_born_per_angstrom, rel=2e-11, abs=2e-12
    )
    assert moved.gauss_closure_vector_angstrom2 == pytest.approx(
        reference.gauss_closure_vector_angstrom2 @ rotation.T, abs=2e-9
    )


def test_r6_is_permutation_covariant():
    positions, radii = _two_site_geometry()
    order = np.array([1, 0])
    reference = r6_inverse_born_reference(positions, radii)
    permuted = r6_inverse_born_reference(positions[order], radii[order])

    assert permuted.inverse_born_per_angstrom == pytest.approx(
        reference.inverse_born_per_angstrom[order], rel=2e-11, abs=2e-12
    )


def test_r6_retains_all_prescribed_azimuth_refinement_levels():
    positions, radii = _two_site_geometry()
    result = r6_inverse_born_reference(positions, radii)

    assert [level.azimuth_order for level in result.levels] == [64, 96, 128]
    assert all(
        level.inverse_cube_quad_error_estimate_per_angstrom3 >= 0.0
        for level in result.levels
    )
    assert all(level.meridian_evaluations > 0 for level in result.levels)


def test_overlapping_active_tubes_fail_closed():
    with pytest.raises(UnsupportedR6Geometry, match="tube separation"):
        r6_inverse_born_reference(
            [[0.0, 0.0, 0.0], [4.2, 0.0, 0.0], [2.1, 3.7, 0.0]],
            [1.5, 1.5, 1.5],
        )


def test_more_than_three_sites_fail_closed():
    with pytest.raises(UnsupportedR6Geometry, match="at most three"):
        r6_inverse_born_reference(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            [1.0, 1.0, 1.0, 1.0],
        )


def test_site_count_guard_precedes_finite_and_pairwise_work(monkeypatch):
    def forbidden_finite_check(*args, **kwargs):
        raise AssertionError("site-count guard did not run before whole-array work")

    monkeypatch.setattr(np, "isfinite", forbidden_finite_check)
    with pytest.raises(UnsupportedR6Geometry, match="at most three"):
        r6_inverse_born_reference(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            [1.0, 1.0, 1.0, 1.0],
        )


@pytest.mark.parametrize("orders", [(32, 64, 96, 128), (64, 96, 129)])
def test_excessive_public_azimuth_work_fails_closed(orders):
    positions, radii = _two_site_geometry()

    with pytest.raises(ValueError, match="at most three|at most 128"):
        r6_inverse_born_reference(positions, radii, azimuth_orders=orders)


def test_azimuth_level_guard_consumes_at_most_four_iterable_items():
    consumed = 0

    def guarded_orders():
        nonlocal consumed
        for value in (16, 32, 64, 96):
            consumed += 1
            yield value
        raise AssertionError("oracle consumed beyond the rejection sentinel")

    positions, radii = _two_site_geometry()
    with pytest.raises(ValueError, match="at most three"):
        r6_inverse_born_reference(positions, radii, azimuth_orders=guarded_orders())

    assert consumed == 4


def test_r6_azimuth_refinement_converges_for_axisymmetric_pair():
    positions, radii = _two_site_geometry()
    result = r6_inverse_born_reference(positions, radii)
    coarse, middle, fine = (level.inverse_cube_per_angstrom3 for level in result.levels)

    coarse_difference = np.max(np.abs(coarse - middle))
    fine_difference = np.max(np.abs(middle - fine))

    assert fine_difference <= max(coarse_difference, 1e-15)
    assert coarse_difference < 1e-10
    assert fine_difference < 1e-10


def test_complete_reference_components_sum_to_total():
    result = cha_continuum_reference(
        [[0.0, 0.0, 0.0]],
        [-0.7],
        [1.5],
        [1.82],
        [0.093],
    )

    assert result.total_kcal_mol == pytest.approx(
        result.polar_kcal_mol + result.cavity_kcal_mol + result.dispersion_kcal_mol,
        abs=2e-12,
    )


def test_complete_reference_single_sphere_cavity_is_analytic():
    result = cha_continuum_reference(
        [[0.0, 0.0, 0.0]],
        [-0.7],
        [1.5],
        [1.82],
        [0.093],
    )
    expected_volume = 4.0 * math.pi * (1.82 + 1.3) ** 3 / 3.0

    assert result.cavity_volume_angstrom3 == pytest.approx(expected_volume, rel=2e-10)
    assert result.cavity_kcal_mol == pytest.approx(
        0.0378 * expected_volume - 0.5692, rel=2e-10
    )


def test_contained_water_sav_matches_full_oxygen_sphere():
    result = _water_reference()
    expected_volume = 4.0 * math.pi * (1.82 + 1.3) ** 3 / 3.0

    assert result.cavity_volume_angstrom3 == pytest.approx(expected_volume, abs=2e-10)


def test_contained_water_reports_scalar_only_cavity_error_units():
    result = _water_reference()
    diagnostics = result.diagnostics

    assert diagnostics["cavity_scalar_reference_available"] is True
    assert diagnostics["cavity_scalar_error_estimate_angstrom3"] >= 0.0
    assert abs(diagnostics["cavity_generic_minus_scalar_reference_angstrom3"]) < 2e-10
    assert "cavity_quadrature_error_angstrom3" not in diagnostics
    assert diagnostics["cavity_mixed_vector_quad_error"] >= 0.0


def test_prepared_water_polar_matches_frozen_center_value():
    result = _water_reference()

    assert result.polar_kcal_mol == pytest.approx(-9.696776508790435, abs=1e-10)


def test_contained_water_dispersion_matches_radial_solid_angle_integral():
    positions = np.array(
        [[0.011, 0.404, 0.0], [0.777, -0.223, 0.0], [-0.788, -0.181, 0.0]]
    )
    rmin = np.array([1.82, 0.3019, 0.3019])
    epsilon = np.array([0.093, 0.0047, 0.0047])
    boundary_radius = 1.82 + 0.557
    sigma = (rmin + 1.7683) * 2.0 ** (-1.0 / 6.0)
    mixed_epsilon = np.sqrt(epsilon * 0.1520)
    density = 0.03333 * 1.129

    expected = 0.0
    for site in range(3):
        offset = float(np.linalg.norm(positions[site] - positions[0]))

        def radial_integrand(distance):
            if distance < sigma[site]:
                return 0.0
            potential = (
                4.0
                * mixed_epsilon[site]
                * ((sigma[site] / distance) ** 12 - (sigma[site] / distance) ** 6)
            )
            if offset == 0.0:
                solid_angle = 0.0 if distance < boundary_radius else 4.0 * math.pi
            elif distance < boundary_radius - offset:
                solid_angle = 0.0
            elif distance < boundary_radius + offset:
                solid_angle = (
                    2.0
                    * math.pi
                    * (
                        1.0
                        + (distance**2 + offset**2 - boundary_radius**2)
                        / (2.0 * distance * offset)
                    )
                )
            else:
                solid_angle = 4.0 * math.pi
            return density * potential * distance**2 * solid_angle

        lower = float(sigma[site])
        split_points = [
            value
            for value in (boundary_radius - offset, boundary_radius + offset)
            if value > lower
        ]
        for upper in split_points:
            expected += quad(
                radial_integrand, lower, upper, epsabs=1e-12, epsrel=1e-12
            )[0]
            lower = upper
        expected += quad(radial_integrand, lower, np.inf, epsabs=1e-12, epsrel=1e-12)[0]

    assert _water_reference().dispersion_kcal_mol == pytest.approx(expected, abs=2e-11)


def test_contained_water_reports_scalar_only_dispersion_error_units():
    result = _water_reference()
    diagnostics = result.diagnostics

    assert diagnostics["dispersion_scalar_reference_available"] is True
    assert diagnostics["dispersion_scalar_error_estimate_kcal_mol"] >= 0.0
    assert (
        abs(diagnostics["dispersion_generic_minus_scalar_reference_kcal_mol"]) < 2e-11
    )
    assert "dispersion_quadrature_error_kcal_mol" not in diagnostics
    assert diagnostics["dispersion_mixed_vector_quad_error"] >= 0.0


def test_noncontained_nonpolar_geometry_has_no_scalar_error_estimate():
    positions, radii = _two_site_geometry()
    result = cha_continuum_reference(
        positions,
        [-0.6, 0.6],
        radii,
        [1.0, 1.0],
        [0.1, 0.1],
    )

    assert result.diagnostics["cavity_scalar_reference_available"] is False
    assert result.diagnostics["cavity_scalar_error_estimate_angstrom3"] is None
    assert result.diagnostics["dispersion_scalar_reference_available"] is False
    assert result.diagnostics["dispersion_scalar_error_estimate_kcal_mol"] is None


def test_complete_reference_is_rigid_motion_invariant():
    positions, radii = _two_site_geometry()
    rotation = _rotation()
    kwargs = dict(
        charges_e=[-0.6, 0.6],
        intrinsic_radii_angstrom=radii,
        lj_rmin_angstrom=[1.82, 0.3019],
        lj_epsilon_kcal_mol=[0.093, 0.0047],
    )
    reference = cha_continuum_reference(positions, **kwargs)
    moved = cha_continuum_reference(positions @ rotation.T + [7.3, -5.1, 2.7], **kwargs)

    assert moved.total_kcal_mol == pytest.approx(reference.total_kcal_mol, rel=2e-10)


def test_reference_does_not_import_production_geometry_or_scalar_helpers():
    source_path = BENCHMARK_DIR / "cha_continuum_reference.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    assert not any(
        forbidden in module
        for module in imported
        for forbidden in (
            "torch_continuum_r6",
            "torch_continuum_ses",
            "torch_chagb",
        )
    )


@pytest.mark.parametrize(
    ("positions", "radii", "message"),
    [
        ([[0.0, 0.0]], [1.0], "shape"),
        ([[0.0, 0.0, 0.0]], [0.0], "positive"),
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [1.0, 1.0], "distinct"),
    ],
)
def test_invalid_r6_inputs_fail_closed(positions, radii, message):
    with pytest.raises((TypeError, ValueError), match=message):
        r6_inverse_born_reference(positions, radii)
