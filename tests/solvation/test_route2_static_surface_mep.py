from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_static_surface_mep import (
    build_route2_smd_exterior_probe_surface,
    route2_weighted_surface_mep_discrepancy,
)


def test_single_atom_probe_has_all_twenty_six_geometry_only_candidates():
    surface = build_route2_smd_exterior_probe_surface(
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([1.5]),
        clearance_angstrom=1.0,
    )

    assert surface.candidate_count == 26
    assert surface.retained_point_count == 26
    assert np.all(surface.parent_atom_indices == 0)
    # 1.5 + 1.0 Å expressed in bohr.
    assert np.allclose(np.linalg.norm(surface.surface_points_bohr, axis=1), 2.5 / 0.5291772105638411)


def test_probe_filter_uses_only_the_declared_expanded_sphere_geometry():
    positions = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    radii = np.asarray([1.0, 1.0])
    first = build_route2_smd_exterior_probe_surface(
        positions, radii, clearance_angstrom=0.25
    )
    second = build_route2_smd_exterior_probe_surface(
        positions, radii, clearance_angstrom=0.25
    )

    assert first.candidate_count == 52
    assert first.retained_point_count == second.retained_point_count
    np.testing.assert_array_equal(first.surface_points_bohr, second.surface_points_bohr)
    points_angstrom = first.surface_points_bohr * 0.5291772105638411
    distances = np.linalg.norm(points_angstrom[:, None, :] - positions[None, :, :], axis=2)
    assert np.all(distances + 2.0e-12 >= (radii + 0.25)[None, :])


def test_weighted_surface_metrics_are_exact_for_matched_and_constant_errors():
    weights = np.asarray([1.0, 2.0, 3.0])
    reference = np.asarray([2.0, -1.0, 3.0])
    matched = route2_weighted_surface_mep_discrepancy(reference, reference, weights)
    assert matched == {
        "weighted_relative_l2": 0.0,
        "weighted_rmse_hartree_per_e": 0.0,
        "maximum_absolute_error_hartree_per_e": 0.0,
        "relative_infinity_norm": 0.0,
    }

    offset = route2_weighted_surface_mep_discrepancy(reference + 2.0, reference, weights)
    assert offset["weighted_rmse_hartree_per_e"] == pytest.approx(2.0)
    assert offset["maximum_absolute_error_hartree_per_e"] == pytest.approx(2.0)
    assert offset["relative_infinity_norm"] == pytest.approx(2.0 / 3.0)


def test_surface_metrics_reject_nonpositive_or_mismatched_inputs():
    with pytest.raises(ValueError, match="positive"):
        route2_weighted_surface_mep_discrepancy(
            np.ones(2), np.ones(2), np.asarray([1.0, 0.0])
        )
    with pytest.raises(ValueError, match="matched"):
        route2_weighted_surface_mep_discrepancy(
            np.ones(2), np.ones(3), np.ones(2)
        )
