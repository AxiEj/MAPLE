from __future__ import annotations

import numpy as np
import pytest
from ase.units import kB

from maple.function.dispatcher.solvfe.membership import (
    SoftCutoffMembership,
    smooth_surface_geometry,
    soft_occupancy_weights,
)


def _definition(**overrides) -> SoftCutoffMembership:
    values = {
        "vdw_radii_angstrom": {1: 1.20, 6: 1.70, 8: 1.52},
        "lambda_s_angstrom": 1.50,
        "softness_angstrom": 0.10,
        "surface_smoothing_angstrom": 0.05,
        "shell_boundary_id": "smooth-union-solute-vdw-surface-v1",
    }
    values.update(overrides)
    return SoftCutoffMembership(**values)


def test_member_and_empty_fields_are_exact_complements():
    definition = _definition()
    signed_distances = np.asarray([-2.0, 1.5, 5.0])
    temperature = 298.15
    beta = 1.0 / (kB * temperature)

    evaluation = definition.evaluate_signed_distances(
        signed_distances,
        temperature_k=temperature,
    )

    assert evaluation.membership + evaluation.nonmembership == pytest.approx(
        np.ones(3),
        abs=1.0e-15,
    )
    assert np.exp(-beta * evaluation.member_potential_ev) == pytest.approx(
        evaluation.membership,
        rel=1.0e-13,
        abs=1.0e-15,
    )
    assert np.exp(-beta * evaluation.empty_potential_ev) == pytest.approx(
        evaluation.nonmembership,
        rel=1.0e-13,
        abs=1.0e-15,
    )


@pytest.mark.parametrize(
    ("signed_distance", "potential_name", "derivative_name"),
    [
        (1.65, "member_potential_ev", "member_derivative_ev_per_angstrom"),
        (1.35, "empty_potential_ev", "empty_derivative_ev_per_angstrom"),
    ],
)
def test_soft_field_derivatives_match_finite_difference(
    signed_distance,
    potential_name,
    derivative_name,
):
    definition = _definition()
    temperature = 298.15
    step = 1.0e-6

    plus = definition.evaluate_signed_distances(
        np.asarray([signed_distance + step]),
        temperature_k=temperature,
    )
    minus = definition.evaluate_signed_distances(
        np.asarray([signed_distance - step]),
        temperature_k=temperature,
    )
    center = definition.evaluate_signed_distances(
        np.asarray([signed_distance]),
        temperature_k=temperature,
    )
    finite_difference = (
        getattr(plus, potential_name)[0] - getattr(minus, potential_name)[0]
    ) / (2.0 * step)

    assert getattr(center, derivative_name)[0] == pytest.approx(
        finite_difference,
        abs=1.0e-10,
    )


def test_soft_occupancy_weights_match_two_water_expansion():
    b1 = 0.25
    b2 = 0.80

    weights = soft_occupancy_weights(np.asarray([b1, b2]))

    assert weights == pytest.approx(
        np.asarray(
            [
                (1.0 - b1) * (1.0 - b2),
                b1 * (1.0 - b2) + (1.0 - b1) * b2,
                b1 * b2,
            ]
        )
    )
    assert np.sum(weights) == pytest.approx(1.0, abs=1.0e-15)


def test_soft_occupancy_supports_batches_and_preserves_normalization():
    memberships = np.asarray(
        [
            [0.0, 0.5, 1.0],
            [0.1, 0.2, 0.3],
        ]
    )

    weights = soft_occupancy_weights(memberships)

    assert weights.shape == (2, 4)
    assert np.all(weights >= 0.0)
    assert np.sum(weights, axis=-1) == pytest.approx(np.ones(2), abs=1.0e-15)
    assert weights[0] == pytest.approx([0.0, 0.5, 0.5, 0.0])


def test_active_support_may_cover_every_possible_occupancy():
    weights = soft_occupancy_weights(
        np.asarray([0.25]),
        active_occupancy_max=1,
    )

    assert weights == pytest.approx([0.75, 0.25, 0.0])


@pytest.mark.parametrize(
    "memberships",
    [
        np.asarray([np.nan]),
        np.asarray([-0.01]),
        np.asarray([1.01]),
        np.asarray([]),
    ],
)
def test_soft_occupancy_rejects_invalid_memberships(memberships):
    with pytest.raises(ValueError, match="memberships"):
        soft_occupancy_weights(memberships)


def test_membership_hash_is_order_independent_and_parameter_sensitive():
    first = _definition(vdw_radii_angstrom={8: 1.52, 1: 1.20, 6: 1.70})
    reordered = _definition(vdw_radii_angstrom={1: 1.20, 6: 1.70, 8: 1.52})
    changed_softness = _definition(softness_angstrom=0.20)
    changed_surface = _definition(surface_smoothing_angstrom=0.10)

    assert first.content_hash == reordered.content_hash
    assert first.content_hash != changed_softness.content_hash
    assert first.content_hash != changed_surface.content_hash


def test_hard_limit_is_approached_without_overflow():
    definition = _definition(softness_angstrom=1.0e-3)

    evaluation = definition.evaluate_signed_distances(
        np.asarray([-10.0, 10.0]),
        temperature_k=298.15,
    )

    assert np.all(np.isfinite(evaluation.member_potential_ev))
    assert np.all(np.isfinite(evaluation.empty_potential_ev))
    assert evaluation.membership == pytest.approx([1.0, 0.0], abs=1.0e-15)
    assert evaluation.nonmembership == pytest.approx([0.0, 1.0], abs=1.0e-15)


def test_smooth_surface_geometry_is_exact_for_one_center():
    geometry = smooth_surface_geometry(
        point_positions_angstrom=np.asarray([[2.0, 0.0, 0.0]]),
        center_positions_angstrom=np.asarray([[4.0, 0.0, 0.0]]),
        center_radii_angstrom=np.asarray([1.8]),
        surface_smoothing_angstrom=0.05,
    )

    assert geometry.signed_distances_angstrom == pytest.approx([0.2])
    np.testing.assert_allclose(geometry.center_weights, [[1.0]])
    np.testing.assert_allclose(
        geometry.center_gradient_vectors,
        [[[-1.0, 0.0, 0.0]]],
    )
    assert geometry.periodic_image_shells == 0


def test_smooth_surface_geometry_blends_two_centers_at_seam():
    smoothing = 0.20
    geometry = smooth_surface_geometry(
        point_positions_angstrom=np.asarray([[2.0, 0.0, 0.0]]),
        center_positions_angstrom=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        ),
        center_radii_angstrom=np.asarray([1.0, 1.8]),
        surface_smoothing_angstrom=smoothing,
    )

    expected = -smoothing * np.log(
        np.exp(-1.0 / smoothing) + np.exp(-0.2 / smoothing)
    )
    assert geometry.signed_distances_angstrom == pytest.approx([expected])
    np.testing.assert_allclose(
        geometry.center_weights,
        [[1.0 / (1.0 + np.exp(4.0)), 1.0 / (1.0 + np.exp(-4.0))]],
    )
    np.testing.assert_allclose(
        geometry.center_gradient_vectors,
        [
            [
                [geometry.center_weights[0, 0], 0.0, 0.0],
                [-geometry.center_weights[0, 1], 0.0, 0.0],
            ]
        ],
    )


def test_smooth_surface_geometry_is_order_invariant_and_continuous_at_seam():
    centers = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    radii = np.asarray([1.0, 1.0])
    smoothing = 0.10

    seam = smooth_surface_geometry(
        point_positions_angstrom=np.asarray([[0.0, 0.0, 0.0]]),
        center_positions_angstrom=centers,
        center_radii_angstrom=radii,
        surface_smoothing_angstrom=smoothing,
    )
    left = smooth_surface_geometry(
        point_positions_angstrom=np.asarray([[-1.0e-7, 0.0, 0.0]]),
        center_positions_angstrom=centers,
        center_radii_angstrom=radii,
        surface_smoothing_angstrom=smoothing,
    )
    right = smooth_surface_geometry(
        point_positions_angstrom=np.asarray([[1.0e-7, 0.0, 0.0]]),
        center_positions_angstrom=centers,
        center_radii_angstrom=radii,
        surface_smoothing_angstrom=smoothing,
    )
    reordered = smooth_surface_geometry(
        point_positions_angstrom=np.asarray([[0.0, 0.0, 0.0]]),
        center_positions_angstrom=centers[::-1],
        center_radii_angstrom=radii[::-1],
        surface_smoothing_angstrom=smoothing,
    )

    assert seam.signed_distances_angstrom == pytest.approx(
        [-smoothing * np.log(2.0)]
    )
    np.testing.assert_allclose(seam.center_weights, [[0.5, 0.5]])
    seam_gradient = np.sum(seam.center_gradient_vectors, axis=1)
    left_gradient = np.sum(left.center_gradient_vectors, axis=1)
    right_gradient = np.sum(right.center_gradient_vectors, axis=1)
    np.testing.assert_allclose(seam_gradient, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        left_gradient,
        right_gradient,
        atol=4.0e-6,
    )
    np.testing.assert_allclose(
        reordered.signed_distances_angstrom,
        seam.signed_distances_angstrom,
    )
    np.testing.assert_allclose(
        reordered.center_weights[:, ::-1],
        seam.center_weights,
    )


def test_smooth_surface_geometry_uses_periodic_images():
    geometry = smooth_surface_geometry(
        point_positions_angstrom=np.asarray([[9.8, 0.0, 0.0]]),
        center_positions_angstrom=np.asarray([[0.2, 0.0, 0.0]]),
        center_radii_angstrom=np.asarray([1.5]),
        surface_smoothing_angstrom=0.05,
        cell_angstrom=np.eye(3) * 10.0,
        pbc=np.ones(3, dtype=bool),
    )

    assert geometry.signed_distances_angstrom == pytest.approx([-1.1])
    np.testing.assert_allclose(
        geometry.center_gradient_vectors,
        [[[-1.0, 0.0, 0.0]]],
        atol=1.0e-14,
    )
    assert geometry.periodic_image_shells >= 1


def test_periodic_image_surface_is_smooth_at_minimum_image_cut_locus():
    smoothing = 0.05
    cell = np.eye(3) * 5.0
    center = np.asarray([[0.0, 0.0, 0.0]])
    radii = np.asarray([1.5])
    step = 1.0e-6

    seam = smooth_surface_geometry(
        point_positions_angstrom=np.asarray([[2.5, 0.0, 0.0]]),
        center_positions_angstrom=center,
        center_radii_angstrom=radii,
        surface_smoothing_angstrom=smoothing,
        cell_angstrom=cell,
        pbc=np.ones(3, dtype=bool),
    )
    left = smooth_surface_geometry(
        point_positions_angstrom=np.asarray([[2.5 - step, 0.0, 0.0]]),
        center_positions_angstrom=center,
        center_radii_angstrom=radii,
        surface_smoothing_angstrom=smoothing,
        cell_angstrom=cell,
        pbc=np.ones(3, dtype=bool),
    )
    right = smooth_surface_geometry(
        point_positions_angstrom=np.asarray([[2.5 + step, 0.0, 0.0]]),
        center_positions_angstrom=center,
        center_radii_angstrom=radii,
        surface_smoothing_angstrom=smoothing,
        cell_angstrom=cell,
        pbc=np.ones(3, dtype=bool),
    )

    seam_gradient = np.sum(seam.center_gradient_vectors, axis=1)[0, 0]
    left_gradient = np.sum(left.center_gradient_vectors, axis=1)[0, 0]
    right_gradient = np.sum(right.center_gradient_vectors, axis=1)[0, 0]
    finite_difference = (
        right.signed_distances_angstrom[0]
        - left.signed_distances_angstrom[0]
    ) / (2.0 * step)

    assert seam.periodic_image_shells == 1
    assert seam.signed_distances_angstrom[0] == pytest.approx(
        1.0 - smoothing * np.log(2.0),
        abs=1.0e-14,
    )
    assert seam_gradient == pytest.approx(0.0, abs=1.0e-14)
    assert finite_difference == pytest.approx(0.0, abs=1.0e-10)
    assert left_gradient == pytest.approx(-right_gradient, abs=1.0e-12)
