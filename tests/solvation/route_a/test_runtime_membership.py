from __future__ import annotations

import numpy as np
import pytest
from ase.units import kB

from maple.function.dispatcher.solvfe.membership import (
    SoftCutoffMembership,
    nearest_surface_geometry,
    soft_occupancy_weights,
)


def _definition(**overrides) -> SoftCutoffMembership:
    values = {
        "vdw_radii_angstrom": {1: 1.20, 6: 1.70, 8: 1.52},
        "lambda_s_angstrom": 1.50,
        "softness_angstrom": 0.10,
        "shell_boundary_id": "nearest-solute-vdw-surface-v1",
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

    assert first.content_hash == reordered.content_hash
    assert first.content_hash != changed_softness.content_hash


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


def test_nearest_surface_geometry_selects_minimum_signed_distance():
    geometry = nearest_surface_geometry(
        point_positions_angstrom=np.asarray([[2.0, 0.0, 0.0]]),
        center_positions_angstrom=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        ),
        center_radii_angstrom=np.asarray([1.0, 1.8]),
    )

    assert geometry.nearest_center_indices.tolist() == [1]
    assert geometry.signed_distances_angstrom == pytest.approx([0.2])
    np.testing.assert_allclose(
        geometry.unit_vectors_center_to_point,
        [[-1.0, 0.0, 0.0]],
    )


def test_nearest_surface_geometry_uses_minimum_image_when_periodic():
    geometry = nearest_surface_geometry(
        point_positions_angstrom=np.asarray([[9.8, 0.0, 0.0]]),
        center_positions_angstrom=np.asarray([[0.2, 0.0, 0.0]]),
        center_radii_angstrom=np.asarray([1.5]),
        cell_angstrom=np.eye(3) * 10.0,
        pbc=np.ones(3, dtype=bool),
    )

    assert geometry.signed_distances_angstrom == pytest.approx([-1.1])
    np.testing.assert_allclose(
        geometry.unit_vectors_center_to_point,
        [[-1.0, 0.0, 0.0]],
    )
