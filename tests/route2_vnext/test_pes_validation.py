from __future__ import annotations

import numpy as np
import pytest

from maple.solvation.release.pes_validation import (
    closed_rectangular_loop,
    closed_loop_work,
    displace_positions,
    reverse_closed_path,
    summarize_directional_derivatives,
    water_geometry_descriptors,
    water_vibrational_directions,
)

WATER = np.asarray(
    [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9266, 0.0]],
    dtype=float,
)


def test_water_vibrational_directions_are_internal_orthonormal_modes():
    directions = water_vibrational_directions(WATER)
    assert tuple(directions) == (
        "symmetric_stretch",
        "asymmetric_stretch",
        "bend",
    )
    matrix = np.stack([value.reshape(-1) for value in directions.values()])
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=2.0e-15)
    for direction in directions.values():
        np.testing.assert_allclose(np.sum(direction, axis=0), 0.0, atol=2.0e-15)

    base = water_geometry_descriptors(WATER)
    symmetric = water_geometry_descriptors(
        displace_positions(WATER, directions, {"symmetric_stretch": 1.0e-3})
    )
    asymmetric = water_geometry_descriptors(
        displace_positions(WATER, directions, {"asymmetric_stretch": 1.0e-3})
    )
    bent = water_geometry_descriptors(
        displace_positions(WATER, directions, {"bend": 1.0e-3})
    )
    assert symmetric["oh1_A"] > base["oh1_A"]
    assert symmetric["oh2_A"] > base["oh2_A"]
    assert (asymmetric["oh1_A"] - base["oh1_A"]) * (
        asymmetric["oh2_A"] - base["oh2_A"]
    ) < 0.0
    assert abs(bent["hoh_angle_deg"] - base["hoh_angle_deg"]) > 1.0e-2


def test_rectangular_loop_is_closed_reversible_and_has_fixed_edge_count():
    path = closed_rectangular_loop(subdivisions_per_edge=4)
    assert len(path) == 17
    assert path[0] == path[-1] == (-1.0, -1.0)
    assert path[4] == (1.0, -1.0)
    assert path[8] == (1.0, 1.0)
    assert path[12] == (-1.0, 1.0)
    reverse = reverse_closed_path(path)
    assert reverse[0] == reverse[-1] == path[0]
    assert reverse == tuple(reversed(path))
    with pytest.raises(ValueError, match="positive even"):
        closed_rectangular_loop(subdivisions_per_edge=3)


def test_simpson_loop_work_is_zero_for_quadratic_pes_and_detects_curl():
    coefficients = closed_rectangular_loop(subdivisions_per_edge=4)
    positions = tuple(
        np.asarray([[0.3 * first, 0.2 * second, 0.0]]) for first, second in coefficients
    )
    stiffness = np.diag([2.0, 3.0, 1.0])
    conservative_forces = tuple(-(position @ stiffness.T) for position in positions)
    conservative = closed_loop_work(
        positions,
        conservative_forces,
        subdivisions_per_edge=4,
    )
    assert conservative["simpson_work_eV"] == pytest.approx(0.0, abs=1.0e-16)
    assert conservative["gate_passed"] is True

    curl_forces = tuple(
        np.asarray([[-position[0, 1], position[0, 0], 0.0]]) for position in positions
    )
    nonconservative = closed_loop_work(
        positions,
        curl_forces,
        subdivisions_per_edge=4,
    )
    # Curl is 2 and the rectangle area is 0.24, hence Green's theorem gives 0.48.
    assert abs(nonconservative["simpson_work_eV"]) == pytest.approx(0.48)
    assert nonconservative["gate_passed"] is False


def test_directional_summary_applies_absolute_and_away_from_zero_relative_gates():
    # E(x)=2*x+3*x^2 at x=0 has exact central derivative 2 for every step.
    samples = tuple(
        (step, 2.0 * step + 3.0 * step**2, -2.0 * step + 3.0 * step**2)
        for step in (4.0e-4, 2.0e-4, 1.0e-4)
    )
    exact = summarize_directional_derivatives(2.0, samples)
    assert exact["all_gates_passed"] is True
    assert max(item["absolute_error_eV_per_A"] for item in exact["records"]) < 1e-12

    wrong = summarize_directional_derivatives(2.01, samples)
    assert wrong["all_gates_passed"] is False
    assert all(item["relative_gate_applies"] for item in wrong["records"])

    near_zero = summarize_directional_derivatives(
        1.0e-5,
        ((1.0e-3, 1.0e-8, -1.0e-8),),
    )
    assert near_zero["records"][0]["relative_gate_applies"] is False
    assert near_zero["all_gates_passed"] is True


def test_validation_helpers_reject_nonfinite_or_open_inputs():
    directions = water_vibrational_directions(WATER)
    with pytest.raises(ValueError, match="unknown collective direction"):
        displace_positions(WATER, directions, {"torsion": 0.1})
    with pytest.raises(ValueError, match="finite"):
        water_vibrational_directions(WATER * np.nan)
    path = closed_rectangular_loop(subdivisions_per_edge=4)
    positions = [np.asarray([[a, b, 0.0]]) for a, b in path]
    positions[-1] = positions[-1] + 1.0e-2
    forces = [np.zeros((1, 3)) for _ in positions]
    with pytest.raises(ValueError, match="geometrically closed"):
        closed_loop_work(positions, forces, subdivisions_per_edge=4)
