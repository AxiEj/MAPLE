from __future__ import annotations

import math

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    _swig_switch_with_derivative,
    fibonacci_swig_inspired_solvent_accessible_surface_area_position_vjp,
    fibonacci_swig_inspired_solvent_accessible_surface_areas,
    smd_water_cds,
    smd_water_cds_fibonacci_swig_inspired,
    smd_water_cds_fibonacci_swig_inspired_position_gradient,
)


def test_swig_switch_is_smooth_at_both_boundaries():
    epsilon = 1.0e-7
    coordinates = np.asarray(
        [-epsilon, 0.0, epsilon, 1.0 - epsilon, 1.0, 1.0 + epsilon]
    )

    values, derivatives = _swig_switch_with_derivative(coordinates)

    assert values[0] == values[1] == 0.0
    assert values[4] == values[5] == 1.0
    assert derivatives[0] == derivatives[1] == 0.0
    assert derivatives[4] == derivatives[5] == 0.0
    assert values[2] < 1.1e-19
    assert 1.0 - values[3] < 2.0e-15
    assert derivatives[2] < 3.1e-13
    assert derivatives[3] < 3.1e-13


def _coordinate_finite_difference(
    positions: np.ndarray,
    scalar,
    *,
    step_angstrom: float = 1.0e-6,
) -> np.ndarray:
    gradient = np.empty_like(positions)
    for atom_index in range(len(positions)):
        for coordinate in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom_index, coordinate] += step_angstrom
            minus[atom_index, coordinate] -= step_angstrom
            gradient[atom_index, coordinate] = (
                scalar(plus) - scalar(minus)
            ) / (2.0 * step_angstrom)
    return gradient


def test_fibonacci_swig_inspired_area_is_exact_for_separated_spheres():
    positions = np.asarray([[0.0, 0.0, 0.0], [20.0, -3.0, 1.0]])
    radii = np.asarray([1.6, 2.1])

    areas = fibonacci_swig_inspired_solvent_accessible_surface_areas(
        positions,
        radii,
        grid_points=194,
    )

    np.testing.assert_allclose(areas, 4.0 * math.pi * radii**2, atol=1.0e-13)


def test_fibonacci_swig_inspired_area_vjp_matches_finite_difference():
    positions = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [1.35, 0.10, -0.05],
            [-0.45, 1.30, 0.20],
            [0.20, -1.15, 0.35],
        ]
    )
    radii = np.asarray([2.10, 1.92, 1.60, 1.95])
    cotangent = np.asarray([0.7, -0.2, 0.4, 0.9])
    grid_points = 302

    analytic = (
        fibonacci_swig_inspired_solvent_accessible_surface_area_position_vjp(
            positions,
            radii,
            cotangent,
            grid_points=grid_points,
        )
    )
    finite_difference = _coordinate_finite_difference(
        positions,
        lambda displaced: float(
            np.dot(
                cotangent,
                fibonacci_swig_inspired_solvent_accessible_surface_areas(
                    displaced,
                    radii,
                    grid_points=grid_points,
                ),
            )
        ),
    )

    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=2.0e-7,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(analytic.sum(axis=0), 0.0, atol=2.0e-13)


def test_fibonacci_swig_inspired_cds_gradient_matches_energy_difference():
    symbols = ("C", "O", "H", "H", "H", "H")
    positions = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [1.43, 0.00, 0.00],
            [-0.63, 0.90, 0.00],
            [-0.63, -0.45, 0.78],
            [-0.63, -0.45, -0.78],
            [1.80, 0.80, 0.00],
        ]
    )
    grid_points = 590

    analytic = smd_water_cds_fibonacci_swig_inspired_position_gradient(
        symbols,
        positions,
        grid_points=grid_points,
    )
    finite_difference = _coordinate_finite_difference(
        positions,
        lambda displaced: smd_water_cds_fibonacci_swig_inspired(
            symbols,
            displaced,
            grid_points=grid_points,
        ).energy_hartree,
    )

    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=2.0e-7,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(analytic.sum(axis=0), 0.0, atol=2.0e-13)


@pytest.mark.parametrize(
    ("symbols", "positions", "reference_kcal_mol"),
    [
        (
            ("O", "H", "H"),
            [[0, 0, 0], [0.958, 0, 0], [-0.239, 0.927, 0]],
            1.442140966301685,
        ),
        (
            ("C", "H", "H", "H", "H"),
            [
                [0, 0, 0],
                [0.629, 0.629, 0.629],
                [-0.629, -0.629, 0.629],
                [-0.629, 0.629, -0.629],
                [0.629, -0.629, -0.629],
            ],
            2.756661165552430,
        ),
        (
            ("C", "O", "H", "H", "H", "H"),
            [
                [0, 0, 0],
                [1.43, 0, 0],
                [-0.63, 0.9, 0],
                [-0.63, -0.45, 0.78],
                [-0.63, -0.45, -0.78],
                [1.8, 0.8, 0],
            ],
            2.592480030180290,
        ),
    ],
)
def test_fibonacci_swig_inspired_cds_is_within_static_nwchem_tolerance(
    symbols,
    positions,
    reference_kcal_mol,
):
    result = smd_water_cds_fibonacci_swig_inspired(
        symbols,
        np.asarray(positions, dtype=float),
    )
    assert result.energy_kcal_mol == pytest.approx(
        reference_kcal_mol,
        abs=0.001,
    )


def test_fibonacci_swig_candidate_does_not_change_canonical_energy():
    symbols = ("C", "O", "H", "H", "H", "H")
    positions = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [1.43, 0.00, 0.00],
            [-0.63, 0.90, 0.00],
            [-0.63, -0.45, 0.78],
            [-0.63, -0.45, -0.78],
            [1.80, 0.80, 0.00],
        ]
    )

    assert smd_water_cds(
        symbols,
        positions,
    ).energy_kcal_mol == pytest.approx(2.5850021037890576, abs=1.0e-12)


@pytest.mark.parametrize(
    ("positions", "radii", "cotangent", "message"),
    [
        (
            np.zeros((2, 2)),
            np.ones(2),
            np.ones(2),
            "coordinates must be finite",
        ),
        (
            np.full((2, 3), np.nan),
            np.ones(2),
            np.ones(2),
            "coordinates must be finite",
        ),
        (
            np.zeros((2, 3)),
            np.asarray([1.0, 0.0]),
            np.ones(2),
            "radii must be one finite positive",
        ),
        (
            np.zeros((2, 3)),
            np.ones(2),
            np.asarray([1.0, np.nan]),
            "area cotangent must be finite",
        ),
    ],
)
def test_fibonacci_swig_inspired_area_vjp_rejects_invalid_inputs(
    positions,
    radii,
    cotangent,
    message,
):
    with pytest.raises(ValueError, match=message):
        fibonacci_swig_inspired_solvent_accessible_surface_area_position_vjp(
            positions,
            radii,
            cotangent,
            grid_points=194,
        )
