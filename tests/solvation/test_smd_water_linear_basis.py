from __future__ import annotations

import math

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2,
    SMD_WATER_TENSION_PARAMETER_NAMES,
    aqueous_atomic_surface_tension_basis,
    aqueous_atomic_surface_tension_position_vjp_from_coefficients,
    aqueous_atomic_surface_tensions,
    aqueous_atomic_surface_tensions_from_coefficients,
    aqueous_cds_tension_design_row,
)


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
            gradient[atom_index, coordinate] = (scalar(plus) - scalar(minus)) / (
                2.0 * step_angstrom
            )
    return gradient


def _rotation() -> np.ndarray:
    axis = np.asarray([0.31, -0.47, 0.83], dtype=float)
    axis /= np.linalg.norm(axis)
    angle = 0.713
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        math.cos(angle) * np.eye(3)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * cross
    )


def _geometry() -> tuple[tuple[str, ...], np.ndarray]:
    symbols = ("N", "C", "O", "H", "C", "O", "P", "F")
    positions = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [1.35, 0.00, 0.00],
            [2.55, 0.00, 0.00],
            [1.35, -1.10, 0.10],
            [1.35, 1.50, 0.20],
            [0.00, 1.35, -0.10],
            [0.00, 3.20, 0.00],
            [2.45, 1.45, -0.20],
        ],
        dtype=float,
    )
    return symbols, positions


def test_water_tension_parameter_contract_is_stable_and_immutable():
    assert SMD_WATER_TENSION_PARAMETER_NAMES == (
        "base:H",
        "base:C",
        "base:N",
        "base:O",
        "base:F",
        "base:P",
        "base:S",
        "base:Cl",
        "base:Br",
        "base:I",
        "environment:H-C",
        "environment:H-O",
        "environment:C-C",
        "environment:N-C-coordination-power-1.3",
        "environment:N-C3-short-range",
        "environment:O-C",
        "environment:O-N",
        "environment:O-P",
    )
    coefficients = SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2
    assert coefficients.shape == (len(SMD_WATER_TENSION_PARAMETER_NAMES),)
    assert coefficients.flags.writeable is False
    assert coefficients.tolist() == pytest.approx(
        [
            48.69,
            129.74,
            0.0,
            0.0,
            38.18,
            0.0,
            -9.10,
            9.82,
            -8.72,
            0.0,
            -60.77,
            0.0,
            -72.95,
            -48.22,
            84.10,
            68.69,
            121.98,
            68.85,
        ],
        rel=0.0,
        abs=0.0,
    )
    with pytest.raises(ValueError):
        coefficients[0] = 0.0


def test_linear_basis_exactly_reconstructs_stock_atomic_tensions():
    symbols, positions = _geometry()
    basis = aqueous_atomic_surface_tension_basis(symbols, positions)
    reconstructed = basis @ SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2

    np.testing.assert_allclose(
        reconstructed,
        aqueous_atomic_surface_tensions(symbols, positions),
        rtol=0.0,
        atol=2.0e-14,
    )
    assert basis.shape == (len(symbols), len(SMD_WATER_TENSION_PARAMETER_NAMES))
    # The published aqueous H-O coefficient is zero, but the declared branch
    # remains available for a preregistered linear refit.
    assert np.any(basis[:, SMD_WATER_TENSION_PARAMETER_NAMES.index("environment:H-O")])


def test_design_row_contracts_atom_areas_with_the_same_linear_basis():
    symbols, positions = _geometry()
    atom_areas = np.asarray([11.0, 7.0, 4.0, 6.0, 8.0, 3.0, 9.0, 5.0])
    coefficients = np.linspace(-17.0, 23.0, len(SMD_WATER_TENSION_PARAMETER_NAMES))
    design = aqueous_cds_tension_design_row(symbols, positions, atom_areas)
    tensions = aqueous_atomic_surface_tensions_from_coefficients(
        symbols,
        positions,
        coefficients,
    )

    assert float(np.dot(design, coefficients)) == pytest.approx(
        float(np.dot(atom_areas, tensions) / 1000.0),
        rel=2.0e-15,
        abs=2.0e-15,
    )


def test_arbitrary_coefficient_tension_vjp_matches_finite_difference():
    symbols, positions = _geometry()
    coefficients = np.linspace(-31.0, 29.0, len(SMD_WATER_TENSION_PARAMETER_NAMES))
    cotangent = np.asarray([0.7, -0.3, 0.4, 0.2, -0.5, 0.9, -0.1, 0.6])

    analytic = aqueous_atomic_surface_tension_position_vjp_from_coefficients(
        symbols,
        positions,
        cotangent,
        coefficients,
    )
    finite_difference = _coordinate_finite_difference(
        positions,
        lambda displaced: float(
            np.dot(
                cotangent,
                aqueous_atomic_surface_tensions_from_coefficients(
                    symbols,
                    displaced,
                    coefficients,
                ),
            )
        ),
    )

    np.testing.assert_allclose(analytic, finite_difference, rtol=2.0e-8, atol=2.0e-6)
    np.testing.assert_allclose(analytic.sum(axis=0), np.zeros(3), atol=5.0e-13)


def test_design_row_is_translation_rotation_and_permutation_invariant():
    symbols, positions = _geometry()
    atom_areas = np.asarray([11.0, 7.0, 4.0, 6.0, 8.0, 3.0, 9.0, 5.0])
    reference = aqueous_cds_tension_design_row(symbols, positions, atom_areas)

    transformed = (positions @ _rotation().T) + np.asarray([2.1, -0.7, 1.3])
    np.testing.assert_allclose(
        aqueous_cds_tension_design_row(symbols, transformed, atom_areas),
        reference,
        rtol=2.0e-15,
        atol=2.0e-15,
    )

    permutation = np.asarray([3, 6, 1, 7, 0, 4, 2, 5])
    permuted_symbols = tuple(symbols[index] for index in permutation)
    np.testing.assert_allclose(
        aqueous_cds_tension_design_row(
            permuted_symbols,
            positions[permutation],
            atom_areas[permutation],
        ),
        reference,
        rtol=2.0e-15,
        atol=2.0e-15,
    )


@pytest.mark.parametrize(
    ("positions", "areas", "coefficients", "match"),
    [
        (np.zeros((2, 2)), np.ones(2), np.ones(18), "coordinates"),
        (np.zeros((2, 3)), np.ones(3), np.ones(18), "areas"),
        (np.zeros((2, 3)), np.ones(2), np.ones(17), "coefficients"),
        (np.zeros((2, 3)), np.asarray([1.0, np.nan]), np.ones(18), "areas"),
    ],
)
def test_linear_tension_contract_rejects_invalid_inputs(
    positions: np.ndarray,
    areas: np.ndarray,
    coefficients: np.ndarray,
    match: str,
):
    symbols = ("H", "C")
    if match == "areas":
        with pytest.raises(ValueError, match=match):
            aqueous_cds_tension_design_row(symbols, positions, areas)
    else:
        with pytest.raises(ValueError, match=match):
            aqueous_atomic_surface_tensions_from_coefficients(
                symbols,
                positions,
                coefficients,
            )
