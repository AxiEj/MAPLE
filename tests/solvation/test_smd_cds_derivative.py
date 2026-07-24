from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    HARTREE_TO_KCAL_MOL,
    aqueous_atomic_surface_tension_position_vjp,
    aqueous_atomic_surface_tensions,
    smd_water_cds,
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
            gradient[atom_index, coordinate] = (
                scalar(plus) - scalar(minus)
            ) / (2.0 * step_angstrom)
    return gradient


def test_atomic_tension_position_vjp_matches_all_switching_branches():
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
        ]
    )
    cotangent = np.asarray([0.7, -0.3, 0.4, 0.2, -0.5, 0.9, -0.1, 0.6])

    analytic = aqueous_atomic_surface_tension_position_vjp(
        symbols,
        positions,
        cotangent,
    )
    finite_difference = _coordinate_finite_difference(
        positions,
        lambda displaced: float(
            np.dot(
                cotangent,
                aqueous_atomic_surface_tensions(symbols, displaced),
            )
        ),
    )

    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=1.0e-8,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(analytic.sum(axis=0), 0.0, atol=5.0e-13)


def test_atomic_tension_vjp_gives_fixed_area_cds_energy_component():
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
    cds = smd_water_cds(symbols, positions)
    energy_cotangent = (
        cds.atom_areas_angstrom2 / (1000.0 * HARTREE_TO_KCAL_MOL)
    )

    analytic_hartree_per_angstrom = (
        aqueous_atomic_surface_tension_position_vjp(
            symbols,
            positions,
            energy_cotangent,
        )
    )
    finite_difference = _coordinate_finite_difference(
        positions,
        lambda displaced: float(
            np.dot(
                aqueous_atomic_surface_tensions(symbols, displaced),
                cds.atom_areas_angstrom2,
            )
            / (1000.0 * HARTREE_TO_KCAL_MOL)
        ),
    )

    np.testing.assert_allclose(
        analytic_hartree_per_angstrom,
        finite_difference,
        rtol=1.0e-8,
        atol=2.0e-10,
    )


def test_atomic_tension_position_vjp_fails_closed_for_coincident_active_pair():
    with pytest.raises(ValueError, match="cannot occupy the same position"):
        aqueous_atomic_surface_tension_position_vjp(
            ("H", "C"),
            np.zeros((2, 3)),
            np.ones(2),
        )


@pytest.mark.parametrize(
    ("positions", "cotangent"),
    [
        (np.zeros((2, 2)), np.ones(2)),
        (np.full((2, 3), np.nan), np.ones(2)),
        (np.zeros((2, 3)), np.ones(3)),
        (np.zeros((2, 3)), np.asarray([0.0, np.nan])),
    ],
)
def test_atomic_tension_position_vjp_rejects_invalid_inputs(
    positions,
    cotangent,
):
    with pytest.raises(ValueError, match="must be finite with shape"):
        aqueous_atomic_surface_tension_position_vjp(
            ("H", "C"),
            positions,
            cotangent,
        )
