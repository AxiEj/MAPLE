from __future__ import annotations

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.gto_density import (
    cartesian_multipoles,
    point_asc_reaction_position_vjp,
    point_asc_reaction_potential_gradient,
    point_multipole_potential,
    point_multipole_potential_position_vjp,
)


def _fixed_kernel_inputs():
    positions_angstrom = np.asarray(
        [
            [-0.7, 0.2, 0.1],
            [0.8, -0.3, 0.4],
            [0.1, 0.9, -0.5],
        ]
    )
    surface_bohr = np.asarray(
        [
            [5.2, -1.1, 0.7],
            [-4.8, 2.3, 1.4],
            [1.7, 4.9, -2.2],
            [-2.6, -4.1, 3.3],
        ]
    )
    coefficients = np.asarray(
        [
            [0.4, -0.2, 0.1, 0.3],
            [-0.6, 0.4, -0.1, 0.2],
            [0.2, 0.3, 0.5, -0.4],
        ]
    )
    asc = np.asarray([0.03, -0.02, 0.015, -0.01])
    return positions_angstrom, surface_bohr, coefficients, asc


def test_point_multipole_position_vjp_matches_central_difference():
    positions_angstrom, surface_bohr, coefficients, asc = _fixed_kernel_inputs()
    analytic = point_multipole_potential_position_vjp(
        surface_bohr, positions_angstrom, coefficients, asc
    )
    finite_difference = np.empty_like(positions_angstrom)
    step_angstrom = 1.0e-6

    for atom_index in range(positions_angstrom.shape[0]):
        for coordinate in range(3):
            plus = positions_angstrom.copy()
            minus = positions_angstrom.copy()
            plus[atom_index, coordinate] += step_angstrom
            minus[atom_index, coordinate] -= step_angstrom
            finite_difference[atom_index, coordinate] = (
                np.dot(
                    asc,
                    point_multipole_potential(surface_bohr, plus, coefficients),
                )
                - np.dot(
                    asc,
                    point_multipole_potential(surface_bohr, minus, coefficients),
                )
            ) / (2.0 * step_angstrom)

    assert analytic == pytest.approx(
        finite_difference,
        rel=2.0e-8,
        abs=2.0e-9,
    )


def test_point_asc_position_vjp_matches_central_difference():
    positions_angstrom, surface_bohr, coefficients, asc = _fixed_kernel_inputs()
    charges, dipoles_angstrom = cartesian_multipoles(coefficients)
    dipoles_bohr = dipoles_angstrom / Bohr
    analytic = point_asc_reaction_position_vjp(
        positions_angstrom,
        surface_bohr,
        asc,
        charges,
        dipoles_bohr,
    )
    finite_difference = np.empty_like(positions_angstrom)
    step_angstrom = 1.0e-6

    for atom_index in range(positions_angstrom.shape[0]):
        for coordinate in range(3):
            plus = positions_angstrom.copy()
            minus = positions_angstrom.copy()
            plus[atom_index, coordinate] += step_angstrom
            minus[atom_index, coordinate] -= step_angstrom
            potential_plus, gradient_plus = point_asc_reaction_potential_gradient(
                plus, surface_bohr, asc
            )
            potential_minus, gradient_minus = (
                point_asc_reaction_potential_gradient(minus, surface_bohr, asc)
            )
            coupling_plus = (
                np.dot(charges, potential_plus)
                + np.einsum("ij,ij->", dipoles_bohr, gradient_plus)
            )
            coupling_minus = (
                np.dot(charges, potential_minus)
                + np.einsum("ij,ij->", dipoles_bohr, gradient_minus)
            )
            finite_difference[atom_index, coordinate] = (
                coupling_plus - coupling_minus
            ) / (2.0 * step_angstrom)

    assert analytic == pytest.approx(
        finite_difference,
        rel=2.0e-8,
        abs=2.0e-9,
    )


def test_fixed_kernel_position_derivatives_preserve_reciprocity():
    positions_angstrom, surface_bohr, coefficients, asc = _fixed_kernel_inputs()
    charges, dipoles_angstrom = cartesian_multipoles(coefficients)
    dipoles_bohr = dipoles_angstrom / Bohr

    surface_side = point_multipole_potential_position_vjp(
        surface_bohr,
        positions_angstrom,
        coefficients,
        asc,
    )
    atom_side = point_asc_reaction_position_vjp(
        positions_angstrom,
        surface_bohr,
        asc,
        charges,
        dipoles_bohr,
    )

    assert atom_side == pytest.approx(surface_side, rel=1.0e-12, abs=1.0e-12)
