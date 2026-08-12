from __future__ import annotations

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.gto_density import (
    cartesian_multipoles,
    gaussian_multipole_potential,
    point_asc_reaction_position_vjp,
    point_asc_reaction_potential_gradient,
    point_multipole_potential,
    point_multipole_potential_position_vjp,
    point_multipole_potential_surface_position_vjp,
)
from maple.solvation.coupling.gaussian_multipole_derivatives import (
    gaussian_multipole_potential_position_vjp,
    gaussian_multipole_potential_surface_position_vjp,
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


def test_point_multipole_surface_position_vjp_matches_central_difference():
    positions_angstrom, surface_bohr, coefficients, asc = _fixed_kernel_inputs()
    analytic = point_multipole_potential_surface_position_vjp(
        surface_bohr,
        positions_angstrom,
        coefficients,
        asc,
    )
    finite_difference = np.empty_like(surface_bohr)
    step_bohr = 1.0e-6

    for point_index in range(surface_bohr.shape[0]):
        for coordinate in range(3):
            plus = surface_bohr.copy()
            minus = surface_bohr.copy()
            plus[point_index, coordinate] += step_bohr
            minus[point_index, coordinate] -= step_bohr
            finite_difference[point_index, coordinate] = (
                np.dot(
                    asc,
                    point_multipole_potential(plus, positions_angstrom, coefficients),
                )
                - np.dot(
                    asc,
                    point_multipole_potential(minus, positions_angstrom, coefficients),
                )
            ) / (2.0 * step_bohr)

    assert analytic == pytest.approx(
        finite_difference,
        rel=2.0e-8,
        abs=2.0e-9,
    )


def test_point_multipole_position_vjps_close_under_rigid_translation():
    positions_angstrom, surface_bohr, coefficients, asc = _fixed_kernel_inputs()
    atom_vjp_per_angstrom = point_multipole_potential_position_vjp(
        surface_bohr,
        positions_angstrom,
        coefficients,
        asc,
    )
    surface_vjp_per_bohr = point_multipole_potential_surface_position_vjp(
        surface_bohr,
        positions_angstrom,
        coefficients,
        asc,
    )

    total_vjp_per_angstrom = (
        np.sum(
            atom_vjp_per_angstrom,
            axis=0,
        )
        + np.sum(surface_vjp_per_bohr, axis=0) / Bohr
    )

    assert total_vjp_per_angstrom == pytest.approx(
        np.zeros(3),
        rel=0.0,
        abs=2.0e-12,
    )


@pytest.mark.parametrize("sigma_angstrom", [0.7, 1.5, 3.0])
def test_gaussian_multipole_position_vjps_match_central_difference(
    sigma_angstrom,
):
    positions_angstrom, surface_bohr, coefficients, asc = _fixed_kernel_inputs()
    atom_analytic = gaussian_multipole_potential_position_vjp(
        surface_bohr,
        positions_angstrom,
        coefficients,
        asc,
        sigma_angstrom=sigma_angstrom,
    )
    surface_analytic = gaussian_multipole_potential_surface_position_vjp(
        surface_bohr,
        positions_angstrom,
        coefficients,
        asc,
        sigma_angstrom=sigma_angstrom,
    )
    atom_fd = np.empty_like(positions_angstrom)
    surface_fd = np.empty_like(surface_bohr)
    step = 1.0e-6

    for atom_index in range(positions_angstrom.shape[0]):
        for coordinate in range(3):
            plus = positions_angstrom.copy()
            minus = positions_angstrom.copy()
            plus[atom_index, coordinate] += step
            minus[atom_index, coordinate] -= step
            atom_fd[atom_index, coordinate] = (
                np.dot(
                    asc,
                    gaussian_multipole_potential(
                        surface_bohr,
                        plus,
                        coefficients,
                        sigma_angstrom=sigma_angstrom,
                    ),
                )
                - np.dot(
                    asc,
                    gaussian_multipole_potential(
                        surface_bohr,
                        minus,
                        coefficients,
                        sigma_angstrom=sigma_angstrom,
                    ),
                )
            ) / (2.0 * step)

    step_bohr = step
    for point_index in range(surface_bohr.shape[0]):
        for coordinate in range(3):
            plus = surface_bohr.copy()
            minus = surface_bohr.copy()
            plus[point_index, coordinate] += step_bohr
            minus[point_index, coordinate] -= step_bohr
            surface_fd[point_index, coordinate] = (
                np.dot(
                    asc,
                    gaussian_multipole_potential(
                        plus,
                        positions_angstrom,
                        coefficients,
                        sigma_angstrom=sigma_angstrom,
                    ),
                )
                - np.dot(
                    asc,
                    gaussian_multipole_potential(
                        minus,
                        positions_angstrom,
                        coefficients,
                        sigma_angstrom=sigma_angstrom,
                    ),
                )
            ) / (2.0 * step_bohr)

    np.testing.assert_allclose(atom_analytic, atom_fd, rtol=3e-8, atol=2e-9)
    np.testing.assert_allclose(surface_analytic, surface_fd, rtol=3e-8, atol=2e-9)
    np.testing.assert_allclose(
        atom_analytic.sum(axis=0) + surface_analytic.sum(axis=0) / Bohr,
        np.zeros(3),
        rtol=0.0,
        atol=3e-12,
    )


def test_gaussian_multipole_gradient_is_finite_at_its_center():
    positions = np.array([[0.2, -0.1, 0.3]])
    points = positions / Bohr
    coefficients = np.array([[0.4, -0.2, 0.1, 0.3]])
    cotangent = np.array([0.7])
    atom = gaussian_multipole_potential_position_vjp(
        points, positions, coefficients, cotangent, sigma_angstrom=1.5
    )
    surface = gaussian_multipole_potential_surface_position_vjp(
        points, positions, coefficients, cotangent, sigma_angstrom=1.5
    )
    assert np.all(np.isfinite(atom))
    assert np.all(np.isfinite(surface))
    np.testing.assert_allclose(atom + surface / Bohr, np.zeros((1, 3)), atol=2e-14)


@pytest.mark.parametrize(
    ("surface_bohr", "surface_cotangent", "message"),
    [
        (np.zeros(3), np.zeros(1), "shape"),
        (np.full((1, 3), np.nan), np.zeros(1), "finite"),
        (np.zeros((1, 3)), np.full(1, np.inf), "finite"),
        (np.zeros((2, 3)), np.zeros(1), "length"),
    ],
)
def test_point_multipole_surface_position_vjp_rejects_invalid_inputs(
    surface_bohr,
    surface_cotangent,
    message,
):
    positions_angstrom, _, coefficients, _ = _fixed_kernel_inputs()

    with pytest.raises(ValueError, match=message):
        point_multipole_potential_surface_position_vjp(
            surface_bohr,
            positions_angstrom,
            coefficients,
            surface_cotangent,
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
            potential_minus, gradient_minus = point_asc_reaction_potential_gradient(
                minus, surface_bohr, asc
            )
            coupling_plus = np.dot(charges, potential_plus) + np.einsum(
                "ij,ij->", dipoles_bohr, gradient_plus
            )
            coupling_minus = np.dot(charges, potential_minus) + np.einsum(
                "ij,ij->", dipoles_bohr, gradient_minus
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
