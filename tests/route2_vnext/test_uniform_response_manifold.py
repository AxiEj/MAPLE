from __future__ import annotations

import numpy as np
import pytest

from maple.solvation.release.uniform_response_manifold import (
    affine_uniform_native_field,
    molecular_dipole_eangstrom,
    solve_uniform_response_dipole_closure,
)


def _raw_source(charges: np.ndarray, dipoles_xyz: np.ndarray) -> np.ndarray:
    return np.column_stack((charges, dipoles_xyz[:, (1, 2, 0)]))


def test_affine_native_field_is_translation_covariant_and_two_width_exact() -> None:
    positions = np.asarray(((0.0, 0.2, -0.1), (1.1, -0.4, 0.7)))
    gradient = np.asarray((0.3, -0.2, 0.4))
    baseline = affine_uniform_native_field(positions, gradient)
    translated = affine_uniform_native_field(
        positions + np.asarray((4.0, -2.0, 0.5)), gradient
    )
    assert translated == pytest.approx(baseline, abs=1.0e-15)
    assert baseline[:, 0] == pytest.approx(baseline[:, 1], abs=0.0)
    assert baseline[:, 2:5] == pytest.approx(baseline[:, 5:8], abs=0.0)
    assert baseline[0, 2:5] == pytest.approx((-0.2, 0.4, 0.3), abs=0.0)


def test_linear_uniform_response_closes_dipole_without_changing_charge() -> None:
    positions = np.asarray(((-0.7, 0.1, 0.0), (0.8, -0.2, 0.3)))
    charges = np.asarray((0.25, -0.25))
    dipoles = np.asarray(((0.02, 0.01, -0.03), (-0.01, 0.04, 0.02)))
    zero_source = _raw_source(charges, dipoles)
    susceptibility = np.asarray(
        ((0.72, 0.05, -0.02), (0.05, 0.54, 0.01), (-0.02, 0.01, 0.61))
    )
    target_shift = np.asarray((0.08, -0.04, 0.03))
    target = molecular_dipole_eangstrom(positions, zero_source) + target_shift

    def gradient_from_field(field: np.ndarray) -> np.ndarray:
        return field[0, (4, 2, 3)]

    def evaluate(field: np.ndarray) -> np.ndarray:
        shift = susceptibility @ gradient_from_field(field)
        result = zero_source.copy()
        result[0, (3, 1, 2)] += shift
        return result

    def jvp(_field: np.ndarray, direction: np.ndarray) -> np.ndarray:
        shift = susceptibility @ gradient_from_field(direction)
        result = np.zeros_like(zero_source)
        result[0, (3, 1, 2)] = shift
        return result

    solved = solve_uniform_response_dipole_closure(
        positions_angstrom=positions,
        target_molecular_dipole_eangstrom=target,
        evaluate_source=evaluate,
        source_jvp=jvp,
    )
    assert solved.converged
    assert solved.iterations == 1
    assert solved.residual_norm_eangstrom < 1.0e-13
    assert solved.total_charge_change_e == pytest.approx(0.0, abs=0.0)
    assert solved.potential_gradient_ev_per_e_angstrom == pytest.approx(
        np.linalg.solve(susceptibility, target_shift), abs=2.0e-15
    )


def test_solver_is_rotation_covariant_for_isotropic_response() -> None:
    positions = np.asarray(((-0.8, 0.0, 0.1), (0.4, 0.5, -0.2)))
    zero = _raw_source(np.asarray((0.2, -0.2)), np.zeros((2, 3)))
    target_shift = np.asarray((0.06, -0.03, 0.04))
    angle = 0.43
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )

    def solve(pos: np.ndarray, base: np.ndarray, shift: np.ndarray):
        def grad(field: np.ndarray) -> np.ndarray:
            return field[0, (4, 2, 3)]

        def evaluate(field: np.ndarray) -> np.ndarray:
            result = base.copy()
            result[0, (3, 1, 2)] += 0.5 * grad(field)
            return result

        def jvp(_field: np.ndarray, direction: np.ndarray) -> np.ndarray:
            result = np.zeros_like(base)
            result[0, (3, 1, 2)] = 0.5 * grad(direction)
            return result

        return solve_uniform_response_dipole_closure(
            positions_angstrom=pos,
            target_molecular_dipole_eangstrom=(
                molecular_dipole_eangstrom(pos, base) + shift
            ),
            evaluate_source=evaluate,
            source_jvp=jvp,
        )

    baseline = solve(positions, zero, target_shift)
    rotated_positions = positions @ rotation.T
    charges = zero[:, 0]
    dipoles = zero[:, 1:4][:, (2, 0, 1)] @ rotation.T
    rotated_zero = _raw_source(charges, dipoles)
    rotated = solve(rotated_positions, rotated_zero, target_shift @ rotation.T)
    assert rotated.potential_gradient_ev_per_e_angstrom == pytest.approx(
        baseline.potential_gradient_ev_per_e_angstrom @ rotation.T, abs=2.0e-14
    )


def test_charge_drift_and_singular_response_fail_closed() -> None:
    positions = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    zero = _raw_source(np.asarray((0.1, -0.1)), np.zeros((2, 3)))
    target = np.asarray((0.1, -0.04, 0.03))

    def drift(field: np.ndarray) -> np.ndarray:
        result = zero.copy()
        gradient = field[0, (4, 2, 3)]
        result[0, 0] += gradient[0]
        result[0, (3, 1, 2)] += gradient
        return result

    def drift_jvp(_field: np.ndarray, direction: np.ndarray) -> np.ndarray:
        result = np.zeros_like(zero)
        gradient = direction[0, (4, 2, 3)]
        result[0, 0] = gradient[0]
        result[0, (3, 1, 2)] = gradient
        return result

    with pytest.raises(RuntimeError, match="total charge"):
        solve_uniform_response_dipole_closure(
            positions_angstrom=positions,
            target_molecular_dipole_eangstrom=target,
            evaluate_source=drift,
            source_jvp=drift_jvp,
        )

    with pytest.raises(RuntimeError, match="singular or ill-conditioned"):
        solve_uniform_response_dipole_closure(
            positions_angstrom=positions,
            target_molecular_dipole_eangstrom=target,
            evaluate_source=lambda _field: zero,
            source_jvp=lambda _field, _direction: np.zeros_like(zero),
        )
