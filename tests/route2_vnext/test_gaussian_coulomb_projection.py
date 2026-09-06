from __future__ import annotations

import math

import numpy as np
import pytest

from maple.solvation.coupling.gaussian_coulomb_projection import (
    charge_dipole_constraint_matrix,
    gaussian_charge_dipole_coulomb_gram,
    project_charge_dipoles_in_gaussian_coulomb_metric,
)


def _rotation() -> np.ndarray:
    axis = np.asarray((0.31, -0.47, 0.82), dtype=float)
    axis /= np.linalg.norm(axis)
    angle = 0.73
    cross = np.asarray(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        )
    )
    return (
        math.cos(angle) * np.eye(3)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * cross
    )


def test_one_center_gram_matches_analytic_self_energy() -> None:
    sigma = 1.5
    gram = gaussian_charge_dipole_coulomb_gram(np.zeros((1, 3)), sigma_angstrom=sigma)
    expected = np.diag(
        (
            1.0 / (math.sqrt(math.pi) * sigma),
            1.0 / (6.0 * math.sqrt(math.pi) * sigma**3),
            1.0 / (6.0 * math.sqrt(math.pi) * sigma**3),
            1.0 / (6.0 * math.sqrt(math.pi) * sigma**3),
        )
    )
    np.testing.assert_allclose(gram, expected, rtol=0.0, atol=2.0e-16)


def test_pair_blocks_are_derivatives_of_gaussian_charge_interaction() -> None:
    sigma = 1.5
    positions = np.asarray(((0.1, -0.2, 0.3), (1.4, 0.7, -0.5)))
    gram = gaussian_charge_dipole_coulomb_gram(positions, sigma_angstrom=sigma)
    direction = np.asarray((0.37, -0.19, 0.28))
    step = 1.0e-5

    def qq(displaced: np.ndarray) -> float:
        return float(
            gaussian_charge_dipole_coulomb_gram(displaced, sigma_angstrom=sigma)[0, 4]
        )

    plus_b = positions.copy()
    minus_b = positions.copy()
    plus_b[1] += step * direction
    minus_b[1] -= step * direction
    derivative_b = (qq(plus_b) - qq(minus_b)) / (2.0 * step)
    assert derivative_b == pytest.approx(float(gram[0, 5:8] @ direction), abs=2.0e-11)

    plus_a = positions.copy()
    minus_a = positions.copy()
    plus_a[0] += step * direction
    minus_a[0] -= step * direction
    derivative_a = (qq(plus_a) - qq(minus_a)) / (2.0 * step)
    assert derivative_a == pytest.approx(float(gram[1:4, 4] @ direction), abs=2.0e-11)


def test_projection_closes_constraints_and_kkt_stationarity() -> None:
    positions = np.asarray(((-0.8, 0.2, 0.1), (0.7, -0.4, 0.3), (1.6, 0.5, -0.6)))
    charges = np.asarray((0.12, -0.08, -0.01))
    dipoles = np.asarray(
        ((0.03, -0.01, 0.02), (-0.04, 0.07, 0.01), (0.02, 0.01, -0.03))
    )
    target_charge = 0.0
    target_dipole = np.asarray((0.42, -0.31, 0.17))
    projected = project_charge_dipoles_in_gaussian_coulomb_metric(
        positions_angstrom=positions,
        charges_e=charges,
        dipoles_eangstrom=dipoles,
        target_total_charge_e=target_charge,
        target_molecular_dipole_eangstrom=target_dipole,
        sigma_angstrom=1.5,
    )
    source = np.concatenate((charges[:, None], dipoles), axis=1).reshape(-1)
    result = np.concatenate(
        (projected.charges_e[:, None], projected.dipoles_eangstrom), axis=1
    ).reshape(-1)
    constraint = charge_dipole_constraint_matrix(positions)
    target = np.concatenate(([target_charge], target_dipole))
    np.testing.assert_allclose(constraint @ result, target, atol=2.0e-13, rtol=0.0)

    gram = gaussian_charge_dipole_coulomb_gram(positions, sigma_angstrom=1.5)
    correction_gradient = gram @ (result - source)
    tangent = np.random.default_rng(17).normal(size=len(result))
    tangent -= constraint.T @ np.linalg.solve(
        constraint @ constraint.T, constraint @ tangent
    )
    assert float(correction_gradient @ tangent) == pytest.approx(0.0, abs=3.0e-13)


def test_projection_is_rotation_translation_and_permutation_covariant() -> None:
    positions = np.asarray(((-0.8, 0.2, 0.1), (0.7, -0.4, 0.3), (1.6, 0.5, -0.6)))
    charges = np.asarray((0.12, -0.08, -0.01))
    dipoles = np.asarray(
        ((0.03, -0.01, 0.02), (-0.04, 0.07, 0.01), (0.02, 0.01, -0.03))
    )
    target_charge = 0.0
    target_dipole = np.asarray((0.42, -0.31, 0.17))
    baseline = project_charge_dipoles_in_gaussian_coulomb_metric(
        positions_angstrom=positions,
        charges_e=charges,
        dipoles_eangstrom=dipoles,
        target_total_charge_e=target_charge,
        target_molecular_dipole_eangstrom=target_dipole,
        sigma_angstrom=1.5,
    )

    rotation = _rotation()
    translation = np.asarray((2.3, -1.1, 0.7))
    moved = project_charge_dipoles_in_gaussian_coulomb_metric(
        positions_angstrom=positions @ rotation.T + translation,
        charges_e=charges,
        dipoles_eangstrom=dipoles @ rotation.T,
        target_total_charge_e=target_charge,
        target_molecular_dipole_eangstrom=(
            target_dipole @ rotation.T + target_charge * translation
        ),
        sigma_angstrom=1.5,
    )
    np.testing.assert_allclose(
        moved.charges_e, baseline.charges_e, atol=2.0e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        moved.dipoles_eangstrom,
        baseline.dipoles_eangstrom @ rotation.T,
        atol=3.0e-12,
        rtol=0.0,
    )

    permutation = np.asarray((2, 0, 1))
    permuted = project_charge_dipoles_in_gaussian_coulomb_metric(
        positions_angstrom=positions[permutation],
        charges_e=charges[permutation],
        dipoles_eangstrom=dipoles[permutation],
        target_total_charge_e=target_charge,
        target_molecular_dipole_eangstrom=target_dipole,
        sigma_angstrom=1.5,
    )
    np.testing.assert_allclose(
        permuted.charges_e, baseline.charges_e[permutation], atol=2.0e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        permuted.dipoles_eangstrom,
        baseline.dipoles_eangstrom[permutation],
        atol=3.0e-12,
        rtol=0.0,
    )


def test_one_atom_projection_is_well_posed() -> None:
    result = project_charge_dipoles_in_gaussian_coulomb_metric(
        positions_angstrom=np.asarray(((0.4, -0.3, 0.2),)),
        charges_e=np.asarray((0.7,)),
        dipoles_eangstrom=np.asarray(((0.1, 0.2, -0.4),)),
        target_total_charge_e=-1.0,
        target_molecular_dipole_eangstrom=np.asarray((0.8, -0.1, 0.3)),
        sigma_angstrom=1.5,
    )
    np.testing.assert_allclose(result.charges_e, (-1.0,), atol=2.0e-14)
    np.testing.assert_allclose(
        result.dipoles_eangstrom,
        np.asarray(((1.2, -0.4, 0.5),)),
        atol=2.0e-14,
    )


def test_projection_fails_closed_on_coincident_basis() -> None:
    with pytest.raises(np.linalg.LinAlgError, match="not numerically positive"):
        project_charge_dipoles_in_gaussian_coulomb_metric(
            positions_angstrom=np.zeros((2, 3)),
            charges_e=np.zeros(2),
            dipoles_eangstrom=np.zeros((2, 3)),
            target_total_charge_e=0.0,
            target_molecular_dipole_eangstrom=np.zeros(3),
            sigma_angstrom=1.5,
        )
