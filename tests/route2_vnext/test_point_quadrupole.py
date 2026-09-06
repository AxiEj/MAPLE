from __future__ import annotations

import numpy as np

from maple.solvation.coupling.point_quadrupole import (
    TRACELESS_QUADRUPOLE_BASIS,
    point_traceless_quadrupole_surface_operator,
    quadrupole_rotation_matrix,
    traceless_quadrupole_coefficients,
    traceless_quadrupole_tensors,
)
from maple.solvation.release.cartesian_multipole_mep import (
    cartesian_atomic_multipole_potential,
)


def _rotation() -> np.ndarray:
    axis = np.asarray([0.4, -0.7, 0.2])
    axis /= np.linalg.norm(axis)
    angle = 0.73
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross


def test_traceless_quadrupole_basis_is_orthonormal_and_roundtrips() -> None:
    gram = np.einsum(
        "aij,bij->ab",
        TRACELESS_QUADRUPOLE_BASIS,
        TRACELESS_QUADRUPOLE_BASIS,
    )
    np.testing.assert_allclose(gram, np.eye(5), rtol=0.0, atol=3.0e-16)
    np.testing.assert_allclose(
        np.trace(TRACELESS_QUADRUPOLE_BASIS, axis1=1, axis2=2),
        np.zeros(5),
        rtol=0.0,
        atol=0.0,
    )
    generator = np.random.default_rng(20260827)
    coefficients = generator.normal(size=(4, 5))
    np.testing.assert_allclose(
        traceless_quadrupole_coefficients(
            traceless_quadrupole_tensors(coefficients)
        ),
        coefficients,
        rtol=3.0e-16,
        atol=3.0e-16,
    )


def test_point_quadrupole_operator_matches_cartesian_multipole_kernel() -> None:
    generator = np.random.default_rng(71)
    centers = generator.normal(size=(3, 3))
    points = generator.normal(size=(17, 3)) * 4.0 + np.asarray([7.0, -3.0, 5.0])
    coefficients = generator.normal(size=(3, 5))
    operator = point_traceless_quadrupole_surface_operator(
        points_bohr=points,
        centers_angstrom=centers,
    )
    expected = cartesian_atomic_multipole_potential(
        points_bohr=points,
        centers_angstrom=centers,
        charges_e=np.zeros(3),
        dipoles_eangstrom=np.zeros((3, 3)),
        quadrupoles_eangstrom2=traceless_quadrupole_tensors(coefficients),
    )
    np.testing.assert_allclose(
        operator @ coefficients.reshape(-1),
        expected,
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_point_quadrupole_operator_is_rotation_covariant() -> None:
    generator = np.random.default_rng(93)
    centers = generator.normal(size=(2, 3))
    points = generator.normal(size=(13, 3)) * 3.0 + np.asarray([5.0, 4.0, -6.0])
    coefficients = generator.normal(size=(2, 5))
    rotation = _rotation()
    representation = quadrupole_rotation_matrix(rotation)
    rotated_coefficients = coefficients @ representation.T
    reference = point_traceless_quadrupole_surface_operator(
        points_bohr=points,
        centers_angstrom=centers,
    ) @ coefficients.reshape(-1)
    rotated = point_traceless_quadrupole_surface_operator(
        points_bohr=points @ rotation.T,
        centers_angstrom=centers @ rotation.T,
    ) @ rotated_coefficients.reshape(-1)

    np.testing.assert_allclose(
        representation @ representation.T,
        np.eye(5),
        rtol=0.0,
        atol=8.0e-16,
    )
    np.testing.assert_allclose(rotated, reference, rtol=3.0e-14, atol=3.0e-14)
