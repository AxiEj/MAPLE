from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_response import (
    NeutralDensityCoordinates,
    UnmixedDensityResidualLinearization,
    project_neutral_density_tangent,
    solve_adjoint,
)


class _MatrixLinearMap:
    def __init__(self, matrix: np.ndarray, atom_count: int):
        self.matrix = np.asarray(matrix, dtype=float)
        self.atom_count = atom_count

    def apply(self, values: np.ndarray) -> np.ndarray:
        flat = np.asarray(values, dtype=float).reshape(-1)
        return (self.matrix @ flat).reshape(self.atom_count, 4)

    def adjoint(self, cotangent: np.ndarray) -> np.ndarray:
        flat = np.asarray(cotangent, dtype=float).reshape(-1)
        return (self.matrix.T @ flat).reshape(self.atom_count, 4)


class _MatrixDensityResponse:
    def __init__(self, jacobian: np.ndarray, atom_count: int):
        self.jacobian = np.asarray(jacobian, dtype=float)
        self.atom_count = atom_count

    def jvp(self, field_direction: np.ndarray) -> np.ndarray:
        flat = np.asarray(field_direction, dtype=float).reshape(-1)
        return (self.jacobian @ flat).reshape(self.atom_count, 4)

    def vjp(self, density_cotangent: np.ndarray) -> np.ndarray:
        flat = np.asarray(density_cotangent, dtype=float).reshape(-1)
        return (self.jacobian.T @ flat).reshape(self.atom_count, 4)


def _neutral_projector(atom_count: int) -> np.ndarray:
    dimension = 4 * atom_count
    projector = np.eye(dimension)
    charge_indices = np.arange(0, dimension, 4)
    projector[np.ix_(charge_indices, charge_indices)] -= (
        np.ones((atom_count, atom_count)) / atom_count
    )
    return projector


def test_unmixed_residual_jvp_vjp_match_dense_neutral_subspace_operator():
    atom_count = 3
    dimension = 4 * atom_count
    rng = np.random.default_rng(20260724)
    reaction_matrix = rng.normal(scale=0.08, size=(dimension, dimension))
    response_jacobian = rng.normal(scale=0.06, size=(dimension, dimension))
    projector = _neutral_projector(atom_count)
    dense = projector @ (
        np.eye(dimension) - response_jacobian @ reaction_matrix
    ) @ projector
    linearization = UnmixedDensityResidualLinearization(
        atom_count=atom_count,
        reaction_field=_MatrixLinearMap(reaction_matrix, atom_count),
        density_response=_MatrixDensityResponse(response_jacobian, atom_count),
    )
    direction = project_neutral_density_tangent(
        rng.normal(size=(atom_count, 4))
    )
    cotangent = project_neutral_density_tangent(
        rng.normal(size=(atom_count, 4))
    )

    jvp = linearization.jvp(direction)
    vjp = linearization.vjp(cotangent)

    np.testing.assert_allclose(
        jvp.reshape(-1),
        dense @ direction.reshape(-1),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        vjp.reshape(-1),
        dense.T @ cotangent.reshape(-1),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert np.vdot(cotangent, jvp) == pytest.approx(
        np.vdot(vjp, direction),
        abs=1.0e-12,
    )
    assert abs(float(np.sum(jvp[:, 0]))) <= 1.0e-14
    assert abs(float(np.sum(vjp[:, 0]))) <= 1.0e-14


def test_unmixed_residual_rejects_non_neutral_tangent_vectors():
    atom_count = 2
    identity = np.eye(4 * atom_count)
    linearization = UnmixedDensityResidualLinearization(
        atom_count=atom_count,
        reaction_field=_MatrixLinearMap(identity, atom_count),
        density_response=_MatrixDensityResponse(identity, atom_count),
    )
    non_neutral = np.zeros((atom_count, 4))
    non_neutral[0, 0] = 0.1

    with pytest.raises(ValueError, match="neutral density tangent"):
        linearization.jvp(non_neutral)
    with pytest.raises(ValueError, match="neutral density tangent"):
        linearization.vjp(non_neutral)


def test_neutral_density_projection_only_changes_monopoles():
    values = np.asarray(
        [
            [0.3, 1.0, 2.0, 3.0],
            [-0.1, 4.0, 5.0, 6.0],
            [0.4, 7.0, 8.0, 9.0],
        ]
    )

    projected = project_neutral_density_tangent(values)

    assert float(np.sum(projected[:, 0])) == pytest.approx(0.0, abs=1.0e-15)
    np.testing.assert_array_equal(projected[:, 1:], values[:, 1:])
    np.testing.assert_array_equal(values[:, 0], [0.3, -0.1, 0.4])


def test_neutral_density_coordinates_are_an_isometric_round_trip():
    rng = np.random.default_rng(11)
    coordinates = NeutralDensityCoordinates(4)
    vector = rng.normal(size=coordinates.dimension)
    tangent = coordinates.expand(vector)

    np.testing.assert_allclose(
        coordinates.reduce(tangent),
        vector,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert np.linalg.norm(tangent) == pytest.approx(
        np.linalg.norm(vector),
        abs=1.0e-14,
    )
    assert float(np.sum(tangent[:, 0])) == pytest.approx(0.0, abs=1.0e-15)


def test_matrix_free_adjoint_solver_matches_dense_neutral_solution():
    atom_count = 3
    dimension = 4 * atom_count
    rng = np.random.default_rng(17)
    reaction_matrix = rng.normal(scale=0.05, size=(dimension, dimension))
    response_jacobian = rng.normal(scale=0.04, size=(dimension, dimension))
    projector = _neutral_projector(atom_count)
    dense = projector @ (
        np.eye(dimension) - response_jacobian @ reaction_matrix
    ) @ projector
    linearization = UnmixedDensityResidualLinearization(
        atom_count=atom_count,
        reaction_field=_MatrixLinearMap(reaction_matrix, atom_count),
        density_response=_MatrixDensityResponse(response_jacobian, atom_count),
    )
    coordinates = NeutralDensityCoordinates(atom_count)
    embedding = np.column_stack(
        [
            coordinates.expand(np.eye(coordinates.dimension)[column]).reshape(-1)
            for column in range(coordinates.dimension)
        ]
    )
    rhs = project_neutral_density_tangent(
        rng.normal(size=(atom_count, 4))
    )
    reduced_rhs = coordinates.reduce(rhs)
    expected_reduced = np.linalg.solve(
        embedding.T @ dense.T @ embedding,
        reduced_rhs,
    )

    result = solve_adjoint(
        linearization,
        rhs,
        relative_tolerance=1.0e-11,
        absolute_tolerance=1.0e-13,
    )

    np.testing.assert_allclose(
        coordinates.reduce(result.solution),
        expected_reduced,
        rtol=1.0e-10,
        atol=1.0e-11,
    )
    assert result.residual_callback_count > 0
    assert result.operator_applications >= result.residual_callback_count
    assert result.relative_residual <= 1.0e-11


def test_matrix_free_adjoint_solver_fails_closed_for_singular_operator():
    atom_count = 2
    identity = np.eye(4 * atom_count)
    linearization = UnmixedDensityResidualLinearization(
        atom_count=atom_count,
        reaction_field=_MatrixLinearMap(identity, atom_count),
        density_response=_MatrixDensityResponse(identity, atom_count),
    )
    rhs = project_neutral_density_tangent(
        np.arange(4 * atom_count, dtype=float).reshape(atom_count, 4)
    )

    with pytest.raises(RuntimeError, match="GMRES did not satisfy"):
        solve_adjoint(
            linearization,
            rhs,
            relative_tolerance=1.0e-12,
            absolute_tolerance=0.0,
            max_iterations=2,
        )
