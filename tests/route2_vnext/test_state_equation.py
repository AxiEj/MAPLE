from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest

from maple.solvation.coupling.fixed_point import FixedPointOptions, solve_fixed_point
from maple.solvation.coupling.metrics import ATOMIC_L1_PAIRING
from maple.solvation.coupling.spaces import AffineChargeCoordinates, SourceSpace
from maple.solvation.coupling.state_equation import ReducedStateEquation


def _q(atom_count: int) -> np.ndarray:
    return np.kron(np.eye(atom_count), ATOMIC_L1_PAIRING.block)


@dataclass
class DenseLinearContinuum:
    base_scale: float
    scale_gradient: np.ndarray
    atom_count: int

    def _scale(self, geometry):
        return self.base_scale + float(np.vdot(self.scale_gradient, geometry))

    def evaluate_field(self, geometry, source):
        return (self._scale(geometry) * _q(self.atom_count).T @ source.reshape(-1)).reshape(
            self.atom_count, 4
        )

    def source_jvp(self, geometry, source, source_direction):
        return (
            self._scale(geometry) * _q(self.atom_count).T @ source_direction.reshape(-1)
        ).reshape(self.atom_count, 4)

    def source_vjp(self, geometry, source, field_cotangent):
        return (
            self._scale(geometry) * _q(self.atom_count) @ field_cotangent.reshape(-1)
        ).reshape(self.atom_count, 4)

    def coordinate_vjp(self, geometry, source, field_cotangent):
        coefficient = np.vdot(
            field_cotangent.reshape(-1), _q(self.atom_count).T @ source.reshape(-1)
        )
        return coefficient * self.scale_gradient


@dataclass
class DenseLinearElectronic:
    bias: np.ndarray
    field_matrix: np.ndarray
    geometry_matrix: np.ndarray

    def evaluate_source(self, geometry, field):
        return (
            self.bias
            + self.field_matrix @ field.reshape(-1)
            + self.geometry_matrix @ geometry
        ).reshape(field.shape)

    def field_jvp(self, geometry, field, field_direction):
        return (self.field_matrix @ field_direction.reshape(-1)).reshape(field.shape)

    def field_vjp(self, geometry, field, source_cotangent):
        return (self.field_matrix.T @ source_cotangent.reshape(-1)).reshape(field.shape)

    def coordinate_vjp(self, geometry, field, source_cotangent):
        return self.geometry_matrix.T @ source_cotangent.reshape(-1)


class DenseNonlinearElectronic(DenseLinearElectronic):
    def _argument(self, geometry, field):
        return self.bias + self.field_matrix @ field.reshape(-1) + self.geometry_matrix @ geometry

    def evaluate_source(self, geometry, field):
        return np.tanh(self._argument(geometry, field)).reshape(field.shape)

    def field_jvp(self, geometry, field, field_direction):
        derivative = 1.0 - np.tanh(self._argument(geometry, field)) ** 2
        return (derivative * (self.field_matrix @ field_direction.reshape(-1))).reshape(field.shape)

    def field_vjp(self, geometry, field, source_cotangent):
        derivative = 1.0 - np.tanh(self._argument(geometry, field)) ** 2
        return (self.field_matrix.T @ (derivative * source_cotangent.reshape(-1))).reshape(field.shape)

    def coordinate_vjp(self, geometry, field, source_cotangent):
        derivative = 1.0 - np.tanh(self._argument(geometry, field)) ** 2
        return self.geometry_matrix.T @ (derivative * source_cotangent.reshape(-1))


def _linear_equation(seed=5):
    rng = np.random.default_rng(seed)
    coordinates = AffineChargeCoordinates(2, total_charge=-1.0, monopole_scale=0.8, dipole_scale=1.2)
    size = coordinates.source_dimension
    electronic = DenseLinearElectronic(
        bias=rng.normal(scale=0.2, size=size),
        field_matrix=rng.normal(scale=0.025, size=(size, size)),
        geometry_matrix=rng.normal(scale=0.03, size=(size, 3)),
    )
    continuum = DenseLinearContinuum(0.3, np.array([0.01, -0.02, 0.015]), 2)
    return ReducedStateEquation(coordinates, electronic, continuum)


def test_residual_is_exact_projected_charge_constrained_formula():
    equation = _linear_equation()
    geometry = np.array([0.2, -0.1, 0.4])
    y = np.linspace(-0.2, 0.3, equation.reduced_dimension)
    evaluated = equation.evaluate(geometry, y)
    c = equation.coordinates.expand(y)
    field = equation.continuum.evaluate_field(geometry, c)
    response = equation.electronic.evaluate_source(geometry, field)
    expected = equation.coordinates.reduce_tangent(
        c - equation.coordinates.project_affine(response)
    )
    np.testing.assert_allclose(evaluated.residual, expected, atol=1e-14)
    assert abs(np.sum(np.asarray(evaluated.projected_source)[:, 0]) + 1.0) <= 1e-12


@pytest.mark.parametrize("nonlinear", [False, True])
def test_exact_reduced_jvp_vjp_and_finite_difference(nonlinear):
    equation = _linear_equation(seed=7)
    if nonlinear:
        linear = equation.electronic
        equation = ReducedStateEquation(
            equation.coordinates,
            DenseNonlinearElectronic(linear.bias, linear.field_matrix, linear.geometry_matrix),
            equation.continuum,
        )
    rng = np.random.default_rng(11)
    geometry = rng.normal(scale=0.2, size=3)
    y = rng.normal(scale=0.15, size=equation.reduced_dimension)
    dy = rng.normal(size=equation.reduced_dimension)
    cotangent = rng.normal(size=equation.reduced_dimension)
    step = 1.0e-6
    fd = (equation.residual(geometry, y + step * dy) - equation.residual(geometry, y - step * dy)) / (2 * step)
    jvp = equation.jvp(geometry, y, dy)
    np.testing.assert_allclose(jvp, fd, atol=1e-9, rtol=1e-9)
    assert np.vdot(jvp, cotangent) == pytest.approx(
        np.vdot(dy, equation.vjp(geometry, y, cotangent)), abs=1e-10
    )


def test_linear_root_matches_dense_solution_and_cold_warm_identity():
    equation = _linear_equation(seed=13)
    geometry = np.array([0.1, 0.2, -0.3])
    zero = np.zeros(equation.reduced_dimension)
    offset = equation.residual(geometry, zero)
    jacobian = np.column_stack(
        [equation.jvp(geometry, zero, basis) for basis in np.eye(equation.reduced_dimension)]
    )
    exact = np.linalg.solve(jacobian, -offset)
    options = FixedPointOptions(method="anderson", tolerance=1e-13, max_iterations=80)
    context = "dense-linear-provider/test-profile/geometry-0.1,0.2,-0.3"
    cold = solve_fixed_point(equation, geometry, root_context_id=context, options=options)
    warm = solve_fixed_point(
        equation,
        geometry,
        root_context_id=context,
        initial_y=exact + 1e-4,
        options=options,
    )
    np.testing.assert_allclose(cold.y, exact, atol=1e-12)
    np.testing.assert_allclose(warm.y, exact, atol=1e-12)
    assert cold.actual_unmixed_residual_norm <= options.tolerance
    assert warm.actual_unmixed_residual_norm <= options.tolerance
    assert cold.root_hash == warm.root_hash
    different_context = solve_fixed_point(
        equation,
        geometry,
        root_context_id=context + "/different-provider",
        options=options,
    )
    assert different_context.root_hash != cold.root_hash
    assert cold.initialization == "cold" and warm.initialization == "warm"
    assert cold.field_array().shape == cold.source_array().shape
    assert abs(np.sum(cold.source_array()[:, 0]) + 1.0) <= 1e-12
    with pytest.raises(FrozenInstanceError):
        cold.converged = False


def test_state_kernel_accepts_non_l1_future_source_space():
    source_space = SourceSpace(
        scalar_id="test.future-multipole.v1",
        representation="charge plus two generic multipoles",
        components=("m1", "net_charge", "m2"),
        units=("u1", "e", "u2"),
        charge_component=1,
    )
    coordinates = AffineChargeCoordinates(
        3,
        2.0,
        source_space=source_space,
        component_scales=(0.7, 0.9, 1.1),
    )

    class Continuum:
        def evaluate_field(self, geometry, source):
            return 0.2 * source

        def source_jvp(self, geometry, source, direction):
            return 0.2 * direction

        def source_vjp(self, geometry, source, cotangent):
            return 0.2 * cotangent

        def coordinate_vjp(self, geometry, source, cotangent):
            return np.zeros_like(geometry)

    class Electronic:
        def evaluate_source(self, geometry, field):
            return 0.1 * field + 0.05

        def field_jvp(self, geometry, field, direction):
            return 0.1 * direction

        def field_vjp(self, geometry, field, cotangent):
            return 0.1 * cotangent

        def coordinate_vjp(self, geometry, field, cotangent):
            return np.zeros_like(geometry)

    equation = ReducedStateEquation(coordinates, Electronic(), Continuum())
    state = solve_fixed_point(
        equation,
        np.zeros(2),
        root_context_id="future-multipole/test-provider/geometry-zero",
        options=FixedPointOptions(tolerance=1e-13),
    )
    assert state.source_array().shape == (3, 3)
    assert state.field_array().shape == (3, 3)
    assert abs(np.sum(state.source_array()[:, source_space.charge_component]) - 2.0) <= 1e-12
