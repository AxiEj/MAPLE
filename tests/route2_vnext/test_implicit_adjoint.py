from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from maple.solvation.coupling.adjoint import AdjointOptions
from maple.solvation.coupling.energy import (
    OperationalElectrostaticScalar,
    nonlinear_half_coupling,
)
from maple.solvation.coupling.fixed_point import FixedPointOptions, solve_fixed_point
from maple.solvation.coupling.metrics import ATOMIC_L1_PAIRING
from maple.solvation.coupling.spaces import AffineChargeCoordinates
from maple.solvation.coupling.state_equation import ReducedStateEquation


def _q(atom_count):
    return np.kron(np.eye(atom_count), ATOMIC_L1_PAIRING.block)


@dataclass
class LinearReciprocalContinuum:
    atom_count: int
    base_scale: float
    scale_gradient: np.ndarray
    fixed_topology = True
    linear_response = True
    reciprocal = True

    def scale(self, geometry):
        return self.base_scale + np.vdot(self.scale_gradient, geometry)

    def evaluate_field(self, geometry, source):
        return (self.scale(geometry) * _q(self.atom_count).T @ source.reshape(-1)).reshape(source.shape)

    def source_jvp(self, geometry, source, direction):
        return (self.scale(geometry) * _q(self.atom_count).T @ direction.reshape(-1)).reshape(source.shape)

    def source_vjp(self, geometry, source, cotangent):
        return (self.scale(geometry) * _q(self.atom_count) @ cotangent.reshape(-1)).reshape(source.shape)

    def coordinate_vjp(self, geometry, source, cotangent):
        factor = np.vdot(cotangent.reshape(-1), _q(self.atom_count).T @ source.reshape(-1))
        return factor * self.scale_gradient


@dataclass
class LinearElectronic:
    bias: np.ndarray
    field_matrix: np.ndarray
    geometry_matrix: np.ndarray

    def evaluate_source(self, geometry, field):
        return (self.bias + self.field_matrix @ field.reshape(-1) + self.geometry_matrix @ geometry).reshape(field.shape)

    def field_jvp(self, geometry, field, direction):
        return (self.field_matrix @ direction.reshape(-1)).reshape(field.shape)

    def field_vjp(self, geometry, field, cotangent):
        return (self.field_matrix.T @ cotangent.reshape(-1)).reshape(field.shape)

    def coordinate_vjp(self, geometry, field, cotangent):
        return self.geometry_matrix.T @ cotangent.reshape(-1)


@dataclass
class QuadraticVacuum:
    hessian: np.ndarray
    linear: np.ndarray

    def evaluate_energy(self, geometry):
        return 0.5 * geometry @ self.hessian @ geometry + self.linear @ geometry

    def coordinate_gradient(self, geometry):
        return self.hessian @ geometry + self.linear


@dataclass
class NonlinearContinuum:
    matrix: np.ndarray
    geometry_matrix: np.ndarray

    def argument(self, geometry, source):
        return self.matrix @ source.reshape(-1) + self.geometry_matrix @ geometry

    def evaluate_field(self, geometry, source):
        return np.tanh(self.argument(geometry, source)).reshape(source.shape)

    def source_jvp(self, geometry, source, direction):
        derivative = 1.0 - np.tanh(self.argument(geometry, source)) ** 2
        return (derivative * (self.matrix @ direction.reshape(-1))).reshape(source.shape)

    def source_vjp(self, geometry, source, cotangent):
        derivative = 1.0 - np.tanh(self.argument(geometry, source)) ** 2
        return (self.matrix.T @ (derivative * cotangent.reshape(-1))).reshape(source.shape)

    def coordinate_vjp(self, geometry, source, cotangent):
        derivative = 1.0 - np.tanh(self.argument(geometry, source)) ** 2
        return self.geometry_matrix.T @ (derivative * cotangent.reshape(-1))


def _system():
    rng = np.random.default_rng(29)
    coordinates = AffineChargeCoordinates(2, 0.5, 0.9, 1.1)
    size = coordinates.source_dimension
    continuum = LinearReciprocalContinuum(2, 0.25, np.array([0.02, -0.01, 0.015]))
    electronic = LinearElectronic(
        rng.normal(scale=0.15, size=size),
        rng.normal(scale=0.025, size=(size, size)),
        rng.normal(scale=0.04, size=(size, 3)),
    )
    vacuum = QuadraticVacuum(
        np.array([[1.2, 0.1, 0.0], [0.1, 0.8, -0.05], [0.0, -0.05, 1.0]]),
        np.array([0.2, -0.1, 0.05]),
    )
    equation = ReducedStateEquation(coordinates, electronic, continuum)
    return equation, OperationalElectrostaticScalar(equation, vacuum)


def _root_context(geometry):
    geometry_hash = hashlib.sha256(np.asarray(geometry, dtype=float).tobytes()).hexdigest()
    return f"linear-oracle/fixedtopology-electrostatic-v1/{geometry_hash}"


def test_implicit_adjoint_matches_resolved_scalar_finite_difference():
    equation, scalar = _system()
    geometry = np.array([0.3, -0.2, 0.15])
    primal_options = FixedPointOptions(tolerance=1e-13, max_iterations=80)
    state = solve_fixed_point(
        equation,
        geometry,
        root_context_id=_root_context(geometry),
        options=primal_options,
    )
    gradient = scalar.implicit_gradient(
        geometry,
        state,
        adjoint_options=AdjointOptions(relative_tolerance=1e-13, absolute_tolerance=1e-14),
    )
    step = 1.0e-5
    finite_difference = []
    for axis in range(geometry.size):
        direction = np.zeros_like(geometry)
        direction[axis] = step
        plus_geometry = geometry + direction
        minus_geometry = geometry - direction
        plus_state = solve_fixed_point(
            equation,
            plus_geometry,
            root_context_id=_root_context(plus_geometry),
            options=primal_options,
        )
        minus_state = solve_fixed_point(
            equation,
            minus_geometry,
            root_context_id=_root_context(minus_geometry),
            options=primal_options,
        )
        plus = scalar.evaluate(plus_geometry, plus_state.y).total_energy
        minus = scalar.evaluate(minus_geometry, minus_state.y).total_energy
        finite_difference.append((plus - minus) / (2 * step))
    np.testing.assert_allclose(
        gradient.total_coordinate_gradient, finite_difference, atol=1e-8, rtol=1e-8
    )
    assert gradient.adjoint.converged
    assert gradient.adjoint.true_residual_norm <= gradient.adjoint.acceptance_tolerance


def test_nonlinear_half_coupling_direct_plus_response_matches_fd():
    rng = np.random.default_rng(31)
    atom_count = 2
    size = atom_count * 4
    continuum = NonlinearContinuum(
        rng.normal(scale=0.08, size=(size, size)),
        rng.normal(scale=0.05, size=(size, 3)),
    )
    geometry = rng.normal(scale=0.2, size=3)
    source = rng.normal(scale=0.3, size=(atom_count, 4))
    evaluated = nonlinear_half_coupling(continuum, geometry, source)
    step = 1.0e-6
    finite_difference = np.zeros_like(source)
    for index in range(size):
        direction = np.zeros(size)
        direction[index] = step
        plus = source + direction.reshape(source.shape)
        minus = source - direction.reshape(source.shape)
        plus_energy = nonlinear_half_coupling(continuum, geometry, plus).energy
        minus_energy = nonlinear_half_coupling(continuum, geometry, minus).energy
        finite_difference.reshape(-1)[index] = (plus_energy - minus_energy) / (2 * step)
    np.testing.assert_allclose(
        evaluated.total_source_gradient, finite_difference, atol=1e-9, rtol=1e-9
    )
