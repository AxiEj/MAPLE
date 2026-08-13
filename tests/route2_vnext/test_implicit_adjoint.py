from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import ClassVar

import numpy as np
import pytest

from maple.solvation.api.profiles import (
    EXACT_GTO_COUPLING_CANDIDATE_ID,
    OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1,
)
from maple.solvation.coupling.adjoint import AdjointOptions
from maple.solvation.coupling.energy import (
    OperationalElectrostaticScalar,
    nonlinear_half_coupling,
)
from maple.solvation.coupling.fixed_point import (
    FixedPointOptions,
    roots_numerically_equivalent,
    solve_fixed_point,
)
from maple.solvation.coupling.metrics import ATOMIC_L1_PAIRING, PairingMetric
from maple.solvation.coupling.spaces import AffineChargeCoordinates
from maple.solvation.coupling.spaces import (
    ATOMIC_L1_FIELD_DUAL_SPACE,
    ATOMIC_L1_SOURCE_SPACE,
    FieldDualSpace,
    SourceSpace,
)
from maple.solvation.coupling.state_equation import ReducedStateEquation


def _q(atom_count):
    return np.kron(np.eye(atom_count), ATOMIC_L1_PAIRING.block)


def _configuration_sha256(*values):
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass
class LinearReciprocalContinuum:
    atom_count: int
    base_scale: float
    scale_gradient: np.ndarray
    fixed_topology = True
    linear_response = True
    reciprocal = True
    provider_id: ClassVar[str] = "maple.route2.test.linear-continuum.v1"
    continuum_profile_id: ClassVar[str] = "fixed-topology-linear-reciprocal-cpcm-v1"
    cavity_profile_id: ClassVar[str] = "fixed-topology-amplitude-swig-v1"
    coupling_id: ClassVar[str] = EXACT_GTO_COUPLING_CANDIDATE_ID
    provenance_sha256: ClassVar[str] = "c" * 64
    source_space: ClassVar[SourceSpace] = ATOMIC_L1_SOURCE_SPACE
    field_space: ClassVar[FieldDualSpace] = ATOMIC_L1_FIELD_DUAL_SPACE

    def configuration_sha256(self):
        return _configuration_sha256(
            self.atom_count, self.base_scale, self.scale_gradient
        )

    def scale(self, geometry):
        return self.base_scale + np.vdot(self.scale_gradient, geometry)

    def evaluate_field(self, geometry, source):
        return (
            self.scale(geometry) * _q(self.atom_count).T @ source.reshape(-1)
        ).reshape(source.shape)

    def source_jvp(self, geometry, source, direction):
        return (
            self.scale(geometry) * _q(self.atom_count).T @ direction.reshape(-1)
        ).reshape(source.shape)

    def source_vjp(self, geometry, source, cotangent):
        return (
            self.scale(geometry) * _q(self.atom_count) @ cotangent.reshape(-1)
        ).reshape(source.shape)

    def coordinate_vjp(self, geometry, source, cotangent):
        factor = np.vdot(
            cotangent.reshape(-1), _q(self.atom_count).T @ source.reshape(-1)
        )
        return factor * self.scale_gradient


@dataclass
class LinearElectronic:
    bias: np.ndarray
    field_matrix: np.ndarray
    geometry_matrix: np.ndarray
    provider_id: ClassVar[str] = "maple.route2.test.linear-electronic.v1"
    model_profile_id: ClassVar[str] = "mace-polar-route2-source-field-contract-v1"
    coupling_id: ClassVar[str] = EXACT_GTO_COUPLING_CANDIDATE_ID
    provenance_sha256: ClassVar[str] = "e" * 64
    source_space: ClassVar[SourceSpace] = ATOMIC_L1_SOURCE_SPACE
    field_space: ClassVar[FieldDualSpace] = ATOMIC_L1_FIELD_DUAL_SPACE
    derivative_scale: ClassVar[float] = 1.0

    def configuration_sha256(self):
        return _configuration_sha256(
            self.bias, self.field_matrix, self.geometry_matrix, self.derivative_scale
        )

    def evaluate_source(self, geometry, field):
        return (
            self.bias
            + self.field_matrix @ field.reshape(-1)
            + self.geometry_matrix @ geometry
        ).reshape(field.shape)

    def field_jvp(self, geometry, field, direction):
        return (
            self.derivative_scale * self.field_matrix @ direction.reshape(-1)
        ).reshape(field.shape)

    def field_vjp(self, geometry, field, cotangent):
        return (
            self.derivative_scale * self.field_matrix.T @ cotangent.reshape(-1)
        ).reshape(field.shape)

    def coordinate_vjp(self, geometry, field, cotangent):
        return self.geometry_matrix.T @ cotangent.reshape(-1)


@dataclass
class QuadraticVacuum:
    hessian: np.ndarray
    linear: np.ndarray
    provider_id: ClassVar[str] = "maple.route2.test.quadratic-vacuum.v1"
    model_profile_id: ClassVar[str] = "mace-polar-route2-source-field-contract-v1"
    provenance_sha256: ClassVar[str] = "a" * 64

    def configuration_sha256(self):
        return _configuration_sha256(self.hessian, self.linear)

    def evaluate_energy(self, geometry):
        return 0.5 * geometry @ self.hessian @ geometry + self.linear @ geometry

    def coordinate_gradient(self, geometry):
        return self.hessian @ geometry + self.linear


@dataclass
class NonlinearContinuum:
    matrix: np.ndarray
    geometry_matrix: np.ndarray
    provider_id: ClassVar[str] = "test.nonlinear-continuum.v1"
    continuum_profile_id: ClassVar[str] = "fixed-topology-linear-reciprocal-cpcm-v1"
    cavity_profile_id: ClassVar[str] = "fixed-topology-amplitude-swig-v1"
    coupling_id: ClassVar[str] = EXACT_GTO_COUPLING_CANDIDATE_ID
    provenance_sha256: ClassVar[str] = "b" * 64
    source_space: ClassVar[SourceSpace] = ATOMIC_L1_SOURCE_SPACE
    field_space: ClassVar[FieldDualSpace] = ATOMIC_L1_FIELD_DUAL_SPACE

    def configuration_sha256(self):
        return _configuration_sha256(self.matrix, self.geometry_matrix)

    def argument(self, geometry, source):
        return self.matrix @ source.reshape(-1) + self.geometry_matrix @ geometry

    def evaluate_field(self, geometry, source):
        return np.tanh(self.argument(geometry, source)).reshape(source.shape)

    def source_jvp(self, geometry, source, direction):
        derivative = 1.0 - np.tanh(self.argument(geometry, source)) ** 2
        return (derivative * (self.matrix @ direction.reshape(-1))).reshape(
            source.shape
        )

    def source_vjp(self, geometry, source, cotangent):
        derivative = 1.0 - np.tanh(self.argument(geometry, source)) ** 2
        return (self.matrix.T @ (derivative * cotangent.reshape(-1))).reshape(
            source.shape
        )

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
    return equation, OperationalElectrostaticScalar(
        equation,
        vacuum,
        profile_id=OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1,
    )


def _root_context(geometry):
    geometry_hash = hashlib.sha256(
        np.asarray(geometry, dtype=float).tobytes()
    ).hexdigest()
    return f"linear-oracle/fixedtopology-electrostatic-v1/{geometry_hash}"


def test_implicit_adjoint_matches_resolved_scalar_finite_difference():
    equation, scalar = _system()
    geometry = np.array([0.3, -0.2, 0.15])
    primal_options = FixedPointOptions(tolerance=1e-13, max_iterations=80)
    state = solve_fixed_point(
        equation,
        geometry,
        scalar_id=scalar.scalar_id,
        profile_id=scalar.profile_id,
        scalar_binding=scalar,
        root_context_id=_root_context(geometry),
        options=primal_options,
    )
    warm_state = solve_fixed_point(
        equation,
        geometry,
        scalar_id=scalar.scalar_id,
        profile_id=scalar.profile_id,
        scalar_binding=scalar,
        root_context_id=_root_context(geometry),
        initial_y=np.asarray(state.y) + 1.0e-5,
        options=primal_options,
    )
    assert roots_numerically_equivalent(state, warm_state)
    assert scalar.evaluate(geometry, state.y).total_energy == pytest.approx(
        scalar.evaluate(geometry, warm_state.y).total_energy, abs=1e-12
    )
    gradient = scalar.implicit_gradient(
        geometry,
        state,
        adjoint_options=AdjointOptions(
            relative_tolerance=1e-13, absolute_tolerance=1e-14
        ),
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
            scalar_id=scalar.scalar_id,
            profile_id=scalar.profile_id,
            scalar_binding=scalar,
            root_context_id=_root_context(plus_geometry),
            options=primal_options,
        )
        minus_state = solve_fixed_point(
            equation,
            minus_geometry,
            scalar_id=scalar.scalar_id,
            profile_id=scalar.profile_id,
            scalar_binding=scalar,
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


def test_energy_only_entry_is_same_scalar_and_does_not_call_derivatives(monkeypatch):
    def derivative_must_not_run(*_args, **_kwargs):
        raise AssertionError("energy-only scalar evaluated a derivative")

    monkeypatch.setattr(
        LinearReciprocalContinuum, "source_vjp", derivative_must_not_run
    )
    monkeypatch.setattr(
        LinearReciprocalContinuum, "coordinate_vjp", derivative_must_not_run
    )
    monkeypatch.setattr(QuadraticVacuum, "coordinate_gradient", derivative_must_not_run)
    equation, scalar = _system()
    geometry = np.array([0.3, -0.2, 0.15])
    y = np.linspace(-0.02, 0.03, equation.reduced_dimension)
    source = equation.coordinates.expand(y)
    field = equation.continuum.evaluate_field(geometry, source)
    expected = scalar.vacuum.evaluate_energy(geometry) + 0.5 * scalar.metric.pair(
        source, field
    )
    assert scalar.evaluate_energy(geometry, y) == pytest.approx(expected, abs=1e-15)


def test_implicit_gradient_rejects_wrong_geometry_provider_and_tampered_state():
    equation, scalar = _system()
    geometry = np.array([0.3, -0.2, 0.15])
    state = solve_fixed_point(
        equation,
        geometry,
        scalar_id=scalar.scalar_id,
        profile_id=scalar.profile_id,
        scalar_binding=scalar,
        root_context_id="binding-negative-tests",
        options=FixedPointOptions(tolerance=1e-13),
    )
    with pytest.raises(ValueError, match="different geometry"):
        scalar.implicit_gradient(geometry + np.array([1e-3, 0.0, 0.0]), state)

    wrong_continuum = LinearReciprocalContinuum(
        equation.continuum.atom_count,
        equation.continuum.base_scale,
        equation.continuum.scale_gradient,
    )
    wrong_continuum.provider_id = "test.different-continuum.v1"
    wrong_equation = ReducedStateEquation(
        equation.coordinates, equation.electronic, wrong_continuum
    )
    wrong_scalar = OperationalElectrostaticScalar(wrong_equation, scalar.vacuum)
    with pytest.raises(ValueError, match="different provider/equation"):
        wrong_scalar.implicit_gradient(geometry, state)

    tampered_source = list(state.source)
    tampered_source[0] = tuple(
        value + (1e-3 if index == 0 else 0.0)
        for index, value in enumerate(tampered_source[0])
    )
    with pytest.raises(ValueError, match="root_hash"):
        replace(state, source=tuple(tampered_source))
    with pytest.raises(ValueError, match="root_hash"):
        replace(state, root_context_id="spoof")

    replacement_vacuum = QuadraticVacuum(
        scalar.vacuum.hessian.copy(), scalar.vacuum.linear + 0.01
    )
    replacement_scalar = OperationalElectrostaticScalar(equation, replacement_vacuum)
    with pytest.raises(ValueError, match="different scalar/vacuum"):
        replacement_scalar.implicit_gradient(geometry, state)


def test_provider_configuration_drift_is_rejected_at_every_entry():
    equation, scalar = _system()
    geometry = np.array([0.1, -0.2, 0.05])
    y = np.zeros(equation.reduced_dimension)
    forward_before = equation.electronic.evaluate_source(
        geometry,
        equation.continuum.evaluate_field(geometry, equation.coordinates.expand(y)),
    )
    equation.electronic.derivative_scale = 1.01
    forward_after = equation.electronic.evaluate_source(
        geometry,
        equation.continuum.evaluate_field(geometry, equation.coordinates.expand(y)),
    )
    np.testing.assert_array_equal(forward_before, forward_after)
    with pytest.raises(ValueError, match="configuration drifted"):
        equation.evaluate(geometry, y)

    equation, scalar = _system()
    scalar.vacuum.hessian[0, 0] += 1e-4
    with pytest.raises(ValueError, match="configuration drifted"):
        scalar.evaluate(geometry, y)

    equation, scalar = _system()
    equation.electronic.field_vjp = lambda geometry, field, cotangent: cotangent
    with pytest.raises(TypeError, match="instance-level callable rebinding"):
        scalar.evaluate(geometry, y)


def test_scalar_rejects_missing_or_mismatched_vacuum_model_profile_id():
    equation, scalar = _system()
    missing = QuadraticVacuum(scalar.vacuum.hessian, scalar.vacuum.linear)
    missing.model_profile_id = None
    with pytest.raises(ValueError, match="model-profile ID"):
        OperationalElectrostaticScalar(equation, missing)
    wrong = QuadraticVacuum(scalar.vacuum.hessian, scalar.vacuum.linear)
    wrong.model_profile_id = "wrong-model-profile"
    with pytest.raises(ValueError, match="model-profile ID"):
        OperationalElectrostaticScalar(equation, wrong)


def test_scalar_rejects_wrong_charge_component_and_cavity_identity():
    equation, scalar = _system()
    bad_source = SourceSpace(
        scalar_id=ATOMIC_L1_SOURCE_SPACE.scalar_id,
        representation=ATOMIC_L1_SOURCE_SPACE.representation,
        components=ATOMIC_L1_SOURCE_SPACE.components,
        units=ATOMIC_L1_SOURCE_SPACE.units,
        charge_component=1,
    )
    bad_field = FieldDualSpace(
        scalar_id=ATOMIC_L1_FIELD_DUAL_SPACE.scalar_id,
        representation=ATOMIC_L1_FIELD_DUAL_SPACE.representation,
        components=ATOMIC_L1_FIELD_DUAL_SPACE.components,
        units=ATOMIC_L1_FIELD_DUAL_SPACE.units,
        source_space=bad_source,
        pairing_metric=ATOMIC_L1_PAIRING,
    )
    bad_coordinates = AffineChargeCoordinates(2, 0.5, source_space=bad_source)
    equation.electronic.source_space = bad_source
    equation.electronic.field_space = bad_field
    equation.continuum.source_space = bad_source
    equation.continuum.field_space = bad_field
    bad_equation = ReducedStateEquation(
        bad_coordinates, equation.electronic, equation.continuum
    )
    with pytest.raises(ValueError, match="exact authoritative"):
        OperationalElectrostaticScalar(bad_equation, scalar.vacuum)

    equation, scalar = _system()
    equation.continuum.cavity_profile_id = "wrong-cavity-profile"
    mismatched = ReducedStateEquation(
        equation.coordinates, equation.electronic, equation.continuum
    )
    with pytest.raises(ValueError, match="cavity-profile ID"):
        OperationalElectrostaticScalar(mismatched, scalar.vacuum)


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


def test_half_coupling_rejects_same_shape_wrong_order_units_or_gauge():
    rng = np.random.default_rng(37)
    continuum = NonlinearContinuum(
        rng.normal(scale=0.02, size=(8, 8)),
        rng.normal(scale=0.02, size=(8, 3)),
    )
    bad_fields = (
        (
            ATOMIC_L1_PAIRING.field_components,
            ATOMIC_L1_PAIRING.field_units,
            "wrong-gauge",
        ),
        (
            (
                "potential_gradient_x",
                "potential",
                "potential_gradient_y",
                "potential_gradient_z",
            ),
            ATOMIC_L1_PAIRING.field_units,
            ATOMIC_L1_PAIRING.gauge,
        ),
        (
            ATOMIC_L1_PAIRING.field_components,
            ("wrong",) + ATOMIC_L1_PAIRING.field_units[1:],
            ATOMIC_L1_PAIRING.gauge,
        ),
    )
    for field_components, field_units, gauge in bad_fields:
        bad_metric = PairingMetric(
            scalar_id=ATOMIC_L1_PAIRING.scalar_id,
            source_components=ATOMIC_L1_PAIRING.source_components,
            field_components=field_components,
            source_units=ATOMIC_L1_PAIRING.source_units,
            field_units=field_units,
            field_to_source_indices=ATOMIC_L1_PAIRING.field_to_source_indices,
            gauge=gauge,
            field_convention=ATOMIC_L1_PAIRING.field_convention,
        )
        with pytest.raises(ValueError, match="pairing identity"):
            nonlinear_half_coupling(
                continuum, np.zeros(3), np.zeros((2, 4)), metric=bad_metric
            )

    reordered_source = SourceSpace(
        scalar_id=ATOMIC_L1_SOURCE_SPACE.scalar_id,
        representation=ATOMIC_L1_SOURCE_SPACE.representation,
        components=("net_monopole", "real_l1_m1", "real_l1_m0", "real_l1_m_minus1"),
        units=ATOMIC_L1_SOURCE_SPACE.units,
    )
    continuum.source_space = reordered_source
    with pytest.raises(ValueError, match="source-space identity"):
        nonlinear_half_coupling(
            continuum, np.zeros(3), np.zeros((2, 4)), metric=ATOMIC_L1_PAIRING
        )
