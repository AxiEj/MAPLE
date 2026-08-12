from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
import hashlib
from typing import ClassVar

import numpy as np
import pytest

from maple.solvation.coupling.fixed_point import (
    FixedPointOptions,
    compute_root_hash,
    roots_numerically_equivalent,
    solve_fixed_point,
)
from maple.solvation.coupling.metrics import ATOMIC_L1_PAIRING
from maple.solvation.api.scalar_registry import OPERATIONAL_CPCM_ELECTROSTATIC_V1
from maple.solvation.api.profiles import (
    EXACT_GTO_COUPLING_CANDIDATE_ID,
    OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1,
)
from maple.solvation.coupling.spaces import (
    ATOMIC_L1_FIELD_DUAL_SPACE,
    ATOMIC_L1_SOURCE_SPACE,
    AffineChargeCoordinates,
    FieldDualSpace,
    SourceSpace,
)
from maple.solvation.coupling.state_equation import (
    ReducedStateEquation,
    geometry_sha256,
)

SCALAR_ID = OPERATIONAL_CPCM_ELECTROSTATIC_V1
PROFILE_ID = OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1
SCALAR_SHA = "9" * 64


class _ScalarBinding:
    scalar_id = SCALAR_ID
    profile_id = PROFILE_ID

    def fingerprint_sha256(self):
        return SCALAR_SHA


SCALAR_BINDING = _ScalarBinding()


def _configuration_sha256(*values):
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _q(atom_count: int) -> np.ndarray:
    return np.kron(np.eye(atom_count), ATOMIC_L1_PAIRING.block)


_NESTED_PROVIDER_SCALE = np.asarray([1.0])


def _nested_provider_scale() -> float:
    return float(_NESTED_PROVIDER_SCALE[0])


@dataclass
class DenseLinearContinuum:
    base_scale: float
    scale_gradient: np.ndarray
    atom_count: int
    provider_id: ClassVar[str] = "test.dense-linear-continuum-implementation.v1"
    continuum_profile_id: ClassVar[str] = "fixed-topology-linear-reciprocal-cpcm-v1"
    cavity_profile_id: ClassVar[str] = "fixed-topology-amplitude-swig-v1"
    coupling_id: ClassVar[str] = EXACT_GTO_COUPLING_CANDIDATE_ID
    provenance_sha256: ClassVar[str] = "c" * 64
    source_space: ClassVar[SourceSpace] = ATOMIC_L1_SOURCE_SPACE
    field_space: ClassVar[FieldDualSpace] = ATOMIC_L1_FIELD_DUAL_SPACE

    def configuration_sha256(self):
        return _configuration_sha256(
            self.base_scale, self.scale_gradient, self.atom_count
        )

    def _scale(self, geometry):
        return self.base_scale + float(np.vdot(self.scale_gradient, geometry))

    def evaluate_field(self, geometry, source):
        return (
            self._scale(geometry) * _q(self.atom_count).T @ source.reshape(-1)
        ).reshape(self.atom_count, 4)

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
    provider_id: ClassVar[str] = "test.dense-linear-electronic-implementation.v1"
    model_profile_id: ClassVar[str] = "mace-polar-route2-source-field-contract-v1"
    coupling_id: ClassVar[str] = EXACT_GTO_COUPLING_CANDIDATE_ID
    provenance_sha256: ClassVar[str] = "e" * 64
    source_space: ClassVar[SourceSpace] = ATOMIC_L1_SOURCE_SPACE
    field_space: ClassVar[FieldDualSpace] = ATOMIC_L1_FIELD_DUAL_SPACE

    def configuration_sha256(self):
        return _configuration_sha256(self.bias, self.field_matrix, self.geometry_matrix)

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
        return (
            self.bias
            + self.field_matrix @ field.reshape(-1)
            + self.geometry_matrix @ geometry
        )

    def evaluate_source(self, geometry, field):
        return np.tanh(self._argument(geometry, field)).reshape(field.shape)

    def field_jvp(self, geometry, field, field_direction):
        derivative = 1.0 - np.tanh(self._argument(geometry, field)) ** 2
        return (derivative * (self.field_matrix @ field_direction.reshape(-1))).reshape(
            field.shape
        )

    def field_vjp(self, geometry, field, source_cotangent):
        derivative = 1.0 - np.tanh(self._argument(geometry, field)) ** 2
        return (
            self.field_matrix.T @ (derivative * source_cotangent.reshape(-1))
        ).reshape(field.shape)

    def coordinate_vjp(self, geometry, field, source_cotangent):
        derivative = 1.0 - np.tanh(self._argument(geometry, field)) ** 2
        return self.geometry_matrix.T @ (derivative * source_cotangent.reshape(-1))


class NestedGlobalElectronic(DenseLinearElectronic):
    def field_vjp(self, geometry, field, source_cotangent):
        return _nested_provider_scale() * super().field_vjp(
            geometry, field, source_cotangent
        )


def _linear_equation(seed=5):
    rng = np.random.default_rng(seed)
    coordinates = AffineChargeCoordinates(
        2, total_charge=-1.0, monopole_scale=0.8, dipole_scale=1.2
    )
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
            DenseNonlinearElectronic(
                linear.bias, linear.field_matrix, linear.geometry_matrix
            ),
            equation.continuum,
        )
    rng = np.random.default_rng(11)
    geometry = rng.normal(scale=0.2, size=3)
    y = rng.normal(scale=0.15, size=equation.reduced_dimension)
    dy = rng.normal(size=equation.reduced_dimension)
    cotangent = rng.normal(size=equation.reduced_dimension)
    step = 1.0e-6
    fd = (
        equation.residual(geometry, y + step * dy)
        - equation.residual(geometry, y - step * dy)
    ) / (2 * step)
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
        [
            equation.jvp(geometry, zero, basis)
            for basis in np.eye(equation.reduced_dimension)
        ]
    )
    exact = np.linalg.solve(jacobian, -offset)
    options = FixedPointOptions(method="anderson", tolerance=1e-13, max_iterations=80)
    context = "dense-linear-provider/test-profile/geometry-0.1,0.2,-0.3"
    cold = solve_fixed_point(
        equation,
        geometry,
        scalar_id=SCALAR_ID,
        profile_id=PROFILE_ID,
        scalar_binding=SCALAR_BINDING,
        root_context_id=context,
        options=options,
    )
    warm = solve_fixed_point(
        equation,
        geometry,
        scalar_id=SCALAR_ID,
        profile_id=PROFILE_ID,
        scalar_binding=SCALAR_BINDING,
        root_context_id=context,
        initial_y=exact + 1e-4,
        options=options,
    )
    np.testing.assert_allclose(cold.y, exact, atol=1e-12)
    np.testing.assert_allclose(warm.y, exact, atol=1e-12)
    assert cold.actual_unmixed_residual_norm <= options.tolerance
    assert warm.actual_unmixed_residual_norm <= options.tolerance
    np.testing.assert_allclose(cold.y, warm.y, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(cold.source, warm.source, atol=1e-12, rtol=0.0)
    assert (
        cold.root_hash != warm.root_hash
    )  # exact content identity is not tolerance-based
    different_context = solve_fixed_point(
        equation,
        geometry,
        scalar_id=SCALAR_ID,
        profile_id=PROFILE_ID,
        scalar_binding=SCALAR_BINDING,
        root_context_id=context + "/different-provider",
        options=options,
    )
    assert different_context.root_hash != cold.root_hash
    assert cold.geometry_sha256 == geometry_sha256(geometry)
    assert cold.equation_sha256 == equation.fingerprint_sha256()
    assert cold.initialization == "cold" and warm.initialization == "warm"
    assert cold.field_array().shape == cold.source_array().shape
    assert abs(np.sum(cold.source_array()[:, 0]) + 1.0) <= 1e-12
    with pytest.raises(FrozenInstanceError):
        cold.converged = False
    with pytest.raises(ValueError, match="root_hash"):
        replace(cold, root_context_id="spoofed-context")
    with pytest.raises(ValueError, match="root_hash"):
        replace(cold, converged=False)
    with pytest.raises(ValueError, match="root_hash"):
        replace(cold, iterations=cold.iterations[:-1])
    assert not roots_numerically_equivalent(
        cold,
        replace(
            cold,
            converged=False,
            root_hash=compute_root_hash(
                state_equation_id=cold.state_equation_id,
                geometry_digest=cold.geometry_sha256,
                equation_digest=cold.equation_sha256,
                scalar_digest=cold.scalar_sha256,
                scalar_id=cold.scalar_id,
                profile_id=cold.profile_id,
                primal_tolerance=cold.primal_tolerance,
                root_context_id=cold.root_context_id,
                y=cold.y,
                source=cold.source,
                field=cold.field,
                residual=cold.actual_unmixed_residual,
                initialization=cold.initialization,
                converged=False,
                residual_norm=cold.actual_unmixed_residual_norm,
                iterations=cold.iterations,
            ),
        ),
    )


def test_state_kernel_accepts_non_l1_space_but_release_solver_requires_profile():
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
    from maple.solvation.coupling.metrics import PairingMetric

    pairing = PairingMetric(
        scalar_id="test.future-pairing.v1",
        source_components=source_space.components,
        field_components=("f1", "potential", "f2"),
        source_units=source_space.units,
        field_units=("du1", "eV/e", "du2"),
        field_to_source_indices=(0, 1, 2),
        gauge="test-zero",
        field_convention="positive-energy-dual",
    )
    field_space = FieldDualSpace(
        scalar_id="test.future-field.v1",
        representation="future energy-dual field",
        components=pairing.field_components,
        units=pairing.field_units,
        source_space=source_space,
        pairing_metric=pairing,
    )

    class Continuum:
        provider_id = "test.future-continuum-implementation.v1"
        continuum_profile_id = "fixed-topology-linear-reciprocal-cpcm-v1"
        cavity_profile_id = "fixed-topology-amplitude-swig-v1"
        coupling_id = EXACT_GTO_COUPLING_CANDIDATE_ID
        provenance_sha256 = "1" * 64

        def __init__(self):
            self.source_space = source_space
            self.field_space = field_space

        def configuration_sha256(self):
            return "1" * 64

        def evaluate_field(self, geometry, source):
            return 0.2 * source

        def source_jvp(self, geometry, source, direction):
            return 0.2 * direction

        def source_vjp(self, geometry, source, cotangent):
            return 0.2 * cotangent

        def coordinate_vjp(self, geometry, source, cotangent):
            return np.zeros_like(geometry)

    class Electronic:
        provider_id = "test.future-electronic-implementation.v1"
        model_profile_id = "mace-polar-route2-source-field-contract-v1"
        coupling_id = EXACT_GTO_COUPLING_CANDIDATE_ID
        provenance_sha256 = "2" * 64

        def __init__(self):
            self.source_space = source_space
            self.field_space = field_space

        def configuration_sha256(self):
            return "2" * 64

        def evaluate_source(self, geometry, field):
            return 0.1 * field + 0.05

        def field_jvp(self, geometry, field, direction):
            return 0.1 * direction

        def field_vjp(self, geometry, field, cotangent):
            return 0.1 * cotangent

        def coordinate_vjp(self, geometry, field, cotangent):
            return np.zeros_like(geometry)

    equation = ReducedStateEquation(coordinates, Electronic(), Continuum())
    evaluated = equation.evaluate(np.zeros(2), np.zeros(equation.reduced_dimension))
    assert np.asarray(evaluated.source).shape == (3, 3)
    assert np.asarray(evaluated.field).shape == (3, 3)
    assert (
        abs(
            source_space.total_charge(
                evaluated.source, atom_count=coordinates.atom_count
            )
            - 2.0
        )
        <= 1e-12
    )
    with pytest.raises(ValueError, match="source identity"):
        solve_fixed_point(
            equation,
            np.zeros(2),
            scalar_id=SCALAR_ID,
            profile_id=PROFILE_ID,
            scalar_binding=SCALAR_BINDING,
            root_context_id="future-multipole/test-provider/geometry-zero",
            options=FixedPointOptions(tolerance=1e-13),
        )


@pytest.mark.parametrize(
    "bad_source",
    [
        SourceSpace(
            scalar_id=ATOMIC_L1_SOURCE_SPACE.scalar_id,
            representation=ATOMIC_L1_SOURCE_SPACE.representation,
            components=("net_monopole", "real_l1_m1", "real_l1_m0", "real_l1_m_minus1"),
            units=ATOMIC_L1_SOURCE_SPACE.units,
        ),
        SourceSpace(
            scalar_id=ATOMIC_L1_SOURCE_SPACE.scalar_id,
            representation=ATOMIC_L1_SOURCE_SPACE.representation,
            components=ATOMIC_L1_SOURCE_SPACE.components,
            units=("e", "wrong", "e*angstrom", "e*angstrom"),
        ),
    ],
)
def test_equation_rejects_reordered_or_wrong_unit_source_identity(bad_source):
    equation = _linear_equation()
    equation.electronic.source_space = bad_source
    with pytest.raises(ValueError, match="source_space"):
        ReducedStateEquation(
            equation.coordinates, equation.electronic, equation.continuum
        )


def test_geometry_hash_binds_numeric_shape_and_ase_identity():
    numeric = np.arange(6.0).reshape(2, 3)
    assert geometry_sha256(numeric) != geometry_sha256(numeric.reshape(6))

    class AtomsLike:
        def get_atomic_numbers(self):
            return np.array([1, 8])

        def get_positions(self):
            return numeric

        def get_cell(self):
            return np.eye(3)

        def get_pbc(self):
            return np.array([False, False, False])

    digest = geometry_sha256(AtomsLike())
    assert len(digest) == 64


def test_solver_rejects_unregistered_or_mismatched_profile_ids():
    equation = _linear_equation()
    geometry = np.zeros(3)
    common = dict(
        scalar_binding=SCALAR_BINDING,
        root_context_id="registry-negative",
        options=FixedPointOptions(max_iterations=2),
    )
    with pytest.raises(KeyError, match="Unregistered Route-2 profile"):
        solve_fixed_point(
            equation,
            geometry,
            scalar_id=SCALAR_ID,
            profile_id=SCALAR_ID,  # scalar IDs are never accepted as profile IDs
            **common,
        )


def test_equation_requires_compatibility_profile_ids_and_solver_rejects_mismatch():
    equation = _linear_equation()
    equation.electronic.model_profile_id = None
    with pytest.raises(ValueError, match="model_profile_id"):
        ReducedStateEquation(
            equation.coordinates, equation.electronic, equation.continuum
        )

    equation = _linear_equation()
    equation.continuum.cavity_profile_id = None
    with pytest.raises(ValueError, match="cavity_profile_id"):
        ReducedStateEquation(
            equation.coordinates, equation.electronic, equation.continuum
        )

    equation = _linear_equation()
    equation.continuum.continuum_profile_id = "wrong-continuum-profile"
    mismatched = ReducedStateEquation(
        equation.coordinates, equation.electronic, equation.continuum
    )
    with pytest.raises(ValueError, match="Continuum profile identity"):
        solve_fixed_point(
            mismatched,
            np.zeros(3),
            scalar_id=SCALAR_ID,
            profile_id=PROFILE_ID,
            scalar_binding=SCALAR_BINDING,
            root_context_id="compatibility-mismatch",
        )


def test_equation_rejects_instance_level_derivative_rebinding():
    equation = _linear_equation()
    geometry = np.zeros(3)
    y = np.zeros(equation.reduced_dimension)
    equation.evaluate(geometry, y)
    equation.electronic.field_vjp = lambda geometry, field, cotangent: cotangent
    with pytest.raises(TypeError, match="instance-level callable rebinding"):
        equation.evaluate(geometry, y)


def test_equation_fingerprint_binds_referenced_global_helper(monkeypatch):
    equation = _linear_equation()
    before = equation.fingerprint_sha256()
    monkeypatch.setitem(
        DenseLinearContinuum.coordinate_vjp.__globals__,
        "_q",
        lambda atom_count: 2.0 * _q(atom_count),
    )
    with pytest.raises(ValueError, match="configuration drifted"):
        equation.fingerprint_sha256()
    assert before == equation._construction_fingerprint
    with pytest.raises(ValueError, match="does not bind the requested scalar"):
        solve_fixed_point(
            equation,
            np.zeros(3),
            scalar_id="route2-operational-cpcm-fixedtopology-smdcds-v1",
            profile_id=PROFILE_ID,
            scalar_binding=SCALAR_BINDING,
            root_context_id="scalar-profile-mismatch",
        )


def test_equation_fingerprint_recursively_binds_helper_global_arrays(monkeypatch):
    base = _linear_equation()
    electronic = base.electronic
    equation = ReducedStateEquation(
        base.coordinates,
        NestedGlobalElectronic(
            electronic.bias,
            electronic.field_matrix,
            electronic.geometry_matrix,
        ),
        base.continuum,
    )
    before = equation.fingerprint_sha256()
    monkeypatch.setitem(
        NestedGlobalElectronic.field_vjp.__globals__,
        "_NESTED_PROVIDER_SCALE",
        np.asarray([2.0]),
    )
    with pytest.raises(ValueError, match="configuration drifted"):
        equation.fingerprint_sha256()
    assert equation._construction_fingerprint == before
