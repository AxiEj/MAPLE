from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import ClassVar

import numpy as np
import pytest

from maple.solvation.api.profiles import (
    DIAGNOSTIC_RADIAL_GTO_CPCM_ELECTROSTATIC_PROFILE_V1,
    MACE_POLAR_RADIAL_GTO_COUPLING_ID,
    OPERATIONAL_CPCM_RADIAL_GTO_ELECTROSTATIC_PROFILE_V1,
)
from maple.solvation.continuum import (
    ConjugateRadialGTOFixedTopologyCPCMBackend,
)
from maple.solvation.coupling.adjoint import AdjointOptions
from maple.solvation.coupling.energy import OperationalElectrostaticScalar
from maple.solvation.coupling.fixed_point import (
    FixedPointOptions,
    roots_numerically_equivalent,
    solve_fixed_point,
)
from maple.solvation.coupling.spaces import (
    LinearChargeCoordinates,
    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)
from maple.solvation.coupling.state_equation import ReducedStateEquation

_UNIT_SPHERE = np.asarray(
    [
        [1.0, 0.0, 0.0, 1.0 / 6.0],
        [-1.0, 0.0, 0.0, 1.0 / 6.0],
        [0.0, 1.0, 0.0, 1.0 / 6.0],
        [0.0, -1.0, 0.0, 1.0 / 6.0],
        [0.0, 0.0, 1.0, 1.0 / 6.0],
        [0.0, 0.0, -1.0, 1.0 / 6.0],
    ],
    dtype=float,
)
_MODEL_PROFILE_ID = "mace-polar-route2-source-field-contract-v1"


def _content_sha256(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass
class _RadialElectronicOracle:
    bias: np.ndarray
    field_matrix: np.ndarray
    geometry_matrix: np.ndarray
    provider_id: ClassVar[str] = "maple.route2.test.radial-electronic-oracle.v1"
    model_profile_id: ClassVar[str] = _MODEL_PROFILE_ID
    coupling_id: ClassVar[str] = MACE_POLAR_RADIAL_GTO_COUPLING_ID
    provenance_sha256: ClassVar[str] = "e" * 64
    source_space = MACE_POLAR_RADIAL_GTO_SOURCE_SPACE
    field_space = MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE

    def configuration_sha256(self) -> str:
        return _content_sha256(self.bias, self.field_matrix, self.geometry_matrix)

    def evaluate_source(self, geometry, field):
        flat = (
            self.bias
            + self.field_matrix @ np.asarray(field).reshape(-1)
            + self.geometry_matrix @ np.asarray(geometry).reshape(-1)
        )
        return flat.reshape(np.asarray(field).shape)

    def field_jvp(self, geometry, field, field_direction):
        return (self.field_matrix @ np.asarray(field_direction).reshape(-1)).reshape(
            np.asarray(field).shape
        )

    def field_vjp(self, geometry, field, source_cotangent):
        return (self.field_matrix.T @ np.asarray(source_cotangent).reshape(-1)).reshape(
            np.asarray(field).shape
        )

    def coordinate_vjp(self, geometry, field, source_cotangent):
        return self.geometry_matrix.T @ np.asarray(source_cotangent).reshape(-1)


@dataclass
class _QuadraticVacuumOracle:
    diagonal: np.ndarray
    linear: np.ndarray
    provider_id: ClassVar[str] = "maple.route2.test.radial-vacuum-oracle.v1"
    model_profile_id: ClassVar[str] = _MODEL_PROFILE_ID
    provenance_sha256: ClassVar[str] = "a" * 64

    def configuration_sha256(self) -> str:
        return _content_sha256(self.diagonal, self.linear)

    def evaluate_energy(self, geometry) -> float:
        positions = np.asarray(geometry).reshape(-1)
        return float(
            0.5 * np.vdot(self.diagonal * positions, positions)
            + self.linear @ positions
        )

    def coordinate_gradient(self, geometry) -> np.ndarray:
        positions = np.asarray(geometry).reshape(-1)
        return self.diagonal * positions + self.linear


def _system():
    positions = np.asarray([[0.0, 0.0, 0.0], [2.5, 0.2, -0.1]])
    coordinates = LinearChargeCoordinates(
        atom_count=2,
        total_charge=0.0,
        source_space=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
        component_scales=(1.0,) * 8,
    )
    continuum = ConjugateRadialGTOFixedTopologyCPCMBackend(
        ("H", "H"),
        np.asarray([1.1, 1.1]),
        dielectric=78.39,
        unit_sphere=_UNIT_SPHERE,
        switching_constant=4.84566077868,
        runtime_version="radial-operational-scalar-oracle-v1",
    )
    rng = np.random.default_rng(20260813)
    size = coordinates.source_dimension
    learned = np.asarray([0, 2, 3, 4, 8, 10, 11, 12])
    bias = np.zeros(size)
    bias[learned] = rng.normal(scale=0.03, size=learned.size)
    field_matrix = np.zeros((size, size))
    field_matrix[learned] = rng.normal(scale=2.0e-4, size=(learned.size, size))
    geometry_matrix = np.zeros((size, positions.size))
    geometry_matrix[learned] = rng.normal(
        scale=3.0e-3, size=(learned.size, positions.size)
    )
    electronic = _RadialElectronicOracle(bias, field_matrix, geometry_matrix)
    vacuum = _QuadraticVacuumOracle(
        np.linspace(0.05, 0.10, positions.size),
        np.linspace(-0.02, 0.03, positions.size),
    )
    equation = ReducedStateEquation(coordinates, electronic, continuum)
    scalar = OperationalElectrostaticScalar(
        equation,
        vacuum,
        profile_id=DIAGNOSTIC_RADIAL_GTO_CPCM_ELECTROSTATIC_PROFILE_V1,
    )
    return positions, equation, scalar


def _solve(equation, scalar, geometry, *, initial_y=None):
    geometry_digest = hashlib.sha256(np.asarray(geometry).tobytes()).hexdigest()
    return solve_fixed_point(
        equation,
        geometry,
        scalar_id=scalar.scalar_id,
        profile_id=scalar.profile_id,
        scalar_binding=scalar,
        root_context_id=f"radial-operational-scalar/{geometry_digest}",
        initial_y=initial_y,
        options=FixedPointOptions(tolerance=1.0e-13, max_iterations=100),
    )


def test_radial_root_scalar_and_adjoint_are_one_resolved_pes():
    geometry, equation, scalar = _system()
    cold = _solve(equation, scalar, geometry)
    warm = _solve(
        equation,
        scalar,
        geometry,
        initial_y=np.asarray(cold.y) + 1.0e-6,
    )
    assert roots_numerically_equivalent(cold, warm)
    assert cold.actual_unmixed_residual_norm <= cold.primal_tolerance
    evaluated = scalar.evaluate(geometry, cold.y)
    assert evaluated.total_energy == pytest.approx(
        evaluated.vacuum_energy + evaluated.continuum_energy, abs=1.0e-14
    )
    assert MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.total_charge(
        cold.source, atom_count=2
    ) == pytest.approx(0.0, abs=1.0e-12)

    result = scalar.implicit_gradient(
        geometry,
        cold,
        adjoint_options=AdjointOptions(
            relative_tolerance=1.0e-13,
            absolute_tolerance=1.0e-14,
        ),
    )
    direction = np.asarray([[0.20, -0.13, 0.07], [-0.20, 0.13, -0.07]], dtype=float)
    direction /= np.linalg.norm(direction)
    step = 1.0e-5
    plus_geometry = geometry + step * direction
    minus_geometry = geometry - step * direction
    plus_state = _solve(equation, scalar, plus_geometry, initial_y=np.asarray(cold.y))
    minus_state = _solve(equation, scalar, minus_geometry, initial_y=np.asarray(cold.y))
    finite_difference = (
        scalar.evaluate(plus_geometry, plus_state.y).total_energy
        - scalar.evaluate(minus_geometry, minus_state.y).total_energy
    ) / (2.0 * step)
    analytic = np.vdot(result.total_coordinate_gradient, direction.reshape(-1))
    assert analytic == pytest.approx(finite_difference, rel=1.0e-8, abs=2.0e-8)
    assert result.adjoint.converged
    assert result.adjoint.true_residual_norm <= result.adjoint.acceptance_tolerance


def test_operational_water_profile_rejects_unbound_injected_grid_backend():
    _, equation, scalar = _system()
    with pytest.raises(ValueError, match="physical-configuration contract"):
        OperationalElectrostaticScalar(
            equation,
            scalar.vacuum,
            profile_id=OPERATIONAL_CPCM_RADIAL_GTO_ELECTROSTATIC_PROFILE_V1,
        )
