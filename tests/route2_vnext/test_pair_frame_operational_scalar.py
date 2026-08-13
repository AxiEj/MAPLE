from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import ClassVar

import numpy as np
from maple.solvation.api.profiles import (
    DIAGNOSTIC_PAIR_FRAME_CPCM_RADIAL_GTO_PROFILE_V1,
    MACE_POLAR_FIXED_BOX40_MODEL_PROFILE_ID,
    MACE_POLAR_RADIAL_GTO_COUPLING_ID,
)
from maple.solvation.continuum import build_pair_frame_water_cpcm_110_candidate
from maple.solvation.coupling.adjoint import AdjointOptions
from maple.solvation.coupling.energy import OperationalElectrostaticScalar
from maple.solvation.coupling.fixed_point import FixedPointOptions, solve_fixed_point
from maple.solvation.coupling.spaces import (
    LinearChargeCoordinates,
    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)
from maple.solvation.coupling.state_equation import ReducedStateEquation
from maple.solvation.release import load_pes_panel


def _content_sha256(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass
class _ElectronicOracle:
    bias: np.ndarray
    field_matrix: np.ndarray
    geometry_matrix: np.ndarray
    provider_id: ClassVar[str] = "maple.route2.test.pairframe-electronic.v1"
    model_profile_id: ClassVar[str] = MACE_POLAR_FIXED_BOX40_MODEL_PROFILE_ID
    coupling_id: ClassVar[str] = MACE_POLAR_RADIAL_GTO_COUPLING_ID
    provenance_sha256: ClassVar[str] = "e" * 64
    source_space = MACE_POLAR_RADIAL_GTO_SOURCE_SPACE
    field_space = MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE

    def configuration_sha256(self) -> str:
        return _content_sha256(self.bias, self.field_matrix, self.geometry_matrix)

    def evaluate_source(self, geometry, field):
        values = (
            self.bias
            + self.field_matrix @ np.asarray(field).reshape(-1)
            + self.geometry_matrix @ np.asarray(geometry).reshape(-1)
        )
        return values.reshape(np.asarray(field).shape)

    def field_jvp(self, geometry, field, direction):
        return (self.field_matrix @ np.asarray(direction).reshape(-1)).reshape(
            np.asarray(field).shape
        )

    def field_vjp(self, geometry, field, cotangent):
        return (self.field_matrix.T @ np.asarray(cotangent).reshape(-1)).reshape(
            np.asarray(field).shape
        )

    def coordinate_vjp(self, geometry, field, cotangent):
        return self.geometry_matrix.T @ np.asarray(cotangent).reshape(-1)


@dataclass
class _VacuumOracle:
    diagonal: np.ndarray
    provider_id: ClassVar[str] = "maple.route2.test.pairframe-vacuum.v1"
    model_profile_id: ClassVar[str] = MACE_POLAR_FIXED_BOX40_MODEL_PROFILE_ID
    provenance_sha256: ClassVar[str] = "a" * 64

    def configuration_sha256(self) -> str:
        return _content_sha256(self.diagonal)

    def evaluate_energy(self, geometry) -> float:
        flat = np.asarray(geometry).reshape(-1)
        return float(0.5 * np.vdot(self.diagonal * flat, flat))

    def coordinate_gradient(self, geometry) -> np.ndarray:
        return self.diagonal * np.asarray(geometry).reshape(-1)


def test_pair_frame_candidate_closes_one_implicit_operational_scalar():
    atoms = load_pes_panel()[1].atoms
    geometry = np.asarray(atoms.positions)
    coordinates = LinearChargeCoordinates(
        len(atoms),
        total_charge=0.0,
        source_space=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
        component_scales=(1.0,) * 8,
    )
    continuum = build_pair_frame_water_cpcm_110_candidate(atoms.get_chemical_symbols())
    rng = np.random.default_rng(20260813)
    size = coordinates.source_dimension
    electronic = _ElectronicOracle(
        rng.normal(scale=0.01, size=size),
        rng.normal(scale=1e-5, size=(size, size)),
        rng.normal(scale=2e-4, size=(size, geometry.size)),
    )
    vacuum = _VacuumOracle(np.linspace(0.03, 0.08, geometry.size))
    equation = ReducedStateEquation(coordinates, electronic, continuum)
    scalar = OperationalElectrostaticScalar(
        equation,
        vacuum,
        profile_id=DIAGNOSTIC_PAIR_FRAME_CPCM_RADIAL_GTO_PROFILE_V1,
    )

    def solve(values, initial=None):
        digest = hashlib.sha256(np.asarray(values).tobytes()).hexdigest()
        return solve_fixed_point(
            equation,
            values,
            scalar_id=scalar.scalar_id,
            profile_id=scalar.profile_id,
            scalar_binding=scalar,
            root_context_id=f"pairframe-operational/{digest}",
            initial_y=initial,
            options=FixedPointOptions(tolerance=1e-13, max_iterations=100),
        )

    state = solve(geometry)
    result = scalar.implicit_gradient(
        geometry,
        state,
        adjoint_options=AdjointOptions(
            relative_tolerance=1e-13, absolute_tolerance=1e-14
        ),
    )
    direction = rng.normal(size=geometry.shape)
    direction -= np.mean(direction, axis=0)
    direction /= np.linalg.norm(direction)
    analytic = float(
        np.vdot(
            np.asarray(result.total_coordinate_gradient).reshape(geometry.shape),
            direction,
        )
    )
    errors = []
    for step in (2e-4, 1e-4, 5e-5):
        plus = geometry + step * direction
        minus = geometry - step * direction
        plus_state = solve(plus, state.y_array())
        minus_state = solve(minus, state.y_array())
        finite_difference = (
            scalar.evaluate_energy(plus, plus_state.y)
            - scalar.evaluate_energy(minus, minus_state.y)
        ) / (2 * step)
        errors.append(abs(finite_difference - analytic))
    assert errors[-1] < 2e-8
    assert errors[-1] < errors[0]
    assert state.actual_unmixed_residual_norm <= state.primal_tolerance
    assert result.adjoint.true_residual_norm <= result.adjoint.acceptance_tolerance
    assert continuum.capabilities.enabled_tiers == ()
