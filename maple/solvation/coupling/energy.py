"""Canonical operational scalar for the fixed-topology electrostatic profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from maple.solvation.api.scalar_registry import OPERATIONAL_CPCM_ELECTROSTATIC_V1

from .adjoint import AdjointOptions, AdjointResult, solve_reduced_adjoint
from .fixed_point import FixedPointState
from .linearization import ReducedLinearization
from .metrics import ATOMIC_L1_PAIRING, PairingMetric
from .state_equation import ContinuumResponseProvider, ReducedStateEquation


@runtime_checkable
class VacuumScalarProvider(Protocol):
    """Vacuum scalar and its partial Cartesian derivative."""

    def evaluate_energy(self, geometry: Any) -> float: ...
    def coordinate_gradient(self, geometry: Any) -> np.ndarray: ...


@runtime_checkable
class FixedTopologyReciprocalLinearContinuum(ContinuumResponseProvider, Protocol):
    """Provider declaration required by the canonical Phase-3 scalar."""

    fixed_topology: bool
    linear_response: bool
    reciprocal: bool


def _source_array(
    values: object, atom_count: int, component_count: int, name: str
) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    expected = (atom_count, component_count)
    if result.shape != expected or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {expected}; received {result.shape}.")
    return np.array(result, copy=True)


def _source_dual_of_field(field: np.ndarray, metric: PairingMetric) -> np.ndarray:
    return np.einsum("ij,nj->ni", metric.block, field)


def _field_cotangent_of_source(source: np.ndarray, metric: PairingMetric) -> np.ndarray:
    return np.einsum("ji,nj->ni", metric.block, source)


@dataclass(frozen=True)
class HalfCouplingEvaluation:
    energy: float
    direct_source_gradient: tuple[tuple[float, ...], ...]
    response_source_gradient: tuple[tuple[float, ...], ...]
    total_source_gradient: tuple[tuple[float, ...], ...]
    coordinate_gradient: tuple[float, ...]

    def total_source_gradient_array(self) -> np.ndarray:
        result = np.asarray(self.total_source_gradient, dtype=float)
        result.setflags(write=False)
        return result

    def coordinate_gradient_array(self) -> np.ndarray:
        result = np.asarray(self.coordinate_gradient, dtype=float)
        result.setflags(write=False)
        return result


def nonlinear_half_coupling(
    continuum: ContinuumResponseProvider,
    geometry: Any,
    source: object,
    *,
    metric: PairingMetric = ATOMIC_L1_PAIRING,
) -> HalfCouplingEvaluation:
    """Evaluate ``0.5 <c,P_R(c)>_Q`` with direct plus ``J_P.T`` response.

    The expression is valid for nonlinear response maps.  For an admitted
    linear reciprocal map the two source-gradient halves are identical and
    their sum is ``Q P_R(c)``.
    """

    values = np.asarray(source, dtype=float)
    if values.ndim != 2 or values.shape[1] != metric.component_count:
        raise ValueError("source must have shape (atom_count, metric.component_count).")
    if not np.all(np.isfinite(values)):
        raise ValueError("source must be finite.")
    atom_count = values.shape[0]
    field = _source_array(
        continuum.evaluate_field(geometry, values),
        atom_count,
        metric.component_count,
        "continuum field",
    )
    energy = 0.5 * metric.pair(values, field)
    direct = 0.5 * _source_dual_of_field(field, metric)
    field_cotangent = 0.5 * _field_cotangent_of_source(values, metric)
    response = _source_array(
        continuum.source_vjp(geometry, values, field_cotangent),
        atom_count,
        metric.component_count,
        "half-coupling response source gradient",
    )
    coordinate = np.asarray(
        continuum.coordinate_vjp(geometry, values, field_cotangent), dtype=float
    )
    if coordinate.ndim != 1 or not np.all(np.isfinite(coordinate)):
        raise ValueError("continuum coordinate VJP must be a finite vector.")
    total = direct + response
    return HalfCouplingEvaluation(
        energy=float(energy),
        direct_source_gradient=tuple(tuple(float(v) for v in row) for row in direct),
        response_source_gradient=tuple(tuple(float(v) for v in row) for row in response),
        total_source_gradient=tuple(tuple(float(v) for v in row) for row in total),
        coordinate_gradient=tuple(float(v) for v in coordinate),
    )


def reciprocal_linear_half_coupling(
    continuum: ContinuumResponseProvider,
    geometry: Any,
    source: object,
    *,
    metric: PairingMetric = ATOMIC_L1_PAIRING,
    reciprocity_tolerance: float = 1.0e-10,
) -> HalfCouplingEvaluation:
    """Canonical linear reciprocal half coupling, rejecting broken reciprocity."""

    for declaration in ("fixed_topology", "linear_response", "reciprocal"):
        if getattr(continuum, declaration, False) is not True:
            raise ValueError(
                "Canonical operational half coupling requires an explicitly "
                f"declared {declaration}=True continuum provider."
            )

    result = nonlinear_half_coupling(continuum, geometry, source, metric=metric)
    direct = np.asarray(result.direct_source_gradient)
    response = np.asarray(result.response_source_gradient)
    error = float(np.linalg.norm(direct - response))
    scale = max(1.0, float(np.linalg.norm(direct)), float(np.linalg.norm(response)))
    if error > reciprocity_tolerance * scale:
        raise ValueError(
            "Continuum response is not reciprocal under the authoritative pairing: "
            f"gradient-half error={error:.6e}."
        )
    return result


@dataclass(frozen=True)
class OperationalScalarEvaluation:
    scalar_id: str
    vacuum_energy: float
    continuum_energy: float
    total_energy: float
    reduced_gradient: tuple[float, ...]
    direct_coordinate_gradient: tuple[float, ...]


@dataclass(frozen=True)
class OperationalGradientResult:
    scalar: OperationalScalarEvaluation
    adjoint: AdjointResult
    total_coordinate_gradient: tuple[float, ...]

    @property
    def forces(self) -> tuple[float, ...]:
        return tuple(-value for value in self.total_coordinate_gradient)


@dataclass(frozen=True)
class OperationalElectrostaticScalar:
    """``E_vac + 0.5<c,P_R(c)>_Q`` evaluated along the admitted root."""

    equation: ReducedStateEquation
    vacuum: VacuumScalarProvider
    scalar_id: str = OPERATIONAL_CPCM_ELECTROSTATIC_V1
    metric: PairingMetric = ATOMIC_L1_PAIRING

    def evaluate(self, geometry: Any, y: object) -> OperationalScalarEvaluation:
        reduced = self.equation._y(y)
        source = self.equation.coordinates.expand(reduced)
        continuum = reciprocal_linear_half_coupling(
            self.equation.continuum, geometry, source, metric=self.metric
        )
        vacuum_energy = float(self.vacuum.evaluate_energy(geometry))
        if not np.isfinite(vacuum_energy):
            raise ValueError("vacuum energy must be finite.")
        vacuum_gradient = np.asarray(self.vacuum.coordinate_gradient(geometry), dtype=float)
        continuum_gradient = continuum.coordinate_gradient_array()
        if vacuum_gradient.shape != continuum_gradient.shape or not np.all(np.isfinite(vacuum_gradient)):
            raise ValueError("Vacuum and continuum coordinate gradients must be finite and equally shaped.")
        source_gradient = continuum.total_source_gradient_array()
        reduced_gradient = self.equation.coordinates.reduce_source_cotangent(source_gradient)
        direct_coordinate_gradient = vacuum_gradient + continuum_gradient
        return OperationalScalarEvaluation(
            scalar_id=self.scalar_id,
            vacuum_energy=vacuum_energy,
            continuum_energy=continuum.energy,
            total_energy=vacuum_energy + continuum.energy,
            reduced_gradient=tuple(float(value) for value in reduced_gradient),
            direct_coordinate_gradient=tuple(float(value) for value in direct_coordinate_gradient),
        )

    def implicit_gradient(
        self,
        geometry: Any,
        state: FixedPointState,
        *,
        adjoint_options: AdjointOptions = AdjointOptions(),
    ) -> OperationalGradientResult:
        if not state.converged:
            raise ValueError("Implicit differentiation requires a converged primal state.")
        if state.state_equation_id != self.equation.state_equation_id:
            raise ValueError("Primal state and operational scalar use different state equations.")
        scalar = self.evaluate(geometry, state.y)
        linearization = ReducedLinearization.at(self.equation, geometry, state.y)
        adjoint = solve_reduced_adjoint(
            linearization, scalar.reduced_gradient, options=adjoint_options
        )
        residual_coordinate_pullback = self.equation.coordinate_vjp(
            geometry, state.y, adjoint.solution
        )
        direct = np.asarray(scalar.direct_coordinate_gradient)
        if direct.shape != residual_coordinate_pullback.shape:
            raise ValueError("Scalar and residual coordinate gradients must be equally shaped.")
        total = direct - residual_coordinate_pullback
        return OperationalGradientResult(
            scalar=scalar,
            adjoint=adjoint,
            total_coordinate_gradient=tuple(float(value) for value in total),
        )


__all__ = [
    "HalfCouplingEvaluation",
    "FixedTopologyReciprocalLinearContinuum",
    "OperationalElectrostaticScalar",
    "OperationalGradientResult",
    "OperationalScalarEvaluation",
    "VacuumScalarProvider",
    "nonlinear_half_coupling",
    "reciprocal_linear_half_coupling",
]
