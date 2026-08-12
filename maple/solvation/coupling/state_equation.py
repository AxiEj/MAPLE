"""Provider-independent reduced Route-2 state equation.

The public residual is exactly

``T_plus [c - Pi_q M(R, P_R(c))]`` with ``c = c_ref + T y``.

The reduced coordinates ``y`` are dimensionless by contract.  Providers own
their native implementations; this module contains no model-family branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from maple.solvation.api.state_registry import OPERATIONAL_STATE_EQUATION_ID

from .spaces import AffineChargeCoordinates


def _finite_vector(values: object, size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape ({size},); received {result.shape}.")
    return np.array(result, copy=True)


@runtime_checkable
class ElectronicResponseProvider(Protocol):
    """Electronic source map and its exact local derivatives."""

    def evaluate_source(self, geometry: Any, field: np.ndarray) -> np.ndarray: ...
    def field_jvp(
        self, geometry: Any, field: np.ndarray, field_direction: np.ndarray
    ) -> np.ndarray: ...
    def field_vjp(
        self, geometry: Any, field: np.ndarray, source_cotangent: np.ndarray
    ) -> np.ndarray: ...
    def coordinate_vjp(
        self, geometry: Any, field: np.ndarray, source_cotangent: np.ndarray
    ) -> np.ndarray: ...


@runtime_checkable
class ContinuumResponseProvider(Protocol):
    """Reaction-field map and its exact local derivatives."""

    def evaluate_field(self, geometry: Any, source: np.ndarray) -> np.ndarray: ...
    def source_jvp(
        self, geometry: Any, source: np.ndarray, source_direction: np.ndarray
    ) -> np.ndarray: ...
    def source_vjp(
        self, geometry: Any, source: np.ndarray, field_cotangent: np.ndarray
    ) -> np.ndarray: ...
    def coordinate_vjp(
        self, geometry: Any, source: np.ndarray, field_cotangent: np.ndarray
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class StateEvaluation:
    """Immutable snapshot of one residual evaluation."""

    y: tuple[float, ...]
    source: tuple[tuple[float, ...], ...]
    field: tuple[tuple[float, ...], ...]
    response_source: tuple[tuple[float, ...], ...]
    projected_source: tuple[tuple[float, ...], ...]
    residual: tuple[float, ...]

    @property
    def residual_norm(self) -> float:
        return float(np.linalg.norm(self.residual))


@dataclass(frozen=True)
class ReducedStateEquation:
    """Charge-constrained mutual-polarization residual in dimensionless ``y``.

    Phase 3 uses geometry-independent affine coordinates and a static pairing
    metric.  Coordinate partials here therefore include provider maps only.  A
    future geometry-dependent ``c_ref``, ``T``, or ``Q`` must extend the
    coordinate/pairing protocols with their own coordinate pullbacks before it
    can use this implementation.
    """

    coordinates: AffineChargeCoordinates
    electronic: ElectronicResponseProvider
    continuum: ContinuumResponseProvider
    state_equation_id: str = OPERATIONAL_STATE_EQUATION_ID

    def __post_init__(self) -> None:
        if not isinstance(self.coordinates, AffineChargeCoordinates):
            raise TypeError("coordinates must be AffineChargeCoordinates.")
        if not isinstance(self.state_equation_id, str) or not self.state_equation_id.strip():
            raise ValueError("state_equation_id must be non-empty.")

    @property
    def reduced_dimension(self) -> int:
        return self.coordinates.reduced_dimension

    def _y(self, y: object, name: str = "dimensionless reduced coordinates") -> np.ndarray:
        return _finite_vector(y, self.reduced_dimension, name)

    def _source(self, values: object, name: str) -> np.ndarray:
        source = np.asarray(values, dtype=float)
        expected = self.coordinates.source_space.shape(self.coordinates.atom_count)
        if source.shape != expected or not np.all(np.isfinite(source)):
            raise ValueError(f"{name} must be finite with shape {expected}; received {source.shape}.")
        return np.array(source, copy=True)

    def evaluate(self, geometry: Any, y: object) -> StateEvaluation:
        reduced = self._y(y)
        source = self.coordinates.expand(reduced)
        field = self._source(
            self.continuum.evaluate_field(geometry, source), "continuum field"
        )
        response = self._source(
            self.electronic.evaluate_source(geometry, field), "electronic response source"
        )
        projected = self.coordinates.project_affine(response)
        # This spelling deliberately follows the authoritative equation rather
        # than simplifying to y - reduce(projected).
        residual = self.coordinates.reduce_tangent(source - projected)
        return StateEvaluation(
            y=tuple(float(value) for value in reduced),
            source=tuple(tuple(float(value) for value in row) for row in source),
            field=tuple(tuple(float(value) for value in row) for row in field),
            response_source=tuple(tuple(float(value) for value in row) for row in response),
            projected_source=tuple(tuple(float(value) for value in row) for row in projected),
            residual=tuple(float(value) for value in residual),
        )

    def residual(self, geometry: Any, y: object) -> np.ndarray:
        result = np.asarray(self.evaluate(geometry, y).residual, dtype=float)
        result.setflags(write=False)
        return result

    def fixed_point_map(self, geometry: Any, y: object) -> np.ndarray:
        evaluation = self.evaluate(geometry, y)
        result = self.coordinates.reduce(np.asarray(evaluation.projected_source))
        result.setflags(write=False)
        return result

    def jvp(self, geometry: Any, y: object, dy: object) -> np.ndarray:
        evaluation = self.evaluate(geometry, y)
        direction = self._y(dy, "reduced direction")
        source = np.asarray(evaluation.source)
        field = np.asarray(evaluation.field)
        dc = self.coordinates.expand_direction(direction)
        du = self._source(
            self.continuum.source_jvp(geometry, source, dc), "continuum field JVP"
        )
        dm = self._source(
            self.electronic.field_jvp(geometry, field, du), "electronic source JVP"
        )
        projected_dm = self.coordinates.project_tangent(dm)
        result = self.coordinates.reduce_tangent(dc - projected_dm)
        result.setflags(write=False)
        return result

    def vjp(self, geometry: Any, y: object, residual_cotangent: object) -> np.ndarray:
        """Apply ``r_y.T`` using ``T_plus.T`` then the final ``T.T`` pullback."""

        evaluation = self.evaluate(geometry, y)
        cotangent = self._y(residual_cotangent, "residual cotangent")
        source = np.asarray(evaluation.source)
        field = np.asarray(evaluation.field)

        # residual = T_plus (source - Pi_q response).  The cotangent of the
        # bracket is therefore T_plus.T @ residual_cotangent.
        bracket_bar = self.coordinates.lift_reduced_cotangent(cotangent)
        direct_source_bar = bracket_bar
        # Pi tangent = T T_plus, so Pi.T = T_plus.T T.T.
        response_bar = -self.coordinates.lift_reduced_cotangent(
            self.coordinates.reduce_source_cotangent(bracket_bar)
        )
        field_bar = self._source(
            self.electronic.field_vjp(geometry, field, response_bar),
            "electronic field VJP",
        )
        continuum_source_bar = self._source(
            self.continuum.source_vjp(geometry, source, field_bar),
            "continuum source VJP",
        )
        # source = c_ref + T y, hence the final cotangent map is T.T.
        result = self.coordinates.reduce_source_cotangent(
            direct_source_bar + continuum_source_bar
        )
        result.setflags(write=False)
        return result

    def coordinate_vjp(
        self, geometry: Any, y: object, residual_cotangent: object
    ) -> np.ndarray:
        """Apply the partial coordinate pullback ``r_R.T`` at fixed ``y``."""

        evaluation = self.evaluate(geometry, y)
        cotangent = self._y(residual_cotangent, "residual cotangent")
        source = np.asarray(evaluation.source)
        field = np.asarray(evaluation.field)
        bracket_bar = self.coordinates.lift_reduced_cotangent(cotangent)
        response_bar = -self.coordinates.lift_reduced_cotangent(
            self.coordinates.reduce_source_cotangent(bracket_bar)
        )
        field_bar = self._source(
            self.electronic.field_vjp(geometry, field, response_bar),
            "electronic field VJP",
        )
        electronic_geometry_bar = np.asarray(
            self.electronic.coordinate_vjp(geometry, field, response_bar), dtype=float
        )
        continuum_geometry_bar = np.asarray(
            self.continuum.coordinate_vjp(geometry, source, field_bar), dtype=float
        )
        if (
            electronic_geometry_bar.shape != continuum_geometry_bar.shape
            or not np.all(np.isfinite(electronic_geometry_bar))
            or not np.all(np.isfinite(continuum_geometry_bar))
        ):
            raise ValueError("Provider coordinate VJPs must be finite and equally shaped.")
        result = electronic_geometry_bar + continuum_geometry_bar
        result.setflags(write=False)
        return result


__all__ = [
    "ContinuumResponseProvider",
    "ElectronicResponseProvider",
    "ReducedStateEquation",
    "StateEvaluation",
]
