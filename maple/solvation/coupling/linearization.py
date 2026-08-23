"""Exact reduced linearization of the provider-independent state equation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .state_equation import ReducedStateEquation
from .separated_state import SeparatedReducedStateEquation


@runtime_checkable
class ReducedLinearizationOperator(Protocol):
    """Matrix-free reduced residual derivative consumed by the adjoint."""

    @property
    def dimension(self) -> int: ...

    def jvp(self, direction: object) -> np.ndarray: ...

    def vjp(self, cotangent: object) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ReducedLinearization:
    equation: ReducedStateEquation
    geometry: Any
    y: tuple[float, ...]

    @classmethod
    def at(
        cls, equation: ReducedStateEquation, geometry: Any, y: object
    ) -> "ReducedLinearization":
        values = equation._y(y)
        return cls(equation=equation, geometry=geometry, y=tuple(float(v) for v in values))

    @property
    def dimension(self) -> int:
        return self.equation.reduced_dimension

    def jvp(self, direction: object) -> np.ndarray:
        return self.equation.jvp(self.geometry, self.y, direction)

    def vjp(self, cotangent: object) -> np.ndarray:
        return self.equation.vjp(self.geometry, self.y, cotangent)

    def dense_jacobian(self) -> np.ndarray:
        basis = np.eye(self.dimension)
        result = np.column_stack([self.jvp(basis[:, index]) for index in range(self.dimension)])
        result.setflags(write=False)
        return result

    def transpose_error(self, direction: object, cotangent: object) -> float:
        v = np.asarray(direction, dtype=float)
        w = np.asarray(cotangent, dtype=float)
        return abs(float(np.vdot(self.jvp(v), w) - np.vdot(v, self.vjp(w))))


@dataclass(frozen=True, slots=True)
class SeparatedReducedLinearization:
    """Exact reduced derivative for distinct source and native-field spaces."""

    equation: SeparatedReducedStateEquation
    geometry: Any
    y: tuple[float, ...]

    @classmethod
    def at(
        cls,
        equation: SeparatedReducedStateEquation,
        geometry: Any,
        y: object,
    ) -> "SeparatedReducedLinearization":
        if not isinstance(equation, SeparatedReducedStateEquation):
            raise TypeError("equation must satisfy SeparatedReducedStateEquation.")
        values = np.asarray(y, dtype=float)
        if values.shape != (equation.reduced_dimension,) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "separated reduced coordinates have an invalid shape."
            )
        # Binds geometry and provider state immediately rather than deferring a
        # stale-snapshot failure until GMRES has already started.
        equation.fingerprint_sha256()
        equation.residual(geometry, values)
        return cls(
            equation=equation,
            geometry=geometry,
            y=tuple(float(value) for value in values),
        )

    @property
    def dimension(self) -> int:
        return self.equation.reduced_dimension

    def jvp(self, direction: object) -> np.ndarray:
        return self.equation.residual_jvp(self.geometry, self.y, direction)

    def vjp(self, cotangent: object) -> np.ndarray:
        return self.equation.residual_vjp(self.geometry, self.y, cotangent)

    def dense_jacobian(self) -> np.ndarray:
        basis = np.eye(self.dimension)
        result = np.column_stack(
            [self.jvp(basis[:, index]) for index in range(self.dimension)]
        )
        result.setflags(write=False)
        return result

    def transpose_error(self, direction: object, cotangent: object) -> float:
        v = np.asarray(direction, dtype=float)
        w = np.asarray(cotangent, dtype=float)
        return abs(float(np.vdot(self.jvp(v), w) - np.vdot(v, self.vjp(w))))


__all__ = [
    "ReducedLinearization",
    "ReducedLinearizationOperator",
    "SeparatedReducedLinearization",
]
