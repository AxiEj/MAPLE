"""Exact reduced linearization of the provider-independent state equation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .state_equation import ReducedStateEquation


@dataclass(frozen=True)
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


__all__ = ["ReducedLinearization"]
