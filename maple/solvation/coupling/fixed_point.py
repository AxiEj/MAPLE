"""Deterministic Picard/Anderson solver for the reduced state equation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

import numpy as np

from .state_equation import ReducedStateEquation


@dataclass(frozen=True)
class FixedPointOptions:
    method: Literal["picard", "anderson"] = "anderson"
    tolerance: float = 1.0e-12
    max_iterations: int = 200
    damping: float = 1.0
    history: int = 6
    root_hash_decimals: int = 10

    def __post_init__(self) -> None:
        if self.method not in ("picard", "anderson"):
            raise ValueError("method must be 'picard' or 'anderson'.")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer.")
        if not np.isfinite(self.damping) or not 0.0 < self.damping <= 1.0:
            raise ValueError("damping must lie in (0, 1].")
        if type(self.history) is not int or self.history < 1:
            raise ValueError("history must be a positive integer.")
        if type(self.root_hash_decimals) is not int or self.root_hash_decimals < 0:
            raise ValueError("root_hash_decimals must be a non-negative integer.")


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    actual_unmixed_residual_norm: float
    step_norm: float
    history_depth: int

    def __post_init__(self) -> None:
        if type(self.iteration) is not int or self.iteration < 0:
            raise ValueError("iteration must be a non-negative integer.")
        for name in ("actual_unmixed_residual_norm", "step_norm"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if type(self.history_depth) is not int or self.history_depth < 0:
            raise ValueError("history_depth must be a non-negative integer.")


@dataclass(frozen=True)
class FixedPointState:
    state_equation_id: str
    root_context_id: str
    initialization: Literal["cold", "warm"]
    converged: bool
    y: tuple[float, ...]
    source: tuple[tuple[float, ...], ...]
    field: tuple[tuple[float, ...], ...]
    actual_unmixed_residual: tuple[float, ...]
    actual_unmixed_residual_norm: float
    iterations: tuple[IterationRecord, ...]
    root_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.state_equation_id, str) or not self.state_equation_id.strip():
            raise ValueError("state_equation_id must be non-empty.")
        if not isinstance(self.root_context_id, str) or not self.root_context_id.strip():
            raise ValueError("root_context_id must be non-empty.")
        if self.initialization not in ("cold", "warm"):
            raise ValueError("initialization must be 'cold' or 'warm'.")
        if type(self.converged) is not bool:
            raise TypeError("converged must be a bool.")
        y = np.asarray(self.y, dtype=float)
        residual = np.asarray(self.actual_unmixed_residual, dtype=float)
        source = np.asarray(self.source, dtype=float)
        field = np.asarray(self.field, dtype=float)
        if y.ndim != 1 or y.size == 0 or not np.all(np.isfinite(y)):
            raise ValueError("y must be a non-empty finite vector.")
        if residual.shape != y.shape or not np.all(np.isfinite(residual)):
            raise ValueError("actual_unmixed_residual must be finite and match y.")
        if source.ndim != 2 or source.shape[0] < 1 or source.shape[1] < 1 or not np.all(np.isfinite(source)):
            raise ValueError("source must be a non-empty finite two-dimensional array.")
        if field.shape != source.shape or not np.all(np.isfinite(field)):
            raise ValueError("field must be finite and match the source shape.")
        norm = float(self.actual_unmixed_residual_norm)
        if not np.isfinite(norm) or norm < 0.0:
            raise ValueError("actual_unmixed_residual_norm must be finite and non-negative.")
        if not np.isclose(norm, np.linalg.norm(residual), rtol=1.0e-14, atol=1.0e-15):
            raise ValueError("actual_unmixed_residual_norm does not match the residual vector.")
        if not self.iterations or any(not isinstance(item, IterationRecord) for item in self.iterations):
            raise ValueError("iterations must contain immutable IterationRecord values.")
        if (
            not isinstance(self.root_hash, str)
            or len(self.root_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.root_hash)
        ):
            raise ValueError("root_hash must contain exactly 64 lowercase hexadecimal digits.")

    def y_array(self) -> np.ndarray:
        result = np.asarray(self.y, dtype=float)
        result.setflags(write=False)
        return result

    def source_array(self) -> np.ndarray:
        result = np.asarray(self.source, dtype=float)
        result.setflags(write=False)
        return result

    def field_array(self) -> np.ndarray:
        result = np.asarray(self.field, dtype=float)
        result.setflags(write=False)
        return result


class FixedPointConvergenceError(RuntimeError):
    def __init__(self, state: FixedPointState):
        self.state = state
        super().__init__(
            f"State equation did not converge in {len(state.iterations)} iterations; "
            f"actual unmixed reduced residual={state.actual_unmixed_residual_norm:.6e}."
        )


def _canonical_values(values: object, decimals: int) -> list[object]:
    rounded = np.round(np.asarray(values, dtype=float), decimals=decimals)
    rounded[rounded == 0.0] = 0.0
    return rounded.tolist()


def _root_hash(
    state_equation_id: str,
    root_context_id: str,
    y: np.ndarray,
    source: object,
    field: object,
    decimals: int,
) -> str:
    payload = {
        "state_equation_id": state_equation_id,
        # The caller binds geometry/provider/scalar/profile provenance in this
        # context ID.  Canonicalized source and field prevent a y-only hash from
        # colliding across different physical roots while preserving numerical
        # cold/warm equivalence at the declared solver precision.
        "root_context_id": root_context_id,
        "dimensionless_y": _canonical_values(y, decimals),
        "source": _canonical_values(source, decimals),
        "field": _canonical_values(field, decimals),
        "canonical_decimals": decimals,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _anderson_step(
    x_history: list[np.ndarray],
    f_history: list[np.ndarray],
    mapped: np.ndarray,
    damping: float,
    history: int,
) -> tuple[np.ndarray, int]:
    """Type-II Anderson step for ``f(x)=G(x)-x`` with deterministic least squares."""

    depth = min(history, len(f_history) - 1)
    if depth == 0:
        return x_history[-1] + damping * f_history[-1], 0
    start = len(f_history) - depth - 1
    delta_x = np.column_stack(
        [x_history[index + 1] - x_history[index] for index in range(start, len(x_history) - 1)]
    )
    delta_f = np.column_stack(
        [f_history[index + 1] - f_history[index] for index in range(start, len(f_history) - 1)]
    )
    gamma = np.linalg.lstsq(delta_f, f_history[-1], rcond=None)[0]
    accelerated = mapped - (delta_x + delta_f) @ gamma
    return x_history[-1] + damping * (accelerated - x_history[-1]), depth


def solve_fixed_point(
    equation: ReducedStateEquation,
    geometry: Any,
    *,
    root_context_id: str,
    initial_y: object | None = None,
    options: FixedPointOptions = FixedPointOptions(),
    require_convergence: bool = True,
) -> FixedPointState:
    """Solve deterministically and gate only on the actual unmixed residual.

    ``initial_y=None`` is the canonical cold start (all-zero dimensionless
    coordinates).  Supplying a vector is recorded as a warm start.
    """

    if not isinstance(root_context_id, str) or not root_context_id.strip():
        raise ValueError(
            "root_context_id must bind geometry, provider, scalar, and profile identity."
        )
    if not isinstance(options, FixedPointOptions):
        raise TypeError("options must be FixedPointOptions.")
    initialization: Literal["cold", "warm"] = "cold" if initial_y is None else "warm"
    y = (
        np.zeros(equation.reduced_dimension, dtype=float)
        if initial_y is None
        else equation._y(initial_y, "initial reduced coordinates")
    )
    x_history: list[np.ndarray] = []
    f_history: list[np.ndarray] = []
    records: list[IterationRecord] = []
    converged = False

    for iteration in range(options.max_iterations + 1):
        # Never use a mixed/extrapolated residual for the convergence gate.
        residual = np.asarray(equation.residual(geometry, y), dtype=float)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= options.tolerance:
            records.append(IterationRecord(iteration, residual_norm, 0.0, 0))
            converged = True
            break
        if iteration == options.max_iterations:
            records.append(IterationRecord(iteration, residual_norm, 0.0, 0))
            break

        mapped = np.asarray(equation.fixed_point_map(geometry, y), dtype=float)
        fixed_point_residual = mapped - y
        x_history.append(np.array(y, copy=True))
        f_history.append(np.array(fixed_point_residual, copy=True))
        if options.method == "anderson":
            next_y, depth = _anderson_step(
                x_history, f_history, mapped, options.damping, options.history
            )
        else:
            next_y = y + options.damping * fixed_point_residual
            depth = 0
        if not np.all(np.isfinite(next_y)):
            raise ValueError("Fixed-point update produced non-finite reduced coordinates.")
        step_norm = float(np.linalg.norm(next_y - y))
        records.append(IterationRecord(iteration, residual_norm, step_norm, depth))
        y = next_y

    final = equation.evaluate(geometry, y)
    final_residual = np.asarray(final.residual, dtype=float)
    final_norm = float(np.linalg.norm(final_residual))
    state = FixedPointState(
        state_equation_id=equation.state_equation_id,
        root_context_id=root_context_id.strip(),
        initialization=initialization,
        converged=converged and final_norm <= options.tolerance,
        y=tuple(float(value) for value in y),
        source=final.source,
        field=final.field,
        actual_unmixed_residual=tuple(float(value) for value in final_residual),
        actual_unmixed_residual_norm=final_norm,
        iterations=tuple(records),
        root_hash=_root_hash(
            equation.state_equation_id,
            root_context_id.strip(),
            y,
            final.source,
            final.field,
            options.root_hash_decimals,
        ),
    )
    if require_convergence and not state.converged:
        raise FixedPointConvergenceError(state)
    return state


__all__ = [
    "FixedPointConvergenceError",
    "FixedPointOptions",
    "FixedPointState",
    "IterationRecord",
    "solve_fixed_point",
]
