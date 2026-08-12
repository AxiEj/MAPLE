"""Deterministic Picard/Anderson solver for the reduced state equation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal, Protocol

import numpy as np

from maple.solvation.api.profiles import (
    UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID,
    get_solvation_profile,
)
from maple.solvation.api.scalar_registry import get_scalar_definition

from .metrics import get_pairing_metric
from .spaces import get_coordinate_contract, get_field_dual_space, get_source_space
from .state_equation import ReducedStateEquation, geometry_sha256


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _sha256(value: object, name: str) -> str:
    digest = _nonempty(value, name).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal digits.")
    return digest


@dataclass(frozen=True)
class FixedPointOptions:
    method: Literal["picard", "anderson"] = "anderson"
    tolerance: float = 1.0e-12
    max_iterations: int = 200
    damping: float = 1.0
    history: int = 6

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


class ScalarStateBinding(Protocol):
    scalar_id: str
    profile_id: str

    def fingerprint_sha256(self) -> str: ...


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
    geometry_sha256: str
    equation_sha256: str
    scalar_sha256: str
    scalar_id: str
    profile_id: str
    primal_tolerance: float
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
        _nonempty(self.state_equation_id, "state_equation_id")
        _sha256(self.geometry_sha256, "geometry_sha256")
        _sha256(self.equation_sha256, "equation_sha256")
        _sha256(self.scalar_sha256, "scalar_sha256")
        _nonempty(self.scalar_id, "scalar_id")
        _nonempty(self.profile_id, "profile_id")
        _nonempty(self.root_context_id, "root_context_id")
        if not np.isfinite(self.primal_tolerance) or self.primal_tolerance <= 0.0:
            raise ValueError("primal_tolerance must be finite and positive.")
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
        if (
            source.ndim != 2
            or source.shape[0] < 1
            or source.shape[1] < 1
            or not np.all(np.isfinite(source))
        ):
            raise ValueError("source must be a non-empty finite two-dimensional array.")
        if field.shape != source.shape or not np.all(np.isfinite(field)):
            raise ValueError("field must be finite and match the source shape.")
        norm = float(self.actual_unmixed_residual_norm)
        if not np.isfinite(norm) or norm < 0.0:
            raise ValueError(
                "actual_unmixed_residual_norm must be finite and non-negative."
            )
        if not np.isclose(norm, np.linalg.norm(residual), rtol=1.0e-14, atol=1.0e-15):
            raise ValueError(
                "actual_unmixed_residual_norm does not match the residual vector."
            )
        if not self.iterations or any(
            not isinstance(item, IterationRecord) for item in self.iterations
        ):
            raise ValueError(
                "iterations must contain immutable IterationRecord values."
            )
        _sha256(self.root_hash, "root_hash")
        expected_root_hash = compute_root_hash(
            state_equation_id=self.state_equation_id,
            geometry_digest=self.geometry_sha256,
            equation_digest=self.equation_sha256,
            scalar_digest=self.scalar_sha256,
            scalar_id=self.scalar_id,
            profile_id=self.profile_id,
            primal_tolerance=self.primal_tolerance,
            root_context_id=self.root_context_id,
            y=self.y,
            source=self.source,
            field=self.field,
            residual=self.actual_unmixed_residual,
            initialization=self.initialization,
            converged=self.converged,
            residual_norm=self.actual_unmixed_residual_norm,
            iterations=self.iterations,
        )
        if self.root_hash != expected_root_hash:
            raise ValueError("root_hash does not match the bound state contents.")

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


ROOT_EQUIVALENCE_CONTRACT = "route2-root-equivalence-v1"
ROOT_EQUIVALENCE_ABSOLUTE_TOLERANCE = 1.0e-10


def roots_numerically_equivalent(
    first: FixedPointState, second: FixedPointState
) -> bool:
    """Apply the fixed v1 numerical root-equivalence contract.

    Integrity remains exact-content SHA based.  This separate predicate is the
    only place where cold/warm numerical equivalence uses a tolerance.
    """

    if not isinstance(first, FixedPointState) or not isinstance(
        second, FixedPointState
    ):
        raise TypeError("root equivalence requires two FixedPointState values.")
    identities = (
        "state_equation_id",
        "geometry_sha256",
        "equation_sha256",
        "scalar_sha256",
        "scalar_id",
        "profile_id",
    )
    if any(getattr(first, name) != getattr(second, name) for name in identities):
        return False
    if (
        not first.converged
        or not second.converged
        or first.root_context_id != second.root_context_id
        or first.primal_tolerance != second.primal_tolerance
        or first.actual_unmixed_residual_norm > first.primal_tolerance
        or second.actual_unmixed_residual_norm > second.primal_tolerance
    ):
        return False
    tolerance = ROOT_EQUIVALENCE_ABSOLUTE_TOLERANCE
    return all(
        np.allclose(left, right, rtol=0.0, atol=tolerance)
        for left, right in (
            (first.y, second.y),
            (first.source, second.source),
            (first.field, second.field),
            (first.actual_unmixed_residual, second.actual_unmixed_residual),
        )
    )


def _array_content_sha256(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def compute_root_hash(
    *,
    state_equation_id: str,
    geometry_digest: str,
    equation_digest: str,
    scalar_digest: str,
    scalar_id: str,
    profile_id: str,
    primal_tolerance: float,
    root_context_id: str,
    y: object,
    source: object,
    field: object,
    residual: object,
    initialization: str,
    converged: bool,
    residual_norm: float,
    iterations: tuple[IterationRecord, ...],
) -> str:
    payload = {
        "state_equation_id": state_equation_id,
        "geometry_sha256": geometry_digest,
        "equation_sha256": equation_digest,
        "scalar_sha256": scalar_digest,
        "scalar_id": scalar_id,
        "profile_id": profile_id,
        "primal_tolerance": primal_tolerance,
        # This remains diagnostic only; scientific identity is machine-bound by
        # the geometry/equation/scalar/profile fields above.
        "root_context_id": root_context_id,
        "dimensionless_y_content_sha256": _array_content_sha256(y),
        "source_content_sha256": _array_content_sha256(source),
        "field_content_sha256": _array_content_sha256(field),
        "actual_unmixed_residual_content_sha256": _array_content_sha256(residual),
        "initialization": initialization,
        "converged": converged,
        "actual_unmixed_residual_norm": residual_norm,
        "iterations": [
            {
                "iteration": item.iteration,
                "actual_unmixed_residual_norm": item.actual_unmixed_residual_norm,
                "step_norm": item.step_norm,
                "history_depth": item.history_depth,
            }
            for item in iterations
        ],
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
        [
            x_history[index + 1] - x_history[index]
            for index in range(start, len(x_history) - 1)
        ]
    )
    delta_f = np.column_stack(
        [
            f_history[index + 1] - f_history[index]
            for index in range(start, len(f_history) - 1)
        ]
    )
    gamma = np.linalg.lstsq(delta_f, f_history[-1], rcond=None)[0]
    accelerated = mapped - (delta_x + delta_f) @ gamma
    return x_history[-1] + damping * (accelerated - x_history[-1]), depth


def solve_fixed_point(
    equation: ReducedStateEquation,
    geometry: Any,
    *,
    scalar_id: str,
    profile_id: str,
    scalar_binding: ScalarStateBinding,
    root_context_id: str,
    initial_y: object | None = None,
    options: FixedPointOptions = FixedPointOptions(),
    require_convergence: bool = True,
) -> FixedPointState:
    """Solve deterministically and gate only on the actual unmixed residual.

    ``initial_y=None`` is the canonical cold start (all-zero dimensionless
    coordinates).  Supplying a vector is recorded as a warm start.
    """

    scalar_id = _nonempty(scalar_id, "scalar_id")
    profile_id = _nonempty(profile_id, "profile_id")
    scalar_definition = get_scalar_definition(scalar_id)
    profile = get_solvation_profile(profile_id)
    if profile.scalar_id != scalar_id:
        raise ValueError("Registered profile does not bind the requested scalar ID.")
    if profile.state_equation_id != equation.state_equation_id:
        raise ValueError("Registered profile does not bind this state-equation ID.")
    if not callable(getattr(scalar_binding, "fingerprint_sha256", None)):
        raise TypeError("scalar_binding must expose fingerprint_sha256().")
    if scalar_binding.scalar_id != scalar_id or scalar_binding.profile_id != profile_id:
        raise ValueError("Scalar binding identity does not match scalar_id/profile_id.")
    scalar_sha256 = _sha256(
        scalar_binding.fingerprint_sha256(), "scalar binding fingerprint"
    )
    root_context_id = _nonempty(root_context_id, "root_context_id")
    if equation.electronic.model_profile_id != profile.model_profile:
        raise ValueError(
            "Electronic model-profile identity does not match the registry."
        )
    if equation.continuum.continuum_profile_id != profile.continuum_profile:
        raise ValueError("Continuum profile identity does not match the registry.")
    if equation.continuum.cavity_profile_id != profile.cavity_profile:
        raise ValueError(
            "Continuum cavity-profile identity does not match the registry."
        )
    if (
        getattr(
            equation.continuum,
            "configuration_contract_id",
            UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID,
        )
        != profile.continuum_configuration_contract_id
    ):
        raise ValueError(
            "Continuum physical-configuration contract does not match the registry."
        )
    if (
        equation.electronic.coupling_id != profile.coupling_id
        or equation.continuum.coupling_id != profile.coupling_id
    ):
        raise ValueError(
            "Source/receiver coupling identity does not match the registry."
        )
    registered_source = get_source_space(profile.source_space_id)
    registered_field = get_field_dual_space(profile.field_space_id)
    registered_pairing = get_pairing_metric(profile.pairing_id)
    get_coordinate_contract(profile.coordinate_contract_id).validate(
        equation.coordinates
    )
    if equation.source_space.metadata_hash() != registered_source.metadata_hash():
        raise ValueError(
            "State-equation source identity does not match the profile registry."
        )
    if equation.field_space.metadata_hash() != registered_field.metadata_hash():
        raise ValueError(
            "State-equation field identity does not match the profile registry."
        )
    if (
        equation.field_space.pairing_metric.metadata_hash()
        != registered_pairing.metadata_hash()
    ):
        raise ValueError(
            "State-equation pairing identity does not match the profile registry."
        )
    if scalar_definition.state_equation_id != equation.state_equation_id:
        raise ValueError("Registered scalar does not bind this state-equation ID.")
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
            raise ValueError(
                "Fixed-point update produced non-finite reduced coordinates."
            )
        step_norm = float(np.linalg.norm(next_y - y))
        records.append(IterationRecord(iteration, residual_norm, step_norm, depth))
        y = next_y

    final = equation.evaluate(geometry, y)
    final_residual = np.asarray(final.residual, dtype=float)
    final_norm = float(np.linalg.norm(final_residual))
    geometry_digest = geometry_sha256(geometry)
    equation_digest = equation.fingerprint_sha256()
    root_hash = compute_root_hash(
        state_equation_id=equation.state_equation_id,
        geometry_digest=geometry_digest,
        equation_digest=equation_digest,
        scalar_digest=scalar_sha256,
        scalar_id=scalar_id,
        profile_id=profile_id,
        primal_tolerance=options.tolerance,
        root_context_id=root_context_id,
        y=y,
        source=final.source,
        field=final.field,
        residual=final_residual,
        initialization=initialization,
        converged=converged and final_norm <= options.tolerance,
        residual_norm=final_norm,
        iterations=tuple(records),
    )
    state = FixedPointState(
        state_equation_id=equation.state_equation_id,
        geometry_sha256=geometry_digest,
        equation_sha256=equation_digest,
        scalar_sha256=scalar_sha256,
        scalar_id=scalar_id,
        profile_id=profile_id,
        primal_tolerance=options.tolerance,
        root_context_id=root_context_id,
        initialization=initialization,
        converged=converged and final_norm <= options.tolerance,
        y=tuple(float(value) for value in y),
        source=final.source,
        field=final.field,
        actual_unmixed_residual=tuple(float(value) for value in final_residual),
        actual_unmixed_residual_norm=final_norm,
        iterations=tuple(records),
        root_hash=root_hash,
    )
    if require_convergence and not state.converged:
        raise FixedPointConvergenceError(state)
    return state


__all__ = [
    "FixedPointConvergenceError",
    "FixedPointOptions",
    "FixedPointState",
    "IterationRecord",
    "ROOT_EQUIVALENCE_ABSOLUTE_TOLERANCE",
    "ROOT_EQUIVALENCE_CONTRACT",
    "ScalarStateBinding",
    "compute_root_hash",
    "roots_numerically_equivalent",
    "solve_fixed_point",
]
