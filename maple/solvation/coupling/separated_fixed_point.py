"""Deterministic root state for categorically distinct source and field spaces.

The established :mod:`fixed_point` state predates the MACE-POLAR 4-channel
source / 8-channel receiver split and intentionally requires equal source and
field shapes.  This module reuses the same numerical Picard/Anderson kernel,
but records the separated boundary state and permits rectangular coupling.
The root is a property of the state equation; an operational energy ledger is
bound afterwards and must not be smuggled into the root identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

import numpy as np

from .fixed_point import (
    FixedPointOptions,
    IterationRecord,
    ROOT_EQUIVALENCE_ABSOLUTE_TOLERANCE,
    array_content_sha256,
    iterate_reduced_fixed_point,
)
from .separated_state import SeparatedReducedStateEquation
from .state_equation import geometry_sha256


SEPARATED_ROOT_STATE_CONTRACT_ID = "route2-separated-root-state-v1"


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _digest(value: object, *, name: str) -> str:
    result = _text(value, name=name).lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal digits.")
    return result


def _finite_array(
    values: object, *, ndim: int, name: str, nonempty: bool = True
) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != ndim or (nonempty and result.size == 0) or not np.all(
        np.isfinite(result)
    ):
        raise ValueError(f"{name} must be a non-empty finite {ndim}-D array.")
    return result


def compute_separated_root_hash(
    *,
    state_equation_id: str,
    geometry_digest: str,
    equation_digest: str,
    source_space_digest: str,
    receiver_space_digest: str,
    continuum_configuration_digest: str,
    primal_tolerance: float,
    root_context_id: str,
    initialization: str,
    converged: bool,
    y: object,
    source: object,
    boundary_state: object,
    field: object,
    residual: object,
    residual_norm: float,
    iterations: tuple[IterationRecord, ...],
) -> str:
    payload = {
        "contract_id": SEPARATED_ROOT_STATE_CONTRACT_ID,
        "state_equation_id": state_equation_id,
        "geometry_sha256": geometry_digest,
        "equation_sha256": equation_digest,
        "source_space_sha256": source_space_digest,
        "receiver_space_sha256": receiver_space_digest,
        "continuum_configuration_sha256": continuum_configuration_digest,
        "primal_tolerance": primal_tolerance,
        "root_context_id": root_context_id,
        "initialization": initialization,
        "converged": converged,
        "y_content_sha256": array_content_sha256(y),
        "source_content_sha256": array_content_sha256(source),
        "boundary_state_content_sha256": array_content_sha256(boundary_state),
        "field_content_sha256": array_content_sha256(field),
        "residual_content_sha256": array_content_sha256(residual),
        "actual_unmixed_residual_norm": residual_norm,
        "iterations": [
            {
                "iteration": record.iteration,
                "actual_unmixed_residual_norm": (
                    record.actual_unmixed_residual_norm
                ),
                "step_norm": record.step_norm,
                "history_depth": record.history_depth,
            }
            for record in iterations
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SeparatedFixedPointState:
    """Immutable converged-or-failed state with distinct ``C``, ``Sigma``, ``U``."""

    state_equation_id: str
    geometry_sha256: str
    equation_sha256: str
    source_space_sha256: str
    receiver_space_sha256: str
    continuum_configuration_sha256: str
    primal_tolerance: float
    root_context_id: str
    initialization: Literal["cold", "warm"]
    converged: bool
    y: tuple[float, ...]
    source: tuple[tuple[float, ...], ...]
    boundary_state: tuple[float, ...]
    field: tuple[tuple[float, ...], ...]
    actual_unmixed_residual: tuple[float, ...]
    actual_unmixed_residual_norm: float
    iterations: tuple[IterationRecord, ...]
    root_hash: str
    contract_id: str = SEPARATED_ROOT_STATE_CONTRACT_ID

    def __post_init__(self) -> None:
        _text(self.state_equation_id, name="state_equation_id")
        for name in (
            "geometry_sha256",
            "equation_sha256",
            "source_space_sha256",
            "receiver_space_sha256",
            "continuum_configuration_sha256",
            "root_hash",
        ):
            _digest(getattr(self, name), name=name)
        _text(self.root_context_id, name="root_context_id")
        if self.contract_id != SEPARATED_ROOT_STATE_CONTRACT_ID:
            raise ValueError("unsupported separated root-state contract.")
        if not np.isfinite(self.primal_tolerance) or self.primal_tolerance <= 0.0:
            raise ValueError("primal_tolerance must be finite and positive.")
        if self.initialization not in ("cold", "warm"):
            raise ValueError("initialization must be 'cold' or 'warm'.")
        if type(self.converged) is not bool:
            raise TypeError("converged must be exactly bool.")

        y = _finite_array(self.y, ndim=1, name="y")
        source = _finite_array(self.source, ndim=2, name="source")
        boundary = _finite_array(
            self.boundary_state, ndim=1, name="boundary_state"
        )
        field = _finite_array(self.field, ndim=2, name="field")
        residual = _finite_array(
            self.actual_unmixed_residual,
            ndim=1,
            name="actual_unmixed_residual",
        )
        if residual.shape != y.shape:
            raise ValueError("actual_unmixed_residual must match y.")
        if source.shape[0] != field.shape[0]:
            raise ValueError("source and field must contain the same atom count.")
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
            raise ValueError("iterations must contain IterationRecord values.")
        expected = compute_separated_root_hash(
            state_equation_id=self.state_equation_id,
            geometry_digest=self.geometry_sha256,
            equation_digest=self.equation_sha256,
            source_space_digest=self.source_space_sha256,
            receiver_space_digest=self.receiver_space_sha256,
            continuum_configuration_digest=self.continuum_configuration_sha256,
            primal_tolerance=self.primal_tolerance,
            root_context_id=self.root_context_id,
            initialization=self.initialization,
            converged=self.converged,
            y=y,
            source=source,
            boundary_state=boundary,
            field=field,
            residual=residual,
            residual_norm=norm,
            iterations=self.iterations,
        )
        if self.root_hash != expected:
            raise ValueError("root_hash does not match the separated state contents.")

    @staticmethod
    def _readonly(values: object) -> np.ndarray:
        result = np.asarray(values, dtype=float)
        result.setflags(write=False)
        return result

    def y_array(self) -> np.ndarray:
        return self._readonly(self.y)

    def source_array(self) -> np.ndarray:
        return self._readonly(self.source)

    def boundary_state_array(self) -> np.ndarray:
        return self._readonly(self.boundary_state)

    def field_array(self) -> np.ndarray:
        return self._readonly(self.field)


class SeparatedFixedPointConvergenceError(RuntimeError):
    def __init__(self, state: SeparatedFixedPointState) -> None:
        self.state = state
        super().__init__(
            "Separated state equation did not converge in "
            f"{len(state.iterations)} iterations; actual unmixed reduced "
            f"residual={state.actual_unmixed_residual_norm:.6e}."
        )


def solve_separated_fixed_point(
    equation: SeparatedReducedStateEquation,
    geometry: Any,
    *,
    root_context_id: str,
    initial_y: object | None = None,
    options: FixedPointOptions = FixedPointOptions(),
    require_convergence: bool = True,
) -> SeparatedFixedPointState:
    """Solve ``y = T+ M(R, continuum(c_ref+T y))`` deterministically."""

    if not isinstance(equation, SeparatedReducedStateEquation):
        raise TypeError("equation must satisfy SeparatedReducedStateEquation.")
    context = _text(root_context_id, name="root_context_id")
    if not isinstance(options, FixedPointOptions):
        raise TypeError("options must be FixedPointOptions.")
    initialization: Literal["cold", "warm"] = "cold" if initial_y is None else "warm"
    y0 = (
        np.zeros(equation.reduced_dimension, dtype=float)
        if initial_y is None
        else np.asarray(initial_y, dtype=float)
    )
    y, loop_converged, records = iterate_reduced_fixed_point(
        mapping=lambda value: equation.mapping(geometry, value),
        residual=lambda value: equation.residual(geometry, value),
        dimension=equation.reduced_dimension,
        initial_y=y0,
        options=options,
    )
    residual = np.asarray(equation.residual(geometry, y), dtype=float)
    residual_norm = float(np.linalg.norm(residual))
    source = equation.source(y)
    boundary = equation.boundary_state(y)
    field = equation.field(y)
    converged = loop_converged and residual_norm <= options.tolerance
    geometry_digest = geometry_sha256(geometry)
    equation_digest = equation.fingerprint_sha256()
    source_space_digest = equation.source_space.metadata_hash()
    receiver_space_digest = equation.receiver_space.metadata_hash()
    continuum_digest = equation.continuum.configuration_sha256()
    root_hash = compute_separated_root_hash(
        state_equation_id=equation.state_equation_id,
        geometry_digest=geometry_digest,
        equation_digest=equation_digest,
        source_space_digest=source_space_digest,
        receiver_space_digest=receiver_space_digest,
        continuum_configuration_digest=continuum_digest,
        primal_tolerance=options.tolerance,
        root_context_id=context,
        initialization=initialization,
        converged=converged,
        y=y,
        source=source,
        boundary_state=boundary,
        field=field,
        residual=residual,
        residual_norm=residual_norm,
        iterations=records,
    )
    state = SeparatedFixedPointState(
        state_equation_id=equation.state_equation_id,
        geometry_sha256=geometry_digest,
        equation_sha256=equation_digest,
        source_space_sha256=source_space_digest,
        receiver_space_sha256=receiver_space_digest,
        continuum_configuration_sha256=continuum_digest,
        primal_tolerance=options.tolerance,
        root_context_id=context,
        initialization=initialization,
        converged=converged,
        y=tuple(float(value) for value in y),
        source=tuple(tuple(float(value) for value in row) for row in source),
        boundary_state=tuple(float(value) for value in boundary),
        field=tuple(tuple(float(value) for value in row) for row in field),
        actual_unmixed_residual=tuple(float(value) for value in residual),
        actual_unmixed_residual_norm=residual_norm,
        iterations=records,
        root_hash=root_hash,
    )
    if require_convergence and not state.converged:
        raise SeparatedFixedPointConvergenceError(state)
    return state


def separated_roots_numerically_equivalent(
    first: SeparatedFixedPointState,
    second: SeparatedFixedPointState,
) -> bool:
    """Compare two valid states under the fixed numerical root contract."""

    if not isinstance(first, SeparatedFixedPointState) or not isinstance(
        second, SeparatedFixedPointState
    ):
        raise TypeError("root equivalence requires separated root states.")
    identities = (
        "state_equation_id",
        "geometry_sha256",
        "equation_sha256",
        "source_space_sha256",
        "receiver_space_sha256",
        "continuum_configuration_sha256",
        "root_context_id",
        "primal_tolerance",
    )
    if any(getattr(first, name) != getattr(second, name) for name in identities):
        return False
    if (
        not first.converged
        or not second.converged
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
            (first.boundary_state, second.boundary_state),
            (first.field, second.field),
            (first.actual_unmixed_residual, second.actual_unmixed_residual),
        )
    )


__all__ = [
    "SEPARATED_ROOT_STATE_CONTRACT_ID",
    "SeparatedFixedPointConvergenceError",
    "SeparatedFixedPointState",
    "compute_separated_root_hash",
    "separated_roots_numerically_equivalent",
    "solve_separated_fixed_point",
]
