"""Single conjugate Route-2 coupling operator contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .spaces import FieldDualSpace, SourceSpace


@runtime_checkable
class CouplingOperator(Protocol):
    scalar_id: str
    source_space: SourceSpace
    field_space: FieldDualSpace

    def apply_source(self, geometry: Any, source: np.ndarray) -> np.ndarray: ...
    def apply_adjoint(self, geometry: Any, surface_cotangent: np.ndarray) -> np.ndarray: ...
    def source_jvp(self, geometry: Any, source: np.ndarray, source_direction: np.ndarray) -> np.ndarray: ...
    def source_vjp(self, geometry: Any, source: np.ndarray, surface_cotangent: np.ndarray) -> np.ndarray: ...
    def coordinate_vjp(self, geometry: Any, source: np.ndarray, surface_cotangent: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class AdjointValidation:
    source_to_surface: float
    source_from_surface: float
    absolute_error: float
    tolerance: float
    passed: bool


def validate_adjoint_dot_product(
    operator: CouplingOperator,
    geometry: Any,
    source: object,
    surface_cotangent: object,
    *,
    atom_count: int,
    surface_pairing=None,
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance: float = 1.0e-12,
) -> AdjointValidation:
    """Validate ``<B c,sigma> = <c,B* sigma>_Q`` and fail closed."""

    if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
        raise ValueError("adjoint tolerances must be non-negative.")
    if not isinstance(operator.source_space, SourceSpace):
        raise TypeError("operator source_space must be a SourceSpace.")
    if not isinstance(operator.field_space, FieldDualSpace):
        raise TypeError("operator field_space must be a FieldDualSpace.")
    if operator.field_space.source_space != operator.source_space:
        raise ValueError("operator source_space and field_space source binding must match.")
    source_values = operator.source_space.validate(source, atom_count=atom_count)
    cotangent = np.asarray(surface_cotangent, dtype=float)
    if cotangent.size == 0 or not np.all(np.isfinite(cotangent)):
        raise ValueError("surface_cotangent must be finite and non-empty.")
    mapped = np.asarray(operator.apply_source(geometry, source_values), dtype=float)
    if mapped.shape != cotangent.shape or not np.all(np.isfinite(mapped)):
        raise ValueError("apply_source output must be finite and match surface_cotangent shape.")
    field = operator.field_space.validate(
        operator.apply_adjoint(geometry, cotangent), atom_count=atom_count,
        name="coupling adjoint field",
    )
    pairing = np.vdot if surface_pairing is None else surface_pairing
    lhs = float(pairing(mapped, cotangent))
    rhs = operator.field_space.pair(source_values, field, atom_count=atom_count)
    if not np.isfinite(lhs) or not np.isfinite(rhs):
        raise ValueError("adjoint pairings must be finite.")
    tolerance = absolute_tolerance + relative_tolerance * max(abs(lhs), abs(rhs))
    error = abs(lhs - rhs)
    return AdjointValidation(lhs, rhs, error, tolerance, error <= tolerance)


validate_coupling_adjoint = validate_adjoint_dot_product

__all__ = ["AdjointValidation", "CouplingOperator", "validate_adjoint_dot_product", "validate_coupling_adjoint"]
