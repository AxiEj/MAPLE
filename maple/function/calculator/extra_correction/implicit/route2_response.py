"""Matrix-free linearization of the Route-2 mutual-polarization residual.

This module contains fixed-geometry response algebra only.  It deliberately
does not expose a solvent force: cavity, PCM-boundary, CDS, and nuclear
coordinate derivatives are separate terms in the total Route-2 derivative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ReactionFieldLinearMap(Protocol):
    """Fixed-cavity density-to-node-field map and its discrete adjoint."""

    def apply(self, density_direction: np.ndarray) -> np.ndarray:
        """Map an ``(n_atoms, 4)`` density direction to a node-field direction."""

    def adjoint(self, field_cotangent: np.ndarray) -> np.ndarray:
        """Map an ``(n_atoms, 4)`` node-field cotangent back to density space."""


class DensityResponseLinearization(Protocol):
    """MACE density response to the external node field at one fixed state."""

    def jvp(self, field_direction: np.ndarray) -> np.ndarray:
        """Apply the field-to-density Jacobian."""

    def vjp(self, density_cotangent: np.ndarray) -> np.ndarray:
        """Apply the discrete adjoint of the field-to-density Jacobian."""


def _validated_block(
    values: np.ndarray,
    *,
    atom_count: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    expected_shape = (atom_count, 4)
    if array.shape != expected_shape or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must be finite with shape {expected_shape}; "
            f"received {array.shape}."
        )
    return array


def project_neutral_density_tangent(values: np.ndarray) -> np.ndarray:
    """Orthogonally project the monopole block onto zero total charge.

    The three real-spherical ``l=1`` components are unchanged.  The operation
    is the Euclidean projector used to define the discrete neutral tangent
    space; it is not a modification of a converged physical density.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 4 or not np.all(np.isfinite(array)):
        raise ValueError(
            "Density tangent values must be finite with shape (n_atoms, 4)."
        )
    if array.shape[0] == 0:
        raise ValueError("Density tangent values require at least one atom.")
    projected = array.copy()
    projected[:, 0] -= float(np.mean(projected[:, 0]))
    return projected


def _require_neutral_density_tangent(
    values: np.ndarray,
    *,
    name: str,
    tolerance: float,
) -> None:
    charge_sum = abs(float(np.sum(values[:, 0])))
    scale = max(1.0, float(np.linalg.norm(values[:, 0], ord=1)))
    if charge_sum > tolerance * scale:
        raise ValueError(
            f"{name} must lie in the neutral density tangent space "
            f"(monopole sum={float(np.sum(values[:, 0])):.6e})."
        )


@dataclass(frozen=True)
class UnmixedDensityResidualLinearization:
    """Linearization of ``R(c)=Pi0[c-M(P(c))]`` at fixed geometry/cavity.

    ``P`` is the PCM reaction-field map and ``M`` is the MACE-POLAR density
    response.  Numerical SCF mixing is intentionally absent because it is a
    root-finding choice, not part of the converged physical residual.

    Both operands live in the neutral density tangent space.  Node fields use
    the external Cartesian order ``[V, dV/dx, dV/dy, dV/dz]``; the MACE
    adapter owns any internal e3nn permutation and unit conversion.
    """

    atom_count: int
    reaction_field: ReactionFieldLinearMap
    density_response: DensityResponseLinearization
    neutral_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        if self.atom_count <= 0:
            raise ValueError("Route-2 residual linearization requires atoms.")
        if self.neutral_tolerance <= 0.0:
            raise ValueError("Neutral tangent tolerance must be positive.")

    def jvp(self, density_direction: np.ndarray) -> np.ndarray:
        """Apply ``Pi0 (I - J_M J_P)`` without forming a dense Jacobian."""

        direction = _validated_block(
            density_direction,
            atom_count=self.atom_count,
            name="density_direction",
        )
        _require_neutral_density_tangent(
            direction,
            name="density_direction",
            tolerance=self.neutral_tolerance,
        )
        field_direction = _validated_block(
            self.reaction_field.apply(direction),
            atom_count=self.atom_count,
            name="reaction-field JVP",
        )
        response_direction = _validated_block(
            self.density_response.jvp(field_direction),
            atom_count=self.atom_count,
            name="MACE density-response JVP",
        )
        return project_neutral_density_tangent(direction - response_direction)

    def vjp(self, density_cotangent: np.ndarray) -> np.ndarray:
        """Apply ``Pi0 (I - J_P* J_M*)`` in the discrete neutral subspace."""

        cotangent = _validated_block(
            density_cotangent,
            atom_count=self.atom_count,
            name="density_cotangent",
        )
        _require_neutral_density_tangent(
            cotangent,
            name="density_cotangent",
            tolerance=self.neutral_tolerance,
        )
        field_cotangent = _validated_block(
            self.density_response.vjp(cotangent),
            atom_count=self.atom_count,
            name="MACE density-response VJP",
        )
        response_cotangent = _validated_block(
            self.reaction_field.adjoint(field_cotangent),
            atom_count=self.atom_count,
            name="reaction-field VJP",
        )
        return project_neutral_density_tangent(cotangent - response_cotangent)


__all__ = [
    "DensityResponseLinearization",
    "ReactionFieldLinearMap",
    "UnmixedDensityResidualLinearization",
    "project_neutral_density_tangent",
]
