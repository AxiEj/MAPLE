"""Matrix-free linearization of the Route-2 mutual-polarization residual.

This module contains fixed-geometry response algebra only.  It deliberately
does not expose a solvent force: cavity, PCM-boundary, CDS, and nuclear
coordinate derivatives are separate terms in the total Route-2 derivative.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Protocol

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


class ReactionFieldLinearMap(Protocol):
    """Fixed-cavity density-to-node-field map and its discrete adjoint."""

    atom_count: int

    def apply(self, density_direction: np.ndarray) -> np.ndarray:
        """Map an ``(n_atoms, 4)`` density direction to a node-field direction."""
        ...

    def adjoint(self, field_cotangent: np.ndarray) -> np.ndarray:
        """Map an ``(n_atoms, 4)`` node-field cotangent back to density space."""
        ...


class DensityResponseLinearization(Protocol):
    """MACE density response to the external node field at one fixed state."""

    def jvp(self, field_direction: np.ndarray) -> np.ndarray:
        """Apply the field-to-density Jacobian."""
        ...

    def vjp(self, density_cotangent: np.ndarray) -> np.ndarray:
        """Apply the discrete adjoint of the field-to-density Jacobian."""
        ...


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


class NeutralDensityCoordinates:
    """Orthonormal coordinates for the zero-total-monopole tangent space."""

    def __init__(self, atom_count: int, *, neutral_tolerance: float = 1.0e-10):
        if atom_count <= 0:
            raise ValueError("Neutral density coordinates require atoms.")
        if neutral_tolerance <= 0.0:
            raise ValueError("Neutral tangent tolerance must be positive.")
        self.atom_count = int(atom_count)
        self.neutral_tolerance = float(neutral_tolerance)
        self._charge_basis = self._build_charge_basis(self.atom_count)
        self.dimension = 4 * self.atom_count - 1

    @staticmethod
    def _build_charge_basis(atom_count: int) -> np.ndarray:
        basis = np.zeros((atom_count, max(atom_count - 1, 0)), dtype=float)
        for column in range(atom_count - 1):
            count = column + 1
            normalization = np.sqrt(count * (count + 1))
            basis[:count, column] = 1.0 / normalization
            basis[count, column] = -count / normalization
        return basis

    def expand(self, coordinates: np.ndarray) -> np.ndarray:
        """Map a reduced vector isometrically into an ``(n_atoms, 4)`` tangent."""

        vector = np.asarray(coordinates, dtype=float)
        if vector.shape != (self.dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError(
                "Neutral density coordinate vector must be finite with shape "
                f"({self.dimension},); received {vector.shape}."
            )
        charge_dimension = self.atom_count - 1
        values = np.empty((self.atom_count, 4), dtype=float)
        values[:, 0] = self._charge_basis @ vector[:charge_dimension]
        values[:, 1:] = vector[charge_dimension:].reshape(self.atom_count, 3)
        return values

    def reduce(self, values: np.ndarray) -> np.ndarray:
        """Map a neutral tangent into its orthonormal reduced coordinates."""

        tangent = _validated_block(
            values,
            atom_count=self.atom_count,
            name="density tangent",
        )
        _require_neutral_density_tangent(
            tangent,
            name="density tangent",
            tolerance=self.neutral_tolerance,
        )
        return np.concatenate(
            (
                self._charge_basis.T @ tangent[:, 0],
                tangent[:, 1:].reshape(-1),
            )
        )


@dataclass(frozen=True)
class AdjointSolveResult:
    """Verified solution of the neutral-subspace Route-2 adjoint equation."""

    solution: np.ndarray
    residual_callback_count: int
    operator_applications: int
    residual_norm: float
    relative_residual: float
    restart_size: int
    maximum_inner_iterations: int
    method: str = "gmres"


def solve_adjoint(
    linearization: UnmixedDensityResidualLinearization,
    right_hand_side: np.ndarray,
    *,
    relative_tolerance: float = 1.0e-8,
    absolute_tolerance: float = 1.0e-10,
    restart: int | None = None,
    max_iterations: int = 100,
) -> AdjointSolveResult:
    """Solve ``J_c R0* lambda = rhs`` by matrix-free GMRES.

    The Helmert charge basis removes the forbidden uniform-charge mode while
    preserving the Euclidean discrete pairing.  By default the restart size
    spans the reduced density space, capped by ``max_iterations``.  SciPy's
    legacy callback-counting mode makes ``max_iterations`` an actual bound on
    inner Krylov iterations rather than on restart cycles.  The solve fails
    closed on Krylov non-convergence or when a fresh post-solve residual check
    does not satisfy the requested tolerance.
    """

    if relative_tolerance <= 0.0:
        raise ValueError("Adjoint relative tolerance must be positive.")
    if absolute_tolerance < 0.0:
        raise ValueError("Adjoint absolute tolerance cannot be negative.")
    if restart is not None and restart <= 0:
        raise ValueError("Adjoint GMRES restart must be positive.")
    if max_iterations <= 0:
        raise ValueError("Adjoint maximum iterations must be positive.")

    coordinates = NeutralDensityCoordinates(
        linearization.atom_count,
        neutral_tolerance=linearization.neutral_tolerance,
    )
    restart_size = min(
        coordinates.dimension if restart is None else restart,
        coordinates.dimension,
        max_iterations,
    )
    rhs = coordinates.reduce(right_hand_side)
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        return AdjointSolveResult(
            solution=coordinates.expand(np.zeros_like(rhs)),
            residual_callback_count=0,
            operator_applications=0,
            residual_norm=0.0,
            relative_residual=0.0,
            restart_size=restart_size,
            maximum_inner_iterations=max_iterations,
        )

    iterations = 0
    operator_applications = 0

    def matvec(vector: np.ndarray) -> np.ndarray:
        nonlocal operator_applications
        operator_applications += 1
        return coordinates.reduce(
            linearization.vjp(coordinates.expand(vector))
        )

    def callback(_residual) -> None:
        nonlocal iterations
        iterations += 1

    linear_operator_factory: Any = LinearOperator
    operator = linear_operator_factory(
        shape=(coordinates.dimension, coordinates.dimension),
        matvec=matvec,
        dtype=np.float64,
    )
    kwargs = {
        "restart": restart_size,
        "maxiter": max_iterations,
        "callback": callback,
        "atol": absolute_tolerance,
    }
    gmres_parameters = inspect.signature(gmres).parameters
    if "rtol" in gmres_parameters:
        kwargs["rtol"] = relative_tolerance
    else:  # SciPy < 1.14
        kwargs["tol"] = relative_tolerance
    if "callback_type" in gmres_parameters:
        kwargs["callback_type"] = "legacy"

    solution, info = gmres(operator, rhs, **kwargs)
    residual = matvec(solution) - rhs
    residual_norm = float(np.linalg.norm(residual))
    relative_residual = residual_norm / rhs_norm
    threshold = max(absolute_tolerance, relative_tolerance * rhs_norm)
    if info != 0 or residual_norm > threshold + 1.0e-13:
        raise RuntimeError(
            "Route-2 adjoint GMRES did not satisfy the requested tolerance "
            f"(info={info}, residual={residual_norm:.3e}, "
            f"threshold={threshold:.3e})."
        )
    return AdjointSolveResult(
        solution=coordinates.expand(solution),
        residual_callback_count=iterations,
        operator_applications=operator_applications,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        restart_size=restart_size,
        maximum_inner_iterations=max_iterations,
    )


__all__ = [
    "AdjointSolveResult",
    "DensityResponseLinearization",
    "NeutralDensityCoordinates",
    "ReactionFieldLinearMap",
    "UnmixedDensityResidualLinearization",
    "project_neutral_density_tangent",
    "solve_adjoint",
]
