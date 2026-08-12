"""Implicit reduced adjoint with fail-closed true-residual verification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .linearization import ReducedLinearization


@dataclass(frozen=True)
class AdjointOptions:
    relative_tolerance: float = 1.0e-12
    absolute_tolerance: float = 1.0e-14
    max_iterations: int = 200
    restart: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.relative_tolerance) or self.relative_tolerance < 0.0:
            raise ValueError("relative_tolerance must be finite and non-negative.")
        if not np.isfinite(self.absolute_tolerance) or self.absolute_tolerance < 0.0:
            raise ValueError("absolute_tolerance must be finite and non-negative.")
        if self.relative_tolerance == 0.0 and self.absolute_tolerance == 0.0:
            raise ValueError("At least one adjoint tolerance must be positive.")
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer.")
        if self.restart is not None and (type(self.restart) is not int or self.restart < 1):
            raise ValueError("restart must be None or a positive integer.")


@dataclass(frozen=True)
class AdjointResult:
    solution: tuple[float, ...]
    iterations: int
    gmres_info: int
    true_residual_norm: float
    acceptance_tolerance: float
    converged: bool

    def solution_array(self) -> np.ndarray:
        result = np.asarray(self.solution, dtype=float)
        result.setflags(write=False)
        return result


class AdjointConvergenceError(RuntimeError):
    def __init__(self, result: AdjointResult):
        self.result = result
        super().__init__(
            "Reduced adjoint failed true-residual verification: "
            f"info={result.gmres_info}, residual={result.true_residual_norm:.6e}, "
            f"tolerance={result.acceptance_tolerance:.6e}."
        )


def solve_reduced_adjoint(
    linearization: ReducedLinearization,
    scalar_reduced_gradient: object,
    *,
    options: AdjointOptions = AdjointOptions(),
) -> AdjointResult:
    """Solve ``r_y.T lambda = Phi_y.T`` and verify the actual residual."""

    if not isinstance(options, AdjointOptions):
        raise TypeError("options must be AdjointOptions.")
    rhs = np.asarray(scalar_reduced_gradient, dtype=float)
    if rhs.shape != (linearization.dimension,) or not np.all(np.isfinite(rhs)):
        raise ValueError(
            f"scalar_reduced_gradient must be finite with shape ({linearization.dimension},)."
        )
    operator = LinearOperator(
        (linearization.dimension, linearization.dimension),
        # SciPy's Arnoldi implementation updates the returned work vector in
        # place, while public contract arrays are read-only.
        matvec=lambda vector: np.array(linearization.vjp(vector), dtype=float, copy=True),
        dtype=float,
    )
    iterations = 0

    def callback(_residual: object) -> None:
        nonlocal iterations
        iterations += 1

    solution, info = gmres(
        operator,
        rhs,
        rtol=options.relative_tolerance,
        atol=options.absolute_tolerance,
        restart=options.restart,
        maxiter=options.max_iterations,
        callback=callback,
        callback_type="pr_norm",
    )
    solution = np.asarray(solution, dtype=float)
    true_residual = np.asarray(linearization.vjp(solution), dtype=float) - rhs
    true_norm = float(np.linalg.norm(true_residual))
    acceptance = max(
        options.absolute_tolerance,
        options.relative_tolerance * float(np.linalg.norm(rhs)),
    )
    converged = info == 0 and np.all(np.isfinite(solution)) and true_norm <= acceptance
    result = AdjointResult(
        solution=tuple(float(value) for value in solution),
        iterations=iterations,
        gmres_info=int(info),
        true_residual_norm=true_norm,
        acceptance_tolerance=acceptance,
        converged=converged,
    )
    if not converged:
        raise AdjointConvergenceError(result)
    return result


__all__ = [
    "AdjointConvergenceError",
    "AdjointOptions",
    "AdjointResult",
    "solve_reduced_adjoint",
]
