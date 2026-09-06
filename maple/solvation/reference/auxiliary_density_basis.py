"""Conditioned auxiliary-density projection for standard-basis ladders.

The earlier :mod:`auxiliary_density` implementation is source-bound to a
completed ``make_auxbasis`` artifact and remains immutable.  This module uses
an explicitly supplied PySCF auxiliary basis and solves the exact moment-
constrained Coulomb fit in the constraint nullspace.  No Coulomb mode is
truncated or selected from geometry-dependent target performance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ConstraintRestrictedAuxiliaryDensityProjection:
    """One exact-constraint fit and its scale-invariant numerical certificate."""

    constrained_coefficients: np.ndarray
    target_moments: np.ndarray
    constraint_matrix: np.ndarray
    constraint_rank: int
    normalized_restricted_minimum_eigenvalue: float
    normalized_restricted_maximum_eigenvalue: float
    normalized_restricted_condition_number: float
    constrained_moment_max_absolute_error: float
    projected_stationarity_max_absolute_error: float

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.constrained_coefficients, dtype=np.float64)
        if (
            coefficients.ndim != 1
            or coefficients.size < 5
            or not np.all(np.isfinite(coefficients))
        ):
            raise ValueError(
                "constrained_coefficients must be a finite vector of length >=5."
            )
        size = int(coefficients.size)
        coefficients = _readonly(
            coefficients,
            shape=(size,),
            name="constrained_coefficients",
        )
        target = _readonly(self.target_moments, shape=(4,), name="target_moments")
        constraints = _readonly(
            self.constraint_matrix,
            shape=(4, size),
            name="constraint_matrix",
        )
        if self.constraint_rank != 4:
            raise ValueError("constraint_rank must equal four.")
        minimum = float(self.normalized_restricted_minimum_eigenvalue)
        maximum = float(self.normalized_restricted_maximum_eigenvalue)
        condition = float(self.normalized_restricted_condition_number)
        closure = float(self.constrained_moment_max_absolute_error)
        stationarity = float(self.projected_stationarity_max_absolute_error)
        if (
            not math.isfinite(minimum)
            or minimum <= 0.0
            or not math.isfinite(maximum)
            or maximum < minimum
            or not math.isfinite(condition)
            or condition < 1.0
            or not math.isfinite(closure)
            or closure < 0.0
            or not math.isfinite(stationarity)
            or stationarity < 0.0
        ):
            raise ValueError("projection diagnostics are invalid.")
        expected_condition = maximum / minimum
        if not np.isclose(condition, expected_condition, rtol=2.0e-12, atol=0.0):
            raise ValueError("restricted condition number is inconsistent.")
        object.__setattr__(self, "constrained_coefficients", coefficients)
        object.__setattr__(self, "target_moments", target)
        object.__setattr__(self, "constraint_matrix", constraints)
        object.__setattr__(
            self,
            "normalized_restricted_minimum_eigenvalue",
            minimum,
        )
        object.__setattr__(
            self,
            "normalized_restricted_maximum_eigenvalue",
            maximum,
        )
        object.__setattr__(
            self,
            "normalized_restricted_condition_number",
            condition,
        )
        object.__setattr__(
            self,
            "constrained_moment_max_absolute_error",
            closure,
        )
        object.__setattr__(
            self,
            "projected_stationarity_max_absolute_error",
            stationarity,
        )

    @property
    def metric_condition_number(self) -> float:
        """Compatibility name for the constraint-restricted normalized metric."""

        return self.normalized_restricted_condition_number


def coulomb_fit_auxiliary_density_in_basis(
    molecule: Any,
    density: object,
    *,
    auxiliary_basis: Mapping[str, object],
    moment_grid_level: int = 4,
    constraint_rank_relative_tolerance: float = 1.0e-12,
) -> tuple[Any, ConstraintRestrictedAuxiliaryDensityProjection]:
    """Fit one AO density without truncating the normalized Coulomb metric.

    With ``D=diag(J)``, the fitted coordinates are ``y=D^(1/2)c`` and the
    normalized Coulomb metric is ``Jhat=D^(-1/2) J D^(-1/2)``.  The four moment
    constraints become ``A y=t`` with ``A=C D^(-1/2)``.  A full QR supplies a
    basis ``Z`` of ``ker(A)`` and the minimizer is solved directly in that
    subspace.  The condition number reported here is therefore
    ``cond2(Z.T Jhat Z)`` and is invariant to diagonal rescaling of the basis.
    """

    from pyscf import df, dft
    from scipy import linalg

    if not isinstance(auxiliary_basis, Mapping) or not auxiliary_basis:
        raise ValueError("auxiliary_basis must be a nonempty element mapping.")
    size = int(molecule.nao_nr())
    matrix = np.asarray(density, dtype=np.float64)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError("density must be a finite AO square matrix.")
    matrix = np.ascontiguousarray(0.5 * (matrix + matrix.T))
    if (
        isinstance(moment_grid_level, bool)
        or not isinstance(moment_grid_level, int)
        or moment_grid_level < 1
    ):
        raise ValueError("moment_grid_level must be a positive integer.")
    rank_tolerance = float(constraint_rank_relative_tolerance)
    if not math.isfinite(rank_tolerance) or not 0.0 < rank_tolerance < 1.0:
        raise ValueError("constraint rank tolerance must lie strictly in (0,1).")

    auxiliary = df.addons.make_auxmol(
        molecule,
        auxbasis=dict(auxiliary_basis),
    )
    metric = np.asarray(auxiliary.intor("int2c2e"), dtype=np.float64)
    metric = np.ascontiguousarray(0.5 * (metric + metric.T))
    diagonal = np.diag(metric)
    if np.any(diagonal <= 0.0) or not np.all(np.isfinite(diagonal)):
        raise RuntimeError("auxiliary Coulomb metric has a nonpositive diagonal.")
    inverse_sqrt_diagonal = 1.0 / np.sqrt(diagonal)
    normalized_metric = (
        inverse_sqrt_diagonal[:, None]
        * metric
        * inverse_sqrt_diagonal[None, :]
    )
    normalized_metric = 0.5 * (normalized_metric + normalized_metric.T)

    grids = dft.gen_grid.Grids(auxiliary)
    grids.level = moment_grid_level
    grids.build(with_non0tab=False)
    auxiliary_size = int(auxiliary.nao_nr())
    charge_moment = np.zeros(auxiliary_size, dtype=np.float64)
    first_moment = np.zeros((3, auxiliary_size), dtype=np.float64)
    for start in range(0, len(grids.coords), 30000):
        stop = min(start + 30000, len(grids.coords))
        coordinates = np.asarray(grids.coords[start:stop], dtype=np.float64)
        weights = np.asarray(grids.weights[start:stop], dtype=np.float64)
        ao = np.asarray(dft.numint.eval_ao(auxiliary, coordinates), dtype=np.float64)
        charge_moment += np.einsum("g,gp->p", weights, ao, optimize=True)
        first_moment += np.einsum(
            "g,gx,gp->xp",
            weights,
            coordinates,
            ao,
            optimize=True,
        )
    constraints = np.vstack((charge_moment, first_moment))
    scaled_constraints = constraints * inverse_sqrt_diagonal[None, :]
    singular_values = linalg.svdvals(scaled_constraints)
    constraint_rank = int(
        np.count_nonzero(singular_values > singular_values[0] * rank_tolerance)
    )
    if constraint_rank != 4:
        raise RuntimeError("scaled auxiliary moment constraints do not have rank four.")
    q_matrix, _r_matrix = linalg.qr(
        scaled_constraints.T,
        mode="full",
        pivoting=False,
        check_finite=False,
    )
    nullspace = np.ascontiguousarray(q_matrix[:, 4:])
    restricted_metric = nullspace.T @ normalized_metric @ nullspace
    restricted_metric = 0.5 * (restricted_metric + restricted_metric.T)
    eigenvalues = linalg.eigvalsh(
        restricted_metric,
        check_finite=False,
        driver="evr",
    )
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    if minimum <= 0.0 or not np.all(np.isfinite(eigenvalues)):
        raise RuntimeError("constraint-restricted normalized metric is not positive.")
    condition = maximum / minimum

    overlap = np.asarray(molecule.intor_symmetric("int1e_ovlp"), dtype=np.float64)
    position = np.asarray(
        molecule.intor_symmetric("int1e_r", comp=3),
        dtype=np.float64,
    )
    target = np.concatenate(
        (
            [float(np.einsum("ij,ji", matrix, overlap, optimize=True))],
            np.einsum("xij,ji->x", position, matrix, optimize=True),
        )
    )
    constraint_gram = scaled_constraints @ scaled_constraints.T
    particular = scaled_constraints.T @ linalg.solve(
        constraint_gram,
        target,
        assume_a="pos",
        check_finite=False,
    )
    three_center = np.asarray(
        df.incore.aux_e2(molecule, auxiliary, intor="int3c2e", aosym="s1"),
        dtype=np.float64,
    )
    rhs = np.einsum("ijp,ij->p", three_center, matrix, optimize=True)
    normalized_rhs = inverse_sqrt_diagonal * rhs
    reduced_rhs = nullspace.T @ (
        normalized_rhs - normalized_metric @ particular
    )
    reduced_solution = linalg.solve(
        restricted_metric,
        reduced_rhs,
        assume_a="pos",
        check_finite=False,
    )
    normalized_coefficients = particular + nullspace @ reduced_solution
    coefficients = inverse_sqrt_diagonal * normalized_coefficients
    closure = float(np.max(np.abs(constraints @ coefficients - target)))
    stationarity = float(
        np.max(
            np.abs(
                nullspace.T
                @ (normalized_metric @ normalized_coefficients - normalized_rhs)
            )
        )
    )
    return auxiliary, ConstraintRestrictedAuxiliaryDensityProjection(
        constrained_coefficients=coefficients,
        target_moments=target,
        constraint_matrix=constraints,
        constraint_rank=constraint_rank,
        normalized_restricted_minimum_eigenvalue=minimum,
        normalized_restricted_maximum_eigenvalue=maximum,
        normalized_restricted_condition_number=condition,
        constrained_moment_max_absolute_error=closure,
        projected_stationarity_max_absolute_error=stationarity,
    )


__all__ = [
    "ConstraintRestrictedAuxiliaryDensityProjection",
    "coulomb_fit_auxiliary_density_in_basis",
]
