"""Cavity-independent Coulomb fitting of frozen PySCF electron densities.

This module is reference/data-generation infrastructure.  It neither defines
an ML source head nor reads a solvation target.  The fitted density is expanded
in a standard PySCF auxiliary basis and is constrained to preserve electron
count and electronic first moment before any cavity or continuum is chosen.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class AuxiliaryDensityProjection:
    """One Coulomb-fitted density with exact fitted-basis moment constraints."""

    raw_coefficients: np.ndarray
    constrained_coefficients: np.ndarray
    target_moments: np.ndarray
    constraint_matrix: np.ndarray
    metric_rank: int
    metric_condition_number: float
    raw_moment_max_absolute_error: float
    constrained_moment_max_absolute_error: float

    def __post_init__(self) -> None:
        raw = np.asarray(self.raw_coefficients, dtype=np.float64)
        if raw.ndim != 1 or raw.size < 1 or not np.all(np.isfinite(raw)):
            raise ValueError("raw_coefficients must be a finite nonempty vector.")
        size = int(raw.size)
        constrained = _readonly(
            self.constrained_coefficients,
            shape=(size,),
            name="constrained_coefficients",
        )
        target = _readonly(
            self.target_moments,
            shape=(4,),
            name="target_moments",
        )
        constraints = _readonly(
            self.constraint_matrix,
            shape=(4, size),
            name="constraint_matrix",
        )
        if (
            isinstance(self.metric_rank, bool)
            or not isinstance(self.metric_rank, int)
            or not 1 <= self.metric_rank <= size
        ):
            raise ValueError("metric_rank is outside the auxiliary space.")
        condition = float(self.metric_condition_number)
        raw_error = float(self.raw_moment_max_absolute_error)
        constrained_error = float(self.constrained_moment_max_absolute_error)
        if (
            not math.isfinite(condition)
            or condition < 1.0
            or not math.isfinite(raw_error)
            or raw_error < 0.0
            or not math.isfinite(constrained_error)
            or constrained_error < 0.0
        ):
            raise ValueError("projection diagnostics must be finite and nonnegative.")
        raw = np.ascontiguousarray(raw)
        raw.setflags(write=False)
        object.__setattr__(self, "raw_coefficients", raw)
        object.__setattr__(self, "constrained_coefficients", constrained)
        object.__setattr__(self, "target_moments", target)
        object.__setattr__(self, "constraint_matrix", constraints)
        object.__setattr__(self, "metric_condition_number", condition)
        object.__setattr__(self, "raw_moment_max_absolute_error", raw_error)
        object.__setattr__(
            self,
            "constrained_moment_max_absolute_error",
            constrained_error,
        )


def coulomb_fit_auxiliary_density(
    molecule: Any,
    density: object,
    *,
    metric_eigenvalue_relative_cutoff: float = 1.0e-12,
    moment_grid_level: int = 4,
) -> tuple[Any, AuxiliaryDensityProjection]:
    """Fit ``density`` in ``df.make_auxbasis(molecule)`` and preserve moments.

    The projection minimizes the Coulomb-metric density-fitting residual.  A
    four-constraint Schur correction then preserves the electron count and the
    three Cartesian electronic first moments.  The moment integrals of the
    auxiliary functions are evaluated on a deterministic PySCF atom-centred
    grid; their achieved closure is reported explicitly.
    """

    from pyscf import df, dft

    size = int(molecule.nao_nr())
    matrix = np.asarray(density, dtype=np.float64)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError("density must be a finite AO square matrix.")
    matrix = 0.5 * (matrix + matrix.T)
    cutoff = float(metric_eigenvalue_relative_cutoff)
    if not math.isfinite(cutoff) or not 0.0 < cutoff < 1.0:
        raise ValueError("metric eigenvalue cutoff must lie strictly in (0,1).")
    if (
        isinstance(moment_grid_level, bool)
        or not isinstance(moment_grid_level, int)
        or moment_grid_level < 1
    ):
        raise ValueError("moment_grid_level must be a positive integer.")

    auxiliary = df.addons.make_auxmol(
        molecule,
        auxbasis=df.make_auxbasis(molecule),
    )
    metric = np.asarray(auxiliary.intor("int2c2e"), dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (metric + metric.T))
    maximum = float(np.max(eigenvalues))
    retained = eigenvalues > maximum * cutoff
    if maximum <= 0.0 or int(np.count_nonzero(retained)) < 4:
        raise RuntimeError("auxiliary Coulomb metric has insufficient positive rank.")
    retained_values = eigenvalues[retained]
    retained_vectors = eigenvectors[:, retained]
    inverse = retained_vectors @ (
        retained_vectors.T / retained_values[:, None]
    )
    three_center = np.asarray(
        df.incore.aux_e2(molecule, auxiliary, intor="int3c2e", aosym="s1"),
        dtype=np.float64,
    )
    rhs = np.einsum("ijp,ij->p", three_center, matrix, optimize=True)
    raw = inverse @ rhs

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
    constraints = np.vstack((charge_moment, first_moment))
    residual = target - constraints @ raw
    schur = constraints @ inverse @ constraints.T
    schur_scale = max(1.0, float(np.linalg.norm(schur, ord=2)))
    if np.linalg.eigvalsh(0.5 * (schur + schur.T))[0] <= 1.0e-13 * schur_scale:
        raise RuntimeError("auxiliary moment-constraint Schur matrix is singular.")
    constrained = raw + inverse @ constraints.T @ np.linalg.solve(schur, residual)
    return auxiliary, AuxiliaryDensityProjection(
        raw_coefficients=raw,
        constrained_coefficients=constrained,
        target_moments=target,
        constraint_matrix=constraints,
        metric_rank=int(np.count_nonzero(retained)),
        metric_condition_number=float(
            np.max(retained_values) / np.min(retained_values)
        ),
        raw_moment_max_absolute_error=float(
            np.max(np.abs(constraints @ raw - target))
        ),
        constrained_moment_max_absolute_error=float(
            np.max(np.abs(constraints @ constrained - target))
        ),
    )


__all__ = ["AuxiliaryDensityProjection", "coulomb_fit_auxiliary_density"]
