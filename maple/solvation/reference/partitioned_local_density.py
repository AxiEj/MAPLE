"""Unique local density labels from a smooth stockholder partition.

Each reference atomic target is ``w_A(r) * rho_QM(r)``.  It is fitted only in
the auxiliary functions centred on atom ``A`` using that atom's one-centre
overlap metric.  A final four-constraint correction uses the block-diagonal
collection of those local metrics; no molecular Coulomb inverse, pivot, or
mode truncation is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from maple.solvation.coupling.stockholder_partition import (
    PromolecularStockholderPartition,
)


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PartitionedLocalDensityFit:
    coefficients: np.ndarray
    local_target_electron_counts: np.ndarray
    local_metric_condition_numbers: tuple[float, ...]
    exact_electron_count: float
    exact_electronic_first_moment_bohr_e: np.ndarray
    grid_electron_count: float
    grid_electronic_first_moment_bohr_e: np.ndarray
    constraint_max_absolute_residual: float
    constraint_gram_condition_number: float
    auxiliary_dimension: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.auxiliary_dimension, bool)
            or not isinstance(self.auxiliary_dimension, int)
            or self.auxiliary_dimension < 1
        ):
            raise ValueError("auxiliary_dimension must be positive.")
        coefficients = _readonly(
            self.coefficients,
            shape=(self.auxiliary_dimension,),
            name="coefficients",
        )
        local_counts = np.asarray(
            self.local_target_electron_counts,
            dtype=np.float64,
        )
        if (
            local_counts.ndim != 1
            or local_counts.size != len(self.local_metric_condition_numbers)
            or not np.all(np.isfinite(local_counts))
        ):
            raise ValueError("local target electron counts are invalid.")
        local_counts = np.ascontiguousarray(local_counts)
        local_counts.setflags(write=False)
        exact_moment = _readonly(
            self.exact_electronic_first_moment_bohr_e,
            shape=(3,),
            name="exact electronic first moment",
        )
        grid_moment = _readonly(
            self.grid_electronic_first_moment_bohr_e,
            shape=(3,),
            name="grid electronic first moment",
        )
        exact_count = float(self.exact_electron_count)
        grid_count = float(self.grid_electron_count)
        residual = float(self.constraint_max_absolute_residual)
        gram_condition = float(self.constraint_gram_condition_number)
        conditions = tuple(float(value) for value in self.local_metric_condition_numbers)
        if (
            not math.isfinite(exact_count)
            or exact_count <= 0.0
            or not math.isfinite(grid_count)
            or grid_count <= 0.0
            or not math.isfinite(residual)
            or residual < 0.0
            or not math.isfinite(gram_condition)
            or gram_condition < 1.0
            or not conditions
            or any(not math.isfinite(value) or value < 1.0 for value in conditions)
        ):
            raise ValueError("partitioned local-density diagnostics are invalid.")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "local_target_electron_counts", local_counts)
        object.__setattr__(
            self,
            "local_metric_condition_numbers",
            conditions,
        )
        object.__setattr__(self, "exact_electron_count", exact_count)
        object.__setattr__(self, "grid_electron_count", grid_count)
        object.__setattr__(
            self,
            "exact_electronic_first_moment_bohr_e",
            exact_moment,
        )
        object.__setattr__(
            self,
            "grid_electronic_first_moment_bohr_e",
            grid_moment,
        )
        object.__setattr__(self, "constraint_max_absolute_residual", residual)
        object.__setattr__(
            self,
            "constraint_gram_condition_number",
            gram_condition,
        )


def fit_partitioned_local_density(
    molecule: Any,
    density: object,
    *,
    partition: PromolecularStockholderPartition,
    auxiliary_basis: Mapping[str, object],
    grid_level: int = 4,
    batch_size: int = 4000,
) -> tuple[Any, PartitionedLocalDensityFit]:
    """Fit unique atomic stockholder targets in unpartitioned local functions."""

    from pyscf import df, dft
    from scipy import linalg

    if not isinstance(partition, PromolecularStockholderPartition):
        raise TypeError("partition must be PromolecularStockholderPartition.")
    if not isinstance(auxiliary_basis, Mapping) or not auxiliary_basis:
        raise ValueError("auxiliary_basis must be a nonempty element mapping.")
    if isinstance(grid_level, bool) or not isinstance(grid_level, int) or grid_level < 1:
        raise ValueError("grid_level must be a positive integer.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    orbital_dimension = int(molecule.nao_nr())
    matrix = np.asarray(density, dtype=np.float64)
    if matrix.shape != (orbital_dimension, orbital_dimension) or not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError("density must be a finite AO square matrix.")
    matrix = np.ascontiguousarray(0.5 * (matrix + matrix.T))

    auxiliary = df.addons.make_auxmol(
        molecule,
        auxbasis=dict(auxiliary_basis),
    )
    auxiliary_dimension = int(auxiliary.nao_nr())
    atom_slices = np.asarray(auxiliary.aoslice_by_atom(), dtype=np.int64)[:, 2:4]
    overlap_auxiliary = np.asarray(
        auxiliary.intor_symmetric("int1e_ovlp"),
        dtype=np.float64,
    )
    local_metrics = []
    local_conditions = []
    for start, stop in atom_slices:
        block = np.asarray(
            overlap_auxiliary[int(start) : int(stop), int(start) : int(stop)],
            dtype=np.float64,
        )
        block = 0.5 * (block + block.T)
        diagonal = np.sqrt(np.diag(block))
        normalized = block / diagonal[:, None] / diagonal[None, :]
        eigenvalues = linalg.eigvalsh(normalized, check_finite=False)
        if eigenvalues[0] <= 0.0:
            raise RuntimeError("local auxiliary overlap metric is not positive.")
        local_metrics.append(block)
        local_conditions.append(float(eigenvalues[-1] / eigenvalues[0]))

    grids = dft.gen_grid.Grids(molecule)
    grids.level = grid_level
    grids.build(with_non0tab=False)
    right_hand_sides = [np.zeros(stop - start) for start, stop in atom_slices]
    constraints = np.zeros((4, auxiliary_dimension), dtype=np.float64)
    local_counts = np.zeros(int(molecule.natm), dtype=np.float64)
    grid_count = 0.0
    grid_moment = np.zeros(3, dtype=np.float64)
    numbers = np.asarray(molecule.atom_charges(), dtype=np.int64)
    positions = np.asarray(molecule.atom_coords(), dtype=np.float64)
    for start_index in range(0, len(grids.coords), batch_size):
        stop_index = min(start_index + batch_size, len(grids.coords))
        coordinates = np.asarray(
            grids.coords[start_index:stop_index],
            dtype=np.float64,
        )
        quadrature = np.asarray(
            grids.weights[start_index:stop_index],
            dtype=np.float64,
        )
        orbital_ao = np.asarray(
            dft.numint.eval_ao(molecule, coordinates),
            dtype=np.float64,
        )
        rho = np.einsum(
            "gi,ij,gj->g",
            orbital_ao,
            matrix,
            orbital_ao,
            optimize=True,
        )
        auxiliary_ao = np.asarray(
            dft.numint.eval_ao(auxiliary, coordinates),
            dtype=np.float64,
        )
        stockholder = partition.evaluate(
            points_bohr=coordinates,
            atomic_numbers=numbers,
            positions_bohr=positions,
        ).weights
        weighted_density = quadrature * rho
        grid_count += float(np.sum(weighted_density))
        grid_moment += np.einsum(
            "g,gx->x", weighted_density, coordinates, optimize=True
        )
        local_counts += stockholder.T @ weighted_density
        constraints[0] += np.einsum(
            "g,gp->p", quadrature, auxiliary_ao, optimize=True
        )
        constraints[1:] += np.einsum(
            "g,gx,gp->xp",
            quadrature,
            coordinates,
            auxiliary_ao,
            optimize=True,
        )
        for atom_index, (ao_start, ao_stop) in enumerate(atom_slices):
            local_ao = auxiliary_ao[:, int(ao_start) : int(ao_stop)]
            right_hand_sides[atom_index] += np.einsum(
                "g,g,gp->p",
                weighted_density,
                stockholder[:, atom_index],
                local_ao,
                optimize=True,
            )

    coefficients = np.zeros(auxiliary_dimension, dtype=np.float64)
    for (start, stop), metric, rhs in zip(
        atom_slices,
        local_metrics,
        right_hand_sides,
        strict=True,
    ):
        coefficients[int(start) : int(stop)] = linalg.solve(
            metric,
            rhs,
            assume_a="pos",
            check_finite=False,
        )

    overlap_orbital = np.asarray(
        molecule.intor_symmetric("int1e_ovlp"),
        dtype=np.float64,
    )
    position_orbital = np.asarray(
        molecule.intor_symmetric("int1e_r", comp=3),
        dtype=np.float64,
    )
    exact_count = float(np.einsum("ij,ji", matrix, overlap_orbital, optimize=True))
    exact_moment = np.einsum(
        "xij,ji->x",
        position_orbital,
        matrix,
        optimize=True,
    )
    target = np.concatenate(([exact_count], exact_moment))

    inverse_constraint_columns = np.empty(
        (auxiliary_dimension, 4),
        dtype=np.float64,
    )
    for (start, stop), metric in zip(atom_slices, local_metrics, strict=True):
        inverse_constraint_columns[int(start) : int(stop)] = linalg.solve(
            metric,
            constraints[:, int(start) : int(stop)].T,
            assume_a="pos",
            check_finite=False,
        )
    gram = constraints @ inverse_constraint_columns
    gram = 0.5 * (gram + gram.T)
    gram_eigenvalues = linalg.eigvalsh(gram, check_finite=False)
    if gram_eigenvalues[0] <= 0.0:
        raise RuntimeError("partitioned global constraint Gram is not positive.")
    correction = inverse_constraint_columns @ linalg.solve(
        gram,
        target - constraints @ coefficients,
        assume_a="pos",
        check_finite=False,
    )
    coefficients += correction
    residual = float(np.max(np.abs(constraints @ coefficients - target)))
    return auxiliary, PartitionedLocalDensityFit(
        coefficients=coefficients,
        local_target_electron_counts=local_counts,
        local_metric_condition_numbers=tuple(local_conditions),
        exact_electron_count=exact_count,
        exact_electronic_first_moment_bohr_e=exact_moment,
        grid_electron_count=grid_count,
        grid_electronic_first_moment_bohr_e=grid_moment,
        constraint_max_absolute_residual=residual,
        constraint_gram_condition_number=float(
            gram_eigenvalues[-1] / gram_eigenvalues[0]
        ),
        auxiliary_dimension=auxiliary_dimension,
    )


__all__ = ["PartitionedLocalDensityFit", "fit_partitioned_local_density"]
