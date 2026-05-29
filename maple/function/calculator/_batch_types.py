"""Return types for the unified batched calculator interface.

`BatchResult` is the single contract that `CalcABC.calculate_many` returns and
that every evaluator in `_batch_eval` consumes. Keeping it return-value driven
(rather than mutating `calc.results`) prevents the batched paths from
clobbering ASE's single-structure result cache that other modules
(frequency, irc, scan) still depend on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class BatchResult:
    """Result of `CalcABC.calculate_many(atoms_list, properties)`.

    Attributes
    ----------
    energies
        (B,) float64 array of energies (Hartree if the calculator's
        single-structure path returns Hartree). `None` when 'energy' was not
        requested.
    forces
        Length-B list of (N_i, 3) float64 arrays in Hartree/Å. Per-structure
        atom counts may differ across the list; the caller is responsible for
        matching each `forces[i]` to its `atoms_list[i]`. `None` when 'forces'
        was not requested.
    hessians
        Length-B list of (3*N_i, 3*N_i) float64 arrays. `None` when 'hessian'
        was not requested.
    padding_counts
        Optional (B,) int array. Some batch back-ends (notably AIMNet2's
        existing GPU path) work in a padded DOF space and report the padding
        per entry so downstream code can slice. Empty (`None`) when no
        padding is in use.
    """

    energies: Optional[np.ndarray] = None
    forces: Optional[List[np.ndarray]] = None
    hessians: Optional[List[np.ndarray]] = None
    padding_counts: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        lengths = []
        if self.energies is not None:
            energies = np.asarray(self.energies, dtype=np.float64)
            if energies.ndim != 1:
                raise ValueError(
                    f"BatchResult.energies must be a 1D array, got shape "
                    f"{energies.shape}"
                )
            object.__setattr__(self, "energies", energies)
            lengths.append(("energies", len(energies)))
        if self.forces is not None:
            force_arrays = [np.asarray(forces, dtype=np.float64) for forces in self.forces]
            object.__setattr__(self, "forces", force_arrays)
            lengths.append(("forces", len(force_arrays)))
            for i, arr in enumerate(force_arrays):
                if arr.ndim != 2 or arr.shape[1] != 3:
                    raise ValueError(
                        "BatchResult.forces entries must have shape (N, 3), "
                        f"got forces[{i}].shape={arr.shape}"
                    )
        if self.hessians is not None:
            hessian_arrays = [
                np.asarray(hessian, dtype=np.float64)
                for hessian in self.hessians
            ]
            object.__setattr__(self, "hessians", hessian_arrays)
            lengths.append(("hessians", len(hessian_arrays)))
            for i, arr in enumerate(hessian_arrays):
                if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                    raise ValueError(
                        "BatchResult.hessians entries must be square 2D "
                        f"arrays, got hessians[{i}].shape={arr.shape}"
                    )
        if self.padding_counts is not None:
            padding_counts = np.asarray(self.padding_counts, dtype=np.int64)
            if padding_counts.ndim != 1:
                raise ValueError(
                    "BatchResult.padding_counts must be a 1D array, got "
                    f"shape {padding_counts.shape}"
                )
            object.__setattr__(self, "padding_counts", padding_counts)
            lengths.append(("padding_counts", len(padding_counts)))

        if lengths:
            expected_name, expected_len = lengths[0]
            for name, got_len in lengths[1:]:
                if got_len != expected_len:
                    raise ValueError(
                        "BatchResult field lengths must match: "
                        f"{expected_name} has length {expected_len}, "
                        f"{name} has length {got_len}"
                    )

        if self.forces is not None and self.hessians is not None:
            for i, (forces, hessian) in enumerate(zip(self.forces, self.hessians)):
                n_atoms = np.asarray(forces).shape[0]
                expected = (3 * n_atoms, 3 * n_atoms)
                arr = np.asarray(hessian)
                if arr.shape != expected:
                    raise ValueError(
                        "BatchResult.hessians entries must match the "
                        "corresponding force atom count: "
                        f"hessians[{i}].shape={arr.shape}, expected {expected}"
                    )
