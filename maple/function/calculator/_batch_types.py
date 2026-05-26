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
