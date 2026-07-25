"""Return types for the unified batched calculator interface.

`BatchResult` is the single contract returned by `CalcABC.calculate_many`.
Keeping it return-value driven lets native paths avoid mutating ASE's
single-structure cache; the shared sequential fallback restores that cache
before returning.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BatchResult:
    """Result of `CalcABC.calculate_many(atoms_list, properties)`.

    Attributes
    ----------
    energies
        (B,) float64 array of energies in Hartree. `None` when 'energy' was
        not requested.
    forces
        Length-B list of (N_i, 3) float64 arrays in Hartree/Å. Per-structure
        atom counts may differ across the list; the caller is responsible for
        matching each `forces[i]` to its `atoms_list[i]`. `None` when 'forces'
        was not requested.
    """

    energies: np.ndarray | None = None
    forces: list[np.ndarray] | None = None

    def __post_init__(self) -> None:
        lengths = []
        if self.energies is not None:
            energies = np.array(self.energies, dtype=np.float64, copy=True)
            if energies.ndim != 1:
                raise ValueError(
                    f"BatchResult.energies must be a 1D array, got shape "
                    f"{energies.shape}"
                )
            object.__setattr__(self, "energies", energies)
            lengths.append(("energies", len(energies)))
        if self.forces is not None:
            force_arrays = [
                np.array(forces, dtype=np.float64, copy=True) for forces in self.forces
            ]
            object.__setattr__(self, "forces", force_arrays)
            lengths.append(("forces", len(force_arrays)))
            for i, arr in enumerate(force_arrays):
                if arr.ndim != 2 or arr.shape[1] != 3:
                    raise ValueError(
                        "BatchResult.forces entries must have shape (N, 3), "
                        f"got forces[{i}].shape={arr.shape}"
                    )
        if lengths:
            expected_name, expected_len = lengths[0]
            for name, got_len in lengths[1:]:
                if got_len != expected_len:
                    raise ValueError(
                        "BatchResult field lengths must match: "
                        f"{expected_name} has length {expected_len}, "
                        f"{name} has length {got_len}"
                    )
