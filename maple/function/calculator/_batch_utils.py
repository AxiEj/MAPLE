"""Shared helpers for calculator-native batch evaluation.

The public contract stays in :mod:`calculator_base` / ``BatchResult``.  This
module only holds small mechanics that are useful across backend-specific
``calculate_many`` implementations:

* normalize an energy/forces property request;
* keep the sequential fallback result-driven and cache-safe;
* split atom-wise tensors back into ASE-ordered per-structure arrays.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
from ase.calculators.calculator import all_changes

from ._batch_types import BatchResult


def atoms_has_pbc(atoms) -> bool:
    """Return True when an ASE Atoms object has any periodic axis enabled."""
    if atoms is None:
        return False
    return bool(np.any(getattr(atoms, "pbc", False)))


def atoms_list_has_pbc(atoms_list: Sequence) -> bool:
    """Return True when any structure in a candidate batch is periodic."""
    return any(atoms_has_pbc(at) for at in atoms_list)


def normalize_energy_forces_request(properties) -> tuple[tuple[str, ...], bool, bool, list[str]]:
    """Return ``(props, want_energy, want_forces, request)``.

    ``calculate_many`` is intentionally limited to energy/forces.  Numerical
    Hessian assembly belongs to ``FDHessianEvaluator``; analytic batched
    Hessians should be an explicit backend override rather than an accidental
    property side effect.
    """
    props = tuple(properties)
    if "hessian" in props:
        raise NotImplementedError(
            "calculate_many does not assemble Hessians; use FDHessianEvaluator "
            "for numerical Hessians or a backend-specific analytic Hessian path."
        )

    want_energy = "energy" in props
    want_forces = "forces" in props
    request = [p for p in props if p in ("energy", "forces")]
    return props, want_energy, want_forces, request


def empty_batch_result(want_energy: bool, want_forces: bool) -> BatchResult:
    return BatchResult(
        energies=np.zeros(0, dtype=np.float64) if want_energy else None,
        forces=[] if want_forces else None,
    )


def sequential_calculate_many(
    calc,
    atoms_list: Sequence,
    request: Sequence[str],
    want_energy: bool,
    want_forces: bool,
) -> BatchResult:
    """Sequential fallback matching ``CalcABC.calculate_many`` semantics."""
    energies = [] if want_energy else None
    forces_list = [] if want_forces else None

    for at in atoms_list:
        calc.calculate(at, properties=list(request), system_changes=all_changes)
        if want_energy:
            if "free_energy" in calc.results:
                energies.append(float(calc.results["free_energy"]))
            else:
                energies.append(float(calc.results["energy"]))
        if want_forces:
            forces_list.append(np.asarray(calc.results["forces"], dtype=np.float64))

    return BatchResult(
        energies=np.asarray(energies, dtype=np.float64) if energies is not None else None,
        forces=forces_list,
    )


def atom_counts(atoms_list: Sequence) -> list[int]:
    return [len(at) for at in atoms_list]


def split_atomwise_array(values: np.ndarray, counts: Sequence[int]) -> list[np.ndarray]:
    """Split a flat ``(sum(N_i), 3)`` array into per-Atoms force arrays."""
    out = []
    start = 0
    for n_atoms in counts:
        stop = start + int(n_atoms)
        out.append(np.asarray(values[start:stop], dtype=np.float64))
        start = stop
    return out


def grouped_indices_by_numbers(atoms_list: Sequence) -> Iterable[tuple[tuple[int, ...], list[int]]]:
    """Yield original indices grouped by identical atomic-number sequence."""
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i, at in enumerate(atoms_list):
        groups[tuple(int(z) for z in at.get_atomic_numbers())].append(i)
    return groups.items()
