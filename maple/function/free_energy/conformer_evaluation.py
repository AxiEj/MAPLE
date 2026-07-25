"""Model-agnostic gas-energy evaluation for fixed conformer ensembles."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def evaluate_gas_conformer_energies(
    calculator,
    template_atoms,
    positions_angstrom: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    *,
    batch_size: int = 8,
) -> np.ndarray:
    """Return gas-phase conformer energies in Hartree.

    The calculator must implement MAPLE's ``calculate_many`` contract. The
    input geometry/topology comes from ``template_atoms``; only Cartesian
    positions vary. An attached solvent correction is rejected so callers
    cannot accidentally count the Route 1 correction twice.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be a positive integer.")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if getattr(calculator, "solvent_correction", None) is not None:
        raise ValueError(
            "Conformer gas energies require a gas-phase calculator without an "
            "attached solvent correction."
        )
    if not hasattr(calculator, "calculate_many"):
        raise TypeError("calculator must implement calculate_many.")

    positions = np.asarray(positions_angstrom, dtype=np.float64)
    expected_shape = (len(template_atoms), 3)
    if (
        positions.ndim != 3
        or tuple(positions.shape[1:]) != expected_shape
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError(
            "positions_angstrom must be finite with shape "
            f"(n_states, {expected_shape[0]}, 3)."
        )
    if len(positions) == 0:
        raise ValueError("positions_angstrom must contain at least one state.")

    energies: list[np.ndarray] = []
    for start in range(0, len(positions), batch_size):
        structures = []
        for geometry in positions[start : start + batch_size]:
            atoms = template_atoms.copy()
            atoms.set_positions(geometry, apply_constraint=False)
            structures.append(atoms)
        result = calculator.calculate_many(structures, properties=("energy",))
        values = result.energies
        if values is None:
            raise RuntimeError("calculate_many did not return requested energies.")
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (len(structures),) or not np.all(np.isfinite(values)):
            raise RuntimeError("calculate_many returned invalid conformer energies.")
        energies.append(values)

    return np.concatenate(energies)
