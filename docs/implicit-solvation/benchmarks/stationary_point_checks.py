"""Pure stationary-point checks shared by Route 1 validation runners.

The helper intentionally does not interpret an optimizer return value as
convergence.  It classifies only the supplied final forces and Hessian, using
the same mass weighting and rigid-motion projection as MAPLE's ``MWFrequency``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from ase import Atoms

from maple.function.dispatcher.frequency.frequency import (
    HARTREE_AMU_ANGSTROM2_TO_CM1,
    MWFrequency,
)

FORCE_MAX_THRESHOLD_HARTREE_PER_ANGSTROM = 2.5e-4
FORCE_RMS_THRESHOLD_HARTREE_PER_ANGSTROM = 1.5e-4
HESSIAN_ASYMMETRY_ABSOLUTE_BUDGET_HARTREE_PER_ANGSTROM2 = 5.0e-4
HESSIAN_ASYMMETRY_RELATIVE_BUDGET_FACTOR = 1.0e-3


def _signed_frequencies(eigenvalues: np.ndarray) -> np.ndarray:
    return (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * HARTREE_AMU_ANGSTROM2_TO_CM1
    )


def analyze_stationary_point(
    atoms: Atoms,
    hessian: np.ndarray,
    forces: np.ndarray,
    *,
    negative_cutoff_cm1: float = 30.0,
) -> dict[str, Any]:
    """Analyze final E/F/H data without relying on optimizer status.

    Parameters
    ----------
    atoms
        Geometry and atomic masses corresponding exactly to ``hessian`` and
        ``forces``.
    hessian
        Cartesian Hessian in Hartree/Angstrom^2 with shape ``(3N, 3N)``.
    forces
        Cartesian forces in Hartree/Angstrom with shape ``(N, 3)``.
    negative_cutoff_cm1
        A significant imaginary mode has frequency less than the negative of
        this positive magnitude.  Softer modes remain in returned spectra.

    Returns
    -------
    dict
        Raw mass-weighted and rigid-motion-projected spectra, final-force
        metrics, the most negative significant Cartesian mode (if present),
        and independent minimum/first-order-saddle classification flags.

    Notes
    -----
    The function performs no file I/O and does not assert that an optimization
    algorithm converged.  Step-size convergence must be checked separately by
    the runner that owns the optimizer trajectory.
    """
    if not isinstance(atoms, Atoms) or len(atoms) == 0:
        raise ValueError("atoms must be a non-empty ASE Atoms object")

    cutoff = float(negative_cutoff_cm1)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("negative_cutoff_cm1 must be finite and positive")

    atom_count = len(atoms)
    expected_hessian_shape = (3 * atom_count, 3 * atom_count)
    hessian_array = np.asarray(hessian, dtype=np.float64)
    force_array = np.asarray(forces, dtype=np.float64)
    if hessian_array.shape != expected_hessian_shape:
        raise ValueError(
            f"hessian shape {hessian_array.shape} does not match "
            f"{expected_hessian_shape}"
        )
    if force_array.shape != (atom_count, 3):
        raise ValueError(
            f"forces shape {force_array.shape} does not match {(atom_count, 3)}"
        )

    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    masses = np.asarray(atoms.get_masses(), dtype=np.float64)
    if not (
        np.all(np.isfinite(positions))
        and np.all(np.isfinite(masses))
        and np.all(np.isfinite(hessian_array))
        and np.all(np.isfinite(force_array))
    ):
        raise ValueError("atoms, masses, Hessian, and forces must all be finite")
    if np.any(masses <= 0.0):
        raise ValueError("all atomic masses must be finite and positive")

    antisymmetric = hessian_array - hessian_array.T
    maximum_raw_asymmetry = float(np.max(np.abs(antisymmetric)))
    hessian_norm = float(np.linalg.norm(hessian_array))
    relative_raw_asymmetry = (
        float(np.linalg.norm(antisymmetric) / hessian_norm)
        if hessian_norm > 0.0
        else 0.0
    )
    symmetric_hessian = 0.5 * (hessian_array + hessian_array.T)
    maximum_absolute_raw_hessian = float(np.max(np.abs(hessian_array)))
    asymmetry_budget = (
        HESSIAN_ASYMMETRY_ABSOLUTE_BUDGET_HARTREE_PER_ANGSTROM2
        + HESSIAN_ASYMMETRY_RELATIVE_BUDGET_FACTOR
        * maximum_absolute_raw_hessian
    )
    hessian_quality_pass = bool(maximum_raw_asymmetry <= asymmetry_budget)

    inverse_root_mass = np.repeat(1.0 / np.sqrt(masses), 3)
    raw_mass_weighted_hessian = (
        symmetric_hessian
        * inverse_root_mass[:, None]
        * inverse_root_mass[None, :]
    )
    raw_frequencies = _signed_frequencies(
        np.linalg.eigvalsh(raw_mass_weighted_hessian)
    )

    frequency = MWFrequency(os.devnull, atoms)
    frequency.verbosity = 0
    physical_frequencies, physical_modes = frequency.compute_frequencies(
        symmetric_hessian
    )
    physical_frequencies = np.asarray(physical_frequencies, dtype=np.float64)
    physical_modes = np.asarray(physical_modes, dtype=np.float64).reshape(
        3 * atom_count,
        atom_count,
        3,
    )
    negative_indices = np.flatnonzero(physical_frequencies < -cutoff)
    negative_mode = (
        physical_modes[int(negative_indices[0])].copy()
        if negative_indices.size
        else None
    )

    force_max = float(np.max(np.abs(force_array)))
    force_rms = float(np.sqrt(np.mean(np.square(force_array))))
    force_converged = bool(
        force_max <= FORCE_MAX_THRESHOLD_HARTREE_PER_ANGSTROM
        and force_rms <= FORCE_RMS_THRESHOLD_HARTREE_PER_ANGSTROM
    )
    negative_count = int(negative_indices.size)
    is_minimum_candidate = bool(force_converged and negative_count == 0)
    is_first_order_saddle_candidate = bool(force_converged and negative_count == 1)
    is_minimum = bool(is_minimum_candidate and hessian_quality_pass)
    is_first_order_saddle = bool(
        is_first_order_saddle_candidate and hessian_quality_pass
    )

    return {
        "negative_cutoff_cm1": cutoff,
        "force_max_hartree_per_angstrom": force_max,
        "force_rms_hartree_per_angstrom": force_rms,
        "force_converged": force_converged,
        "maximum_raw_hessian_asymmetry_hartree_per_angstrom2": (
            maximum_raw_asymmetry
        ),
        "maximum_absolute_raw_hessian_hartree_per_angstrom2": (
            maximum_absolute_raw_hessian
        ),
        "raw_hessian_asymmetry_budget_hartree_per_angstrom2": asymmetry_budget,
        "relative_raw_hessian_asymmetry_frobenius": relative_raw_asymmetry,
        "hessian_quality_pass": hessian_quality_pass,
        "raw_frequencies_cm1": raw_frequencies,
        "physical_frequencies_cm1": physical_frequencies,
        "physical_modes_cartesian": physical_modes,
        "significant_negative_mode_indices": negative_indices,
        "significant_negative_count": negative_count,
        "negative_mode_cartesian": negative_mode,
        "is_minimum_candidate": is_minimum_candidate,
        "is_first_order_saddle_candidate": is_first_order_saddle_candidate,
        "spectral_minimum_candidate": negative_count == 0,
        "spectral_first_order_saddle_candidate": negative_count == 1,
        "is_minimum": is_minimum,
        "is_first_order_saddle": is_first_order_saddle,
    }


__all__ = [
    "FORCE_MAX_THRESHOLD_HARTREE_PER_ANGSTROM",
    "FORCE_RMS_THRESHOLD_HARTREE_PER_ANGSTROM",
    "HESSIAN_ASYMMETRY_ABSOLUTE_BUDGET_HARTREE_PER_ANGSTROM2",
    "HESSIAN_ASYMMETRY_RELATIVE_BUDGET_FACTOR",
    "analyze_stationary_point",
]
