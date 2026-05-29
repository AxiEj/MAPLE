"""Temperature and kinetic-energy calculations from velocities.

Depends on :mod:`.units`; re-exported by ``...md.utils``.
"""

from typing import Optional

import numpy as np
from ase import Atoms

from .units import AMU_TO_AU, KELVIN_TO_HARTREE


def calculate_temperature(atoms: Atoms, velocities: np.ndarray, n_dof: Optional[int] = None) -> float:
    """
    Calculate instantaneous temperature from velocities.

    Uses the equipartition theorem:
        T = 2 * KE / (N_dof * k_B)

    In the migrated MD code paths, `n_dof` is supplied explicitly from the
    runtime motion/DOF policy so that temperature reporting, thermostat target
    kinetic energy, and logger summaries all use the same active subspace.
    The built-in fallback (`3N` for PBC, `3N-3` for non-PBC) is retained only
    for legacy callers that have not yet been migrated to the central policy.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    velocities : np.ndarray
        Atomic velocities in atomic units (Bohr/a.u. time)
        Shape: (N_atoms, 3)
    n_dof : int, optional
        Active number of degrees of freedom. When omitted, a legacy fallback is
        used (`3N` for PBC, `3N-3` for non-PBC).

    Returns
    -------
    float
        Temperature in Kelvin
    """
    masses = atoms.get_masses() * AMU_TO_AU  # Convert to atomic units
    kinetic = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)

    if n_dof is None:
        n_atoms = len(atoms)
        # Backward-compatible default until all callers migrate to explicit policy.
        if any(atoms.pbc):
            n_dof = 3 * n_atoms
        else:
            n_dof = 3 * n_atoms - 3

    if n_dof <= 0:
        return 0.0

    temperature = 2.0 * kinetic / (n_dof * KELVIN_TO_HARTREE)
    return temperature


def calculate_center_of_mass_kinetic_energy(atoms: Atoms, velocities: np.ndarray) -> float:
    """Return the kinetic energy in net centre-of-mass translation.

    The velocity unit convention matches :func:`calculate_kinetic_energy`
    (atomic units, Bohr/a.u. time), and the returned energy is Hartree.
    """
    masses = atoms.get_masses() * AMU_TO_AU
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        return 0.0
    total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    return float(0.5 * np.dot(total_momentum, total_momentum) / total_mass)


def calculate_kinetic_energy(
    atoms: Atoms,
    velocities: np.ndarray,
    *,
    exclude_com_kinetic: bool = False,
) -> float:
    """
    Calculate kinetic energy.

    KE = 0.5 * sum(m_i * v_i^2)

    When ``exclude_com_kinetic`` is true, the net centre-of-mass translational
    kinetic energy is subtracted.  This is the kinetic subspace needed by NPT
    pressure/barostat paths when the resolved DOF policy says COM translation
    is projected/constrained rather than part of the active thermostat bath.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    velocities : np.ndarray
        Atomic velocities in atomic units
        Shape: (N_atoms, 3)

    Returns
    -------
    float
        Kinetic energy in Hartree
    """
    masses = atoms.get_masses() * AMU_TO_AU
    kinetic = float(0.5 * np.sum(masses[:, np.newaxis] * velocities**2))
    if exclude_com_kinetic:
        kinetic -= calculate_center_of_mass_kinetic_energy(atoms, velocities)
        kinetic = max(kinetic, 0.0)
    return kinetic
