"""Instantaneous pressure from the virial theorem.

Depends on :mod:`.units` and the stress-tensor contract check in
:mod:`.capabilities`; re-exported by ``...md.utils``.
"""

import numpy as np
from ase import Atoms

from .capabilities import validate_stress_tensor
from .thermo import calculate_kinetic_energy
from .units import EV_PER_ANG3_TO_BAR, HARTREE_TO_EV


def compute_instantaneous_pressure(
    atoms: Atoms,
    velocities: np.ndarray,
) -> float:
    """
    Compute instantaneous pressure from the virial theorem.

    P = (2*KE + W) / (3*V)

    where W = -V * (σ_xx + σ_yy + σ_zz) is the virial from the stress tensor.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system with attached calculator.
    velocities : np.ndarray
        Current velocities in atomic units (Bohr/a.u. time), shape (N_atoms, 3).

    Returns
    -------
    float
        Instantaneous pressure in bar.

    References
    ----------
    Allen & Tildesley, Computer Simulation of Liquids, 2nd ed. (2017), §3.3.
    """
    volume = atoms.get_volume()   # Å³
    if volume <= 0.0 or not np.isfinite(volume):
        raise ValueError(f"Pressure calculation requires a finite positive volume, got {volume!r}.")

    # Kinetic contribution (in eV).
    #
    # The kinetic term uses the FULL kinetic energy from all velocities.  A net
    # centre-of-mass (COM) velocity therefore contributes an extra 2*KE_com/(3V)
    # ~ k_B T/V to the pressure — O(1/N) and negligible for a large cell, but
    # resolvable for a small validation cell.  This is DOF-consistent in MAPLE
    # because the COM is projected at initialization and the non-re-exciting
    # v-rescale / c-rescale operators leave it at zero, so KE_com stays ~0 (see
    # the operator-aware DOF policy in semantics.py and the additivity test in
    # tests/dispatcher/md/test_pbc_capabilities.py).  A caller that runs with a
    # deliberately non-zero COM should project it before reading the pressure.
    ke_ev = calculate_kinetic_energy(atoms, velocities) * HARTREE_TO_EV

    # Virial contribution from stress tensor (eV)
    stress = validate_stress_tensor(atoms)   # eV/Å³, Voigt: xx,yy,zz,yz,xz,xy
    # Hydrostatic virial: W = -V * (σ_xx + σ_yy + σ_zz)
    virial_ev = -volume * (stress[0] + stress[1] + stress[2])

    # P = (2*KE + W) / (3*V)  in eV/Å³, then convert to bar
    pressure_ev_ang3 = (2.0 * ke_ev + virial_ev) / (3.0 * volume)
    return pressure_ev_ang3 * EV_PER_ANG3_TO_BAR
