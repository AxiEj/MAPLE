"""Instantaneous pressure from the virial theorem.

Depends on :mod:`.units` and the stress-tensor contract check in
:mod:`.capabilities`; re-exported by ``...md.utils``.
"""

import numpy as np
from ase import Atoms

from .capabilities import validate_stress_tensor
from .units import AMU_ANG2_PER_FS2_TO_EV, AU_TO_FS, BOHR_TO_ANGSTROM, EV_PER_ANG3_TO_BAR


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

    # Kinetic contribution (in eV)
    masses_amu = atoms.get_masses()
    # v in a.u. (Bohr/a.u.time) → convert to Å/fs
    v_ang_per_fs = velocities * BOHR_TO_ANGSTROM / AU_TO_FS
    # KE in eV: 0.5 * m[amu] * v²[Å²/fs²] * (amu·Å²/fs² → eV)
    ke_ev = 0.5 * np.sum(masses_amu[:, np.newaxis] * v_ang_per_fs**2) * AMU_ANG2_PER_FS2_TO_EV

    # Virial contribution from stress tensor (eV)
    stress = validate_stress_tensor(atoms)   # eV/Å³, Voigt: xx,yy,zz,yz,xz,xy
    # Hydrostatic virial: W = -V * (σ_xx + σ_yy + σ_zz)
    virial_ev = -volume * (stress[0] + stress[1] + stress[2])

    # P = (2*KE + W) / (3*V)  in eV/Å³, then convert to bar
    pressure_ev_ang3 = (2.0 * ke_ev + virial_ev) / (3.0 * volume)
    return pressure_ev_ang3 * EV_PER_ANG3_TO_BAR
