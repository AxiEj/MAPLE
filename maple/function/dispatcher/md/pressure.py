"""Instantaneous pressure from the virial theorem.

Depends on :mod:`.units` and the stress-tensor contract check in
:mod:`.capabilities`; re-exported by ``...md.utils``.
"""

import numpy as np
from ase import Atoms

from maple.function.calculator._ase_unit_contract import ASE_STRESS_UNIT

from .capabilities import validate_stress_tensor
from .thermo import calculate_kinetic_energy
from .units import EV_PER_ANG3_TO_BAR, HARTREE_TO_EV


def compute_instantaneous_pressure(
    atoms: Atoms,
    velocities: np.ndarray,
    *,
    exclude_com_kinetic: bool = False,
    stress_ev_per_ang3: np.ndarray | None = None,
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
    exclude_com_kinetic : bool, default=False
        Exclude net centre-of-mass translational kinetic energy from the kinetic
        pressure term.  NPT callers should set this from the resolved DOF policy
        when COM translation is projected/constrained rather than an active bath
        mode.
    stress_ev_per_ang3 : np.ndarray, optional
        Already-validated ASE Voigt stress in eV/Å³.  Supplying it lets the
        evaluator compute stress and pressure from one validated property read.

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

    # Kinetic contribution (in eV).  The caller explicitly chooses whether this
    # is the full 3N kinetic energy or the DOF-policy active subspace with
    # centre-of-mass translation removed.  That explicit switch is important for
    # c-rescale/v-rescale NPT: if COM was projected and the global operators do
    # not re-excite it, Bernetti-Bussi's kinetic pressure term uses K - K_cm.
    ke_ev = calculate_kinetic_energy(
        atoms,
        velocities,
        exclude_com_kinetic=exclude_com_kinetic,
    ) * HARTREE_TO_EV

    # Virial contribution from stress tensor (eV)
    if stress_ev_per_ang3 is not None:
        stress_unit = getattr(getattr(atoms, "calc", None), "maple_stress_unit", ASE_STRESS_UNIT)
        if stress_unit != ASE_STRESS_UNIT:
            raise ValueError(
                "Pressure calculation stress unit contract mismatch: expected "
                f"{ASE_STRESS_UNIT!r}, got {stress_unit!r}."
            )
        stress = np.asarray(stress_ev_per_ang3, dtype=float).reshape(-1)
    else:
        stress = validate_stress_tensor(atoms)
    # eV/Å³, Voigt: xx,yy,zz,yz,xz,xy
    if stress.shape != (6,) or not np.all(np.isfinite(stress)):
        raise ValueError("Pressure calculation requires a finite Voigt stress tensor of length 6.")
    # Hydrostatic virial: W = -V * (σ_xx + σ_yy + σ_zz)
    virial_ev = -volume * (stress[0] + stress[1] + stress[2])

    # P = (2*KE + W) / (3*V)  in eV/Å³, then convert to bar
    pressure_ev_ang3 = (2.0 * ke_ev + virial_ev) / (3.0 * volume)
    return pressure_ev_ang3 * EV_PER_ANG3_TO_BAR
