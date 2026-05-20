"""Single backend property-request entry point for MD (WS3, follows WS2).

``evaluate_md_properties`` is the one place that asks a calculator for
energy/forces/(stress) during a step.  It requests every needed property in a
*single* backend evaluation, so a property-selective adapter (one that computes
only the properties named in ``properties=``) does not pay multiple forward
passes when the step needs energy, forces and stress together.

It returns a frozen result object whose field names carry their units, so the
central evaluator does not itself become a unit-confusion entry point:

* ``energy_ha``           — potential energy in Hartree (MAPLE calculator unit)
* ``forces_au``           — forces in Ha/Bohr (the integrator unit)
* ``forces_ha_per_ang``   — forces in Ha/Å (the raw calculator unit)
* ``stress_ev_per_ang3``  — Voigt stress in eV/Å³ (ASE unit), or None
* ``pressure_bar``        — instantaneous pressure in bar, or None
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_changes

from .utils import (
    HA_PER_ANG_TO_AU,
    compute_instantaneous_pressure,
    validate_stress_tensor,
)


@dataclass(frozen=True)
class MDProperties:
    """Backend properties for one configuration, units encoded in the names."""

    energy_ha: float
    forces_au: np.ndarray
    forces_ha_per_ang: np.ndarray
    stress_ev_per_ang3: Optional[np.ndarray]
    pressure_bar: Optional[float]


def evaluate_md_properties(
    atoms: Atoms,
    *,
    need_stress: bool = False,
    velocities_au: Optional[np.ndarray] = None,
) -> MDProperties:
    """Evaluate energy/forces/(stress) for ``atoms`` in a single backend call.

    Parameters
    ----------
    atoms:
        System with an attached calculator.
    need_stress:
        Request the stress tensor as well (validated via the MAPLE stress unit
        contract).  Required for NPT pressure.
    velocities_au:
        Velocities in atomic units (Bohr/a.u. time).  When provided together
        with ``need_stress``, the instantaneous pressure (bar) is also returned.
    """
    calc = getattr(atoms, "calc", None)
    if calc is None:
        raise ValueError("evaluate_md_properties requires an attached calculator")

    properties = ["energy", "forces"]
    if need_stress:
        properties.append("stress")

    # One backend evaluation for every requested property.  Subsequent accessors
    # read the populated result cache rather than triggering new forward passes.
    calc.calculate(atoms, properties=properties, system_changes=all_changes)

    energy_ha = float(atoms.get_potential_energy())
    forces_ha_per_ang = np.asarray(atoms.get_forces(), dtype=float)
    forces_au = forces_ha_per_ang * HA_PER_ANG_TO_AU

    stress_ev_per_ang3 = None
    pressure_bar = None
    if need_stress:
        stress_ev_per_ang3 = validate_stress_tensor(atoms)
        if velocities_au is not None:
            pressure_bar = compute_instantaneous_pressure(atoms, velocities_au)

    return MDProperties(
        energy_ha=energy_ha,
        forces_au=forces_au,
        forces_ha_per_ang=forces_ha_per_ang,
        stress_ev_per_ang3=stress_ev_per_ang3,
        pressure_bar=pressure_bar,
    )
