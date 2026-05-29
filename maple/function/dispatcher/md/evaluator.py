"""Single backend property-request entry point for MD (WS3, follows WS2).

``evaluate_md_properties`` is the one place that asks a calculator for
energy/forces/(stress) during a step.  It is cache-aware: it triggers a backend
forward pass only when the geometry changed or a required property is missing
for the current configuration, and when it does it requests every required
property together so a property-selective adapter (one that computes only the
properties named in ``properties=``) does not pay a second pass for stress.  In
the loops this means routing the per-step logged reads through here adds no
backend call beyond the force evaluation the integrator already performed.

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

from .capabilities import validate_stress_tensor
from .pressure import compute_instantaneous_pressure
from .units import HA_PER_ANG_TO_AU


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
    """Evaluate energy/forces/(stress) for ``atoms`` with at most one backend pass.

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

    required = ["energy", "forces"]
    if need_stress:
        required.append("stress")

    # Cache-aware single entry point.  Only request a backend forward pass when
    # the geometry actually changed or a required property is not already cached
    # for the current configuration; otherwise read the existing cache.  When a
    # pass *is* needed, request every required property together so a
    # property-selective adapter does not pay a second pass for stress.  This is
    # what keeps routing the per-step logged reads through here free of extra
    # backend calls (the integrator has usually just evaluated forces).
    system_changes = calc.check_state(atoms)
    if system_changes or any(prop not in calc.results for prop in required):
        calc.calculate(atoms, properties=required, system_changes=system_changes or all_changes)

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
