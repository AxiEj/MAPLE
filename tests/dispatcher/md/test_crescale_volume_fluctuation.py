import math

import pytest
from ase import Atoms

from maple.function.dispatcher.md.barostat.crescale import CRescaleBarostat
from maple.function.dispatcher.md.units import (
    EV_PER_ANG3_TO_BAR,
    HARTREE_TO_EV,
    KELVIN_TO_HARTREE,
)


def test_crescale_noise_prefactor_uses_bar_to_ang3_per_ev_conversion():
    temperature = 298.15
    compressibility = 4.5e-5
    tau_p = 2000.0
    timestep = 0.5
    atoms = Atoms("Ar", positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True)

    barostat = CRescaleBarostat(
        atoms=atoms,
        pressure=1.0,
        temperature=temperature,
        tau_p=tau_p,
        timestep=timestep,
        compressibility=compressibility,
    )

    kT_ev = temperature * KELVIN_TO_HARTREE * HARTREE_TO_EV
    expected = math.sqrt(
        kT_ev * (compressibility * EV_PER_ANG3_TO_BAR) * timestep / (2.0 * tau_p)
    )

    # Locks a prior audit false positive: changing the conversion from * to /
    # shrinks this prefactor by about 1.6e6 and the variance by about 2.6e12.
    assert barostat._lam_noise_prefactor == pytest.approx(expected, rel=1e-9)

    recovered = (
        barostat._lam_noise_prefactor**2
        * (2.0 * tau_p)
        / (kT_ev * EV_PER_ANG3_TO_BAR * timestep)
    )
    assert abs(math.log10(recovered / compressibility)) < 0.5
