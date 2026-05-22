"""WS4b — under PBC, runtime COM removal defaults OFF unless set explicitly.

Runtime COM removal subtracts total momentum during dynamics and perturbs
diffusion/VACF under PBC, so the NVT/NPT default of ``remove_com_every=100`` is
forced to 0 for periodic systems unless the user passes the key.  Non-PBC
defaults and explicit user values are left untouched.
"""

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator._ase_unit_contract import (
    ASE_STRESS_UNIT,
    MAPLE_ENERGY_UNIT,
    MAPLE_FORCE_UNIT,
)
from maple.function.dispatcher.md.ensemble.nvt import NVT
from maple.function.dispatcher.md.ensemble.npt import NPT


class _PBCCalc(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self):
        super().__init__()
        self.maple_model_name = "fake-pbc"
        self.maple_pbc_md_supported = True
        self.maple_stress_supported = True
        self.maple_stress_unit = ASE_STRESS_UNIT
        self.maple_energy_unit = MAPLE_ENERGY_UNIT
        self.maple_force_unit = MAPLE_FORCE_UNIT
        self.maple_neighbor_cutoff = 2.0  # < min-image radius of the test cells

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))
        if "stress" in properties:
            self.results["stress"] = np.zeros(6)


class _ClusterCalc(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self):
        super().__init__()
        self.maple_model_name = "fake-cluster"
        self.maple_pbc_md_supported = False
        self.maple_stress_supported = False
        self.maple_energy_unit = MAPLE_ENERGY_UNIT
        self.maple_force_unit = MAPLE_FORCE_UNIT

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))


def _crystal():
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True)
    atoms.calc = _PBCCalc()
    return atoms


def _molecule():
    atoms = Atoms("He3", positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]])
    atoms.calc = _ClusterCalc()
    return atoms


def test_pbc_nvt_defaults_com_off_when_unset(tmp_path):
    sim = NVT(output=str(tmp_path / "nvt.out"), atoms=_crystal(), paras={
        "steps": 0, "verbose": 0, "thermostat": "v-rescale",
    })
    assert sim.params.remove_com_every == 0


def test_pbc_nvt_honors_explicit_com_every(tmp_path):
    sim = NVT(output=str(tmp_path / "nvt.out"), atoms=_crystal(), paras={
        "steps": 0, "verbose": 0, "thermostat": "v-rescale", "remove_com_every": 50,
    })
    assert sim.params.remove_com_every == 50


def test_pbc_npt_defaults_com_off_when_unset(tmp_path):
    sim = NPT(output=str(tmp_path / "npt.out"), atoms=_crystal(), paras={
        "steps": 0, "verbose": 0, "thermostat": "v-rescale", "barostat": "c-rescale",
    })
    assert sim.params.remove_com_every == 0


def test_nonpbc_nvt_keeps_default_com_every(tmp_path):
    sim = NVT(output=str(tmp_path / "nvt.out"), atoms=_molecule(), paras={
        "steps": 0, "verbose": 0, "thermostat": "v-rescale",
    })
    assert sim.params.remove_com_every == 100
