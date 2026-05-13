import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator.set_calculator import SetClaculator
from maple.function.dispatcher.md.ensemble.npt import NPT
from maple.function.dispatcher.md.ensemble.nve import NVE
from maple.function.dispatcher.md.ensemble.nvt import NVT
from maple.function.dispatcher.md.utils import (
    AMU_ANG2_PER_FS2_TO_EV,
    AU_TO_FS,
    BOHR_TO_ANGSTROM,
    EV_PER_ANG3_TO_BAR,
    MDStressUnavailableError,
    compute_instantaneous_pressure,
)
from maple.function.read.command_control import CommandControl


class EnergyForcesCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))


class StressCalculator(EnergyForcesCalculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, stress):
        super().__init__()
        self._stress = np.asarray(stress, dtype=float)
        self.maple_pbc_md_supported = True
        self.maple_stress_supported = True

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["stress"] = self._stress.copy()


def _periodic_atoms(calc):
    atoms = Atoms(
        "He",
        positions=[[0.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.calc = calc
    return atoms


def test_npt_rejects_missing_stress_at_startup(tmp_path):
    calc = EnergyForcesCalculator()
    calc.maple_pbc_md_supported = True
    calc.maple_stress_supported = True
    atoms = _periodic_atoms(calc)

    with pytest.raises(MDStressUnavailableError, match="stress tensor"):
        NPT(
            output=str(tmp_path / "npt.out"),
            atoms=atoms,
            paras={"steps": 0, "verbose": 0},
        )


def test_pressure_uses_kinetic_and_stress_terms():
    stress = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    atoms = _periodic_atoms(StressCalculator(stress))
    velocities = np.array([[0.01, 0.02, 0.03]])

    pressure = compute_instantaneous_pressure(atoms, velocities)

    volume = atoms.get_volume()
    v_ang_per_fs = velocities * BOHR_TO_ANGSTROM / AU_TO_FS
    ke_ev = (
        0.5
        * np.sum(atoms.get_masses()[:, np.newaxis] * v_ang_per_fs**2)
        * AMU_ANG2_PER_FS2_TO_EV
    )
    virial_ev = -volume * np.sum(stress[:3])
    expected = ((2.0 * ke_ev + virial_ev) / (3.0 * volume)) * EV_PER_ANG3_TO_BAR
    kinetic_only = ((2.0 * ke_ev) / (3.0 * volume)) * EV_PER_ANG3_TO_BAR

    assert np.isclose(pressure, expected)
    assert not np.isclose(pressure, kinetic_only)


@pytest.mark.parametrize("ensemble_cls, ensemble", [(NVE, "nve"), (NVT, "nvt"), (NPT, "npt")])
def test_pbc_md_gate_rejects_nonperiodic_backend(ensemble_cls, ensemble, tmp_path):
    calc = EnergyForcesCalculator()
    calc.maple_model_name = "maceoff23s"
    atoms = _periodic_atoms(calc)

    with pytest.raises(ValueError, match="PBC.*periodic MD support"):
        ensemble_cls(
            output=str(tmp_path / f"{ensemble}.out"),
            atoms=atoms,
            paras={"steps": 0, "verbose": 0},
        )


@pytest.mark.parametrize("ensemble_cls, ensemble", [(NVE, "nve"), (NVT, "nvt")])
def test_non_pbc_nve_nvt_allow_unknown_calculator(ensemble_cls, ensemble, tmp_path):
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], pbc=False)
    atoms.calc = EnergyForcesCalculator()

    ensemble_cls(
        output=str(tmp_path / f"{ensemble}.out"),
        atoms=atoms,
        paras={"steps": 0, "verbose": 0},
    )


def test_command_control_rejects_uma_omol_with_hash_pbc(tmp_path):
    output_path = tmp_path / "parse.out"

    with pytest.raises(ValueError, match="PBC is incompatible with UMA task='omol'"):
        CommandControl.from_settings(
            [
                "#model=uma(task=omol)",
                "#md(ensemble=nve)",
                "#pbc(10.0, 10.0, 10.0)",
            ],
            output_path=str(output_path),
        )


def test_set_calculator_rejects_uma_omol_with_extxyz_pbc(tmp_path):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[8.0, 8.0, 8.0], pbc=True)
    output_path = tmp_path / "factory.out"

    with pytest.raises(ValueError, match="PBC is incompatible with UMA task='omol'"):
        SetClaculator(
            torch.device("cpu"),
            "uma",
            str(output_path),
            atoms=atoms,
            model_options={"task": "omol"},
        ).set_calculator()
