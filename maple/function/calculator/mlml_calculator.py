from typing import Optional

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from .set_calculator import SetClaculator
from ..read.filereader.part_reader import PartReader


class MLMLCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        *,
        output: str,
        device: torch.device,
        full_atoms: Atoms,
        high_model: str,
        low_model: str,
        partition_file: str,
        high_model_options: Optional[dict] = None,
        low_model_options: Optional[dict] = None,
        d4: bool = False,
        implicit: str = "None",
        solvent: str = "None",
    ):
        super().__init__()
        self.output = output
        self.device = device
        self.full_atoms = full_atoms

        self.high_model = str(high_model).lower()
        self.low_model = str(low_model).lower()
        self.partition_file = partition_file

        self.high_model_options = high_model_options or {}
        self.low_model_options = low_model_options or {}
        self.d4 = d4
        self.implicit = implicit
        self.solvent = solvent

        partition = PartReader(self.partition_file, natoms=len(full_atoms))
        self.core_indices = partition["core_indices"]
        self.env_indices = partition["env_indices"]

        self.low_calc = SetClaculator(
            self.device,
            self.low_model,
            self.output,
            atoms=self.full_atoms,
            d4=self.d4,
            implicit=self.implicit,
            solvent=self.solvent,
            model_options=self.low_model_options,
        ).set_calculator()

        self.high_calc = SetClaculator(
            self.device,
            self.high_model,
            self.output,
            atoms=self._build_core_atoms(self.full_atoms),
            d4=self.d4,
            implicit=self.implicit,
            solvent=self.solvent,
            model_options=self.high_model_options,
        ).set_calculator()

    def _build_core_atoms(self, atoms: Atoms) -> Atoms:
        core_atoms = atoms[self.core_indices]
        core_atoms = core_atoms.copy()

        for key in ("charge", "mult", "spin"):
            if key in atoms.info:
                core_atoms.info[key] = atoms.info[key]

        return core_atoms

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            atoms = self.full_atoms

        atoms_full = atoms.copy()
        atoms_full.calc = self.low_calc
        e_low_full = float(atoms_full.get_potential_energy())
        f_low_full = np.asarray(atoms_full.get_forces(), dtype=float)

        core_atoms = self._build_core_atoms(atoms)

        core_high = core_atoms.copy()
        core_high.calc = self.high_calc
        e_high_core = float(core_high.get_potential_energy())
        f_high_core = np.asarray(core_high.get_forces(), dtype=float)

        core_low = core_atoms.copy()
        core_low.calc = self.low_calc
        e_low_core = float(core_low.get_potential_energy())
        f_low_core = np.asarray(core_low.get_forces(), dtype=float)

        e_corr = e_high_core - e_low_core
        e_total = e_low_full + e_corr

        f_total = f_low_full.copy()
        core_delta = f_high_core - f_low_core
        f_total[self.core_indices, :] += core_delta

        self.results["energy"] = e_total
        self.results["free_energy"] = e_total
        self.results["forces"] = f_total
        self.results["mlml_energy_low_full"] = e_low_full
        self.results["mlml_energy_high_core"] = e_high_core
        self.results["mlml_energy_low_core"] = e_low_core
        self.results["mlml_energy_correction"] = e_corr
