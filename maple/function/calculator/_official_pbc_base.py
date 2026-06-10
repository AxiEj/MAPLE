"""Shared official ASE-backed PBC calculator adapter."""

from abc import abstractmethod

import numpy as np
from ase.calculators.calculator import all_changes

from maple.function.calculator._ase_unit_contract import ASE_STRESS_UNIT, EV2HARTREE
from .calculator_base import CalcABC


class OfficialPBCAdapterBase(CalcABC):
    """MAPLE unit adapter base for official ASE PBC calculators."""

    implemented_properties = ["energy", "forces", "stress", "free_energy"]
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES = ()
    SUPPORTS_PBC = True
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False
    supported_hessian_modes = ()
    maple_pbc_md_supported = True
    maple_stress_supported = True
    maple_stress_unit = ASE_STRESS_UNIT
    # Official periodic ASE backends own their replicated-image neighbor graph;
    # MAPLE should not force them into the supercell-only single-image MIC scope.
    maple_requires_single_image_mic = False
    maple_periodic_neighborlist_multi_image_safe = True

    @abstractmethod
    def _build_official_calculator(self, *args, **kwargs):
        """Return the wrapped official ASE calculator."""

    def _prepare_atoms(self, atoms):
        return atoms

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        requested = set(properties or ["energy"])

        official_properties = ["energy"]
        if "forces" in requested:
            official_properties.append("forces")
        if "stress" in requested:
            official_properties.append("stress")

        self._official_calculator.calculate(
            atoms=self._prepare_atoms(atoms),
            properties=official_properties,
            system_changes=system_changes,
        )

        official_results = self._official_calculator.results
        if "energy" in official_results:
            energy = float(official_results["energy"]) * EV2HARTREE
            self.results["energy"] = energy
            self.results["free_energy"] = energy
        if "free_energy" in official_results:
            self.results["free_energy"] = float(official_results["free_energy"]) * EV2HARTREE
        if "forces" in official_results:
            self.results["forces"] = np.asarray(official_results["forces"], dtype=float) * EV2HARTREE
        if "stress" in official_results:
            self.results["stress"] = np.asarray(official_results["stress"], dtype=float)
