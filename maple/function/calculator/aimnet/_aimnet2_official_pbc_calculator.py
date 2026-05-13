import importlib
from typing import Literal

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from ..calculator_base import CalcABC
from .options import AIMNET_PBC_MODELS


EV2HARTREE = 1.0 / 27.211386245988


def _load_aimnet2_classes(model: str):
    try:
        calculators = importlib.import_module("aimnet.calculators")
        return calculators.AIMNet2Calculator, calculators.AIMNet2ASE
    except ImportError as exc:
        raise ImportError(
            f"Model '{model}' requires the official AIMNet2 package. "
            'Install with: pip install -e ".[pbc-aimnet]"'
        ) from exc
    except AttributeError as exc:
        raise ImportError(
            f"Model '{model}' requires an AIMNet2 package exposing AIMNet2Calculator "
            "and AIMNet2ASE. Install/upgrade with: pip install -e \".[pbc-aimnet]\""
        ) from exc


class AIMNet2OfficialPBCCalculator(CalcABC):
    """MAPLE unit adapter around the official AIMNet2 ASE PBC calculator."""

    implemented_properties = ["energy", "forces", "stress", "free_energy"]
    supported_hessian_modes = ()

    def __init__(
        self,
        device: torch.device,
        model: str = "aimnet2-pbc",
        coulomb_method: str = "dsf",
        cutoff: float = 15.0,
        dsf_alpha: float = 0.2,
        ewald_accuracy: float = 1e-6,
        implicit: Literal["gbsa", "none"] = "gbsa",
        solvent: str = "none",
        base_calculator_cls=None,
        ase_calculator_cls=None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model = model
        self.hessian = None

        official_model = AIMNET_PBC_MODELS.get(model)
        if official_model is None:
            raise ValueError(f"Unsupported AIMNet2 PBC model: '{model}'")

        if base_calculator_cls is None or ase_calculator_cls is None:
            base_calculator_cls, ase_calculator_cls = _load_aimnet2_classes(model)

        base_calculator = base_calculator_cls(official_model, device=str(self.device))
        if not hasattr(base_calculator, "set_lrcoulomb_method"):
            raise RuntimeError("Official AIMNet2 calculator does not expose set_lrcoulomb_method.")
        base_calculator.set_lrcoulomb_method(
            str(coulomb_method).lower(),
            cutoff=float(cutoff),
            dsf_alpha=float(dsf_alpha),
            ewald_accuracy=float(ewald_accuracy),
        )

        self._official_calculator = ase_calculator_cls(base_calculator)

        if implicit == "gbsa" and solvent != "none":
            raise NotImplementedError("Implicit solvent is not supported for AIMNet2 PBC backends.")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        requested = set(properties or ["energy"])

        official_properties = ["energy"]
        if "forces" in requested:
            official_properties.append("forces")
        if "stress" in requested:
            official_properties.append("stress")

        self._official_calculator.calculate(
            atoms=atoms,
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
            # Stress intentionally stays in ASE-native eV/Angstrom^3 for MAPLE's NPT pressure path.
            self.results["stress"] = np.asarray(official_results["stress"], dtype=float)
