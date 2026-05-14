import importlib
from typing import Literal

import torch
from ase.calculators.calculator import all_changes

from .._ase_unit_contract import (
    ase_properties_for_maple_request,
    copy_ase_results_to_maple_units,
)
from ..calculator_base import CalcABC
from .options import AIMNET_PBC_MODELS


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
        self.maple_neighbor_cutoff = float(cutoff)

        if implicit == "gbsa" and solvent != "none":
            raise NotImplementedError("Implicit solvent is not supported for AIMNet2 PBC backends.")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self._official_calculator.calculate(
            atoms=atoms,
            properties=ase_properties_for_maple_request(properties),
            system_changes=system_changes,
        )
        copy_ase_results_to_maple_units(self._official_calculator.results, self.results)
