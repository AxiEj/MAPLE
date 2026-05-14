import importlib
from typing import Literal, Optional

import torch
from ase.calculators.calculator import all_changes

from .._ase_unit_contract import (
    ase_properties_for_maple_request,
    copy_ase_results_to_maple_units,
)
from ..calculator_base import CalcABC
from .options import MACE_PBC_DEFAULT_HEADS, MACE_PBC_MODELS


def _load_mace_mp(model: str):
    try:
        calculators = importlib.import_module("mace.calculators")
        return calculators.mace_mp
    except ImportError as exc:
        raise ImportError(
            f"Model '{model}' requires the official MACE package. "
            'Install with: pip install -e ".[pbc-mace]"'
        ) from exc
    except AttributeError as exc:
        raise ImportError(
            f"Model '{model}' requires a MACE package exposing mace.calculators.mace_mp. "
            'Install/upgrade with: pip install -e ".[pbc-mace]"'
        ) from exc


class MACEOfficialPBCCalculator(CalcABC):
    """MAPLE unit adapter around official MACE foundation ASE calculators."""

    implemented_properties = ["energy", "forces", "stress", "free_energy"]
    supported_hessian_modes = ()

    def __init__(
        self,
        device: torch.device,
        model: str = "mace-mp-pbc",
        foundation: Optional[str] = None,
        default_dtype: Literal["float32", "float64"] = "float32",
        dispersion: bool = False,
        head: Optional[str] = None,
        implicit: Literal["gbsa", "none"] = "gbsa",
        solvent: str = "none",
        mace_mp_factory=None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model = model
        self.hessian = None

        default_foundation = MACE_PBC_MODELS.get(model)
        if default_foundation is None:
            raise ValueError(f"Unsupported MACE PBC model: '{model}'")

        if implicit == "gbsa" and solvent != "none":
            raise NotImplementedError("Implicit solvent is not supported for MACE PBC backends.")

        if mace_mp_factory is None:
            mace_mp_factory = _load_mace_mp(model)

        factory_kwargs = {
            "model": foundation or default_foundation,
            "device": str(self.device),
            "default_dtype": str(default_dtype),
            "dispersion": bool(dispersion),
        }
        effective_head = head or MACE_PBC_DEFAULT_HEADS.get(model)
        if effective_head is not None:
            factory_kwargs["head"] = effective_head

        self._official_calculator = mace_mp_factory(**factory_kwargs)
        self.maple_neighbor_cutoff = self._extract_neighbor_cutoff()

    def _extract_neighbor_cutoff(self) -> "Optional[float]":
        calc = self._official_calculator
        models = getattr(calc, "models", None)
        if not models:
            return getattr(calc, "r_max", None)
        rmax = getattr(models[0], "r_max", None)
        if rmax is None:
            return None
        try:
            return float(rmax.item() if hasattr(rmax, "item") else rmax)
        except Exception:
            return None

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self._official_calculator.calculate(
            atoms=atoms,
            properties=ase_properties_for_maple_request(properties),
            system_changes=system_changes,
        )
        copy_ase_results_to_maple_units(self._official_calculator.results, self.results)
