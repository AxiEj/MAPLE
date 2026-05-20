"""Official MACE-MP periodic backend adapter.

MACE-MP-0: Batatia et al., arXiv:2401.00096 (2023)
             — short-range materials potential.
"""

import importlib
from typing import Literal, Optional

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from maple.function.calculator._ase_unit_contract import ASE_STRESS_UNIT, EV2HARTREE
from ..calculator_base import CalcABC
from .options import MACE_PBC_MODELS, MACE_PBC_OFFICIAL_FOUNDATIONS


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
    """MAPLE unit adapter around official MACE-MP ASE calculators."""

    implemented_properties = ["energy", "forces", "stress", "free_energy"]
    supported_hessian_modes = ()
    maple_pbc_md_supported = True
    maple_stress_supported = True
    maple_stress_unit = ASE_STRESS_UNIT
    supported_maple_properties = frozenset({"energy", "forces", "stress", "MD"})

    def __init__(
        self,
        device: torch.device,
        model: str = "mace-mp-pbc-medium",
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
        self.model_name = model
        self.hessian = None

        default_foundation = MACE_PBC_MODELS.get(model)
        if default_foundation is None:
            raise ValueError(f"Unsupported MACE PBC model: '{model}'")

        if implicit == "gbsa" and solvent != "none":
            raise NotImplementedError("Implicit solvent is not supported for MACE PBC backends.")

        if mace_mp_factory is None:
            mace_mp_factory = _load_mace_mp(model)

        foundation_name = foundation or default_foundation
        official_foundation = MACE_PBC_OFFICIAL_FOUNDATIONS[foundation_name]
        factory_kwargs = {
            "model": official_foundation,
            "device": str(self.device),
            "default_dtype": str(default_dtype),
            "dispersion": bool(dispersion),
        }
        if head is not None:
            factory_kwargs["head"] = str(head)

        self._official_calculator = mace_mp_factory(**factory_kwargs)
        self.neighbor_cutoff_A = self._extract_neighbor_cutoff()

    def _extract_neighbor_cutoff(self) -> Optional[float]:
        calc = self._official_calculator
        models = getattr(calc, "models", None)
        if not models:
            return getattr(calc, "r_max", None)

        r_max = getattr(models[0], "r_max", None)
        if r_max is None:
            return None
        try:
            return float(r_max.item() if hasattr(r_max, "item") else r_max)
        except Exception:
            return None

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
            self.results["stress"] = np.asarray(official_results["stress"], dtype=float)
