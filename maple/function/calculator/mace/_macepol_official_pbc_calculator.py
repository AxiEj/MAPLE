"""Official MACE-Polar periodic backend adapter.

MACE-POLAR-1: Baldwin et al., arXiv:2602.19411 (2026)
              — polar molecular potential with short + long-range electrostatics.

Runtime requirements: mace-torch>=0.3.16 plus graph_longrange. Install via
``pip install -e \".[pbc-macepol]\"``.
"""

import importlib
from typing import Literal, Optional

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from maple.function.calculator._ase_unit_contract import EV2HARTREE
from ..calculator_base import CalcABC
from .options import MACEPOL_PBC_MODELS


def _load_mace_polar(model: str):
    try:
        calculators = importlib.import_module("mace.calculators")
        return calculators.mace_polar
    except ImportError as exc:
        raise ImportError(
            f"Model '{model}' requires the official MACE-Polar stack. "
            'Install with: pip install -e ".[pbc-macepol]" '
            "(mace-torch>=0.3.16 plus graph_longrange)."
        ) from exc
    except AttributeError as exc:
        raise ImportError(
            f"Model '{model}' requires a MACE package exposing mace.calculators.mace_polar. "
            'Install/upgrade with: pip install -e ".[pbc-macepol]" '
            "(mace-torch>=0.3.16 plus graph_longrange)."
        ) from exc


class MACEPolOfficialPBCCalculator(CalcABC):
    """MAPLE unit adapter around official PolarMACE ASE calculators."""

    implemented_properties = ["energy", "forces", "stress", "free_energy"]
    supported_hessian_modes = ()
    maple_pbc_md_supported = True
    maple_stress_supported = True
    supported_maple_properties = frozenset({"energy", "forces", "stress", "MD"})

    def __init__(
        self,
        device: torch.device,
        model: str = "macepol-pbc-medium",
        default_dtype: Literal["float32", "float64"] = "float32",
        implicit: Literal["gbsa", "none"] = "gbsa",
        solvent: str = "none",
        mace_polar_factory=None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model_name = model
        self.hessian = None

        upstream_model = MACEPOL_PBC_MODELS.get(model)
        if upstream_model is None:
            raise ValueError(f"Unsupported MACE-Polar PBC model: '{model}'")

        if implicit == "gbsa" and solvent != "none":
            raise NotImplementedError("Implicit solvent is not supported for MACE-Polar PBC backends.")

        if mace_polar_factory is None:
            mace_polar_factory = _load_mace_polar(model)

        self._official_calculator = mace_polar_factory(
            model=upstream_model,
            device=str(self.device),
            default_dtype=str(default_dtype),
        )
        self.neighbor_cutoff_A = self._extract_neighbor_cutoff()

    def _extract_neighbor_cutoff(self) -> Optional[float]:
        calc = self._official_calculator
        models = getattr(calc, "models", None)
        if not models:
            r_max = getattr(calc, "r_max", None)
        else:
            r_max = getattr(models[0], "r_max", None)

        if r_max is None:
            return None
        try:
            return float(r_max.item() if hasattr(r_max, "item") else r_max)
        except Exception:
            return None

    def _normalized_polar_atoms(self, atoms):
        if atoms is None:
            return None

        proxy = atoms.copy()
        spin = proxy.info.get("spin", proxy.info.get("mult", 1))
        charge = proxy.info.get("charge", 0)
        proxy.info["spin"] = float(spin)
        proxy.info["charge"] = float(charge)
        # Upstream PolarMACE examples and the local legacy traced wrapper use a
        # per-atom external-field array.  Supplying explicit zeros keeps the ASE
        # adapter contract stable when inputs omit an external field.
        if "external_field" in proxy.info:
            external_field = np.asarray(proxy.info["external_field"], dtype=float)
            if external_field.shape != (len(proxy), 3):
                raise ValueError(
                    "MACE-Polar PBC external_field must have shape "
                    f"({len(proxy)}, 3); got {external_field.shape}."
                )
        else:
            external_field = np.zeros((len(proxy), 3), dtype=float)
        proxy.info["external_field"] = external_field
        return proxy

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        requested = set(properties or ["energy"])

        official_properties = ["energy"]
        if "forces" in requested:
            official_properties.append("forces")
        if "stress" in requested:
            official_properties.append("stress")

        polar_atoms = self._normalized_polar_atoms(atoms)
        self._official_calculator.calculate(
            atoms=polar_atoms,
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
