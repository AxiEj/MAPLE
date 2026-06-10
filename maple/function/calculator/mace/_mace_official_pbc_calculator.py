"""Official MACE-MP periodic backend adapter.

MACE-MP-0: Batatia et al., arXiv:2401.00096 (2023)
             — short-range materials potential.
"""

import importlib
from typing import Literal, Optional

import torch

from maple.function.calculator._official_pbc_base import OfficialPBCAdapterBase
from ..calculator_base import register_calculator
from ..pbc_option_registry import validate_model_pbc_options
from ._official_pbc_common import extract_mace_r_max
from .options import (
    MACE_PBC_MODELS,
    MACE_PBC_OFFICIAL_FOUNDATIONS,
    MACE_PBC_OPTION_KEYS,
)


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


@register_calculator
class MACEOfficialPBCCalculator(OfficialPBCAdapterBase):
    """MAPLE unit adapter around official MACE-MP ASE calculators."""

    MODEL_NAMES = tuple(MACE_PBC_MODELS)
    SUPPORTS_CHARGE_MULT = False
    OPTION_KEYS = tuple(MACE_PBC_OPTION_KEYS)

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        mace_options = validate_model_pbc_options(model, options)
        return {
            "foundation": mace_options.get("foundation"),
            "default_dtype": mace_options.get("default_dtype", "float32"),
            "dispersion": mace_options.get("dispersion", False),
            "head": mace_options.get("head"),
        }

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

        self._official_calculator = self._build_official_calculator(
            mace_mp_factory,
            factory_kwargs,
        )
        self.neighbor_cutoff_A = extract_mace_r_max(self._official_calculator)

    def _build_official_calculator(self, mace_mp_factory, factory_kwargs):
        return mace_mp_factory(**factory_kwargs)
