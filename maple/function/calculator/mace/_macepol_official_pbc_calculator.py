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

from maple.function.calculator._official_pbc_base import OfficialPBCAdapterBase
from ..calculator_base import register_calculator
from ..pbc_option_registry import validate_model_pbc_options
from ._official_pbc_common import extract_mace_r_max
from .options import (
    MACEPOL_PBC_MODELS,
    MACEPOL_PBC_OPTION_KEYS,
)


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


@register_calculator
class MACEPolOfficialPBCCalculator(OfficialPBCAdapterBase):
    """MAPLE unit adapter around official PolarMACE ASE calculators."""

    MODEL_NAMES = tuple(MACEPOL_PBC_MODELS)
    SUPPORTS_CHARGE_MULT = True
    OPTION_KEYS = tuple(MACEPOL_PBC_OPTION_KEYS)

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        macepol_options = validate_model_pbc_options(model, options)
        return {
            "default_dtype": macepol_options.get("default_dtype", "float32"),
        }

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

        self._official_calculator = self._build_official_calculator(
            mace_polar_factory,
            {
                "model": upstream_model,
                "device": str(self.device),
                "default_dtype": str(default_dtype),
            },
        )
        self.neighbor_cutoff_A = extract_mace_r_max(self._official_calculator)

    def _build_official_calculator(self, mace_polar_factory, factory_kwargs):
        return mace_polar_factory(**factory_kwargs)

    def _normalized_polar_atoms(self, atoms):
        if atoms is None:
            return None

        proxy = atoms.copy()
        # MAPLE readers use atoms.info["spin"] for the theoretical total
        # spin S=(mult-1)/2.  The upstream PolarMACE ASE adapter expects its
        # own spin label, which follows the multiplicity-style convention used
        # in the official examples.  Keep the conversion local to this adapter
        # and reserve ``macepol_spin`` for deliberate upstream-label overrides.
        spin = proxy.info.pop("macepol_spin", proxy.info.get("mult", 1))
        charge = proxy.info.get("charge", 0)
        proxy.info["spin"] = float(spin)
        proxy.info["charge"] = float(charge)
        # The current official PolarMACE ASE adapter expects a single global
        # external-field vector and reshapes it to (1, 3).  MAPLE accepts that
        # representation directly; for compatibility with older per-atom zero
        # placeholders, identical per-atom rows are reduced to one vector.
        if "external_field" in proxy.info:
            external_field = np.asarray(proxy.info["external_field"], dtype=float)
            if external_field.shape == (3,):
                pass
            elif external_field.shape == (1, 3):
                external_field = external_field.reshape(3)
            elif external_field.shape == (len(proxy), 3) and np.allclose(external_field, external_field[0]):
                external_field = external_field[0].copy()
            else:
                raise ValueError(
                    "MACE-Polar PBC external_field must have shape "
                    f"(3,) or a uniform ({len(proxy)}, 3); got {external_field.shape}."
                )
        else:
            external_field = np.zeros(3, dtype=float)
        proxy.info["external_field"] = external_field
        return proxy

    def _prepare_atoms(self, atoms):
        return self._normalized_polar_atoms(atoms)
