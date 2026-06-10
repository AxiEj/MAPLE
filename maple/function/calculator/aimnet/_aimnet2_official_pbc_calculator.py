"""Official AIMNet2 periodic backend adapter.

AIMNet2 (long-range Coulomb framing):
             Anstine, Zubatyuk & Isayev, *Chemical Science* (2025),
             DOI 10.1039/D4SC08572H.
"""

import importlib
from typing import Literal, Optional

import torch

from maple.function.calculator._official_pbc_base import OfficialPBCAdapterBase
from ..calculator_base import register_calculator
from ..pbc_option_registry import validate_model_pbc_options
from .options import (
    AIMNET_PBC_COULOMB_METHODS,
    AIMNET_PBC_MODELS,
    AIMNET_PBC_OPTION_KEYS,
)

# AIMNet2's short-range AEV descriptor uses a fixed 5.0 Å cutoff for the local
# environment graph, independent of the long-range Coulomb method.  The MD
# admission gate compares the *effective* local cutoff against the MIC radius,
# so this is the lower bound the AIMNet2 PBC calculator must always satisfy.
AIMNET2_SHORT_RANGE_CUTOFF_A = 5.0


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


@register_calculator
class AIMNet2OfficialPBCCalculator(OfficialPBCAdapterBase):
    """MAPLE unit adapter around the official AIMNet2 ASE PBC calculator."""

    MODEL_NAMES = tuple(AIMNET_PBC_MODELS)
    SUPPORTS_CHARGE_MULT = True
    SUPPORTED_COULOMB_METHODS = tuple(AIMNET_PBC_COULOMB_METHODS)
    OPTION_KEYS = tuple(AIMNET_PBC_OPTION_KEYS)

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        aimnet_options = validate_model_pbc_options(model, options)
        return {
            "coulomb_method": aimnet_options.get("coulomb", "dsf"),
            "cutoff": aimnet_options.get("cutoff", 15.0),
            "dsf_alpha": aimnet_options.get("dsf_alpha", 0.2),
            "ewald_accuracy": aimnet_options.get("ewald_accuracy", 1e-6),
            "pme_cutoff": aimnet_options.get("pme_cutoff"),
        }

    def __init__(
        self,
        device: torch.device,
        model: str = "aimnet2-pbc",
        coulomb_method: str = "dsf",
        cutoff: float = 15.0,
        dsf_alpha: float = 0.2,
        ewald_accuracy: float = 1e-6,
        pme_cutoff: Optional[float] = None,
        implicit: Literal["gbsa", "none"] = "gbsa",
        solvent: str = "none",
        base_calculator_cls=None,
        ase_calculator_cls=None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model_name = model
        self.hessian = None

        official_model = AIMNET_PBC_MODELS.get(model)
        if official_model is None:
            raise ValueError(f"Unsupported AIMNet2 PBC model: '{model}'")

        if base_calculator_cls is None or ase_calculator_cls is None:
            base_calculator_cls, ase_calculator_cls = _load_aimnet2_classes(model)

        base_calculator = base_calculator_cls(official_model, device=str(self.device))
        if not hasattr(base_calculator, "set_lrcoulomb_method"):
            raise RuntimeError("Official AIMNet2 calculator does not expose set_lrcoulomb_method.")

        method = str(coulomb_method).lower()
        if method not in AIMNET_PBC_COULOMB_METHODS:
            raise ValueError(
                f"Unsupported AIMNet2 PBC Coulomb method: '{method}'. "
                f"Supported: {sorted(AIMNET_PBC_COULOMB_METHODS)}"
            )

        lrcoulomb_kwargs = {
            "cutoff": float(cutoff),
            "dsf_alpha": float(dsf_alpha),
            "ewald_accuracy": float(ewald_accuracy),
        }
        if pme_cutoff is not None:
            lrcoulomb_kwargs["pme_cutoff"] = float(pme_cutoff)

        base_calculator.set_lrcoulomb_method(method, **lrcoulomb_kwargs)
        self._official_calculator = self._build_official_calculator(
            base_calculator,
            ase_calculator_cls,
        )

        # Record the long-range method and its public cutoff verbatim for the
        # provenance manifest; do not conflate them with the *effective* local
        # neighbor cutoff the MD admission gate compares against the MIC radius.
        self.lrcoulomb_method = method
        self.lrcoulomb_cutoff_A = float(cutoff)
        self.long_range_coulomb_cutoff_A = float(cutoff)
        self.local_descriptor_cutoff_A = AIMNET2_SHORT_RANGE_CUTOFF_A
        # Effective neighbor cutoff for MD admission.  DSF is a real cutoff-based
        # method, so its public ``cutoff`` participates in the MIC bound (but the
        # 5 Å AEV short-range descriptor is always present).  Ewald and PME
        # ignore the public cutoff at runtime — they estimate their real-space
        # cutoff from accuracy + cell geometry — so the only cutoff MAPLE can
        # honestly gate against is the AEV short-range descriptor.
        if method == "dsf":
            self.short_range_realspace_cutoff_A = float(cutoff)
            self.neighbor_cutoff_A = max(AIMNET2_SHORT_RANGE_CUTOFF_A, float(cutoff))
            # DSF is a finite real-space cutoff method.  MAPLE's production
            # admission therefore treats it as single-image MIC scoped; using a
            # cutoff longer than half the shortest periodic vector can introduce
            # discontinuous image-shell changes in stress/volume validation.
            self.maple_requires_single_image_mic = True
            self.maple_periodic_neighborlist_multi_image_safe = False
        else:  # "ewald" / "pme"
            self.short_range_realspace_cutoff_A = None
            self.neighbor_cutoff_A = AIMNET2_SHORT_RANGE_CUTOFF_A
            self.maple_requires_single_image_mic = False
            self.maple_periodic_neighborlist_multi_image_safe = True

        if implicit == "gbsa" and solvent != "none":
            raise NotImplementedError("Implicit solvent is not supported for AIMNet2 PBC backends.")

    def _build_official_calculator(self, base_calculator, ase_calculator_cls):
        return ase_calculator_cls(base_calculator)
