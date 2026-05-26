"""Official AIMNet2 periodic backend adapter.

AIMNet2 (long-range Coulomb framing):
             Anstine, Zubatyuk & Isayev, *Chemical Science* (2025),
             DOI 10.1039/D4SC08572H.
"""

import importlib
from typing import Literal, Optional

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from maple.function.calculator._ase_unit_contract import ASE_STRESS_UNIT, EV2HARTREE
from ..calculator_base import CalcABC
from .options import AIMNET_PBC_COULOMB_METHODS, AIMNET_PBC_MODELS

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


class AIMNet2OfficialPBCCalculator(CalcABC):
    """MAPLE unit adapter around the official AIMNet2 ASE PBC calculator."""

    implemented_properties = ["energy", "forces", "stress", "free_energy"]
    supported_hessian_modes = ()
    maple_pbc_md_supported = True
    maple_stress_supported = True
    maple_stress_unit = ASE_STRESS_UNIT
    supported_maple_properties = frozenset({"energy", "forces", "stress", "MD"})

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
        self._official_calculator = ase_calculator_cls(base_calculator)

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
        else:  # "ewald" / "pme"
            self.short_range_realspace_cutoff_A = None
            self.neighbor_cutoff_A = AIMNET2_SHORT_RANGE_CUTOFF_A

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
            self.results["stress"] = np.asarray(official_results["stress"], dtype=float)
