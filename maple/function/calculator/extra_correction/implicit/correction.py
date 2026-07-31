"""Route-2 composition boundary for MACE-POLAR plus SMD continuum."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .ddpcm_smd import PyDDXSMDImplicitSolvation
from .result import SolvationResult
from .smd import SMDImplicitSolvation


class ImplicitSolvationCorrection:
    """Prepare and evaluate one explicitly selected Route-2 SMD provider."""

    def __init__(
        self,
        atoms,
        charge_options: dict[str, Any],
        solvation_options: dict[str, Any],
        *,
        output: str | os.PathLike[str] | None = None,
    ):
        self.atoms = atoms
        self.charge_options = dict(charge_options)
        self.solvation_options = dict(solvation_options)
        if self.charge_options:
            raise ValueError(
                "Route 2 obtains its density from MACE-POLAR; remove #charge(...)."
            )
        if self.solvation_options.get("experimental") is not True:
            raise ValueError(
                "Route 2 is an uncertified research path; "
                "set experimental=true explicitly."
            )
        method = str(self.solvation_options.get("method", "")).lower()
        if method != "smd":
            raise ValueError("The Route-2 branch supports method='smd' only.")
        if "profile" not in self.solvation_options:
            raise ValueError(
                "Route 2 SMD requires an explicit versioned profile."
            )
        provider_name = str(
            self.solvation_options.get("provider", "pcmsolver")
        ).lower()
        providers = {
            "pcmsolver": SMDImplicitSolvation,
            "pyddx": PyDDXSMDImplicitSolvation,
        }
        try:
            provider_type = providers[provider_name]
        except KeyError as exc:
            raise ValueError(
                "Route 2 provider must be pcmsolver or pyddx."
            ) from exc

        self.method = "smd"
        self.mode = str(self.solvation_options.get("response", "scf")).lower()
        output_path = Path(output).resolve() if output else Path.cwd() / "maple.out"
        self.audit_dir = output_path.with_suffix(output_path.suffix + ".implicit")
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.provider = provider_type(
            atoms,
            self.solvation_options,
            audit_dir=self.audit_dir,
        )
        self.supported_properties = set(self.provider.supported_properties)
        self._write_audit_manifest()

    def _write_audit_manifest(self) -> None:
        provenance = getattr(self.provider, "provenance", None)
        if callable(provenance):
            provenance = provenance()
        manifest = {
            "schema_version": 2,
            "charge": None,
            "solvation": provenance,
            "charge_options": {},
            "solvation_options": self.solvation_options,
            "energy_composition": "E_MAPLE_gas + delta_G_solv",
            "response_lifecycle": f"density-coupled-{self.mode}",
            "elements": self.atoms.get_chemical_symbols(),
            "positions_angstrom": np.asarray(
                self.atoms.get_positions(), dtype=float
            ).tolist(),
        }
        (self.audit_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    def _write_public_result_ledger(self, result: SolvationResult) -> None:
        """Persist the checked leaf/derived energy split for one evaluation."""

        if not result.leaf_components_hartree:
            return
        payload = {
            "schema_version": 1,
            "energy_hartree": float(result.energy_hartree),
            "leaf_components_hartree": dict(result.leaf_components_hartree),
            "derived_totals_hartree": dict(result.derived_totals_hartree),
            "profile": result.provenance.get("profile"),
            "provider": result.provenance.get("provider"),
            "component_contract": (
                "derived totals are checked from leaves; do not sum the "
                "legacy flat components map"
            ),
        }
        output_path = self.audit_dir / "route2-public-result-ledger.json"
        temporary_path = output_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary_path.replace(output_path)

    def evaluate(
        self,
        atoms,
        *,
        need_forces: bool = False,
        calculator=None,
    ) -> SolvationResult:
        if need_forces:
            raise NotImplementedError(
                "Route 2 SMD does not expose forces before the "
                "solution-phase PES validation gate passes."
            )
        result = self.provider.evaluate(
            atoms,
            need_forces=need_forces,
            calculator=calculator,
        )
        self._write_public_result_ledger(result)
        return result
