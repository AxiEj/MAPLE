"""Route-2 composition boundary for MACE-POLAR plus SMD/IEFPCM."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .result import SolvationResult
from .smd import SMDImplicitSolvation


class ImplicitSolvationCorrection:
    """Prepare and evaluate the locked Route-2 SMD provider."""

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
                "Route 2 is an energy-only research proof-of-concept; "
                "set experimental=true explicitly."
            )
        method = str(self.solvation_options.get("method", "")).lower()
        if method != "smd":
            raise ValueError("The Route-2 branch supports method='smd' only.")

        self.method = "smd"
        self.mode = str(self.solvation_options.get("response", "scf")).lower()
        output_path = Path(output).resolve() if output else Path.cwd() / "maple.out"
        self.audit_dir = output_path.with_suffix(output_path.suffix + ".implicit")
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.provider = SMDImplicitSolvation(
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

    def evaluate(
        self,
        atoms,
        *,
        need_forces: bool = False,
        calculator=None,
    ) -> SolvationResult:
        return self.provider.evaluate(
            atoms,
            need_forces=need_forces,
            calculator=calculator,
        )
