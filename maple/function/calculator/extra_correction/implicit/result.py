from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SolvationResult:
    energy_hartree: float
    forces_hartree_per_angstrom: np.ndarray | None = None
    components_hartree: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    quantity_kind: str = "solvation_free_energy_correction"

    def __post_init__(self):
        if self.quantity_kind not in {
            "solvation_free_energy_correction",
            "geometry_level_energy_correction",
        }:
            raise ValueError(f"Unknown solvent result quantity: {self.quantity_kind!r}")
