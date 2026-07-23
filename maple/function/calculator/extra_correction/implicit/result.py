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
