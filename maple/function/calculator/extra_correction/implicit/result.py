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


@dataclass
class SolvationDirectionalResult:
    """Energy, force, and HVP from one solvent scalar graph.

    The direction is dimensionless in flattened Cartesian ordering.  Therefore
    ``hvp_hartree_per_angstrom2`` has the same units as a Cartesian Hessian
    acting on that direction.
    """

    hvp_hartree_per_angstrom2: np.ndarray
    forces_hartree_per_angstrom: np.ndarray
    energy_hartree: float
    provenance: dict[str, Any] = field(default_factory=dict)
