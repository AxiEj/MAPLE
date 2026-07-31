from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np


@dataclass
class SolvationResult:
    energy_hartree: float
    forces_hartree_per_angstrom: np.ndarray | None = None
    components_hartree: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SinglePointDerivativeEvidence:
    """Research-only derivative evidence that is not an ASE/PES result.

    Route 2's pyddx derivative expression is retained for falsification and
    finite-difference checks, but it has not passed the solution-phase PES
    admission gate. Keeping it separate from :class:`SolvationResult`
    prevents an ASE caller from interpreting it as an available force field.
    """

    energy_hartree: float
    forces_hartree_per_angstrom: np.ndarray
    components_hartree: Mapping[str, float]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        forces = np.array(self.forces_hartree_per_angstrom, dtype=float, copy=True)
        if forces.ndim != 2 or forces.shape[1:] != (3,):
            raise ValueError(
                "Single-point derivative evidence requires an (n_atoms, 3) force array."
            )
        forces.setflags(write=False)
        object.__setattr__(self, "forces_hartree_per_angstrom", forces)
        object.__setattr__(
            self,
            "components_hartree",
            MappingProxyType(
                {
                    str(name): float(value)
                    for name, value in self.components_hartree.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )
