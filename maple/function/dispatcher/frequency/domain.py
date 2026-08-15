"""Shared domain admission for molecular vibrational mathematics."""

from __future__ import annotations

import numpy as np
from ase import Atoms


def validate_molecular_vibrational_domain(
    atoms: Atoms,
    *,
    operation: str,
) -> None:
    """Reject domains that need periodic or reduced-coordinate mathematics."""

    if np.any(atoms.get_pbc()):
        raise ValueError(f"{operation} requires a non-periodic system.")
    if atoms.constraints:
        raise NotImplementedError(
            f"{operation} with ASE constraints is not implemented; "
            "a reduced-coordinate Hessian/mass contract is required."
        )


__all__ = ["validate_molecular_vibrational_domain"]
