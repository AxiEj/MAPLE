"""Shared contract for named Route-2 V0 molecular external-potential scalars.

The molecular ideal-gas and projected molecular-HNC functionals must consume
one configuration-wise scalar ``u(Gamma)``.  They do not need to know whether
that scalar came from the early promolecular Thomas--Fermi control or from a
later zero-field MACE cluster-energy construction.  This marker keeps those
two deliberately different source definitions from being silently combined
while permitting either to enter the same stationary liquid equations.
"""

from __future__ import annotations

import numpy as np


class Route2V0MolecularExternalPotentialContract:
    """Marker base for one validated molecular external-potential source.

    Concrete immutable sources must provide ``configurations`` and an immutable
    finite ``external_potential_hartree`` vector.  The marker is intentionally
    narrow: source-specific geometry, force, and provenance invariants remain
    owned by their concrete construction modules.
    """


def require_molecular_external_potential_values(
    value: object,
    *,
    configuration_count: int,
) -> np.ndarray:
    """Validate and return the finite configuration-energy vector of a source."""

    if not isinstance(value, Route2V0MolecularExternalPotentialContract):
        raise TypeError(
            "Molecular functional requires a declared Route-2 V0 molecular "
            "external-potential source."
        )
    energies = np.asarray(
        getattr(value, "external_potential_hartree", None),
        dtype=float,
    )
    if energies.shape != (configuration_count,) or not np.all(np.isfinite(energies)):
        raise ValueError(
            "Molecular external-potential source must provide finite energies for "
            "every declared configuration."
        )
    return energies


__all__ = [
    "Route2V0MolecularExternalPotentialContract",
    "require_molecular_external_potential_values",
]
