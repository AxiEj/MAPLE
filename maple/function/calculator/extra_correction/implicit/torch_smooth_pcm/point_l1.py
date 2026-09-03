"""Point-l<=1 physical constants and smooth sphere-pair validation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numpy as np
from ase.units import Bohr, Hartree

COULOMB_EV_ANGSTROM_PER_E2 = Hartree * Bohr
POINT_SOURCE_SHELL_EVENT_TOLERANCE_ANGSTROM = 0.05

@dataclass(frozen=True)
class HarmonicSpherePairTopology:
    topology_sha256: str
    relations: tuple[tuple[int, int, str], ...]
    minimum_tangency_margin_angstrom: float | None


def harmonic_sphere_pair_topology(positions_angstrom: object, radii_angstrom: object) -> HarmonicSpherePairTopology:
    """Classify pairs without treating smooth internal/external tangency as an event."""
    positions = np.asarray(positions_angstrom, dtype=float)
    radii = np.asarray(radii_angstrom, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or radii.shape != (len(positions),):
        raise ValueError("positions/radii have invalid shape")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("positions and radii must be finite; radii must be positive")
    relations=[]; margins=[]
    for i in range(len(positions)):
        for j in range(i+1,len(positions)):
            d=float(np.linalg.norm(positions[j]-positions[i]))
            if not np.isfinite(d) or d <= 1e-12:
                raise ValueError("sphere centres must be distinct")
            inner=abs(float(radii[i]-radii[j])); outer=float(radii[i]+radii[j])
            relation="nested" if d < inner else "intersecting" if d < outer else "separated"
            relations.append((i,j,relation)); margins.append(min(abs(d-inner),abs(d-outer)))
    payload={"radii":radii.tolist(),"relations":relations}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return HarmonicSpherePairTopology(digest,tuple(relations),min(margins) if margins else None)

__all__=["COULOMB_EV_ANGSTROM_PER_E2","POINT_SOURCE_SHELL_EVENT_TOLERANCE_ANGSTROM","HarmonicSpherePairTopology","harmonic_sphere_pair_topology"]
