"""Exact point-monopole boundary data in complete real-harmonic spaces.

For a point charge at displacement ``d`` from a target-sphere centre, the
boundary potential on radius ``a`` has coefficients

``4*pi/(2*l+1) * r_<**l / r_>**(l+1) * Y_lm(d_hat)``.

This module owns the NumPy reference implementation and the rigid-motion
invariant inside/outside stratum identity.  It does not own a continuum solve
or any public capability.  A source centre crossing a target shell
(``|d| == a``) is a genuine derivative event and therefore fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE

from .harmonic_coefficients import (
    PerAtomHarmonicSpace,
    _bounded_lmax,
    _real_harmonic_design,
)
from .harmonic_single_layer import COULOMB_EV_ANGSTROM_PER_E2

HARMONIC_POINT_SOURCE_CONTRACT_ID = (
    "maple.route2.continuum.harmonic-point-monopole-boundary-map.v1"
)
HARMONIC_POINT_SOURCE_PROVIDER_ID = (
    "maple.route2.continuum.harmonic-point-monopole.impl.v1"
)
POINT_SOURCE_SHELL_EVENT_TOLERANCE_ANGSTROM = 1.0e-12


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _positive_radius(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _positions(values: object) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if (
        result.ndim != 2
        or result.shape[0] < 1
        or result.shape[1] != 3
        or not np.all(np.isfinite(result))
    ):
        raise ValueError("positions_angstrom must be finite with shape (N,3).")
    return np.array(result, copy=True)


def _radii(values: object, *, atom_count: int) -> tuple[float, ...]:
    result = tuple(
        _positive_radius(value, name=f"radii_angstrom[{index}]")
        for index, value in enumerate(values)
    )
    if len(result) != atom_count:
        raise ValueError("radii_angstrom must have one value per atom.")
    return result


@dataclass(frozen=True, slots=True)
class HarmonicPointSourceTopology:
    """Rigid-motion-invariant point-centre/target-shell stratum identity."""

    topology_sha256: str
    relations: tuple[tuple[int, int, str], ...]
    minimum_shell_margin_angstrom: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_id": HARMONIC_POINT_SOURCE_CONTRACT_ID,
            "topology_sha256": self.topology_sha256,
            "relations": [list(item) for item in self.relations],
            "minimum_shell_margin_angstrom": self.minimum_shell_margin_angstrom,
        }


def point_source_topology(
    positions_angstrom: object,
    radii_angstrom: object,
) -> HarmonicPointSourceTopology:
    """Classify every distinct source centre as inside/outside each target shell."""

    positions = _positions(positions_angstrom)
    radii = _radii(radii_angstrom, atom_count=len(positions))
    relations: list[tuple[int, int, str]] = []
    margins: list[float] = []
    for target, radius in enumerate(radii):
        for source in range(len(positions)):
            if target == source:
                relations.append((target, source, "self"))
                continue
            distance = float(np.linalg.norm(positions[source] - positions[target]))
            if not np.isfinite(distance) or distance <= 1.0e-12:
                raise ValueError("distinct point-source centres must not coincide.")
            margin = abs(distance - radius)
            if margin <= POINT_SOURCE_SHELL_EVENT_TOLERANCE_ANGSTROM:
                raise ValueError("point source lies on a target sphere event surface.")
            relations.append(
                (target, source, "inside" if distance < radius else "outside")
            )
            margins.append(margin)
    payload = {
        "contract_id": HARMONIC_POINT_SOURCE_CONTRACT_ID,
        "provider_id": HARMONIC_POINT_SOURCE_PROVIDER_ID,
        "radii_angstrom": radii,
        "relations": relations,
    }
    return HarmonicPointSourceTopology(
        topology_sha256=_sha(payload),
        relations=tuple(relations),
        minimum_shell_margin_angstrom=min(margins) if margins else None,
    )


def point_monopole_harmonic_coefficients(
    displacement_angstrom: object,
    *,
    target_radius_angstrom: float,
    lmax: int,
) -> np.ndarray:
    """Return boundary-potential coefficients per unit point charge.

    ``displacement_angstrom`` points from the target sphere centre to the
    source centre.  The result uses atom-major-compatible ``l,m`` ordering and
    has units ``eV/e`` per source charge in ``e``.
    """

    displacement = np.asarray(displacement_angstrom, dtype=float)
    if displacement.shape != (3,) or not np.all(np.isfinite(displacement)):
        raise ValueError("displacement_angstrom must be finite with shape (3,).")
    radius = _positive_radius(target_radius_angstrom, name="target_radius_angstrom")
    maximum = _bounded_lmax(lmax)
    distance = float(np.linalg.norm(displacement))
    result = np.zeros(((maximum + 1) ** 2,), dtype=float)
    if distance <= 1.0e-12:
        result[0] = COULOMB_EV_ANGSTROM_PER_E2 * math.sqrt(4.0 * np.pi) / radius
        return result
    if abs(distance - radius) <= POINT_SOURCE_SHELL_EVENT_TOLERANCE_ANGSTROM:
        raise ValueError("point source lies on a target sphere event surface.")

    harmonics = _real_harmonic_design((displacement / distance)[None, :], lmax=maximum)[
        0
    ]
    for ell in range(maximum + 1):
        radial = (
            distance**ell / radius ** (ell + 1)
            if distance < radius
            else radius**ell / distance ** (ell + 1)
        )
        section = slice(ell * ell, (ell + 1) * (ell + 1))
        result[section] = (
            COULOMB_EV_ANGSTROM_PER_E2
            * 4.0
            * np.pi
            * radial
            / (2 * ell + 1)
            * harmonics[section]
        )
    return result


def point_l0_harmonic_source_operator(
    *,
    positions_angstrom: object,
    radii_angstrom: object,
    lmax: int,
) -> np.ndarray:
    """Map atomic ``[q,0,0,0]`` sources to raw harmonic boundary data."""

    positions = _positions(positions_angstrom)
    radii = _radii(radii_angstrom, atom_count=len(positions))
    maximum = _bounded_lmax(lmax)
    point_source_topology(positions, radii)
    space = PerAtomHarmonicSpace(atom_count=len(positions), lmax=maximum)
    source_dimension = len(positions) * ATOMIC_L1_SOURCE_SPACE.component_count
    operator = np.zeros((space.dimension, source_dimension), dtype=float)
    block_dimension = space.single_atom_dimension
    source_components = ATOMIC_L1_SOURCE_SPACE.component_count
    for target, radius in enumerate(radii):
        row = slice(target * block_dimension, (target + 1) * block_dimension)
        for source in range(len(positions)):
            operator[row, source * source_components] = (
                point_monopole_harmonic_coefficients(
                    positions[source] - positions[target],
                    target_radius_angstrom=radius,
                    lmax=maximum,
                )
            )
    operator.setflags(write=False)
    return operator


__all__ = [
    "HARMONIC_POINT_SOURCE_CONTRACT_ID",
    "HARMONIC_POINT_SOURCE_PROVIDER_ID",
    "POINT_SOURCE_SHELL_EVENT_TOLERANCE_ANGSTROM",
    "HarmonicPointSourceTopology",
    "point_l0_harmonic_source_operator",
    "point_monopole_harmonic_coefficients",
    "point_source_topology",
]
