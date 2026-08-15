"""Fixed symmetry-preserving radial embeddings for a four-channel source.

This module does not change the learned atom-centred monopoles or dipoles.  It
only represents every original ``(q,l=1)`` block as one convex mixture of
normalized Gaussian radial primitives.  A common mixture is used for every
element and every member of an angular irrep, so the map preserves total
charge, every atom-centred dipole, permutations, and SO(3) covariance.

The current object is a research coupling candidate.  It does not admit a
quantitative PCM source or any E/F/H/V/M capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from maple.function.calculator.extra_correction.implicit.gto_galerkin import (
    AtomCenteredL1GTOBasis,
)

FIXED_RADIAL_SOURCE_EMBEDDING_CONTRACT_VERSION = (
    "route2-fixed-radial-source-embedding-v1"
)
MACE_POLAR_FIXED_RADIAL_SOURCE_RESEARCH_COUPLING_ID = (
    "maple.route2.coupling.macepolar-original-source-fixed-radial-research.v1"
)
MACE_POLAR_FIXED_RADIAL_SOURCE_RESEARCH_PROFILE_ID = (
    "route2-research-macepolar-original-source-fixed-radial-embedding-v1"
)


def _source4(values: object) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if (
        result.ndim != 2
        or result.shape[0] < 1
        or result.shape[1] != 4
        or not np.all(np.isfinite(result))
    ):
        raise ValueError("source must be finite with shape (n_atoms,4).")
    return np.array(result, copy=True)


@dataclass(frozen=True, slots=True)
class FixedRadialSourceEmbedding:
    """One immutable convex Gaussian mixture for all source components."""

    sigmas_angstrom: tuple[float, ...]
    weights: tuple[float, ...]
    contract_version: str = FIXED_RADIAL_SOURCE_EMBEDDING_CONTRACT_VERSION
    coupling_id: str = MACE_POLAR_FIXED_RADIAL_SOURCE_RESEARCH_COUPLING_ID
    profile_id: str = MACE_POLAR_FIXED_RADIAL_SOURCE_RESEARCH_PROFILE_ID

    def __post_init__(self) -> None:
        sigmas = tuple(float(value) for value in self.sigmas_angstrom)
        weights = tuple(float(value) for value in self.weights)
        if not sigmas or len(sigmas) != len(weights):
            raise ValueError(
                "sigmas_angstrom and weights must have equal nonzero length."
            )
        if (
            not np.all(np.isfinite(sigmas))
            or not np.all(np.asarray(sigmas) > 0.0)
            or len(set(sigmas)) != len(sigmas)
        ):
            raise ValueError("Gaussian sigmas must be finite, positive, and unique.")
        if not np.all(np.isfinite(weights)) or not np.all(np.asarray(weights) > 0.0):
            raise ValueError("Radial weights must be finite and strictly positive.")
        if abs(float(sum(weights)) - 1.0) > 2.0e-15:
            raise ValueError("Radial weights must sum to one without normalization.")
        if self.contract_version != FIXED_RADIAL_SOURCE_EMBEDDING_CONTRACT_VERSION:
            raise ValueError("Unsupported fixed radial embedding contract version.")
        if self.coupling_id != MACE_POLAR_FIXED_RADIAL_SOURCE_RESEARCH_COUPLING_ID:
            raise ValueError("Fixed radial embedding coupling identity is immutable.")
        if self.profile_id != MACE_POLAR_FIXED_RADIAL_SOURCE_RESEARCH_PROFILE_ID:
            raise ValueError("Fixed radial embedding profile identity is immutable.")
        object.__setattr__(self, "sigmas_angstrom", sigmas)
        object.__setattr__(self, "weights", weights)

    @property
    def configuration_sha256(self) -> str:
        payload = {
            "contract_version": self.contract_version,
            "coupling_id": self.coupling_id,
            "profile_id": self.profile_id,
            "sigmas_angstrom": list(self.sigmas_angstrom),
            "weights": list(self.weights),
            "normalization": "each primitive carries the original physical q/p moment",
            "scope": "same mixture for every atom, element, l, and m",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def radial_count(self) -> int:
        return len(self.sigmas_angstrom)

    def expand(self, source: object) -> np.ndarray:
        """Return ``(atom,radial,raw-lm)`` normalized GTO coefficients."""

        values = _source4(source)
        return values[:, None, :] * np.asarray(self.weights)[None, :, None]

    def reduce_cotangent(self, radial_cotangent: object) -> np.ndarray:
        """Apply the exact transpose of :meth:`expand`."""

        values = np.asarray(radial_cotangent, dtype=float)
        if (
            values.ndim != 3
            or values.shape[0] < 1
            or values.shape[1:] != (self.radial_count, 4)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                "radial_cotangent must be finite with shape (n_atoms,n_radial,4)."
            )
        return np.einsum("r,nrc->nc", np.asarray(self.weights), values)

    def surface_operator(
        self,
        surface_points_bohr: object,
        atom_positions_angstrom: object,
    ) -> np.ndarray:
        """Return the fixed-geometry map from original source to surface MEP."""

        positions = np.asarray(atom_positions_angstrom, dtype=float)
        basis = AtomCenteredL1GTOBasis(self.sigmas_angstrom)
        native = np.asarray(
            basis.surface_operator(surface_points_bohr, positions), dtype=float
        ).reshape(-1, len(positions), self.radial_count, 4)
        operator = np.einsum("snrc,r->snc", native, np.asarray(self.weights)).reshape(
            native.shape[0], len(positions) * 4
        )
        operator.setflags(write=False)
        return operator

    def surface_potential(
        self,
        surface_points_bohr: object,
        atom_positions_angstrom: object,
        source: object,
    ) -> np.ndarray:
        values = _source4(source)
        operator = self.surface_operator(surface_points_bohr, atom_positions_angstrom)
        if operator.shape[1] != values.size:
            raise ValueError("source atom count does not match atom positions.")
        return operator @ values.reshape(-1)


__all__ = [
    "FIXED_RADIAL_SOURCE_EMBEDDING_CONTRACT_VERSION",
    "MACE_POLAR_FIXED_RADIAL_SOURCE_RESEARCH_COUPLING_ID",
    "MACE_POLAR_FIXED_RADIAL_SOURCE_RESEARCH_PROFILE_ID",
    "FixedRadialSourceEmbedding",
]
