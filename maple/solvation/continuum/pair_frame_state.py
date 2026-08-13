"""Immutable evidence state for the disabled pair-frame C-PCM candidate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.api.profiles import (
    PAIR_FRAME_WATER_CPCM_110_CONFIGURATION_CONTRACT_ID,
)
from maple.solvation.api.scalar_registry import OPERATIONAL_CPCM_ELECTROSTATIC_V1
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING

from .fixed_topology_cpcm import CONTINUUM_PROFILE_ID
from .pair_frame_geometry import frame_topology_sha256
from maple.solvation.surfaces.fixed_topology import CAVITY_PROFILE_ID

PAIR_FRAME_CPCM_PROVIDER_ID = (
    "maple.route2.continuum.ordered-pair-frame-ensemble-radial-gto-cpcm.impl.v1"
)
PAIR_FRAME_CPCM_COUPLING_ID = "route2-coupling-mace-polar-native-radial-gto-v1"
PAIR_FRAME_CPCM_GRID_POINTS_PER_ATOM = 110


def pair_frame_state_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class PairFrameCPCMState:
    """Immutable scalar/field snapshot for one ensemble continuum solve."""

    provider_id: str
    configuration_sha256: str
    provenance_sha256: str
    geometry_sha256: str
    frame_topology_sha256: str
    frame_activity_sha256: str
    state_hash: str
    source_values: tuple[float, ...]
    field_values: tuple[float, ...]
    polarization_energy_eV: float
    atom_count: int
    frame_count: int
    active_frame_count: int
    continuum_profile_id: str = CONTINUUM_PROFILE_ID
    cavity_profile_id: str = CAVITY_PROFILE_ID
    configuration_contract_id: str = PAIR_FRAME_WATER_CPCM_110_CONFIGURATION_CONTRACT_ID
    coupling_id: str = PAIR_FRAME_CPCM_COUPLING_ID
    scalar_id: str = OPERATIONAL_CPCM_ELECTROSTATIC_V1
    capabilities: CapabilityStatus = CapabilityStatus()

    def __post_init__(self) -> None:
        if self.provider_id != PAIR_FRAME_CPCM_PROVIDER_ID:
            raise ValueError("pair-frame state provider identity is invalid.")
        if (
            self.continuum_profile_id != CONTINUUM_PROFILE_ID
            or self.cavity_profile_id != CAVITY_PROFILE_ID
            or self.configuration_contract_id
            != PAIR_FRAME_WATER_CPCM_110_CONFIGURATION_CONTRACT_ID
            or self.coupling_id != PAIR_FRAME_CPCM_COUPLING_ID
            or self.scalar_id != OPERATIONAL_CPCM_ELECTROSTATIC_V1
        ):
            raise ValueError("pair-frame state profile identities are invalid.")
        if isinstance(self.atom_count, bool) or self.atom_count < 1:
            raise ValueError("pair-frame state atom_count must be positive.")
        for value, name in (
            (self.configuration_sha256, "configuration_sha256"),
            (self.provenance_sha256, "provenance_sha256"),
            (self.geometry_sha256, "geometry_sha256"),
            (self.frame_topology_sha256, "frame_topology_sha256"),
            (self.frame_activity_sha256, "frame_activity_sha256"),
            (self.state_hash, "state_hash"),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA256 digest.")
        expected_frame_count = self.atom_count * (self.atom_count - 1)
        if self.frame_count != expected_frame_count:
            raise ValueError("pair-frame state does not contain every ordered pair.")
        if self.frame_topology_sha256 != frame_topology_sha256(
            self.atom_count, PAIR_FRAME_CPCM_GRID_POINTS_PER_ATOM
        ):
            raise ValueError("pair-frame state topology identity is invalid.")
        if not 1 <= self.active_frame_count <= self.frame_count:
            raise ValueError("pair-frame state active-frame count is invalid.")
        source = self.source
        field = self.reaction_field
        energy = float(self.polarization_energy_eV)
        if not math.isfinite(energy):
            raise ValueError("pair-frame state energy must be finite.")
        paired = 0.5 * MACE_POLAR_RADIAL_GTO_PAIRING.pair(source, field)
        if abs(energy - paired) > max(2.0e-13, 2.0e-13 * abs(energy)):
            raise ValueError("pair-frame state violates the half-coupling scalar.")
        expected = pair_frame_state_sha256(
            {
                "contract": "ordered-pair-frame-ensemble-cpcm-state-v1",
                "provider_id": self.provider_id,
                "configuration_sha256": self.configuration_sha256,
                "provenance_sha256": self.provenance_sha256,
                "geometry_sha256": self.geometry_sha256,
                "frame_topology_sha256": self.frame_topology_sha256,
                "frame_activity_sha256": self.frame_activity_sha256,
                "frame_count": self.frame_count,
                "active_frame_count": self.active_frame_count,
                "source": source.tolist(),
                "field": field.tolist(),
                "polarization_energy_eV": energy,
            }
        )
        if self.state_hash != expected:
            raise ValueError("pair-frame state hash does not match its contents.")
        if self.capabilities.enabled_tiers:
            raise ValueError("pair-frame E/F/H/V/M capabilities remain closed.")

    @property
    def source(self) -> np.ndarray:
        result = np.asarray(self.source_values, dtype=float).reshape(self.atom_count, 8)
        result.setflags(write=False)
        return result

    @property
    def reaction_field(self) -> np.ndarray:
        result = np.asarray(self.field_values, dtype=float).reshape(self.atom_count, 8)
        result.setflags(write=False)
        return result


__all__ = [
    "PAIR_FRAME_CPCM_COUPLING_ID",
    "PAIR_FRAME_CPCM_GRID_POINTS_PER_ATOM",
    "PAIR_FRAME_CPCM_PROVIDER_ID",
    "PairFrameCPCMState",
    "pair_frame_state_sha256",
]
