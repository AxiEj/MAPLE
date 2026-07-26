from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .protocol import canonical_sha256


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PackingBiasState:
    label: str
    bias_scale: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError(
                "Packing bias state label must be a non-empty string."
            )
        if (
            isinstance(self.bias_scale, (bool, np.bool_))
            or not math.isfinite(float(self.bias_scale))
            or not 0.0 <= float(self.bias_scale) <= 1.0
        ):
            raise ValueError(
                "Packing bias scale must be finite and in [0, 1]."
            )


@dataclass(frozen=True)
class PackingSchedule:
    """Hash-bound state contract for a pure-water soft-cavity calculation."""

    states: tuple[PackingBiasState, ...]
    target_state_index: int
    full_field_state_index: int
    membership_definition_hash: str
    observation_volume_hash: str
    boundary_adapter_hash: str
    conditioning_measure_id: str
    solute_measure_hash: str
    water_hamiltonian_hash: str
    oxygen_atom_map_hash: str
    solute_atom_map_hash: str
    cell_hash: str
    active_occupancy_max: int
    temperature_k: float
    ensemble: str
    pressure_bar: float | None
    boundary_conditions: str

    def __post_init__(self) -> None:
        states = tuple(self.states)
        if len(states) < 2 or any(
            not isinstance(state, PackingBiasState) for state in states
        ):
            raise ValueError(
                "Packing schedule requires at least two PackingBiasState values."
            )
        labels = tuple(state.label for state in states)
        scales = np.asarray(
            [float(state.bias_scale) for state in states],
            dtype=float,
        )
        if len(set(labels)) != len(labels):
            raise ValueError("Packing schedule labels must be unique.")
        if len(set(scales.tolist())) != len(scales):
            raise ValueError("Packing schedule bias scales must be unique.")
        differences = np.diff(scales)
        if not (
            np.all(differences > 0.0) or np.all(differences < 0.0)
        ):
            raise ValueError(
                "Packing schedule bias scales must be strictly monotonic."
            )
        if np.count_nonzero(scales == 0.0) != 1:
            raise ValueError(
                "Packing schedule requires one unbiased target state at scale 0."
            )
        if np.count_nonzero(scales == 1.0) != 1:
            raise ValueError(
                "Packing schedule requires one full-field state at scale 1."
            )

        target_index = self._validate_state_index(
            self.target_state_index,
            len(states),
            "target_state_index",
        )
        full_index = self._validate_state_index(
            self.full_field_state_index,
            len(states),
            "full_field_state_index",
        )
        if target_index == full_index:
            raise ValueError(
                "Packing target and full-field states must be different."
            )
        if float(states[target_index].bias_scale) != 0.0:
            raise ValueError(
                "target_state_index must identify the scale-0 target state."
            )
        if float(states[full_index].bias_scale) != 1.0:
            raise ValueError(
                "full_field_state_index must identify the scale-1 full-field state."
            )

        for name in (
            "membership_definition_hash",
            "observation_volume_hash",
            "boundary_adapter_hash",
            "solute_measure_hash",
            "water_hamiltonian_hash",
            "oxygen_atom_map_hash",
            "solute_atom_map_hash",
            "cell_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 hash.")
        if (
            isinstance(self.active_occupancy_max, (bool, np.bool_))
            or int(self.active_occupancy_max) != self.active_occupancy_max
            or int(self.active_occupancy_max) < 1
        ):
            raise ValueError(
                "active_occupancy_max must include n=0 and at least n=1."
            )
        if (
            not isinstance(self.conditioning_measure_id, str)
            or not self.conditioning_measure_id
        ):
            raise ValueError(
                "conditioning_measure_id must be a non-empty string."
            )
        if (
            isinstance(self.temperature_k, (bool, np.bool_))
            or not math.isfinite(float(self.temperature_k))
            or float(self.temperature_k) <= 0.0
        ):
            raise ValueError("temperature_k must be finite and positive.")
        if self.ensemble != "NVT":
            raise ValueError(
                "Packing production schedules currently require NVT; "
                "NPT cell-ensemble identity and stress are not implemented."
            )
        if self.pressure_bar is not None:
            raise ValueError(
                "NVT packing schedules must set pressure_bar to null."
            )
        if self.boundary_conditions != "periodic-3d":
            raise ValueError(
                "Packing boundary_conditions must be periodic-3d."
            )

        object.__setattr__(self, "states", states)
        object.__setattr__(self, "target_state_index", target_index)
        object.__setattr__(self, "full_field_state_index", full_index)
        object.__setattr__(
            self,
            "active_occupancy_max",
            int(self.active_occupancy_max),
        )

    @staticmethod
    def _validate_state_index(
        value: int,
        state_count: int,
        name: str,
    ) -> int:
        if (
            isinstance(value, (bool, np.bool_))
            or int(value) != value
            or not 0 <= int(value) < state_count
        ):
            raise ValueError(
                f"{name} must identify one packing schedule state."
            )
        return int(value)

    @property
    def target_state(self) -> PackingBiasState:
        return self.states[self.target_state_index]

    @property
    def full_field_state(self) -> PackingBiasState:
        return self.states[self.full_field_state_index]

    @property
    def row_labels(self) -> tuple[str, ...]:
        return tuple(state.label for state in self.states)

    @property
    def content_hash(self) -> str:
        return canonical_sha256(
            {
                "contract_id": "packing-schedule-v1",
                "states": [
                    {
                        "label": state.label,
                        "bias_scale": float(state.bias_scale),
                    }
                    for state in self.states
                ],
                "target_state_index": self.target_state_index,
                "full_field_state_index": self.full_field_state_index,
                "membership_definition_hash": (
                    self.membership_definition_hash
                ),
                "observation_volume_hash": self.observation_volume_hash,
                "boundary_adapter_hash": self.boundary_adapter_hash,
                "conditioning_measure_id": self.conditioning_measure_id,
                "solute_measure_hash": self.solute_measure_hash,
                "water_hamiltonian_hash": self.water_hamiltonian_hash,
                "oxygen_atom_map_hash": self.oxygen_atom_map_hash,
                "solute_atom_map_hash": self.solute_atom_map_hash,
                "cell_hash": self.cell_hash,
                "active_occupancy_max": self.active_occupancy_max,
                "temperature_k": float(self.temperature_k),
                "ensemble": self.ensemble,
                "pressure_bar": (
                    None
                    if self.pressure_bar is None
                    else float(self.pressure_bar)
                ),
                "boundary_conditions": self.boundary_conditions,
            }
        )


__all__ = ["PackingBiasState", "PackingSchedule"]
