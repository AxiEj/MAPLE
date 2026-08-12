"""Provider-independent continuum response contracts."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.coupling.spaces import FieldDualSpace, SourceSpace


@runtime_checkable
class ContinuumBackend(Protocol):
    provider_id: str
    continuum_profile_id: str
    cavity_profile_id: str
    configuration_contract_id: str
    coupling_id: str
    scalar_id: str

    def configuration_sha256(self) -> str: ...

    provenance_sha256: str
    capabilities: CapabilityStatus
    source_space: SourceSpace
    field_space: FieldDualSpace
    fixed_topology: bool
    linear_response: bool
    reciprocal: bool

    def build_state(self, geometry: Any, source: np.ndarray): ...
    def energy(self, geometry: Any, source: np.ndarray) -> float: ...
    def field(self, geometry: Any, source: np.ndarray) -> np.ndarray: ...
    def evaluate_field(self, geometry: Any, source: np.ndarray) -> np.ndarray: ...
    def source_jvp(
        self, geometry: Any, source: np.ndarray, source_direction: np.ndarray
    ) -> np.ndarray: ...
    def source_vjp(
        self, geometry: Any, source: np.ndarray, field_cotangent: np.ndarray
    ) -> np.ndarray: ...
    def coordinate_vjp(
        self, geometry: Any, source: np.ndarray, field_cotangent: np.ndarray
    ) -> np.ndarray: ...


__all__ = ["ContinuumBackend"]
