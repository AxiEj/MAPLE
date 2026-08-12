"""Provider-independent immutable-surface contracts for Route 2."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SurfaceSnapshot(Protocol):
    """Read-only identity of one cavity discretization at one geometry."""

    provider_id: str
    cavity_profile_id: str
    configuration_sha256: str
    provenance_sha256: str
    topology_hash: str
    state_hash: str
    candidate_count: int
    fixed_topology: bool


@runtime_checkable
class SurfaceProvider(Protocol):
    """Build a validated immutable surface snapshot."""

    provider_id: str
    cavity_profile_id: str
    configuration_sha256: str
    provenance_sha256: str
    fixed_topology: bool

    def build_state(self, geometry: Any) -> SurfaceSnapshot: ...


__all__ = ["SurfaceProvider", "SurfaceSnapshot"]
