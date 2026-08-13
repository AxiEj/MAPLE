"""Candidate-neutral topology identity for frozen Route-2 panel runners."""

from __future__ import annotations


def continuum_topology_hash(continuum, geometry) -> str:
    """Return the immutable member/surface topology identity at ``geometry``."""

    diagnostics = getattr(continuum, "frame_diagnostics", None)
    if callable(diagnostics):
        value = diagnostics(geometry).get("frame_topology_sha256")
    else:
        provider = getattr(continuum, "surface_provider", None)
        build_state = getattr(provider, "build_state", None)
        if not callable(build_state):
            raise TypeError("continuum does not expose a topology identity contract.")
        value = build_state(geometry).topology_hash
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("continuum topology identity must be a SHA256 digest.")
    return value


__all__ = ["continuum_topology_hash"]
