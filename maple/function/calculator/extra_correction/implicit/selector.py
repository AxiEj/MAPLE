"""Pure provider selection for implicit-solvent public and direct APIs."""

from __future__ import annotations

from typing import Any, Mapping

_LEGACY_OPENMM_MODELS = frozenset({"hct", "obc1", "gbn", "gbn2"})
_OPENMM_ONLY_OPTIONS = frozenset(
    {"platform", "precision", "device_index", "opencl_platform_index"}
)


def resolve_implicit_provider(options: Mapping[str, Any]) -> str:
    """Resolve one provider without importing or probing any backend.

    Explicit selection always wins.  For GB, the public OBC-II default is the
    independent Torch implementation; legacy Amber GB models retain OpenMM.
    Options that only OpenMM can interpret never silently switch providers.
    """

    method = str(options.get("method", "")).strip().lower()
    explicit = options.get("provider")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ValueError("Solvation provider must be a non-empty string.")
        provider = explicit.strip().lower()
    elif method == "gb":
        model = str(options.get("model", "obc2")).strip().lower()
        if model == "chagb":
            provider = "private-torch-cha"
        else:
            provider = "openmm" if model in _LEGACY_OPENMM_MODELS else "torch"
    elif method == "pb":
        provider = "apbs"
    else:
        return ""

    if method == "gb" and provider == "torch":
        conflicts = sorted(_OPENMM_ONLY_OPTIONS.intersection(options))
        if options.get("inner") is not None:
            conflicts.append("inner")
        if str(options.get("nonpolar", "ace")).strip().lower() == "lcpo":
            conflicts.append("nonpolar=lcpo")
        if conflicts:
            raise ValueError(
                "Independent Torch OBC-II does not use OpenMM-only options "
                f"({', '.join(conflicts)}); request provider=openmm explicitly."
            )
    return provider


__all__ = ["resolve_implicit_provider"]
