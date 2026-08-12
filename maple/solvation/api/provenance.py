"""Immutable provenance records shared by model, continuum, and cavity APIs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _metadata_items(
    value: Mapping[str, object] | tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    normalized = tuple(sorted((_text(k, "metadata key"), _text(v, "metadata value")) for k, v in items))
    if len({key for key, _ in normalized}) != len(normalized):
        raise ValueError("Provenance metadata keys must be unique.")
    return normalized


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Content-addressed identity for one scientific provider or artifact."""

    identity: str
    kind: str
    version: str
    sha256: str
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", _text(self.identity, "identity"))
        object.__setattr__(self, "kind", _text(self.kind, "kind"))
        object.__setattr__(self, "version", _text(self.version, "version"))
        digest = _text(self.sha256, "sha256").lower()
        if not _SHA256.fullmatch(digest):
            raise ValueError("sha256 must contain exactly 64 lowercase hexadecimal digits.")
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "metadata", _metadata_items(self.metadata))


@dataclass(frozen=True, slots=True)
class RuntimeProvenance:
    python: str
    platform: str
    dependencies: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "python", _text(self.python, "python"))
        object.__setattr__(self, "platform", _text(self.platform, "platform"))
        object.__setattr__(self, "dependencies", _metadata_items(self.dependencies))


@dataclass(frozen=True, slots=True)
class ProvenanceBundle:
    model: ProvenanceRecord
    continuum: ProvenanceRecord
    cavity: ProvenanceRecord
    runtime: RuntimeProvenance

    def __post_init__(self) -> None:
        expected = {"model": self.model, "continuum": self.continuum, "cavity": self.cavity}
        for kind, record in expected.items():
            if not isinstance(record, ProvenanceRecord):
                raise TypeError(f"{kind} provenance must be a ProvenanceRecord.")
            if record.kind != kind:
                raise ValueError(f"{kind} provenance record must declare kind={kind!r}.")
        if not isinstance(self.runtime, RuntimeProvenance):
            raise TypeError("runtime provenance must be RuntimeProvenance.")


__all__ = ["ProvenanceBundle", "ProvenanceRecord", "RuntimeProvenance"]
