"""Explicit MLIP factories and immutable historical-artifact dispositions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from .route2_plugin_contracts import (
    ElectronicSourceProvider,
    validate_plugin_source_provider,
)
from .route2_plugin_errors import PluginContractError

ArtifactDisposition = Literal[
    "accepted-current",
    "legacy-unresolved",
    "rejected",
    "superseded",
]


def _nonempty(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


@dataclass(frozen=True)
class ArtifactDispositionRecord:
    """Immutable disposition for a historical artifact; the artifact is not edited."""

    artifact_id: str
    artifact_path: str
    artifact_sha256: str
    disposition: ArtifactDisposition
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artifact_id", _nonempty(self.artifact_id, name="artifact id")
        )
        path = _nonempty(self.artifact_path, name="artifact path")
        if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise ValueError(
                "artifact path must be repository-relative and normalized."
            )
        object.__setattr__(self, "artifact_path", path)
        sha256 = _nonempty(self.artifact_sha256, name="artifact sha256").lower()
        if len(sha256) != 64 or any(
            character not in "0123456789abcdef" for character in sha256
        ):
            raise ValueError(
                "artifact sha256 must contain 64 lowercase hexadecimal characters."
            )
        object.__setattr__(self, "artifact_sha256", sha256)
        object.__setattr__(
            self, "rationale", _nonempty(self.rationale, name="artifact rationale")
        )
        if self.disposition not in {
            "accepted-current",
            "legacy-unresolved",
            "rejected",
            "superseded",
        }:
            raise ValueError("unsupported artifact disposition.")


@dataclass(frozen=True)
class ArtifactDispositionRegistry:
    """Append-only-style registry used instead of rewriting historical JSON."""

    records: tuple[ArtifactDispositionRecord, ...] = ()

    def __post_init__(self) -> None:
        identifiers = [record.artifact_id for record in self.records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(
                "artifact disposition registry has duplicate artifact ids."
            )

    def disposition_for(self, artifact_id: str) -> ArtifactDispositionRecord | None:
        return next(
            (record for record in self.records if record.artifact_id == artifact_id),
            None,
        )

    @classmethod
    def from_mapping(cls, payload: object) -> "ArtifactDispositionRegistry":
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("artifact disposition registry requires schema_version=1.")
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise TypeError("artifact disposition registry records must be a list.")
        records = []
        for raw in raw_records:
            if not isinstance(raw, dict):
                raise TypeError("artifact disposition entries must be mappings.")
            records.append(
                ArtifactDispositionRecord(
                    artifact_id=raw.get("artifact_id", ""),
                    artifact_path=raw.get("artifact_path", ""),
                    artifact_sha256=raw.get("artifact_sha256", ""),
                    disposition=raw.get("disposition", ""),
                    rationale=raw.get("rationale", ""),
                )
            )
        return cls(tuple(records))


class Route2PluginRegistry:
    """Explicit model-family factory registry; no method-name discovery occurs."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., ElectronicSourceProvider]] = {}

    def register(
        self,
        plugin_id: str,
        factory: Callable[..., ElectronicSourceProvider],
        *,
        replace: bool = False,
    ) -> None:
        key = _nonempty(plugin_id, name="plug-in registry id")
        if not callable(factory):
            raise TypeError("plug-in factory must be callable.")
        if key in self._factories and not replace:
            raise ValueError(f"plug-in factory already registered: {key!r}.")
        self._factories[key] = factory

    def create(
        self, plugin_id: str, /, *args: object, **kwargs: object
    ) -> ElectronicSourceProvider:
        key = _nonempty(plugin_id, name="plug-in registry id")
        try:
            factory = self._factories[key]
        except KeyError as exc:
            raise PluginContractError(
                f"No Route-2 plug-in factory is registered for {key!r}."
            ) from exc
        plugin = factory(*args, **kwargs)
        provider = validate_plugin_source_provider(plugin)
        if provider.plugin_id != key:
            raise PluginContractError(
                f"plug-in factory registered for {key!r} returned {provider.plugin_id!r}."
            )
        return provider

    @property
    def registered_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


__all__ = [
    "ArtifactDisposition",
    "ArtifactDispositionRecord",
    "ArtifactDispositionRegistry",
    "Route2PluginRegistry",
]
