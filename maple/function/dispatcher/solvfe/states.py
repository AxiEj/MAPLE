from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .protocol import canonical_sha256


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class CalculatorSpec:
    role: str
    name: str
    checkpoint: str
    sha256: str
    license_acknowledged: bool
    capabilities: frozenset[str]
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (
            ("role", self.role),
            ("name", self.name),
            ("checkpoint", self.checkpoint),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Calculator {label} must be a non-empty string.")
        if not _SHA256_RE.fullmatch(str(self.sha256)):
            raise ValueError("Calculator sha256 must be 64 lowercase hex characters.")
        if self.license_acknowledged is not True:
            raise ValueError(
                f"Calculator role '{self.role}' requires explicit license acknowledgement."
            )
        if not self.capabilities or any(
            not isinstance(item, str) or not item
            for item in self.capabilities
        ):
            raise ValueError("Calculator capabilities must be a non-empty string set.")
        object.__setattr__(self, "options", _frozen_mapping(self.options))

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "checkpoint": self.checkpoint,
            "sha256": self.sha256,
            "license_acknowledged": self.license_acknowledged,
            "capabilities": sorted(self.capabilities),
            "options": dict(self.options),
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.as_dict())


@dataclass(frozen=True)
class SolvFERequest:
    method: str
    solvent: str
    temperature: float
    pressure_bar: float
    standard_state: str
    protocol: str
    experimental: bool
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.method != "qct":
            raise ValueError("Route A requires method=qct.")
        if self.solvent != "water":
            raise ValueError("Route A v1 requires solvent=water.")
        if self.standard_state != "1m":
            raise ValueError("Route A v1 requires standard_state=1m.")
        if self.experimental is not True:
            raise ValueError("Route A requires experimental=true.")
        if isinstance(self.temperature, bool) or self.temperature != 298.15:
            raise ValueError("Route A v1 requires temperature=298.15 K.")
        if isinstance(self.pressure_bar, bool) or self.pressure_bar != 1.0:
            raise ValueError("Route A v1 requires pressure_bar=1.0.")
        if not isinstance(self.protocol, str) or not self.protocol.strip():
            raise ValueError("Route A requires a non-empty protocol path.")
        if not isinstance(self.dry_run, bool):
            raise ValueError("Route A dry_run must be true or false.")

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "SolvFERequest":
        return cls(
            method=str(params.get("method", "")).lower(),
            solvent=str(params.get("solvent", "")).lower(),
            temperature=params.get("temperature"),
            pressure_bar=params.get("pressure_bar"),
            standard_state=str(params.get("standard_state", "")).lower(),
            protocol=str(params.get("protocol", "")),
            experimental=params.get("experimental"),
            dry_run=params.get("dry_run", False),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "solvent": self.solvent,
            "temperature": self.temperature,
            "pressure_bar": self.pressure_bar,
            "standard_state": self.standard_state,
            "protocol": self.protocol,
            "experimental": self.experimental,
            "dry_run": self.dry_run,
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.as_dict())
