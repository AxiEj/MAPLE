"""Hash-bound solvent-side short-range provenance for Route-2 V0.

The RISM short-range remainder is derived only from the frozen MDL/input/XVV/
Cvv source at its native SMEAR.  This certificate binds that derivation to the
other solvent files without carrying a MACE checkpoint, fitted coefficient, or
target-solvation label.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

V0_RISM_SHORT_RANGE_SOURCE_CONSTRUCTION = "route2-v0-rism-short-range-source-v1"
V0_RISM_SHORT_RANGE_SOURCE_SCOPE = "solvent-side"
V0_RISM_SHORT_RANGE_DERIVATION = "native-smear-split-from-rism1d-cvv-v1"
V0_RISM_SHORT_RANGE_HASH_ROLES = frozenset(
    {
        "site_model",
        "rism1d_input",
        "xvv",
        "cvv",
        "thermodynamic_output",
        "provenance_statement",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ROOT_KEYS = frozenset(
    {
        "construction",
        "source_scope",
        "derivation",
        "closure",
        "temperature_kelvin",
        "pressure_bar",
        "coulomb_tail_start_angstrom",
        "coulomb_tail_tolerance_dimensionless",
        "target_solvation_labels_used",
        "source_sha256",
    }
)


def _strict_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object with string keys.")
    return value


def _strict_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, name: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{name} keys differ; missing={missing}, extra={extra}.")


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value.strip()


def _positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{name} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


@dataclass(frozen=True)
class Route2V0RismShortRangeSource:
    """One strict solvent-side native-SMEAR derivation certificate."""

    closure: str
    temperature_kelvin: float
    pressure_bar: float
    coulomb_tail_start_angstrom: float
    coulomb_tail_tolerance_dimensionless: float
    source_sha256: Mapping[str, str]
    target_solvation_labels_used: bool = False
    construction: str = V0_RISM_SHORT_RANGE_SOURCE_CONSTRUCTION
    source_scope: str = V0_RISM_SHORT_RANGE_SOURCE_SCOPE
    derivation: str = V0_RISM_SHORT_RANGE_DERIVATION

    def __post_init__(self) -> None:
        closure = _nonempty(self.closure, name="Short-range source closure")
        temperature = _positive(self.temperature_kelvin, name="Short-range temperature")
        pressure = _positive(self.pressure_bar, name="Short-range pressure")
        tail_start = _positive(
            self.coulomb_tail_start_angstrom,
            name="Short-range Coulomb-tail start",
        )
        tail_tolerance = _positive(
            self.coulomb_tail_tolerance_dimensionless,
            name="Short-range Coulomb-tail tolerance",
        )
        hashes = _strict_mapping(self.source_sha256, name="Short-range source hashes")
        _strict_keys(hashes, V0_RISM_SHORT_RANGE_HASH_ROLES, name="source_sha256")
        normalized_hashes: dict[str, str] = {}
        for role, value in hashes.items():
            digest = _nonempty(value, name=f"Short-range {role} SHA-256").lower()
            if _SHA256.fullmatch(digest) is None:
                raise ValueError(
                    f"Short-range {role} SHA-256 must be a lowercase digest."
                )
            normalized_hashes[role] = digest
        if self.target_solvation_labels_used is not False:
            raise ValueError(
                "Short-range source must explicitly exclude target labels."
            )
        if self.construction != V0_RISM_SHORT_RANGE_SOURCE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 RISM short-range construction.")
        if self.source_scope != V0_RISM_SHORT_RANGE_SOURCE_SCOPE:
            raise ValueError("RISM short-range source scope must be solvent-side.")
        if self.derivation != V0_RISM_SHORT_RANGE_DERIVATION:
            raise ValueError("Unsupported RISM short-range derivation.")
        object.__setattr__(self, "closure", closure)
        object.__setattr__(self, "temperature_kelvin", temperature)
        object.__setattr__(self, "pressure_bar", pressure)
        object.__setattr__(self, "coulomb_tail_start_angstrom", tail_start)
        object.__setattr__(self, "coulomb_tail_tolerance_dimensionless", tail_tolerance)
        object.__setattr__(
            self,
            "source_sha256",
            MappingProxyType(dict(sorted(normalized_hashes.items()))),
        )


def parse_route2_v0_rism_short_range_source(text: str) -> Route2V0RismShortRangeSource:
    """Parse one strict JSON solvent-side short-range certificate."""

    if not isinstance(text, str):
        raise TypeError("RISM short-range certificate must be text.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("RISM short-range certificate must be valid JSON.") from exc
    root = _strict_mapping(payload, name="RISM short-range certificate")
    _strict_keys(root, _ROOT_KEYS, name="RISM short-range certificate")
    return Route2V0RismShortRangeSource(
        closure=root["closure"],
        temperature_kelvin=root["temperature_kelvin"],
        pressure_bar=root["pressure_bar"],
        coulomb_tail_start_angstrom=root["coulomb_tail_start_angstrom"],
        coulomb_tail_tolerance_dimensionless=root[
            "coulomb_tail_tolerance_dimensionless"
        ],
        source_sha256=_strict_mapping(root["source_sha256"], name="source_sha256"),
        target_solvation_labels_used=root["target_solvation_labels_used"],
        construction=root["construction"],
        source_scope=root["source_scope"],
        derivation=root["derivation"],
    )


def load_route2_v0_rism_short_range_source(
    path: str | Path,
) -> Route2V0RismShortRangeSource:
    """Load one strict JSON solvent-side short-range certificate."""

    return parse_route2_v0_rism_short_range_source(
        Path(path).read_text(encoding="utf-8")
    )


__all__ = [
    "V0_RISM_SHORT_RANGE_DERIVATION",
    "V0_RISM_SHORT_RANGE_HASH_ROLES",
    "V0_RISM_SHORT_RANGE_SOURCE_CONSTRUCTION",
    "V0_RISM_SHORT_RANGE_SOURCE_SCOPE",
    "Route2V0RismShortRangeSource",
    "load_route2_v0_rism_short_range_source",
    "parse_route2_v0_rism_short_range_source",
]
