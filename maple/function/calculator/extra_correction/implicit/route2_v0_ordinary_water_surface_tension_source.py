"""Source-only IAPWS surface-tension anchor for ordinary liquid water.

This source is intentionally *not* a cSPC/E state record.  The cSPC/E RISM
model adds coincident hydrogen Lennard-Jones sites, so ordinary-water IAPWS
thermophysics cannot be silently relabelled as a state of that model.  The
record supplies only an independently sourced liquid--vapour surface-tension
target for a future pure-water bridge certificate.  It cannot construct a
finite-k correlation, choose a Gaussian width, admit a molecular liquid, or
produce an accuracy result.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ase.units import Bohr, Hartree

from .route2_v0_bulk_liquid_state_source import (
    V0_BULK_LIQUID_STATE_REQUIRED_NO_TARGET_KEYS,
    Route2V0BulkLiquidPropertySource,
)

V0_ORDINARY_WATER_SURFACE_TENSION_SOURCE_CONSTRUCTION = (
    "route2-v0-ordinary-water-surface-tension-source-v1"
)
V0_ORDINARY_WATER_SURFACE_TENSION_SOURCE_STATUS = (
    "source-only-ordinary-water-surface-tension-not-molecular-rism-or-accuracy-admitted"
)

_SCHEMA_VERSION = 1
_ROOT_KEYS = frozenset(
    {
        "protocol_id",
        "schema_version",
        "status",
        "substance",
        "state",
        "formula",
        "surface_tension_newton_per_meter",
        "property_source",
        "no_target_policy",
        "claim_boundary",
        "not_claimed",
    }
)
_STATE_KEYS = frozenset({"temperature_kelvin"})
_FORMULA_KEYS = frozenset(
    {
        "critical_temperature_kelvin",
        "prefactor_newton_per_meter",
        "exponent",
        "linear_coefficient",
    }
)
_PROPERTY_SOURCE_KEYS = frozenset(
    {
        "property",
        "origin",
        "document_url",
        "document_sha256",
        "source_locator",
        "retrieved_utc",
    }
)
_EXACT_SUBSTANCE = "ordinary-water-liquid-vapor-interface"
_TRIPLE_POINT_KELVIN = 273.16
_IAPWS_TEMPERATURE_KELVIN = 298.0
_IAPWS_CRITICAL_TEMPERATURE_KELVIN = 647.096
_IAPWS_PREFACTOR_NEWTON_PER_METER = 0.2358
_IAPWS_EXPONENT = 1.256
_IAPWS_LINEAR_COEFFICIENT = -0.625
_IAPWS_DOCUMENT_URL = "https://iapws.org/documents/release/Surf-H2O.download"
_IAPWS_DOCUMENT_SHA256 = (
    "77d07ae4c6d473806b73f982d4c031b927db02a5718bd3c25cd7a03c3321587d"
)
_EXPECTED_CLAIM_BOUNDARY = (
    "This source is the IAPWS ordinary-water liquid-vapor surface-tension target "
    "at 298.0 K. It can become an independently sourced target for a future "
    "same-scalar outer planar-interface certificate; it is not a cSPC/E "
    "molecular-RISM state, an HNC correlation, a bridge width selection, a liquid "
    "asset, a solvation free energy, a force/PES result, or an accuracy result."
)
_EXPECTED_NOT_CLAIMED = (
    (
        "The cSPC/E RISM model has coincident hydrogen Lennard-Jones sites and must "
        "not inherit this ordinary-water target as a model state."
    ),
    (
        "This source does not determine a finite-k molecular direct correlation, a "
        "closure, a short-range solvent interaction, or a molecular liquid functional."
    ),
    (
        "This source does not choose a Gaussian kernel width; the width must follow a "
        "nested same-scalar coexistence and planar stationary certificate."
    ),
    (
        "This source uses no solvation target labels and does not establish a solvation "
        "accuracy result."
    ),
)
_ELECTRONIC_CHARGE_JOULE = 1.602176634e-19
_HARTREE_PER_BOHR2_TO_NEWTON_PER_METER = (
    Hartree * _ELECTRONIC_CHARGE_JOULE / (Bohr * 1.0e-10) ** 2
)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object with string keys.")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise TypeError(f"{name} must be a JSON object with string keys.")
    return {cast(str, key): item for key, item in raw.items()}


def _strict_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise ValueError(f"{name} keys differ: missing={missing}, extra={extra}.")


def _text(value: object, *, name: str) -> str:
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


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonempty_strings(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f"{name} must be a sequence.")
    result = tuple(_text(item, name=name) for item in value)
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return result


def _surface_tension_formula(
    *,
    temperature_kelvin: float,
    critical_temperature_kelvin: float,
    prefactor_newton_per_meter: float,
    exponent: float,
    linear_coefficient: float,
) -> float:
    """Evaluate IAPWS R1-76(2014) without a fitted local parameter."""

    reduced_distance = 1.0 - temperature_kelvin / critical_temperature_kelvin
    if reduced_distance <= 0.0:
        raise ValueError("Surface-tension temperature must remain below critical.")
    value = (
        prefactor_newton_per_meter
        * reduced_distance**exponent
        * (1.0 + linear_coefficient * reduced_distance)
    )
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("IAPWS surface-tension formula produced an invalid value.")
    return value


@dataclass(frozen=True)
class Route2V0OrdinaryWaterSurfaceTensionSource:
    """One content-addressed ordinary-water target, explicitly outside cSPC/E."""

    temperature_kelvin: float
    critical_temperature_kelvin: float
    prefactor_newton_per_meter: float
    exponent: float
    linear_coefficient: float
    surface_tension_newton_per_meter: float
    property_source: Route2V0BulkLiquidPropertySource
    claim_boundary: str
    not_claimed: tuple[str, ...]
    construction: str = V0_ORDINARY_WATER_SURFACE_TENSION_SOURCE_CONSTRUCTION
    status: str = V0_ORDINARY_WATER_SURFACE_TENSION_SOURCE_STATUS

    def __post_init__(self) -> None:
        temperature = _positive(self.temperature_kelvin, name="Temperature")
        critical = _positive(
            self.critical_temperature_kelvin,
            name="Critical temperature",
        )
        if not _TRIPLE_POINT_KELVIN <= temperature < critical:
            raise ValueError(
                "IAPWS surface-tension source temperature must lie from the triple "
                "point to below the critical point."
            )
        prefactor = _positive(
            self.prefactor_newton_per_meter,
            name="Surface-tension prefactor",
        )
        exponent = _positive(self.exponent, name="Surface-tension exponent")
        linear = _finite(
            self.linear_coefficient,
            name="Surface-tension linear coefficient",
        )
        surface_tension = _positive(
            self.surface_tension_newton_per_meter,
            name="Surface tension",
        )
        if (
            temperature,
            critical,
            prefactor,
            exponent,
            linear,
        ) != (
            _IAPWS_TEMPERATURE_KELVIN,
            _IAPWS_CRITICAL_TEMPERATURE_KELVIN,
            _IAPWS_PREFACTOR_NEWTON_PER_METER,
            _IAPWS_EXPONENT,
            _IAPWS_LINEAR_COEFFICIENT,
        ):
            raise ValueError("IAPWS R1-76(2014) constants changed.")
        source = cast(object, self.property_source)
        if not isinstance(source, Route2V0BulkLiquidPropertySource):
            raise TypeError("Surface-tension source requires a property source record.")
        if source.property_name != "surface_tension_newton_per_meter":
            raise ValueError(
                "Surface-tension source has an incompatible property name."
            )
        if source.origin != "independent_measurement":
            raise ValueError(
                "Ordinary-water surface tension must retain independent-measurement provenance."
            )
        if (
            source.document_url != _IAPWS_DOCUMENT_URL
            or source.document_sha256 != _IAPWS_DOCUMENT_SHA256
        ):
            raise ValueError("IAPWS R1-76(2014) source identity changed.")
        expected = _surface_tension_formula(
            temperature_kelvin=temperature,
            critical_temperature_kelvin=critical,
            prefactor_newton_per_meter=prefactor,
            exponent=exponent,
            linear_coefficient=linear,
        )
        if not math.isclose(
            surface_tension,
            expected,
            rel_tol=1.0e-14,
            abs_tol=1.0e-16,
        ):
            raise ValueError(
                "IAPWS surface-tension record does not reproduce its declared formula."
            )
        claim = _text(self.claim_boundary, name="Surface-tension claim boundary")
        nonclaims = _nonempty_strings(
            self.not_claimed,
            name="Surface-tension non-claim",
        )
        if claim != _EXPECTED_CLAIM_BOUNDARY:
            raise ValueError("Surface-tension claim boundary changed.")
        if nonclaims != _EXPECTED_NOT_CLAIMED:
            raise ValueError("Surface-tension non-claim boundary changed.")
        if self.construction != V0_ORDINARY_WATER_SURFACE_TENSION_SOURCE_CONSTRUCTION:
            raise ValueError("Unsupported ordinary-water surface-tension construction.")
        if self.status != V0_ORDINARY_WATER_SURFACE_TENSION_SOURCE_STATUS:
            raise ValueError(
                "Ordinary-water surface-tension record must remain source-only."
            )
        object.__setattr__(self, "temperature_kelvin", temperature)
        object.__setattr__(self, "critical_temperature_kelvin", critical)
        object.__setattr__(self, "prefactor_newton_per_meter", prefactor)
        object.__setattr__(self, "exponent", exponent)
        object.__setattr__(self, "linear_coefficient", linear)
        object.__setattr__(self, "surface_tension_newton_per_meter", surface_tension)
        object.__setattr__(self, "property_source", source)
        object.__setattr__(self, "claim_boundary", claim)
        object.__setattr__(self, "not_claimed", nonclaims)

    @property
    def reduced_temperature_distance(self) -> float:
        """Return ``tau = 1 - T/Tc`` in the IAPWS convention."""

        return 1.0 - self.temperature_kelvin / self.critical_temperature_kelvin

    @property
    def surface_tension_hartree_per_bohr2(self) -> float:
        """Return the exact-SI-converted target usable by a future certificate."""

        return self.surface_tension_newton_per_meter / (
            _HARTREE_PER_BOHR2_TO_NEWTON_PER_METER
        )

    @property
    def is_molecular_liquid_asset(self) -> bool:
        """A macroscopic target never supplies a molecular liquid functional."""

        return False

    @property
    def is_molecular_rism_state_source(self) -> bool:
        """This ordinary-water source cannot be relabelled as cSPC/E RISM state."""

        return False


def _property_source_from_json(value: object) -> Route2V0BulkLiquidPropertySource:
    payload = _mapping(value, name="Surface-tension property source")
    _strict_keys(
        payload,
        _PROPERTY_SOURCE_KEYS,
        name="Surface-tension property source",
    )
    return Route2V0BulkLiquidPropertySource(
        property_name=_text(payload["property"], name="Surface-tension property"),
        origin=_text(payload["origin"], name="Surface-tension source origin"),
        document_url=_text(
            payload["document_url"],
            name="Surface-tension document URL",
        ),
        document_sha256=_text(
            payload["document_sha256"],
            name="Surface-tension document SHA-256",
        ),
        source_locator=_text(
            payload["source_locator"],
            name="Surface-tension source locator",
        ),
        retrieved_utc=_text(
            payload["retrieved_utc"],
            name="Surface-tension retrieval timestamp",
        ),
    )


def parse_route2_v0_ordinary_water_surface_tension_source(
    text: str,
) -> Route2V0OrdinaryWaterSurfaceTensionSource:
    """Parse one strict source-only IAPWS ordinary-water target record."""

    if not isinstance(text, str):
        raise TypeError("Ordinary-water surface-tension source must be text.")
    try:
        payload = cast(object, json.loads(text))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Ordinary-water surface-tension source must be valid JSON."
        ) from exc
    root = _mapping(payload, name="Ordinary-water surface-tension source")
    _strict_keys(root, _ROOT_KEYS, name="Ordinary-water surface-tension source")
    if root["protocol_id"] != V0_ORDINARY_WATER_SURFACE_TENSION_SOURCE_CONSTRUCTION:
        raise ValueError("Unsupported ordinary-water surface-tension protocol.")
    if root["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("Unsupported ordinary-water surface-tension schema.")
    if root["status"] != V0_ORDINARY_WATER_SURFACE_TENSION_SOURCE_STATUS:
        raise ValueError(
            "Ordinary-water surface-tension record must remain source-only."
        )
    if root["substance"] != _EXACT_SUBSTANCE:
        raise ValueError("Ordinary-water surface-tension substance changed.")
    state = _mapping(root["state"], name="Ordinary-water surface-tension state")
    _strict_keys(state, _STATE_KEYS, name="Ordinary-water surface-tension state")
    formula = _mapping(root["formula"], name="IAPWS surface-tension formula")
    _strict_keys(formula, _FORMULA_KEYS, name="IAPWS surface-tension formula")
    policy = _mapping(root["no_target_policy"], name="Surface-tension no-target policy")
    _strict_keys(
        policy,
        V0_BULK_LIQUID_STATE_REQUIRED_NO_TARGET_KEYS,
        name="Surface-tension no-target policy",
    )
    if any(value is not False for value in policy.values()):
        raise ValueError("Surface-tension no-target policy flags must be false.")
    return Route2V0OrdinaryWaterSurfaceTensionSource(
        temperature_kelvin=_positive(state["temperature_kelvin"], name="Temperature"),
        critical_temperature_kelvin=_positive(
            formula["critical_temperature_kelvin"],
            name="Critical temperature",
        ),
        prefactor_newton_per_meter=_positive(
            formula["prefactor_newton_per_meter"],
            name="Surface-tension prefactor",
        ),
        exponent=_positive(formula["exponent"], name="Surface-tension exponent"),
        linear_coefficient=_finite(
            formula["linear_coefficient"],
            name="Surface-tension linear coefficient",
        ),
        surface_tension_newton_per_meter=_positive(
            root["surface_tension_newton_per_meter"],
            name="Surface tension",
        ),
        property_source=_property_source_from_json(root["property_source"]),
        claim_boundary=_text(
            root["claim_boundary"], name="Surface-tension claim boundary"
        ),
        not_claimed=_nonempty_strings(
            root["not_claimed"], name="Surface-tension non-claim"
        ),
    )


def load_route2_v0_ordinary_water_surface_tension_source(
    path: str | Path,
) -> Route2V0OrdinaryWaterSurfaceTensionSource:
    """Load one strict source-only IAPWS ordinary-water target record."""

    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(
            f"Ordinary-water surface-tension source is absent: {source_path}."
        )
    return parse_route2_v0_ordinary_water_surface_tension_source(
        source_path.read_text(encoding="utf-8")
    )


__all__ = [
    "V0_ORDINARY_WATER_SURFACE_TENSION_SOURCE_CONSTRUCTION",
    "V0_ORDINARY_WATER_SURFACE_TENSION_SOURCE_STATUS",
    "Route2V0OrdinaryWaterSurfaceTensionSource",
    "load_route2_v0_ordinary_water_surface_tension_source",
    "parse_route2_v0_ordinary_water_surface_tension_source",
]
