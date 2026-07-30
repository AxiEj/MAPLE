"""Source-only macroscopic state for the molecular-RISM Route-2 V0 branch.

A fixed-charge molecular RISM liquid needs a static dielectric constraint, but
it does not contain a separate solvent-electronic polarization degree of
freedom.  Its 1D bulk input therefore consumes ``epsilon(0)``, not an optical
``epsilon(infinity)``.  The latter remains mandatory only for the separately
labelled nonlocal-dielectric continuum control.

This module prevents those two state contracts from being accidentally merged.
It hash-binds the scalar pure-liquid state that a molecular-RISM input can
actually cross-check: temperature, pressure, molecular number density, static
dielectric constant, isothermal compressibility, and surface tension.  It
never constructs a finite-wavevector direct correlation, a closure, a bridge,
or a solvation free energy.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .route2_v0_bulk_liquid_state_source import (
    V0_BULK_LIQUID_STATE_REQUIRED_NO_TARGET_KEYS,
    Route2V0BulkLiquidPropertySource,
)
from .route2_v0_rism_molecular_source import Route2V0Rism1dInput

V0_MOLECULAR_RISM_BULK_STATE_SOURCE_CONSTRUCTION = (
    "route2-v0-molecular-rism-bulk-state-source-v1"
)
V0_MOLECULAR_RISM_BULK_STATE_SOURCE_STATUS = (
    "source-only-molecular-rism-state-not-liquid-or-accuracy-admitted"
)
V0_MOLECULAR_RISM_BULK_STATE_REQUIRED_PROPERTIES = frozenset(
    {
        "temperature_kelvin",
        "pressure_bar",
        "molecular_number_density_angstrom3",
        "static_dielectric_constant",
        "isothermal_compressibility_pa_inverse",
        "surface_tension_newton_per_meter",
    }
)

_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SOLVENT_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
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
_ROOT_KEYS = frozenset(
    {
        "protocol_id",
        "schema_version",
        "status",
        "solvent_id",
        "model",
        "state",
        "property_sources",
        "no_target_policy",
        "claim_boundary",
        "not_claimed",
    }
)
_MODEL_KEYS = frozenset({"identifier", "source_sha256"})

_BOLTZMANN_J_K = 1.380649e-23
_ANGSTROM3_PER_M3 = 1.0e30


def _strict_mapping(value: object, *, name: str) -> Mapping[str, object]:
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


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value.strip()


def _digest(value: object, *, name: str) -> str:
    result = _nonempty(value, name=name).lower()
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return result


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{name} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _string_sequence(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f"{name} must be a sequence.")
    result = tuple(_nonempty(item, name=name) for item in value)
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return result


def _state_match(observed: float, expected: float, *, name: str) -> None:
    if not math.isclose(observed, expected, rel_tol=1.0e-8, abs_tol=1.0e-12):
        message = f"Molecular-RISM bulk-state source and rism1d {name} disagree: "
        message += f"source={observed!r}, rism1d={expected!r}."
        raise ValueError(message)


def _property_source_from_json(value: object) -> Route2V0BulkLiquidPropertySource:
    payload = _strict_mapping(value, name="Molecular-RISM property source")
    _strict_keys(
        payload,
        _PROPERTY_SOURCE_KEYS,
        name="Molecular-RISM property source",
    )
    property_name = _nonempty(payload["property"], name="Bulk-state property")
    if property_name not in V0_MOLECULAR_RISM_BULK_STATE_REQUIRED_PROPERTIES:
        raise ValueError("Unsupported molecular-RISM bulk-state property.")
    return Route2V0BulkLiquidPropertySource(
        property_name=property_name,
        origin=_nonempty(payload["origin"], name="Bulk-state source origin"),
        document_url=_nonempty(
            payload["document_url"], name="Bulk-state document URL"
        ),
        document_sha256=_digest(
            payload["document_sha256"], name="Bulk-state document SHA-256"
        ),
        source_locator=_nonempty(
            payload["source_locator"], name="Bulk-state source locator"
        ),
        retrieved_utc=_nonempty(
            payload["retrieved_utc"], name="Bulk-state retrieval timestamp"
        ),
    )


@dataclass(frozen=True)
class Route2V0MolecularRismBulkStateSource:
    """One source-only macroscopic state usable by a molecular-RISM input.

    The one-component zero-mode helpers deliberately expose only the scalar
    compressibility consequence.  A molecular-site direct-correlation matrix
    ``C_ab(k)`` remains an independent frozen-liquid requirement.
    """

    solvent_id: str
    model_identifier: str
    model_source_sha256: str
    temperature_kelvin: float
    pressure_bar: float
    molecular_number_density_angstrom3: float
    static_dielectric_constant: float
    isothermal_compressibility_pa_inverse: float
    surface_tension_newton_per_meter: float
    property_sources: tuple[Route2V0BulkLiquidPropertySource, ...]
    claim_boundary: str
    not_claimed: tuple[str, ...]
    construction: str = V0_MOLECULAR_RISM_BULK_STATE_SOURCE_CONSTRUCTION
    status: str = V0_MOLECULAR_RISM_BULK_STATE_SOURCE_STATUS

    def __post_init__(self) -> None:
        solvent_id = _nonempty(self.solvent_id, name="Solvent identifier")
        if _SOLVENT_ID.fullmatch(solvent_id) is None:
            raise ValueError("Solvent identifier must be a lowercase stable slug.")
        identifier = _nonempty(
            self.model_identifier,
            name="Molecular-RISM bulk-state model identifier",
        )
        digest = _digest(
            self.model_source_sha256,
            name="Molecular-RISM bulk-state model source SHA-256",
        )
        temperature = _finite_positive(self.temperature_kelvin, name="Temperature")
        pressure = _finite_positive(self.pressure_bar, name="Pressure")
        density = _finite_positive(
            self.molecular_number_density_angstrom3,
            name="Molecular number density",
        )
        dielectric = _finite_positive(
            self.static_dielectric_constant,
            name="Static dielectric constant",
        )
        if dielectric <= 1.0:
            raise ValueError("Static dielectric constant must exceed one.")
        compressibility = _finite_positive(
            self.isothermal_compressibility_pa_inverse,
            name="Isothermal compressibility",
        )
        surface_tension = _finite_positive(
            self.surface_tension_newton_per_meter,
            name="Surface tension",
        )
        source_values = cast(object, self.property_sources)
        if not isinstance(source_values, tuple):
            raise TypeError(
                "Molecular-RISM bulk-state property sources must be source records."
            )
        raw_sources = tuple(cast(tuple[object, ...], source_values))
        if not raw_sources or any(
            not isinstance(source, Route2V0BulkLiquidPropertySource)
            for source in raw_sources
        ):
            raise TypeError(
                "Molecular-RISM bulk-state property sources must be source records."
            )
        sources = tuple(
            cast(Route2V0BulkLiquidPropertySource, source) for source in raw_sources
        )
        names = tuple(source.property_name for source in sources)
        if (
            len(set(names)) != len(names)
            or frozenset(names) != V0_MOLECULAR_RISM_BULK_STATE_REQUIRED_PROPERTIES
        ):
            raise ValueError(
                "Molecular-RISM property sources must cover exactly the required state properties."
            )
        claim = _nonempty(self.claim_boundary, name="Molecular-RISM claim boundary")
        nonclaims = _string_sequence(
            self.not_claimed,
            name="Molecular-RISM bulk-state non-claims",
        )
        if self.construction != V0_MOLECULAR_RISM_BULK_STATE_SOURCE_CONSTRUCTION:
            raise ValueError("Unsupported molecular-RISM bulk-state construction.")
        if self.status != V0_MOLECULAR_RISM_BULK_STATE_SOURCE_STATUS:
            raise ValueError("Molecular-RISM bulk-state record must remain source-only.")
        object.__setattr__(self, "solvent_id", solvent_id)
        object.__setattr__(self, "model_identifier", identifier)
        object.__setattr__(self, "model_source_sha256", digest)
        object.__setattr__(self, "temperature_kelvin", temperature)
        object.__setattr__(self, "pressure_bar", pressure)
        object.__setattr__(self, "molecular_number_density_angstrom3", density)
        object.__setattr__(self, "static_dielectric_constant", dielectric)
        object.__setattr__(
            self,
            "isothermal_compressibility_pa_inverse",
            compressibility,
        )
        object.__setattr__(self, "surface_tension_newton_per_meter", surface_tension)
        object.__setattr__(self, "property_sources", sources)
        object.__setattr__(self, "claim_boundary", claim)
        object.__setattr__(self, "not_claimed", nonclaims)

    @property
    def is_molecular_liquid_asset(self) -> bool:
        """Return false because the molecular finite-k liquid is still absent."""

        return False

    @property
    def number_structure_factor_zero_mode(self) -> float:
        """Return the scalar compressibility sum-rule consequence ``S_NN(0)``."""

        density_m3 = self.molecular_number_density_angstrom3 * _ANGSTROM3_PER_M3
        value = (
            density_m3
            * _BOLTZMANN_J_K
            * self.temperature_kelvin
            * self.isothermal_compressibility_pa_inverse
        )
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("Bulk-state compressibility gives an invalid S_NN(0).")
        return value

    @property
    def number_channel_direct_correlation_zero_mode_angstrom3(self) -> float:
        """Return the one-component OZ zero mode without inventing ``C_ab(k)``."""

        return (
            1.0 - 1.0 / self.number_structure_factor_zero_mode
        ) / self.molecular_number_density_angstrom3

    def property_source_for(
        self,
        property_name: str,
    ) -> Route2V0BulkLiquidPropertySource:
        """Return the exact source record for one declared state property."""

        name = _nonempty(property_name, name="Molecular-RISM bulk-state property")
        for source in self.property_sources:
            if source.property_name == name:
                return source
        raise ValueError(f"Molecular-RISM source lacks property {name!r}.")

    def verify_model_source_digest(self, source_sha256: str) -> None:
        """Require the exact molecular model bound before a RISM use."""

        observed = _digest(
            source_sha256,
            name="Molecular-RISM molecular model source SHA-256",
        )
        if observed != self.model_source_sha256:
            raise ValueError(
                "Molecular-RISM bulk-state and molecular model source digest disagree."
            )

    def verify_rism1d_state(self, rism1d_input: Route2V0Rism1dInput) -> None:
        """Cross-check exactly the static state observable by fixed-charge RISM."""

        input_value = cast(object, rism1d_input)
        if not isinstance(input_value, Route2V0Rism1dInput):
            raise TypeError("Molecular-RISM bulk state requires a parsed rism1d input.")
        parsed_input = input_value
        _state_match(
            self.temperature_kelvin,
            parsed_input.temperature_kelvin,
            name="temperature",
        )
        _state_match(
            self.molecular_number_density_angstrom3,
            parsed_input.molecular_number_density_angstrom3,
            name="density",
        )
        _state_match(
            self.static_dielectric_constant,
            parsed_input.dielectric_constant,
            name="static dielectric",
        )


def parse_route2_v0_molecular_rism_bulk_state_source(
    text: str,
) -> Route2V0MolecularRismBulkStateSource:
    """Parse one strict source-only molecular-RISM state record."""

    text_value = cast(object, text)
    if not isinstance(text_value, str):
        raise TypeError("Molecular-RISM bulk-state source must be text.")
    try:
        payload = cast(object, json.loads(text_value))
    except json.JSONDecodeError as exc:
        raise ValueError("Molecular-RISM bulk-state source must be valid JSON.") from exc
    root = _strict_mapping(payload, name="Molecular-RISM bulk-state source")
    _strict_keys(root, _ROOT_KEYS, name="Molecular-RISM bulk-state source")
    if root["protocol_id"] != V0_MOLECULAR_RISM_BULK_STATE_SOURCE_CONSTRUCTION:
        raise ValueError("Unsupported molecular-RISM bulk-state protocol.")
    if root["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("Unsupported molecular-RISM bulk-state schema.")
    if root["status"] != V0_MOLECULAR_RISM_BULK_STATE_SOURCE_STATUS:
        raise ValueError("Molecular-RISM bulk-state record must remain source-only.")
    model = _strict_mapping(root["model"], name="Molecular-RISM molecular model")
    _strict_keys(model, _MODEL_KEYS, name="Molecular-RISM molecular model")
    state = _strict_mapping(root["state"], name="Molecular-RISM liquid state")
    _strict_keys(
        state,
        V0_MOLECULAR_RISM_BULK_STATE_REQUIRED_PROPERTIES,
        name="Molecular-RISM liquid state",
    )
    property_sources_value = root["property_sources"]
    if not isinstance(property_sources_value, list):
        raise TypeError("Molecular-RISM property_sources must be a list.")
    property_sources = cast(list[object], property_sources_value)
    policy = _strict_mapping(
        root["no_target_policy"],
        name="Molecular-RISM no-target policy",
    )
    _strict_keys(
        policy,
        V0_BULK_LIQUID_STATE_REQUIRED_NO_TARGET_KEYS,
        name="Molecular-RISM no-target policy",
    )
    if any(value is not False for value in policy.values()):
        raise ValueError("Molecular-RISM no-target policy flags must be false.")
    return Route2V0MolecularRismBulkStateSource(
        solvent_id=_nonempty(root["solvent_id"], name="Solvent identifier"),
        model_identifier=_nonempty(model["identifier"], name="Model identifier"),
        model_source_sha256=_digest(
            model["source_sha256"],
            name="Molecular-RISM model source SHA-256",
        ),
        temperature_kelvin=_finite_positive(
            state["temperature_kelvin"],
            name="Temperature",
        ),
        pressure_bar=_finite_positive(state["pressure_bar"], name="Pressure"),
        molecular_number_density_angstrom3=_finite_positive(
            state["molecular_number_density_angstrom3"],
            name="Molecular number density",
        ),
        static_dielectric_constant=_finite_positive(
            state["static_dielectric_constant"],
            name="Static dielectric constant",
        ),
        isothermal_compressibility_pa_inverse=_finite_positive(
            state["isothermal_compressibility_pa_inverse"],
            name="Isothermal compressibility",
        ),
        surface_tension_newton_per_meter=_finite_positive(
            state["surface_tension_newton_per_meter"],
            name="Surface tension",
        ),
        property_sources=tuple(
            _property_source_from_json(value) for value in property_sources
        ),
        claim_boundary=_nonempty(
            root["claim_boundary"],
            name="Molecular-RISM claim boundary",
        ),
        not_claimed=_string_sequence(
            root["not_claimed"],
            name="Molecular-RISM bulk-state non-claims",
        ),
    )


def load_route2_v0_molecular_rism_bulk_state_source(
    path: str | Path,
) -> Route2V0MolecularRismBulkStateSource:
    """Load one strict source-only molecular-RISM state record."""

    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"Molecular-RISM bulk-state source is absent: {source_path}.")
    return parse_route2_v0_molecular_rism_bulk_state_source(
        source_path.read_text(encoding="utf-8")
    )


__all__ = [
    "V0_MOLECULAR_RISM_BULK_STATE_REQUIRED_PROPERTIES",
    "V0_MOLECULAR_RISM_BULK_STATE_SOURCE_CONSTRUCTION",
    "V0_MOLECULAR_RISM_BULK_STATE_SOURCE_STATUS",
    "Route2V0MolecularRismBulkStateSource",
    "load_route2_v0_molecular_rism_bulk_state_source",
    "parse_route2_v0_molecular_rism_bulk_state_source",
]
