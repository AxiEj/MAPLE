"""Fail-closed pure-liquid bulk-state sources for Route-2 V0.

A molecular site model and a few macroscopic liquid properties are not a
molecular liquid free-energy functional.  This module deliberately captures
only the independently sourced state anchors that a later all-atom RISM/MDFT
asset must cross-check: temperature, pressure, number density, static and
optical dielectric limits, compressibility, and surface tension.

The record is source-only.  It never manufactures a finite-wavevector direct
correlation, a short-range solute--solvent potential, a bridge functional, or
a solvation free energy.  Those missing objects remain fail-closed gates.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .route2_v0_rism_molecular_source import Route2V0Rism1dInput

V0_BULK_LIQUID_STATE_SOURCE_CONSTRUCTION = "route2-v0-bulk-liquid-state-source-v1"
V0_BULK_LIQUID_STATE_SOURCE_STATUS = (
    "source-only-bulk-state-not-molecular-liquid-or-accuracy-admitted"
)
V0_BULK_LIQUID_STATE_REQUIRED_NO_TARGET_KEYS = frozenset(
    {
        "post_training",
        "fine_tuning",
        "experimental_solvation_fit",
        "map_or_uq_calibration",
        "target_solvation_labels_used",
    }
)
V0_BULK_LIQUID_STATE_REQUIRED_PROPERTIES = frozenset(
    {
        "temperature_kelvin",
        "pressure_bar",
        "molecular_number_density_angstrom3",
        "static_dielectric_constant",
        "optical_dielectric_constant",
        "isothermal_compressibility_pa_inverse",
        "surface_tension_newton_per_meter",
    }
)

_SCHEMA_VERSION = 1
_ORIGINS = frozenset(
    {
        "independent_measurement",
        "ab_initio",
        "upstream_model_validation",
    }
)
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SOLVENT_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
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

# SI exact value.  The public state is in Angstrom^-3 because that is the
# unit consumed by the molecular RISM parser.
_BOLTZMANN_J_K = 1.380649e-23
_ANGSTROM3_PER_M3 = 1.0e30


def _strict_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object.")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise TypeError(f"{name} must be an object.")
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


def _digest(value: object, *, name: str) -> str:
    result = _nonempty(value, name=name).lower()
    if _DIGEST.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return result


def _string_sequence(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list.")
    values = tuple(_nonempty(item, name=name) for item in cast(list[object], value))
    if not values:
        raise ValueError(f"{name} must not be empty.")
    return values


def _state_match(observed: float, expected: float, *, name: str) -> None:
    if not math.isclose(observed, expected, rel_tol=1.0e-8, abs_tol=1.0e-12):
        message = (
            f"Bulk liquid-state source and rism1d {name} disagree: "
            + f"source={observed!r}, rism1d={expected!r}."
        )
        raise ValueError(message)


@dataclass(frozen=True)
class Route2V0BulkLiquidPropertySource:
    """One content-addressed source for exactly one pure-liquid property."""

    property_name: str
    origin: str
    document_url: str
    document_sha256: str
    source_locator: str
    retrieved_utc: str

    def __post_init__(self) -> None:
        property_name = _nonempty(self.property_name, name="Bulk-state property")
        if property_name not in V0_BULK_LIQUID_STATE_REQUIRED_PROPERTIES:
            raise ValueError("Unsupported Route-2 V0 bulk-state property.")
        origin = _nonempty(self.origin, name="Bulk-state property source origin")
        if origin not in _ORIGINS:
            raise ValueError("Unsupported bulk-state property source origin.")
        url = _nonempty(self.document_url, name="Bulk-state document URL")
        if not url.startswith("https://"):
            raise ValueError("Bulk-state document URL must use HTTPS.")
        digest = _digest(self.document_sha256, name="Bulk-state document SHA-256")
        locator = _nonempty(self.source_locator, name="Bulk-state source locator")
        retrieved = _nonempty(
            self.retrieved_utc,
            name="Bulk-state retrieval timestamp",
        )
        if _UTC_TIMESTAMP.fullmatch(retrieved) is None:
            raise ValueError(
                "Bulk-state retrieval timestamp must be UTC RFC3339 seconds."
            )
        object.__setattr__(self, "property_name", property_name)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "document_url", url)
        object.__setattr__(self, "document_sha256", digest)
        object.__setattr__(self, "source_locator", locator)
        object.__setattr__(self, "retrieved_utc", retrieved)


@dataclass(frozen=True)
class Route2V0BulkLiquidStateSource:
    """One source-only pure-liquid state bound to one molecular model identity.

    The optional one-component zero-mode helpers express exactly what density
    and compressibility constrain.  They are intentionally *not* a molecular
    site--site direct-correlation construction.
    """

    solvent_id: str
    model_identifier: str
    model_source_sha256: str
    temperature_kelvin: float
    pressure_bar: float
    molecular_number_density_angstrom3: float
    static_dielectric_constant: float
    optical_dielectric_constant: float
    isothermal_compressibility_pa_inverse: float
    surface_tension_newton_per_meter: float
    property_sources: tuple[Route2V0BulkLiquidPropertySource, ...]
    claim_boundary: str
    not_claimed: tuple[str, ...]
    construction: str = V0_BULK_LIQUID_STATE_SOURCE_CONSTRUCTION
    status: str = V0_BULK_LIQUID_STATE_SOURCE_STATUS

    def __post_init__(self) -> None:
        solvent_id = _nonempty(self.solvent_id, name="Solvent identifier")
        if _SOLVENT_ID.fullmatch(solvent_id) is None:
            raise ValueError("Solvent identifier must be a lowercase stable slug.")
        model_identifier = _nonempty(
            self.model_identifier,
            name="Bulk-state model identifier",
        )
        model_digest = _digest(
            self.model_source_sha256,
            name="Bulk-state model source SHA-256",
        )
        temperature = _finite_positive(self.temperature_kelvin, name="Temperature")
        pressure = _finite_positive(self.pressure_bar, name="Pressure")
        density = _finite_positive(
            self.molecular_number_density_angstrom3,
            name="Molecular number density",
        )
        epsilon_static = _finite_positive(
            self.static_dielectric_constant,
            name="Static dielectric constant",
        )
        epsilon_optical = _finite_positive(
            self.optical_dielectric_constant,
            name="Optical dielectric constant",
        )
        compressibility = _finite_positive(
            self.isothermal_compressibility_pa_inverse,
            name="Isothermal compressibility",
        )
        surface_tension = _finite_positive(
            self.surface_tension_newton_per_meter,
            name="Surface tension",
        )
        if epsilon_static <= 1.0:
            raise ValueError("Static dielectric constant must exceed one.")
        if epsilon_optical < 1.0 or epsilon_optical > epsilon_static:
            raise ValueError(
                "Optical dielectric constant must lie in [1, static dielectric]."
            )
        property_values = cast(object, self.property_sources)
        if not isinstance(property_values, tuple):
            raise TypeError("Bulk-state property sources must be source records.")
        properties = tuple(cast(tuple[object, ...], property_values))
        if not properties or any(
            not isinstance(value, Route2V0BulkLiquidPropertySource)
            for value in properties
        ):
            raise TypeError("Bulk-state property sources must be source records.")
        typed_properties = tuple(
            cast(Route2V0BulkLiquidPropertySource, value) for value in properties
        )
        property_names = tuple(value.property_name for value in typed_properties)
        if (
            len(set(property_names)) != len(property_names)
            or frozenset(property_names) != V0_BULK_LIQUID_STATE_REQUIRED_PROPERTIES
        ):
            raise ValueError(
                "Bulk-state property sources must cover exactly the required state "
                + "properties."
            )
        claim = _nonempty(self.claim_boundary, name="Bulk-state claim boundary")
        not_claimed_value = cast(object, self.not_claimed)
        if not isinstance(not_claimed_value, Sequence) or isinstance(
            not_claimed_value,
            str,
        ):
            raise TypeError("Bulk-state non-claims must be a sequence.")
        not_claimed = tuple(
            _nonempty(value, name="Bulk-state non-claim") for value in not_claimed_value
        )
        if not not_claimed:
            raise ValueError("Bulk-state non-claims must not be empty.")
        if self.construction != V0_BULK_LIQUID_STATE_SOURCE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 bulk-state construction.")
        if self.status != V0_BULK_LIQUID_STATE_SOURCE_STATUS:
            raise ValueError("Bulk-state record must remain source-only.")
        object.__setattr__(self, "solvent_id", solvent_id)
        object.__setattr__(self, "model_identifier", model_identifier)
        object.__setattr__(self, "model_source_sha256", model_digest)
        object.__setattr__(self, "temperature_kelvin", temperature)
        object.__setattr__(self, "pressure_bar", pressure)
        object.__setattr__(self, "molecular_number_density_angstrom3", density)
        object.__setattr__(self, "static_dielectric_constant", epsilon_static)
        object.__setattr__(self, "optical_dielectric_constant", epsilon_optical)
        object.__setattr__(
            self, "isothermal_compressibility_pa_inverse", compressibility
        )
        object.__setattr__(self, "surface_tension_newton_per_meter", surface_tension)
        object.__setattr__(self, "property_sources", typed_properties)
        object.__setattr__(self, "claim_boundary", claim)
        object.__setattr__(self, "not_claimed", not_claimed)

    @property
    def is_molecular_liquid_asset(self) -> bool:
        """Return false: finite-k liquid structure is deliberately absent."""

        return False

    @property
    def number_structure_factor_zero_mode(self) -> float:
        """Return ``S_NN(0)=rho k_B T kappa_T`` for a scalar number channel.

        This is a compressibility sum-rule anchor in the one-component
        reduction.  Molecular RISM needs a full site/orientational matrix and
        cannot be reconstructed from this number alone.
        """

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
        """Return the one-component OZ zero mode implied by compressibility.

        ``S_NN(0) = [1-rho*c_NN(0)]^-1``.  The value is a single projected
        number-channel constraint, never a replacement for ``C_ab(k)``.
        """

        return (
            1.0 - 1.0 / self.number_structure_factor_zero_mode
        ) / self.molecular_number_density_angstrom3

    def property_source_for(
        self, property_name: str
    ) -> Route2V0BulkLiquidPropertySource:
        """Return the exact source record for one declared state property."""

        name = _nonempty(property_name, name="Bulk-state property")
        for source in self.property_sources:
            if source.property_name == name:
                return source
        raise ValueError(f"Bulk-state source lacks property {name!r}.")

    def verify_model_source_digest(self, source_sha256: str) -> None:
        """Require the caller's molecular model to equal the frozen digest.

        A state source owns no molecular coordinates or site parameters.  A
        later liquid-asset loader must therefore make this explicit join
        before it can use the scalar state with a model source.
        """

        observed = _digest(source_sha256, name="Molecular model source SHA-256")
        if observed != self.model_source_sha256:
            raise ValueError(
                "Bulk liquid-state source and molecular model source digest disagree."
            )

    def verify_rism1d_state(self, rism1d_input: Route2V0Rism1dInput) -> None:
        """Cross-check the state shared with an eventual RISM source.

        RISM input exposes only temperature, molecular density, and static
        dielectric.  Passing this check does not supply the correlation,
        closure accuracy, short-range interaction, or a liquid endpoint.
        """

        _state_match(
            self.temperature_kelvin,
            rism1d_input.temperature_kelvin,
            name="temperature",
        )
        _state_match(
            self.molecular_number_density_angstrom3,
            rism1d_input.molecular_number_density_angstrom3,
            name="density",
        )
        _state_match(
            self.static_dielectric_constant,
            rism1d_input.dielectric_constant,
            name="dielectric",
        )


def _property_source_from_json(value: object) -> Route2V0BulkLiquidPropertySource:
    payload = _strict_mapping(value, name="Bulk-state property source")
    _strict_keys(payload, _PROPERTY_SOURCE_KEYS, name="Bulk-state property source")
    return Route2V0BulkLiquidPropertySource(
        property_name=_nonempty(payload["property"], name="Bulk-state property"),
        origin=_nonempty(payload["origin"], name="Bulk-state source origin"),
        document_url=_nonempty(
            payload["document_url"],
            name="Bulk-state document URL",
        ),
        document_sha256=_digest(
            payload["document_sha256"],
            name="Bulk-state document SHA-256",
        ),
        source_locator=_nonempty(
            payload["source_locator"],
            name="Bulk-state source locator",
        ),
        retrieved_utc=_nonempty(
            payload["retrieved_utc"],
            name="Bulk-state retrieval timestamp",
        ),
    )


def parse_route2_v0_bulk_liquid_state_source(
    text: str,
) -> Route2V0BulkLiquidStateSource:
    """Parse one strict, source-only Route-2 V0 pure-liquid state record."""

    try:
        payload = cast(object, json.loads(text))
    except json.JSONDecodeError as exc:
        raise ValueError("Bulk liquid-state source must be valid JSON.") from exc
    root = _strict_mapping(payload, name="Bulk liquid-state source")
    _strict_keys(root, _ROOT_KEYS, name="Bulk liquid-state source")
    if root["protocol_id"] != V0_BULK_LIQUID_STATE_SOURCE_CONSTRUCTION:
        raise ValueError("Unsupported Route-2 V0 bulk-state protocol.")
    if root["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("Unsupported Route-2 V0 bulk-state schema.")
    if root["status"] != V0_BULK_LIQUID_STATE_SOURCE_STATUS:
        raise ValueError("Bulk-state record must remain source-only.")
    model = _strict_mapping(root["model"], name="Bulk-state molecular model")
    _strict_keys(model, _MODEL_KEYS, name="Bulk-state molecular model")
    state = _strict_mapping(root["state"], name="Bulk liquid state")
    _strict_keys(
        state,
        V0_BULK_LIQUID_STATE_REQUIRED_PROPERTIES,
        name="Bulk liquid state",
    )
    property_sources_value = root["property_sources"]
    if not isinstance(property_sources_value, list):
        raise TypeError("Bulk-state property_sources must be a list.")
    property_sources = cast(list[object], property_sources_value)
    policy = _strict_mapping(
        root["no_target_policy"], name="Bulk-state no-target policy"
    )
    _strict_keys(
        policy,
        V0_BULK_LIQUID_STATE_REQUIRED_NO_TARGET_KEYS,
        name="Bulk-state no-target policy",
    )
    if any(value is not False for value in policy.values()):
        raise ValueError("Bulk-state no-target policy flags must be false.")
    return Route2V0BulkLiquidStateSource(
        solvent_id=_nonempty(root["solvent_id"], name="Solvent identifier"),
        model_identifier=_nonempty(
            model["identifier"],
            name="Bulk-state model identifier",
        ),
        model_source_sha256=_digest(
            model["source_sha256"],
            name="Bulk-state model source SHA-256",
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
        optical_dielectric_constant=_finite_positive(
            state["optical_dielectric_constant"],
            name="Optical dielectric constant",
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
            root["claim_boundary"], name="Bulk-state claim boundary"
        ),
        not_claimed=_string_sequence(root["not_claimed"], name="Bulk-state non-claims"),
    )


def load_route2_v0_bulk_liquid_state_source(
    path: str | Path,
) -> Route2V0BulkLiquidStateSource:
    """Load one strict source-only Route-2 V0 pure-liquid state record."""

    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"Bulk liquid-state source is absent: {source_path}.")
    return parse_route2_v0_bulk_liquid_state_source(
        source_path.read_text(encoding="utf-8")
    )


__all__ = [
    "V0_BULK_LIQUID_STATE_REQUIRED_NO_TARGET_KEYS",
    "V0_BULK_LIQUID_STATE_REQUIRED_PROPERTIES",
    "V0_BULK_LIQUID_STATE_SOURCE_CONSTRUCTION",
    "V0_BULK_LIQUID_STATE_SOURCE_STATUS",
    "Route2V0BulkLiquidPropertySource",
    "Route2V0BulkLiquidStateSource",
    "load_route2_v0_bulk_liquid_state_source",
    "parse_route2_v0_bulk_liquid_state_source",
]
