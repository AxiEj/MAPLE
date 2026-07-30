"""Fail-closed source records for the Route-2 V0 Lorentz dielectric control.

The Lorentz/Yukawa spectrum needs one finite-wavevector orientational
correlation length in addition to the static and optical dielectric limits.
The limits belong to :mod:`route2_v0_bulk_liquid_state_source`; this module
records the independent provenance of that length and joins it to exactly the
same molecular-model and bulk-state identity.

The result is deliberately a source-only electrostatic input.  It cannot
construct a molecular susceptibility, a cavity, dispersion, a standard-state
ledger, or a total solvation free energy.
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
    Route2V0BulkLiquidStateSource,
)

V0_LORENTZ_DIELECTRIC_SOURCE_CONSTRUCTION = (
    "route2-v0-lorentz-nonlocal-dielectric-source-v1"
)
V0_LORENTZ_DIELECTRIC_SOURCE_STATUS = (
    "source-only-lorentz-response-not-liquid-or-accuracy-admitted"
)

_SCHEMA_VERSION = 1
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SOLVENT_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_ORIGINS = frozenset(
    {
        "independent_measurement",
        "ab_initio",
        "upstream_model_validation",
    }
)
_ROOT_KEYS = frozenset(
    {
        "protocol_id",
        "schema_version",
        "status",
        "solvent_id",
        "model",
        "response",
        "correlation_length_source",
        "no_target_policy",
        "claim_boundary",
        "not_claimed",
    }
)
_MODEL_KEYS = frozenset({"identifier", "source_sha256"})
_RESPONSE_KEYS = frozenset(
    {
        "static_dielectric_constant",
        "optical_dielectric_constant",
        "orientational_correlation_length_bohr",
    }
)
_CORRELATION_LENGTH_SOURCE_KEYS = frozenset(
    {
        "origin",
        "document_url",
        "document_sha256",
        "source_locator",
        "retrieved_utc",
    }
)


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
    if _DIGEST.fullmatch(result) is None:
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


def _same_scalar(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-12)


@dataclass(frozen=True)
class Route2V0LorentzCorrelationLengthSource:
    """Content-addressed provenance for one orientational correlation length."""

    origin: str
    document_url: str
    document_sha256: str
    source_locator: str
    retrieved_utc: str

    def __post_init__(self) -> None:
        origin = _nonempty(self.origin, name="Lorentz correlation source origin")
        if origin not in _ORIGINS:
            raise ValueError("Unsupported Lorentz correlation source origin.")
        url = _nonempty(self.document_url, name="Lorentz correlation document URL")
        if not url.startswith("https://"):
            raise ValueError("Lorentz correlation document URL must use HTTPS.")
        digest = _digest(
            self.document_sha256,
            name="Lorentz correlation document SHA-256",
        )
        locator = _nonempty(
            self.source_locator,
            name="Lorentz correlation source locator",
        )
        retrieved = _nonempty(
            self.retrieved_utc,
            name="Lorentz correlation retrieval timestamp",
        )
        if _UTC_TIMESTAMP.fullmatch(retrieved) is None:
            raise ValueError(
                "Lorentz correlation retrieval timestamp must be UTC RFC3339 seconds."
            )
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "document_url", url)
        object.__setattr__(self, "document_sha256", digest)
        object.__setattr__(self, "source_locator", locator)
        object.__setattr__(self, "retrieved_utc", retrieved)


@dataclass(frozen=True)
class Route2V0LorentzNonlocalDielectricSource:
    """One source-only finite-wavevector dielectric response record.

    The record duplicates the two dielectric limits so that it can prove it
    belongs to the exact :class:`Route2V0BulkLiquidStateSource` used by the
    spectrum.  It is not an independent replacement for the bulk-state
    record, and it remains below total-solvation admission.
    """

    solvent_id: str
    model_identifier: str
    model_source_sha256: str
    static_dielectric_constant: float
    optical_dielectric_constant: float
    orientational_correlation_length_bohr: float | None
    correlation_length_source: Route2V0LorentzCorrelationLengthSource | None
    claim_boundary: str
    not_claimed: tuple[str, ...]
    construction: str = V0_LORENTZ_DIELECTRIC_SOURCE_CONSTRUCTION
    status: str = V0_LORENTZ_DIELECTRIC_SOURCE_STATUS

    def __post_init__(self) -> None:
        solvent_id = _nonempty(self.solvent_id, name="Solvent identifier")
        if _SOLVENT_ID.fullmatch(solvent_id) is None:
            raise ValueError("Solvent identifier must be a lowercase stable slug.")
        identifier = _nonempty(
            self.model_identifier,
            name="Lorentz dielectric model identifier",
        )
        digest = _digest(
            self.model_source_sha256,
            name="Lorentz dielectric model source SHA-256",
        )
        static = _finite_positive(
            self.static_dielectric_constant,
            name="Lorentz static dielectric constant",
        )
        optical = _finite_positive(
            self.optical_dielectric_constant,
            name="Lorentz optical dielectric constant",
        )
        if static <= 1.0:
            raise ValueError("Lorentz static dielectric constant must exceed one.")
        if optical < 1.0 or optical > static:
            raise ValueError(
                "Lorentz optical dielectric constant must lie in [1, static dielectric]."
            )
        if static == optical:
            if self.orientational_correlation_length_bohr is not None:
                raise ValueError(
                    "A constant Lorentz dielectric response must not carry an "
                    "unidentifiable orientational correlation length."
                )
            if self.correlation_length_source is not None:
                raise ValueError(
                    "A constant Lorentz dielectric response must not carry a "
                    "correlation-length source."
                )
            length: float | None = None
            correlation_source: Route2V0LorentzCorrelationLengthSource | None = None
        else:
            length = _finite_positive(
                self.orientational_correlation_length_bohr,
                name="Lorentz orientational correlation length",
            )
            if not isinstance(
                self.correlation_length_source,
                Route2V0LorentzCorrelationLengthSource,
            ):
                raise TypeError("Lorentz correlation length requires a source record.")
            correlation_source = self.correlation_length_source
        claim = _nonempty(self.claim_boundary, name="Lorentz claim boundary")
        nonclaims = _string_sequence(
            self.not_claimed,
            name="Lorentz non-claims",
        )
        if self.construction != V0_LORENTZ_DIELECTRIC_SOURCE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 Lorentz dielectric source.")
        if self.status != V0_LORENTZ_DIELECTRIC_SOURCE_STATUS:
            raise ValueError("Lorentz dielectric source must remain source-only.")
        object.__setattr__(self, "solvent_id", solvent_id)
        object.__setattr__(self, "model_identifier", identifier)
        object.__setattr__(self, "model_source_sha256", digest)
        object.__setattr__(self, "static_dielectric_constant", static)
        object.__setattr__(self, "optical_dielectric_constant", optical)
        object.__setattr__(self, "orientational_correlation_length_bohr", length)
        object.__setattr__(self, "correlation_length_source", correlation_source)
        object.__setattr__(self, "claim_boundary", claim)
        object.__setattr__(self, "not_claimed", nonclaims)

    @property
    def is_molecular_liquid_asset(self) -> bool:
        """Return false because a Lorentz response is not a liquid functional."""

        return False

    def verify_bulk_liquid_state_source(
        self,
        bulk_state_source: Route2V0BulkLiquidStateSource,
    ) -> None:
        """Require exactly one matching source-only bulk-state record."""

        if not isinstance(bulk_state_source, Route2V0BulkLiquidStateSource):
            raise TypeError(
                "Lorentz dielectric response requires a Route-2 bulk-liquid state source."
            )
        if (
            bulk_state_source.solvent_id != self.solvent_id
            or bulk_state_source.model_identifier != self.model_identifier
            or bulk_state_source.model_source_sha256 != self.model_source_sha256
        ):
            raise ValueError(
                "Lorentz dielectric response and bulk-state model identity disagree."
            )
        if not _same_scalar(
            bulk_state_source.static_dielectric_constant,
            self.static_dielectric_constant,
        ) or not _same_scalar(
            bulk_state_source.optical_dielectric_constant,
            self.optical_dielectric_constant,
        ):
            raise ValueError(
                "Lorentz dielectric response and bulk-state dielectric limits disagree."
            )


def _correlation_length_source_from_json(
    value: object,
) -> Route2V0LorentzCorrelationLengthSource | None:
    if value is None:
        return None
    payload = _strict_mapping(value, name="Lorentz correlation length source")
    _strict_keys(
        payload,
        _CORRELATION_LENGTH_SOURCE_KEYS,
        name="Lorentz correlation length source",
    )
    return Route2V0LorentzCorrelationLengthSource(
        origin=_nonempty(payload["origin"], name="Lorentz correlation source origin"),
        document_url=_nonempty(
            payload["document_url"],
            name="Lorentz correlation document URL",
        ),
        document_sha256=_digest(
            payload["document_sha256"],
            name="Lorentz correlation document SHA-256",
        ),
        source_locator=_nonempty(
            payload["source_locator"],
            name="Lorentz correlation source locator",
        ),
        retrieved_utc=_nonempty(
            payload["retrieved_utc"],
            name="Lorentz correlation retrieval timestamp",
        ),
    )


def parse_route2_v0_lorentz_nonlocal_dielectric_source(
    text: str,
) -> Route2V0LorentzNonlocalDielectricSource:
    """Parse one strict source-only Lorentz dielectric response record."""

    try:
        payload = cast(object, json.loads(text))
    except json.JSONDecodeError as exc:
        raise ValueError("Lorentz dielectric source must be valid JSON.") from exc
    root = _strict_mapping(payload, name="Lorentz dielectric source")
    _strict_keys(root, _ROOT_KEYS, name="Lorentz dielectric source")
    if root["protocol_id"] != V0_LORENTZ_DIELECTRIC_SOURCE_CONSTRUCTION:
        raise ValueError("Unsupported Route-2 V0 Lorentz dielectric source protocol.")
    if root["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("Unsupported Route-2 V0 Lorentz dielectric source schema.")
    if root["status"] != V0_LORENTZ_DIELECTRIC_SOURCE_STATUS:
        raise ValueError("Lorentz dielectric source must remain source-only.")
    model = _strict_mapping(root["model"], name="Lorentz dielectric molecular model")
    _strict_keys(model, _MODEL_KEYS, name="Lorentz dielectric molecular model")
    response = _strict_mapping(root["response"], name="Lorentz dielectric response")
    _strict_keys(response, _RESPONSE_KEYS, name="Lorentz dielectric response")
    policy = _strict_mapping(
        root["no_target_policy"],
        name="Lorentz dielectric no-target policy",
    )
    _strict_keys(
        policy,
        V0_BULK_LIQUID_STATE_REQUIRED_NO_TARGET_KEYS,
        name="Lorentz dielectric no-target policy",
    )
    if any(value is not False for value in policy.values()):
        raise ValueError("Lorentz dielectric no-target policy flags must be false.")
    return Route2V0LorentzNonlocalDielectricSource(
        solvent_id=_nonempty(root["solvent_id"], name="Solvent identifier"),
        model_identifier=_nonempty(
            model["identifier"],
            name="Lorentz dielectric model identifier",
        ),
        model_source_sha256=_digest(
            model["source_sha256"],
            name="Lorentz dielectric model source SHA-256",
        ),
        static_dielectric_constant=_finite_positive(
            response["static_dielectric_constant"],
            name="Lorentz static dielectric constant",
        ),
        optical_dielectric_constant=_finite_positive(
            response["optical_dielectric_constant"],
            name="Lorentz optical dielectric constant",
        ),
        orientational_correlation_length_bohr=(
            None
            if response["orientational_correlation_length_bohr"] is None
            else _finite_positive(
                response["orientational_correlation_length_bohr"],
                name="Lorentz orientational correlation length",
            )
        ),
        correlation_length_source=_correlation_length_source_from_json(
            root["correlation_length_source"]
        ),
        claim_boundary=_nonempty(root["claim_boundary"], name="Lorentz claim boundary"),
        not_claimed=_string_sequence(root["not_claimed"], name="Lorentz non-claims"),
    )


def load_route2_v0_lorentz_nonlocal_dielectric_source(
    path: str | Path,
) -> Route2V0LorentzNonlocalDielectricSource:
    """Load one strict source-only Lorentz dielectric response record."""

    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"Lorentz dielectric source is absent: {source_path}.")
    return parse_route2_v0_lorentz_nonlocal_dielectric_source(
        source_path.read_text(encoding="utf-8")
    )


__all__ = [
    "V0_LORENTZ_DIELECTRIC_SOURCE_CONSTRUCTION",
    "V0_LORENTZ_DIELECTRIC_SOURCE_STATUS",
    "Route2V0LorentzCorrelationLengthSource",
    "Route2V0LorentzNonlocalDielectricSource",
    "load_route2_v0_lorentz_nonlocal_dielectric_source",
    "parse_route2_v0_lorentz_nonlocal_dielectric_source",
]
