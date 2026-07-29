"""LSNN protocol contracts and a fail-closed adapter scaffold.

This module defines explicit runtime contracts for LSNN-style alchemical
workflows. The adapter accepts an injected callable runtime that returns Hartree
energies and dU/dlambda derivatives; no official OpenMM runtime is claimed or
implicitly assumed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast


_LSNN_DEFAULT_CARD = (
    Path(__file__).resolve().parent.parent
    / "calculator"
    / "model_cards"
    / "lsnn-v1.yaml"
)


class LSNNConfigError(ValueError):
    """Invalid LSNN adapter configuration."""


class LSNNRuntimeMissingError(RuntimeError):
    """No runtime adapter is available for executing LSNN evaluations."""


def _finite_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise LSNNConfigError(f"{field_name} must be a finite integer value.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LSNNConfigError(f"{field_name} must be a finite integer value.") from exc
    if not math.isfinite(parsed) or parsed != math.floor(parsed):
        raise LSNNConfigError(f"{field_name} must be a finite integer value.")
    return int(parsed)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _finite_float(value: object, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LSNNConfigError(f"{field_name} must be a finite numeric value.") from exc
    if not math.isfinite(parsed):
        raise LSNNConfigError(f"{field_name} must be finite: {value!r}.")
    return parsed


def _normalize_sha256(value: object, field_name: str) -> str:
    raw = str(value).strip().lower()
    if not raw:
        raise LSNNConfigError(f"{field_name} must be a non-empty SHA256 hex string.")
    if len(raw) != 64 or any(ch not in "0123456789abcdef" for ch in raw):
        raise LSNNConfigError(
            f"{field_name} must be a 64-character lowercase hexadecimal SHA256 digest."
        )
    return raw


def _parse_charge_range(value: object, field_name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    raw = re.sub(r"\s*to\s*", "-", str(value).strip(), flags=re.IGNORECASE)
    raw = raw.replace(" ", "").replace("_", "").replace("+", "")
    if not raw:
        return None

    normalized = raw.replace("to", "-", 1)

    if "-" in normalized:
        match = re.fullmatch(r"([+-]?\d+)-([+-]?\d+)", normalized)
        if not match:
            raise LSNNConfigError(
                f"{field_name} must be a numeric range like 'a to b' or 'a-b'."
            )
        lo = int(match.group(1))
        hi = int(match.group(2))
        if lo <= hi:
            return lo, hi
        return hi, lo

    parsed = _finite_int(normalized, field_name)
    return parsed, parsed


def _parse_multiplicity_values(value: object, field_name: str) -> tuple[int, ...]:
    if value is None:
        raise LSNNConfigError(f"{field_name} is required for audited LSNN domains.")

    if isinstance(value, (int, float, str)):
        parsed = _finite_int(value, field_name)
        if parsed <= 0:
            raise LSNNConfigError(f"{field_name} must be a positive integer.")
        return (parsed,)

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise LSNNConfigError(f"{field_name} must be an integer or sequence of integers.")

    parsed_values: set[int] = set()
    for item in value:
        parsed = _finite_int(item, field_name)
        if parsed <= 0:
            raise LSNNConfigError(f"{field_name} must contain only positive integers.")
        parsed_values.add(parsed)
    if not parsed_values:
        raise LSNNConfigError(f"{field_name} must contain at least one multiplicity.")
    return tuple(sorted(parsed_values))


def _canonical_element_sequence(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise LSNNConfigError(f"{field_name} must be a non-empty sequence.")
    canonical = tuple(str(item).strip() for item in value)
    if not canonical or any(not item for item in canonical):
        raise LSNNConfigError(f"{field_name} must contain at least one non-empty symbol.")
    return tuple(canonical)


@dataclass(frozen=True)
class LSNNModelCard:
    """Immutable metadata container loaded from the JSON-compatible YAML stub."""

    model_id: str
    version: str
    source_revision: str | None
    source_url: str | None
    model_checksum_sha256: str | None
    weight_artifacts: tuple[dict[str, str], ...]
    license: str | None
    elements: tuple[str, ...]
    charge_range: str | None
    audited_domain_enabled: bool
    audited_elements: tuple[str, ...]
    audited_charge_range: tuple[int, int] | None
    audited_multiplicities: tuple[int, ...]
    spin_support: bool
    solvent: str
    temperature_kelvin: float | None
    energy_reference: str
    solvent_scope: str
    supports_absolute_solvation: bool
    supports_alchemical_lambda: bool
    supports_ti: bool
    supports_mbar: bool
    supports_water_only: bool
    scientific_status: str
    default_eligible: bool

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "LSNNModelCard":
        artifacts = payload.get("weight_artifacts", ())
        if not isinstance(artifacts, list):
            raise LSNNConfigError("model_card.weight_artifacts must be a list.")

        if not artifacts:
            raise LSNNConfigError("model_card.weight_artifacts must not be empty.")

        cleaned: list[dict[str, str]] = []
        for item in artifacts:
            if not isinstance(item, Mapping):
                raise LSNNConfigError("Each weight artifact must be a mapping.")
            filename = str(item.get("filename", "")).strip()
            if not filename:
                raise LSNNConfigError("Each weight artifact must define filename.")
            raw_sha256 = item.get("sha256")
            if raw_sha256 is None:
                raise LSNNConfigError(
                    f"LSNN requires explicit SHA pin for artifact {filename!r};"
                    " omission is fail-closed."
                )
            sha = _normalize_sha256(raw_sha256, f"weight_artifacts[{filename}].sha256")
            source_path = str(item.get("source_path", filename)).strip()
            if not source_path:
                raise LSNNConfigError(
                    f"Weight artifact {filename!r} must define a non-empty source_path."
                )
            cleaned.append(
                {
                    "filename": filename,
                    "source_path": source_path,
                    "sha256": sha,
                }
            )

        elements = payload.get("elements", ())
        if not isinstance(elements, list):
            raise LSNNConfigError("model_card.elements must be a sequence.")
        normalized_elements = tuple(str(item) for item in elements)

        audited_domain = payload.get("audited_domain", {})
        if not isinstance(audited_domain, Mapping):
            raise LSNNConfigError("model_card.audited_domain must be a mapping when present.")

        audited_domain_enabled = bool(audited_domain.get("enabled", False))
        if audited_domain_enabled:
            audited_elements = _canonical_element_sequence(
                audited_domain.get("elements", normalized_elements),
                "audited_domain.elements",
            )
            audited_charge_range = _parse_charge_range(
                audited_domain.get("charge_range", payload.get("charge_range")),
                "audited_domain.charge_range",
            )
            if audited_charge_range is None:
                raise LSNNConfigError(
                    "audited_domain.charge_range is required when audited_domain.enabled=true."
                )
            audited_multiplicities = _parse_multiplicity_values(
                audited_domain.get("multiplicities", payload.get("multiplicity")),
                "audited_domain.multiplicities",
            )
        else:
            audited_elements = ()
            audited_charge_range = None
            audited_multiplicities = ()

        return cls(
            model_id=str(payload.get("model_id", "lsnn-v1")).strip().lower(),
            version=str(payload.get("version", "v1")),
            source_revision=_optional_str(payload.get("source_revision")),
            source_url=_optional_str(payload.get("source_url")),
            model_checksum_sha256=_optional_str(payload.get("model_checksum_sha256")),
            weight_artifacts=tuple(cleaned),
            license=_optional_str(payload.get("license")),
            elements=normalized_elements,
            charge_range=_optional_str(payload.get("charge_range")),
            audited_domain_enabled=audited_domain_enabled,
            audited_elements=audited_elements,
            audited_charge_range=audited_charge_range,
            audited_multiplicities=audited_multiplicities,
            spin_support=bool(payload.get("spin_support", False)),
            solvent=str(payload.get("solvent", "water")),
            temperature_kelvin=_optional_float(payload.get("temperature_kelvin")),
            energy_reference=str(payload.get("energy_reference", "unknown")),
            solvent_scope=str(payload.get("solvent_scope", "unknown")),
            supports_absolute_solvation=bool(payload.get("supports_absolute_solvation", False)),
            supports_alchemical_lambda=bool(payload.get("supports_alchemical_lambda", True)),
            supports_ti=bool(payload.get("supports_ti", True)),
            supports_mbar=bool(payload.get("supports_mbar", True)),
            supports_water_only=bool(payload.get("supports_water_only", True)),
            scientific_status=str(payload.get("scientific_status", "experimental")),
            default_eligible=bool(payload.get("default_eligible", False)),
        )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return _finite_float(value, "temperature_kelvin")


def load_lsnn_model_card(path: str | Path | None = None) -> LSNNModelCard:
    """Load a JSON-compatible card; PyYAML is intentionally not required."""
    card_path = Path(path) if path is not None else _LSNN_DEFAULT_CARD
    try:
        payload = json.loads(card_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LSNNConfigError(f"LSNN model card not found: {card_path}") from exc
    except json.JSONDecodeError as exc:
        raise LSNNConfigError(
            f"LSNN model card is not JSON-compatible: {card_path}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise LSNNConfigError("LSNN model card payload must be a JSON object.")

    model_card = LSNNModelCard.from_mapping(payload)
    if model_card.model_id == "lsnn-v1" and not model_card.source_revision:
        raise LSNNConfigError("LSNN v1 model card must include pinned source_revision.")
    return model_card


def verify_sha256_file(file_path: str | Path, expected: str | None) -> None:
    """Fail closed unless checksum matches explicit expectation."""
    if not expected:
        raise LSNNConfigError(
            f"LSNN checksum expectation missing for {Path(file_path).name!r}; explicit SHA is required."
        )
    _normalize_sha256(expected, f"expected sha256 for {Path(file_path).name}")
    target = Path(file_path)
    if not target.is_file():
        raise LSNNConfigError(f"LSNN weight artifact not found: {target}")
    actual = _sha256_hex(target)
    if expected.lower() != actual:
        raise LSNNConfigError(
            f"LSNN model checksum mismatch for {target.name}; "
            f"expected {expected.lower()}, got {actual}."
        )


def _sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expect_water_only(solvent: str) -> None:
    if solvent.strip().lower() != "water":
        raise LSNNConfigError(
            "LSNN protocol gate is water-only in this branch; solvent must be water."
        )


def _finite_sequence(values: Sequence[object], field_name: str) -> tuple[float, ...]:
    return tuple(_finite_float(value, f"{field_name}[{index}]") for index, value in enumerate(values))


@dataclass(frozen=True)
class LSNNProtocolRequest:
    """One alchemical geometry + coupling state."""

    symbols: tuple[str, ...]
    positions_angstrom: tuple[float, ...]
    lambda_electrostatics: float
    lambda_sterics: float
    solvent: str = "water"
    total_charge: int | None = None
    multiplicity: int | None = None
    progress: float | None = None
    temperature_kelvin: float = 298.15

    def __post_init__(self) -> None:
        if not self.symbols:
            raise LSNNConfigError("LSNN request requires at least one atom.")
        if len(self.positions_angstrom) != len(self.symbols) * 3:
            raise LSNNConfigError("positions_angstrom must have 3 values per atom.")
        _finite_sequence(self.positions_angstrom, "positions_angstrom")

        for label, value in {
            "lambda_electrostatics": self.lambda_electrostatics,
            "lambda_sterics": self.lambda_sterics,
            "temperature_kelvin": self.temperature_kelvin,
        }.items():
            finite_value = _finite_float(value, label)
            if label.startswith("lambda") and not (0.0 <= finite_value <= 1.0):
                raise LSNNConfigError(f"{label} must be in [0, 1].")
            if label == "temperature_kelvin" and finite_value <= 0.0:
                raise LSNNConfigError("temperature_kelvin must be positive.")

        if self.progress is not None:
            _finite_float(self.progress, "progress")
            if not (0.0 <= float(self.progress) <= 1.0):
                raise LSNNConfigError("progress must be in [0, 1] if provided.")

        object.__setattr__(
            self,
            "total_charge",
            _finite_int(self.total_charge, "total_charge"),
        )
        object.__setattr__(
            self,
            "multiplicity",
            _finite_int(self.multiplicity, "multiplicity"),
        )
        if self.multiplicity is not None and self.multiplicity <= 0:
            raise LSNNConfigError("multiplicity must be >= 1.")


@dataclass(frozen=True)
class LSNNRuntimeResponse:
    """Runtime output expected from a concrete LSNN executor."""

    energy_hartree: float
    dU_dlambda_electrostatics: float
    dU_dlambda_sterics: float
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _ = _finite_float(self.energy_hartree, "energy_hartree")
        _ = _finite_float(self.dU_dlambda_electrostatics, "dU_dlambda_electrostatics")
        _ = _finite_float(self.dU_dlambda_sterics, "dU_dlambda_sterics")


@dataclass(frozen=True)
class LSNNProtocolResult:
    """Typed per-state result with explicit coupling derivatives."""

    request: LSNNProtocolRequest
    energy: float
    lambda_electrostatics: float
    lambda_sterics: float
    dU_dlambda_electrostatics: float
    dU_dlambda_sterics: float

    @property
    def dU_dlambda(self) -> float:
        return self.dU_dlambda_electrostatics + self.dU_dlambda_sterics


@dataclass(frozen=True)
class LSNNTiResult:
    """Scaffold-only single-geometry TI output (not a certified TI estimate)."""

    protocol_status: str
    protocol_type: str
    points: int
    delta_g_hartree: float
    hysteresis_hartree: float | None
    lambda_start: float
    lambda_end: float
    reverse_delta_g_hartree: float | None = None


@dataclass(frozen=True)
class LSNNMbarRequest:
    """Typed MBAR input state."""

    request: LSNNProtocolRequest
    n_frames: int = 1

    def __post_init__(self) -> None:
        if self.n_frames < 1:
            raise LSNNConfigError("LSNNMbarRequest.n_frames must be >= 1.")


@dataclass(frozen=True)
class LSNNMbarResult:
    """Scaffold MBAR output with required uncertainty/sampling/unit/provenance fields."""

    points: int
    delta_g_hartree: float
    overlap: float
    effective_sample_size: float
    uncertainty: float
    uncertainty_units: str
    sampling: Mapping[str, float]
    provenance: Mapping[str, object]
    protocol_type: str
    protocol_status: str
    diagnostics: Mapping[str, float] | None = None


LSNNRuntimeCallable = Callable[[LSNNProtocolRequest], LSNNRuntimeResponse]
LSNNMBARCallable = Callable[[Sequence[LSNNMbarRequest]], Mapping[str, object]]


class LSNNProtocolAdapter:
    """Fail-closed protocol adapter for LSNN-style alchemical workflows."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        card_path: str | Path | None = None,
        runtime: LSNNRuntimeCallable | None = None,
    ) -> None:
        self.model_card = load_lsnn_model_card(card_path)
        self.model_path = Path(model_path)

        self._validate_model_path_and_checksum()
        if runtime is None:
            raise LSNNRuntimeMissingError(
                "No LSNN runtime injected. Provide runtime=callable in this branch."
            )
        self._runtime = runtime

    def _validate_model_path_and_checksum(self) -> None:
        if not self.model_path.is_file():
            raise LSNNConfigError(f"LSNN model file not found: {self.model_path}")

        expected = None
        for artifact in self.model_card.weight_artifacts:
            if artifact["filename"] == self.model_path.name:
                expected = artifact["sha256"]
                break
        if expected is None:
            raise LSNNConfigError(
                f"LSNN model path {self.model_path.name!r} is not a pinned artifact for "
                "this protocol card."
            )
        verify_sha256_file(self.model_path, expected)

    def evaluate(self, request: LSNNProtocolRequest) -> LSNNProtocolResult:
        self._validate_audited_domain(request)
        _expect_water_only(request.solvent)
        if not self.model_card.supports_alchemical_lambda:
            raise LSNNConfigError("LSNN model card does not expose alchemical lambdas.")

        if not self.model_card.supports_water_only:
            raise LSNNConfigError(
                "LSNN model card is not in a water-only branch; this adapter gates water-only paths."
            )

        response = self._runtime(request)
        if not isinstance(response, LSNNRuntimeResponse):
            raise LSNNConfigError(
                "LSNN runtime must return LSNNRuntimeResponse with Hartree and dU/dlambda fields."
            )
        return LSNNProtocolResult(
            request=request,
            energy=float(_finite_float(response.energy_hartree, "runtime.energy_hartree")),
            lambda_electrostatics=float(request.lambda_electrostatics),
            lambda_sterics=float(request.lambda_sterics),
            dU_dlambda_electrostatics=float(_finite_float(response.dU_dlambda_electrostatics, "runtime.dU_dlambda_electrostatics")),
            dU_dlambda_sterics=float(_finite_float(response.dU_dlambda_sterics, "runtime.dU_dlambda_sterics")),
        )

    def evaluate_ti(
        self,
        states: Sequence[LSNNProtocolRequest],
        *,
        reverse_states: Sequence[LSNNProtocolRequest] | None = None,
    ) -> LSNNTiResult:
        return self._evaluate_ti_scaffold(states, reverse_states=reverse_states)

    def evaluate_ti_scaffold(
        self,
        states: Sequence[LSNNProtocolRequest],
        *,
        reverse_states: Sequence[LSNNProtocolRequest] | None = None,
    ) -> LSNNTiResult:
        return self._evaluate_ti_scaffold(states, reverse_states=reverse_states)

    def _evaluate_ti_scaffold(
        self,
        states: Sequence[LSNNProtocolRequest],
        *,
        reverse_states: Sequence[LSNNProtocolRequest] | None = None,
    ) -> LSNNTiResult:
        if not self.model_card.supports_ti:
            raise LSNNConfigError("LSNN model card has TI disabled.")
        if len(states) < 2:
            raise LSNNConfigError("TI requires at least two states.")

        forward_delta = self._integrate_ti_path(states, direction="forward")
        reverse_delta: float | None = None
        if reverse_states is not None:
            if len(reverse_states) < 2:
                raise LSNNConfigError("Reverse TI requires at least two states.")
            self._validate_reverse_endpoints(states, reverse_states)
            reverse_delta = self._integrate_ti_path(reverse_states, direction="reverse")

        hysteresis = (
            None
            if reverse_delta is None
            else abs(forward_delta + reverse_delta) * 0.5
        )

        ordered_states = tuple(states)
        return LSNNTiResult(
            protocol_status="scaffold",
            protocol_type="single-geometry-ti",
            points=len(ordered_states),
            delta_g_hartree=forward_delta,
            hysteresis_hartree=hysteresis,
            lambda_start=float(_finite_float(ordered_states[0].progress, "first state progress"))
            if ordered_states[0].progress is not None
            else 0.0,
            lambda_end=float(_finite_float(ordered_states[-1].progress, "last state progress"))
            if ordered_states[-1].progress is not None
            else float(len(ordered_states) - 1),
            reverse_delta_g_hartree=reverse_delta,
        )

    @staticmethod
    def _validate_reverse_endpoints(
        forward_states: Sequence[LSNNProtocolRequest],
        reverse_states: Sequence[LSNNProtocolRequest],
    ) -> None:
        def endpoint(state: LSNNProtocolRequest) -> tuple[float, float, str, float]:
            return (
                float(state.lambda_electrostatics),
                float(state.lambda_sterics),
                state.solvent.strip().lower(),
                float(state.temperature_kelvin),
            )

        if (
            endpoint(reverse_states[0]) != endpoint(forward_states[-1])
            or endpoint(reverse_states[-1]) != endpoint(forward_states[0])
        ):
            raise LSNNConfigError(
                "Reverse TI endpoints must exactly reverse the forward lambda, "
                "solvent, and temperature endpoints."
            )

    def _integrate_ti_path(
        self, states: Sequence[LSNNProtocolRequest], *, direction: str
    ) -> float:
        if direction not in {"forward", "reverse"}:
            raise LSNNConfigError(f"Unsupported TI direction {direction!r}.")

        ordered = tuple(states)
        if not ordered:
            raise LSNNConfigError("TI path may not be empty.")

        result_points = tuple(self.evaluate(state) for state in ordered)

        prev = result_points[0]
        if prev.request.progress is None:
            raise LSNNConfigError("All TI states must set progress.")

        target_sign = 1.0 if direction == "forward" else -1.0
        total = 0.0
        for current in result_points[1:]:
            if current.request.progress is None:
                raise LSNNConfigError("All TI states must set progress.")
            prev_progress = prev.request.progress
            curr_progress = current.request.progress
            if prev_progress is None or curr_progress is None:
                raise LSNNConfigError("All TI states must set progress.")
            delta_progress = curr_progress - prev_progress
            if not math.isfinite(delta_progress):
                raise LSNNConfigError("TI state progress must be finite and valid.")
            if delta_progress == 0.0:
                raise LSNNConfigError("TI state progress must be strictly monotonic.")
            if delta_progress / abs(delta_progress) != target_sign:
                raise LSNNConfigError(
                    "TI forward states must progress forward and reverse states must progress in reverse."
                )

            scale_e = (float(current.lambda_electrostatics) - float(prev.lambda_electrostatics)) / delta_progress
            scale_s = (float(current.lambda_sterics) - float(prev.lambda_sterics)) / delta_progress
            integrand = 0.5 * (
                (float(prev.dU_dlambda_electrostatics) + float(current.dU_dlambda_electrostatics))
                * scale_e
                + (float(prev.dU_dlambda_sterics) + float(current.dU_dlambda_sterics))
                * scale_s
            )
            if not math.isfinite(integrand):
                raise LSNNConfigError("TI derivative accumulation produced non-finite value.")
            if not math.isfinite(delta_progress):
                raise LSNNConfigError("TI progress spacing must be finite.")
            total += float(integrand * delta_progress)
            prev = current

        return _finite_float(total, "TI integral")

    def evaluate_mbar(self, states: Sequence[LSNNMbarRequest]) -> LSNNMbarResult:
        if not self.model_card.supports_mbar:
            raise LSNNConfigError("LSNN model card has MBAR disabled.")
        if not states:
            raise LSNNConfigError("MBAR requires one or more states.")

        mbar = getattr(self._runtime, "mbar", None)
        if not callable(mbar):
            raise LSNNRuntimeMissingError(
                "Runtime does not provide an MBAR estimator hook."
            )
        state_tuple = tuple(states)
        _ = tuple(self.evaluate(state.request) for state in state_tuple)
        payload = cast(Mapping[str, object], mbar(state_tuple))
        if not isinstance(payload, Mapping):
            raise LSNNConfigError("MBAR runtime hook must return a mapping payload.")

        delta = float(
            _finite_float(
                payload.get("delta_g_hartree"),
                "MBAR payload delta_g_hartree",
            )
        )
        overlap = float(
            _finite_float(
                payload.get("overlap"),
                "MBAR payload overlap",
            )
        )
        if not (0.0 <= overlap <= 1.0):
            raise LSNNConfigError("MBAR payload overlap must be in [0, 1].")
        ess = float(_finite_float(payload.get("effective_sample_size"), "MBAR payload effective_sample_size"))
        if ess < 0.0:
            raise LSNNConfigError("MBAR payload effective_sample_size must be >= 0.")

        uncertainty = float(_finite_float(payload.get("uncertainty"), "MBAR payload uncertainty"))
        uncertainty_units = _optional_str(payload.get("uncertainty_units"))
        if uncertainty_units not in {"hartree", "kcal/mol", "kJ/mol"}:
            raise LSNNConfigError(
                "MBAR payload uncertainty_units must be one of: hartree, kcal/mol, kJ/mol."
            )

        sampling_raw = payload.get("sampling")
        if not isinstance(sampling_raw, Mapping) or not sampling_raw:
            raise LSNNConfigError("MBAR payload must provide non-empty sampling metadata.")
        sampling: dict[str, float] = {}
        for key, value in sampling_raw.items():
            if not isinstance(key, str) or not key:
                raise LSNNConfigError("MBAR payload sampling keys must be non-empty strings.")
            sampling[key] = _finite_float(value, f"sampling[{key}]")

        provenance_raw = payload.get("provenance")
        if not isinstance(provenance_raw, Mapping) or not provenance_raw:
            raise LSNNConfigError("MBAR payload must provide non-empty provenance metadata.")
        provenance = dict(provenance_raw)

        diagnostics: dict[str, float] = {}
        for key, value in payload.items():
            if key in {
                "delta_g_hartree",
                "overlap",
                "effective_sample_size",
                "uncertainty",
                "uncertainty_units",
                "sampling",
                "provenance",
            }:
                continue
            if isinstance(value, (float, int)) and math.isfinite(float(value)):
                diagnostics[key] = float(value)

        return LSNNMbarResult(
            points=len(state_tuple),
            delta_g_hartree=delta,
            overlap=overlap,
            effective_sample_size=ess,
            uncertainty=uncertainty,
            uncertainty_units=uncertainty_units,
            sampling=sampling,
            provenance=provenance,
            protocol_type="scaffold-mbar",
            protocol_status="scaffold",
            diagnostics=diagnostics,
        )

    def _validate_audited_domain(self, request: LSNNProtocolRequest) -> None:
        if not self.model_card.audited_domain_enabled:
            raise LSNNConfigError(
                "LSNN public card lacks an explicitly audited domain; execution is blocked."
            )

        if request.total_charge is None or request.multiplicity is None:
            raise LSNNConfigError(
                "Request must provide total_charge and multiplicity for audited-domain checks."
            )

        if self.model_card.audited_elements and set(request.symbols).difference(
            self.model_card.audited_elements
        ):
            missing = sorted(
                set(request.symbols).difference(self.model_card.audited_elements)
            )
            raise LSNNConfigError(
                f"Request contains symbols outside audited LSNN domain: {missing}."
            )

        if self.model_card.audited_charge_range is not None:
            min_charge, max_charge = self.model_card.audited_charge_range
            if request.total_charge < min_charge or request.total_charge > max_charge:
                raise LSNNConfigError(
                    f"total_charge={request.total_charge} outside audited domain {min_charge}..{max_charge}."
                )

        if request.multiplicity not in self.model_card.audited_multiplicities:
            raise LSNNConfigError(
                f"multiplicity={request.multiplicity} not in audited domain {self.model_card.audited_multiplicities}."
            )
