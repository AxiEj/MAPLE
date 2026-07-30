"""Deterministic generation provenance for Route-2 V0 RISM assets.

The frozen XVV contains a wall-clock timestamp and the textual solver report
contains CPU timings, so raw output hashes are expected to differ.  This
module instead verifies two independent source-identical runs through:

* exact Cvv and thermodynamic self-test hashes;
* an XVV hash with only the ``DATE`` value in its first ``%VERSION`` line
  normalized, while retaining each raw first line so every raw XVV hash can
  be reconstructed; and
* the complete residual sequences embedded in the hash-bound provenance file.

The certificate carries no target-solvation values and performs no fitting.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

V0_RISM_GENERATION_SOURCE_CONSTRUCTION = "route2-v0-rism-generation-provenance-v1"
V0_RISM_GENERATION_SOURCE_STATUS = (
    "source-complete-candidate-not-production-or-accuracy-admitted"
)
V0_RISM_GENERATION_HASH_ROLES = frozenset(
    {
        "site_model",
        "rism1d_input",
        "xvv",
        "cvv",
        "thermodynamic_output",
    }
)
V0_RISM_REQUIRED_EXCLUDED_TARGET_LABEL_SETS = frozenset(
    {"mnsol", "freesolv", "development", "confirmation", "blind"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ROOT_KEYS = frozenset(
    {
        "construction",
        "status",
        "claim_boundary",
        "source_literature",
        "generator",
        "no_target_policy",
        "frozen_source_sha256",
        "residual_tolerance",
        "runs",
        "not_claimed",
    }
)
_GENERATOR_KEYS = frozenset(
    {
        "package",
        "version",
        "build",
        "channel_url",
        "package_sha256",
        "license_expression",
        "rism1d_binary_sha256",
        "site_model_package_path",
        "command",
    }
)
_LITERATURE_KEYS = frozenset({"role", "title", "doi", "url"})
_NO_TARGET_KEYS = frozenset(
    {
        "post_training",
        "fine_tuning",
        "experimental_solvation_fit",
        "map_or_uq_calibration",
        "target_solvation_labels_used",
        "excluded_target_label_sets",
    }
)
_RUN_KEYS = frozenset(
    {
        "raw_xvv_sha256",
        "normalized_xvv_sha256",
        "xvv_first_line",
        "cvv_sha256",
        "thermodynamic_output_sha256",
        "transcript_sha256",
        "transcript",
    }
)
_STEP = re.compile(
    r"^\s*step=\s*(?P<step>\d+)\s+Res=\s*(?P<residual>\S+)",
    flags=re.MULTILINE,
)
_VERSION_DATE = re.compile(
    rb"(?P<prefix>\bDATE[ \t]*=[ \t]*)"
    rb"(?P<value>[^ \t\r\n]+[ \t]+[^ \t\r\n]+)"
    rb"(?P<suffix>[ \t]*\r?)$"
)


def _strict_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object with string keys.")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
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


def _digest(value: object, *, name: str) -> str:
    digest = _nonempty(value, name=name).lower()
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return digest


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{name} must be finite and positive.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive.") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _finite_float(value: str, *, name: str) -> float:
    try:
        number = float(value.replace("D", "E").replace("d", "e"))
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _string_list(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list.")
    result = tuple(_nonempty(item, name=f"{name} item") for item in value)
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return result


def _normalize_xvv_version_date(content: bytes) -> bytes:
    """Replace only the first-line DATE value while preserving version metadata."""

    first, separator, remainder = content.partition(b"\n")
    if not separator or not first.startswith(b"%VERSION"):
        raise ValueError("RISM XVV must start with a timestamp-bearing %VERSION line.")
    normalized_first, replacements = _VERSION_DATE.subn(
        rb"\g<prefix><TIMESTAMP>\g<suffix>",
        first,
    )
    if replacements != 1:
        raise ValueError(
            "RISM XVV first line must contain exactly one terminal DATE value."
        )
    return normalized_first + separator + remainder


def normalized_rism1d_xvv_sha256(path: str | Path) -> str:
    """Hash XVV bytes after normalizing only the first-line DATE value."""

    return hashlib.sha256(
        _normalize_xvv_version_date(Path(path).read_bytes())
    ).hexdigest()


def _steps(body: str, *, name: str) -> tuple[tuple[int, float], ...]:
    values = tuple(
        (
            int(match.group("step")),
            _finite_float(match.group("residual"), name=f"{name} residual"),
        )
        for match in _STEP.finditer(body)
    )
    if not values:
        raise ValueError(f"{name} transcript contains no residual steps.")
    observed = tuple(step for step, _ in values)
    if observed != tuple(range(1, len(values) + 1)):
        raise ValueError(f"{name} transcript steps must be consecutive from one.")
    if any(residual < 0.0 for _, residual in values):
        raise ValueError(f"{name} transcript residuals must be nonnegative.")
    return values


def _convergence_records(
    transcript: str,
) -> tuple[tuple[tuple[int, float], ...], tuple[tuple[int, float], ...]]:
    if (
        transcript.count("relaxing RISM:") != 1
        or transcript.count("relaxing RISM DT:") != 1
    ):
        raise ValueError(
            "RISM generation transcript must contain one primary and one "
            "temperature-derivative solve."
        )
    primary_tail = transcript.split("relaxing RISM:", 1)[1]
    if "done." not in primary_tail:
        raise ValueError("RISM primary transcript is missing its completion marker.")
    primary = _steps(primary_tail.split("done.", 1)[0], name="Primary RISM")

    derivative_tail = transcript.split("relaxing RISM DT:", 1)[1]
    if "outputting Xvv" not in derivative_tail:
        raise ValueError(
            "RISM temperature-derivative transcript is missing its output marker."
        )
    derivative = _steps(
        derivative_tail.split("outputting Xvv", 1)[0],
        name="Temperature-derivative RISM",
    )
    return primary, derivative


@dataclass(frozen=True)
class Route2V0RismGenerationRun:
    """One independently executed source-identical ``rism1d`` run."""

    raw_xvv_sha256: str
    normalized_xvv_sha256: str
    xvv_first_line: str
    cvv_sha256: str
    thermodynamic_output_sha256: str
    transcript_sha256: str
    transcript: str = field(repr=False)
    residual_tolerance: float = field(repr=False, compare=False)
    primary_iterations: int = field(init=False)
    primary_final_residual: float = field(init=False)
    temperature_derivative_iterations: int = field(init=False)
    temperature_derivative_final_residual: float = field(init=False)

    def __post_init__(self) -> None:
        raw_xvv = _digest(self.raw_xvv_sha256, name="Raw XVV SHA-256")
        normalized_xvv = _digest(
            self.normalized_xvv_sha256,
            name="Normalized XVV SHA-256",
        )
        xvv_first_line = self.xvv_first_line
        if (
            not isinstance(xvv_first_line, str)
            or not xvv_first_line.strip()
            or not xvv_first_line.startswith("%VERSION")
            or "\n" in xvv_first_line
            or "\r" in xvv_first_line
        ):
            raise ValueError(
                "XVV first line must be one timestamp-bearing %VERSION line."
            )
        _ = _normalize_xvv_version_date(xvv_first_line.encode("utf-8") + b"\n")
        cvv = _digest(self.cvv_sha256, name="Cvv SHA-256")
        thermodynamic = _digest(
            self.thermodynamic_output_sha256,
            name="Thermodynamic-output SHA-256",
        )
        transcript_sha = _digest(
            self.transcript_sha256,
            name="RISM transcript SHA-256",
        )
        transcript = self.transcript
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("RISM transcript must be a nonempty string.")
        observed_transcript_sha = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
        if observed_transcript_sha != transcript_sha:
            raise ValueError("RISM transcript content hash does not match its digest.")
        tolerance = _positive_float(
            self.residual_tolerance,
            name="RISM residual tolerance",
        )
        primary, derivative = _convergence_records(transcript)
        primary_final = primary[-1][1]
        derivative_final = derivative[-1][1]
        if primary_final > tolerance or derivative_final > tolerance:
            raise ValueError("RISM generation run did not meet its residual tolerance.")
        object.__setattr__(self, "raw_xvv_sha256", raw_xvv)
        object.__setattr__(self, "normalized_xvv_sha256", normalized_xvv)
        object.__setattr__(self, "xvv_first_line", xvv_first_line)
        object.__setattr__(self, "cvv_sha256", cvv)
        object.__setattr__(self, "thermodynamic_output_sha256", thermodynamic)
        object.__setattr__(self, "transcript_sha256", transcript_sha)
        object.__setattr__(self, "transcript", transcript)
        object.__setattr__(self, "residual_tolerance", tolerance)
        object.__setattr__(self, "primary_iterations", primary[-1][0])
        object.__setattr__(self, "primary_final_residual", primary_final)
        object.__setattr__(
            self,
            "temperature_derivative_iterations",
            derivative[-1][0],
        )
        object.__setattr__(
            self,
            "temperature_derivative_final_residual",
            derivative_final,
        )


@dataclass(frozen=True)
class Route2V0RismGenerationSource:
    """Two-run deterministic provenance for one frozen RISM source."""

    claim_boundary: str
    source_literature: tuple[Mapping[str, str], ...]
    generator: Mapping[str, str]
    frozen_source_sha256: Mapping[str, str]
    residual_tolerance: float
    runs: tuple[Route2V0RismGenerationRun, ...]
    not_claimed: tuple[str, ...]
    excluded_target_label_sets: tuple[str, ...]
    construction: str = V0_RISM_GENERATION_SOURCE_CONSTRUCTION
    status: str = V0_RISM_GENERATION_SOURCE_STATUS

    def __post_init__(self) -> None:
        if self.construction != V0_RISM_GENERATION_SOURCE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 RISM generation provenance.")
        if self.status != V0_RISM_GENERATION_SOURCE_STATUS:
            raise ValueError("RISM generation provenance must remain candidate-only.")
        claim = _nonempty(self.claim_boundary, name="RISM claim boundary")
        literature = tuple(self.source_literature)
        if not literature:
            raise ValueError("RISM generation provenance needs primary literature.")
        normalized_literature: list[Mapping[str, str]] = []
        for index, source in enumerate(literature):
            entry = _strict_mapping(source, name=f"source_literature[{index}]")
            _strict_keys(entry, _LITERATURE_KEYS, name=f"source_literature[{index}]")
            normalized_literature.append(
                MappingProxyType(
                    {
                        key: _nonempty(
                            entry[key],
                            name=f"source_literature[{index}].{key}",
                        )
                        for key in sorted(_LITERATURE_KEYS)
                    }
                )
            )
        generator = _strict_mapping(self.generator, name="RISM generator")
        _strict_keys(generator, _GENERATOR_KEYS, name="RISM generator")
        normalized_generator = {
            key: _nonempty(generator[key], name=f"generator.{key}")
            for key in _GENERATOR_KEYS
        }
        normalized_generator["package_sha256"] = _digest(
            generator["package_sha256"],
            name="generator.package_sha256",
        )
        normalized_generator["rism1d_binary_sha256"] = _digest(
            generator["rism1d_binary_sha256"],
            name="generator.rism1d_binary_sha256",
        )
        hashes = _strict_mapping(
            self.frozen_source_sha256,
            name="Frozen RISM source hashes",
        )
        _strict_keys(
            hashes,
            V0_RISM_GENERATION_HASH_ROLES,
            name="frozen_source_sha256",
        )
        normalized_hashes = {
            role: _digest(value, name=f"frozen_source_sha256.{role}")
            for role, value in hashes.items()
        }
        tolerance = _positive_float(
            self.residual_tolerance,
            name="RISM residual tolerance",
        )
        runs = tuple(self.runs)
        if len(runs) != 2 or any(
            not isinstance(run, Route2V0RismGenerationRun) for run in runs
        ):
            raise ValueError(
                "RISM generation provenance requires exactly two validated runs."
            )
        if any(run.residual_tolerance != tolerance for run in runs):
            raise ValueError("RISM generation runs must use the declared tolerance.")
        if runs[0].raw_xvv_sha256 != normalized_hashes["xvv"]:
            raise ValueError("The first RISM run must be the frozen raw XVV source.")
        if any(run.cvv_sha256 != normalized_hashes["cvv"] for run in runs):
            raise ValueError("Independent RISM runs must reproduce the frozen Cvv.")
        if any(
            run.thermodynamic_output_sha256 != normalized_hashes["thermodynamic_output"]
            for run in runs
        ):
            raise ValueError(
                "Independent RISM runs must reproduce the frozen thermodynamic output."
            )
        normalized_xvv_hashes = {run.normalized_xvv_sha256 for run in runs}
        if len(normalized_xvv_hashes) != 1:
            raise ValueError(
                "Independent RISM runs must reproduce timestamp-normalized XVV."
            )
        if runs[0].raw_xvv_sha256 == runs[1].raw_xvv_sha256:
            raise ValueError(
                "Independent RISM runs must have distinct raw XVV evidence."
            )
        if runs[0].transcript_sha256 == runs[1].transcript_sha256:
            raise ValueError(
                "Independent RISM runs must have distinct transcript evidence."
            )
        excluded = tuple(value.lower() for value in self.excluded_target_label_sets)
        if len(set(excluded)) != len(excluded):
            raise ValueError("Excluded target-label sets must be unique.")
        missing = V0_RISM_REQUIRED_EXCLUDED_TARGET_LABEL_SETS.difference(excluded)
        if missing:
            raise ValueError(
                "RISM generation provenance must exclude target-label sets: "
                f"{', '.join(sorted(missing))}."
            )
        not_claimed = tuple(self.not_claimed)
        if not not_claimed:
            raise ValueError("RISM generation provenance must state its open gates.")
        object.__setattr__(self, "claim_boundary", claim)
        object.__setattr__(self, "source_literature", tuple(normalized_literature))
        object.__setattr__(
            self,
            "generator",
            MappingProxyType(dict(sorted(normalized_generator.items()))),
        )
        object.__setattr__(
            self,
            "frozen_source_sha256",
            MappingProxyType(dict(sorted(normalized_hashes.items()))),
        )
        object.__setattr__(self, "residual_tolerance", tolerance)
        object.__setattr__(self, "runs", runs)
        object.__setattr__(
            self,
            "not_claimed",
            tuple(_nonempty(value, name="not_claimed item") for value in not_claimed),
        )
        object.__setattr__(self, "excluded_target_label_sets", excluded)

    @property
    def normalized_xvv_sha256(self) -> str:
        """Return the common timestamp-normalized XVV digest."""

        return self.runs[0].normalized_xvv_sha256

    def verify_xvv_reproduction(self, path: str | Path) -> None:
        """Reconstruct and verify each run from one frozen XVV body."""

        source = Path(path)
        content = source.read_bytes()
        first, separator, remainder = content.partition(b"\n")
        if not separator or not first.startswith(b"%VERSION"):
            raise ValueError(
                "RISM XVV must start with a timestamp-bearing %VERSION line."
            )
        normalized = hashlib.sha256(_normalize_xvv_version_date(content)).hexdigest()
        if normalized != self.normalized_xvv_sha256:
            raise ValueError(
                "Frozen solvent XVV does not match the reproducible normalized digest."
            )
        for index, run in enumerate(self.runs, start=1):
            reconstructed = run.xvv_first_line.encode("utf-8") + separator + remainder
            if hashlib.sha256(reconstructed).hexdigest() != run.raw_xvv_sha256:
                raise ValueError(
                    "RISM generation run "
                    f"{index} raw XVV cannot be reconstructed from its timestamp "
                    "line and the frozen normalized body."
                )
            if (
                hashlib.sha256(_normalize_xvv_version_date(reconstructed)).hexdigest()
                != run.normalized_xvv_sha256
            ):
                raise ValueError(
                    f"RISM generation run {index} changes non-DATE XVV metadata."
                )


def parse_route2_v0_rism_generation_source(
    text: str,
) -> Route2V0RismGenerationSource:
    """Parse one strict two-run generation-provenance JSON document."""

    if not isinstance(text, str):
        raise TypeError("RISM generation provenance must be text.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("RISM generation provenance must be valid JSON.") from exc
    root = _strict_mapping(payload, name="RISM generation provenance")
    _strict_keys(root, _ROOT_KEYS, name="RISM generation provenance")

    no_target = _strict_mapping(root["no_target_policy"], name="no_target_policy")
    _strict_keys(no_target, _NO_TARGET_KEYS, name="no_target_policy")
    for key in _NO_TARGET_KEYS.difference({"excluded_target_label_sets"}):
        if no_target[key] is not False:
            raise ValueError(f"no_target_policy.{key} must be explicitly false.")
    excluded = _string_list(
        no_target["excluded_target_label_sets"],
        name="excluded_target_label_sets",
    )
    tolerance = _positive_float(
        root["residual_tolerance"],
        name="residual_tolerance",
    )
    run_values = root["runs"]
    if not isinstance(run_values, list):
        raise TypeError("RISM generation runs must be a list.")
    runs: list[Route2V0RismGenerationRun] = []
    for index, value in enumerate(run_values):
        run = _strict_mapping(value, name=f"runs[{index}]")
        _strict_keys(run, _RUN_KEYS, name=f"runs[{index}]")
        runs.append(
            Route2V0RismGenerationRun(
                raw_xvv_sha256=run["raw_xvv_sha256"],
                normalized_xvv_sha256=run["normalized_xvv_sha256"],
                xvv_first_line=run["xvv_first_line"],
                cvv_sha256=run["cvv_sha256"],
                thermodynamic_output_sha256=run["thermodynamic_output_sha256"],
                transcript_sha256=run["transcript_sha256"],
                transcript=run["transcript"],
                residual_tolerance=tolerance,
            )
        )
    literature_values = root["source_literature"]
    if not isinstance(literature_values, list):
        raise TypeError("source_literature must be a list.")
    return Route2V0RismGenerationSource(
        claim_boundary=root["claim_boundary"],
        source_literature=tuple(
            _strict_mapping(value, name=f"source_literature[{index}]")
            for index, value in enumerate(literature_values)
        ),
        generator=_strict_mapping(root["generator"], name="generator"),
        frozen_source_sha256=_strict_mapping(
            root["frozen_source_sha256"],
            name="frozen_source_sha256",
        ),
        residual_tolerance=tolerance,
        runs=tuple(runs),
        not_claimed=_string_list(root["not_claimed"], name="not_claimed"),
        excluded_target_label_sets=excluded,
        construction=root["construction"],
        status=root["status"],
    )


def load_route2_v0_rism_generation_source(
    path: str | Path,
) -> Route2V0RismGenerationSource:
    """Load one strict RISM generation-provenance document."""

    return parse_route2_v0_rism_generation_source(
        Path(path).read_text(encoding="utf-8")
    )


__all__ = [
    "V0_RISM_GENERATION_HASH_ROLES",
    "V0_RISM_GENERATION_SOURCE_CONSTRUCTION",
    "V0_RISM_GENERATION_SOURCE_STATUS",
    "V0_RISM_REQUIRED_EXCLUDED_TARGET_LABEL_SETS",
    "Route2V0RismGenerationRun",
    "Route2V0RismGenerationSource",
    "load_route2_v0_rism_generation_source",
    "normalized_rism1d_xvv_sha256",
    "parse_route2_v0_rism_generation_source",
]
