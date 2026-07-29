"""Benchmark identity, leakage, metrics, and artifact contracts for Route 4.

The module deliberately does not download or redistribute benchmark data.  It
binds local records to a complete protocol identity, enforces conservative
training-overlap labels, and writes reproducible result artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _normalized_ids(values: Iterable[str], *, field: str) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{field} must contain one or more non-empty identifiers.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} contains duplicate identifiers.")
    return normalized


@dataclass(frozen=True)
class BenchmarkIdentity:
    """Everything that must match before two accuracy panels are compared."""

    dataset: str
    dataset_version: str
    record_ids: tuple[str, ...]
    temperature_kelvin: float
    standard_state: str
    protonation_policy: str
    tautomer_policy: str
    conformer_policy: str
    geometry_protocol: str
    solvent_protocol: str
    potential: str
    solvation_backend: str
    cavity_model: str
    sampling_protocol: str
    estimator: str
    experimental_provenance: str

    def __post_init__(self) -> None:
        for field in (
            "dataset",
            "dataset_version",
            "standard_state",
            "protonation_policy",
            "tautomer_policy",
            "conformer_policy",
            "geometry_protocol",
            "solvent_protocol",
            "potential",
            "solvation_backend",
            "cavity_model",
            "sampling_protocol",
            "estimator",
            "experimental_provenance",
        ):
            if not str(getattr(self, field)).strip():
                raise ValueError(f"{field} must be explicit and non-empty.")
        if (
            not math.isfinite(float(self.temperature_kelvin))
            or self.temperature_kelvin <= 0
        ):
            raise ValueError("temperature_kelvin must be finite and positive.")
        object.__setattr__(
            self,
            "record_ids",
            _normalized_ids(self.record_ids, field="record_ids"),
        )

    def canonical_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["record_ids"] = list(self.record_ids)
        return payload

    def _hash_payload(self, payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._hash_payload(self.canonical_payload())

    @property
    def panel_fingerprint(self) -> str:
        """Fingerprint for scientifically comparable panels (identity minus potential/model id)."""
        payload = self.canonical_payload()
        payload = dict(payload)
        payload.pop("potential", None)
        return self._hash_payload(payload)


class LeakageStatus(str, Enum):
    KNOWN_OVERLAP = "known_overlap"
    OVERLAP_UNKNOWN = "overlap_unknown"
    STRICT_HOLDOUT = "strict_holdout"


@dataclass(frozen=True)
class ModelTrainingEvidence:
    """Auditable evidence about the exact records used to train a model."""

    training_datasets: tuple[str, ...] = ()
    known_training_record_ids: tuple[str, ...] = ()
    explicitly_excluded_datasets: tuple[str, ...] = ()
    explicitly_excluded_record_ids: tuple[str, ...] = ()
    record_accounting_complete: bool = False
    evidence_source: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "training_datasets",
            "known_training_record_ids",
            "explicitly_excluded_datasets",
            "explicitly_excluded_record_ids",
        ):
            values = tuple(
                value
                for value in (str(item).strip() for item in getattr(self, field))
                if value
            )
            object.__setattr__(self, field, values)


@dataclass(frozen=True)
class LeakageAudit:
    status: LeakageStatus
    overlapping_record_ids: tuple[str, ...]
    rationale: str
    evidence_source: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "overlapping_record_ids": list(self.overlapping_record_ids),
            "rationale": self.rationale,
            "evidence_source": self.evidence_source,
        }


def audit_training_overlap(
    identity: BenchmarkIdentity,
    evidence: ModelTrainingEvidence,
) -> LeakageAudit:
    """Label overlap conservatively; missing evidence is never a holdout."""

    benchmark_ids = set(identity.record_ids)
    overlaps = tuple(
        sorted(benchmark_ids.intersection(evidence.known_training_record_ids))
    )
    if overlaps:
        return LeakageAudit(
            LeakageStatus.KNOWN_OVERLAP,
            overlaps,
            "One or more benchmark record identifiers are documented training records.",
            evidence.evidence_source,
        )

    dataset_key = identity.dataset.casefold()
    used_datasets = {item.casefold() for item in evidence.training_datasets}
    excluded_datasets = {
        item.casefold() for item in evidence.explicitly_excluded_datasets
    }
    excluded_records = set(evidence.explicitly_excluded_record_ids)
    every_record_excluded = benchmark_ids.issubset(excluded_records)

    if (
        evidence.record_accounting_complete
        and (dataset_key in excluded_datasets or every_record_excluded)
        and dataset_key not in used_datasets
    ):
        return LeakageAudit(
            LeakageStatus.STRICT_HOLDOUT,
            (),
            "Complete training accounting explicitly excludes this dataset or every benchmark record.",
            evidence.evidence_source,
        )

    if dataset_key in used_datasets:
        rationale = (
            "The benchmark dataset is named in training evidence, but exact record overlap "
            "cannot be resolved."
        )
    else:
        rationale = (
            "Training evidence does not completely account for every benchmark record; "
            "absence of a documented overlap is not holdout proof."
        )
    return LeakageAudit(
        LeakageStatus.OVERLAP_UNKNOWN,
        (),
        rationale,
        evidence.evidence_source,
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denominator)


def _kendall_tau_b(left: np.ndarray, right: np.ndarray) -> float:
    concordant = discordant = ties_left = ties_right = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            dx = np.sign(left[j] - left[i])
            dy = np.sign(right[j] - right[i])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_left += 1
            elif dy == 0:
                ties_right += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_left) * (concordant + discordant + ties_right)
    )
    if denominator == 0:
        return float("nan")
    return float((concordant - discordant) / denominator)


def _metric_bundle(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - reference
    absolute = np.abs(error)
    ss_total = float(np.sum((reference - reference.mean()) ** 2))
    r_squared = (
        float(1.0 - np.sum(error**2) / ss_total) if ss_total > 0 else float("nan")
    )
    return {
        "mae_kcal_mol": float(absolute.mean()),
        "rmse_kcal_mol": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error_kcal_mol": float(np.median(absolute)),
        "mean_signed_error_kcal_mol": float(error.mean()),
        "r_squared": r_squared,
        "spearman_rho": _correlation(
            _average_ranks(reference), _average_ranks(prediction)
        ),
        "kendall_tau_b": _kendall_tau_b(reference, prediction),
    }


def summarize_predictions(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize finite predictions and preserve coverage/failure accounting."""

    if not records:
        raise ValueError("At least one benchmark record is required.")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative.")

    seen: set[str] = set()
    usable: list[tuple[float, float]] = []
    failures = 0
    for record in records:
        record_id = str(record.get("record_id", "")).strip()
        if not record_id:
            raise ValueError("Every record must define a non-empty record_id.")
        if record_id in seen:
            raise ValueError(f"Duplicate benchmark record_id: {record_id}")
        seen.add(record_id)
        reference = record.get("experimental_kcal_mol")
        prediction = record.get("predicted_kcal_mol")
        try:
            reference_value = float(reference)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Benchmark record {record_id!r} has invalid experimental_kcal_mol."
            ) from exc
        if not math.isfinite(reference_value):
            raise ValueError(
                f"Benchmark record {record_id!r} has non-finite experimental_kcal_mol."
            )

        try:
            prediction_value = float(prediction)
        except (TypeError, ValueError):
            failures += 1
            continue
        if not math.isfinite(prediction_value):
            failures += 1
            continue
        usable.append((reference_value, prediction_value))

    count = len(usable)
    result: dict[str, Any] = {
        "record_count": len(records),
        "evaluated_count": count,
        "failure_count": failures,
        "coverage": count / len(records),
        "failure_rate": failures / len(records),
    }
    if not usable:
        result["metrics"] = None
        result["mae_bootstrap_95_ci_kcal_mol"] = None
        return result

    reference = np.asarray([item[0] for item in usable], dtype=float)
    prediction = np.asarray([item[1] for item in usable], dtype=float)
    result["metrics"] = _metric_bundle(reference, prediction)

    if bootstrap_samples == 0:
        result["mae_bootstrap_95_ci_kcal_mol"] = None
    else:
        generator = np.random.default_rng(seed)
        indices = generator.integers(0, count, size=(bootstrap_samples, count))
        absolute_error = np.abs(prediction - reference)
        bootstrap_mae = absolute_error[indices].mean(axis=1)
        low, high = np.percentile(bootstrap_mae, (2.5, 97.5))
        result["mae_bootstrap_95_ci_kcal_mol"] = [float(low), float(high)]
    return result


@dataclass(frozen=True)
class PairedComparison:
    panel_fingerprint: str
    run_fingerprint_a: str
    run_fingerprint_b: str
    record_count: int
    mean_delta_absolute_error_kcal_mol: float
    wins_a: int
    ties: int
    wins_b: int

    @property
    def benchmark_fingerprint(self) -> str:
        """Backward-compatible name for the shared scientific panel identity."""
        return self.panel_fingerprint


def paired_comparison(
    identity_a: BenchmarkIdentity,
    records_a: Sequence[Mapping[str, Any]],
    identity_b: BenchmarkIdentity,
    records_b: Sequence[Mapping[str, Any]],
) -> PairedComparison:
    """Compare models only when scientific panel dimensions are identical.

    Full run identifiers may differ (e.g., potential/model version) as long as the
    panel context used for the comparison is identical.
    """

    if identity_a.panel_fingerprint != identity_b.panel_fingerprint:
        raise ValueError(
            "Paired accuracy comparison requires identical scientific panel identities."
        )

    def collect(records: Sequence[Mapping[str, Any]]) -> dict[str, tuple[float, float]]:
        collected: dict[str, tuple[float, float]] = {}
        for record in records:
            record_id = str(record.get("record_id", "")).strip()
            if record_id in collected:
                raise ValueError(f"Duplicate paired record_id: {record_id}")
            try:
                reference = float(record["experimental_kcal_mol"])
                prediction = float(record["predicted_kcal_mol"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Paired record {record_id!r} lacks a finite prediction/reference."
                ) from exc
            if not (
                record_id and math.isfinite(reference) and math.isfinite(prediction)
            ):
                raise ValueError(
                    f"Paired record {record_id!r} lacks a finite prediction/reference."
                )
            collected[record_id] = (reference, prediction)
        return collected

    left = collect(records_a)
    right = collect(records_b)
    expected = set(identity_a.record_ids)
    if set(left) != expected or set(right) != expected:
        raise ValueError(
            "Paired comparison requires exactly one successful result for every identity record."
        )

    deltas: list[float] = []
    wins_a = ties = wins_b = 0
    for record_id in identity_a.record_ids:
        reference_a, prediction_a = left[record_id]
        reference_b, prediction_b = right[record_id]
        if reference_a != reference_b:
            raise ValueError(
                f"Experimental value differs for paired record {record_id!r}."
            )
        error_a = abs(prediction_a - reference_a)
        error_b = abs(prediction_b - reference_b)
        deltas.append(error_a - error_b)
        if math.isclose(error_a, error_b, rel_tol=0.0, abs_tol=1e-12):
            ties += 1
        elif error_a < error_b:
            wins_a += 1
        else:
            wins_b += 1

    return PairedComparison(
        panel_fingerprint=identity_a.panel_fingerprint,
        run_fingerprint_a=identity_a.fingerprint,
        run_fingerprint_b=identity_b.fingerprint,
        record_count=len(identity_a.record_ids),
        mean_delta_absolute_error_kcal_mol=float(np.mean(deltas)),
        wins_a=wins_a,
        ties=ties,
        wins_b=wins_b,
    )


class BenchmarkResultStore:
    """Atomic, path-safe writer for the Route-4 benchmark artifact schema."""

    REQUIRED_ARTIFACTS = frozenset(
        {
            "model_card.yaml",
            "protocol.yaml",
            "environment.lock",
            "per_record.csv",
            "failures.csv",
            "diagnostics.json",
            "uncertainty.csv",
            "leakage_report.json",
            "speed.json",
            "summary.json",
        }
    )

    def __init__(self, root: str | Path, model_id: str) -> None:
        if not model_id or Path(model_id).name != model_id or model_id in {".", ".."}:
            raise ValueError("model_id must be one path-safe directory name.")
        self.directory = Path(root) / model_id
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_filename(filename: str) -> str:
        normalized = str(filename).strip()
        if (
            not normalized
            or Path(normalized).name != normalized
            or normalized in {".", ".."}
        ):
            raise ValueError("artifact filename must be one path-safe file name.")
        return normalized

    def _atomic_text(self, filename: str, text: str) -> Path:
        filename = self._validate_filename(filename)
        target = self.directory / filename
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.directory,
            delete=False,
        ) as handle:
            handle.write(text)
            temp_name = handle.name
        os.replace(temp_name, target)
        return target

    def write_json(self, filename: str, payload: Mapping[str, Any]) -> Path:
        return self._atomic_text(
            filename,
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        )

    def write_json_yaml(self, filename: str, payload: Mapping[str, Any]) -> Path:
        if not filename.endswith((".yaml", ".yml")):
            raise ValueError("JSON-compatible YAML filename must end in .yaml or .yml.")
        return self.write_json(filename, payload)

    def write_csv(
        self,
        filename: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        fieldnames: Sequence[str],
    ) -> Path:
        filename = self._validate_filename(filename)
        if not fieldnames:
            raise ValueError("CSV fieldnames must not be empty.")
        with tempfile.NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            dir=self.directory,
            delete=False,
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            temp_name = handle.name
        target = self.directory / filename
        os.replace(temp_name, target)
        return target

    def write_environment_lock(self, text: str) -> Path:
        if not text.strip():
            raise ValueError("environment.lock content must not be empty.")
        return self._atomic_text("environment.lock", text.rstrip() + "\n")

    def missing_required_artifacts(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                name
                for name in self.REQUIRED_ARTIFACTS
                if not (self.directory / name).is_file()
            )
        )

    def assert_complete(self) -> None:
        missing = self.missing_required_artifacts()
        if missing:
            raise RuntimeError(
                "Benchmark result directory is incomplete; missing: "
                + ", ".join(missing)
            )
