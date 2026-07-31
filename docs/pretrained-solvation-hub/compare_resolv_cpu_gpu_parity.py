#!/usr/bin/env python3
"""Require bitwise-binary64 identity for full-panel ReSolv CPU/GPU audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import io
import json
import math
import struct
import time
import types
from collections.abc import Mapping, Sequence
from pathlib import Path

EXPECTED_RECORDS = 162
EXPECTED_CLASSIFIED_PRIMARY_GROUPS = 26
EXPECTED_UNCLASSIFIED_RECORDS = 9
EXPECTED_LABELS_INCLUDING_UNCLASSIFIED = 27
UNCLASSIFIED_NON_GROUP_BUCKET = "__unclassified_non_group__"
IDENTITY_FIELDS = ("k_index", "trajectory_id", "mobley_id")
PROVENANCE_IDENTITY_FIELDS = (
    "source_revision",
    "artifact_sha256",
    "manifest_sha256",
    "trajectory_receipts",
    "code_sha256",
    "executing_module_code_sha256",
)

def _code_object_sha256(code: types.CodeType) -> str:
    """Return a deterministic structural digest without marshal alias effects."""

    digest = hashlib.sha256()

    def update(value: object) -> None:
        if isinstance(value, types.CodeType):
            digest.update(b"code:")
            for field_name in (
                "co_argcount",
                "co_posonlyargcount",
                "co_kwonlyargcount",
                "co_nlocals",
                "co_stacksize",
                "co_flags",
                "co_code",
                "co_consts",
                "co_names",
                "co_varnames",
                "co_filename",
                "co_name",
                "co_qualname",
                "co_firstlineno",
                "co_linetable",
                "co_lnotab",
                "co_exceptiontable",
                "co_freevars",
                "co_cellvars",
            ):
                update(getattr(value, field_name, None))
        elif isinstance(value, tuple):
            digest.update(b"tuple:")
            digest.update(str(len(value)).encode("ascii"))
            for item in value:
                update(item)
        elif isinstance(value, frozenset):
            digest.update(b"frozenset:")
            item_digests = sorted(
                _code_constant_sha256(item) for item in value
            )
            for item_digest in item_digests:
                digest.update(item_digest.encode("ascii"))
        elif isinstance(value, bytes):
            digest.update(b"bytes:")
            digest.update(str(len(value)).encode("ascii"))
            digest.update(value)
        elif isinstance(value, str):
            encoded = value.encode("utf-8")
            digest.update(b"str:")
            digest.update(str(len(encoded)).encode("ascii"))
            digest.update(encoded)
        elif value is None or value is Ellipsis or isinstance(
            value, (bool, int, float, complex)
        ):
            digest.update(type(value).__name__.encode("ascii"))
            digest.update(b":")
            digest.update(repr(value).encode("ascii"))
        else:
            raise TypeError(
                f"Unsupported code constant type: {type(value).__name__}"
            )
        digest.update(b";")

    def _code_constant_sha256(value: object) -> str:
        child = hashlib.sha256()
        child.update(repr(value).encode("utf-8"))
        return child.hexdigest()

    update(code)
    return digest.hexdigest()


_MODULE_FRAME = inspect.currentframe()
if _MODULE_FRAME is None:
    raise RuntimeError("Cannot attest the executing comparator module code.")
_EXECUTING_MODULE_CODE = _MODULE_FRAME.f_code
EXECUTING_MODULE_CODE_SHA256 = _code_object_sha256(_EXECUTING_MODULE_CODE)
del _MODULE_FRAME


class ReSolvParityError(RuntimeError):
    """The strict CPU/GPU parity comparison cannot be trusted."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_from_bytes(content: bytes, *, label: str) -> object:
    try:
        return json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReSolvParityError(f"{label} is not valid UTF-8 JSON.") from exc


def _attest_comparator_start_source(
    source_bytes: bytes,
    *,
    filename: str,
) -> dict[str, str]:
    try:
        compiled = compile(source_bytes, filename, "exec")
    except (SyntaxError, ValueError) as exc:
        raise ReSolvParityError(
            "Retained comparator source cannot be compiled."
        ) from exc
    compiled_code_sha256 = _code_object_sha256(compiled)
    if compiled_code_sha256 != EXECUTING_MODULE_CODE_SHA256:
        raise ReSolvParityError(
            "Executing comparator module code does not match retained start "
            "source bytes."
        )
    return {
        "source_sha256": _sha256_bytes(source_bytes),
        "module_code_sha256": compiled_code_sha256,
    }


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ReSolvParityError(f"{label} must be a finite binary64 value.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReSolvParityError(f"{label} must be a finite binary64 value.") from exc
    if not math.isfinite(result):
        raise ReSolvParityError(f"{label} must be finite.")
    return result


def _identity(record: Mapping[str, object]) -> tuple[int, int, str]:
    try:
        raw_k_index = record["k_index"]
        raw_trajectory_id = record["trajectory_id"]
        raw_mobley_id = record["mobley_id"]
        if (
            isinstance(raw_k_index, bool)
            or not isinstance(raw_k_index, (int, str))
            or isinstance(raw_trajectory_id, bool)
            or not isinstance(raw_trajectory_id, (int, str))
            or not isinstance(raw_mobley_id, str)
        ):
            raise TypeError
        result = (
            int(raw_k_index),
            int(raw_trajectory_id),
            raw_mobley_id,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReSolvParityError(
            "Each record requires k_index, trajectory_id, and mobley_id."
        ) from exc
    if not result[2]:
        raise ReSolvParityError("mobley_id must not be empty.")
    return result


def _records_by_identity(
    audit: Mapping[str, object],
    *,
    label: str,
) -> dict[tuple[int, int, str], dict[str, object]]:
    if audit.get("scope") != "full_published_test_split":
        raise ReSolvParityError(f"{label} audit is not a full-panel result.")
    if audit.get("paper_split_reproduction_eligible") is not True:
        raise ReSolvParityError(
            f"{label} audit did not pass its paper-split coverage gates."
        )
    records = audit.get("records")
    if not isinstance(records, list):
        raise ReSolvParityError(f"{label} audit lacks a records list.")
    if len(records) != EXPECTED_RECORDS:
        raise ReSolvParityError(
            f"{label} audit must contain exactly {EXPECTED_RECORDS} records."
        )
    indexed: dict[tuple[int, int, str], dict[str, object]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ReSolvParityError(f"{label} audit contains a non-object record.")
        record = dict(raw)
        identity = _identity(record)
        if identity in indexed:
            raise ReSolvParityError(
                f"{label} audit contains duplicate identity {identity}."
            )
        indexed[identity] = record
    labels = {
        str(record.get("primary_functional_group", "")).strip()
        for record in indexed.values()
    }
    labels.discard("")
    classified_groups = labels - {UNCLASSIFIED_NON_GROUP_BUCKET}
    unclassified_record_count = sum(
        record.get("primary_functional_group") == UNCLASSIFIED_NON_GROUP_BUCKET
        for record in indexed.values()
    )
    if len(classified_groups) != EXPECTED_CLASSIFIED_PRIMARY_GROUPS:
        raise ReSolvParityError(
            f"{label} audit must contain exactly "
            f"{EXPECTED_CLASSIFIED_PRIMARY_GROUPS} classified primary "
            "functional groups."
        )
    if len(labels) != EXPECTED_LABELS_INCLUDING_UNCLASSIFIED:
        raise ReSolvParityError(
            f"{label} audit must contain exactly "
            f"{EXPECTED_LABELS_INCLUDING_UNCLASSIFIED} labels including the "
            "explicit non-group bucket."
        )
    if unclassified_record_count != EXPECTED_UNCLASSIFIED_RECORDS:
        raise ReSolvParityError(
            f"{label} audit must contain exactly "
            f"{EXPECTED_UNCLASSIFIED_RECORDS} unclassified records."
        )
    if (
        audit.get("primary_functional_group_count")
        != EXPECTED_CLASSIFIED_PRIMARY_GROUPS
        or audit.get("classified_primary_functional_group_count")
        != EXPECTED_CLASSIFIED_PRIMARY_GROUPS
        or audit.get("primary_functional_group_label_count_including_unclassified")
        != EXPECTED_LABELS_INCLUDING_UNCLASSIFIED
        or audit.get("unclassified_record_count")
        != EXPECTED_UNCLASSIFIED_RECORDS
    ):
        raise ReSolvParityError(
            f"{label} audit primary-functional-group receipt is inconsistent."
        )
    metrics = audit.get("metrics")
    if (
        not isinstance(metrics, Mapping)
        or not isinstance(metrics.get("pooled"), Mapping)
        or metrics["pooled"].get("n") != EXPECTED_RECORDS
        or not isinstance(metrics.get("by_primary_functional_group"), Mapping)
        or len(metrics["by_primary_functional_group"])
        != EXPECTED_LABELS_INCLUDING_UNCLASSIFIED
        or UNCLASSIFIED_NON_GROUP_BUCKET
        not in metrics["by_primary_functional_group"]
    ):
        raise ReSolvParityError(f"{label} audit metric coverage is inconsistent.")
    if any(record.get("split", "test") != "test" for record in indexed.values()):
        raise ReSolvParityError(f"{label} audit contains a non-test record.")
    return indexed


def _require_runtime(audit: Mapping[str, object], *, platform: str) -> None:
    runtime = audit.get("runtime")
    worker = audit.get("worker")
    if not isinstance(runtime, Mapping) or not isinstance(worker, Mapping):
        raise ReSolvParityError(f"{platform} audit lacks runtime/device receipts.")
    if runtime.get("requested_platform") != platform:
        raise ReSolvParityError(
            f"{platform} audit requested-platform receipt does not match."
        )
    if runtime.get("x64_enabled") is not True:
        raise ReSolvParityError(f"{platform} audit did not use JAX x64.")
    expected_backends = {"cpu"} if platform == "cpu" else {"gpu", "cuda", "rocm"}
    devices = runtime.get("devices")
    if not isinstance(devices, list) or not devices:
        raise ReSolvParityError(f"{platform} audit exposed no JAX devices.")
    observed_platforms = {
        str(device.get("platform", "")).lower()
        for device in devices
        if isinstance(device, Mapping)
    }
    if not observed_platforms or not observed_platforms.issubset(expected_backends):
        raise ReSolvParityError(
            f"{platform} audit device platforms do not match."
        )
    if worker.get("backend") not in expected_backends:
        raise ReSolvParityError(
            f"{platform} audit worker backend does not match its platform."
        )
    dtype = worker.get("dtype")
    if not isinstance(dtype, Mapping) or dtype.get("jax_enable_x64") is not True:
        raise ReSolvParityError(f"{platform} audit worker did not preserve JAX x64.")


def _require_repeat_stability(
    audit: Mapping[str, object],
    *,
    platform: str,
) -> int:
    repeat_determinism = audit.get("repeat_determinism")
    if not isinstance(repeat_determinism, Mapping):
        raise ReSolvParityError(
            f"{platform} canonical audit lacks repeat_determinism evidence."
        )
    repeat_count = repeat_determinism.get("repeat_count")
    if (
        isinstance(repeat_count, bool)
        or not isinstance(repeat_count, int)
        or repeat_count < 3
    ):
        raise ReSolvParityError(
            f"{platform} canonical audit requires repeat_count >= 3."
        )
    if repeat_determinism.get("identity_keyed_bitwise_stable") is not True:
        raise ReSolvParityError(
            f"{platform} canonical audit did not attest identity-keyed "
            "bitwise repeat stability."
        )
    return repeat_count


def _binary64_hex(value: float) -> str:
    return struct.pack(">d", value).hex()


def _ordered_bits(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return (~bits & 0xFFFFFFFFFFFFFFFF) if bits & (1 << 63) else bits | (1 << 63)


def _ulp_delta(left: float, right: float) -> int:
    return abs(_ordered_bits(left) - _ordered_bits(right))


def _metrics(records: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    errors = [
        abs(
            _finite_float(record["prediction_kcal_mol"], "prediction")
            - _finite_float(record["experimental_kcal_mol"], "experiment")
        )
        for record in records
    ]
    return {
        "n": len(errors),
        "mae_kcal_mol": sum(errors) / len(errors),
        "rmse_kcal_mol": math.sqrt(
            sum(error * error for error in errors) / len(errors)
        ),
        "maxae_kcal_mol": max(errors),
    }


def _metric_comparison(
    cpu_records: Sequence[Mapping[str, object]],
    gpu_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    cpu = _metrics(cpu_records)
    gpu = _metrics(gpu_records)
    return {
        "cpu": cpu,
        "gpu": gpu,
        "gpu_minus_cpu": {
            key: float(gpu[key]) - float(cpu[key])
            for key in ("mae_kcal_mol", "rmse_kcal_mol", "maxae_kcal_mol")
        },
    }


def compare_audits(
    cpu_audit: Mapping[str, object],
    gpu_audit: Mapping[str, object],
    *,
    diagnostic_abs_tolerance: float = 1.0e-12,
) -> dict[str, object]:
    """Compare complete audits; any one-ULP prediction change fails admission."""

    tolerance = _finite_float(
        diagnostic_abs_tolerance,
        "diagnostic_abs_tolerance",
    )
    if tolerance < 0.0:
        raise ReSolvParityError("diagnostic_abs_tolerance must be non-negative.")
    _require_repeat_stability(cpu_audit, platform="CPU")
    _require_repeat_stability(gpu_audit, platform="GPU")
    _require_runtime(cpu_audit, platform="cpu")
    _require_runtime(gpu_audit, platform="gpu")

    cpu_provenance = cpu_audit.get("provenance")
    gpu_provenance = gpu_audit.get("provenance")
    if not isinstance(cpu_provenance, Mapping) or not isinstance(
        gpu_provenance, Mapping
    ):
        raise ReSolvParityError("Both audits require provenance receipts.")
    for field in PROVENANCE_IDENTITY_FIELDS:
        if cpu_provenance.get(field) != gpu_provenance.get(field):
            raise ReSolvParityError(
                f"CPU/GPU provenance identity mismatch: {field}."
            )

    cpu_by_id = _records_by_identity(cpu_audit, label="CPU")
    gpu_by_id = _records_by_identity(gpu_audit, label="GPU")
    if set(cpu_by_id) != set(gpu_by_id):
        missing = sorted(set(cpu_by_id) - set(gpu_by_id))
        extra = sorted(set(gpu_by_id) - set(cpu_by_id))
        raise ReSolvParityError(
            "CPU/GPU record identity sets differ: "
            f"missing={missing[:3]}, extra={extra[:3]}."
        )

    rows: list[dict[str, object]] = []
    strict = True
    for identity in sorted(cpu_by_id):
        cpu_record = cpu_by_id[identity]
        gpu_record = gpu_by_id[identity]
        for field in (
            "experimental_kcal_mol",
            "experimental_uncertainty_kcal_mol",
            "primary_functional_group",
            "smiles",
            "split",
        ):
            if cpu_record.get(field) != gpu_record.get(field):
                raise ReSolvParityError(
                    f"CPU/GPU record metadata mismatch for {identity}: {field}."
                )
        cpu_prediction = _finite_float(
            cpu_record.get("prediction_kcal_mol"),
            "CPU prediction",
        )
        gpu_prediction = _finite_float(
            gpu_record.get("prediction_kcal_mol"),
            "GPU prediction",
        )
        experiment = _finite_float(
            cpu_record.get("experimental_kcal_mol"),
            "experiment",
        )
        exact = _binary64_hex(cpu_prediction) == _binary64_hex(gpu_prediction)
        strict = strict and exact
        absolute_delta = abs(gpu_prediction - cpu_prediction)
        cpu_error = abs(cpu_prediction - experiment)
        gpu_error = abs(gpu_prediction - experiment)
        rows.append(
            {
                "k_index": identity[0],
                "trajectory_id": identity[1],
                "mobley_id": identity[2],
                "primary_functional_group": cpu_record[
                    "primary_functional_group"
                ],
                "cpu_prediction_kcal_mol": cpu_prediction,
                "gpu_prediction_kcal_mol": gpu_prediction,
                "cpu_prediction_ieee754_binary64_hex": _binary64_hex(
                    cpu_prediction
                ),
                "gpu_prediction_ieee754_binary64_hex": _binary64_hex(
                    gpu_prediction
                ),
                "bitwise_binary64_equal": exact,
                "absolute_delta_kcal_mol": absolute_delta,
                "ulp_delta": _ulp_delta(cpu_prediction, gpu_prediction),
                "diagnostic_within_abs_tolerance": (
                    absolute_delta <= tolerance
                ),
                "cpu_absolute_experimental_error_kcal_mol": cpu_error,
                "gpu_absolute_experimental_error_kcal_mol": gpu_error,
                "exact_experimental_error_worsening_kcal_mol": (
                    gpu_error - cpu_error
                ),
            }
        )

    label_names = sorted(
        {
            str(record["primary_functional_group"])
            for record in cpu_by_id.values()
        }
    )
    classified_group_names = [
        label
        for label in label_names
        if label != UNCLASSIFIED_NON_GROUP_BUCKET
    ]
    cpu_records = [cpu_by_id[key] for key in sorted(cpu_by_id)]
    gpu_records = [gpu_by_id[key] for key in sorted(gpu_by_id)]
    by_group = {}
    for group in classified_group_names:
        cpu_group = [
            record
            for record in cpu_records
            if record["primary_functional_group"] == group
        ]
        gpu_group = [
            record
            for record in gpu_records
            if record["primary_functional_group"] == group
        ]
        by_group[group] = _metric_comparison(cpu_group, gpu_group)
    cpu_unclassified = [
        record
        for record in cpu_records
        if record["primary_functional_group"] == UNCLASSIFIED_NON_GROUP_BUCKET
    ]
    gpu_unclassified = [
        record
        for record in gpu_records
        if record["primary_functional_group"] == UNCLASSIFIED_NON_GROUP_BUCKET
    ]

    return {
        "schema_version": 1,
        "candidate": "ReSolv",
        "gate": "strict_zero_loss_cpu_gpu_binary64",
        "strict_zero_loss": strict,
        "strict_failure_count": sum(
            not bool(row["bitwise_binary64_equal"]) for row in rows
        ),
        "diagnostic_abs_tolerance_kcal_mol": tolerance,
        "diagnostic_within_tolerance_count": sum(
            bool(row["diagnostic_within_abs_tolerance"]) for row in rows
        ),
        "coverage": {
            "record_count": len(rows),
            "classified_primary_functional_group_count": len(
                classified_group_names
            ),
            "unclassified_record_count": len(cpu_unclassified),
            "label_count_including_unclassified": len(label_names),
            "full_162_record_26_group_9_unclassified_panel": True,
        },
        "identity": {
            field: cpu_provenance[field]
            for field in PROVENANCE_IDENTITY_FIELDS
        },
        "runtime_device_receipts": {
            "cpu": {
                "runtime": cpu_audit["runtime"],
                "worker": cpu_audit["worker"],
            },
            "gpu": {
                "runtime": gpu_audit["runtime"],
                "worker": gpu_audit["worker"],
            },
        },
        "evidence_file_receipts": {
            "cpu": cpu_audit.get("evidence_file_sha256"),
            "gpu": gpu_audit.get("evidence_file_sha256"),
        },
        "metrics_parity": {
            "pooled": _metric_comparison(cpu_records, gpu_records),
            "by_classified_primary_functional_group": by_group,
            "unclassified_non_group_bucket": _metric_comparison(
                cpu_unclassified,
                gpu_unclassified,
            ),
        },
        "records": rows,
    }


def load_audit_directory(path: Path, *, platform: str) -> dict[str, object]:
    path = Path(path)
    evidence_paths = {
        "audit": path / "audit.json",
        "records": path / "records.csv",
        "manifest": path / "manifest.json",
        "receipt": path / "receipt.json",
        "runtime_diagnostic": path / "runtime-diagnostic.json",
    }
    for required in evidence_paths.values():
        if not required.is_file():
            raise ReSolvParityError(f"Missing {platform} audit artifact: {required}")
    # Each evidence file crosses the filesystem boundary exactly once.  Hashing
    # and parsing below use these retained bytes, never a reopened path.
    evidence = {
        label: evidence_path.read_bytes()
        for label, evidence_path in evidence_paths.items()
    }
    audit_raw = _json_from_bytes(evidence["audit"], label=f"{platform} audit.json")
    manifest_raw = _json_from_bytes(
        evidence["manifest"],
        label=f"{platform} manifest.json",
    )
    receipt_raw = _json_from_bytes(
        evidence["receipt"],
        label=f"{platform} receipt.json",
    )
    runtime_diagnostic_raw = _json_from_bytes(
        evidence["runtime_diagnostic"],
        label=f"{platform} runtime-diagnostic.json",
    )
    if not isinstance(audit_raw, Mapping) or not isinstance(receipt_raw, Mapping):
        raise ReSolvParityError(f"{platform} audit or receipt is not an object.")
    if not isinstance(runtime_diagnostic_raw, Mapping):
        raise ReSolvParityError(
            f"{platform} runtime diagnostic is not an object."
        )
    if not isinstance(manifest_raw, list):
        raise ReSolvParityError(f"{platform} manifest is not a record list.")
    audit = dict(audit_raw)
    receipt = dict(receipt_raw)
    if receipt.get("audit_json_sha256") != _sha256_bytes(evidence["audit"]):
        raise ReSolvParityError(f"{platform} audit.json receipt mismatch.")
    if receipt.get("records_csv_sha256") != _sha256_bytes(evidence["records"]):
        raise ReSolvParityError(f"{platform} records.csv receipt mismatch.")
    if receipt.get("manifest_json_sha256") != _sha256_bytes(evidence["manifest"]):
        raise ReSolvParityError(f"{platform} manifest.json receipt mismatch.")
    if receipt.get("runtime_diagnostic_sha256") != _sha256_bytes(
        evidence["runtime_diagnostic"]
    ):
        raise ReSolvParityError(
            f"{platform} runtime-diagnostic.json receipt mismatch."
        )
    provenance = audit.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("manifest_sha256")
        != _sha256_bytes(evidence["manifest"])
    ):
        raise ReSolvParityError(f"{platform} manifest provenance mismatch.")

    audit_records = _records_by_identity(audit, label=platform.upper())
    try:
        csv_text = evidence["records"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReSolvParityError(
            f"{platform} records.csv is not valid UTF-8."
        ) from exc
    csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
    csv_by_identity: dict[tuple[int, int, str], dict[str, str]] = {}
    for row in csv_rows:
        identity = _identity(row)
        if identity in csv_by_identity:
            raise ReSolvParityError(
                f"{platform} records.csv contains duplicate identity {identity}."
            )
        csv_by_identity[identity] = row
    if set(csv_by_identity) != set(audit_records):
        raise ReSolvParityError(
            f"{platform} records.csv identities differ from audit.json."
        )
    for identity, audit_record in audit_records.items():
        csv_prediction = _finite_float(
            csv_by_identity[identity].get("prediction_kcal_mol"),
            f"{platform} CSV prediction",
        )
        audit_prediction = _finite_float(
            audit_record.get("prediction_kcal_mol"),
            f"{platform} audit prediction",
        )
        if _binary64_hex(csv_prediction) != _binary64_hex(audit_prediction):
            raise ReSolvParityError(
                f"{platform} records.csv prediction differs for {identity}."
            )

    manifest_by_identity: dict[tuple[int, int, str], Mapping[str, object]] = {}
    for raw in manifest_raw:
        if not isinstance(raw, Mapping):
            raise ReSolvParityError(
                f"{platform} manifest contains a non-object record."
            )
        identity = _identity(raw)
        if identity in manifest_by_identity:
            raise ReSolvParityError(
                f"{platform} manifest contains duplicate identity {identity}."
            )
        manifest_by_identity[identity] = raw
    if set(manifest_by_identity) != set(audit_records):
        raise ReSolvParityError(
            f"{platform} manifest identities differ from audit.json."
        )
    audit_repeat_count = _require_repeat_stability(audit, platform=platform)
    runtime_repeat_count = runtime_diagnostic_raw.get("repeat_count")
    if (
        runtime_repeat_count != audit_repeat_count
        or runtime_diagnostic_raw.get("scope")
        != "three_or_more_cold_worker_replays_not_full_task_timing"
    ):
        raise ReSolvParityError(
            f"{platform} runtime diagnostic repeat evidence is inconsistent."
        )
    audit["evidence_file_sha256"] = {
        "audit_json": _sha256_bytes(evidence["audit"]),
        "records_csv": _sha256_bytes(evidence["records"]),
        "manifest_json": _sha256_bytes(evidence["manifest"]),
        "receipt_json": _sha256_bytes(evidence["receipt"]),
        "runtime_diagnostic_json": _sha256_bytes(
            evidence["runtime_diagnostic"]
        ),
    }
    return audit


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-audit-dir", type=Path, required=True)
    parser.add_argument("--gpu-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-abs-tolerance", type=float, default=1.0e-12)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    comparator_path = Path(__file__).resolve()
    comparator_start_bytes = comparator_path.read_bytes()
    comparator_attestation = _attest_comparator_start_source(
        comparator_start_bytes,
        filename=str(comparator_path),
    )
    args = _parse_args(argv)
    started = time.perf_counter()
    cpu = load_audit_directory(args.cpu_audit_dir, platform="cpu")
    gpu = load_audit_directory(args.gpu_audit_dir, platform="gpu")
    report = compare_audits(
        cpu,
        gpu,
        diagnostic_abs_tolerance=args.diagnostic_abs_tolerance,
    )
    comparator_end_bytes = comparator_path.read_bytes()
    if _sha256_bytes(comparator_end_bytes) != comparator_attestation[
        "source_sha256"
    ]:
        raise ReSolvParityError(
            "Comparator source changed while parity comparison was running."
        )
    report["comparator_sha256"] = comparator_attestation["source_sha256"]
    report["comparator_module_code_sha256"] = comparator_attestation[
        "module_code_sha256"
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "cpu-gpu-parity.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    runtime_path = args.output_dir / "cpu-gpu-parity-runtime.json"
    runtime_path.write_text(
        json.dumps(
            {"comparison_wall_seconds": time.perf_counter() - started},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    receipt = {
        "cpu_gpu_parity_json_sha256": _sha256(report_path),
        "cpu_gpu_parity_runtime_json_sha256": _sha256(runtime_path),
    }
    receipt_path = args.output_dir / "cpu-gpu-parity-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({**receipt, "strict_zero_loss": report["strict_zero_loss"]}))
    return 0 if report["strict_zero_loss"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
