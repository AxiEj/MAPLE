#!/usr/bin/env python3
"""Recompute the full published ReSolv test split from pinned endpoint data.

This runner never generates a conformer or propagates MD.  It verifies the
official source, two checkpoints, database, and every selected trajectory
before delegating deserialization and JAX execution to the isolated ReSolv
Python environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import types
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maple.function.solvfe.resolv_protocol import (
    RESOLV_DATABASE_SHA256,
    RESOLV_FULL_TEST_MANIFEST_SHA256,
    RESOLV_IMPORTED_MODULE_CODE_SHA256,
    RESOLV_UPSTREAM_REVISION,
    RESOLV_VACUUM_MODEL_SHA256,
    RESOLV_WATER_MODEL_SHA256,
    ReSolvArtifactSet,
    ReSolvProtocolError,
    build_resolv_manifest,
    sha256_file,
)

PROTOCOL_SEMANTICS: dict[str, object] = {
    "protocol_kind": "dedicated_hydration_free_energy",
    "estimator": "BAR",
    "ordinary_calculator": False,
    "additive_continuum": False,
    "fresh_conformer_generation": False,
    "fresh_md_propagation": False,
    "solvent": "water",
    "temperature_kelvin": 298.15,
    "evaluation_split": "published_FreeSolv_derived_test_split",
}
KBT_KCAL_MOL = 298.15 * 0.00198720426
BAR_ENDPOINT_SAMPLES = 40
BAR_NUMERICAL_EPSILON = sys.float_info.epsilon
UNCLASSIFIED_NON_GROUP_BUCKET = "__unclassified_non_group__"

RUNNER_PATH = Path(__file__).resolve()
WORKER_PATH = (
    REPOSITORY_ROOT / "maple/function/solvfe/_resolv_worker.py"
).resolve()
PROTOCOL_PATH = (
    REPOSITORY_ROOT / "maple/function/solvfe/resolv_protocol.py"
).resolve()


def _code_object_sha256(code: types.CodeType) -> str:
    """Return a deterministic structural digest of executable module code."""

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
            for item_digest in sorted(
                hashlib.sha256(repr(item).encode("utf-8")).hexdigest()
                for item in value
            ):
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

    update(code)
    return digest.hexdigest()


IMPORTED_RUNNER_CODE_SHA256 = _code_object_sha256(sys._getframe().f_code)


class ReSolvAuditError(RuntimeError):
    """The ReSolv audit cannot continue without violating its evidence gates."""


def _code_sha256() -> dict[str, str]:
    return {
        "audit_runner": sha256_file(RUNNER_PATH),
        "worker": sha256_file(WORKER_PATH),
        "protocol": sha256_file(PROTOCOL_PATH),
    }


def _read_code_snapshot() -> tuple[dict[str, str], bytes]:
    """Read the admitted local code once and retain the exact worker bytes."""

    sources = {
        "audit_runner": RUNNER_PATH.read_bytes(),
        "worker": WORKER_PATH.read_bytes(),
        "protocol": PROTOCOL_PATH.read_bytes(),
    }
    compiled_runner_sha256 = _code_object_sha256(
        compile(
            sources["audit_runner"],
            str(RUNNER_PATH),
            "exec",
            dont_inherit=True,
        )
    )
    if compiled_runner_sha256 != IMPORTED_RUNNER_CODE_SHA256:
        raise ReSolvAuditError(
            "ReSolv runner source changed after Python loaded its code object."
        )
    compiled_protocol_sha256 = _code_object_sha256(
        compile(
            sources["protocol"],
            str(PROTOCOL_PATH),
            "exec",
            dont_inherit=True,
        )
    )
    if compiled_protocol_sha256 != RESOLV_IMPORTED_MODULE_CODE_SHA256:
        raise ReSolvAuditError(
            "ReSolv protocol source changed after Python imported its code object."
        )
    return (
        {
            label: hashlib.sha256(content).hexdigest()
            for label, content in sources.items()
        },
        sources["worker"],
    )


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ReSolvAuditError(f"{label} must be a finite number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ReSolvAuditError(f"{label} must be a finite number.") from exc
    if not math.isfinite(parsed):
        raise ReSolvAuditError(f"{label} must be finite, observed {value!r}.")
    return parsed


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ReSolvAuditError(f"{label} must be an integer.")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ReSolvAuditError(f"{label} must be an integer.") from exc
    if isinstance(value, str) and str(parsed) != value:
        raise ReSolvAuditError(f"{label} must use canonical integer syntax.")
    return parsed


def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _bar_work_array(value: object, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != BAR_ENDPOINT_SAMPLES:
        raise ReSolvAuditError(
            f"{label} must contain exactly {BAR_ENDPOINT_SAMPLES} values."
        )
    return [
        _finite_float(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    ]


def diagnose_equal_endpoint_bar(
    bar_diagnostics: object,
    *,
    upstream_delta_g_kcal_mol: object,
) -> dict[str, object]:
    """Independently solve and diagnose equal-40-frame BAR in binary64."""

    if not isinstance(bar_diagnostics, Mapping):
        raise ReSolvAuditError("BAR diagnostics must be a mapping.")
    work = bar_diagnostics.get("dimensionless_work")
    if not isinstance(work, Mapping):
        raise ReSolvAuditError("BAR diagnostics lack dimensionless work arrays.")
    a = _bar_work_array(work.get("vacuum_ensemble"), "vacuum BAR work")
    b = _bar_work_array(work.get("water_ensemble"), "water BAR work")

    def residual(f_value: float) -> float:
        c0 = [_stable_sigmoid(f_value - value) for value in a]
        c1 = [_stable_sigmoid(value - f_value) for value in b]
        return sum(c0) / len(c0) - sum(c1) / len(c1)

    lower = min((*a, *b))
    upper = max((*a, *b))
    lower_residual = residual(lower)
    upper_residual = residual(upper)
    bracket_tolerance = 64.0 * BAR_NUMERICAL_EPSILON
    if lower_residual > bracket_tolerance or upper_residual < -bracket_tolerance:
        raise ReSolvAuditError(
            "Equal-endpoint BAR residual is not bracketed within 64 eps."
        )

    if lower == upper:
        root = lower
    else:
        root = (lower + upper) / 2.0
        for _ in range(200):
            root_residual = residual(root)
            if root_residual == 0.0:
                lower = root
                upper = root
                break
            relative_width = (upper - lower) / max(1.0, abs(root))
            if relative_width <= 1.0e-12 and abs(root_residual) <= 1.0e-12:
                break
            if root_residual < 0.0:
                lower = root
            else:
                upper = root
            next_root = (lower + upper) / 2.0
            if next_root == root:
                break
            root = next_root
        else:
            raise ReSolvAuditError("Equal-endpoint BAR bisection did not converge.")

    root_residual = residual(root)
    if abs(root_residual) > 1.0e-12:
        raise ReSolvAuditError(
            "Equal-endpoint BAR root residual exceeds 1e-12."
        )
    reconstructed_delta_g = KBT_KCAL_MOL * root
    upstream_delta_g = _finite_float(
        upstream_delta_g_kcal_mol,
        "upstream_delta_g_kcal_mol",
    )
    delta_g_mismatch = abs(upstream_delta_g - reconstructed_delta_g)
    if delta_g_mismatch > 1.0e-10:
        raise ReSolvAuditError(
            "Upstream BAR result disagrees with the independent equal-endpoint "
            "root by more than 1e-10 kcal/mol."
        )

    c0 = [_stable_sigmoid(root - value) for value in a]
    c1 = [_stable_sigmoid(value - root) for value in b]
    mean_c0 = sum(c0) / len(c0)
    mean_c1 = sum(c1) / len(c1)
    overlap = mean_c0 + mean_c1

    def kish_ess(values: Sequence[float]) -> float:
        total = sum(values)
        squared = sum(value * value for value in values)
        if squared == 0.0:
            return 0.0
        return total * total / squared

    vacuum_ess = kish_ess(c0)
    water_ess = kish_ess(c1)
    information = (
        len(c0) * sum(value * (1.0 - value) for value in c0) / len(c0)
        + len(c1) * sum(value * (1.0 - value) for value in c1) / len(c1)
    )
    variance: float | None
    conditional_se: float | None
    if information == 0.0:
        variance = None
        conditional_se = None
    else:
        variance = 1.0 / information - 1.0 / len(c0) - 1.0 / len(c1)
        if variance < -64.0 * BAR_NUMERICAL_EPSILON:
            raise ReSolvAuditError(
                "Conditional BAR IID variance is negative beyond 64 eps."
            )
        variance = max(0.0, variance)
        conditional_se = KBT_KCAL_MOL * math.sqrt(variance)

    overlap_support_pass = (
        overlap >= 0.25 and min(vacuum_ess, water_ess) >= 10.0
    )
    statistical_support_eligible = (
        overlap_support_pass and conditional_se is not None
    )
    return {
        "bar_root_dimensionless": root,
        "bar_root_residual": root_residual,
        "bar_reconstructed_delta_g_kcal_mol": reconstructed_delta_g,
        "bar_upstream_root_abs_mismatch_kcal_mol": delta_g_mismatch,
        "bar_overlap_omega": overlap,
        "bar_vacuum_direction_kish_ess": vacuum_ess,
        "bar_water_direction_kish_ess": water_ess,
        "bar_min_directional_kish_ess": min(vacuum_ess, water_ess),
        "bar_effective_overlap_mass_total": BAR_ENDPOINT_SAMPLES * overlap,
        "bar_conditional_iid_information": information,
        "bar_conditional_iid_variance_dimensionless": variance,
        "bar_conditional_iid_se_kcal_mol": conditional_se,
        "bar_overlap_support_pass": overlap_support_pass,
        "statistical_support_eligible": statistical_support_eligible,
        "bar_uncertainty_inferential": False,
        "bar_confidence_interval_available": False,
        "bar_bootstrap_performed": False,
        "bar_uncertainty_limitation": (
            "Conditional asymptotic IID diagnostic only; autocorrelation and "
            "independent replica information are unavailable, so no inferential "
            "uncertainty or confidence interval is reported."
        ),
    }


def _error_metrics(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    absolute_errors = [
        _finite_float(record["absolute_error_kcal_mol"], "absolute error")
        for record in records
    ]
    if not absolute_errors:
        raise ReSolvAuditError("Cannot compute metrics for an empty record set.")
    squared_errors = [value * value for value in absolute_errors]
    return {
        "n": len(absolute_errors),
        "mae_kcal_mol": sum(absolute_errors) / len(absolute_errors),
        "rmse_kcal_mol": math.sqrt(sum(squared_errors) / len(squared_errors)),
        "maxae_kcal_mol": max(absolute_errors),
    }


def aggregate_results(
    records: Sequence[Mapping[str, object]],
    *,
    minimum_primary_groups: int = 10,
    require_bar_diagnostics: bool = False,
) -> dict[str, object]:
    """Recompute record, pooled, and primary-functional-group errors."""

    if minimum_primary_groups < 1:
        raise ReSolvAuditError("minimum_primary_groups must be positive.")
    normalized: list[dict[str, object]] = []
    for raw in records:
        record = dict(raw)
        prediction = _finite_float(
            record.get("prediction_kcal_mol"),
            "prediction_kcal_mol",
        )
        experiment = _finite_float(
            record.get("experimental_kcal_mol"),
            "experimental_kcal_mol",
        )
        signed_error = prediction - experiment
        record["prediction_kcal_mol"] = prediction
        record["experimental_kcal_mol"] = experiment
        record["signed_error_kcal_mol"] = signed_error
        record["absolute_error_kcal_mol"] = abs(signed_error)

        groups = record.get("groups", [])
        if not isinstance(groups, (list, tuple)):
            raise ReSolvAuditError("Each record's groups field must be a sequence.")
        cleaned_groups = [str(value).strip() for value in groups if str(value).strip()]
        record["groups"] = cleaned_groups
        primary = str(
            record.get("primary_functional_group")
            or (cleaned_groups[0] if cleaned_groups else "unclassified")
        ).strip()
        record["primary_functional_group"] = (
            UNCLASSIFIED_NON_GROUP_BUCKET
            if not primary or primary.lower() == "unclassified"
            else primary
        )
        normalized.append(record)

    if not normalized:
        raise ReSolvAuditError("The ReSolv audit returned no records.")

    grouped: dict[str, list[dict[str, object]]] = {}
    for record in normalized:
        group = str(record["primary_functional_group"])
        grouped.setdefault(group, []).append(record)
    by_group = {
        group: _error_metrics(group_records)
        for group, group_records in sorted(grouped.items())
    }
    classified_groups = {
        group
        for group in grouped
        if group != UNCLASSIFIED_NON_GROUP_BUCKET
    }
    unclassified_record_count = len(
        grouped.get(UNCLASSIFIED_NON_GROUP_BUCKET, [])
    )

    failures: list[str] = []
    if len(classified_groups) < minimum_primary_groups:
        failures.append("classified_primary_functional_groups_below_10")
    if len(normalized) != 162:
        failures.append("published_test_panel_incomplete")
    if any(record.get("split", "test") != "test" for record in normalized):
        failures.append("non_test_record_present")
    # This is a reproduction of the paper's FreeSolv-derived split, not an
    # external blind validation.  Keep that scientific distinction explicit.
    failures.append("not_independent_external_validation")

    statistical_support: dict[str, object] = {
        "status": "not_evaluated",
        "statistical_support_eligible": False,
    }
    if require_bar_diagnostics:
        required_fields = (
            "bar_overlap_omega",
            "bar_min_directional_kish_ess",
            "bar_conditional_iid_se_kcal_mol",
            "statistical_support_eligible",
        )
        if any(
            any(field not in record for field in required_fields)
            for record in normalized
        ):
            raise ReSolvAuditError(
                "A record lacks required scalar BAR diagnostics."
            )
        overlaps = [
            _finite_float(record["bar_overlap_omega"], "BAR overlap")
            for record in normalized
        ]
        minimum_ess = [
            _finite_float(
                record["bar_min_directional_kish_ess"],
                "BAR minimum directional Kish ESS",
            )
            for record in normalized
        ]
        finite_se = [
            _finite_float(value, "conditional BAR IID SE")
            for record in normalized
            if (value := record["bar_conditional_iid_se_kcal_mol"]) is not None
        ]
        support_failure_count = sum(
            record["statistical_support_eligible"] is not True
            for record in normalized
        )
        statistical_support = {
            "status": "conditional_diagnostic_only_non_inferential",
            "statistical_support_eligible": support_failure_count == 0,
            "support_failure_count": support_failure_count,
            "minimum_overlap_omega": min(overlaps),
            "median_overlap_omega": statistics.median(overlaps),
            "minimum_directional_kish_ess": min(minimum_ess),
            "maximum_conditional_iid_se_kcal_mol": (
                max(finite_se) if finite_se else None
            ),
            "conditional_iid_se_unavailable_count": (
                len(normalized) - len(finite_se)
            ),
            "inferential_uncertainty": False,
            "confidence_interval_available": False,
            "bootstrap_performed": False,
            "limitation": (
                "Autocorrelation and independent replica information are "
                "unavailable; the IID quantity is diagnostic only."
            ),
        }
        if support_failure_count:
            failures.append("bar_statistical_support_failed")

    return {
        "protocol_semantics": dict(PROTOCOL_SEMANTICS),
        "records": normalized,
        "metrics": {
            "pooled": _error_metrics(normalized),
            "by_primary_functional_group": by_group,
        },
        "primary_functional_group_count": len(classified_groups),
        "classified_primary_functional_group_count": len(classified_groups),
        "primary_functional_group_label_count_including_unclassified": len(grouped),
        "unclassified_record_count": unclassified_record_count,
        "minimum_classified_primary_functional_groups_required": minimum_primary_groups,
        "statistical_support": statistical_support,
        "statistical_support_eligible": statistical_support[
            "statistical_support_eligible"
        ],
        "scientific_accuracy_eligible": not failures,
        "paper_split_reproduction_eligible": not any(
            failure
            in {
                "classified_primary_functional_groups_below_10",
                "published_test_panel_incomplete",
                "non_test_record_present",
                "bar_statistical_support_failed",
            }
            for failure in failures
        ),
        "eligibility_failures": failures,
    }


def detect_jax_runtime(
    *,
    device_records: Sequence[Mapping[str, object]],
    x64_enabled: bool,
) -> dict[str, object]:
    """Normalize a JAX device probe without pretending CPU JAX is GPU JAX."""

    devices = [dict(record) for record in device_records]
    gpu_devices = [
        record
        for record in devices
        if str(record.get("platform", "")).lower() in {"gpu", "cuda", "rocm"}
        or "gpu" in str(record.get("device_kind", "")).lower()
    ]
    if gpu_devices:
        status = "available"
        reason = None
    else:
        status = "unavailable"
        reason = (
            "The selected ReSolv Python/JAX runtime exposes no CUDA/ROCm device; "
            "no GPU accuracy or timing result was fabricated."
        )
    return {
        "devices": devices,
        "x64_enabled": bool(x64_enabled),
        "gpu_status": status,
        "gpu_results_present": bool(gpu_devices),
        "gpu_unavailable_reason": reason,
    }


def _validate_requested_runtime(
    *,
    requested_platform: str,
    runtime_probe: Mapping[str, object],
    worker_response: Mapping[str, object] | None = None,
) -> None:
    """Fail closed when the requested backend or binary64 mode was not used."""

    if requested_platform not in {"cpu", "gpu"}:
        raise ReSolvAuditError("requested_platform must be 'cpu' or 'gpu'.")
    if runtime_probe.get("x64_enabled") is not True:
        raise ReSolvAuditError("Requested ReSolv runtime did not enable JAX x64.")
    devices = runtime_probe.get("devices")
    if not isinstance(devices, list) or not devices:
        raise ReSolvAuditError("Requested ReSolv runtime exposed no JAX devices.")
    observed_platforms = {
        str(device.get("platform", "")).lower()
        for device in devices
        if isinstance(device, Mapping)
    }
    allowed = {"cpu"} if requested_platform == "cpu" else {"gpu", "cuda", "rocm"}
    if not observed_platforms or not observed_platforms.issubset(allowed):
        raise ReSolvAuditError(
            "Requested ReSolv platform does not match probed JAX devices: "
            f"requested {requested_platform}, observed {sorted(observed_platforms)}."
        )
    if worker_response is None:
        return
    if worker_response.get("backend") not in allowed:
        raise ReSolvAuditError(
            "Requested ReSolv platform does not match worker backend: "
            f"requested {requested_platform}, observed "
            f"{worker_response.get('backend')!r}."
        )
    dtype = worker_response.get("dtype")
    if not isinstance(dtype, Mapping) or dtype.get("jax_enable_x64") is not True:
        raise ReSolvAuditError("ReSolv worker did not preserve JAX x64 execution.")


def validate_artifacts(
    artifacts: ReSolvArtifactSet,
    *,
    required_k_indices: Iterable[int],
) -> dict[str, object]:
    """Translate the protocol validator into the runner's error type."""

    try:
        return artifacts.validate(required_k_indices=tuple(required_k_indices))
    except ReSolvProtocolError as exc:
        raise ReSolvAuditError(str(exc)) from exc


def _tracked_blob_receipt(root: Path, path: Path) -> dict[str, str]:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ReSolvAuditError(
            f"Trajectory lies outside the pinned ReSolv checkout: {path}"
        ) from exc

    tree = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "HEAD", "--", relative],
        text=True,
        capture_output=True,
        check=False,
    )
    fields = tree.stdout.strip().split()
    if tree.returncode != 0 or len(fields) < 4 or fields[1] != "blob":
        raise ReSolvAuditError(
            f"Trajectory is not a tracked blob at the pinned ReSolv revision: {relative}"
        )
    expected_blob = fields[2]
    observed_blob_run = subprocess.run(
        ["git", "-C", str(root), "hash-object", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    observed_blob = observed_blob_run.stdout.strip()
    if observed_blob_run.returncode != 0 or observed_blob != expected_blob:
        raise ReSolvAuditError(
            "Trajectory differs from the pinned Git blob: "
            f"{relative}; expected {expected_blob}, observed {observed_blob or 'unknown'}."
        )
    return {
        "relative_path": relative,
        "git_blob": expected_blob,
        "sha256": sha256_file(path),
    }


def _probe_jax(python_executable: Path, *, platform: str) -> dict[str, object]:
    script = r"""
import json
import jax
payload = {
    "x64_enabled": bool(jax.config.jax_enable_x64),
    "devices": [
        {
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
            "id": int(device.id),
        }
        for device in jax.devices()
    ],
}
print(json.dumps(payload))
"""
    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = "true"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if platform == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        [str(python_executable), "-c", script],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise ReSolvAuditError(
            f"JAX runtime probe failed for {python_executable}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ReSolvAuditError("JAX runtime probe returned invalid JSON.") from exc
    if not isinstance(payload, Mapping):
        raise ReSolvAuditError("JAX runtime probe did not return an object.")
    raw_devices = payload.get("devices")
    x64_enabled = payload.get("x64_enabled")
    if not isinstance(raw_devices, list) or not isinstance(x64_enabled, bool):
        raise ReSolvAuditError("JAX runtime probe payload has invalid fields.")
    devices: list[dict[str, object]] = []
    for index, raw_device in enumerate(raw_devices):
        if not isinstance(raw_device, Mapping):
            raise ReSolvAuditError(
                f"JAX runtime probe device {index} is not an object."
            )
        platform_name = raw_device.get("platform")
        device_kind = raw_device.get("device_kind")
        if (
            not isinstance(platform_name, str)
            or not platform_name
            or not isinstance(device_kind, str)
            or not device_kind
        ):
            raise ReSolvAuditError(
                f"JAX runtime probe device {index} has invalid identity fields."
            )
        devices.append(
            {
                "platform": platform_name,
                "device_kind": device_kind,
                "id": _strict_int(
                    raw_device.get("id"), f"JAX runtime probe device {index} id"
                ),
            }
        )
    return detect_jax_runtime(
        device_records=devices,
        x64_enabled=x64_enabled,
    )


def _probe_atom_counts(
    python_executable: Path,
    records: Sequence[Mapping[str, object]],
) -> dict[int, int]:
    """Use the pinned RDKit runtime to reproduce Chem.AddHs atom counts."""

    script = r"""
import json
import sys
from rdkit import Chem
records = json.load(sys.stdin)
result = {}
for record in records:
    molecule = Chem.MolFromSmiles(record["smiles"])
    if molecule is None:
        raise ValueError(f"RDKit rejected SMILES: {record['smiles']}")
    result[str(record["k_index"])] = Chem.AddHs(molecule).GetNumAtoms()
print(json.dumps(result, sort_keys=True))
"""
    completed = subprocess.run(
        [str(python_executable), "-c", script],
        input=json.dumps(
            [
                {
                    "k_index": _strict_int(record["k_index"], "record k_index"),
                    "smiles": str(record["smiles"]),
                }
                for record in records
            ]
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ReSolvAuditError(
            "Pinned RDKit atom-count probe failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        counts = {
            _strict_int(key, "atom-count k_index"): _strict_int(
                value, "atom count"
            )
            for key, value in payload.items()
        }
    except (IndexError, AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ReSolvAuditError(
            "Pinned RDKit atom-count probe returned invalid JSON."
        ) from exc
    if set(counts) != {
        _strict_int(record["k_index"], "record k_index") for record in records
    }:
        raise ReSolvAuditError("Pinned RDKit atom-count probe omitted a record.")
    return counts


def _shape_bounded_batches(
    records: Sequence[Mapping[str, object]],
    *,
    atom_counts: Mapping[int, int],
    max_shapes_per_worker: int,
) -> list[list[dict[str, object]]]:
    if max_shapes_per_worker < 1:
        raise ReSolvAuditError("max_shapes_per_worker must be positive.")
    by_shape: dict[int, list[dict[str, object]]] = {}
    for raw in records:
        record = dict(raw)
        k_index = _strict_int(record["k_index"], "record k_index")
        if k_index not in atom_counts:
            raise ReSolvAuditError(
                f"Missing atom-count identity for k_index={k_index}."
            )
        by_shape.setdefault(atom_counts[k_index], []).append(record)

    batches: list[list[dict[str, object]]] = []
    pending: list[dict[str, object]] = []
    pending_shapes = 0
    for atom_count in sorted(by_shape):
        if pending and pending_shapes >= max_shapes_per_worker:
            batches.append(pending)
            pending = []
            pending_shapes = 0
        pending.extend(by_shape[atom_count])
        pending_shapes += 1
    if pending:
        batches.append(pending)
    return batches


def _build_worker_records(
    artifacts: ReSolvArtifactSet,
    records: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    worker_records: list[dict[str, object]] = []
    trajectory_receipts: list[dict[str, object]] = []
    for record in records:
        k_index = _strict_int(record["k_index"], "record k_index")
        vacuum = artifacts.vacuum_trajectory_path(k_index)
        water = artifacts.water_trajectory_path(k_index)
        vacuum_receipt = _tracked_blob_receipt(artifacts.upstream_root, vacuum)
        water_receipt = _tracked_blob_receipt(artifacts.upstream_root, water)
        worker_records.append(
            {
                "k_index": k_index,
                "trajectory_id": _strict_int(
                    record["trajectory_id"], "record trajectory_id"
                ),
                "mobley_id": record["mobley_id"],
                "smiles": record["smiles"],
                "vacuum_trajectory": str(vacuum),
                "water_trajectory": str(water),
                "vacuum_trajectory_sha256": vacuum_receipt["sha256"],
                "water_trajectory_sha256": water_receipt["sha256"],
            }
        )
        trajectory_receipts.append(
            {
                "k_index": k_index,
                "vacuum": vacuum_receipt,
                "water": water_receipt,
            }
        )
    return worker_records, trajectory_receipts


def _run_worker(
    *,
    worker_path: Path,
    python_executable: Path,
    request: Mapping[str, object],
    platform: str,
    timeout_seconds: float,
) -> tuple[dict[str, object], float]:
    worker = Path(worker_path).resolve()
    if not worker.is_file():
        raise ReSolvAuditError(f"ReSolv sidecar worker is missing: {worker}")
    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = "true"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if platform == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    elif platform == "gpu":
        env.pop("JAX_PLATFORMS", None)
    else:
        raise ReSolvAuditError("platform must be 'cpu' or 'gpu'.")

    with tempfile.TemporaryDirectory(prefix="maple-resolv-audit-") as temp_dir:
        request_path = Path(temp_dir) / "request.json"
        response_path = Path(temp_dir) / "response.json"
        request_path.write_text(
            json.dumps(request, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [
                    str(python_executable),
                    str(worker),
                    "--request",
                    str(request_path),
                    "--response",
                    str(response_path),
                ],
                text=True,
                capture_output=True,
                env=env,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ReSolvAuditError(
                f"ReSolv worker exceeded timeout {timeout_seconds:g} s."
            ) from exc
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            raise ReSolvAuditError(
                "ReSolv worker failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        if not response_path.is_file():
            raise ReSolvAuditError("ReSolv worker wrote no response artifact.")
        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReSolvAuditError("ReSolv worker response is not valid JSON.") from exc
    if response.get("ok") is not True:
        raise ReSolvAuditError(
            "ReSolv worker returned a non-success response: "
            f"{response.get('error') or response.get('record_errors') or 'unknown error'}"
        )
    if response.get("source_revision") != RESOLV_UPSTREAM_REVISION:
        raise ReSolvAuditError("ReSolv worker response changed source revision.")
    dtype = response.get("dtype")
    if not isinstance(dtype, Mapping) or dtype.get("jax_enable_x64") is not True:
        raise ReSolvAuditError("ReSolv worker did not preserve JAX x64 execution.")
    source_snapshot = response.get("source_snapshot")
    if (
        not isinstance(source_snapshot, Mapping)
        or source_snapshot.get("revision") != RESOLV_UPSTREAM_REVISION
        or source_snapshot.get("source") != "git-archive"
        or source_snapshot.get("exported_roots")
        != ["chemtrain", "jax_sgmc", "util"]
    ):
        raise ReSolvAuditError(
            "ReSolv worker did not attest the pinned private source snapshot."
        )
    archive_sha256 = source_snapshot.get("archive_sha256")
    if (
        not isinstance(archive_sha256, str)
        or len(archive_sha256) != 64
        or any(character not in "0123456789abcdef" for character in archive_sha256)
    ):
        raise ReSolvAuditError(
            "ReSolv worker source snapshot lacks a valid archive SHA256."
        )
    worker_records = response.get("records")
    if not isinstance(worker_records, list):
        raise ReSolvAuditError("ReSolv worker response lacks records.")
    for record in worker_records:
        if record.get("bar_output_dtype") != "float64":
            raise ReSolvAuditError(
                "ReSolv worker returned a non-float64 BAR result."
            )
        if record.get("snapshots_per_endpoint") != 40:
            raise ReSolvAuditError(
                "ReSolv worker returned an endpoint other than the fixed 40 frames."
            )
        for field in (
            "vacuum_self_max_abs_diff_kcal_mol",
            "water_self_max_abs_diff_kcal_mol",
        ):
            if _finite_float(record.get(field), field) > 1.0e-6:
                raise ReSolvAuditError(
                    f"ReSolv worker self-energy identity failed: {field}."
                )
    return response, elapsed


def _run_worker_batches(
    *,
    worker_path: Path,
    python_executable: Path,
    base_request: Mapping[str, object],
    batches: Sequence[Sequence[Mapping[str, object]]],
    platform: str,
    timeout_seconds: float,
) -> tuple[dict[str, object], float]:
    """Run bounded shape groups so XLA executable caches cannot exhaust RAM."""

    started = time.perf_counter()
    batch_responses: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for batch in batches:
        response, _ = _run_worker(
            worker_path=worker_path,
            python_executable=python_executable,
            request={**dict(base_request), "records": list(batch)},
            platform=platform,
            timeout_seconds=timeout_seconds,
        )
        batch_responses.append(response)
        batch_records = response.get("records")
        if not isinstance(batch_records, list):
            raise ReSolvAuditError("A ReSolv worker batch omitted records.")
        records.extend(batch_records)
    first = batch_responses[0]
    for response in batch_responses[1:]:
        for field in (
            "source_revision",
            "backend",
            "devices",
            "dtype",
            "source_snapshot",
            "platform_attestation",
            "runtime_environment",
        ):
            if response.get(field) != first.get(field):
                raise ReSolvAuditError(
                    f"ReSolv worker batches disagree on {field}."
                )
    combined = {
        **first,
        "records": records,
        "record_errors": [],
        "runtime": {
            "batch_count": len(batch_responses),
            "batch_runtime": [
                response.get("runtime") for response in batch_responses
            ],
        },
    }
    return combined, time.perf_counter() - started


def _merge_prediction_records(
    manifest_records: Sequence[Mapping[str, object]],
    worker_records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_index = {
        _strict_int(record["k_index"], "worker k_index"): record
        for record in worker_records
    }
    if len(by_index) != len(worker_records):
        raise ReSolvAuditError("Worker returned duplicate k_index values.")
    merged: list[dict[str, object]] = []
    for source in manifest_records:
        k_index = _strict_int(source["k_index"], "manifest k_index")
        if k_index not in by_index:
            raise ReSolvAuditError(
                f"Worker response omitted ReSolv k_index={k_index}."
            )
        prediction = by_index[k_index]
        prediction_value = prediction["delta_g_kcal_mol"]
        bar_diagnostics = diagnose_equal_endpoint_bar(
            prediction.get("bar_diagnostics"),
            upstream_delta_g_kcal_mol=prediction_value,
        )
        merged.append(
            {
                **dict(source),
                "prediction_kcal_mol": prediction_value,
                "snapshots_per_endpoint": prediction.get(
                    "snapshots_per_endpoint"
                ),
                "vacuum_self_max_abs_diff_kcal_mol": prediction.get(
                    "vacuum_self_max_abs_diff_kcal_mol"
                ),
                "water_self_max_abs_diff_kcal_mol": prediction.get(
                    "water_self_max_abs_diff_kcal_mol"
                ),
                **bar_diagnostics,
            }
        )
    return merged


def _record_identity(record: Mapping[str, object]) -> tuple[int, int, str]:
    try:
        identity = (
            _strict_int(record["k_index"], "record k_index"),
            _strict_int(record["trajectory_id"], "record trajectory_id"),
            str(record["mobley_id"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReSolvAuditError(
            "ReSolv record identity requires k_index, trajectory_id, and mobley_id."
        ) from exc
    if not identity[2]:
        raise ReSolvAuditError("ReSolv mobley_id must not be empty.")
    return identity


def _predictions_by_identity(
    manifest_records: Sequence[Mapping[str, object]],
    worker_records: Sequence[Mapping[str, object]],
) -> dict[tuple[int, int, str], float]:
    """Bind worker k-indices back to immutable manifest identities."""

    manifest_by_index: dict[int, Mapping[str, object]] = {}
    identities: set[tuple[int, int, str]] = set()
    for record in manifest_records:
        identity = _record_identity(record)
        k_index = identity[0]
        if k_index in manifest_by_index or identity in identities:
            raise ReSolvAuditError("Manifest contains duplicate ReSolv identities.")
        manifest_by_index[k_index] = record
        identities.add(identity)

    predictions: dict[tuple[int, int, str], float] = {}
    seen_indices: set[int] = set()
    for record in worker_records:
        try:
            k_index = _strict_int(record["k_index"], "worker k_index")
        except (KeyError, TypeError, ValueError) as exc:
            raise ReSolvAuditError("Worker record lacks a valid k_index.") from exc
        if k_index in seen_indices:
            raise ReSolvAuditError("Worker returned duplicate k_index values.")
        seen_indices.add(k_index)
        source = manifest_by_index.get(k_index)
        if source is None:
            raise ReSolvAuditError(
                f"Worker returned unexpected ReSolv k_index={k_index}."
            )
        identity = _record_identity(source)
        predictions[identity] = _finite_float(
            record.get("delta_g_kcal_mol"),
            "repeat prediction",
        )
    if set(predictions) != identities:
        missing = sorted(identities - set(predictions))
        raise ReSolvAuditError(
            f"Worker response omitted ReSolv identities: {missing[:3]}."
        )
    return predictions


def _prediction_bits_by_identity(
    predictions: Mapping[tuple[int, int, str], float],
) -> dict[tuple[int, int, str], bytes]:
    """Return the literal IEEE-754 binary64 representation for each result."""

    return {
        identity: struct.pack(">d", value)
        for identity, value in predictions.items()
    }


def _portable_manifest(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Remove host-specific absolute roots while preserving endpoint identity."""

    portable: list[dict[str, object]] = []
    for source in records:
        record = dict(source)
        record["vacuum_trajectory"] = Path(
            str(record["vacuum_trajectory"])
        ).name
        record["water_trajectory"] = Path(
            str(record["water_trajectory"])
        ).name
        portable.append(record)
    return portable


def _portable_manifest_payload(
    records: Sequence[Mapping[str, object]],
    *,
    require_frozen_full_test_identity: bool,
) -> tuple[bytes, str]:
    """Serialize once and optionally enforce the frozen 162-record identity."""

    payload = json.dumps(records, indent=2, sort_keys=True).encode("utf-8")
    observed = hashlib.sha256(payload).hexdigest()
    if (
        require_frozen_full_test_identity
        and observed != RESOLV_FULL_TEST_MANIFEST_SHA256
    ):
        raise ReSolvAuditError(
            "Pinned ReSolv full-test manifest identity drifted: expected "
            f"{RESOLV_FULL_TEST_MANIFEST_SHA256}, observed {observed}."
        )
    return payload, observed


def _write_records_csv(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "k_index",
        "trajectory_id",
        "mobley_id",
        "smiles",
        "iupac",
        "split",
        "primary_functional_group",
        "groups",
        "prediction_kcal_mol",
        "experimental_kcal_mol",
        "experimental_uncertainty_kcal_mol",
        "signed_error_kcal_mol",
        "absolute_error_kcal_mol",
        "snapshots_per_endpoint",
        "vacuum_self_max_abs_diff_kcal_mol",
        "water_self_max_abs_diff_kcal_mol",
        "bar_root_dimensionless",
        "bar_root_residual",
        "bar_reconstructed_delta_g_kcal_mol",
        "bar_upstream_root_abs_mismatch_kcal_mol",
        "bar_overlap_omega",
        "bar_vacuum_direction_kish_ess",
        "bar_water_direction_kish_ess",
        "bar_min_directional_kish_ess",
        "bar_effective_overlap_mass_total",
        "bar_conditional_iid_information",
        "bar_conditional_iid_variance_dimensionless",
        "bar_conditional_iid_se_kcal_mol",
        "bar_overlap_support_pass",
        "statistical_support_eligible",
        "bar_uncertainty_inferential",
        "bar_confidence_interval_available",
        "bar_bootstrap_performed",
        "bar_uncertainty_limitation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in fieldnames}
            groups = record.get("groups")
            if not isinstance(groups, (list, tuple)):
                raise ReSolvAuditError("Record groups must be a sequence.")
            row["groups"] = "; ".join(str(value) for value in groups)
            writer.writerow(row)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path.home() / ".cache/maple/resolv/ReSolv",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        type=Path,
        default=Path.home() / "miniconda3/envs/maple-resolv/bin/python",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--max-shapes-per-worker",
        type=int,
        default=4,
        help="Bound XLA shape caches per subprocess to avoid whole-panel RAM growth.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Engineering-only prefix limit; omitted means all 162 test records.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=14400.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.repeats < 1:
        raise ReSolvAuditError("--repeats must be at least one.")
    if args.limit is not None and args.limit < 1:
        raise ReSolvAuditError("--limit must be positive.")
    if not args.python_executable.is_file():
        raise ReSolvAuditError(
            f"ReSolv Python executable is missing: {args.python_executable}"
        )
    initial_code_sha256, immutable_worker_bytes = _read_code_snapshot()

    artifacts = ReSolvArtifactSet(upstream_root=args.upstream_root)
    manifest = build_resolv_manifest(artifacts)
    selected = [record for record in manifest if record["split"] == "test"]
    if args.limit is not None:
        selected = selected[: args.limit]
    validate_artifacts(
        artifacts,
        required_k_indices=(
            _strict_int(record["k_index"], "manifest k_index")
            for record in selected
        ),
    )
    runtime_probe = _probe_jax(
        args.python_executable,
        platform=args.platform,
    )
    _validate_requested_runtime(
        requested_platform=args.platform,
        runtime_probe=runtime_probe,
    )

    worker_records, trajectory_receipts = _build_worker_records(
        artifacts,
        selected,
    )
    atom_counts = _probe_atom_counts(args.python_executable, worker_records)
    batches = _shape_bounded_batches(
        worker_records,
        atom_counts=atom_counts,
        max_shapes_per_worker=args.max_shapes_per_worker,
    )
    portable_selected = _portable_manifest(selected)
    full_panel = len(selected) == 162
    manifest_payload, manifest_sha256 = _portable_manifest_payload(
        portable_selected,
        require_frozen_full_test_identity=full_panel,
    )
    assert artifacts.vacuum_model_path is not None
    assert artifacts.water_model_path is not None
    assert artifacts.database_path is not None
    request = {
        "protocol": dict(PROTOCOL_SEMANTICS),
        "upstream_root": str(artifacts.upstream_root),
        "source_revision": RESOLV_UPSTREAM_REVISION,
        "vacuum_model_path": str(artifacts.vacuum_model_path),
        "vacuum_model_sha256": RESOLV_VACUUM_MODEL_SHA256,
        "water_model_path": str(artifacts.water_model_path),
        "water_model_sha256": RESOLV_WATER_MODEL_SHA256,
        "database_path": str(artifacts.database_path),
        "database_sha256": RESOLV_DATABASE_SHA256,
        "temperature_kelvin": 298.15,
        "shape_batching": {
            "distinct_total_atom_counts": sorted(set(atom_counts.values())),
            "max_shapes_per_worker": args.max_shapes_per_worker,
            "worker_batch_count": len(batches),
        },
    }

    responses: list[dict[str, object]] = []
    elapsed_runs: list[float] = []
    with tempfile.TemporaryDirectory(
        prefix="maple-resolv-worker-snapshot-"
    ) as snapshot_dir:
        worker_snapshot = Path(snapshot_dir) / "_resolv_worker.py"
        worker_snapshot.write_bytes(immutable_worker_bytes)
        if sha256_file(worker_snapshot) != initial_code_sha256["worker"]:
            raise ReSolvAuditError(
                "Immutable ReSolv worker snapshot SHA256 mismatch."
            )
        for _ in range(args.repeats):
            response, elapsed = _run_worker_batches(
                worker_path=worker_snapshot,
                python_executable=args.python_executable,
                base_request=request,
                batches=batches,
                platform=args.platform,
                timeout_seconds=args.timeout_seconds,
            )
            responses.append(response)
            elapsed_runs.append(elapsed)

    _validate_requested_runtime(
        requested_platform=args.platform,
        runtime_probe=runtime_probe,
        worker_response=responses[0],
    )
    first_records = responses[0].get("records")
    if not isinstance(first_records, list):
        raise ReSolvAuditError("ReSolv worker response lacks a records list.")
    first_predictions = _predictions_by_identity(portable_selected, first_records)
    first_prediction_bits = _prediction_bits_by_identity(first_predictions)
    for repeat_index, response in enumerate(responses[1:], start=2):
        _validate_requested_runtime(
            requested_platform=args.platform,
            runtime_probe=runtime_probe,
            worker_response=response,
        )
        other_records = response.get("records")
        if not isinstance(other_records, list):
            raise ReSolvAuditError(
                f"ReSolv worker repeat {repeat_index} lacks records."
            )
        other_predictions = _predictions_by_identity(
            portable_selected,
            other_records,
        )
        if first_prediction_bits != _prediction_bits_by_identity(other_predictions):
            raise ReSolvAuditError(
                f"ReSolv repeat {repeat_index} changed one or more binary64 "
                "predictions."
            )

    merged = _merge_prediction_records(portable_selected, first_records)
    aggregate = aggregate_results(
        merged,
        minimum_primary_groups=10,
        require_bar_diagnostics=True,
    )
    if _code_sha256() != initial_code_sha256:
        raise ReSolvAuditError(
            "ReSolv audit code changed while the audit was running."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_bytes(manifest_payload)
    result = {
        "schema_version": 1,
        "candidate": "ReSolv",
        "scope": (
            "full_published_test_split"
            if full_panel
            else "engineering_prefix_not_accuracy_evidence"
        ),
        "provenance": {
            "source_revision": RESOLV_UPSTREAM_REVISION,
            "code_sha256": initial_code_sha256,
            "executing_module_code_sha256": {
                "audit_runner": IMPORTED_RUNNER_CODE_SHA256,
                "protocol": RESOLV_IMPORTED_MODULE_CODE_SHA256,
            },
            "artifact_sha256": {
                "vacuum_model": RESOLV_VACUUM_MODEL_SHA256,
                "water_model": RESOLV_WATER_MODEL_SHA256,
                "database": RESOLV_DATABASE_SHA256,
            },
            "manifest_sha256": manifest_sha256,
            "full_test_manifest_identity": {
                "expected_sha256": RESOLV_FULL_TEST_MANIFEST_SHA256,
                "observed_sha256": manifest_sha256,
                "verified": full_panel,
            },
            "trajectory_receipts": trajectory_receipts,
            "shape_batching": request["shape_batching"],
        },
        "runtime": {
            **runtime_probe,
            "requested_platform": args.platform,
        },
        "worker": {
            "backend": responses[0].get("backend"),
            "devices": responses[0].get("devices"),
            "dtype": responses[0].get("dtype"),
            "environment": responses[0].get("runtime_environment"),
            "platform_attestation": responses[0].get("platform_attestation"),
            "source_snapshot": responses[0].get("source_snapshot"),
        },
        "performance_gate": {
            "matched_pure_qm_baseline_seconds": None,
            "speed_advantage_vs_pure_qm_proven": False,
            "speed_gate_failure": "matched_pure_qm_full_task_baseline_missing",
        },
        "repeat_determinism": {
            "repeat_count": args.repeats,
            "identity_keyed_bitwise_stable": True,
            "minimum_repeats_for_parity_admission": 3,
        },
        **aggregate,
    }
    result_path = args.output_dir / "audit.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_records_csv(args.output_dir / "records.csv", result["records"])
    runtime_diagnostic_path = args.output_dir / "runtime-diagnostic.json"
    runtime_diagnostic_path.write_text(
        json.dumps(
            {
                "scope": (
                    "three_or_more_cold_worker_replays_not_full_task_timing"
                    if args.repeats >= 3 and full_panel
                    else "worker_replay_diagnostic_only_not_performance_admission"
                ),
                "repeat_count": args.repeats,
                "identity_keyed_bitwise_stable": True,
                "cold_worker_replay_seconds": elapsed_runs,
                "median_cold_worker_replay_seconds": statistics.median(elapsed_runs),
                "worker_runtime": [
                    response.get("runtime") for response in responses
                ],
                "excluded_from_timing": [
                    "manifest_and_trajectory_validation",
                    "atom_count_probe_and_batch_preparation",
                    "aggregation_and_diagnostics",
                    "artifact_writes",
                ],
                "performance_admission_eligible": False,
                "matched_pure_qm_baseline": False,
                "speed_advantage_vs_pure_qm_proven": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    receipt = {
        "audit_json_sha256": sha256_file(result_path),
        "records_csv_sha256": sha256_file(args.output_dir / "records.csv"),
        "manifest_json_sha256": sha256_file(manifest_path),
        "runtime_diagnostic_sha256": sha256_file(runtime_diagnostic_path),
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), **receipt}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
