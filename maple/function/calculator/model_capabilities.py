"""Machine-readable capability and provenance contracts for model composition."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

MODEL_CARD_EXTENSIONS = (".yaml", ".yml", ".json")
CONSERVATIVE_NUMERICAL_FREQUENCY_TYPE = "effective_solution_pmf"
GPU_EVIDENCE_ROOT = Path(__file__).resolve().parents[3]
GPU_PARITY_SCHEMA_VERSION = 2
GPU_COMPARISON_MANIFEST_SCHEMA_VERSION = 1
GPU_PARITY_VERDICT = "pass_no_loss"
GPU_FORBIDDEN_PRECISION_CONTROLS = (
    "autocast",
    "bfloat16",
    "fast_math",
    "float16",
    "reduced_matmul",
    "relaxed_convergence",
    "shortened_sampling",
    "tf32",
)
GPU_OBSERVABLE_COMPARISONS = {
    "absolute",
    "equal",
    "non_degradation",
}
GPU_ADMITTED_SCALAR_DTYPES = {"float32", "float64"}
GPU_ADMITTED_MATMUL_PRECISIONS = {"highest", "ieee"}
GPU_STRICT_ACCURACY_CONTRACTS = {
    "coverage": "equal",
    "literature_panel_metrics": "non_degradation",
    "maximum_error": "non_degradation",
    "per_case_accuracy": "non_degradation",
}

HessianMode = Literal["analytic", "autograd", "finite_difference", "none"]
SolvationMode = Literal["none", "native", "additive", "alchemical", "property_only"]
EnergyReference = Literal["absolute", "relative", "unknown"]


class ModelCardError(ValueError):
    """Raised when a model-provenance card cannot be loaded or validated."""


def _strict_bool(
    payload: Mapping[str, object],
    name: str,
    *,
    default: bool = False,
) -> bool:
    value = payload.get(name, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    raise ModelCardError(
        f"Model-card field {name!r} must be a boolean, got {type(value).__name__}."
    )


def _normalized_string_sequence(raw: object, name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ModelCardError(f"Model-card field {name!r} must be a sequence.")
    return tuple(
        sorted(
            {
                str(value).strip().lower().replace("-", "_")
                for value in raw
                if str(value).strip()
            }
        )
    )


def _normalized_label(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _validated_sha256(value: object, name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64:
        raise ModelCardError(
            f"Model-card field {name!r} must contain 64 hexadecimal characters."
        )
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ModelCardError(f"Model-card field {name!r} must be hexadecimal.") from exc
    return digest


def _require_nonempty_model_card_string(
    payload: Mapping[str, object],
    name: str,
) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ModelCardError(f"Model-card field {name!r} must be a non-empty string.")
    return value.strip()


def _observable_contracts(
    raw: object,
) -> tuple[tuple[str, str, str, float], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise ModelCardError(
            "Model-card gpu_acceleration.observable_contracts must be a mapping."
        )

    contracts: list[tuple[str, str, str, float]] = []
    normalized_names: set[str] = set()
    for raw_name, raw_record in raw.items():
        name = _normalized_label(raw_name)
        if not name:
            raise ModelCardError("GPU observable contract names must be non-empty.")
        if name in normalized_names:
            raise ModelCardError(
                f"GPU observable contract name {name!r} is duplicated after "
                "normalization."
            )
        normalized_names.add(name)
        if not isinstance(raw_record, Mapping):
            raise ModelCardError(f"GPU observable contract {name!r} must be a mapping.")
        unit = _require_nonempty_model_card_string(raw_record, "unit")
        comparison = _normalized_label(
            _require_nonempty_model_card_string(raw_record, "comparison")
        )
        if comparison not in GPU_OBSERVABLE_COMPARISONS:
            raise ModelCardError(
                f"GPU observable contract {name!r} has unsupported comparison "
                f"{comparison!r}."
            )
        threshold = raw_record.get("threshold")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or float(threshold) < 0.0
        ):
            raise ModelCardError(
                f"GPU observable contract {name!r} threshold must be a finite "
                "non-negative number."
            )
        contracts.append((name, unit, comparison, float(threshold)))
    return tuple(sorted(contracts))


def _checkpoint_sha256s(payload: Mapping[str, object]) -> tuple[str, ...]:
    """Collect every explicitly named checkpoint digest from a model card."""

    digests: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key) == "checkpoint_sha256":
                    digests.add(_validated_sha256(child, "checkpoint_sha256"))
                else:
                    visit(child)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for child in value:
                visit(child)

    visit(payload)
    return tuple(sorted(digests))


def is_explicit_cpu_device(device: object) -> bool:
    device_name = str(device).strip().lower()
    return device_name == "cpu" or device_name.startswith("cpu:")


def _require_parity_mapping(
    payload: Mapping[str, object],
    name: str,
) -> Mapping[str, object]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"GPU parity artifact field {name!r} must be a mapping.")
    return value


def _require_nonempty_parity_string(
    payload: Mapping[str, object],
    name: str,
) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"GPU parity artifact field {name!r} must be a non-empty string."
        )
    return value.strip()


def _validated_evidence_sha256(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"GPU evidence field {name!r} must be a SHA256 string.")
    digest = value.strip().lower()
    if len(digest) != 64:
        raise ValueError(
            f"GPU evidence field {name!r} must contain 64 hexadecimal characters."
        )
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"GPU evidence field {name!r} must be hexadecimal.") from exc
    return digest


def _gpu_output_sha256(
    *,
    record_id: str,
    observable: str,
    role: str,
    values: object,
) -> str:
    payload = {
        "observable": observable,
        "record_id": record_id,
        "role": role,
        "values": values,
    }
    try:
        encoded = (json.dumps(payload, allow_nan=False, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"GPU parity output {record_id!r}/{observable!r}/{role!r} must be "
            "finite JSON."
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _finite_numeric_tensor(
    value: object,
    *,
    label: str,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if isinstance(value, bool):
        raise ValueError(f"GPU parity output {label} must contain finite numbers.")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"GPU parity output {label} must contain finite numbers.")
        return (), (number,)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(
            f"GPU parity output {label} must be a finite numeric scalar or array."
        )
    if not value:
        raise ValueError(f"GPU parity output {label} must not be empty.")

    children = [
        _finite_numeric_tensor(child, label=f"{label}[{index}]")
        for index, child in enumerate(value)
    ]
    child_shape = children[0][0]
    if any(shape != child_shape for shape, _ in children[1:]):
        raise ValueError(f"GPU parity output {label} must not be ragged.")
    numbers = tuple(number for _, values in children for number in values)
    return (len(children), *child_shape), numbers


def _validate_gpu_paired_outputs(
    record: Mapping[str, object],
    *,
    observable: str,
    comparison: str,
    manifest_output_sha256s: Mapping[str, Mapping[str, str]],
) -> tuple[int, float]:
    raw_pairs = record.get("paired_outputs")
    if not isinstance(raw_pairs, Sequence) or isinstance(
        raw_pairs, (str, bytes, bytearray)
    ):
        raise ValueError(
            f"GPU parity observable {observable!r} paired_outputs must be a sequence."
        )
    if not raw_pairs:
        raise ValueError(
            f"GPU parity observable {observable!r} paired_outputs must not be empty."
        )

    seen_record_ids: set[str] = set()
    comparison_count = 0
    observed_maximum = 0.0
    for index, pair in enumerate(raw_pairs):
        if not isinstance(pair, Mapping):
            raise ValueError(
                f"GPU parity observable {observable!r} paired output {index} "
                "must be a mapping."
            )
        record_id = _require_nonempty_parity_string(pair, "record_id")
        if record_id in seen_record_ids:
            raise ValueError(
                f"GPU parity observable {observable!r} record_id {record_id!r} "
                "is duplicated."
            )
        seen_record_ids.add(record_id)
        try:
            manifest_hashes = manifest_output_sha256s[record_id]
        except KeyError as exc:
            raise ValueError(
                f"GPU parity observable {observable!r} record_id {record_id!r} "
                "is not present in the comparison manifest."
            ) from exc

        reference_values = pair.get("reference_values")
        accelerator_values = pair.get("accelerator_values")
        reference_shape, reference_numbers = _finite_numeric_tensor(
            reference_values,
            label=f"{observable}.{record_id}.reference_values",
        )
        accelerator_shape, accelerator_numbers = _finite_numeric_tensor(
            accelerator_values,
            label=f"{observable}.{record_id}.accelerator_values",
        )
        if reference_shape != accelerator_shape:
            raise ValueError(
                f"GPU parity observable {observable!r} record {record_id!r} "
                "reference and accelerator output shapes differ."
            )

        for role, values in (
            ("reference", reference_values),
            ("accelerator", accelerator_values),
        ):
            declared_sha256 = _validated_evidence_sha256(
                pair.get(f"{role}_sha256"),
                f"{observable}.{record_id}.{role}_sha256",
            )
            computed_sha256 = _gpu_output_sha256(
                record_id=record_id,
                observable=observable,
                role=role,
                values=values,
            )
            if declared_sha256 != computed_sha256:
                raise ValueError(
                    f"GPU parity observable {observable!r} record {record_id!r} "
                    f"{role} output SHA256 does not match its values."
                )
            if declared_sha256 != manifest_hashes[role]:
                raise ValueError(
                    f"GPU parity observable {observable!r} record {record_id!r} "
                    f"{role} output SHA256 does not match the comparison manifest."
                )

        comparison_count += len(reference_numbers)
        if comparison == "equal":
            pair_maximum = 0.0 if reference_numbers == accelerator_numbers else 1.0
        elif comparison == "absolute":
            pair_maximum = max(
                abs(accelerator - reference)
                for reference, accelerator in zip(
                    reference_numbers, accelerator_numbers
                )
            )
        else:
            pair_maximum = max(
                0.0,
                max(
                    accelerator - reference
                    for reference, accelerator in zip(
                        reference_numbers, accelerator_numbers
                    )
                ),
            )
        observed_maximum = max(observed_maximum, pair_maximum)

    if seen_record_ids != set(manifest_output_sha256s):
        raise ValueError(
            f"GPU parity observable {observable!r} paired_outputs do not exactly "
            "cover the comparison manifest records."
        )
    return comparison_count, observed_maximum


def _validate_gpu_comparison_manifest(
    manifest: Path,
    *,
    comparison_manifest_id: str,
    model_id: str,
    model_version: str,
    checkpoint_sha256s: tuple[str, ...],
    task_modes: tuple[str, ...],
    required_observables: tuple[str, ...],
) -> tuple[
    Mapping[str, int],
    Mapping[str, Mapping[str, Mapping[str, str]]],
]:
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"GPU comparison manifest must be readable JSON: {manifest}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("GPU comparison manifest must contain one JSON object.")
    if payload.get("schema_version") != GPU_COMPARISON_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "GPU comparison manifest schema_version does not match the admitted "
            f"version {GPU_COMPARISON_MANIFEST_SCHEMA_VERSION}."
        )
    if (
        _require_nonempty_parity_string(payload, "comparison_manifest_id")
        != comparison_manifest_id
    ):
        raise ValueError("GPU comparison manifest ID does not match the model card.")
    if _require_nonempty_parity_string(payload, "model_id").lower() != model_id:
        raise ValueError(
            "GPU comparison manifest model_id does not match the model card."
        )
    if _require_nonempty_parity_string(payload, "model_version") != model_version:
        raise ValueError(
            "GPU comparison manifest model_version does not match the model card."
        )
    manifest_checkpoints = _normalized_string_sequence(
        payload.get("checkpoint_sha256s"),
        "comparison_manifest.checkpoint_sha256s",
    )
    if manifest_checkpoints != checkpoint_sha256s:
        raise ValueError(
            "GPU comparison manifest checkpoint identities do not match the "
            "model card."
        )
    manifest_task_modes = _normalized_string_sequence(
        payload.get("task_modes"),
        "comparison_manifest.task_modes",
    )
    if manifest_task_modes != task_modes:
        raise ValueError(
            "GPU comparison manifest task/mode coverage does not match the model "
            "card."
        )

    records = payload.get("records")
    if not isinstance(records, Sequence) or isinstance(
        records, (str, bytes, bytearray)
    ):
        raise ValueError("GPU comparison manifest records must be a sequence.")
    if not records:
        raise ValueError("GPU comparison manifest records must not be empty.")
    record_ids: set[str] = set()
    output_sha256s: dict[str, dict[str, dict[str, str]]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(
                f"GPU comparison manifest record {index} must be a mapping."
            )
        record_id = _require_nonempty_parity_string(record, "record_id")
        if record_id in record_ids:
            raise ValueError(
                f"GPU comparison manifest record_id {record_id!r} is duplicated."
            )
        record_ids.add(record_id)
        _require_nonempty_parity_string(record, "panel_id")
        _validated_evidence_sha256(
            record.get("configuration_sha256"),
            f"records[{index}].configuration_sha256",
        )
        _validated_evidence_sha256(
            record.get("reference_sha256"),
            f"records[{index}].reference_sha256",
        )
        raw_output_sha256s = record.get("observable_output_sha256s")
        if not isinstance(raw_output_sha256s, Mapping):
            raise ValueError(
                f"GPU comparison manifest record {index} "
                "observable_output_sha256s must be a mapping."
            )
        if set(raw_output_sha256s) != set(required_observables):
            raise ValueError(
                f"GPU comparison manifest record {index} output hashes must "
                "exactly cover every required observable."
            )
        record_hashes: dict[str, dict[str, str]] = {}
        for observable in required_observables:
            raw_hashes = raw_output_sha256s.get(observable)
            if not isinstance(raw_hashes, Mapping) or set(raw_hashes) != {
                "reference",
                "accelerator",
            }:
                raise ValueError(
                    f"GPU comparison manifest record {index} observable "
                    f"{observable!r} must bind reference and accelerator outputs."
                )
            record_hashes[observable] = {
                role: _validated_evidence_sha256(
                    raw_hashes.get(role),
                    (
                        f"records[{index}].observable_output_sha256s."
                        f"{observable}.{role}"
                    ),
                )
                for role in ("reference", "accelerator")
            }
        output_sha256s[record_id] = record_hashes

    raw_counts = payload.get("observable_counts")
    if not isinstance(raw_counts, Mapping):
        raise ValueError("GPU comparison manifest observable_counts must be a mapping.")
    if set(raw_counts) != set(required_observables):
        raise ValueError(
            "GPU comparison manifest observable_counts must exactly match every "
            "required observable."
        )
    counts: dict[str, int] = {}
    for name in required_observables:
        count = raw_counts.get(name)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(
                f"GPU comparison manifest observable count {name!r} must be a "
                "positive integer."
            )
        counts[name] = count
    return counts, output_sha256s


def _validate_gpu_parity_artifact(
    artifact: Path,
    *,
    model_id: str,
    model_version: str,
    checkpoint_sha256s: tuple[str, ...],
    comparison_manifest_id: str,
    comparison_manifest_sha256: str,
    allowed_tasks: tuple[str, ...],
    allowed_inference_modes: tuple[str, ...],
    allowed_hardware: tuple[str, ...],
    compute_dtype: str,
    matmul_precision: str,
    required_runtime_packages: tuple[str, ...],
    required_observables: tuple[str, ...],
    observable_contracts: tuple[tuple[str, str, str, float], ...],
    manifest_observable_counts: Mapping[str, int],
    manifest_output_sha256s: Mapping[
        str,
        Mapping[str, Mapping[str, str]],
    ],
    runtime_context: Mapping[str, object],
) -> None:
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"GPU parity artifact must be readable JSON: {artifact}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("GPU parity artifact must contain one JSON object.")
    if payload.get("schema_version") != GPU_PARITY_SCHEMA_VERSION:
        raise ValueError(
            "GPU parity artifact schema_version does not match the admitted "
            f"version {GPU_PARITY_SCHEMA_VERSION}."
        )
    if payload.get("verdict") != GPU_PARITY_VERDICT:
        raise ValueError(f"GPU parity artifact verdict must be {GPU_PARITY_VERDICT!r}.")
    if _require_nonempty_parity_string(payload, "model_id").lower() != model_id:
        raise ValueError("GPU parity artifact model_id does not match the model card.")
    if _require_nonempty_parity_string(payload, "model_version") != model_version:
        raise ValueError(
            "GPU parity artifact model_version does not match the model card."
        )
    if (
        _require_nonempty_parity_string(payload, "comparison_manifest_id")
        != comparison_manifest_id
    ):
        raise ValueError(
            "GPU parity artifact comparison manifest ID does not match the model card."
        )
    if (
        _require_nonempty_parity_string(payload, "comparison_manifest_sha256").lower()
        != comparison_manifest_sha256
    ):
        raise ValueError(
            "GPU parity artifact comparison manifest SHA256 does not match the "
            "model card."
        )

    evidence_checkpoints = _normalized_string_sequence(
        payload.get("checkpoint_sha256s"),
        "checkpoint_sha256s",
    )
    if evidence_checkpoints != checkpoint_sha256s:
        raise ValueError(
            "GPU parity artifact checkpoint identities do not match the model card."
        )
    evidence_tasks = tuple(
        _normalize_task_name(task)
        for task in _normalized_string_sequence(
            payload.get("allowed_tasks"),
            "allowed_tasks",
        )
    )
    if tuple(sorted(set(evidence_tasks))) != allowed_tasks:
        raise ValueError(
            "GPU parity artifact allowed_tasks do not match the model card."
        )
    evidence_modes = _normalized_string_sequence(
        payload.get("allowed_inference_modes"),
        "allowed_inference_modes",
    )
    if evidence_modes != allowed_inference_modes:
        raise ValueError(
            "GPU parity artifact inference modes do not match the model card."
        )
    expected_task_modes = tuple(
        sorted(
            f"{task}:{mode}"
            for task in allowed_tasks
            for mode in allowed_inference_modes
        )
    )
    evidence_task_modes = _normalized_string_sequence(
        payload.get("validated_task_modes"),
        "validated_task_modes",
    )
    if evidence_task_modes != expected_task_modes:
        raise ValueError(
            "GPU parity artifact validated_task_modes do not cover the exact "
            "admitted task/mode combinations."
        )

    precision = _require_parity_mapping(payload, "precision_controls")
    if precision.get("same_scalar_precision") is not True:
        raise ValueError(
            "GPU parity artifact must prove the same accepted scalar precision."
        )
    reference_dtype = _require_nonempty_parity_string(
        precision, "reference_dtype"
    ).lower()
    accelerator_dtype = _require_nonempty_parity_string(
        precision, "accelerator_dtype"
    ).lower()
    if accelerator_dtype != reference_dtype:
        raise ValueError(
            "GPU parity artifact reference and accelerator dtypes must match."
        )
    if accelerator_dtype != compute_dtype:
        raise ValueError(
            "GPU parity artifact dtype does not match the model-card compute_dtype."
        )
    if (
        _normalized_label(
            _require_nonempty_parity_string(precision, "matmul_precision")
        )
        != matmul_precision
    ):
        raise ValueError(
            "GPU parity artifact matmul precision does not match the model card."
        )
    for name in GPU_FORBIDDEN_PRECISION_CONTROLS:
        if precision.get(name) is not False:
            raise ValueError(
                f"GPU parity artifact must explicitly record {name}=false."
            )

    runtime = _require_parity_mapping(payload, "runtime")
    for name in ("reference_backend", "accelerator_backend", "hardware"):
        _require_nonempty_parity_string(runtime, name)
    hardware = (
        _require_nonempty_parity_string(runtime, "hardware").lower().replace("-", "_")
    )
    if hardware not in allowed_hardware:
        raise ValueError(
            "GPU parity artifact hardware is not admitted by the model card."
        )
    software_versions = runtime.get("software_versions")
    if not isinstance(software_versions, Mapping) or not software_versions:
        raise ValueError(
            "GPU parity artifact runtime.software_versions must be a non-empty mapping."
        )
    if any(
        not str(name).strip() or not str(version).strip()
        for name, version in software_versions.items()
    ):
        raise ValueError(
            "GPU parity artifact runtime software names and versions must be non-empty."
        )
    normalized_versions = {
        str(name).strip().lower().replace("-", "_"): str(version).strip()
        for name, version in software_versions.items()
    }
    if any(name not in normalized_versions for name in required_runtime_packages):
        raise ValueError(
            "GPU parity artifact omits a required runtime package version."
        )

    artifact_backend = (
        _require_nonempty_parity_string(runtime, "accelerator_backend")
        .lower()
        .replace("-", "_")
    )
    current_backend = (
        _require_nonempty_parity_string(runtime_context, "accelerator_backend")
        .lower()
        .replace("-", "_")
    )
    current_hardware = (
        _require_nonempty_parity_string(runtime_context, "hardware")
        .lower()
        .replace("-", "_")
    )
    current_versions = runtime_context.get("software_versions")
    if not isinstance(current_versions, Mapping):
        raise ValueError(
            "Accelerator runtime context software_versions must be a mapping."
        )
    normalized_current_versions = {
        str(name).strip().lower().replace("-", "_"): str(version).strip()
        for name, version in current_versions.items()
    }
    if current_backend != artifact_backend:
        raise ValueError(
            "Current accelerator backend does not match the parity artifact."
        )
    if current_hardware != hardware:
        raise ValueError(
            "Current accelerator hardware does not match the parity artifact."
        )
    for package in required_runtime_packages:
        if normalized_current_versions.get(package) != normalized_versions[package]:
            raise ValueError(
                f"Current runtime package {package!r} does not match the parity artifact."
            )

    current_precision = _require_parity_mapping(runtime_context, "precision_controls")
    if (
        _normalized_label(
            _require_nonempty_parity_string(current_precision, "compute_dtype")
        )
        != compute_dtype
    ):
        raise ValueError(
            "Current accelerator compute dtype does not match the model card."
        )
    if (
        _normalized_label(
            _require_nonempty_parity_string(current_precision, "matmul_precision")
        )
        != matmul_precision
    ):
        raise ValueError(
            "Current accelerator matmul precision does not match the model card."
        )
    for name in GPU_FORBIDDEN_PRECISION_CONTROLS:
        if current_precision.get(name) is not False:
            raise ValueError(
                f"Current accelerator precision state must record {name}=false."
            )

    observables = _require_parity_mapping(payload, "observables")
    if set(observables) != set(required_observables):
        raise ValueError(
            "GPU parity artifact observables must exactly match every exposed "
            "quantity and scientific-accuracy contract."
        )
    contract_by_name = {
        name: {
            "unit": unit,
            "comparison": comparison,
            "threshold": threshold,
        }
        for name, unit, comparison, threshold in observable_contracts
    }
    for name in required_observables:
        record = observables.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(
                f"GPU parity artifact is missing required observable {name!r}."
            )
        contract = contract_by_name[name]
        if _require_nonempty_parity_string(record, "unit") != contract["unit"]:
            raise ValueError(
                f"GPU parity observable {name!r} unit does not match the model card."
            )
        comparison = _normalized_label(
            _require_nonempty_parity_string(record, "comparison")
        )
        if comparison != contract["comparison"]:
            raise ValueError(
                f"GPU parity observable {name!r} comparison does not match the "
                "model card."
            )
        threshold = record.get("threshold")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or float(threshold) < 0.0
        ):
            raise ValueError(
                f"GPU parity observable {name!r} threshold must be a finite "
                "non-negative number."
            )
        if float(threshold) > float(contract["threshold"]):
            raise ValueError(
                f"GPU parity observable {name!r} chooses a threshold looser than "
                "the model-card contract."
            )
        paired_count, recomputed_maximum = _validate_gpu_paired_outputs(
            record,
            observable=name,
            comparison=comparison,
            manifest_output_sha256s={
                record_id: hashes[name]
                for record_id, hashes in manifest_output_sha256s.items()
            },
        )
        comparison_count = record.get("comparison_count")
        if (
            isinstance(comparison_count, bool)
            or not isinstance(comparison_count, int)
            or comparison_count <= 0
        ):
            raise ValueError(
                f"GPU parity observable {name!r} needs a positive comparison_count."
            )
        if comparison_count != manifest_observable_counts[name]:
            raise ValueError(
                f"GPU parity observable {name!r} comparison_count does not match "
                "the frozen comparison manifest."
            )
        if comparison_count != paired_count:
            raise ValueError(
                f"GPU parity observable {name!r} comparison_count does not match "
                "its hash-bound paired outputs."
            )
        observed_maximum = record.get("observed_maximum")
        if (
            isinstance(observed_maximum, bool)
            or not isinstance(observed_maximum, (int, float))
            or not math.isfinite(float(observed_maximum))
        ):
            raise ValueError(
                f"GPU parity observable {name!r} observed_maximum must be finite."
            )
        if comparison in {"absolute", "equal"} and float(observed_maximum) < 0.0:
            raise ValueError(
                f"GPU parity observable {name!r} observed_maximum cannot be "
                "negative for an absolute/equal comparison."
            )
        if float(observed_maximum) != recomputed_maximum:
            raise ValueError(
                f"GPU parity observable {name!r} observed_maximum does not match "
                "the recomputed hash-bound paired outputs."
            )
        recomputed_passed = recomputed_maximum <= float(threshold)
        if record.get("passed") is not recomputed_passed:
            raise ValueError(
                f"GPU parity observable {name!r} passed verdict does not match "
                "the recomputed comparison."
            )
        if not recomputed_passed:
            raise ValueError(
                f"GPU parity observable {name!r} exceeds its card-bounded threshold."
            )


@dataclass(frozen=True)
class GPUAccelerationPolicy:
    """Fail-closed evidence gate for CUDA/GPU execution."""

    no_loss_parity_verified: bool = False
    status: str = "unverified"
    evidence_artifact: str | None = None
    evidence_sha256: str | None = None
    compute_dtype: str | None = None
    matmul_precision: str | None = None
    comparison_manifest_artifact: str | None = None
    comparison_manifest_id: str | None = None
    comparison_manifest_sha256: str | None = None
    allowed_tasks: tuple[str, ...] = ()
    allowed_inference_modes: tuple[str, ...] = ()
    allowed_hardware: tuple[str, ...] = ()
    required_runtime_packages: tuple[str, ...] = ()
    observable_contracts: tuple[tuple[str, str, str, float], ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "GPUAccelerationPolicy":
        raw = payload.get("gpu_acceleration")
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ModelCardError("Model-card gpu_acceleration must be a mapping.")

        verified = _strict_bool(raw, "no_loss_parity_verified", default=False)
        artifact = raw.get("evidence_artifact")
        sha256 = raw.get("evidence_sha256")
        artifact = str(artifact).strip() if artifact is not None else None
        sha256 = str(sha256).strip().lower() if sha256 is not None else None
        allowed_tasks = tuple(
            _normalize_task_name(task)
            for task in _normalized_string_sequence(
                raw.get("allowed_tasks"), "gpu_acceleration.allowed_tasks"
            )
        )
        allowed_modes = _normalized_string_sequence(
            raw.get("allowed_inference_modes"),
            "gpu_acceleration.allowed_inference_modes",
        )
        allowed_hardware = _normalized_string_sequence(
            raw.get("allowed_hardware"),
            "gpu_acceleration.allowed_hardware",
        )
        runtime_packages = _normalized_string_sequence(
            raw.get("required_runtime_packages"),
            "gpu_acceleration.required_runtime_packages",
        )
        compute_dtype = _normalized_label(raw.get("compute_dtype", ""))
        matmul_precision = _normalized_label(raw.get("matmul_precision", ""))
        manifest_artifact = raw.get("comparison_manifest_artifact")
        manifest_artifact = (
            str(manifest_artifact).strip() if manifest_artifact is not None else None
        )
        manifest_id = str(raw.get("comparison_manifest_id", "")).strip() or None
        manifest_sha256 = raw.get("comparison_manifest_sha256")
        manifest_sha256 = (
            _validated_sha256(
                manifest_sha256,
                "gpu_acceleration.comparison_manifest_sha256",
            )
            if manifest_sha256 is not None
            else None
        )
        observable_contracts = _observable_contracts(raw.get("observable_contracts"))

        if verified:
            if not artifact or not sha256:
                raise ModelCardError(
                    "Verified GPU acceleration requires a parity evidence artifact "
                    "and SHA256."
                )
            artifact_path = Path(artifact)
            if artifact_path.is_absolute() or ".." in artifact_path.parts:
                raise ModelCardError(
                    "GPU parity evidence_artifact must be a repository-relative path "
                    "without parent traversal."
                )
            sha256 = _validated_sha256(
                sha256,
                "gpu_acceleration.evidence_sha256",
            )
            if not allowed_tasks:
                raise ModelCardError(
                    "Verified GPU acceleration requires explicit allowed_tasks."
                )
            if not allowed_modes:
                raise ModelCardError(
                    "Verified GPU acceleration requires explicit "
                    "allowed_inference_modes."
                )
            if compute_dtype not in GPU_ADMITTED_SCALAR_DTYPES:
                raise ModelCardError(
                    "Verified GPU acceleration requires a concrete compute_dtype "
                    f"from {sorted(GPU_ADMITTED_SCALAR_DTYPES)}."
                )
            if matmul_precision not in GPU_ADMITTED_MATMUL_PRECISIONS:
                raise ModelCardError(
                    "Verified GPU acceleration requires an explicit matmul_precision "
                    f"from {sorted(GPU_ADMITTED_MATMUL_PRECISIONS)}."
                )
            if not allowed_hardware:
                raise ModelCardError(
                    "Verified GPU acceleration requires explicit allowed_hardware."
                )
            if not runtime_packages:
                raise ModelCardError(
                    "Verified GPU acceleration requires explicit "
                    "required_runtime_packages."
                )
            if not manifest_id or not manifest_sha256:
                raise ModelCardError(
                    "Verified GPU acceleration requires an immutable comparison "
                    "manifest ID and SHA256."
                )
            if not manifest_artifact:
                raise ModelCardError(
                    "Verified GPU acceleration requires a repository-local "
                    "comparison_manifest_artifact."
                )
            manifest_path = Path(manifest_artifact)
            if manifest_path.is_absolute() or ".." in manifest_path.parts:
                raise ModelCardError(
                    "GPU comparison_manifest_artifact must be a repository-relative "
                    "path without parent traversal."
                )
            if not observable_contracts:
                raise ModelCardError(
                    "Verified GPU acceleration requires card-bound observable "
                    "contracts."
                )

        return cls(
            no_loss_parity_verified=verified,
            status=str(raw.get("status", "unverified")).strip() or "unverified",
            evidence_artifact=artifact,
            evidence_sha256=sha256,
            compute_dtype=compute_dtype or None,
            matmul_precision=matmul_precision or None,
            comparison_manifest_artifact=manifest_artifact,
            comparison_manifest_id=manifest_id,
            comparison_manifest_sha256=manifest_sha256,
            allowed_tasks=tuple(sorted(set(allowed_tasks))),
            allowed_inference_modes=allowed_modes,
            allowed_hardware=allowed_hardware,
            required_runtime_packages=runtime_packages,
            observable_contracts=observable_contracts,
        )

    def validate(
        self,
        device: object,
        *,
        model_id: str,
        model_version: str,
        checkpoint_sha256s: tuple[str, ...],
        required_observables: tuple[str, ...],
        task: str | None = None,
        inference_mode: str | None = None,
        evidence_root: Path | None = None,
        runtime_context: Mapping[str, object] | None = None,
    ) -> None:
        if is_explicit_cpu_device(device):
            return
        if task is None or not str(task).strip():
            raise ValueError(
                f"Accelerator execution for {model_id!r} requires an explicit "
                "execution task."
            )
        if not self.no_loss_parity_verified:
            raise ValueError(
                f"Accelerator execution for {model_id!r} is disabled: no frozen no-loss "
                f"CPU/GPU parity evidence is admitted ({self.status})."
            )
        if inference_mode is None or not str(inference_mode).strip():
            raise ValueError(
                f"Accelerator execution for {model_id!r} requires an explicit "
                "inference mode."
            )
        if runtime_context is None:
            raise ValueError(
                f"Accelerator execution for {model_id!r} requires a current "
                "runtime and hardware fingerprint."
            )

        root = (evidence_root or GPU_EVIDENCE_ROOT).resolve()
        artifact = (root / str(self.evidence_artifact)).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"GPU parity evidence for {model_id!r} resolves outside the "
                "admitted evidence root."
            ) from exc
        if not artifact.is_file():
            raise ValueError(
                f"GPU parity evidence for {model_id!r} is missing: {artifact}."
            )
        actual_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual_sha256 != self.evidence_sha256:
            raise ValueError(
                f"GPU parity evidence SHA256 mismatch for {model_id!r}: "
                f"expected {self.evidence_sha256}, got {actual_sha256}."
            )
        manifest = (root / str(self.comparison_manifest_artifact)).resolve()
        try:
            manifest.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"GPU comparison manifest for {model_id!r} resolves outside the "
                "admitted evidence root."
            ) from exc
        if not manifest.is_file():
            raise ValueError(
                f"GPU comparison manifest for {model_id!r} is missing: {manifest}."
            )
        manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifest_sha256 != self.comparison_manifest_sha256:
            raise ValueError(
                f"GPU comparison manifest SHA256 mismatch for {model_id!r}: "
                f"expected {self.comparison_manifest_sha256}, got {manifest_sha256}."
            )
        task_modes = tuple(
            sorted(
                f"{covered_task}:{mode}"
                for covered_task in self.allowed_tasks
                for mode in self.allowed_inference_modes
            )
        )
        (
            manifest_observable_counts,
            manifest_output_sha256s,
        ) = _validate_gpu_comparison_manifest(
            manifest,
            comparison_manifest_id=str(self.comparison_manifest_id),
            model_id=model_id,
            model_version=model_version,
            checkpoint_sha256s=checkpoint_sha256s,
            task_modes=task_modes,
            required_observables=required_observables,
        )
        _validate_gpu_parity_artifact(
            artifact,
            model_id=model_id,
            model_version=model_version,
            checkpoint_sha256s=checkpoint_sha256s,
            comparison_manifest_id=str(self.comparison_manifest_id),
            comparison_manifest_sha256=str(self.comparison_manifest_sha256),
            allowed_tasks=self.allowed_tasks,
            allowed_inference_modes=self.allowed_inference_modes,
            allowed_hardware=self.allowed_hardware,
            compute_dtype=str(self.compute_dtype),
            matmul_precision=str(self.matmul_precision),
            required_runtime_packages=self.required_runtime_packages,
            required_observables=required_observables,
            observable_contracts=self.observable_contracts,
            manifest_observable_counts=manifest_observable_counts,
            manifest_output_sha256s=manifest_output_sha256s,
            runtime_context=runtime_context,
        )

        normalized_task = _normalize_task_name(task)
        if normalized_task not in self.allowed_tasks:
            raise ValueError(
                f"GPU execution for {model_id!r} has no admitted parity evidence "
                f"for task {normalized_task!r}."
            )
        normalized_mode = _normalized_label(inference_mode)
        if (
            self.allowed_inference_modes
            and normalized_mode not in self.allowed_inference_modes
        ):
            raise ValueError(
                f"GPU inference mode {normalized_mode!r} for {model_id!r} has no "
                "admitted no-loss parity evidence."
            )


@dataclass(frozen=True)
class ModelCapabilities:
    """Capabilities of one potential or property model.

    The defaults are deliberately closed.  A task is enabled only by explicit
    card evidence; absence of a field never implies support.
    """

    energy: bool = False
    forces: bool = False
    conservative_forces: bool = False
    hessian: HessianMode = "none"
    supports_md: bool = False
    supports_pbc: bool = False
    supports_charge: bool = False
    supports_multiplicity: bool = False
    supports_multifragment: bool = False
    requires_topology: bool = False
    requires_partial_charges: bool = False
    solvation_mode: SolvationMode = "none"
    energy_reference: EnergyReference = "unknown"
    supports_absolute_solvation: bool = False
    supports_alchemical_lambda: bool = False

    @property
    def supports_numerical_hessian(self) -> bool:
        return self.hessian == "finite_difference"

    @property
    def supports_energy_derived_forces(self) -> bool:
        return self.forces and self.conservative_forces

    def gpu_required_observables(self) -> tuple[str, ...]:
        required = set(GPU_STRICT_ACCURACY_CONTRACTS)
        if self.energy:
            required.add("energy")
        if self.forces:
            required.add("forces")
        if self.hessian != "none":
            required.add("hessian")
        if self.supports_md or self.forces:
            required.add("trajectory_or_sampled_observables")
        if self.supports_pbc:
            required.add("virial")
        if self.supports_alchemical_lambda:
            required.add("lambda_derivatives")
        if self.supports_absolute_solvation:
            required.update({"final_free_energy", "uncertainty"})
        return tuple(sorted(required))

    def gpu_exposed_tasks(self) -> tuple[str, ...]:
        tasks: set[str] = set()
        if self.energy:
            tasks.add("sp")
        if self.energy and self.forces and self.conservative_forces:
            tasks.update({"opt", "scan", "ts", "irc"})
        if self.hessian != "none":
            tasks.add("frequency")
        if self.supports_md or self.forces:
            tasks.add("md")
        if self.supports_absolute_solvation:
            tasks.add("absolute_solvation_free_energy")
        if self.supports_alchemical_lambda:
            tasks.add("alchemical_free_energy")
        return tuple(sorted(tasks))

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ModelCapabilities":
        raw = payload.get("capabilities", payload)
        if not isinstance(raw, Mapping):
            raise ModelCardError("Model-card capabilities must be a mapping.")

        hessian = raw.get("hessian")
        if hessian is None:
            hessian = (
                "finite_difference"
                if _strict_bool(raw, "supports_numerical_hessian", default=False)
                else "none"
            )
        hessian = str(hessian).strip().lower()
        if hessian not in {"analytic", "autograd", "finite_difference", "none"}:
            raise ModelCardError(f"Unsupported hessian capability: {hessian!r}.")

        conservative = _strict_bool(
            raw,
            "conservative_forces",
            default=_strict_bool(raw, "supports_energy_derived_forces", default=False),
        )
        forces = _strict_bool(raw, "forces", default=conservative)
        solvation_mode = str(raw.get("solvation_mode", "none")).strip().lower()
        if solvation_mode not in {
            "none",
            "native",
            "additive",
            "alchemical",
            "property_only",
        }:
            raise ModelCardError(
                f"Unsupported solvation_mode capability: {solvation_mode!r}."
            )
        energy_reference = (
            str(raw.get("energy_reference", payload.get("energy_reference", "unknown")))
            .strip()
            .lower()
        )
        if energy_reference == "relative_or_unknown":
            energy_reference = "unknown"
        if energy_reference not in {"absolute", "relative", "unknown"}:
            raise ModelCardError(
                f"Unsupported energy_reference capability: {energy_reference!r}."
            )

        capabilities = cls(
            energy=_strict_bool(raw, "energy", default=False),
            forces=forces,
            conservative_forces=conservative,
            hessian=hessian,  # type: ignore[arg-type]
            supports_md=_strict_bool(raw, "supports_md", default=False),
            supports_pbc=_strict_bool(raw, "supports_pbc", default=False),
            supports_charge=_strict_bool(raw, "supports_charge", default=False),
            supports_multiplicity=_strict_bool(
                raw, "supports_multiplicity", default=False
            ),
            supports_multifragment=_strict_bool(
                raw, "supports_multifragment", default=False
            ),
            requires_topology=_strict_bool(raw, "requires_topology", default=False),
            requires_partial_charges=_strict_bool(
                raw, "requires_partial_charges", default=False
            ),
            solvation_mode=solvation_mode,  # type: ignore[arg-type]
            energy_reference=energy_reference,  # type: ignore[arg-type]
            supports_absolute_solvation=_strict_bool(
                raw,
                "supports_absolute_solvation",
                default=_strict_bool(
                    payload, "supports_absolute_solvation", default=False
                ),
            ),
            supports_alchemical_lambda=_strict_bool(
                raw,
                "supports_alchemical_lambda",
                default=_strict_bool(
                    payload, "supports_alchemical_lambda", default=False
                ),
            ),
        )
        capabilities._validate_consistency()
        return capabilities

    def _validate_consistency(self) -> None:
        if self.conservative_forces and not (self.energy and self.forces):
            raise ModelCardError(
                "conservative_forces=true requires both energy=true and forces=true."
            )
        if self.hessian != "none" and not (
            self.energy and self.forces and self.conservative_forces
        ):
            raise ModelCardError(
                "Hessian support requires an energy model with conservative forces."
            )

    def validate_task(self, task: str) -> None:
        task = _normalize_task_name(task)
        supported_tasks = {
            "sp",
            "opt",
            "frequency",
            "md",
            "scan",
            "ts",
            "irc",
            "absolute_solvation_free_energy",
            "alchemical_free_energy",
        }
        if task not in supported_tasks:
            raise ValueError(
                f"Unsupported or unaudited model task {task!r}; "
                "task capability validation is fail-closed."
            )
        if task == "sp" and not self.energy:
            raise ValueError("The model card does not enable energy evaluation.")
        if task in {"opt", "scan", "ts", "irc"} and not (
            self.energy and self.forces and self.conservative_forces
        ):
            raise ValueError(
                f"Task {task!r} requires energy-derived conservative forces."
            )
        if task == "frequency" and not (
            self.energy
            and self.forces
            and self.conservative_forces
            and self.hessian != "none"
        ):
            raise ValueError(
                "Frequency analysis requires a conservative energy/force model and Hessian support."
            )
        if task == "md" and not self.supports_md:
            raise ValueError("The model card does not enable molecular dynamics.")
        if (
            task == "absolute_solvation_free_energy"
            and not self.supports_absolute_solvation
        ):
            raise ValueError(
                "The model card does not enable absolute solvation free energy."
            )
        if task == "alchemical_free_energy" and not self.supports_alchemical_lambda:
            raise ValueError("The model card does not enable alchemical lambda.")


@dataclass(frozen=True)
class SolvationCapabilities:
    """Capabilities of an additive solvent backend."""

    energy: bool = True
    forces: bool = False
    conservative_forces: bool = False
    hessian: HessianMode = "none"
    supports_pbc: bool = False
    multisolvent: bool = False
    energy_reference: EnergyReference = "unknown"
    supports_absolute_solvation: bool = False

    @classmethod
    def from_backend(cls, backend) -> "SolvationCapabilities":
        supported = {
            str(item).strip().lower()
            for item in getattr(backend, "supported_properties", {"energy"})
        }
        return cls(
            energy="energy" in supported,
            forces="forces" in supported,
            conservative_forces=bool(getattr(backend, "conservative_forces", False)),
            hessian=getattr(backend, "hessian_capability", "none"),
            supports_pbc=bool(getattr(backend, "supports_pbc", False)),
            multisolvent=bool(getattr(backend, "multisolvent", False)),
            energy_reference=getattr(backend, "energy_reference", "unknown"),
            supports_absolute_solvation=bool(
                getattr(backend, "supports_absolute_solvation", False)
            ),
        )


def _normalize_task_name(task: object) -> str:
    normalized = str(task).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "single_point": "sp",
        "energy": "sp",
        "optimization": "opt",
        "geometry_optimization": "opt",
        "freq": "frequency",
        "numerical_frequency": "frequency",
        "short_md": "md",
        "molecular_dynamics": "md",
        "absolute_solvation": "absolute_solvation_free_energy",
        "alchemical": "alchemical_free_energy",
    }
    return aliases.get(normalized, normalized)


def _normalize_forbidden_tasks(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ModelCardError("Model-card forbidden_tasks must be a sequence.")
    normalized = {_normalize_task_name(item) for item in raw if str(item).strip()}
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class ModelProvenanceCard:
    """Structured model metadata used for scientific capability checks."""

    model_id: str
    version: str
    capabilities: ModelCapabilities
    payload: Mapping[str, object]
    forbidden_tasks: tuple[str, ...] = ()
    gpu_acceleration: GPUAccelerationPolicy = GPUAccelerationPolicy()
    checkpoint_sha256s: tuple[str, ...] = ()

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        model_name: str,
    ) -> "ModelProvenanceCard":
        if not isinstance(payload, Mapping):
            raise ModelCardError("Model provenance payload must be a mapping.")
        model_id = str(payload.get("model_id", model_name)).strip().lower()
        if not model_id:
            raise ModelCardError("Model card model_id cannot be empty.")
        version = str(payload.get("version", "unknown")).strip() or "unknown"
        capabilities = ModelCapabilities.from_payload(payload)
        gpu_acceleration = GPUAccelerationPolicy.from_payload(payload)
        checkpoint_sha256s = _checkpoint_sha256s(payload)
        forbidden_tasks = _normalize_forbidden_tasks(payload.get("forbidden_tasks"))
        if gpu_acceleration.no_loss_parity_verified:
            if not checkpoint_sha256s:
                raise ModelCardError(
                    "Verified GPU acceleration requires at least one pinned "
                    "checkpoint_sha256 identity."
                )
            exposed_tasks = capabilities.gpu_exposed_tasks()
            if gpu_acceleration.allowed_tasks != exposed_tasks:
                raise ModelCardError(
                    "Verified GPU acceleration must cover every mechanically exposed "
                    "task. Product-level forbidden_tasks do not waive parity coverage "
                    "because direct energy/force APIs remain reusable by external "
                    "drivers."
                )
            required_observables = capabilities.gpu_required_observables()
            contract_by_name = {
                name: (unit, comparison, threshold)
                for name, unit, comparison, threshold in (
                    gpu_acceleration.observable_contracts
                )
            }
            if set(contract_by_name) != set(required_observables):
                raise ModelCardError(
                    "Verified GPU acceleration observable contracts must exactly "
                    "cover every exposed quantity and scientific-accuracy gate."
                )
            for name, comparison in GPU_STRICT_ACCURACY_CONTRACTS.items():
                _, actual_comparison, threshold = contract_by_name[name]
                if actual_comparison != comparison or threshold != 0.0:
                    raise ModelCardError(
                        f"GPU accuracy contract {name!r} must use comparison "
                        f"{comparison!r} with zero degradation threshold."
                    )
        return cls(
            model_id=model_id,
            version=version,
            capabilities=capabilities,
            payload=dict(payload),
            forbidden_tasks=forbidden_tasks,
            gpu_acceleration=gpu_acceleration,
            checkpoint_sha256s=checkpoint_sha256s,
        )

    def validate_task(self, task: str) -> None:
        normalized = _normalize_task_name(task)
        if normalized in self.forbidden_tasks:
            raise ValueError(
                f"Model card {self.model_id!r} explicitly forbids task {normalized!r}."
            )
        self.capabilities.validate_task(normalized)

    def validate_device(
        self,
        device: object,
        *,
        task: str | None = None,
        inference_mode: str | None = None,
        evidence_root: Path | None = None,
        runtime_context: Mapping[str, object] | None = None,
    ) -> None:
        self.gpu_acceleration.validate(
            device,
            model_id=self.model_id,
            model_version=self.version,
            checkpoint_sha256s=self.checkpoint_sha256s,
            required_observables=self.capabilities.gpu_required_observables(),
            task=task,
            inference_mode=inference_mode,
            evidence_root=evidence_root,
            runtime_context=runtime_context,
        )


def read_json_compatible_yaml(path: Path) -> Mapping[str, object]:
    """Load JSON syntax from a .yaml/.yml/.json card without adding PyYAML."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelCardError(f"Model card not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelCardError(
            f"Model card at {path} must use JSON-compatible YAML syntax."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ModelCardError(f"Model card payload at {path} must be a JSON object.")
    return dict(payload)


def _normalize_model_name(model_name: str) -> str:
    return str(model_name).strip().lower().replace("_", "-")


def load_model_provenance_card(
    model_name: str,
    model_card_root: Path,
) -> ModelProvenanceCard:
    normalized = _normalize_model_name(model_name)
    for extension in MODEL_CARD_EXTENSIONS:
        card_path = model_card_root / f"{normalized}{extension}"
        if card_path.is_file():
            return ModelProvenanceCard.from_payload(
                read_json_compatible_yaml(card_path),
                model_name=normalized,
            )
    raise ModelCardError(
        f"Model provenance card for model {model_name!r} was not found in {model_card_root}."
    )


@dataclass(frozen=True)
class SolventCombinationProfile:
    """Metadata emitted after a legal potential/solvent composition."""

    frequency_type: str | None = None


class CombinationValidator:
    """Validate potential, solvent, derivative, and free-energy task boundaries."""

    def __init__(
        self,
        *,
        model_capabilities: ModelCapabilities,
        implicit: str = "none",
        hessian_mode: str | None = None,
        solvation_capabilities: SolvationCapabilities | None = None,
        task: str | None = None,
    ) -> None:
        self.model_capabilities = model_capabilities
        self.implicit = str(implicit).strip().lower()
        self.hessian_mode = (
            hessian_mode.lower() if isinstance(hessian_mode, str) else None
        )
        self.solvation_capabilities = solvation_capabilities
        self.task = str(task).strip().lower() if task is not None else None

    def validate(self) -> SolventCombinationProfile | None:
        if self.task is not None:
            self.model_capabilities.validate_task(self.task)

        additive_requested = self.implicit not in {"", "none"}
        if additive_requested and self.model_capabilities.solvation_mode == "native":
            raise ValueError(
                "Native solution-phase potentials cannot be combined with an additive solvent backend."
            )
        if self.model_capabilities.solvation_mode == "property_only" and self.task:
            raise ValueError(
                "Property-only solvation predictors are benchmark baselines, not calculator backends."
            )

        if self.hessian_mode != "numerical" or not additive_requested:
            return None
        if self.solvation_capabilities is None:
            raise ValueError(
                "Numerical solvent Hessian requires explicit solvent-backend capabilities."
            )
        if not (
            self.model_capabilities.energy
            and self.model_capabilities.forces
            and self.model_capabilities.conservative_forces
            and self.model_capabilities.hessian == "finite_difference"
        ):
            raise ValueError(
                "Numerical solvent Hessian requires a conservative base model with finite-difference Hessian support."
            )
        if not (
            self.solvation_capabilities.energy
            and self.solvation_capabilities.forces
            and self.solvation_capabilities.conservative_forces
        ):
            raise ValueError(
                "Numerical solvent Hessian requires an energy-consistent solvent force backend."
            )
        return SolventCombinationProfile(
            frequency_type=CONSERVATIVE_NUMERICAL_FREQUENCY_TYPE
        )


def validate_model_card_task(card: ModelProvenanceCard, task: str) -> None:
    card.validate_task(task)
