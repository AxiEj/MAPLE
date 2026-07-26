from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from .bulk_water import (
    BulkWaterValidationError,
    _array_sha256,
    _require_sha256,
)
from .bulk_water_npt import BulkWaterNPTConfig
from .bulk_water_npt import NPT_IMPLEMENTATION_PATHS
from .bulk_water_evidence import (
    recompute_bulk_water_npt_evidence,
    require_bulk_water_npt_summary_consistency,
)
from .protocol import canonical_sha256, raw_sha256
from .provenance import require_coherent_git_status


PREREGISTERED_NPT_CONFIG_WITHOUT_SEED_SHA256 = (
    "59c39954c8df951fae3189dae2b483c78f347ebc2acc3937bd724263360136bf"
)
_IMPLEMENTATION_KEYS = {
    "schema",
    "project_root",
    "git_head",
    "git_dirty",
    "git_status_sha256",
    "implementation_file_sha256",
}
_CORE_TRAJECTORY_ARRAYS = {
    "production_positions_angstrom",
    "production_cells_angstrom",
    "production_velocities_angstrom_per_ase_time",
}
_ENGINEERING_CHECKS = {
    "finite_observations",
    "all_md_steps_checked",
    "temperature_below_emergency_limit",
    "force_below_emergency_limit",
    "water_topology_preserved",
    "cell_cutoff_safe",
}
_CAMPAIGN_PROTOCOL_LIMITS = {
    "temperature_relative_tolerance": 0.05,
    "pressure_mean_tolerance_bar": 500.0,
    "density_half_drift_limit": 0.01,
    "density_relative_error_limit": 0.03,
    "maximum_temperature_k": 1_000.0,
    "maximum_force_ev_per_angstrom": 50.0,
}


@dataclass(frozen=True)
class BulkWaterCampaignConfig:
    """Acceptance contract for independent NPT density replicas."""

    required_replicas: int = 3
    density_relative_error_limit: float = 0.03
    replica_density_spread_limit: float = 0.02

    def __post_init__(self) -> None:
        if type(self.required_replicas) is not int:
            raise BulkWaterValidationError(
                "required_replicas must be an integer."
            )
        if self.required_replicas < 3:
            raise BulkWaterValidationError(
                "required_replicas must be at least three."
            )
        for name in (
            "density_relative_error_limit",
            "replica_density_spread_limit",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise BulkWaterValidationError(
                    f"{name} must be finite and positive."
                )
        if self.density_relative_error_limit > 0.03:
            raise BulkWaterValidationError(
                "density_relative_error_limit cannot exceed the "
                "preregistered 3% maximum."
            )
        if self.replica_density_spread_limit > 0.02:
            raise BulkWaterValidationError(
                "replica_density_spread_limit cannot exceed the "
                "preregistered 2% maximum."
            )

    @property
    def content_hash(self) -> str:
        return canonical_sha256(
            {
                "schema": "maple-route-a-bulk-water-campaign-config-v1",
                **asdict(self),
            }
        )


def load_bulk_water_npt_artifact(
    artifact_directory: str | Path,
) -> dict[str, Any]:
    """Load and verify every retained component of one NPT artifact."""

    root = Path(artifact_directory).expanduser().resolve()
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.json"
    arrays_path = root / "arrays.npz"
    if not all(
        path.is_file()
        for path in (manifest_path, summary_path, arrays_path)
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_ARTIFACT_MISSING: each replica requires manifest.json, "
            "summary.json and arrays.npz."
        )
    try:
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BulkWaterValidationError(
            "CAMPAIGN_ARTIFACT_INVALID: JSON evidence cannot be parsed."
        ) from exc
    if (
        manifest.get("schema")
        != "maple-route-a-bulk-water-npt-artifact-v1"
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_ARTIFACT_INVALID: unsupported NPT manifest schema."
        )
    if (
        summary.get("schema")
        != "maple-route-a-bulk-water-npt-summary-v1"
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_ARTIFACT_INVALID: unsupported NPT summary schema."
        )
    declared_summary_hash = _nested(
        manifest,
        "files",
        "summary",
        "canonical_sha256",
    )
    if canonical_sha256(summary) != declared_summary_hash:
        raise BulkWaterValidationError(
            "CAMPAIGN_ARTIFACT_HASH_MISMATCH: summary hash changed."
        )
    declared_arrays_hash = _nested(
        manifest,
        "files",
        "arrays",
        "sha256",
    )
    if raw_sha256(arrays_path) != declared_arrays_hash:
        raise BulkWaterValidationError(
            "CAMPAIGN_ARTIFACT_HASH_MISMATCH: arrays byte hash changed."
        )
    result_hash = summary.get("result_hash")
    summary_preimage = dict(summary)
    summary_preimage.pop("result_hash", None)
    if (
        not isinstance(result_hash, str)
        or canonical_sha256(summary_preimage) != result_hash
        or manifest.get("result_hash") != result_hash
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_ARTIFACT_HASH_MISMATCH: result hash changed."
        )
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
            semantic_hashes = {
                name: _array_sha256(arrays[name])
                for name in sorted(arrays)
            }
    except (OSError, ValueError) as exc:
        raise BulkWaterValidationError(
            "CAMPAIGN_ARTIFACT_INVALID: arrays archive cannot be read."
        ) from exc
    if (
        semantic_hashes != manifest.get("semantic_array_sha256")
        or semantic_hashes
        != _nested(
            summary,
            "trajectory",
            "semantic_array_sha256",
        )
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_ARTIFACT_HASH_MISMATCH: semantic array hashes changed."
        )
    config = _validated_config(summary, index=0)
    identity = _identity(summary)
    model_cutoff = float(
        _nested(
            summary,
            "calculator",
            "calculator",
            "interaction_cutoff_angstrom",
        )
    )
    if not math.isclose(
        model_cutoff,
        config.interaction_cutoff_angstrom,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_CALCULATOR_INVALID: config cutoff differs from "
            "calculator provenance."
        )
    recomputed = recompute_bulk_water_npt_evidence(
        arrays,
        config=config,
        water_count=identity["water_count"],
    )
    require_bulk_water_npt_summary_consistency(summary, recomputed)
    return summary


def _required_mapping(
    value: Any,
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BulkWaterValidationError(
            f"CAMPAIGN_ARTIFACT_INVALID: {label} must be a mapping."
        )
    return value


def _nested(
    value: Mapping[str, Any],
    *keys: str,
) -> Any:
    current: Any = value
    for key in keys:
        current = _required_mapping(
            current,
            label=".".join(keys),
        ).get(key)
    return current


def _identity(
    replica: Mapping[str, Any],
) -> dict[str, Any]:
    implementation = _required_mapping(
        replica.get("implementation"),
        label="implementation",
    )
    if set(implementation) != _IMPLEMENTATION_KEYS:
        raise BulkWaterValidationError(
            "CAMPAIGN_IMPLEMENTATION_INVALID: implementation evidence must "
            "use the exact Route A provenance schema."
        )
    if (
        implementation.get("schema")
        != "maple-route-a-bulk-water-implementation-v1"
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_IMPLEMENTATION_INVALID: implementation schema differs."
        )
    project_root = implementation.get("project_root")
    if (
        not isinstance(project_root, str)
        or not project_root
        or not Path(project_root).is_absolute()
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_IMPLEMENTATION_INVALID: project_root must be absolute."
        )
    git_head = str(implementation.get("git_head", ""))
    if len(git_head) not in {40, 64} or any(
        character not in "0123456789abcdef"
        for character in git_head
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_IMPLEMENTATION_INVALID: Git HEAD is invalid."
        )
    if type(implementation.get("git_dirty")) is not bool:
        raise BulkWaterValidationError(
            "CAMPAIGN_IMPLEMENTATION_INVALID: Git dirty state must be boolean."
        )
    git_status_hash = _require_sha256(
        implementation.get("git_status_sha256", ""),
        label="implementation Git status SHA256",
    )
    try:
        require_coherent_git_status(
            implementation["git_dirty"],
            git_status_hash,
        )
    except ValueError as exc:
        raise BulkWaterValidationError(
            "CAMPAIGN_IMPLEMENTATION_INVALID: "
            f"{exc}"
        ) from exc
    file_hashes = _required_mapping(
        implementation.get("implementation_file_sha256"),
        label="implementation.implementation_file_sha256",
    )
    if set(file_hashes) != NPT_IMPLEMENTATION_PATHS:
        raise BulkWaterValidationError(
            "CAMPAIGN_IMPLEMENTATION_INVALID: implementation file hashes "
            "do not exactly cover the NPT implementation surface."
        )
    normalized_file_hashes = {
        str(path): _require_sha256(
            value,
            label=f"implementation hash for {path}",
        )
        for path, value in sorted(file_hashes.items())
    }
    implementation_identity = canonical_sha256(
        {
            "schema": implementation["schema"],
            "git_head": git_head,
            "git_dirty": implementation["git_dirty"],
            "git_status_sha256": git_status_hash,
            "implementation_file_sha256": normalized_file_hashes,
        }
    )
    semantic_hashes = _required_mapping(
        _nested(
            replica,
            "trajectory",
            "semantic_array_sha256",
        ),
        label="trajectory.semantic_array_sha256",
    )
    missing_core_arrays = _CORE_TRAJECTORY_ARRAYS - semantic_hashes.keys()
    if missing_core_arrays:
        raise BulkWaterValidationError(
            "CAMPAIGN_TRAJECTORY_INVALID: core trajectory hashes are "
            "missing: "
            + ", ".join(sorted(missing_core_arrays))
            + "."
        )
    core_trajectory_hash = canonical_sha256(
        {
            name: _require_sha256(
                semantic_hashes[name],
                label=f"trajectory hash for {name}",
            )
            for name in sorted(_CORE_TRAJECTORY_ARRAYS)
        }
    )
    config_without_seed = _require_sha256(
        _nested(
            replica,
            "config",
            "content_hash_without_seed",
        ),
        label="config without-seed hash",
    )
    source_hash = _require_sha256(
        _nested(replica, "source", "sha256"),
        label="source SHA256",
    )
    checkpoint_hash = _require_sha256(
        _nested(
            replica,
            "calculator",
            "calculator",
            "checkpoint_sha256",
        ),
        label="checkpoint SHA256",
    )
    numerical_precision = _nested(
        replica,
        "calculator",
        "calculator",
        "default_dtype",
    )
    if (
        not isinstance(numerical_precision, str)
        or numerical_precision not in {"float32", "float64"}
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_CALCULATOR_INVALID: numerical precision must be "
            "float32 or float64."
        )
    water_count = _nested(
        replica,
        "source",
        "topology",
        "water_count",
    )
    if type(water_count) is not int or water_count <= 0:
        raise BulkWaterValidationError(
            "CAMPAIGN_SOURCE_INVALID: water count must be a positive integer."
        )
    return {
        "config_without_seed": config_without_seed,
        "source_sha256": source_hash,
        "water_count": water_count,
        "checkpoint_sha256": checkpoint_hash,
        "numerical_precision": numerical_precision,
        "implementation": implementation_identity,
        "git_dirty": implementation["git_dirty"],
        "core_trajectory": core_trajectory_hash,
        "ensemble": _nested(replica, "trajectory", "ensemble"),
        "integrator": _nested(replica, "trajectory", "integrator"),
    }


def _validated_config(
    replica: Mapping[str, Any],
    *,
    index: int,
) -> BulkWaterNPTConfig:
    raw_config = _required_mapping(
        replica.get("config"),
        label=f"replica[{index}].config",
    )
    field_names = {field.name for field in fields(BulkWaterNPTConfig)}
    metadata_names = {
        "schema",
        "content_hash",
        "content_hash_without_seed",
    }
    if set(raw_config) != field_names | metadata_names:
        raise BulkWaterValidationError(
            "CAMPAIGN_CONFIG_INVALID: replica "
            f"{index} has an incomplete or unexpected NPT config."
        )
    if raw_config.get("schema") != (
        "maple-route-a-bulk-water-npt-config-v1"
    ):
        raise BulkWaterValidationError(
            f"CAMPAIGN_CONFIG_INVALID: replica {index} config schema differs."
        )
    try:
        config = BulkWaterNPTConfig(
            **{
                name: raw_config[name]
                for name in sorted(field_names)
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BulkWaterValidationError(
            f"CAMPAIGN_CONFIG_INVALID: replica {index} config is invalid."
        ) from exc
    if (
        raw_config.get("content_hash") != config.content_hash
        or raw_config.get("content_hash_without_seed")
        != config.content_hash_without_seed
    ):
        raise BulkWaterValidationError(
            f"CAMPAIGN_CONFIG_INVALID: replica {index} config hash changed."
        )
    return config


def _validated_replica_gate_evidence(
    replica: Mapping[str, Any],
    *,
    config: BulkWaterNPTConfig,
    index: int,
) -> tuple[bool, bool, bool]:
    trajectory = _required_mapping(
        replica.get("trajectory"),
        label=f"replica[{index}].trajectory",
    )
    diagnostics = _required_mapping(
        replica.get("diagnostics"),
        label=f"replica[{index}].diagnostics",
    )
    gates = _required_mapping(
        replica.get("gates"),
        label=f"replica[{index}].gates",
    )
    try:
        duration = float(trajectory["production_duration_ps"])
        frame_count = int(trajectory["production_frame_count"])
        temperature_mean = float(
            diagnostics["production_temperature_mean_k"]
        )
        pressure_mean = float(
            diagnostics["production_pressure_mean_bar"]
        )
        density_mean = float(
            diagnostics["production_density_mean_g_per_ml"]
        )
        density_drift = float(
            diagnostics["production_density_half_relative_drift"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BulkWaterValidationError(
            "CAMPAIGN_NUMERIC_INVALID: replica "
            f"{index} diagnostic evidence is incomplete."
        ) from exc
    numeric_values = (
        duration,
        temperature_mean,
        pressure_mean,
        density_mean,
        density_drift,
    )
    if (
        not all(math.isfinite(value) for value in numeric_values)
        or frame_count <= 0
        or density_mean <= 0.0
        or density_drift < 0.0
        or not math.isclose(
            duration,
            config.production_duration_ps,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or frame_count != config.production_frame_count
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_NUMERIC_INVALID: replica "
            f"{index} trajectory evidence conflicts with its config."
        )

    engineering_checks = _required_mapping(
        gates.get("engineering_checks"),
        label=f"replica[{index}].gates.engineering_checks",
    )
    if set(engineering_checks) != _ENGINEERING_CHECKS or not all(
        type(value) is bool
        for value in engineering_checks.values()
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_GATE_MISMATCH: replica "
            f"{index} engineering checks must exactly match the "
            "preregistered boolean gate schema."
        )
    engineering_passed = all(engineering_checks.values())
    if gates.get("engineering_stability_passed") is not engineering_passed:
        raise BulkWaterValidationError(
            "CAMPAIGN_GATE_MISMATCH: replica "
            f"{index} engineering gate is internally inconsistent."
        )

    density_reference = config.density_reference_g_per_ml
    temperature_relative_error = abs(
        temperature_mean - config.temperature_k
    ) / config.temperature_k
    density_relative_error = abs(
        density_mean - density_reference
    ) / density_reference
    diagnostic_passed = bool(
        engineering_passed
        and duration >= config.minimum_diagnostic_duration_ps
        and frame_count >= config.minimum_diagnostic_frames
        and temperature_relative_error
        <= config.temperature_relative_tolerance
        and density_relative_error
        <= config.density_relative_error_limit
        and density_drift <= config.density_half_drift_limit
        and abs(pressure_mean - config.pressure_bar)
        <= config.pressure_mean_tolerance_bar
    )
    if (
        gates.get("minimum_npt_diagnostic_eligible")
        is not diagnostic_passed
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_GATE_MISMATCH: replica "
            f"{index} diagnostic gate is not reproduced by numeric evidence."
        )

    paper_duration_passed = config.paper_duration_fidelity
    if (
        gates.get("paper_duration_fidelity_passed")
        is not paper_duration_passed
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_GATE_MISMATCH: replica "
            f"{index} paper-duration gate conflicts with its config."
        )
    return engineering_passed, diagnostic_passed, paper_duration_passed


def _validate_matching_identities(
    identities: Sequence[Mapping[str, Any]],
) -> None:
    reference = identities[0]
    labels = {
        "config_without_seed": "non-seed protocol",
        "source_sha256": "source",
        "water_count": "water count",
        "checkpoint_sha256": "checkpoint",
        "numerical_precision": "numerical precision",
        "implementation": "implementation",
        "ensemble": "ensemble",
        "integrator": "integrator",
    }
    for index, identity in enumerate(identities[1:], start=1):
        for key, label in labels.items():
            if identity.get(key) != reference.get(key):
                raise BulkWaterValidationError(
                    "CAMPAIGN_IDENTITY_MISMATCH: "
                    f"replica {index} has a different {label} identity."
                )


def _validate_campaign_protocol(
    config: BulkWaterNPTConfig,
    *,
    index: int,
) -> None:
    if (
        config.content_hash_without_seed
        != PREREGISTERED_NPT_CONFIG_WITHOUT_SEED_SHA256
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_PROTOCOL_MISMATCH: replica "
            f"{index} does not match the preregistered non-seed NPT protocol."
        )
    for name, maximum in _CAMPAIGN_PROTOCOL_LIMITS.items():
        if float(getattr(config, name)) > maximum + 1.0e-12:
            raise BulkWaterValidationError(
                "CAMPAIGN_PROTOCOL_WEAKENED: replica "
                f"{index} declares {name}={getattr(config, name)!r}, above "
                f"the campaign maximum {maximum!r}."
            )
    if not config.paper_duration_fidelity:
        raise BulkWaterValidationError(
            "CAMPAIGN_PROTOCOL_MISMATCH: replica "
            f"{index} lacks paper-duration fidelity."
        )


def _evaluate_bulk_water_replica_summaries(
    replicas: Sequence[Mapping[str, Any]],
    *,
    config: BulkWaterCampaignConfig,
) -> dict[str, Any]:
    """Aggregate already verified summaries without pooling correlated frames.

    This private statistical helper exists so its arithmetic can be tested
    without fabricating filesystem artifacts. Public callers must use
    :func:`evaluate_bulk_water_replica_campaign`, which verifies each complete
    artifact before reaching this function.
    """

    records = [dict(replica) for replica in replicas]
    if len(records) < config.required_replicas:
        raise BulkWaterValidationError(
            "CAMPAIGN_REPLICA_COUNT: at least "
            f"{config.required_replicas} replicas are required."
        )
    for index, record in enumerate(records):
        if (
            record.get("schema")
            != "maple-route-a-bulk-water-npt-summary-v1"
        ):
            raise BulkWaterValidationError(
                "CAMPAIGN_ARTIFACT_INVALID: replica "
                f"{index} is not a Route A NPT summary."
            )
    configs = [
        _validated_config(record, index=index)
        for index, record in enumerate(records)
    ]
    for index, replica_config in enumerate(configs):
        _validate_campaign_protocol(
            replica_config,
            index=index,
        )
    seeds = [config.seed for config in configs]
    if len(set(seeds)) != len(seeds):
        raise BulkWaterValidationError(
            "CAMPAIGN_SEED_REUSE: independent replicas require distinct seeds."
        )
    result_hashes = [
        _require_sha256(
            record.get("result_hash", ""),
            label=f"replica {index} result_hash",
        )
        for index, record in enumerate(records)
    ]
    for index, (record, result_hash) in enumerate(
        zip(records, result_hashes, strict=True)
    ):
        preimage = dict(record)
        preimage.pop("result_hash", None)
        if canonical_sha256(preimage) != result_hash:
            raise BulkWaterValidationError(
                "CAMPAIGN_ARTIFACT_HASH_MISMATCH: replica "
                f"{index} result hash does not bind its summary."
            )
    if len(set(result_hashes)) != len(result_hashes):
        raise BulkWaterValidationError(
            "CAMPAIGN_ARTIFACT_REUSE: replicas require distinct result hashes."
        )
    identities = [_identity(record) for record in records]
    _validate_matching_identities(identities)
    if any(identity["git_dirty"] for identity in identities):
        raise BulkWaterValidationError(
            "CAMPAIGN_IMPLEMENTATION_INVALID: formal replicas require a "
            "clean committed implementation."
        )
    core_trajectory_hashes = [
        identity["core_trajectory"] for identity in identities
    ]
    if len(set(core_trajectory_hashes)) != len(core_trajectory_hashes):
        raise BulkWaterValidationError(
            "CAMPAIGN_TRAJECTORY_REUSE: independent replicas require "
            "distinct core trajectory hashes."
        )
    if identities[0]["ensemble"] != "NPT":
        raise BulkWaterValidationError(
            "CAMPAIGN_ENSEMBLE_INVALID: density replicas must be NPT."
        )
    gate_evidence = [
        _validated_replica_gate_evidence(
            record,
            config=replica_config,
            index=index,
        )
        for index, (record, replica_config) in enumerate(
            zip(records, configs, strict=True)
        )
    ]

    densities = np.asarray(
        [
            float(
                _nested(
                    record,
                    "diagnostics",
                    "production_density_mean_g_per_ml",
                )
            )
            for record in records
        ],
        dtype=float,
    )
    within_replica_sem = np.asarray(
        [
            float(
                _nested(
                    record,
                    "diagnostics",
                    "production_density_block_sem_g_per_ml",
                )
            )
            for record in records
        ],
        dtype=float,
    )
    reference_values = np.asarray(
        [
            float(
                _nested(
                    record,
                    "config",
                    "density_reference_g_per_ml",
                )
            )
            for record in records
        ],
        dtype=float,
    )
    if (
        not np.all(np.isfinite(densities))
        or np.any(densities <= 0.0)
        or not np.all(np.isfinite(within_replica_sem))
        or np.any(within_replica_sem < 0.0)
        or not np.all(np.isfinite(reference_values))
        or np.any(reference_values <= 0.0)
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_NUMERIC_INVALID: density evidence must be finite and "
            "uncertainties must be non-negative."
        )
    if not np.allclose(
        reference_values,
        reference_values[0],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_REFERENCE_MISMATCH: replicas use different density "
            "references."
        )
    density_reference = float(reference_values[0])
    density_mean = float(np.mean(densities))
    between_replica_sem = float(
        np.std(densities, ddof=1) / math.sqrt(len(densities))
    )
    within_mean_sem = float(
        math.sqrt(float(np.sum(within_replica_sem**2)))
        / len(densities)
    )
    conservative_sem = math.hypot(
        between_replica_sem,
        within_mean_sem,
    )
    density_relative_error = float(
        abs(density_mean - density_reference) / density_reference
    )
    replica_relative_spread = float(
        (np.max(densities) - np.min(densities)) / density_mean
    )
    interval_degrees_of_freedom = len(densities) - 1
    interval_critical_value = float(
        student_t.ppf(0.975, interval_degrees_of_freedom)
    )
    density_interval = [
        density_mean - interval_critical_value * conservative_sem,
        density_mean + interval_critical_value * conservative_sem,
    ]
    density_acceptance_interval = [
        density_reference * (1.0 - config.density_relative_error_limit),
        density_reference * (1.0 + config.density_relative_error_limit),
    ]
    density_interval_within_acceptance = bool(
        density_interval[0] >= density_acceptance_interval[0]
        and density_interval[1] <= density_acceptance_interval[1]
    )

    engineering_passed = all(item[0] for item in gate_evidence)
    diagnostic_passed = all(item[1] for item in gate_evidence)
    paper_duration_passed = all(item[2] for item in gate_evidence)
    independent_replicas_passed = bool(
        engineering_passed
        and diagnostic_passed
        and paper_duration_passed
    )
    density_accuracy_passed = bool(
        independent_replicas_passed
        and density_interval_within_acceptance
        and replica_relative_spread
        <= config.replica_density_spread_limit
    )
    gates = {
        "engineering_replicas_passed": engineering_passed,
        "diagnostic_replicas_passed": diagnostic_passed,
        "paper_duration_replicas_passed": paper_duration_passed,
        "independent_replicas_passed": independent_replicas_passed,
        "npt_density_accuracy_passed": density_accuracy_passed,
        "finite_size_validation_passed": False,
        "external_rdf_validation_passed": False,
        "cross_engine_validation_passed": False,
        "hamiltonian_freeze_eligible": False,
        "route_a_accuracy_claim_allowed": False,
    }
    summary = {
        "schema": "maple-route-a-bulk-water-campaign-summary-v1",
        "config": {
            **asdict(config),
            "content_hash": config.content_hash,
        },
        "replica_count": len(records),
        "replica_result_hashes": result_hashes,
        "seeds": seeds,
        "identity": identities[0],
        "diagnostics": {
            "density_reference_g_per_ml": density_reference,
            "replica_density_means_g_per_ml": densities.tolist(),
            "replica_density_block_sem_g_per_ml": (
                within_replica_sem.tolist()
            ),
            "density_mean_g_per_ml": density_mean,
            "density_between_replica_sem_g_per_ml": (
                between_replica_sem
            ),
            "density_within_mean_sem_g_per_ml": within_mean_sem,
            "density_conservative_sem_g_per_ml": conservative_sem,
            "density_interval_degrees_of_freedom": (
                interval_degrees_of_freedom
            ),
            "density_student_t_95_critical_value": (
                interval_critical_value
            ),
            "density_student_t_95_interval_g_per_ml": density_interval,
            "density_acceptance_interval_g_per_ml": (
                density_acceptance_interval
            ),
            "density_interval_within_acceptance": (
                density_interval_within_acceptance
            ),
            "density_relative_error": density_relative_error,
            "replica_density_relative_spread": (
                replica_relative_spread
            ),
        },
        "gates": gates,
        "interpretation": (
            "A passing replica-density campaign remains insufficient to "
            "freeze the Route A water Hamiltonian. Finite-size, external RDF "
            "and independent OpenMM/Monte-Carlo-barostat cross-engine gates "
            "remain explicit and false."
        ),
    }
    summary["campaign_hash"] = canonical_sha256(summary)
    return summary


def evaluate_bulk_water_replica_campaign(
    artifact_directories: Sequence[str | Path],
    *,
    config: BulkWaterCampaignConfig,
) -> dict[str, Any]:
    """Verify artifact directories, then aggregate independent replicas."""

    if isinstance(artifact_directories, (str, bytes, PathLike)):
        raise BulkWaterValidationError(
            "CAMPAIGN_ARTIFACT_INVALID: artifact directories must be "
            "provided as a sequence."
        )
    directories = list(artifact_directories)
    if any(
        not isinstance(directory, (str, PathLike))
        for directory in directories
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_ARTIFACT_INVALID: the public campaign evaluator "
            "accepts artifact directories, not summary mappings."
        )
    summaries = [
        load_bulk_water_npt_artifact(directory)
        for directory in directories
    ]
    return _evaluate_bulk_water_replica_summaries(
        summaries,
        config=config,
    )
