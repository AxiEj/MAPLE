#!/usr/bin/env python3
"""Aggregate complete MNSol partition shards into a two-member matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, BENCHMARK_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from benchmark_core import canonical_json_bytes, sha256_file, write_json_atomic
from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
from mnsol_partition import (
    PARTITION_ARTIFACT,
    validate_frozen_mnsol_partition_selection,
)
from mnsol_response_ablation import (
    aggregate_method_metrics,
    paired_method_comparison,
)
from maple.function.route2_solvents import route2_solvent_spec
import run_mnsol_macepolar_response_ablation as runner

ALLOWED_METHODS = ("mace_fixed_l1", "mace_scf_l1")
FULL_METHODS = tuple(runner.ABLATION_METHODS)
SUPPORTED_PARTITIONS = frozenset({"development"})
AGGREGATOR_ARTIFACT_NAME = "route2-mnsol-macepolar-two-member-matrix-v2"
SOURCE_RUN_KIND = "partition-record-shard"
AGGREGATE_RUN_KIND = "partition-two-member-matrix"
REQUIRED_STAGE = "scf"
SCHEMA_VERSION = 2
REQUIRED_CONTINUUM_EQUATION = "ddpcm"
SOURCE_RUNNER_PATH = (
    "docs/implicit-solvation/benchmarks/run_mnsol_macepolar_response_ablation.py"
)
EXPECTED_MACE_POLAR_CHECKPOINT = {
    "identifier": "polar-1-m",
    "release_url": (
        "https://github.com/ACEsuit/mace-foundations/releases/download/"
        "mace_polar_1/MACE-POLAR-1-M.model"
    ),
    "sha256": "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a",
    "size_bytes": 68133235,
}
EXPECTED_MACE_TORCH_VERSION = "0.3.16"
EXPECTED_GRAPH_LONGRANGE_VERSION = "0.4.0"
EXPECTED_TORCH_VERSION = "2.12.0+cu130"
EXPECTED_MACE_LONG_RANGE_EVALUATOR_PROFILE = (
    "graph-longrange-molecular-realspace-v1"
)
EXPECTED_PYDDX_VERSION = "0.8.0"
EXPECTED_RUNTIME_DEVICE = "cpu"
EXPECTED_MACE_DTYPE = "torch.float64"
EXPECTED_SCF_RUNTIME_THREADS = 1
EXPECTED_SOLVER_TOLERANCE = 1.0e-12
EXPECTED_MACE_LONG_RANGE_LMAX = 15
EXPECTED_MACE_LONG_RANGE_N_LEBEDEV = 1202
EXPECTED_MACE_LONG_RANGE_ETA = 0.1

DATASET_HASH_FIELDS = (
    "source_artifact_sha256",
    "table_sha256",
    "normalized_bundle_sha256",
)
METHOD_FIELDS = (
    "total_solvation_kcal_mol",
    "signed_error_kcal_mol",
    "absolute_error_kcal_mol",
    "wall_seconds",
    "solute_polarization_kcal_mol",
    "continuum_polarization_kcal_mol",
    "electrostatic_kcal_mol",
    "smd_cds_kcal_mol",
)
MACE_SCF_NOMINAL_REASON = "nominal-density-and-energy-v1"
MACE_SCF_FINITE_RESOLUTION_REASON = "finite-resolution-stagnation-v1"
MACE_SCF_CONVERGENCE_REASONS = frozenset(
    {MACE_SCF_NOMINAL_REASON, MACE_SCF_FINITE_RESOLUTION_REASON}
)
MACE_SCF_NOMINAL_MONOPOLE_TOLERANCE_E = 2.0e-12
MACE_SCF_NOMINAL_DIPOLE_TOLERANCE_E_ANGSTROM = 2.0e-12
MACE_SCF_FINITE_RESOLUTION_MONOPOLE_CEILING_E = 1.0e-10
MACE_SCF_FINITE_RESOLUTION_DIPOLE_CEILING_E_ANGSTROM = 1.0e-10
MACE_SCF_FINITE_RESOLUTION_FIELD_SPAN_TOLERANCE = 1.0e-10
MACE_SCF_FINITE_RESOLUTION_ENERGY_TOLERANCE_EV = 1.0e-10
MACE_SCF_POLARIZATION_IDENTITY_TOLERANCE_EV = 2.0e-10
MACE_SCF_CONVERGENCE_HISTORY_FIELDS = (
    "start_iteration",
    "end_iteration",
    "root_monopole_span_e",
    "root_dipole_span_e_angstrom",
    "residual_monopole_span_e",
    "residual_dipole_span_e_angstrom",
    "potential_span_ev",
    "gradient_span_ev_per_angstrom",
    "maximum_monopole_residual_e",
    "maximum_dipole_residual_e_angstrom",
    "maximum_energy_delta_ev",
    "intrinsic_energy_span_ev",
)
MACE_SCF_CONVERGENCE_MAP_REPLAY_FIELDS = (
    "replay_count",
    "evaluation_count",
    "includes_online_candidate",
    "all_field_arrays_identical",
    "all_response_arrays_identical",
    "field_sha256",
    "response_sha256",
    "maximum_monopole_residual_e",
    "maximum_dipole_residual_e_angstrom",
    "intrinsic_ledger_span_ev",
    "pcm_ledger_span_ev",
    "electrostatic_ledger_span_ev",
    "maximum_polarization_identity_error_ev",
)
PUBLIC_SOURCE_FILES = (
    "docs/implicit-solvation/benchmarks/aggregate_mnsol_response_partition.py",
    "docs/implicit-solvation/benchmarks/benchmark_core.py",
    "docs/implicit-solvation/benchmarks/mnsol_dataset.py",
    "docs/implicit-solvation/benchmarks/mnsol_partition.py",
    "docs/implicit-solvation/benchmarks/mnsol_response_ablation.py",
    "docs/implicit-solvation/benchmarks/run_mnsol_macepolar_response_ablation.py",
)
CONTINUUM_EQUATION_TO_PROFILE = {
    equation: arm.profile for equation, arm in runner.CONTINUUM_ARMS.items()
}
EXPECTED_FINITE_RESOLUTION_RUNTIME_IDENTITY = {
    "profile": CONTINUUM_EQUATION_TO_PROFILE[REQUIRED_CONTINUUM_EQUATION],
    "continuum_equation": REQUIRED_CONTINUUM_EQUATION,
    "mace_checkpoint_identifier": EXPECTED_MACE_POLAR_CHECKPOINT["identifier"],
    "mace_checkpoint_release_url": EXPECTED_MACE_POLAR_CHECKPOINT["release_url"],
    "mace_checkpoint_sha256": EXPECTED_MACE_POLAR_CHECKPOINT["sha256"],
    "mace_checkpoint_size_bytes": EXPECTED_MACE_POLAR_CHECKPOINT["size_bytes"],
    "mace_torch_version": EXPECTED_MACE_TORCH_VERSION,
    "graph_longrange_version": EXPECTED_GRAPH_LONGRANGE_VERSION,
    "mace_long_range_evaluator_profile": (
        EXPECTED_MACE_LONG_RANGE_EVALUATOR_PROFILE
    ),
    "mace_dtype": EXPECTED_MACE_DTYPE,
    "device": EXPECTED_RUNTIME_DEVICE,
    "torch_threads": EXPECTED_SCF_RUNTIME_THREADS,
    "torch_version": EXPECTED_TORCH_VERSION,
    "pyddx_version": EXPECTED_PYDDX_VERSION,
    "pyddx_n_proc": 1,
    "pyddx_solver_tolerance": EXPECTED_SOLVER_TOLERANCE,
    "lmax": EXPECTED_MACE_LONG_RANGE_LMAX,
    "n_lebedev": EXPECTED_MACE_LONG_RANGE_N_LEBEDEV,
    "eta": EXPECTED_MACE_LONG_RANGE_ETA,
}
FINITE_RESOLUTION_RUNTIME_DIGEST_FIELDS = (
    "atomic_numbers_sha256",
    "positions_angstrom_sha256",
    "cavity_radii_angstrom_sha256",
)


@dataclass(frozen=True)
class _Shard:
    run_kind: str
    stage: str
    execution_git_head: str
    continuum_equation: str
    continuum_profile: str
    checkpoints: dict[str, dict[str, Any]]
    record: Mapping[str, Any]


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _require_hex(value: object, *, length: int, label: str) -> str:
    text = str(value)
    if len(text) != length:
        raise ValueError(f"{label} must be {length} characters long.")
    try:
        int(text, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a {length}-hex digest.") from exc
    return text.lower()


def _require_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return value


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _nonnegative_float(value: object, *, label: str) -> float:
    number = _finite_float(value, label=label)
    if number < 0.0:
        raise ValueError(f"{label} must be non-negative.")
    return number


def _require_mapping(
    value: object,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    return dict(value)


def _validated_finite_resolution_runtime_identity(
    value: Mapping[str, Any],
    *,
    selection_index: int,
    method: str,
    canonical_solvent: str,
) -> dict[str, Any]:
    label = (
        f"Record {selection_index} method {method} finite-resolution "
        "runtime_identity"
    )
    identity = dict(value)
    expected = {
        **EXPECTED_FINITE_RESOLUTION_RUNTIME_IDENTITY,
        "solvent": canonical_solvent,
    }
    drifted = [
        field
        for field, expected_value in expected.items()
        if identity.get(field) != expected_value
    ]
    if drifted:
        raise ValueError(
            f"{label} drifted from the frozen runtime lock: "
            + ", ".join(drifted)
            + "."
        )
    dielectric = _finite_float(
        identity.get("continuum_dielectric"),
        label=f"{label} continuum_dielectric",
    )
    expected_dielectric = route2_solvent_spec(
        canonical_solvent
    ).descriptors.dielectric
    if dielectric != expected_dielectric:
        raise ValueError(
            f"{label} continuum_dielectric does not match the frozen "
            f"{canonical_solvent} descriptor."
        )
    normalized = {
        field: identity[field]
        for field in EXPECTED_FINITE_RESOLUTION_RUNTIME_IDENTITY
    }
    normalized["solvent"] = canonical_solvent
    normalized["continuum_dielectric"] = dielectric
    for field in FINITE_RESOLUTION_RUNTIME_DIGEST_FIELDS:
        normalized[field] = _require_hex(
            identity.get(field),
            length=64,
            label=f"{label} {field}",
        )
    return normalized


def _require_not_above(
    value: float,
    *,
    ceiling: float,
    label: str,
) -> None:
    if value > ceiling:
        raise ValueError(f"{label} exceeds the frozen convergence gate.")


def _validate_scf_convergence(
    value: Mapping[str, Any],
    *,
    selection_index: int,
    method: str,
    canonical_solvent: str,
    scf_iterations: int,
) -> dict[str, Any]:
    if method != "mace_scf_l1":
        raise ValueError(f"Record {selection_index} method {method} cannot validate scf_convergence.")

    reason = str(value.get("reason", "")).strip()
    if reason not in MACE_SCF_CONVERGENCE_REASONS:
        raise ValueError(
            f"Record {selection_index} method {method} SCF convergence "
            f"reason is unsupported."
        )

    online_candidate_iteration = _require_int(
        value.get("online_candidate_iteration"),
        label=f"Record {selection_index} method {method} online_candidate_iteration",
    )
    if online_candidate_iteration <= 0:
        raise ValueError(
            f"Record {selection_index} method {method} online_candidate_iteration "
            "must be a positive integer."
        )
    if online_candidate_iteration != scf_iterations:
        raise ValueError(
            f"Record {selection_index} method {method} scf_iterations must "
            "equal online_candidate_iteration."
        )

    final_monopole_residual_e = _nonnegative_float(
        value.get("final_monopole_residual_e"),
        label=(
            f"Record {selection_index} method {method} "
            "final_monopole_residual_e"
        ),
    )
    final_dipole_residual_e_angstrom = _nonnegative_float(
        value.get("final_dipole_residual_e_angstrom"),
        label=(
            f"Record {selection_index} method {method} "
            "final_dipole_residual_e_angstrom"
        ),
    )

    result: dict[str, Any] = {
        "reason": reason,
        "online_candidate_iteration": online_candidate_iteration,
        "final_monopole_residual_e": final_monopole_residual_e,
        "final_dipole_residual_e_angstrom": final_dipole_residual_e_angstrom,
        "runtime_identity": None,
        "history_window": None,
        "fresh_map_replay": None,
    }

    if reason == MACE_SCF_NOMINAL_REASON:
        _require_not_above(
            final_monopole_residual_e,
            ceiling=MACE_SCF_NOMINAL_MONOPOLE_TOLERANCE_E,
            label=(
                f"Record {selection_index} method {method} "
                "final_monopole_residual_e"
            ),
        )
        _require_not_above(
            final_dipole_residual_e_angstrom,
            ceiling=MACE_SCF_NOMINAL_DIPOLE_TOLERANCE_E_ANGSTROM,
            label=(
                f"Record {selection_index} method {method} "
                "final_dipole_residual_e_angstrom"
            ),
        )
        for field in ("runtime_identity", "history_window", "fresh_map_replay"):
            if field not in value or value.get(field) is not None:
                raise ValueError(
                    f"Record {selection_index} method {method} "
                    f"{field} must be null for reason {reason}."
                )
        return result

    runtime_identity = _require_mapping(
        value.get("runtime_identity"),
        label=f"Record {selection_index} method {method} runtime_identity",
    )
    if not runtime_identity:
        raise ValueError(
            f"Record {selection_index} method {method} runtime_identity cannot be empty."
        )
    runtime_identity = _validated_finite_resolution_runtime_identity(
        runtime_identity,
        selection_index=selection_index,
        method=method,
        canonical_solvent=canonical_solvent,
    )
    _require_not_above(
        final_monopole_residual_e,
        ceiling=MACE_SCF_FINITE_RESOLUTION_MONOPOLE_CEILING_E,
        label=(
            f"Record {selection_index} method {method} "
            "final_monopole_residual_e"
        ),
    )
    _require_not_above(
        final_dipole_residual_e_angstrom,
        ceiling=MACE_SCF_FINITE_RESOLUTION_DIPOLE_CEILING_E_ANGSTROM,
        label=(
            f"Record {selection_index} method {method} "
            "final_dipole_residual_e_angstrom"
        ),
    )

    history_window = _require_mapping(
        value.get("history_window"),
        label=f"Record {selection_index} method {method} history_window",
    )
    start_iteration = _require_int(
        history_window.get("start_iteration"),
        label=(
            f"Record {selection_index} method {method} history_window "
            "start_iteration"
        ),
    )
    end_iteration = _require_int(
        history_window.get("end_iteration"),
        label=(
            f"Record {selection_index} method {method} history_window "
            "end_iteration"
        ),
    )
    if start_iteration <= 0 or end_iteration <= 0:
        raise ValueError(
            f"Record {selection_index} method {method} history_window "
            "iteration bounds must be positive."
        )
    if start_iteration > end_iteration:
        raise ValueError(
            f"Record {selection_index} method {method} history_window "
            "start_iteration must not exceed end_iteration."
        )
    if end_iteration != online_candidate_iteration:
        raise ValueError(
            f"Record {selection_index} method {method} history_window "
            "must end at online_candidate_iteration."
        )
    if end_iteration - start_iteration + 1 != 7:
        raise ValueError(
            f"Record {selection_index} method {method} history_window "
            "must contain exactly seven online iterations."
        )
    normalized_history: dict[str, float | int] = {
        "start_iteration": start_iteration,
        "end_iteration": end_iteration,
    }
    for field in MACE_SCF_CONVERGENCE_HISTORY_FIELDS:
        if field in ("start_iteration", "end_iteration"):
            continue
        metric = _nonnegative_float(
            history_window.get(field),
            label=(
                f"Record {selection_index} method {method} "
                f"history_window {field}"
            ),
        )
        normalized_history[field] = metric
        if field in (
            "root_monopole_span_e",
            "residual_monopole_span_e",
        ):
            ceiling = MACE_SCF_NOMINAL_MONOPOLE_TOLERANCE_E
        elif field in (
            "root_dipole_span_e_angstrom",
            "residual_dipole_span_e_angstrom",
        ):
            ceiling = MACE_SCF_NOMINAL_DIPOLE_TOLERANCE_E_ANGSTROM
        elif field == "maximum_monopole_residual_e":
            ceiling = MACE_SCF_FINITE_RESOLUTION_MONOPOLE_CEILING_E
        elif field == "maximum_dipole_residual_e_angstrom":
            ceiling = MACE_SCF_FINITE_RESOLUTION_DIPOLE_CEILING_E_ANGSTROM
        elif field in ("potential_span_ev", "gradient_span_ev_per_angstrom"):
            ceiling = MACE_SCF_FINITE_RESOLUTION_FIELD_SPAN_TOLERANCE
        else:
            ceiling = MACE_SCF_FINITE_RESOLUTION_ENERGY_TOLERANCE_EV
        _require_not_above(
            metric,
            ceiling=ceiling,
            label=(
                f"Record {selection_index} method {method} "
                f"history_window {field}"
            ),
        )

    fresh_map_replay = _require_mapping(
        value.get("fresh_map_replay"),
        label=f"Record {selection_index} method {method} fresh_map_replay",
    )
    replay_count = _require_int(
        fresh_map_replay.get("replay_count"),
        label=(
            f"Record {selection_index} method {method} "
            "fresh_map_replay replay_count"
        ),
    )
    if replay_count != 3:
        raise ValueError(
            f"Record {selection_index} method {method} fresh_map_replay "
            "replay_count must be 3."
        )
    evaluation_count = _require_int(
        fresh_map_replay.get("evaluation_count"),
        label=(
            f"Record {selection_index} method {method} "
            "fresh_map_replay evaluation_count"
        ),
    )
    if evaluation_count != replay_count + 1:
        raise ValueError(
            f"Record {selection_index} method {method} fresh_map_replay "
            "evaluation_count must include the online candidate."
        )
    for field in (
        "includes_online_candidate",
        "all_field_arrays_identical",
        "all_response_arrays_identical",
    ):
        flag = fresh_map_replay.get(field)
        if flag is not True:
            raise ValueError(
                f"Record {selection_index} method {method} "
                f"fresh_map_replay {field} must be true."
            )
    _require_hex(
        fresh_map_replay.get("field_sha256"),
        length=64,
        label=f"Record {selection_index} method {method} fresh_map_replay field_sha256",
    )
    _require_hex(
        fresh_map_replay.get("response_sha256"),
        length=64,
        label=f"Record {selection_index} method {method} fresh_map_replay response_sha256",
    )
    for field in MACE_SCF_CONVERGENCE_MAP_REPLAY_FIELDS:
        if field in (
            "replay_count",
            "evaluation_count",
            "includes_online_candidate",
            "all_field_arrays_identical",
            "all_response_arrays_identical",
        ):
            continue
        if field in ("field_sha256", "response_sha256"):
            continue
        metric = _nonnegative_float(
            fresh_map_replay.get(field),
            label=(
                f"Record {selection_index} method {method} "
                f"fresh_map_replay {field}"
            ),
        )
        ceiling = (
            MACE_SCF_POLARIZATION_IDENTITY_TOLERANCE_EV
            if field == "maximum_polarization_identity_error_ev"
            else (
                MACE_SCF_FINITE_RESOLUTION_MONOPOLE_CEILING_E
                if field == "maximum_monopole_residual_e"
                else (
                    MACE_SCF_FINITE_RESOLUTION_DIPOLE_CEILING_E_ANGSTROM
                    if field == "maximum_dipole_residual_e_angstrom"
                    else MACE_SCF_FINITE_RESOLUTION_ENERGY_TOLERANCE_EV
                )
            )
        )
        _require_not_above(
            metric,
            ceiling=ceiling,
            label=(
                f"Record {selection_index} method {method} "
                f"fresh_map_replay {field}"
            ),
        )

    normalized_fresh_map_replay: dict[str, Any] = {
        "replay_count": replay_count,
        "evaluation_count": evaluation_count,
        "includes_online_candidate": True,
        "all_field_arrays_identical": True,
        "all_response_arrays_identical": True,
        "field_sha256": _require_hex(
            fresh_map_replay["field_sha256"],
            length=64,
            label="field_sha256",
        ),
        "response_sha256": _require_hex(
            fresh_map_replay["response_sha256"],
            length=64,
            label="response_sha256",
        ),
    }
    for field in MACE_SCF_CONVERGENCE_MAP_REPLAY_FIELDS:
        if field in normalized_fresh_map_replay:
            continue
        normalized_fresh_map_replay[field] = float(fresh_map_replay[field])
    result["runtime_identity"] = runtime_identity
    result["history_window"] = normalized_history
    result["fresh_map_replay"] = normalized_fresh_map_replay
    if final_monopole_residual_e > float(
        normalized_history["maximum_monopole_residual_e"]
    ):
        raise ValueError(
            f"Record {selection_index} method {method} final monopole "
            "residual exceeds its history-window maximum."
        )
    if final_dipole_residual_e_angstrom > float(
        normalized_history["maximum_dipole_residual_e_angstrom"]
    ):
        raise ValueError(
            f"Record {selection_index} method {method} final dipole "
            "residual exceeds its history-window maximum."
        )
    if final_monopole_residual_e != float(
        normalized_fresh_map_replay["maximum_monopole_residual_e"]
    ):
        raise ValueError(
            f"Record {selection_index} method {method} fresh-map monopole "
            "residual does not reproduce the online candidate."
        )
    if final_dipole_residual_e_angstrom != float(
        normalized_fresh_map_replay[
            "maximum_dipole_residual_e_angstrom"
        ]
    ):
        raise ValueError(
            f"Record {selection_index} method {method} fresh-map dipole "
            "residual does not reproduce the online candidate."
        )
    return result


def _dataset_hashes(
    value: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, str]:
    if set(value) != set(DATASET_HASH_FIELDS):
        raise ValueError(f"{label} dataset metadata is incomplete.")
    return {
        key: _require_hex(value[key], length=64, label=key)
        for key in DATASET_HASH_FIELDS
    }


def _selection_records(
    selection_manifest: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    if selection_manifest.get("artifact") != PARTITION_ARTIFACT:
        raise ValueError("Selection manifest has unexpected artifact.")
    if (
        str(selection_manifest.get("selection_status", "")).strip()
        != "frozen-before-partition-run"
    ):
        raise ValueError("Selection manifest is not frozen before partition run.")
    partition = str(selection_manifest.get("partition", "")).strip()
    if partition not in SUPPORTED_PARTITIONS:
        raise ValueError("Selection manifest partition is unsupported.")

    rows = selection_manifest.get("selected_records")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Selection manifest selected_records must be a non-empty list."
        )

    indexed: dict[int, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Selection row #{position} is not a JSON object.")
        index = _require_int(
            row.get("selection_index"),
            label="selection_index",
        )
        if index in indexed:
            raise ValueError("Selection manifest contains duplicate selection indices.")
        if str(row.get("partition", "")).strip() != partition:
            raise ValueError("Selection manifest contains cross-partition rows.")
        solvent = str(row.get("canonical_solvent", "")).strip()
        if not solvent:
            raise ValueError(f"Selection row {index} missing canonical_solvent.")
        prior_pilot_overlap = row.get("prior_pilot_geometry_overlap")
        if not isinstance(prior_pilot_overlap, bool):
            raise ValueError(
                f"Selection row {index} prior_pilot_geometry_overlap must be boolean."
            )
        indexed[index] = {
            "partition": partition,
            "canonical_solvent": solvent,
            "opaque_record_id": _require_hex(
                row.get("opaque_record_id"),
                length=64,
                label="opaque_record_id",
            ),
            "geometry_sha256": _require_hex(
                row.get("geometry_sha256"),
                length=64,
                label="geometry_sha256",
            ),
            "prior_pilot_geometry_overlap": prior_pilot_overlap,
        }

    if set(indexed) != set(range(len(indexed))):
        raise ValueError("Selection manifest indices are not contiguous from 0.")
    return indexed


def _checkpoint_size(
    checkpoint: Mapping[str, Any],
    *,
    label: str,
) -> int:
    value = checkpoint.get("size_bytes", checkpoint.get("bytes"))
    size = _require_int(value, label=f"{label} checkpoint size_bytes")
    if size <= 0:
        raise ValueError(f"{label} checkpoint size_bytes must be positive.")
    return size


def _checkpoints(
    value: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for name in ("aimnet2", "mace_polar"):
        checkpoint = value.get(name)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"Private fragment missing {name} checkpoint provenance.")
        normalized[name] = {
            "sha256": _require_hex(
                checkpoint.get("sha256"),
                length=64,
                label=f"{name} checkpoint sha256",
            ),
            "size_bytes": _checkpoint_size(checkpoint, label=name),
        }
    mace = value["mace_polar"]
    assert isinstance(mace, Mapping)
    for key in ("identifier", "release_url"):
        text = str(mace.get(key, "")).strip()
        if not text:
            raise ValueError(f"Private fragment missing MACE checkpoint {key}.")
        normalized["mace_polar"][key] = text
    if normalized["mace_polar"] != EXPECTED_MACE_POLAR_CHECKPOINT:
        raise ValueError(
            "Private fragment MACE-POLAR checkpoint is not the frozen official "
            "polar-1-m checkpoint."
        )
    return normalized


def _validated_shard(
    fragment: Mapping[str, Any],
    *,
    protocol_fingerprint: str,
    selection_fingerprint: str,
    dataset: Mapping[str, str],
) -> _Shard:
    exact = {
        "artifact": runner.ARTIFACT_NAME,
        "schema_version": runner.SCHEMA_VERSION,
        "scf_convergence_contract_version": (
            runner.SCF_CONVERGENCE_CONTRACT_VERSION
        ),
        "visibility": "private-user-supplied-mnsol-row-level",
        "do_not_commit": True,
        "status": "complete",
        "complete_panel": False,
        "run_kind": SOURCE_RUN_KIND,
        "maximum_response_stage": REQUIRED_STAGE,
    }
    for key, expected in exact.items():
        if fragment.get(key) != expected:
            raise ValueError(f"Private fragment {key} is unexpected or incomplete.")
    if (
        _require_hex(
            fragment.get("protocol_fingerprint"),
            length=64,
            label="protocol_fingerprint",
        )
        != protocol_fingerprint
    ):
        raise ValueError("Private fragment protocol fingerprint drifted.")
    if (
        _require_hex(
            fragment.get("selection_fingerprint"),
            length=64,
            label="selection_fingerprint",
        )
        != selection_fingerprint
    ):
        raise ValueError("Private fragment selection fingerprint drifted.")
    if tuple(fragment.get("evaluated_methods") or ()) != FULL_METHODS:
        raise ValueError("Private fragment methods are not the full SCF method set.")

    equation = str(fragment.get("continuum_equation", "")).strip()
    profile = str(fragment.get("continuum_profile", "")).strip()
    if equation != REQUIRED_CONTINUUM_EQUATION:
        raise ValueError(
            "Two-member development matrix requires the frozen ddPCM equation."
        )
    if profile != CONTINUUM_EQUATION_TO_PROFILE[equation]:
        raise ValueError("Private fragment continuum profile does not match equation.")

    fragment_dataset = fragment.get("dataset")
    if not isinstance(fragment_dataset, Mapping):
        raise ValueError("Private fragment dataset metadata is missing.")
    if (
        _dataset_hashes(
            fragment_dataset,
            label="Private fragment",
        )
        != dataset
    ):
        raise ValueError("Private fragment dataset metadata drifted.")

    checkpoint_value = fragment.get("checkpoints")
    if not isinstance(checkpoint_value, Mapping):
        raise ValueError("Private fragment checkpoints are missing.")
    records = fragment.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("Private partition shard must contain exactly one record.")
    return _Shard(
        run_kind=SOURCE_RUN_KIND,
        stage=REQUIRED_STAGE,
        execution_git_head=_require_hex(
            fragment.get("execution_git_head"),
            length=40,
            label="execution_git_head",
        ),
        continuum_equation=equation,
        continuum_profile=profile,
        checkpoints=_checkpoints(checkpoint_value),
        record=records[0],
    )


def _validated_methods(
    methods: Mapping[str, Any],
    *,
    selection_index: int,
    canonical_solvent: str,
    experimental_kcal_mol: float,
) -> dict[str, dict[str, Any]]:
    if tuple(methods) != FULL_METHODS:
        raise ValueError(
            f"Record {selection_index} does not contain the full SCF "
            "method set in source-runner order."
        )

    normalized: dict[str, dict[str, Any]] = {}
    for method in FULL_METHODS:
        payload = methods[method]
        if not isinstance(payload, Mapping):
            raise ValueError(f"Record {selection_index} method {method} is malformed.")
        if not set(METHOD_FIELDS).issubset(payload):
            raise ValueError(f"Record {selection_index} method {method} field drifted.")
        row: dict[str, Any] = {
            field: _finite_float(
                payload[field],
                label=f"Record {selection_index} method {method} {field}",
            )
            for field in METHOD_FIELDS
        }
        if method == "mace_scf_l1":
            scf_iterations = _require_int(
                payload.get("scf_iterations"),
                label=(
                    f"Record {selection_index} method {method} scf_iterations"
                ),
            )
            if scf_iterations <= 0:
                raise ValueError(
                    f"Record {selection_index} method {method} "
                    "scf_iterations must be positive."
                )
            unmixed_density_residual = _nonnegative_float(
                payload.get("unmixed_density_residual_inf_e"),
                label=(
                    f"Record {selection_index} method {method} "
                    "unmixed_density_residual_inf_e"
                ),
            )
            half_coupling_identity_error_ev = _nonnegative_float(
                payload.get("half_coupling_identity_error_ev"),
                label=(
                    f"Record {selection_index} method {method} "
                    "half_coupling_identity_error_ev"
                ),
            )
            _require_not_above(
                half_coupling_identity_error_ev,
                ceiling=MACE_SCF_POLARIZATION_IDENTITY_TOLERANCE_EV,
                label=(
                    f"Record {selection_index} method {method} "
                    "half_coupling_identity_error_ev"
                ),
            )
            if "scf_convergence" not in payload:
                raise ValueError(
                    f"Record {selection_index} method {method} scf_convergence "
                    "is required."
                )
            row["scf_convergence"] = _validate_scf_convergence(
                _require_mapping(
                    payload["scf_convergence"],
                    label=(
                        f"Record {selection_index} method {method} scf_convergence"
                    ),
                ),
                selection_index=selection_index,
                method=method,
                canonical_solvent=canonical_solvent,
                scf_iterations=scf_iterations,
            )
            convergence = row["scf_convergence"]
            channel_maximum = max(
                float(convergence["final_monopole_residual_e"]),
                float(convergence["final_dipole_residual_e_angstrom"]),
            )
            if unmixed_density_residual != channel_maximum:
                raise ValueError(
                    f"Record {selection_index} method {method} legacy "
                    "unmixed_density_residual_inf_e does not match the "
                    "channel maximum."
                )
            if convergence["reason"] == MACE_SCF_FINITE_RESOLUTION_REASON:
                replay_identity_maximum = float(
                    convergence["fresh_map_replay"][
                        "maximum_polarization_identity_error_ev"
                    ]
                )
                if (
                    half_coupling_identity_error_ev
                    > replay_identity_maximum
                ):
                    raise ValueError(
                        f"Record {selection_index} method {method} final "
                        "half-coupling identity exceeds fresh-map evidence."
                    )
            row["scf_iterations"] = scf_iterations
            row["unmixed_density_residual_inf_e"] = (
                unmixed_density_residual
            )
            row["half_coupling_identity_error_ev"] = (
                half_coupling_identity_error_ev
            )
        if row["wall_seconds"] < 0.0:
            raise ValueError(
                f"Record {selection_index} method {method} wall time " "is negative."
            )
        identities = (
            (
                row["electrostatic_kcal_mol"],
                row["solute_polarization_kcal_mol"]
                + row["continuum_polarization_kcal_mol"],
                "electrostatic ledger",
            ),
            (
                row["total_solvation_kcal_mol"],
                row["electrostatic_kcal_mol"] + row["smd_cds_kcal_mol"],
                "total ledger",
            ),
            (
                row["signed_error_kcal_mol"],
                row["total_solvation_kcal_mol"] - experimental_kcal_mol,
                "signed-error ledger",
            ),
            (
                row["absolute_error_kcal_mol"],
                abs(row["signed_error_kcal_mol"]),
                "absolute-error ledger",
            ),
        )
        for observed, expected, label in identities:
            if not np.isclose(
                observed,
                expected,
                rtol=0.0,
                atol=1.0e-10,
            ):
                raise ValueError(
                    f"Record {selection_index} method {method} " f"{label} drifted."
                )
        normalized[method] = row
    return {method: normalized[method] for method in ALLOWED_METHODS}


def _aggregate_scf_convergence(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    history_maxima: dict[str, float] = {
        field: 0.0 for field in MACE_SCF_CONVERGENCE_HISTORY_FIELDS
    }
    replay_maxima: dict[str, float] = {
        field: 0.0 for field in MACE_SCF_CONVERGENCE_MAP_REPLAY_FIELDS
        if field
        not in (
            "replay_count",
            "evaluation_count",
            "includes_online_candidate",
            "all_field_arrays_identical",
            "all_response_arrays_identical",
            "field_sha256",
            "response_sha256",
        )
    }
    nominal_count = 0
    finite_count = 0
    max_final_monopole_residual_e = 0.0
    max_final_dipole_residual_e_angstrom = 0.0

    for record in records:
        convergence = record["methods"]["mace_scf_l1"]["scf_convergence"]
        reason = str(convergence["reason"])
        reasons[reason] = reasons.get(reason, 0) + 1

        if reason == MACE_SCF_NOMINAL_REASON:
            nominal_count += 1
        else:
            finite_count += 1

        max_final_monopole_residual_e = max(
            max_final_monopole_residual_e,
            float(convergence["final_monopole_residual_e"]),
        )
        max_final_dipole_residual_e_angstrom = max(
            max_final_dipole_residual_e_angstrom,
            float(convergence["final_dipole_residual_e_angstrom"]),
        )

        if reason == MACE_SCF_FINITE_RESOLUTION_REASON:
            history_window = convergence["history_window"]
            for field in MACE_SCF_CONVERGENCE_HISTORY_FIELDS:
                candidate = history_window[field]
                value = float(candidate)
                if value > history_maxima[field]:
                    history_maxima[field] = value

            fresh_map_replay = convergence["fresh_map_replay"]
            for field in replay_maxima:
                candidate = fresh_map_replay[field]
                value = float(candidate)
                if value > replay_maxima[field]:
                    replay_maxima[field] = value

    return {
        "record_count": len(records),
        "reason_counts": reasons,
        "nominal_count": nominal_count,
        "finite_count": finite_count,
        "max_final_monopole_residual_e": max_final_monopole_residual_e,
        "max_final_dipole_residual_e_angstrom": max_final_dipole_residual_e_angstrom,
        "finite_history_window_maximums": history_maxima,
        "finite_map_replay_maximums": replay_maxima,
    }


def _validated_record(
    row: Mapping[str, Any],
    *,
    selection_records: Mapping[int, Mapping[str, Any]],
    partition: str,
) -> dict[str, Any]:
    index = _require_int(
        row.get("selection_index"),
        label="selection_index",
    )
    if index not in selection_records:
        raise ValueError(
            f"Private record index {index} is outside the frozen " "selection manifest."
        )
    expected = selection_records[index]
    prior_pilot_overlap = row.get("prior_pilot_geometry_overlap")
    if not isinstance(prior_pilot_overlap, bool):
        raise ValueError(
            f"Record {index} prior_pilot_geometry_overlap must be boolean."
        )
    actual_identity = {
        "partition": str(row.get("partition", "")).strip(),
        "canonical_solvent": str(row.get("canonical_solvent", "")).strip(),
        "opaque_record_id": _require_hex(
            row.get("opaque_record_id"),
            length=64,
            label="opaque_record_id",
        ),
        "geometry_sha256": _require_hex(
            row.get("geometry_sha256"),
            length=64,
            label="geometry_sha256",
        ),
        "prior_pilot_geometry_overlap": prior_pilot_overlap,
    }
    if actual_identity["partition"] != partition:
        raise ValueError("cross-partition private record detected.")
    if actual_identity != expected:
        raise ValueError(f"Record {index} identity drifted from selection manifest.")

    experimental = _finite_float(
        row.get("experimental_delta_g_kcal_mol"),
        label=f"Record {index} experimental_delta_g_kcal_mol",
    )
    methods = row.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError(f"Record {index} has no method map.")
    return {
        "selection_index": index,
        **actual_identity,
        "experimental_delta_g_kcal_mol": experimental,
        "methods": _validated_methods(
            methods,
            selection_index=index,
            canonical_solvent=str(actual_identity["canonical_solvent"]),
            experimental_kcal_mol=experimental,
        ),
    }


def _singleton(values: set[str], *, message: str) -> str:
    if len(values) != 1:
        raise ValueError(message)
    return next(iter(values))


def _threshold_metrics(
    records: Sequence[Mapping[str, Any]],
    method: str,
) -> dict[str, Any]:
    errors = np.asarray(
        [record["methods"][method]["absolute_error_kcal_mol"] for record in records],
        dtype=float,
    )
    count = int(errors.size)
    at_or_above_one = int(np.count_nonzero(errors >= 1.0))
    at_or_above_one_five = int(np.count_nonzero(errors >= 1.5))
    return {
        "n": count,
        "ge_1_0_count": at_or_above_one,
        "ge_1_0_fraction": at_or_above_one / count,
        "ge_1_5_count": at_or_above_one_five,
        "ge_1_5_fraction": at_or_above_one_five / count,
    }


def _source_hashes() -> dict[str, str]:
    return {path: sha256_file(REPO_ROOT / path) for path in PUBLIC_SOURCE_FILES}


def _git_stdout(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            "Could not verify source execution commit: " + result.stderr.strip()
        )
    return result.stdout.strip()


def _execution_source_identity(execution_git_head: str) -> dict[str, str]:
    commit_check = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "cat-file",
            "-e",
            f"{execution_git_head}^{{commit}}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if commit_check.returncode != 0:
        raise ValueError("Shard execution_git_head is not a local Git commit.")
    ancestor_check = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "merge-base",
            "--is-ancestor",
            execution_git_head,
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor_check.returncode != 0:
        raise ValueError(
            "Shard execution commit is not an ancestor of the aggregation checkout."
        )

    runner_bytes = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "show",
            f"{execution_git_head}:{SOURCE_RUNNER_PATH}",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return {
        "execution_git_head": execution_git_head,
        "execution_git_tree_sha1": _require_hex(
            _git_stdout("rev-parse", f"{execution_git_head}^{{tree}}"),
            length=40,
            label="execution_git_tree_sha1",
        ),
        "source_runner_path": SOURCE_RUNNER_PATH,
        "source_runner_git_blob_sha1": _require_hex(
            _git_stdout(
                "rev-parse",
                f"{execution_git_head}:{SOURCE_RUNNER_PATH}",
            ),
            length=40,
            label="source_runner_git_blob_sha1",
        ),
        "source_runner_sha256": hashlib.sha256(runner_bytes).hexdigest(),
    }


def _source_shard_identity(
    fragments: Sequence[Mapping[str, Any]],
) -> tuple[list[str], str]:
    shard_hashes = sorted(
        hashlib.sha256(canonical_json_bytes(fragment)).hexdigest()
        for fragment in fragments
    )
    set_hash = hashlib.sha256(canonical_json_bytes(shard_hashes)).hexdigest()
    return shard_hashes, set_hash


def aggregate_private_two_member_shards(
    fragments: Sequence[Mapping[str, Any]],
    *,
    selection_records: Mapping[int, Mapping[str, Any]],
    selection_artifact_sha256: str,
    selection_fingerprint: str,
    protocol_fingerprint: str,
    dataset: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and aggregate one frozen MNSol partition."""

    if not fragments:
        raise ValueError("At least one private shard is required.")
    if not selection_records:
        raise ValueError("Selection manifest cannot be empty.")
    protocol_hash = _require_hex(
        protocol_fingerprint,
        length=64,
        label="protocol_fingerprint",
    )
    selection_hash = _require_hex(
        selection_fingerprint,
        length=64,
        label="selection_fingerprint",
    )
    selection_artifact_hash = _require_hex(
        selection_artifact_sha256,
        length=64,
        label="selection_artifact_sha256",
    )
    dataset_hashes = _dataset_hashes(dataset, label="Input")

    expected_indices = set(selection_records)
    if expected_indices != set(range(len(selection_records))):
        raise ValueError("Selection records must use contiguous indices starting at 0.")
    partitions = {
        str(record.get("partition", "")).strip()
        for record in selection_records.values()
    }
    partition = _singleton(
        partitions,
        message="Selection records must share one partition.",
    )
    if partition not in SUPPORTED_PARTITIONS:
        raise ValueError("Selection records use an unsupported partition.")

    shards = [
        _validated_shard(
            fragment,
            protocol_fingerprint=protocol_hash,
            selection_fingerprint=selection_hash,
            dataset=dataset_hashes,
        )
        for fragment in fragments
    ]
    execution_head = _singleton(
        {shard.execution_git_head for shard in shards},
        message="Private fragments have different execution heads.",
    )
    execution_source = _execution_source_identity(execution_head)
    source_shard_hashes, source_shard_set_hash = _source_shard_identity(fragments)
    equation = _singleton(
        {shard.continuum_equation for shard in shards},
        message="Private fragments have mismatched continuum equations.",
    )
    profile = _singleton(
        {shard.continuum_profile for shard in shards},
        message="Private fragments have mismatched continuum profiles.",
    )
    checkpoints = shards[0].checkpoints
    if any(shard.checkpoints != checkpoints for shard in shards[1:]):
        raise ValueError("Private fragments have different checkpoint provenance.")

    by_index: dict[int, dict[str, Any]] = {}
    for shard in shards:
        record = _validated_record(
            shard.record,
            selection_records=selection_records,
            partition=partition,
        )
        index = record["selection_index"]
        if index in by_index:
            raise ValueError("Private record selection indices are duplicate.")
        by_index[index] = record
    if set(by_index) != expected_indices:
        raise ValueError("Private fragments must cover all frozen partition indices.")

    records = [by_index[index] for index in sorted(by_index)]
    metrics = {
        method: {
            **aggregate_method_metrics(records, method),
            **_threshold_metrics(records, method),
        }
        for method in ALLOWED_METHODS
    }
    comparison = paired_method_comparison(
        records,
        left=ALLOWED_METHODS[0],
        right=ALLOWED_METHODS[1],
    )
    overlap_records = [
        record for record in records if record["prior_pilot_geometry_overlap"]
    ]
    overlap_unique_geometry_count = len(
        {record["geometry_sha256"] for record in overlap_records}
    )
    selection = {
        "partition": partition,
        "record_count": len(records),
        "full_preregistered_record_count": len(selection_records),
        "complete_partition": True,
        "solvent_count": len({record["canonical_solvent"] for record in records}),
        "unique_geometry_count": len({record["geometry_sha256"] for record in records}),
        "prior_pilot_geometry_overlap_record_count": len(overlap_records),
        "prior_pilot_geometry_overlap_unique_geometry_count": (
            overlap_unique_geometry_count
        ),
        "selection_indices": sorted(by_index),
        "preregistered_members": list(ALLOWED_METHODS),
    }
    scientific_identity = {
        "dataset_scope": f"frozen MNSol {partition} partition",
        "paired_methods": list(ALLOWED_METHODS),
        "methods": {
            "mace_fixed_l1": "gas MACE l<=1; U(c0)+G_CDS",
            "mace_scf_l1": (
                "nominal same-root c*=M(P(c*)) or energy-only "
                "finite-resolution approximate candidate under the frozen "
                "residual policy; DeltaE_model+U(c*)+G_CDS"
            ),
        },
        "reaction_field_projector": "local-jet",
        "shared_electrostatics": "pyddx ddPCM",
        "shared_cavity": "PySCF 2.13.1 SMD Coulomb radii",
        "shared_nonpolar_model": "PySCF 2.13.1 SMD-CDS",
        "continuum_axes_shared": True,
    }
    common = {
        "artifact": AGGREGATOR_ARTIFACT_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "complete_panel": True,
        "complete_partition": True,
        "run_kind": AGGREGATE_RUN_KIND,
        "source_run_kind": SOURCE_RUN_KIND,
        "source_shard_schema_version": runner.SCHEMA_VERSION,
        "scf_convergence_contract_version": (
            runner.SCF_CONVERGENCE_CONTRACT_VERSION
        ),
        "source_runner_evaluated_methods": list(FULL_METHODS),
        "maximum_response_stage": REQUIRED_STAGE,
        "protocol_fingerprint": protocol_hash,
        "selection_artifact_sha256": selection_artifact_hash,
        "selection_fingerprint": selection_hash,
        "shard_execution_git_head": execution_head,
        "source_execution": execution_source,
        "source_shards": {
            "count": len(fragments),
            "canonical_set_sha256": source_shard_set_hash,
        },
        "scf_convergence_summary": _aggregate_scf_convergence(records),
        "continuum_equation": equation,
        "continuum_profile": profile,
        "dataset": dataset_hashes,
        "scientific_identity": scientific_identity,
        "selection": selection,
        "aggregated_methods": list(ALLOWED_METHODS),
        "aggregate_metrics": metrics,
        "paired_method_comparisons": {
            "mace_fixed_l1__mace_scf_l1": comparison,
        },
        "aggregator_source_files_sha256": _source_hashes(),
    }
    private = {
        **common,
        "visibility": "private-user-supplied-mnsol-row-level",
        "do_not_commit": True,
        "claim_boundary": (
            "This row-level artifact covers the frozen development partition "
            "and exactly two preregistered members. It is not full-653 MNSol "
            f"evidence; {len(overlap_records)} records reuse geometries already "
            f"inspected in the pilot ({overlap_unique_geometry_count} unique "
            "geometries), and confirmation remains sealed. Finite-resolution "
            "acceptance records a repeatable, energy-only, finite-precision "
            "approximate fixed-point candidate under the frozen residual "
            "policy; it does not establish energetic accuracy or force "
            "agreement."
        ),
        "member_checkpoint": checkpoints["mace_polar"],
        "source_runner_checkpoints": checkpoints,
        "source_shard_canonical_sha256": source_shard_hashes,
        "records": records,
    }
    public = {
        **common,
        "visibility": "public-aggregate-only",
        "do_not_commit": False,
        "claim_boundary": (
            "This aggregate covers the frozen development partition and "
            "exactly two preregistered members. It contains no row-level MNSol "
            f"values; {len(overlap_records)} records reuse geometries already "
            f"inspected in the pilot ({overlap_unique_geometry_count} unique "
            "geometries). Finite-resolution acceptance records a repeatable, "
            "energy-only, finite-precision approximate fixed-point candidate "
            "under the frozen residual policy; it does not establish energetic "
            "accuracy or force agreement. It is not full-653 MNSol certification and does not "
            "unseal the confirmation partition."
        ),
        "member_checkpoint": checkpoints["mace_polar"],
    }
    return private, public


def _require_below(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must remain below {root}.") from exc
    return resolved


def _validated_output_paths(
    private_output: Path,
    public_output: Path,
) -> tuple[Path, Path]:
    private = _require_below(
        private_output,
        REPO_ROOT / ".omx" / "benchmarks",
        label="Private two-member matrix",
    )
    public = _require_below(
        public_output,
        REPO_ROOT / "docs" / "implicit-solvation" / "benchmarks",
        label="Public two-member aggregate",
    )
    for path in (private, public):
        if path.exists():
            raise FileExistsError(path)
    return private, public


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--pilot-selection", type=Path, required=True)
    parser.add_argument(
        "--private-shard",
        action="append",
        type=Path,
        required=True,
        help="Private one-record partition shard; repeat for every row.",
    )
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    return parser


def _main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    private_output, public_output = _validated_output_paths(
        args.private_output,
        args.public_output,
    )
    shard_paths = [
        _require_below(
            path,
            REPO_ROOT / ".omx" / "benchmarks",
            label="Private response shard",
        )
        for path in args.private_shard
    ]

    selection_manifest = _load_json_object(
        args.selection,
        label="selection manifest",
    )
    selection_records = _selection_records(selection_manifest)

    protocol = load_mnsol_protocol(args.protocol)
    dataset = load_mnsol_v2012(args.source, protocol)
    pilot_manifest = _load_json_object(
        args.pilot_selection,
        label="pilot selection manifest",
    )
    validate_frozen_mnsol_partition_selection(
        selection_manifest,
        dataset,
        protocol,
        pilot_manifest,
    )
    dataset_hashes = {
        "source_artifact_sha256": dataset.source_artifact_sha256,
        "table_sha256": dataset.table_sha256,
        "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
    }
    fragments = [_load_json_object(path, label="private shard") for path in shard_paths]
    private, public = aggregate_private_two_member_shards(
        fragments,
        selection_records=selection_records,
        selection_artifact_sha256=sha256_file(args.selection),
        selection_fingerprint=str(selection_manifest["selection_fingerprint"]),
        protocol_fingerprint=protocol.fingerprint,
        dataset=dataset_hashes,
    )
    write_json_atomic(private_output, private)
    write_json_atomic(public_output, public)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
