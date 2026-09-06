#!/usr/bin/env python3
"""Audit AIMNet2 first-order source response over the MNSol-653 element union."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

for _thread_environment_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ[_thread_environment_name] = "1"

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "docs" / "implicit-solvation" / "benchmarks"
for search_path in (REPO_ROOT, BENCHMARK_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from benchmark_core import (
    numerical_runtime_identity,
    sha256_file,
    single_threaded_numerics,
    write_json_atomic,
)
from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
from maple.function.calculator.aimnet._aimnet2_calculator import (
    AIMNET2_RAW_CHARGE_TOLERANCE_E,
)
from maple.function.calculator.aimnet._aimnet2_float64_source import (
    AIMNET_FIRST_ORDER_ENERGY_PARITY_ABSOLUTE_TOLERANCE_EV,
    AIMNET_FLOAT64_RUNTIME_VERSION,
    AIMNET_SECOND_ORDER_CHARGE_PARITY_ABSOLUTE_TOLERANCE_E,
    AIMNET_SECOND_ORDER_CHARGE_VJP_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A,
    AIMNET_SECOND_ORDER_ENERGY_PARITY_ABSOLUTE_TOLERANCE_EV,
    AIMNET_SECOND_ORDER_GRADIENT_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A,
    AIMNet2ReconstructedFloat64SourceCalculator,
)
from maple.solvation.release.aimnet2_multisolvent_force import canonical_sha256

ARTIFACT = "route2-aimnet2-mnsol653-source-domain-prequalification-private-v1"
PROTOCOL_ARTIFACT = "route2-aimnet2-mnsol653-source-domain-prequalification-protocol-v1"
PRIOR_ARTIFACT = "route2-mnsol-aimnet2-full-frozen-charge-matrix-v1"
SELECTION_SCHEMA = "maple-pure-mace-polar-analytic-gaussian-qm-attribution-prereg-v2"
EXPECTED_PRIOR_RECORD_COUNT = 653
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
SOURCE_DOMAIN_CLAIM_BOUNDARY = (
    "Passing proves only source-runtime first-order availability over the exact "
    "653-panel element union. It does not prove solvation accuracy, "
    "same-total-scalar forces, public E/F/H/V/M, OPT, FREQ, TS, IRC, MD, stress, "
    "or virial support."
)
SOURCE_DOMAIN_EXECUTION_CONTRACT = {
    "clean_committed_checkout": True,
    "private_output_below_ignored_omx_only": True,
    "all_selected_geometries_attempted": True,
    "negative_results_retained": True,
    "no_domain_widening_before_a_passing_hash_bound_artifact": True,
}
_ARTIFACT_BINDING_KEYS = {
    "execution_git_commit",
    "execution_git_tree",
    "protocol_file_sha256",
    "mnsol_dataset_zip_sha256",
    "mnsol_protocol_sha256",
    "prior_private_653_sha256",
    "selection_manifest_file_sha256",
    "selection_manifest_canonical_sha256",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "runtime_kind",
    "runtime_provenance_sha256",
    "numerical_runtime_sha256",
    "covered_atomic_numbers",
    "selected_records",
    "thresholds",
    "claim_boundary",
}


def _object(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object: {path}")
    return value


def _clean_commit() -> tuple[str, str]:
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError(
            "Source-domain evidence requires a clean committed checkout."
        )
    values = []
    for revision in ("HEAD", "HEAD^{tree}"):
        values.append(
            subprocess.run(
                ["git", "-C", str(REPO_ROOT), "rev-parse", revision],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    if any(len(value) != 40 for value in values):
        raise RuntimeError("Could not resolve execution commit/tree.")
    return values[0], values[1]


def _assert_same_clean_commit(commit: str, tree: str) -> None:
    observed_commit, observed_tree = _clean_commit()
    if observed_commit != commit or observed_tree != tree:
        raise RuntimeError(
            "Execution commit/tree changed during source-domain evidence."
        )


def _private_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to((REPO_ROOT / ".omx").resolve())
    except ValueError as exc:
        raise ValueError(
            "Source-domain output must remain below repository .omx."
        ) from exc
    relative = resolved.relative_to(REPO_ROOT).as_posix()
    ignored = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "check-ignore",
            "-v",
            "--no-index",
            "--",
            relative,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if ignored.returncode != 0 or not ignored.stdout.strip():
        raise RuntimeError("Private source-domain output is not ignored by Git.")
    origin = ignored.stdout.split("\t", 1)[0].rsplit(":", 2)[0]
    origin_path = Path(origin)
    if not origin_path.is_absolute():
        origin_path = REPO_ROOT / origin_path
    tracked_ignore = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", ".gitignore"],
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        origin_path.resolve() != (REPO_ROOT / ".gitignore").resolve()
        or tracked_ignore.returncode != 0
    ):
        raise RuntimeError(
            "Private source-domain output must be protected by the tracked "
            "repository .gitignore."
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _runtime_identity(calculator: object) -> tuple[dict[str, Any], str]:
    provider = getattr(calculator, "runtime_provenance", None)
    if not callable(provider):
        raise TypeError("AIMNet2 source calculator exposes no runtime provenance.")
    provenance = provider()
    if not isinstance(provenance, dict):
        raise TypeError("AIMNet2 runtime provenance must be a JSON object.")
    try:
        json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "AIMNet2 runtime provenance must be finite JSON metadata."
        ) from exc
    if provenance.get("runtime_kind") != AIMNET_FLOAT64_RUNTIME_VERSION:
        raise ValueError("AIMNet2 runtime provenance kind drifted.")
    return provenance, canonical_sha256(provenance)


def _configure_torch_threads():
    import torch

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Torch did not enter the preregistered single-thread mode.")
    return torch


def _validate_numerical_runtime(
    artifact: Mapping[str, Any], expected_sha256: object
) -> None:
    runtime = artifact.get("numerical_runtime")
    if not isinstance(runtime, Mapping):
        raise TypeError("source-domain numerical runtime must be a mapping.")
    try:
        json.dumps(
            runtime,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "source-domain numerical runtime must be finite JSON metadata."
        ) from exc
    if canonical_sha256(runtime) != expected_sha256:
        raise ValueError("source-domain numerical runtime identity drifted.")
    environment = runtime.get("thread_environment")
    pools = runtime.get("threadpools")
    if not isinstance(environment, Mapping) or any(
        environment.get(name) != "1"
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    ):
        raise ValueError("source-domain thread environment is not single-threaded.")
    if (
        not isinstance(pools, Sequence)
        or isinstance(pools, (str, bytes))
        or not pools
        or any(
            not isinstance(pool, Mapping)
            or isinstance(pool.get("num_threads"), bool)
            or pool.get("num_threads") != 1
            for pool in pools
        )
        or runtime.get("torch_num_threads") != 1
        or runtime.get("torch_num_interop_threads") != 1
    ):
        raise ValueError("source-domain effective numerical threads are not one.")


def _canonical_manifest_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "artifact_sha256"}
    )


def _thresholds() -> dict[str, float]:
    return {
        "raw_charge_residual_absolute_e": AIMNET2_RAW_CHARGE_TOLERANCE_E,
        "projected_charge_sum_absolute_e": 1.0e-10,
        "ordinary_decomposed_energy_absolute_eV": (
            AIMNET_FIRST_ORDER_ENERGY_PARITY_ABSOLUTE_TOLERANCE_EV
        ),
        "ordinary_decomposed_charge_max_absolute_e": (
            AIMNET_SECOND_ORDER_CHARGE_PARITY_ABSOLUTE_TOLERANCE_E
        ),
        "ordinary_decomposed_charge_vjp_max_absolute_eV_per_A": (
            AIMNET_SECOND_ORDER_CHARGE_VJP_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A
        ),
        "decomposed_repeat_energy_absolute_eV": (
            AIMNET_SECOND_ORDER_ENERGY_PARITY_ABSOLUTE_TOLERANCE_EV
        ),
        "decomposed_repeat_charge_max_absolute_e": (
            AIMNET_SECOND_ORDER_CHARGE_PARITY_ABSOLUTE_TOLERANCE_E
        ),
        "decomposed_repeat_intrinsic_gradient_max_absolute_eV_per_A": (
            AIMNET_SECOND_ORDER_GRADIENT_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A
        ),
        "decomposed_repeat_charge_vjp_max_absolute_eV_per_A": (
            AIMNET_SECOND_ORDER_CHARGE_VJP_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A
        ),
    }


def _validate_protocol(
    protocol: Mapping[str, Any],
    *,
    source: Path,
    mnsol_protocol: Path,
    prior_private: Path,
    selection_manifest: Path,
    checkpoint: Path,
) -> None:
    if (
        protocol.get("artifact") != PROTOCOL_ARTIFACT
        or protocol.get("schema_version") != 1
        or protocol.get("status") != "preregistered-not-executed"
        or protocol.get("thresholds") != _thresholds()
        or protocol.get("capabilities") != NO_CAPABILITIES
        or protocol.get("claim_boundary") != SOURCE_DOMAIN_CLAIM_BOUNDARY
        or protocol.get("execution_contract") != SOURCE_DOMAIN_EXECUTION_CONTRACT
    ):
        raise ValueError("Source-domain prequalification protocol drifted.")
    runtime = protocol.get("runtime")
    pinned = protocol.get("pinned_inputs")
    selection = protocol.get("selection")
    response = protocol.get("response")
    if (
        not isinstance(runtime, Mapping)
        or not isinstance(pinned, Mapping)
        or not isinstance(selection, Mapping)
        or not isinstance(response, Mapping)
    ):
        raise ValueError("Source-domain runtime/input protocol is malformed.")
    if (
        selection.get("expected_atomic_numbers") != [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
        or selection.get("expected_geometry_count") != 15
        or selection.get("selection_reads_experimental_targets") is not False
        or selection.get("selection_reads_aimnet2_or_continuum_outputs") is not False
        or selection.get("reselection_after_any_result") is not False
        or selection.get("base_geometry_policy")
        != "first occurrence of each geometry_sha256 in the frozen 24-record target-free matched-QM attribution panel"
        or selection.get("coverage_completion_policy")
        != "for each still-missing atomic number in ascending order, select the not-yet-selected 653-panel geometry minimizing (geometry_sha256, opaque_record_id); one geometry may close multiple elements"
    ):
        raise ValueError("Source-domain target-free selection contract drifted.")
    if (
        response.get("charge_cotangent_policy")
        != "normalized zero-mean linear atom-index ramp; geometry/target/model-output independent"
        or response.get("requested_total_charge_e") != 0
        or response.get("multiplicity") != 1
        or response.get("ordinary_vs_decomposed_hard_gates")
        != ["energy", "charges", "charge_position_vjp"]
        or response.get("ordinary_embedded_dftd3_intrinsic_gradient_difference")
        != "report-only"
        or response.get("decomposed_repeat_hard_gates")
        != ["energy", "charges", "intrinsic_gradient", "charge_position_vjp"]
    ):
        raise ValueError("Source-domain response contract drifted.")
    if (
        runtime.get("kind") != AIMNET_FLOAT64_RUNTIME_VERSION
        or runtime.get("checkpoint_sha256") != sha256_file(checkpoint)
        or runtime.get("checkpoint_bytes") != checkpoint.stat().st_size
    ):
        raise ValueError("Source-domain AIMNet2 runtime identity drifted.")
    expected_hashes = {
        "mnsol_dataset_zip_sha256": sha256_file(source),
        "mnsol_protocol_sha256": sha256_file(mnsol_protocol),
        "prior_private_653_sha256": sha256_file(prior_private),
        "selection_manifest_file_sha256": sha256_file(selection_manifest),
    }
    if any(pinned.get(name) != value for name, value in expected_hashes.items()):
        raise ValueError("A source-domain pinned input hash drifted.")


def select_audit_geometries(
    *,
    dataset: Any,
    prior_records: Sequence[Mapping[str, Any]],
    manifest_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    """Select target-free base geometries and deterministic element completion."""

    by_opaque = {str(record["opaque_record_id"]): record for record in prior_records}
    if len(by_opaque) != len(prior_records):
        raise ValueError("Prior MNSol opaque record IDs are not unique.")

    selected: list[dict[str, Any]] = []
    selected_geometries: set[str] = set()
    for manifest_record in manifest_records:
        geometry_sha256 = str(manifest_record["geometry_sha256"])
        if geometry_sha256 in selected_geometries:
            continue
        prior = by_opaque.get(str(manifest_record["opaque_record_id"]))
        if prior is None:
            raise ValueError("Target-free manifest record is absent from MNSol-653.")
        if prior.get("geometry_sha256") != geometry_sha256:
            raise ValueError("Target-free manifest geometry identity drifted.")
        selected.append(dict(prior))
        selected_geometries.add(geometry_sha256)

    geometry_records: dict[str, list[Mapping[str, Any]]] = {}
    all_atomic_numbers: set[int] = set()
    for record in prior_records:
        geometry = dataset.geometries.get(str(record["geometry_handle"]))
        if geometry is None or geometry.sha256 != record.get("geometry_sha256"):
            raise ValueError("Prior MNSol geometry identity drifted.")
        all_atomic_numbers.update(int(value) for value in geometry.atomic_numbers)
        geometry_records.setdefault(geometry.sha256, []).append(record)

    covered = set()
    for record in selected:
        geometry = dataset.geometries[str(record["geometry_handle"])]
        covered.update(int(value) for value in geometry.atomic_numbers)
    for atomic_number in sorted(all_atomic_numbers - covered):
        if atomic_number in covered:
            continue
        candidates = []
        for geometry_sha256, records in geometry_records.items():
            if geometry_sha256 in selected_geometries:
                continue
            representative = min(
                records, key=lambda item: str(item["opaque_record_id"])
            )
            geometry = dataset.geometries[str(representative["geometry_handle"])]
            if atomic_number in geometry.atomic_numbers:
                candidates.append(
                    (
                        geometry_sha256,
                        str(representative["opaque_record_id"]),
                        representative,
                    )
                )
        if not candidates:
            raise ValueError(f"No MNSol geometry covers atomic number {atomic_number}.")
        geometry_sha256, _opaque, representative = min(candidates)
        selected.append(dict(representative))
        selected_geometries.add(geometry_sha256)
        geometry = dataset.geometries[str(representative["geometry_handle"])]
        covered.update(int(value) for value in geometry.atomic_numbers)

    return selected, tuple(sorted(all_atomic_numbers))


def _cotangent(atom_count: int) -> np.ndarray:
    if atom_count < 2:
        raise ValueError("Source-domain audit requires at least two atoms.")
    values = np.arange(atom_count, dtype=float)
    values -= float(np.mean(values))
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Deterministic source cotangent is singular.")
    return values / norm


def _row_passes(row: Mapping[str, Any], thresholds: Mapping[str, float]) -> bool:
    if row.get("status") != "measured":
        return False
    parity = row.get("parity")
    if not isinstance(parity, Mapping):
        return False
    comparisons = {
        "raw_charge_residual_absolute_e": abs(float(row["raw_charge_residual_e"])),
        "projected_charge_sum_absolute_e": abs(float(row["projected_charge_sum_e"])),
        "ordinary_decomposed_energy_absolute_eV": float(
            parity["energy_absolute_error_eV"]
        ),
        "ordinary_decomposed_charge_max_absolute_e": float(
            parity["charge_max_absolute_error_e"]
        ),
        "ordinary_decomposed_charge_vjp_max_absolute_eV_per_A": float(
            parity["charge_vjp_max_absolute_error_eV_per_A"]
        ),
        "decomposed_repeat_energy_absolute_eV": float(
            parity["repeat_energy_absolute_error_eV"]
        ),
        "decomposed_repeat_charge_max_absolute_e": float(
            parity["repeat_charge_max_absolute_error_e"]
        ),
        "decomposed_repeat_intrinsic_gradient_max_absolute_eV_per_A": float(
            parity["repeat_intrinsic_gradient_max_absolute_error_eV_per_A"]
        ),
        "decomposed_repeat_charge_vjp_max_absolute_eV_per_A": float(
            parity["repeat_charge_vjp_max_absolute_error_eV_per_A"]
        ),
    }
    return all(
        math.isfinite(value) and 0.0 <= value <= float(thresholds[name])
        for name, value in comparisons.items()
    )


def _selection_identity(
    selected: Sequence[Mapping[str, Any]], dataset: Any
) -> list[dict[str, Any]]:
    result = []
    for ordinal, record in enumerate(selected):
        geometry = dataset.geometries[str(record["geometry_handle"])]
        result.append(
            {
                "ordinal": ordinal,
                "selection_index": int(record["selection_index"]),
                "partition": str(record["partition"]),
                "opaque_record_id": str(record["opaque_record_id"]),
                "geometry_handle": str(record["geometry_handle"]),
                "geometry_sha256": str(geometry.sha256),
                "atom_count": len(geometry.atomic_numbers),
                "atomic_numbers": sorted(
                    set(int(value) for value in geometry.atomic_numbers)
                ),
            }
        )
    return result


def _artifact_bindings(
    *,
    execution_git_commit: str,
    execution_git_tree: str,
    protocol_file_sha256: str,
    source: Path,
    mnsol_protocol: Path,
    prior_private: Path,
    selection_manifest: Path,
    selection_manifest_canonical_sha256: str,
    checkpoint: Path,
    runtime_provenance_sha256: str,
    numerical_runtime_sha256: str,
    covered_atomic_numbers: Sequence[int],
    selected_records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "execution_git_commit": execution_git_commit,
        "execution_git_tree": execution_git_tree,
        "protocol_file_sha256": protocol_file_sha256,
        "mnsol_dataset_zip_sha256": sha256_file(source),
        "mnsol_protocol_sha256": sha256_file(mnsol_protocol),
        "prior_private_653_sha256": sha256_file(prior_private),
        "selection_manifest_file_sha256": sha256_file(selection_manifest),
        "selection_manifest_canonical_sha256": (selection_manifest_canonical_sha256),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "runtime_kind": AIMNET_FLOAT64_RUNTIME_VERSION,
        "runtime_provenance_sha256": runtime_provenance_sha256,
        "numerical_runtime_sha256": numerical_runtime_sha256,
        "covered_atomic_numbers": list(covered_atomic_numbers),
        "selected_records": [dict(record) for record in selected_records],
        "thresholds": dict(_thresholds()),
        "claim_boundary": str(protocol["claim_boundary"]),
    }


def validate_source_domain_artifact(
    artifact: Mapping[str, Any], *, expected_bindings: Mapping[str, object]
) -> None:
    """Verify the complete private source-domain artifact and every row gate."""

    if set(expected_bindings) != _ARTIFACT_BINDING_KEYS:
        raise ValueError("source-domain artifact bindings are incomplete or excessive.")
    if (
        artifact.get("artifact") != ARTIFACT
        or artifact.get("schema_version") != 1
        or artifact.get("status") != "complete"
        or artifact.get("do_not_commit") is not True
        or artifact.get("capabilities") != NO_CAPABILITIES
    ):
        raise ValueError("source-domain artifact identity/status drifted.")
    for key in _ARTIFACT_BINDING_KEYS - {"selected_records"}:
        if artifact.get(key) != expected_bindings[key]:
            raise ValueError(f"source-domain artifact binding drifted: {key}.")
    provenance = artifact.get("runtime_provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("source-domain runtime provenance must be a mapping.")
    try:
        json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "source-domain runtime provenance must be finite JSON metadata."
        ) from exc
    if (
        canonical_sha256(provenance) != expected_bindings["runtime_provenance_sha256"]
        or provenance.get("runtime_kind") != AIMNET_FLOAT64_RUNTIME_VERSION
    ):
        raise ValueError("source-domain runtime provenance identity drifted.")
    _validate_numerical_runtime(
        artifact,
        expected_bindings["numerical_runtime_sha256"],
    )
    expected_records = expected_bindings["selected_records"]
    rows = artifact.get("rows")
    if not isinstance(expected_records, Sequence) or isinstance(
        expected_records, (str, bytes)
    ):
        raise TypeError("expected source-domain selection must be a sequence.")
    if not isinstance(rows, list) or len(rows) != len(expected_records):
        raise ValueError("source-domain artifact row count drifted.")
    expected_id_keys = {
        "ordinal",
        "selection_index",
        "partition",
        "opaque_record_id",
        "geometry_handle",
        "geometry_sha256",
        "atom_count",
        "atomic_numbers",
    }
    passed = 0
    covered: set[int] = set()
    for ordinal, (row, expected) in enumerate(zip(rows, expected_records, strict=True)):
        if not isinstance(row, Mapping) or not isinstance(expected, Mapping):
            raise TypeError(
                "source-domain rows and expected identities must be mappings."
            )
        if any(row.get(key) != expected.get(key) for key in expected_id_keys):
            raise ValueError(f"source-domain row {ordinal} identity drifted.")
        covered.update(int(value) for value in row["atomic_numbers"])
        runtime_seconds = row.get("runtime_seconds")
        if (
            isinstance(runtime_seconds, bool)
            or not isinstance(runtime_seconds, (int, float))
            or not math.isfinite(float(runtime_seconds))
            or float(runtime_seconds) < 0.0
        ):
            raise ValueError(f"source-domain row {ordinal} runtime is invalid.")
        if row.get("status") == "measured":
            energy = row.get("energy_eV")
            intrinsic_norm = row.get("intrinsic_gradient_norm_eV_per_A")
            charge_vjp_norm = row.get("charge_vjp_norm_eV_per_A")
            if (
                isinstance(energy, bool)
                or not isinstance(energy, (int, float))
                or not math.isfinite(float(energy))
                or isinstance(intrinsic_norm, bool)
                or not isinstance(intrinsic_norm, (int, float))
                or not math.isfinite(float(intrinsic_norm))
                or float(intrinsic_norm) < 0.0
                or isinstance(charge_vjp_norm, bool)
                or not isinstance(charge_vjp_norm, (int, float))
                or not math.isfinite(float(charge_vjp_norm))
                or float(charge_vjp_norm) < 0.0
            ):
                raise ValueError(
                    f"source-domain row {ordinal} response scalars are invalid."
                )
            raw = np.asarray(row.get("raw_charges_e"), dtype=float)
            projected = np.asarray(row.get("projected_charges_e"), dtype=float)
            if (
                raw.shape != (int(row["atom_count"]),)
                or projected.shape != raw.shape
                or not np.all(np.isfinite(raw))
                or not np.all(np.isfinite(projected))
            ):
                raise ValueError(f"source-domain row {ordinal} charges are invalid.")
            if not math.isclose(
                float(np.sum(raw)),
                float(row.get("raw_charge_residual_e")),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ) or not math.isclose(
                float(np.sum(projected)),
                float(row.get("projected_charge_sum_e")),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"source-domain row {ordinal} charge ledger drifted.")
            expected_projected = raw - float(np.sum(raw)) / len(raw)
            if not np.allclose(
                projected,
                expected_projected,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    f"source-domain row {ordinal} affine charge projection drifted."
                )
            parity = row.get("parity")
            required_parity = {
                "energy_absolute_error_eV",
                "charge_max_absolute_error_e",
                "intrinsic_gradient_max_absolute_error_eV_per_A",
                "charge_vjp_max_absolute_error_eV_per_A",
                "repeat_energy_absolute_error_eV",
                "repeat_charge_max_absolute_error_e",
                "repeat_intrinsic_gradient_max_absolute_error_eV_per_A",
                "repeat_charge_vjp_max_absolute_error_eV_per_A",
                "ordinary_forward_intrinsic_gradient_report_only_max_absolute_error_eV_per_A",
            }
            if (
                not isinstance(parity, Mapping)
                or set(parity) != required_parity
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                    for value in parity.values()
                )
            ):
                raise ValueError(
                    f"source-domain row {ordinal} parity ledger is malformed."
                )
            try:
                gate_passed = _row_passes(row, _thresholds())
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"source-domain row {ordinal} gate ledger is malformed."
                ) from exc
            if row.get("gate_passed") is not gate_passed:
                raise ValueError(f"source-domain row {ordinal} gate result drifted.")
            passed += int(gate_passed)
        elif row.get("status") == "failed":
            if (
                row.get("gate_passed") is not False
                or not isinstance(row.get("error_type"), str)
                or not row.get("error_type")
                or not isinstance(row.get("error"), str)
            ):
                raise ValueError(f"source-domain row {ordinal} failure is malformed.")
        else:
            raise ValueError(f"source-domain row {ordinal} status is invalid.")
    failed = len(rows) - passed
    all_passed = passed == len(rows)
    if (
        sorted(covered) != list(expected_bindings["covered_atomic_numbers"])
        or artifact.get("record_count") != len(rows)
        or artifact.get("passed_record_count") != passed
        or artifact.get("failed_record_count") != failed
        or artifact.get("all_gates_passed") is not all_passed
    ):
        raise ValueError("source-domain artifact aggregate ledger drifted.")
    observed_sha256 = artifact.get("artifact_sha256")
    expected_sha256 = canonical_sha256(
        {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    )
    if observed_sha256 != expected_sha256:
        raise ValueError("source-domain artifact SHA256 is invalid.")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    from ase import Atoms

    commit, tree = _clean_commit()
    source = args.source.expanduser().resolve(strict=True)
    mnsol_protocol = args.mnsol_protocol.expanduser().resolve(strict=True)
    prior_private = args.prior_private.expanduser().resolve(strict=True)
    selection_manifest = args.selection_manifest.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    protocol_path = args.protocol.expanduser().resolve(strict=True)
    output = _private_output(args.output)
    if output.exists():
        raise FileExistsError(f"Private source-domain artifact exists: {output}")

    protocol = _object(protocol_path, name="source-domain protocol")
    manifest = _object(selection_manifest, name="target-free selection manifest")
    prior = _object(prior_private, name="prior private MNSol artifact")
    _validate_protocol(
        protocol,
        source=source,
        mnsol_protocol=mnsol_protocol,
        prior_private=prior_private,
        selection_manifest=selection_manifest,
        checkpoint=checkpoint,
    )
    if (
        manifest.get("schema") != SELECTION_SCHEMA
        or manifest.get("artifact_sha256") != _canonical_manifest_sha256(manifest)
        or manifest.get("artifact_sha256")
        != protocol["pinned_inputs"]["selection_manifest_canonical_sha256"]
    ):
        raise ValueError("Target-free selection manifest identity drifted.")
    if (
        prior.get("artifact") != PRIOR_ARTIFACT
        or prior.get("complete_panel") is not True
        or prior.get("do_not_commit") is not True
    ):
        raise ValueError("Prior private MNSol artifact identity drifted.")
    prior_records = prior.get("records")
    panel = manifest.get("panel")
    if (
        not isinstance(prior_records, list)
        or len(prior_records) != EXPECTED_PRIOR_RECORD_COUNT
        or not isinstance(panel, Mapping)
        or not isinstance(panel.get("records"), list)
    ):
        raise ValueError("Source-domain input record sets are malformed.")
    dataset = load_mnsol_v2012(source, load_mnsol_protocol(mnsol_protocol))
    selected, atomic_numbers = select_audit_geometries(
        dataset=dataset,
        prior_records=prior_records,
        manifest_records=panel["records"],
    )
    expected_numbers = tuple(protocol["selection"]["expected_atomic_numbers"])
    if (
        atomic_numbers != expected_numbers
        or len(selected) != protocol["selection"]["expected_geometry_count"]
    ):
        raise ValueError("Source-domain target-free coverage selection drifted.")

    torch = _configure_torch_threads()
    calculator = AIMNet2ReconstructedFloat64SourceCalculator(
        model_path=checkpoint, device="cpu"
    )
    runtime_provenance, runtime_provenance_sha256 = _runtime_identity(calculator)
    selected_records = _selection_identity(selected, dataset)
    rows = []
    for ordinal, record in enumerate(selected):
        geometry = dataset.geometries[str(record["geometry_handle"])]
        atoms = Atoms(
            numbers=geometry.atomic_numbers,
            positions=geometry.coordinates_angstrom,
            info={"charge": 0, "mult": 1},
        )
        started = time.perf_counter()
        try:
            response = calculator.charge_position_response(
                atoms, _cotangent(len(atoms))
            )
            state = response.charge_state
            row: dict[str, Any] = {
                "ordinal": ordinal,
                "selection_index": int(record["selection_index"]),
                "partition": str(record["partition"]),
                "opaque_record_id": str(record["opaque_record_id"]),
                "geometry_handle": str(record["geometry_handle"]),
                "geometry_sha256": geometry.sha256,
                "atom_count": len(atoms),
                "atomic_numbers": sorted(set(int(value) for value in atoms.numbers)),
                "status": "measured",
                "energy_eV": float(state.energy_ev),
                "raw_charges_e": np.asarray(state.raw_charges_e).tolist(),
                "projected_charges_e": np.asarray(state.charges_e).tolist(),
                "raw_charge_residual_e": float(state.raw_charge_residual_e),
                "projected_charge_sum_e": float(np.sum(state.charges_e)),
                "intrinsic_gradient_norm_eV_per_A": float(
                    np.linalg.norm(response.intrinsic_energy_gradient_ev_per_angstrom)
                ),
                "charge_vjp_norm_eV_per_A": float(
                    np.linalg.norm(response.charge_position_vjp_ev_per_angstrom)
                ),
                "parity": calculator.last_ordinary_decomposed_parity(),
            }
            row["gate_passed"] = _row_passes(row, protocol["thresholds"])
        except Exception as exc:  # retain negative evidence for every geometry
            row = {
                "ordinal": ordinal,
                "selection_index": int(record["selection_index"]),
                "partition": str(record["partition"]),
                "opaque_record_id": str(record["opaque_record_id"]),
                "geometry_handle": str(record["geometry_handle"]),
                "geometry_sha256": geometry.sha256,
                "atom_count": len(atoms),
                "atomic_numbers": sorted(set(int(value) for value in atoms.numbers)),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "gate_passed": False,
            }
        row["runtime_seconds"] = time.perf_counter() - started
        rows.append(row)

    numerical_runtime = numerical_runtime_identity(torch_module=torch)
    numerical_runtime_sha256 = canonical_sha256(numerical_runtime)
    artifact_bindings = _artifact_bindings(
        execution_git_commit=commit,
        execution_git_tree=tree,
        protocol_file_sha256=sha256_file(protocol_path),
        source=source,
        mnsol_protocol=mnsol_protocol,
        prior_private=prior_private,
        selection_manifest=selection_manifest,
        selection_manifest_canonical_sha256=str(manifest["artifact_sha256"]),
        checkpoint=checkpoint,
        runtime_provenance_sha256=runtime_provenance_sha256,
        numerical_runtime_sha256=numerical_runtime_sha256,
        covered_atomic_numbers=atomic_numbers,
        selected_records=selected_records,
        protocol=protocol,
    )
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "complete" if len(rows) == len(selected) else "incomplete",
        "do_not_commit": True,
        "execution_git_commit": commit,
        "execution_git_tree": tree,
        "protocol_file_sha256": sha256_file(protocol_path),
        "mnsol_dataset_zip_sha256": sha256_file(source),
        "mnsol_protocol_sha256": sha256_file(mnsol_protocol),
        "prior_private_653_sha256": sha256_file(prior_private),
        "selection_manifest_file_sha256": sha256_file(selection_manifest),
        "selection_manifest_canonical_sha256": str(manifest["artifact_sha256"]),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "runtime_kind": AIMNET_FLOAT64_RUNTIME_VERSION,
        "runtime_provenance": runtime_provenance,
        "runtime_provenance_sha256": runtime_provenance_sha256,
        "numerical_runtime": numerical_runtime,
        "numerical_runtime_sha256": numerical_runtime_sha256,
        "covered_atomic_numbers": list(atomic_numbers),
        "record_count": len(rows),
        "passed_record_count": sum(bool(row["gate_passed"]) for row in rows),
        "failed_record_count": sum(not bool(row["gate_passed"]) for row in rows),
        "all_gates_passed": all(bool(row["gate_passed"]) for row in rows),
        "thresholds": _thresholds(),
        "rows": rows,
        "capabilities": dict(NO_CAPABILITIES),
        "claim_boundary": protocol["claim_boundary"],
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    validate_source_domain_artifact(
        payload,
        expected_bindings=artifact_bindings,
    )
    _assert_same_clean_commit(commit, tree)
    write_json_atomic(output, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mnsol-protocol", type=Path, required=True)
    parser.add_argument("--prior-private", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            BENCHMARK_DIR
            / "route2-aimnet2-mnsol653-source-domain-prequalification-protocol-v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    with single_threaded_numerics():
        payload = _run(args)
    print(args.output.expanduser().resolve())
    print(payload["artifact_sha256"])
    print(json.dumps({"all_gates_passed": payload["all_gates_passed"]}))
    if not payload["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
