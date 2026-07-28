#!/usr/bin/env python3
"""Aggregate complete MNSol partition shards into a two-member matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

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
import run_mnsol_macepolar_response_ablation as runner

REPO_ROOT = Path(__file__).resolve().parents[3]

ALLOWED_METHODS = ("mace_fixed_l1", "mace_scf_l1")
FULL_METHODS = tuple(runner.ABLATION_METHODS)
SUPPORTED_PARTITIONS = frozenset({"development"})
AGGREGATOR_ARTIFACT_NAME = "route2-mnsol-macepolar-two-member-matrix-v1"
SOURCE_RUN_KIND = "partition-record-shard"
AGGREGATE_RUN_KIND = "partition-two-member-matrix"
REQUIRED_STAGE = "scf"
SCHEMA_VERSION = 1
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
        "schema_version": SCHEMA_VERSION,
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
    experimental_kcal_mol: float,
) -> dict[str, dict[str, float]]:
    if tuple(methods) != FULL_METHODS:
        raise ValueError(
            f"Record {selection_index} does not contain the full SCF "
            "method set in source-runner order."
        )

    normalized: dict[str, dict[str, float]] = {}
    for method in FULL_METHODS:
        payload = methods[method]
        if not isinstance(payload, Mapping):
            raise ValueError(f"Record {selection_index} method {method} is malformed.")
        if not set(METHOD_FIELDS).issubset(payload):
            raise ValueError(f"Record {selection_index} method {method} field drifted.")
        row = {
            field: _finite_float(
                payload[field],
                label=f"Record {selection_index} method {method} {field}",
            )
            for field in METHOD_FIELDS
        }
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
            "mace_scf_l1": ("same-root c*=M(P(c*)); DeltaE_model+U(c*)+G_CDS"),
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
            "geometries), and confirmation remains sealed."
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
            "geometries). It is not full-653 MNSol certification and does not "
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
