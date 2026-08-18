#!/usr/bin/env python3
"""Finalize the private 653-record smooth-ddPCM run as public-safe evidence.

The expensive measurement is bound to the clean commit recorded in every
private shard.  This separate clean-commit reducer revalidates the pinned
MNSol inputs, every task fingerprint and numerical ledger, the ordered shard
set, and the raw public aggregate before adding machine-readable provenance
and fail-closed capability metadata.  No row-level MNSol data are emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, BENCHMARK_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from benchmark_core import canonical_json_bytes, sha256_file, write_json_atomic
from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
import run_mnsol_aimnet2_smooth_partition_ddpcm as runner
from maple.solvation.release.evidence import (
    RepositorySnapshot,
    collect_loaded_repository_sources,
    committed_source_hashes,
    runtime_record,
)

FINALIZER_ARTIFACT = "route2-mnsol-aimnet2-smooth-partition-ddpcm-full-v1"
CAPABILITIES_CLOSED = {"E": False, "F": False, "H": False, "M": False, "V": False}
REQUIRED_SOURCE_PATHS = (
    "docs/implicit-solvation/benchmarks/finalize_mnsol_aimnet2_smooth_partition_ddpcm.py",
    "docs/implicit-solvation/benchmarks/run_mnsol_aimnet2_smooth_partition_ddpcm.py",
)


def _object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ("git", *args),
        cwd=root,
        check=False,
        capture_output=True,
        text=not binary,
    )
    if result.returncode != 0:
        diagnostic = result.stderr or result.stdout
        if isinstance(diagnostic, bytes):
            diagnostic = diagnostic.decode("utf-8", errors="replace")
        raise RuntimeError(f"git {' '.join(args)} failed: {str(diagnostic).strip()}")
    return result.stdout


def _git_tree(root: Path, commit: str) -> str:
    value = str(_git(root, "rev-parse", f"{commit}^{{tree}}")).strip()
    if len(value) != 40:
        raise RuntimeError("Measurement Git tree is not a full object identity.")
    return value


def _git_source_hashes(
    root: Path, commit: str, relative_paths: Sequence[str]
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in sorted(set(relative_paths)):
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Invalid repository source path: {relative!r}.")
        blob = _git(root, "show", f"{commit}:{relative}", binary=True)
        if not isinstance(blob, bytes):
            raise RuntimeError("Git source capture unexpectedly returned text.")
        hashes[relative] = hashlib.sha256(blob).hexdigest()
    if not hashes:
        raise ValueError("Measurement source binding cannot be empty.")
    return hashes


def _sha256_manifest(entries: Sequence[tuple[str, str]]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(entries))).hexdigest()


def _validate_private(
    *,
    private_dir: Path,
    tasks: Sequence[Mapping[str, Any]],
    prior_private: Path,
    prior_public: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[str, str]]]:
    private_path = private_dir / "private.json"
    private = _object(private_path, label="private aggregate")
    if (
        private.get("artifact") != runner.ARTIFACT
        or private.get("schema_version") != 1
        or private.get("complete_panel") is not True
        or private.get("do_not_commit") is not True
        or private.get("prior_private_sha256") != sha256_file(prior_private)
        or private.get("prior_public_sha256") != sha256_file(prior_public)
    ):
        raise ValueError("Private aggregate identity/provenance contract drifted.")
    execution_head = str(private.get("execution_git_head", ""))
    if len(execution_head) != 40:
        raise ValueError("Private aggregate lacks a full measurement Git commit.")
    records = private.get("records")
    if not isinstance(records, list) or len(records) != runner.EXPECTED_RECORD_COUNT:
        raise ValueError("Private aggregate must contain exactly 653 records.")
    if len(tasks) != runner.EXPECTED_RECORD_COUNT:
        raise ValueError("Pinned MNSol task reconstruction must contain 653 records.")

    shard_dir = private_dir / "shards"
    shard_paths = sorted(shard_dir.glob("*.json"))
    expected_names = [
        f"{index:04d}.json" for index in range(runner.EXPECTED_RECORD_COUNT)
    ]
    if [path.name for path in shard_paths] != expected_names:
        raise ValueError("Private shard set is incomplete or has unexpected names.")

    validated: list[dict[str, Any]] = []
    shard_hashes: list[tuple[str, str]] = []
    for index, (task, record, shard_path) in enumerate(
        zip(tasks, records, shard_paths, strict=True)
    ):
        if not isinstance(record, Mapping):
            raise TypeError(f"Private record {index} is not a mapping.")
        shard = _object(shard_path, label=f"private shard {index}")
        if shard != record:
            raise ValueError(
                f"Private aggregate record {index} differs from its shard."
            )
        if not runner._valid_shard(shard, task=task, execution_git_head=execution_head):
            raise ValueError(f"Private shard {index} failed exact task/ledger replay.")
        validated.append(dict(record))
        shard_hashes.append((shard_path.name, sha256_file(shard_path)))
    return private, validated, shard_hashes


def _validate_raw_public(
    raw: Mapping[str, Any],
    *,
    private: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    prior_public: Mapping[str, Any],
) -> None:
    if (
        raw.get("artifact") != runner.ARTIFACT
        or raw.get("schema_version") != 1
        or raw.get("complete_panel") is not True
        or raw.get("execution_git_head") != private.get("execution_git_head")
        or raw.get("record_count") != runner.EXPECTED_RECORD_COUNT
        or raw.get("unique_geometry_count")
        != len({str(record["geometry_handle"]) for record in records})
    ):
        raise ValueError("Raw public aggregate identity/completeness drifted.")
    source = raw.get("source")
    if not isinstance(source, Mapping) or (
        source.get("prior_artifact") != runner.PRIOR_ARTIFACT
        or source.get("prior_private_sha256") != private.get("prior_private_sha256")
        or source.get("prior_public_sha256") != private.get("prior_public_sha256")
        or source.get("checkpoint") != prior_public.get("checkpoint")
        or source.get("continuum_field_supplied_to_aimnet2") is not False
        or source.get("electronic_scf_iteration") is not False
    ):
        raise ValueError("Raw public source/checkpoint provenance drifted.")
    if raw.get("aggregate_metrics") != runner._metrics(records):
        raise ValueError("Raw public aggregate metrics do not replay.")
    if raw.get("partition_metrics") != runner._grouped_metrics(
        records, lambda record: str(record["partition"])
    ):
        raise ValueError("Raw public partition metrics do not replay.")
    if raw.get("solvent_metrics") != runner._grouped_metrics(
        records, lambda record: str(record["canonical_solvent"])
    ):
        raise ValueError("Raw public solvent metrics do not replay.")
    if raw.get("atom_count_bin_metrics") != runner._grouped_metrics(
        records, runner._atom_bin
    ):
        raise ValueError("Raw public atom-count metrics do not replay.")
    if raw.get("method") != {
        "continuum": "smooth-partition-harmonic-ddpcm",
        "surface_lmax": 4,
        "partition_lmax": runner.AIMNET2_SMOOTH_PARTITION_PARTITION_LMAX,
        "maximum_algebraic_degree": (runner.HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE),
        "transition_width_angstrom2": 0.18,
        "nonpolar": "PySCF-2.13.1 SMD-CDS reused from frozen prior artifact",
        "fit_or_calibration": False,
    }:
        raise ValueError("Raw public method identity drifted.")
    preflight = raw.get("factor_degree_preflight")
    if not isinstance(preflight, Mapping) or preflight != {
        "maximum_factor_count": max(int(record["factor_count"]) for record in records),
        "maximum_required_algebraic_degree": max(
            int(record["required_algebraic_degree"]) for record in records
        ),
        "algebraic_degree_bound": runner.HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE,
        "failed_record_count": 0,
    }:
        raise ValueError("Raw public algebraic-degree preflight does not replay.")
    runner._assert_public_safe(raw)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--prior-private", type=Path, required=True)
    parser.add_argument("--prior-public", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--raw-public", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repository = RepositorySnapshot.capture(REPO_ROOT)
    private_dir = args.private_output_dir.expanduser().resolve(strict=True)
    try:
        private_dir.relative_to((REPO_ROOT / ".omx").resolve())
    except ValueError as exc:
        raise ValueError(
            "Private smooth-ddPCM evidence must remain below .omx."
        ) from exc

    protocol = load_mnsol_protocol(args.protocol)
    dataset = load_mnsol_v2012(args.source, protocol)
    prior_private = _object(args.prior_private, label="prior private aggregate")
    tasks, _degree_preflight = runner._prepare_tasks(dataset, prior_private)
    private, records, shard_hashes = _validate_private(
        private_dir=private_dir,
        tasks=tasks,
        prior_private=args.prior_private,
        prior_public=args.prior_public,
    )
    prior_public = _object(args.prior_public, label="prior public aggregate")
    raw_public = _object(args.raw_public, label="raw public aggregate")
    _validate_raw_public(
        raw_public,
        private=private,
        records=records,
        prior_public=prior_public,
    )

    measurement_head = str(private["execution_git_head"])
    if (
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", measurement_head, repository.head),
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        != 0
    ):
        raise RuntimeError(
            "Measurement commit is not an ancestor of the reducer commit."
        )
    measurement_tree = _git_tree(REPO_ROOT, measurement_head)

    loaded_sources = collect_loaded_repository_sources(
        REPO_ROOT, required_paths=REQUIRED_SOURCE_PATHS
    )
    aggregation_sources = committed_source_hashes(repository, loaded_sources)
    measurement_paths = tuple(
        path
        for path in loaded_sources
        if subprocess.run(
            ("git", "cat-file", "-e", f"{measurement_head}:{path}"),
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    measurement_sources = _git_source_hashes(
        REPO_ROOT, measurement_head, measurement_paths
    )

    private_path = private_dir / "private.json"
    record_payload_sha256 = hashlib.sha256(canonical_json_bytes(records)).hexdigest()
    scientific_measurement = {
        "record_payload_sha256": record_payload_sha256,
        "aggregate_metrics": raw_public["aggregate_metrics"],
        "partition_metrics": raw_public["partition_metrics"],
        "solvent_metrics": raw_public["solvent_metrics"],
        "atom_count_bin_metrics": raw_public["atom_count_bin_metrics"],
        "factor_degree_preflight": raw_public["factor_degree_preflight"],
        "method": raw_public["method"],
        "dataset": raw_public["dataset"],
    }
    finalized = dict(raw_public)
    finalized.update(
        {
            "artifact": FINALIZER_ARTIFACT,
            "visibility": "public-aggregate-only",
            "do_not_commit": False,
            "capabilities": CAPABILITIES_CLOSED,
            "measurement_sha256": hashlib.sha256(
                canonical_json_bytes(scientific_measurement)
            ).hexdigest(),
            "measurement_provenance": {
                "git_head": measurement_head,
                "git_tree": measurement_tree,
                "source_files_sha256": measurement_sources,
                "private_artifact_sha256": sha256_file(private_path),
                "raw_public_artifact_sha256": sha256_file(args.raw_public),
                "record_payload_sha256": record_payload_sha256,
                "shards": {
                    "count": len(shard_hashes),
                    "provenance_sha256": _sha256_manifest(shard_hashes),
                },
            },
            "aggregation_provenance": {
                "git_head": repository.head,
                "git_tree": repository.tree,
                "source_files_sha256": aggregation_sources,
                "runtime": runtime_record(),
            },
            "decision": {
                "complete_653_record_accuracy_panel": True,
                "accuracy_admitted": False,
                "force_admitted": False,
                "hessian_admitted": False,
                "variational_electronic_scf_admitted": False,
                "profile_enabled": False,
                "reason": (
                    "The complete scalar panel is evidence, not an automatic "
                    "capability transition; force, order-convergence, path, HVP, "
                    "and workflow gates remain separately open."
                ),
            },
            "claim_boundary": (
                str(raw_public["claim_boundary"])
                + " The finalizer replays all 653 private ledgers and publishes "
                "only aggregate-safe hashes and statistics; E/F/H/V/M remain false."
            ),
        }
    )
    runner._assert_public_safe(finalized)
    repository.assert_unchanged()
    write_json_atomic(args.public_output, finalized)
    print(
        json.dumps(
            {
                "artifact": finalized["artifact"],
                "measurement_sha256": finalized["measurement_sha256"],
                "record_count": finalized["record_count"],
                "capabilities": finalized["capabilities"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
