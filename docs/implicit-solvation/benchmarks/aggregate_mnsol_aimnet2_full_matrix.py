#!/usr/bin/env python3
"""Aggregate all frozen AIMNet2 MNSol shards without publishing row data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, BENCHMARK_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from benchmark_core import canonical_json_bytes, sha256_file, write_json_atomic
from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
from mnsol_partition import validate_frozen_mnsol_partition_selection
import run_mnsol_aimnet2_multisolvent_pilot as runner

ARTIFACT = "route2-mnsol-aimnet2-full-frozen-charge-matrix-v1"
SOURCE_ARTIFACT = "route2-mnsol-aimnet2-multisolvent-pilot-v1"
SOURCE_RUN_KIND = "partition-record-shard"
METHODS = tuple(method for method, _ in runner.METHODS)
PARTITIONS = ("development", "confirmation")
EXPECTED_RECORD_COUNT = 653
EXPECTED_PARTITION_COUNTS = {"development": 505, "confirmation": 148}
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "records",
        "entry_number",
        "geometry_handle",
        "solute_name",
        "formula",
        "coordinates",
        "charges_e",
        "experimental_delta_g_kcal_mol",
        "opaque_record_id",
    }
)


def _object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _clean_head() -> str:
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("MNSol aggregation requires a clean source checkout.")
    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(head) != 40:
        raise RuntimeError("Could not resolve aggregation Git commit.")
    return head


def _private_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = (REPO_ROOT / ".omx").resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Row-level MNSol aggregate must remain below .omx.") from exc
    return resolved


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _method_metrics(
    records: Sequence[Mapping[str, Any]], method: str
) -> dict[str, float | int]:
    errors = np.asarray(
        [record["methods"][method]["signed_error_kcal_mol"] for record in records],
        dtype=float,
    )
    predicted = np.asarray(
        [record["methods"][method]["total_solvation_kcal_mol"] for record in records],
        dtype=float,
    )
    experimental = np.asarray(
        [record["experimental_delta_g_kcal_mol"] for record in records], dtype=float
    )
    absolute = np.abs(errors)
    if (
        errors.size == 0
        or not np.all(np.isfinite(errors))
        or not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(experimental))
    ):
        raise ValueError("Full MNSol metrics require finite nonempty records.")
    return {
        "record_count": int(errors.size),
        "mean_signed_error_kcal_mol": float(np.mean(errors)),
        "mean_absolute_error_kcal_mol": float(np.mean(absolute)),
        "root_mean_square_error_kcal_mol": float(np.sqrt(np.mean(errors**2))),
        "median_absolute_error_kcal_mol": float(np.median(absolute)),
        "p90_absolute_error_kcal_mol": float(np.quantile(absolute, 0.90)),
        "p95_absolute_error_kcal_mol": float(np.quantile(absolute, 0.95)),
        "p99_absolute_error_kcal_mol": float(np.quantile(absolute, 0.99)),
        "maximum_absolute_error_kcal_mol": float(np.max(absolute)),
        "mean_predicted_delta_g_kcal_mol": float(np.mean(predicted)),
        "mean_experimental_delta_g_kcal_mol": float(np.mean(experimental)),
        "maximum_half_coupling_identity_error_ev": float(
            max(
                record["methods"][method]["half_coupling_identity_error_ev"]
                for record in records
            )
        ),
    }


def _paired_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    pcm = np.asarray(
        [record["methods"]["ddpcm"]["absolute_error_kcal_mol"] for record in records]
    )
    cosmo = np.asarray(
        [record["methods"]["ddcosmo"]["absolute_error_kcal_mol"] for record in records]
    )
    energy_difference = np.asarray(
        [
            record["methods"]["ddcosmo"]["total_solvation_kcal_mol"]
            - record["methods"]["ddpcm"]["total_solvation_kcal_mol"]
            for record in records
        ]
    )
    tolerance = 1.0e-12
    return {
        "record_count": len(records),
        "mean_ddcosmo_minus_ddpcm_kcal_mol": float(np.mean(energy_difference)),
        "ddcosmo_lower_absolute_error_count": int(np.sum(cosmo < pcm - tolerance)),
        "ddpcm_lower_absolute_error_count": int(np.sum(pcm < cosmo - tolerance)),
        "absolute_error_tie_count": int(np.sum(np.abs(cosmo - pcm) <= tolerance)),
        "tie_tolerance_kcal_mol": tolerance,
    }


def _metric_block(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "record_count": len(records),
        "unique_geometry_count": len({record["geometry_handle"] for record in records}),
        "methods": {method: _method_metrics(records, method) for method in METHODS},
        "paired_continuum_comparison": _paired_metrics(records),
    }


def _grouped_metrics(
    records: Sequence[Mapping[str, Any]],
    key: Callable[[Mapping[str, Any]], str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[key(record)].append(record)
    return {name: _metric_block(grouped[name]) for name in sorted(grouped)}


def _atom_bin(record: Mapping[str, Any]) -> str:
    count = int(record["atom_count"])
    if count <= 5:
        return "01-05"
    if count <= 10:
        return "06-10"
    if count <= 20:
        return "11-20"
    return "21-plus"


def _cluster_balanced_metrics(
    records: Sequence[Mapping[str, Any]], method: str
) -> dict[str, float | int]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[str(record["geometry_handle"])].append(
            float(record["methods"][method]["signed_error_kcal_mol"])
        )
    cluster_errors = np.asarray(
        [float(np.mean(values)) for values in grouped.values()], dtype=float
    )
    return {
        "unique_geometry_count": len(cluster_errors),
        "mean_signed_error_kcal_mol": float(np.mean(cluster_errors)),
        "mean_absolute_cluster_error_kcal_mol": float(np.mean(np.abs(cluster_errors))),
        "root_mean_square_cluster_error_kcal_mol": float(
            np.sqrt(np.mean(cluster_errors**2))
        ),
    }


def _validate_method_record(record: Mapping[str, Any], *, index: int) -> None:
    methods = record.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(METHODS):
        raise ValueError(f"Shard {index} continuum method set drifted.")
    for method in METHODS:
        values = methods[method]
        if not isinstance(values, Mapping):
            raise ValueError(f"Shard {index} method {method} is invalid.")
        for field in (
            "polarization_energy_hartree",
            "polarization_energy_kcal_mol",
            "smd_cds_energy_kcal_mol",
            "total_solvation_kcal_mol",
            "signed_error_kcal_mol",
            "absolute_error_kcal_mol",
            "half_coupling_identity_error_ev",
        ):
            _finite(values.get(field), label=f"Shard {index} {method} {field}")
        expected_error = float(values["total_solvation_kcal_mol"]) - float(
            record["experimental_delta_g_kcal_mol"]
        )
        if not math.isclose(
            float(values["signed_error_kcal_mol"]),
            expected_error,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"Shard {index} method {method} error ledger drifted.")


def _validate_shard(
    private: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    selected: Any,
    index: int,
    selection_manifest: Mapping[str, Any],
    checkpoint_sha256: str,
) -> Mapping[str, Any]:
    for payload, label in ((private, "private"), (summary, "summary")):
        if (
            payload.get("artifact") != SOURCE_ARTIFACT
            or payload.get("schema_version") != 1
            or payload.get("complete_panel") is not False
            or payload.get("do_not_commit") is not True
            or payload.get("run_kind") != SOURCE_RUN_KIND
            or payload.get("protocol_fingerprint")
            != selection_manifest.get("protocol_fingerprint")
            or payload.get("selection_fingerprint")
            != selection_manifest.get("selection_fingerprint")
            or payload.get("checkpoint", {}).get("sha256") != checkpoint_sha256
        ):
            raise ValueError(f"Shard {index} {label} contract drifted.")
    records = private.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError(f"Shard {index} must contain exactly one private record.")
    record = records[0]
    item = selected.eligible_record
    expected = {
        "selection_index": index,
        "canonical_solvent": selected.canonical_solvent,
        "partition": item.partition,
        "opaque_record_id": selected.opaque_record_id,
        "entry_number": item.record.entry_number,
        "geometry_handle": item.record.geometry_handle,
        "geometry_sha256": item.geometry.sha256,
        "prior_pilot_geometry_overlap": selected.prior_pilot_geometry_overlap,
        "experimental_delta_g_kcal_mol": item.record.delta_g_kcal_mol,
    }
    if any(record.get(field) != value for field, value in expected.items()):
        raise ValueError(f"Shard {index} does not match its frozen selection row.")
    _validate_method_record(record, index=index)
    if summary.get("selection_indices") != [index]:
        raise ValueError(f"Shard {index} public selection index drifted.")
    if summary.get("aggregate_metrics") != {
        method: runner._metrics([record], method) for method in METHODS
    }:
        raise ValueError(f"Shard {index} public single-row metrics drifted.")
    if summary.get("paired_method_comparison") != runner._paired_method_comparison(
        [record]
    ):
        raise ValueError(f"Shard {index} public paired metrics drifted.")
    if private.get("execution_git_head") != summary.get("execution_git_head"):
        raise ValueError(f"Shard {index} execution commit disagrees across outputs.")
    return record


def _load_partition(
    *,
    dataset: Any,
    protocol: Any,
    pilot_manifest: Mapping[str, Any],
    selection_path: Path,
    shard_root: Path,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], list[Mapping[str, Any]], list[dict[str, Any]]]:
    manifest = _object(selection_path, label="selection manifest")
    selection = validate_frozen_mnsol_partition_selection(
        manifest, dataset, protocol, pilot_manifest
    )
    partition = str(manifest["partition"])
    if len(selection) != EXPECTED_PARTITION_COUNTS[partition]:
        raise ValueError(f"Frozen {partition} selection count drifted.")
    records: list[Mapping[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for index, selected in enumerate(selection):
        directory = shard_root / f"index-{index:03d}"
        private_path = directory / "private.json"
        summary_path = directory / "summary.json"
        private = _object(private_path, label=f"{partition} private shard {index}")
        summary = _object(summary_path, label=f"{partition} summary shard {index}")
        records.append(
            _validate_shard(
                private,
                summary,
                selected=selected,
                index=index,
                selection_manifest=manifest,
                checkpoint_sha256=checkpoint_sha256,
            )
        )
        provenance.append(
            {
                "partition": partition,
                "selection_index": index,
                "execution_git_head": private["execution_git_head"],
                "private_sha256": sha256_file(private_path),
                "summary_sha256": sha256_file(summary_path),
            }
        )
    return manifest, records, provenance


def _source_hashes() -> dict[str, str]:
    paths = (
        "docs/implicit-solvation/benchmarks/aggregate_mnsol_aimnet2_full_matrix.py",
        "docs/implicit-solvation/benchmarks/benchmark_core.py",
        "docs/implicit-solvation/benchmarks/mnsol_dataset.py",
        "docs/implicit-solvation/benchmarks/mnsol_partition.py",
        "docs/implicit-solvation/benchmarks/mnsol_pilot.py",
        "docs/implicit-solvation/benchmarks/run_mnsol_aimnet2_multisolvent_pilot.py",
    )
    return {path: sha256_file(REPO_ROOT / path) for path in paths}


def _assert_public_safe(value: object, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        leaked = FORBIDDEN_PUBLIC_KEYS.intersection(value)
        if leaked:
            raise ValueError(f"Public aggregate leaks private keys at {path}: {leaked}")
        for key, item in value.items():
            _assert_public_safe(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_public_safe(item, path=f"{path}[{index}]")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate the complete AIMNet2 MNSol matrix."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--pilot-selection", type=Path, required=True)
    parser.add_argument("--development-selection", type=Path, required=True)
    parser.add_argument("--development-shards", type=Path, required=True)
    parser.add_argument("--confirmation-selection", type=Path, required=True)
    parser.add_argument("--confirmation-shards", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name in (
        "source",
        "protocol",
        "pilot_selection",
        "development_selection",
        "confirmation_selection",
        "checkpoint",
    ):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, name, path)
    for name in ("development_shards", "confirmation_shards"):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(path)
        setattr(args, name, path)
    args.private_output = _private_path(args.private_output)
    args.public_output = args.public_output.expanduser().resolve()
    aggregation_head = _clean_head()

    protocol = load_mnsol_protocol(args.protocol)
    dataset = load_mnsol_v2012(args.source, protocol)
    pilot_manifest = _object(args.pilot_selection, label="pilot selection")
    checkpoint_sha256 = sha256_file(args.checkpoint)
    development_manifest, development, development_provenance = _load_partition(
        dataset=dataset,
        protocol=protocol,
        pilot_manifest=pilot_manifest,
        selection_path=args.development_selection,
        shard_root=args.development_shards,
        checkpoint_sha256=checkpoint_sha256,
    )
    confirmation_manifest, confirmation, confirmation_provenance = _load_partition(
        dataset=dataset,
        protocol=protocol,
        pilot_manifest=pilot_manifest,
        selection_path=args.confirmation_selection,
        shard_root=args.confirmation_shards,
        checkpoint_sha256=checkpoint_sha256,
    )
    records = development + confirmation
    if len(records) != EXPECTED_RECORD_COUNT:
        raise ValueError("Complete AIMNet2 MNSol record count drifted.")
    if len({record["opaque_record_id"] for record in records}) != len(records):
        raise ValueError("Complete AIMNet2 MNSol matrix contains duplicate rows.")
    provenance = development_provenance + confirmation_provenance
    execution_heads = {item["execution_git_head"] for item in provenance}
    if execution_heads != {aggregation_head}:
        raise ValueError(
            "Every MNSol shard and aggregation must use the same source commit."
        )
    provenance_digest = hashlib.sha256(canonical_json_bytes(provenance)).hexdigest()

    private = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "visibility": "private-user-supplied-mnsol-row-level",
        "do_not_commit": True,
        "complete_panel": True,
        "execution_git_head": aggregation_head,
        "records": records,
        "shard_provenance": provenance,
    }
    write_json_atomic(args.private_output, private)

    untouched_confirmation = [
        record for record in confirmation if not record["prior_pilot_geometry_overlap"]
    ]
    public: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "visibility": "public-aggregate-only",
        "do_not_commit": False,
        "complete_panel": True,
        "run_kind": "complete-development-confirmation-matrix",
        "status": "complete-not-a-force-scalar-admission",
        "execution_git_head": aggregation_head,
        "claim_boundary": (
            "This is the complete 653-record neutral absolute MNSol matrix inside "
            "the frozen Route-2 domain for one-shot AIMNet2 NQE point charges, "
            "full-resolution pyddx ddPCM or separately named scaled ddCOSMO, and "
            "PySCF 2.13.1 SMD-CDS. No model output or experimental value selected, "
            "fit, calibrated, tempered, or changed the method. The result is "
            "chemical-accuracy evidence for this energy protocol, not the lower-"
            "order harmonic force scalar, strict original SMD, E/F/H/V/M admission, "
            "or an optimizer, frequency, transition-state, or dynamics claim."
        ),
        "scientific_identity": {
            "solute_source": "one-shot AIMNet2 NQE point monopoles",
            "continuum_equations": ["pyddx ddPCM", "pyddx scaled ddCOSMO"],
            "continuum_resolution": {"lmax": 15, "n_lebedev": 1202},
            "nonpolar_model": "PySCF 2.13.1 SMD-CDS",
            "electronic_scf_iteration": False,
            "continuum_field_supplied_to_aimnet2": False,
            "strict_original_smd_equivalence": False,
            "same_as_harmonic_force_scalar": False,
        },
        "dataset": {
            "name": "MNSol",
            "version": "2012",
            "source_artifact_sha256": dataset.source_artifact_sha256,
            "table_sha256": dataset.table_sha256,
            "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
            "standard_state": protocol.standard_state,
            "temperature_k": protocol.temperature_k,
            "row_level_data_emitted": False,
        },
        "coverage": {
            "eligible_record_count": EXPECTED_RECORD_COUNT,
            "completed_record_count": len(records),
            "failed_record_count": 0,
            "coverage_fraction": 1.0,
            "unique_geometry_count": len(
                {record["geometry_handle"] for record in records}
            ),
            "partition_record_counts": dict(
                sorted(Counter(record["partition"] for record in records).items())
            ),
            "solvent_record_counts": dict(
                sorted(
                    Counter(record["canonical_solvent"] for record in records).items()
                )
            ),
            "prior_pilot_overlap_record_count": sum(
                bool(record["prior_pilot_geometry_overlap"]) for record in records
            ),
            "untouched_confirmation_record_count": len(untouched_confirmation),
        },
        "selection": {
            "development_fingerprint": development_manifest["selection_fingerprint"],
            "confirmation_fingerprint": confirmation_manifest["selection_fingerprint"],
            "selection_used_experimental_values": False,
            "selection_used_model_outputs": False,
            "selection_excluded_eligible_records": False,
        },
        "aggregate_metrics": {
            "all_records": _metric_block(records),
            "by_partition": _grouped_metrics(
                records, lambda record: record["partition"]
            ),
            "by_solvent": _grouped_metrics(
                records, lambda record: record["canonical_solvent"]
            ),
            "by_atom_count_bin": _grouped_metrics(records, _atom_bin),
            "untouched_confirmation": _metric_block(untouched_confirmation),
            "unique_geometry_cluster_balanced": {
                method: _cluster_balanced_metrics(records, method) for method in METHODS
            },
        },
        "checkpoint": {
            "filename": args.checkpoint.name,
            "bytes": args.checkpoint.stat().st_size,
            "sha256": checkpoint_sha256,
            "redistributed": False,
        },
        "shards": {
            "count": len(provenance),
            "provenance_sha256": provenance_digest,
            "all_same_execution_commit": True,
        },
        "mace_polar_pairing": {
            "status": "not-yet-complete-on-the-same-653-record-matrix",
            "paired_comparison_emitted": False,
        },
        "capabilities": {"E": False, "F": False, "H": False, "V": False, "M": False},
        "runtime": {
            "aggregation_python": platform.python_version(),
            "aggregation_numpy": np.__version__,
        },
        "source_files_sha256": _source_hashes(),
    }
    _assert_public_safe(public)
    public["measurement_sha256"] = hashlib.sha256(
        canonical_json_bytes(public)
    ).hexdigest()
    write_json_atomic(args.public_output, public)
    print(args.private_output)
    print(args.public_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
