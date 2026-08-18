#!/usr/bin/env python3
"""Run the complete frozen-charge MNSol panel through smooth harmonic ddPCM.

The restricted row-level MNSol data and frozen AIMNet2 charges remain below
``.omx``.  Only aggregate metrics are written to the requested public output.
The runner is resumable at one private shard per record and requires a clean
tracked checkout so every result is bound to one implementation commit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, BENCHMARK_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from benchmark_core import canonical_json_bytes, sha256_file, write_json_atomic
from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    HARTREE_TO_KCAL_MOL,
    route2_coulomb_radii,
)
from maple.function.route2_smd_profiles import DDPCM_MULTISOLVENT_SMD_PROFILE
from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.continuum.aimnet2_smooth_partition_ddpcm import (
    AIMNET2_SMOOTH_PARTITION_PARTITION_LMAX,
    build_aimnet2_frozen_charge_smooth_partition_ddpcm_candidate,
)
from maple.solvation.continuum.harmonic_exposure import (
    HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE,
)

ARTIFACT = "route2-mnsol-aimnet2-smooth-partition-ddpcm-full-v1"
PRIOR_ARTIFACT = "route2-mnsol-aimnet2-full-frozen-charge-matrix-v1"
EXPECTED_RECORD_COUNT = 653
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "records",
        "entry_number",
        "geometry_handle",
        "geometry_sha256",
        "solute_name",
        "formula",
        "coordinates_angstrom",
        "charges_e",
        "experimental_delta_g_kcal_mol",
        "opaque_record_id",
    }
)


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _clean_head() -> str:
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("Full MNSol execution requires a clean source checkout.")
    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(head) != 40:
        raise RuntimeError("Could not resolve the execution Git commit.")
    return head


def _private_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    private_root = (REPO_ROOT / ".omx").resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError as exc:
        raise ValueError("Private MNSol output must remain below .omx.") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _task_fingerprint(task: Mapping[str, Any]) -> str:
    return _sha(
        {
            key: task[key]
            for key in (
                "selection_index",
                "partition_selection_index",
                "opaque_record_id",
                "geometry_sha256",
                "canonical_solvent",
                "atomic_numbers",
                "coordinates_angstrom",
                "charges_e",
                "cavity_radii_angstrom",
                "smd_cds_energy_kcal_mol",
                "experimental_delta_g_kcal_mol",
                "reference_ddpcm_energy_hartree",
            )
        }
    )


def _factor_degree_preflight(
    coordinates: Sequence[Sequence[float]],
    radii: Sequence[float],
    *,
    transition_width: float = 0.18,
) -> tuple[int, int]:
    """Return the largest exact product count and required algebraic degree."""

    positions = np.asarray(coordinates, dtype=float)
    radius_values = np.asarray(radii, dtype=float)
    maximum = 0
    for atom_i, radius_i in enumerate(radius_values):
        inside_count = 0
        centered_count = 0
        centered_buried = False
        for atom_j, radius_j in enumerate(radius_values):
            if atom_i == atom_j:
                continue
            distance = float(np.linalg.norm(positions[atom_j] - positions[atom_i]))
            minimum = (distance - radius_i) ** 2 - radius_j**2
            maximum_value = (distance + radius_i) ** 2 - radius_j**2
            if minimum < 0.0:
                inside_count += 1
            if maximum_value <= -transition_width:
                centered_buried = True
            elif minimum < transition_width:
                centered_count += 1
        maximum = max(
            maximum,
            inside_count,
            0 if centered_buried else centered_count,
        )
    return maximum, (maximum + 1) * AIMNET2_SMOOTH_PARTITION_PARTITION_LMAX


def _prepare_tasks(
    dataset: Any,
    prior: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if (
        prior.get("artifact") != PRIOR_ARTIFACT
        or prior.get("complete_panel") is not True
        or prior.get("do_not_commit") is not True
    ):
        raise ValueError("Prior frozen-charge MNSol aggregate contract drifted.")
    records = prior.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_RECORD_COUNT:
        raise ValueError("Prior aggregate must contain exactly 653 private records.")
    partition_counts = {"development": 505, "confirmation": 148}
    ordered = []
    for partition, count in partition_counts.items():
        values = sorted(
            (item for item in records if item.get("partition") == partition),
            key=lambda item: int(item["selection_index"]),
        )
        if len(values) != count or [
            int(item["selection_index"]) for item in values
        ] != list(range(count)):
            raise ValueError(f"Prior {partition} selection indices are incomplete.")
        ordered.extend(values)

    tasks = []
    maximum_factor_count = 0
    maximum_required_degree = 0
    for panel_index, record in enumerate(ordered):
        handle = str(record["geometry_handle"])
        geometry = dataset.geometries.get(handle)
        if geometry is None or geometry.sha256 != record.get("geometry_sha256"):
            raise ValueError(
                f"Prior record geometry drifted at index {record['selection_index']}."
            )
        symbols = tuple(
            __import__("ase").data.chemical_symbols[number]
            for number in geometry.atomic_numbers
        )
        expected_radii = tuple(
            float(value)
            for value in route2_coulomb_radii(
                symbols,
                solvent=str(record["canonical_solvent"]),
                profile=DDPCM_MULTISOLVENT_SMD_PROFILE,
            )
        )
        recorded_radii = tuple(
            float(value) for value in record["cavity_radii_angstrom"]
        )
        if expected_radii != recorded_radii:
            raise ValueError(
                f"Cavity radii drifted at index {record['selection_index']}."
            )
        charges = tuple(float(value) for value in record["charges_e"])
        if len(charges) != len(geometry.atomic_numbers) or not np.all(
            np.isfinite(charges)
        ):
            raise ValueError(
                f"Frozen charges are invalid at index {record['selection_index']}."
            )
        factor_count, required_degree = _factor_degree_preflight(
            geometry.coordinates_angstrom,
            recorded_radii,
        )
        maximum_factor_count = max(maximum_factor_count, factor_count)
        maximum_required_degree = max(maximum_required_degree, required_degree)
        if required_degree > HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE:
            raise ValueError(
                f"Record {record['selection_index']} requires algebraic degree "
                f"{required_degree}, above the bounded contract."
            )
        methods = record.get("methods")
        if not isinstance(methods, Mapping) or "ddpcm" not in methods:
            raise ValueError(
                f"Prior ddPCM reference is missing at index {record['selection_index']}."
            )
        task = {
            "selection_index": panel_index,
            "partition_selection_index": int(record["selection_index"]),
            "opaque_record_id": str(record["opaque_record_id"]),
            "partition": str(record["partition"]),
            "geometry_handle": handle,
            "geometry_sha256": geometry.sha256,
            "canonical_solvent": str(record["canonical_solvent"]),
            "atom_count": len(geometry.atomic_numbers),
            "atomic_numbers": tuple(int(value) for value in geometry.atomic_numbers),
            "coordinates_angstrom": geometry.coordinates_angstrom,
            "charges_e": charges,
            "cavity_radii_angstrom": recorded_radii,
            "smd_cds_energy_kcal_mol": _finite(
                record["smd_cds_energy_kcal_mol"], label="SMD-CDS energy"
            ),
            "experimental_delta_g_kcal_mol": _finite(
                record["experimental_delta_g_kcal_mol"], label="experimental energy"
            ),
            "reference_ddpcm_energy_hartree": _finite(
                methods["ddpcm"]["polarization_energy_hartree"],
                label="reference ddPCM energy",
            ),
            "factor_count": factor_count,
            "required_algebraic_degree": required_degree,
        }
        task["task_sha256"] = _task_fingerprint(task)
        tasks.append(task)
    return tasks, {
        "maximum_factor_count": maximum_factor_count,
        "maximum_required_algebraic_degree": maximum_required_degree,
    }


def _run_task(task: Mapping[str, Any], execution_git_head: str) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import torch
    from ase import Atoms

    torch.set_num_threads(1)
    atoms = Atoms(
        numbers=task["atomic_numbers"],
        positions=task["coordinates_angstrom"],
    )
    source = np.zeros((len(atoms), 4), dtype=float)
    source[:, 0] = np.asarray(task["charges_e"], dtype=float)
    started = time.perf_counter()
    candidate = build_aimnet2_frozen_charge_smooth_partition_ddpcm_candidate(
        tuple(atoms.get_chemical_symbols()),
        solvent=str(task["canonical_solvent"]),
        dtype=torch.float64,
        device="cpu",
    )
    if tuple(candidate.radii_angstrom) != tuple(task["cavity_radii_angstrom"]):
        raise RuntimeError("Worker cavity radii differ from the preflight contract.")
    continuum_energy_ev = candidate.energy_eV(atoms, source)
    elapsed = time.perf_counter() - started
    continuum_energy_kcal = continuum_energy_ev / HARTREE_TO_EV * HARTREE_TO_KCAL_MOL
    reference_ev = float(task["reference_ddpcm_energy_hartree"]) * HARTREE_TO_EV
    total = continuum_energy_kcal + float(task["smd_cds_energy_kcal_mol"])
    signed_error = total - float(task["experimental_delta_g_kcal_mol"])
    return {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "do_not_commit": True,
        "execution_git_head": execution_git_head,
        "selection_index": int(task["selection_index"]),
        "partition_selection_index": int(task["partition_selection_index"]),
        "task_sha256": str(task["task_sha256"]),
        "opaque_record_id": str(task["opaque_record_id"]),
        "partition": str(task["partition"]),
        "geometry_handle": str(task["geometry_handle"]),
        "geometry_sha256": str(task["geometry_sha256"]),
        "canonical_solvent": str(task["canonical_solvent"]),
        "atom_count": int(task["atom_count"]),
        "factor_count": int(task["factor_count"]),
        "required_algebraic_degree": int(task["required_algebraic_degree"]),
        "continuum_energy_ev": continuum_energy_ev,
        "continuum_energy_kcal_mol": continuum_energy_kcal,
        "reference_ddpcm_energy_ev": reference_ev,
        "smooth_minus_reference_ev": continuum_energy_ev - reference_ev,
        "smooth_minus_reference_kcal_mol": (
            (continuum_energy_ev - reference_ev) / HARTREE_TO_EV * HARTREE_TO_KCAL_MOL
        ),
        "smd_cds_energy_kcal_mol": float(task["smd_cds_energy_kcal_mol"]),
        "experimental_delta_g_kcal_mol": float(task["experimental_delta_g_kcal_mol"]),
        "predicted_delta_g_kcal_mol": total,
        "signed_error_kcal_mol": signed_error,
        "absolute_error_kcal_mol": abs(signed_error),
        "configuration_sha256": candidate.configuration_sha256(),
        "provenance_sha256": candidate.provenance_sha256,
        "timing_seconds": elapsed,
    }


def _valid_shard(
    value: object,
    *,
    task: Mapping[str, Any],
    execution_git_head: str,
) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("artifact") == ARTIFACT
        and value.get("schema_version") == 1
        and value.get("do_not_commit") is True
        and value.get("execution_git_head") == execution_git_head
        and value.get("selection_index") == task["selection_index"]
        and value.get("task_sha256") == task["task_sha256"]
        and isinstance(value.get("continuum_energy_ev"), (int, float))
        and math.isfinite(float(value["continuum_energy_ev"]))
    )


def _metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    errors = np.asarray([record["signed_error_kcal_mol"] for record in records])
    absolute = np.abs(errors)
    parity = np.asarray(
        [record["smooth_minus_reference_kcal_mol"] for record in records]
    )
    reference_errors = errors - parity
    return {
        "record_count": len(records),
        "mean_signed_error_kcal_mol": float(np.mean(errors)),
        "mean_absolute_error_kcal_mol": float(np.mean(absolute)),
        "root_mean_square_error_kcal_mol": float(np.sqrt(np.mean(errors**2))),
        "median_absolute_error_kcal_mol": float(np.median(absolute)),
        "p90_absolute_error_kcal_mol": float(np.quantile(absolute, 0.90)),
        "p95_absolute_error_kcal_mol": float(np.quantile(absolute, 0.95)),
        "p99_absolute_error_kcal_mol": float(np.quantile(absolute, 0.99)),
        "maximum_absolute_error_kcal_mol": float(np.max(absolute)),
        "reference_highres_mean_signed_error_kcal_mol": float(
            np.mean(reference_errors)
        ),
        "reference_highres_mean_absolute_error_kcal_mol": float(
            np.mean(np.abs(reference_errors))
        ),
        "reference_highres_root_mean_square_error_kcal_mol": float(
            np.sqrt(np.mean(reference_errors**2))
        ),
        "smooth_vs_reference_mean_signed_kcal_mol": float(np.mean(parity)),
        "smooth_vs_reference_mean_absolute_kcal_mol": float(np.mean(np.abs(parity))),
        "smooth_vs_reference_maximum_absolute_kcal_mol": float(np.max(np.abs(parity))),
    }


def _grouped_metrics(
    records: Sequence[Mapping[str, Any]],
    key: Callable[[Mapping[str, Any]], str],
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    return {name: _metrics(groups[name]) for name in sorted(groups)}


def _atom_bin(record: Mapping[str, Any]) -> str:
    count = int(record["atom_count"])
    if count <= 5:
        return "01-05"
    if count <= 10:
        return "06-10"
    if count <= 20:
        return "11-20"
    return "21-plus"


def _assert_public_safe(value: object) -> None:
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_PUBLIC_KEYS.intersection(value)
        if forbidden:
            raise ValueError(
                f"Public aggregate leaks private keys: {sorted(forbidden)}"
            )
        for child in value.values():
            _assert_public_safe(child)
    elif isinstance(value, list):
        for child in value:
            _assert_public_safe(child)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--prior-private", type=Path, required=True)
    parser.add_argument("--prior-public", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.workers <= 0 or args.workers > 16:
        raise ValueError("--workers must lie in [1, 16].")
    execution_git_head = _clean_head()
    output_dir = _private_directory(args.private_output_dir)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_mnsol_protocol(args.protocol)
    dataset = load_mnsol_v2012(args.source, protocol)
    prior = _object(args.prior_private, label="prior private aggregate")
    prior_public = _object(args.prior_public, label="prior public aggregate")
    checkpoint = prior_public.get("checkpoint")
    if (
        prior_public.get("artifact") != PRIOR_ARTIFACT
        or prior_public.get("complete_panel") is not True
        or not isinstance(checkpoint, Mapping)
        or len(str(checkpoint.get("sha256", ""))) != 64
        or int(checkpoint.get("bytes", 0)) <= 0
    ):
        raise ValueError("Prior public aggregate/checkpoint provenance is invalid.")
    tasks, degree_preflight = _prepare_tasks(dataset, prior)
    prior_sha256 = sha256_file(args.prior_private)
    prior_public_sha256 = sha256_file(args.prior_public)

    records: dict[int, dict[str, Any]] = {}
    pending = []
    for task in tasks:
        shard_path = shard_dir / f"{task['selection_index']:04d}.json"
        if shard_path.is_file():
            value = _object(shard_path, label="private shard")
            if _valid_shard(value, task=task, execution_git_head=execution_git_head):
                records[int(task["selection_index"])] = value
                continue
        pending.append(task)

    if pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = {
                pool.submit(_run_task, task, execution_git_head): task
                for task in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                value = future.result()
                index = int(task["selection_index"])
                write_json_atomic(shard_dir / f"{index:04d}.json", value)
                records[index] = value
                print(
                    json.dumps(
                        {
                            "completed_new": completed,
                            "pending_new": len(pending),
                            "selection_index": index,
                            "timing_seconds": value["timing_seconds"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    if sorted(records) != list(range(EXPECTED_RECORD_COUNT)):
        raise RuntimeError("Full smooth ddPCM panel is incomplete.")
    ordered = [records[index] for index in range(EXPECTED_RECORD_COUNT)]
    elapsed_values = np.asarray([record["timing_seconds"] for record in ordered])
    private = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "complete_panel": True,
        "do_not_commit": True,
        "execution_git_head": execution_git_head,
        "prior_private_sha256": prior_sha256,
        "prior_public_sha256": prior_public_sha256,
        "records": ordered,
    }
    write_json_atomic(output_dir / "private.json", private)

    public = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "complete_panel": True,
        "execution_git_head": execution_git_head,
        "record_count": len(ordered),
        "unique_geometry_count": len({record["geometry_handle"] for record in ordered}),
        "dataset": {
            "table_sha256": dataset.table_sha256,
            "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
            "temperature_k": protocol.temperature_k,
            "standard_state": protocol.standard_state,
        },
        "source": {
            "model": "AIMNet2 one-shot frozen NQE point monopoles",
            "prior_artifact": PRIOR_ARTIFACT,
            "prior_private_sha256": prior_sha256,
            "prior_public_sha256": prior_public_sha256,
            "checkpoint": checkpoint,
            "continuum_field_supplied_to_aimnet2": False,
            "electronic_scf_iteration": False,
        },
        "method": {
            "continuum": "smooth-partition-harmonic-ddpcm",
            "surface_lmax": 4,
            "partition_lmax": AIMNET2_SMOOTH_PARTITION_PARTITION_LMAX,
            "maximum_algebraic_degree": HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE,
            "transition_width_angstrom2": 0.18,
            "nonpolar": "PySCF-2.13.1 SMD-CDS reused from frozen prior artifact",
            "fit_or_calibration": False,
        },
        "factor_degree_preflight": {
            **degree_preflight,
            "algebraic_degree_bound": HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE,
            "failed_record_count": 0,
        },
        "aggregate_metrics": _metrics(ordered),
        "partition_metrics": _grouped_metrics(
            ordered, lambda record: str(record["partition"])
        ),
        "solvent_metrics": _grouped_metrics(
            ordered, lambda record: str(record["canonical_solvent"])
        ),
        "atom_count_bin_metrics": _grouped_metrics(ordered, _atom_bin),
        "timing_seconds": {
            "sum": float(np.sum(elapsed_values)),
            "mean": float(np.mean(elapsed_values)),
            "median": float(np.median(elapsed_values)),
            "maximum": float(np.max(elapsed_values)),
            "worker_count": args.workers,
        },
        "claim_boundary": (
            "Complete 653-record scalar accuracy and numerical-continuum parity for "
            "the frozen one-shot AIMNet2 source. This artifact does not by itself "
            "admit forces, Hessians, optimization, frequency, transition-state, IRC, "
            "molecular-dynamics, or self-consistent electronic polarization tasks."
        ),
    }
    _assert_public_safe(public)
    write_json_atomic(args.public_output, public)
    print(json.dumps(public["aggregate_metrics"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
