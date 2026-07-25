#!/usr/bin/env python3
"""Run the label-free APBS molecular-surface 129^3-to-161^3 grid follow-up."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_core import (  # noqa: E402
    canonical_json_bytes,
    load_json,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402
from run_apbs_ace_screen import (  # noqa: E402
    apply_execution_controls,
    evaluate_apbs_provider,
    load_evaluation_protocol,
    resolve_apbs,
)


def load_followup_protocol(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    path = Path(path).resolve()
    protocol = load_json(path)
    if protocol.get("schema_version") != 1:
        raise ValueError("Only APBS grid-follow-up schema version 1 is supported.")
    if protocol.get("protocol_id") != "maple-route1-apbs-mol-grid-followup-v1":
        raise ValueError("Unexpected APBS grid-follow-up protocol ID.")
    boundary = protocol["execution_boundary"]
    required = {
        "reads_target_properties": False,
        "changes_parent_accuracy_decision": False,
        "no_fit_or_residual_model": True,
        "no_gas_phase_mm_energy": True,
        "force_claim": False,
    }
    if boundary != required:
        raise ValueError("Unexpected APBS grid-follow-up execution boundary.")
    parent_path = path.parent / protocol["parent_evidence"]["protocol"]
    if sha256_file(parent_path) != protocol["parent_evidence"]["protocol_sha256"]:
        raise ValueError("Parent APBS/ACE protocol hash mismatch.")
    parent, parent_fingerprint = load_evaluation_protocol(parent_path)
    if parent_fingerprint != protocol["parent_evidence"]["protocol_fingerprint"]:
        raise ValueError("Parent APBS/ACE protocol fingerprint mismatch.")
    fingerprint = sha256_bytes(canonical_json_bytes(protocol))
    return protocol, parent, fingerprint


def load_source_manifest(
    protocol_path: str | Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    evidence = protocol["parent_evidence"]
    path = protocol_path.parent / evidence["source_manifest"]
    if sha256_file(path) != evidence["source_manifest_sha256"]:
        raise ValueError("APBS grid-follow-up source-manifest hash mismatch.")
    manifest = load_json(path)
    if int(manifest.get("case_count", -1)) != int(evidence["expected_case_count"]):
        raise ValueError("APBS grid-follow-up case count mismatch.")
    payload = json.dumps(manifest, sort_keys=True).lower()
    if "experimental" in payload or "target_propert" in payload:
        raise ValueError("Grid-follow-up source manifest is not label-free.")
    ids = [row["compound_id"] for row in manifest["records"]]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("Grid-follow-up compound IDs must be sorted and unique.")
    return manifest


def _record_path(work_dir: Path, compound_id: str) -> Path:
    return work_dir / "records" / f"{compound_id}.json"


def _evaluate_case(
    row: dict[str, Any],
    *,
    protocol: dict[str, Any],
    parent: dict[str, Any],
    fingerprint: str,
    source_work_dir: Path,
    repository_root: Path,
    work_dir: Path,
    apbs: dict[str, str],
) -> dict[str, str]:
    compound_id = row["compound_id"]
    destination = _record_path(work_dir, compound_id)
    if destination.is_file():
        existing = load_json(destination)
        if (
            existing.get("status") == "success"
            and existing.get("protocol_fingerprint") == fingerprint
            and existing.get("source_mol2_sha256") == row["source_mol2_sha256"]
            and existing.get("parent_grid_record_sha256")
            == row["parent_grid_record_sha256"]
        ):
            return {"compound_id": compound_id, "status": "skipped"}
        raise ValueError(f"Incompatible existing follow-up record: {destination}")

    parent_record = repository_root / row["parent_grid_record_relative_path"]
    if sha256_file(parent_record) != row["parent_grid_record_sha256"]:
        raise ValueError(f"Parent grid-record hash mismatch for {compound_id}.")
    parent_payload = load_json(parent_record)
    parent_energy = float(parent_payload["reference_grid_polar_kcal_mol"])
    if abs(parent_energy - float(row["parent_129_grid_polar_kcal_mol"])) > 1.0e-12:
        raise ValueError(f"Parent grid energy mismatch for {compound_id}.")

    source_mol2 = source_work_dir / row["source_mol2_relative_path"]
    if sha256_file(source_mol2) != row["source_mol2_sha256"]:
        raise ValueError(f"Source MOL2 hash mismatch for {compound_id}.")
    charges = [float(value) for value in row["am1bcc_charges_e"]]
    atoms = MOL2Reader(str(source_mol2), charge=0, mult=1)
    if len(atoms) != len(charges):
        raise ValueError(f"MOL2/charge length mismatch for {compound_id}.")
    evaluated = evaluate_apbs_provider(
        atoms,
        charges,
        protocol=parent,
        executable=apbs["path"],
        grid=protocol["apbs"]["followup_grid"],
    )
    followup_energy = float(evaluated["polar_kcal_mol"])
    if not math.isfinite(followup_energy):
        raise ValueError(f"Non-finite follow-up energy for {compound_id}.")
    record = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "compound_id": compound_id,
        "status": "success",
        "source_mol2_sha256": row["source_mol2_sha256"],
        "am1bcc_charge_vector_sha256": sha256_bytes(
            canonical_json_bytes(charges)
        ),
        "parent_grid_record_sha256": row["parent_grid_record_sha256"],
        "parent_129_grid_polar_kcal_mol": parent_energy,
        "followup_161_grid_polar_kcal_mol": followup_energy,
        "signed_difference_161_minus_129_kcal_mol": (
            followup_energy - parent_energy
        ),
        "followup_input_sha256": evaluated["input_sha256"],
        "provider_sha256": {"apbs": apbs["sha256"]},
    }
    write_json_atomic(destination, record)
    return {"compound_id": compound_id, "status": "success"}


def run(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, parent, fingerprint = load_followup_protocol(protocol_path)
    apply_execution_controls(protocol, args.workers)
    manifest = load_source_manifest(protocol_path, protocol)
    apbs = resolve_apbs(parent, args.apbs)
    source_work_dir = Path(args.source_work_dir).resolve()
    repository_root = Path(args.repository_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    successes = skips = failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _evaluate_case,
                row,
                protocol=protocol,
                parent=parent,
                fingerprint=fingerprint,
                source_work_dir=source_work_dir,
                repository_root=repository_root,
                work_dir=work_dir,
                apbs=apbs,
            ): row["compound_id"]
            for row in manifest["records"]
        }
        for future in as_completed(futures):
            compound_id = futures[future]
            try:
                status = future.result()["status"]
            except Exception as exc:
                failures += 1
                print(f"FAIL {compound_id}: {exc}", file=sys.stderr, flush=True)
            else:
                successes += status == "success"
                skips += status == "skipped"
                print(f"{status.upper()} {compound_id}", flush=True)
    if failures:
        raise RuntimeError(
            f"Grid follow-up failed for {failures} cases "
            f"({successes} completed, {skips} skipped)."
        )
    print(
        f"Grid follow-up complete: {successes} completed, {skips} skipped, "
        "0 failed."
    )


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = Path(args.protocol).resolve()
    protocol, _parent, fingerprint = load_followup_protocol(protocol_path)
    manifest = load_source_manifest(protocol_path, protocol)
    work_dir = Path(args.work_dir).resolve()
    differences: list[float] = []
    hashes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for source in manifest["records"]:
        compound_id = source["compound_id"]
        path = _record_path(work_dir, compound_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        record = load_json(path)
        if (
            record.get("status") != "success"
            or record.get("protocol_fingerprint") != fingerprint
        ):
            raise ValueError(f"Invalid grid-follow-up record: {path}")
        if record["parent_grid_record_sha256"] != source[
            "parent_grid_record_sha256"
        ]:
            raise ValueError(f"Parent record mismatch for {compound_id}.")
        if "experimental" in json.dumps(record, sort_keys=True).lower():
            raise ValueError(f"Grid-follow-up record leaked a label: {compound_id}")
        difference = float(record["signed_difference_161_minus_129_kcal_mol"])
        differences.append(difference)
        hashes[compound_id] = sha256_file(path)
        rows.append(
            {
                "compound_id": compound_id,
                "signed_difference_161_minus_129_kcal_mol": difference,
            }
        )
    absolute = np.abs(np.asarray(differences, dtype=np.float64))
    gate = protocol["numerical_gate"]
    maximum = float(np.max(absolute))
    p90 = float(np.quantile(absolute, 0.9))
    checks = {
        "maximum": maximum <= float(gate["maximum_absolute_difference_kcal_mol"]),
        "p90": p90 <= float(gate["p90_absolute_difference_kcal_mol"]),
    }
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "claim_scope": protocol["claim_scope"],
        "case_count": len(rows),
        "label_use_boundary": {
            "target_properties_read": False,
            "fit_or_residual_model": False,
            "gas_phase_mm_energy": False,
        },
        "mean_signed_difference_kcal_mol": float(np.mean(differences)),
        "mean_absolute_difference_kcal_mol": float(np.mean(absolute)),
        "p90_absolute_difference_kcal_mol": p90,
        "maximum_absolute_difference_kcal_mol": maximum,
        "gate": {
            "checks": checks,
            "passed": all(checks.values()),
            "interpretation": gate["interpretation"],
        },
        "parent_accuracy_decision_unchanged": True,
        "largest_absolute_differences": sorted(
            rows,
            key=lambda row: abs(
                row["signed_difference_161_minus_129_kcal_mol"]
            ),
            reverse=True,
        )[:10],
        "record_sha256": hashes,
    }
    write_json_atomic(args.summary, summary)
    print(f"Wrote grid-follow-up summary to {Path(args.summary).resolve()}")
    return summary


def _defaults() -> dict[str, Path]:
    return {
        "protocol": SCRIPT_DIR / "apbs_mol_grid_followup_protocol.json",
        "source_work_dir": (
            REPOSITORY_ROOT / ".omx/benchmarks/neutral-water-freesolv-route1-20260723"
        ),
        "repository_root": REPOSITORY_ROOT,
        "work_dir": (
            REPOSITORY_ROOT / ".omx/benchmarks/apbs-mol-grid-followup-route1-20260725"
        ),
        "summary": SCRIPT_DIR / "apbs-mol-grid-followup-2026-07-25.json",
    }


def main() -> None:
    defaults = _defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "summarize", "all"))
    parser.add_argument("--protocol", default=str(defaults["protocol"]))
    parser.add_argument("--source-work-dir", default=str(defaults["source_work_dir"]))
    parser.add_argument("--repository-root", default=str(defaults["repository_root"]))
    parser.add_argument("--work-dir", default=str(defaults["work_dir"]))
    parser.add_argument("--summary", default=str(defaults["summary"]))
    parser.add_argument("--apbs")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.phase in {"run", "all"}:
        run(args)
    if args.phase in {"summarize", "all"}:
        summarize(args)


if __name__ == "__main__":
    main()
