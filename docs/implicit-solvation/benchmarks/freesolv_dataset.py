#!/usr/bin/env python3
"""Hash-pinned FreeSolv dataset preparation shared only by Route 2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

from benchmark_core import (
    canonical_json_bytes,
    classify_candidate,
    fetch_and_verify_artifacts,
    load_json,
    load_protocol,
    partition_for_smiles,
    safe_extract_tar,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _parse_database_text(path: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = [field.strip() for field in raw.split(";")]
        if len(fields) < 10:
            raise ValueError(f"Malformed FreeSolv database row: {raw!r}")
        compound_id = fields[0]
        if compound_id in records:
            raise ValueError(f"Duplicate FreeSolv compound ID: {compound_id}")
        records[compound_id] = {
            "compound_id": compound_id,
            "smiles": fields[1],
            "name": fields[2],
            "experimental_kcal_mol": fields[3],
            "experimental_uncertainty_kcal_mol": fields[4],
            "experimental_reference": fields[7],
            "notes": fields[9],
        }
    return records

def prepare(args: argparse.Namespace) -> None:
    protocol, fingerprint = load_protocol(args.protocol)
    work_dir = Path(args.work_dir).resolve()
    dataset_dir = work_dir / "dataset"
    source_dir = Path(args.source_dir).resolve() if args.source_dir else None
    hashes = fetch_and_verify_artifacts(
        protocol, dataset_dir, source_dir=source_dir
    )
    safe_extract_tar(dataset_dir / "mol2files_gaff.tar.gz", dataset_dir)

    text_records = _parse_database_text(dataset_dir / "database.txt")
    json_records = load_json(dataset_dir / "database.json")
    expected_count = int(protocol["dataset"]["expected_record_count"])
    if set(text_records) != set(json_records) or len(text_records) != expected_count:
        raise ValueError(
            "Pinned FreeSolv database.txt/database.json identities do not reconcile with "
            f"expected_record_count={expected_count}."
        )

    allowed = set(protocol["domain"]["allowed_elements"])
    charge_screen = float(protocol["domain"]["input_charge_sum_screen_e"])
    pilots = set(protocol["partition"]["pilot_development_only"])
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for compound_id in sorted(text_records):
        row = text_records[compound_id]
        full = json_records[compound_id]
        json_name = str(full.get("iupac", "")).strip()
        if json_name != row["name"].strip():
            raise ValueError(
                "Name mismatch between pinned FreeSolv files for "
                f"{compound_id}: database.txt={row['name']!r}, "
                f"database.json={json_name!r}."
            )
        json_experimental = float(full["expt"])
        text_experimental = float(row["experimental_kcal_mol"])
        if not math.isclose(
            json_experimental,
            text_experimental,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Experimental free-energy mismatch between pinned FreeSolv "
                f"files for {compound_id}: database.txt={text_experimental}, "
                f"database.json={json_experimental}."
            )
        json_uncertainty = float(full["d_expt"])
        text_uncertainty = float(
            row["experimental_uncertainty_kcal_mol"]
        )
        if not math.isclose(
            json_uncertainty,
            text_uncertainty,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Experimental uncertainty mismatch between pinned FreeSolv "
                f"files for {compound_id}: database.txt={text_uncertainty}, "
                f"database.json={json_uncertainty}."
            )
        mol2_path = dataset_dir / "mol2files_gaff" / f"{compound_id}.mol2"
        try:
            if not mol2_path.is_file():
                raise FileNotFoundError(mol2_path)
            atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
            elements = set(atoms.get_chemical_symbols())
            unsupported = sorted(elements - allowed)
            if unsupported:
                raise ValueError("unsupported elements: " + ", ".join(unsupported))
            input_charge_sum = float(atoms.get_initial_charges().sum())
            if abs(input_charge_sum) > charge_screen:
                raise ValueError(
                    f"input charge sum {input_charge_sum:.8f} exceeds neutral screen"
                )
        except FileNotFoundError as exc:
            exclusions.append(
                {"compound_id": compound_id, "reason_code": "missing_mol2", "detail": str(exc)}
            )
            continue
        except Exception as exc:
            exclusions.append(
                {"compound_id": compound_id, "reason_code": "invalid_mol2", "detail": str(exc)}
            )
            continue

        smiles = str(full.get("smiles", row["smiles"])).strip()
        if smiles != row["smiles"].strip():
            raise ValueError(f"SMILES mismatch between pinned FreeSolv files for {compound_id}.")
        partition = partition_for_smiles(
            smiles,
            seed=str(protocol["partition"]["seed"]),
            development_fraction=float(protocol["partition"]["development_fraction"]),
            forced_development=compound_id in pilots,
        )
        candidate = {
            "compound_id": compound_id,
            "name": row["name"],
            "smiles": smiles,
            "structure_group_sha256": sha256_bytes(smiles.encode("utf-8")),
            "partition": partition,
            "pilot_development_only": compound_id in pilots,
            "experimental_kcal_mol": float(row["experimental_kcal_mol"]),
            "experimental_uncertainty_kcal_mol": float(
                row["experimental_uncertainty_kcal_mol"]
            ),
            "experimental_reference": row["experimental_reference"],
            "mol2_relative_path": f"dataset/mol2files_gaff/{compound_id}.mol2",
            "mol2_sha256": sha256_file(mol2_path),
            "input_charge_sum_e": input_charge_sum,
            "dataset_record_sha256": sha256_bytes(canonical_json_bytes(full)),
            **classify_candidate(atoms, list(full.get("groups", []))),
        }
        candidates.append(candidate)

    if len(candidates) + len(exclusions) != expected_count:
        raise ValueError("Prepared candidate and exclusion counts do not match the dataset.")
    candidate_ids = {candidate["compound_id"] for candidate in candidates}
    missing_pilots = sorted(pilots - candidate_ids)
    if missing_pilots:
        raise ValueError("Pilot development records were excluded or missing: " + ", ".join(missing_pilots))
    leaked = [
        candidate["compound_id"]
        for candidate in candidates
        if candidate["pilot_development_only"] and candidate["partition"] != "development"
    ]
    if leaked:
        raise ValueError("Pilot records leaked into confirmation: " + ", ".join(leaked))

    manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "dataset_commit": protocol["dataset"]["commit"],
        "dataset_artifact_sha256": hashes,
        "candidate_count": len(candidates),
        "exclusion_count": len(exclusions),
        "partition_counts": {
            name: sum(candidate["partition"] == name for candidate in candidates)
            for name in ("development", "confirmation")
        },
        "candidates": candidates,
        "exclusions": exclusions,
    }
    write_json_atomic(work_dir / "prepared.json", manifest)
    print(
        f"Prepared {len(candidates)} candidates "
        f"({manifest['partition_counts']['development']} development, "
        f"{manifest['partition_counts']['confirmation']} confirmation); "
        f"excluded {len(exclusions)}."
    )

def _load_prepared(work_dir: Path, fingerprint: str) -> dict[str, Any]:
    path = work_dir / "prepared.json"
    if not path.is_file():
        raise ValueError("Benchmark is not prepared; run the prepare phase first.")
    manifest = load_json(path)
    if manifest.get("protocol_fingerprint") != fingerprint:
        raise ValueError("Prepared benchmark protocol fingerprint does not match this protocol.")
    for name, expected in manifest["dataset_artifact_sha256"].items():
        observed = sha256_file(work_dir / "dataset" / name)
        if observed != expected:
            raise ValueError(f"Prepared dataset artifact changed after preparation: {name}.")
    return manifest

def freeze_confirmation(args: argparse.Namespace) -> None:
    protocol, fingerprint = load_protocol(args.protocol)
    work_dir = Path(args.work_dir).resolve()
    _load_prepared(work_dir, fingerprint)
    path = work_dir / protocol["confirmation"]["lock_filename"]
    record_dir = work_dir / "records" / "confirmation"
    if path.exists():
        raise FileExistsError("Confirmation lock already exists and is immutable.")
    if record_dir.exists() and any(record_dir.glob("*.json")):
        raise ValueError("Confirmation records already exist; refusing a post-hoc lock.")
    lock = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "proposed_default": args.proposed_default,
        "pass_rule": args.pass_rule,
        "frozen_at_utc": utc_now(),
        "one_shot": True,
        "failed_confirmation_must_not_trigger_tuning": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(canonical_json_bytes(lock).decode("utf-8"))
    except FileExistsError:
        raise FileExistsError("Confirmation lock already exists and is immutable.") from None
    print(f"Frozen confirmation rule in {path} ({sha256_file(path)}).")
