#!/usr/bin/env python3
"""Prepare, run, and summarize the pinned MAPLE FreeSolv GB evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core import (  # noqa: E402
    classify_candidate,
    ensure_confirmation_lock,
    expected_attempt_ids,
    fetch_and_verify_artifacts,
    load_json,
    load_protocol,
    partition_for_smiles,
    safe_extract_tar,
    sha256_bytes,
    sha256_file,
    summarize_errors,
    write_json_atomic,
    canonical_json_bytes,
)
from maple.function.calculator.extra_correction.implicit.charges import (  # noqa: E402
    prepare_charges,
)
from maple.function.calculator.extra_correction.implicit.openmm_gb import (  # noqa: E402
    KJ_PER_MOL_PER_HARTREE,
    OpenMMGB,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402


KCAL_PER_HARTREE = KJ_PER_MOL_PER_HARTREE / 4.184


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


def _provider_environment(protocol: dict[str, Any], executable: str) -> dict[str, Any]:
    try:
        openmm_version = importlib.metadata.version("openmm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError("The pinned benchmark requires OpenMM.") from exc
    required_openmm = str(protocol["providers"]["openmm"]["required_version"])
    if openmm_version != required_openmm:
        raise ValueError(
            f"Protocol requires OpenMM {required_openmm}, observed {openmm_version}. "
            "Change the environment or author and review a new protocol."
        )
    resolved = shutil.which(executable)
    amber = {
        "requested_executable": executable,
        "resolved_executable": resolved,
        "executable_sha256": sha256_file(resolved) if resolved else None,
        "required_version": protocol["providers"]["ambertools"]["required_version"],
    }
    return {
        "openmm_version": openmm_version,
        "ambertools": amber,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _record_path(work_dir: Path, partition: str, attempt_id: str) -> Path:
    return work_dir / "records" / partition / f"{attempt_id}.json"


def _charge_failure_audit(
    work_dir: Path, compound_id: str, charge_method: str
) -> dict[str, Any]:
    audit_dir = work_dir / "provider-audit" / compound_id / charge_method
    command_path = audit_dir / f"charges-{charge_method}.command.json"
    command_record: dict[str, Any] = {}
    if command_path.is_file():
        try:
            command_record = load_json(command_path)
        except (OSError, ValueError):
            command_record = {}
    return {
        "provider": "ambertools-antechamber",
        "command": command_record.get("command"),
        "returncode": command_record.get("returncode"),
        "audit_dir": str(audit_dir),
        "command_record": str(command_path) if command_path.is_file() else None,
    }


def _base_record(
    protocol: dict[str, Any],
    fingerprint: str,
    candidate: dict[str, Any],
    charge_method: str,
    gb_model: str,
    environment: dict[str, Any],
) -> dict[str, Any]:
    attempt_id = f"{candidate['compound_id']}__{charge_method}__{gb_model}"
    return {
        "schema_version": protocol["result_schema_version"],
        "attempt_id": attempt_id,
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "partition": candidate["partition"],
        "compound_id": candidate["compound_id"],
        "name": candidate["name"],
        "smiles": candidate["smiles"],
        "structure_group_sha256": candidate["structure_group_sha256"],
        "mol2_sha256": candidate["mol2_sha256"],
        "dataset_record_sha256": candidate["dataset_record_sha256"],
        "experimental_kcal_mol": candidate["experimental_kcal_mol"],
        "experimental_reference": candidate["experimental_reference"],
        "charge_method": charge_method,
        "gb_model": gb_model,
        "nonpolar": protocol["methods"]["nonpolar"],
        "environment": environment,
        "bins": {
            "functional_groups": candidate["functional_groups"],
            "element_class": candidate["element_class"],
            "size": candidate["size_bin"],
            "heteroatom_count": candidate["heteroatom_bin"],
            "flexibility": candidate["flexibility_bin"],
        },
        "literature": {
            "charge": protocol["literature"][charge_method],
            "gb": protocol["literature"]["obc" if gb_model in {"obc1", "obc2"} else gb_model],
        },
        "recorded_at_utc": utc_now(),
    }


def _load_or_prepare_charges(
    atoms,
    candidate: dict[str, Any],
    charge_method: str,
    protocol: dict[str, Any],
    fingerprint: str,
    work_dir: Path,
    executable: str,
) -> tuple[list[float], dict[str, Any]]:
    cache_path = work_dir / "charges" / candidate["compound_id"] / f"{charge_method}.json"
    if cache_path.is_file():
        cache = load_json(cache_path)
        if (
            cache.get("protocol_fingerprint") != fingerprint
            or cache.get("mol2_sha256") != candidate["mol2_sha256"]
            or cache.get("charge_method") != charge_method
        ):
            raise ValueError(f"Charge cache fingerprint mismatch: {cache_path}")
        return list(cache["charges_e"]), dict(cache["provenance"])

    audit_dir = work_dir / "provider-audit" / candidate["compound_id"] / charge_method
    result = prepare_charges(
        atoms,
        {
            "source": "maple",
            "method": charge_method,
            "mode": "fixed",
            "geometry": "keep",
            "executable": executable,
        },
        audit_dir,
    )
    cache = {
        "schema_version": 1,
        "protocol_fingerprint": fingerprint,
        "mol2_sha256": candidate["mol2_sha256"],
        "charge_method": charge_method,
        "charges_e": result.charges.tolist(),
        "sum_e": float(result.charges.sum()),
        "provenance": result.provenance,
    }
    write_json_atomic(cache_path, cache)
    return list(cache["charges_e"]), dict(cache["provenance"])


def run(args: argparse.Namespace) -> None:
    protocol, fingerprint = load_protocol(args.protocol)
    work_dir = Path(args.work_dir).resolve()
    manifest = _load_prepared(work_dir, fingerprint)
    if args.partition == "confirmation":
        ensure_confirmation_lock(work_dir, fingerprint)
    executable = args.antechamber or protocol["providers"]["ambertools"]["executable"]
    environment = _provider_environment(protocol, executable)
    candidates = [
        candidate
        for candidate in manifest["candidates"]
        if candidate["partition"] == args.partition
    ]
    if args.max_compounds is not None:
        if args.max_compounds <= 0:
            raise ValueError("--max-compounds must be positive.")
        candidates = candidates[: args.max_compounds]
        print("WARNING: max-compounds creates an incomplete smoke run; summarize will reject it.")

    charge_methods = protocol["methods"]["charge_methods"]
    gb_models = protocol["methods"]["gb_models"]
    completed = 0
    skipped = 0
    for candidate in candidates:
        mol2_path = work_dir / candidate["mol2_relative_path"]
        if sha256_file(mol2_path) != candidate["mol2_sha256"]:
            raise ValueError(f"MOL2 changed after preparation: {candidate['compound_id']}")
        atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
        for charge_method in charge_methods:
            pending_models = [
                model
                for model in gb_models
                if not _record_path(
                    work_dir,
                    args.partition,
                    f"{candidate['compound_id']}__{charge_method}__{model}",
                ).exists()
            ]
            skipped += len(gb_models) - len(pending_models)
            if not pending_models:
                continue
            try:
                charge_values, charge_provenance = _load_or_prepare_charges(
                    atoms,
                    candidate,
                    charge_method,
                    protocol,
                    fingerprint,
                    work_dir,
                    executable,
                )
            except Exception as exc:
                for model in pending_models:
                    record = _base_record(
                        protocol,
                        fingerprint,
                        candidate,
                        charge_method,
                        model,
                        environment,
                    )
                    record.update(
                        status="failure",
                        failure={
                            "phase": "charge",
                            **_charge_failure_audit(
                                work_dir, candidate["compound_id"], charge_method
                            ),
                            "exception_class": type(exc).__name__,
                            "reason": str(exc),
                        },
                    )
                    write_json_atomic(
                        _record_path(work_dir, args.partition, record["attempt_id"]), record
                    )
                    completed += 1
                continue

            for model in pending_models:
                record = _base_record(
                    protocol,
                    fingerprint,
                    candidate,
                    charge_method,
                    model,
                    environment,
                )
                record["charge_provenance"] = charge_provenance
                record["charges_e"] = charge_values
                try:
                    provider = OpenMMGB(
                        atoms,
                        charge_values,
                        model=model,
                        nonpolar=protocol["methods"]["nonpolar"],
                        platform=protocol["methods"]["openmm_platform"],
                    )
                    result = provider.evaluate(atoms, need_forces=False)
                    polar = result.components_hartree["polar"] * KCAL_PER_HARTREE
                    nonpolar = result.components_hartree["nonpolar"] * KCAL_PER_HARTREE
                    predicted = result.energy_hartree * KCAL_PER_HARTREE
                    error = predicted - candidate["experimental_kcal_mol"]
                    record.update(
                        status="success",
                        provider_provenance=result.provenance,
                        components_kcal_mol={"polar": polar, "nonpolar": nonpolar},
                        predicted_kcal_mol=predicted,
                        signed_error_kcal_mol=error,
                        absolute_error_kcal_mol=abs(error),
                    )
                except Exception as exc:
                    record.update(
                        status="failure",
                        failure={
                            "phase": "gb",
                            "provider": "openmm",
                            "command": None,
                            "returncode": None,
                            "exception_class": type(exc).__name__,
                            "reason": str(exc),
                        },
                    )
                write_json_atomic(
                    _record_path(work_dir, args.partition, record["attempt_id"]), record
                )
                completed += 1
    print(f"Wrote {completed} attempt records; resumed/skipped {skipped} existing records.")


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


def _seed_for(base_seed: int, key: str) -> int:
    return (base_seed + int(sha256_bytes(key.encode("utf-8"))[:8], 16)) % (2**32)


def _metric_block(
    records: list[dict[str, Any]],
    *,
    expected_count: int,
    statistics: dict[str, Any],
    seed_key: str,
) -> dict[str, Any]:
    errors = [
        float(record["signed_error_kcal_mol"])
        for record in records
        if record["status"] == "success"
    ]
    return summarize_errors(
        errors,
        expected_count=expected_count,
        resamples=int(statistics["bootstrap_resamples"]),
        confidence=float(statistics["bootstrap_confidence"]),
        seed=_seed_for(int(statistics["bootstrap_seed"]), seed_key),
    )


def summarize(args: argparse.Namespace) -> None:
    protocol, fingerprint = load_protocol(args.protocol)
    work_dir = Path(args.work_dir).resolve()
    manifest = _load_prepared(work_dir, fingerprint)
    confirmation_lock = None
    if args.partition == "confirmation":
        confirmation_lock = ensure_confirmation_lock(work_dir, fingerprint)
    candidates = {
        candidate["compound_id"]: candidate
        for candidate in manifest["candidates"]
        if candidate["partition"] == args.partition
    }
    expected = expected_attempt_ids(
        candidates,
        protocol["methods"]["charge_methods"],
        protocol["methods"]["gb_models"],
    )
    record_dir = work_dir / "records" / args.partition
    missing = [attempt_id for attempt_id in expected if not (record_dir / f"{attempt_id}.json").is_file()]
    extra = sorted(
        path.stem for path in record_dir.glob("*.json") if path.stem not in set(expected)
    ) if record_dir.exists() else []
    if missing or extra:
        raise ValueError(
            f"Attempt reconciliation failed: missing={len(missing)}, extra={len(extra)}. "
            "Partial or post-hoc result sets cannot be summarized."
        )
    records: list[dict[str, Any]] = []
    record_hashes: dict[str, str] = {}
    for attempt_id in expected:
        path = record_dir / f"{attempt_id}.json"
        record = load_json(path)
        if (
            record.get("attempt_id") != attempt_id
            or record.get("protocol_fingerprint") != fingerprint
            or record.get("partition") != args.partition
        ):
            raise ValueError(f"Record identity/protocol mismatch: {path}")
        if record.get("status") not in {"success", "failure"}:
            raise ValueError(f"Record has invalid status: {path}")
        records.append(record)
        record_hashes[attempt_id] = sha256_file(path)

    statistics = protocol["statistics"]
    methods: dict[str, Any] = {}
    for charge_method in protocol["methods"]["charge_methods"]:
        for gb_model in protocol["methods"]["gb_models"]:
            key = f"{charge_method}/{gb_model}"
            group = [
                record
                for record in records
                if record["charge_method"] == charge_method and record["gb_model"] == gb_model
            ]
            methods[key] = _metric_block(
                group,
                expected_count=len(candidates),
                statistics=statistics,
                seed_key=f"{args.partition}/{key}/overall",
            )

    strata: dict[str, Any] = {}
    dimensions = {
        "functional_group": sorted(
            {
                group
                for candidate in candidates.values()
                for group in candidate["functional_groups"]
            }
        ),
        "element_class": sorted({candidate["element_class"] for candidate in candidates.values()}),
        "size": sorted({candidate["size_bin"] for candidate in candidates.values()}),
        "heteroatom_count": sorted(
            {candidate["heteroatom_bin"] for candidate in candidates.values()}
        ),
        "flexibility": sorted(
            {candidate["flexibility_bin"] for candidate in candidates.values()}
        ),
    }
    candidate_bin_keys = {
        "functional_group": "functional_groups",
        "element_class": "element_class",
        "size": "size_bin",
        "heteroatom_count": "heteroatom_bin",
        "flexibility": "flexibility_bin",
    }
    for dimension, labels in dimensions.items():
        strata[dimension] = {}
        for label in labels:
            if dimension == "functional_group":
                member_ids = {
                    compound_id
                    for compound_id, candidate in candidates.items()
                    if label in candidate[candidate_bin_keys[dimension]]
                }
            else:
                member_ids = {
                    compound_id
                    for compound_id, candidate in candidates.items()
                    if candidate[candidate_bin_keys[dimension]] == label
                }
            strata[dimension][label] = {}
            for charge_method in protocol["methods"]["charge_methods"]:
                for gb_model in protocol["methods"]["gb_models"]:
                    method_key = f"{charge_method}/{gb_model}"
                    group = [
                        record
                        for record in records
                        if record["compound_id"] in member_ids
                        and record["charge_method"] == charge_method
                        and record["gb_model"] == gb_model
                    ]
                    strata[dimension][label][method_key] = _metric_block(
                        group,
                        expected_count=len(member_ids),
                        statistics=statistics,
                        seed_key=f"{args.partition}/{dimension}/{label}/{method_key}",
                    )

    failures = [
        {
            "attempt_id": record["attempt_id"],
            "compound_id": record["compound_id"],
            "charge_method": record["charge_method"],
            "gb_model": record["gb_model"],
            **record["failure"],
        }
        for record in records
        if record["status"] == "failure"
    ]
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "claim_scope": protocol["claim_scope"],
        "partition": args.partition,
        "candidate_count": len(candidates),
        "attempt_count": len(records),
        "success_count": sum(record["status"] == "success" for record in records),
        "failure_count": len(failures),
        "methods": methods,
        "strata": strata,
        "failures": failures,
        "record_sha256": record_hashes,
        "confirmation_lock_sha256": (
            sha256_file(work_dir / protocol["confirmation"]["lock_filename"])
            if confirmation_lock is not None
            else None
        ),
        "interpretation": (
            "Neutral-water single-geometry evaluation only. These metrics are not a "
            "sampled finite-temperature free energy or broad chemical-domain certification."
        ),
    }
    write_json_atomic(args.output, summary)
    print(f"Wrote deterministic {args.partition} summary to {Path(args.output).resolve()}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="fetch, hash, classify, and partition")
    prepare_parser.add_argument("--protocol", required=True)
    prepare_parser.add_argument("--work-dir", required=True)
    prepare_parser.add_argument("--source-dir", help="offline directory containing pinned artifacts")
    prepare_parser.set_defaults(handler=prepare)

    run_parser = subparsers.add_parser("run", help="run the fixed charge and five-GB matrix")
    run_parser.add_argument("--protocol", required=True)
    run_parser.add_argument("--work-dir", required=True)
    run_parser.add_argument("--partition", choices=("development", "confirmation"), required=True)
    run_parser.add_argument("--antechamber", help="optional explicit Antechamber executable")
    run_parser.add_argument(
        "--max-compounds",
        type=int,
        help="smoke-only incomplete run; summaries intentionally reject missing attempts",
    )
    run_parser.set_defaults(handler=run)

    freeze_parser = subparsers.add_parser(
        "freeze-confirmation", help="immutably record the default and pass rule before confirmation"
    )
    freeze_parser.add_argument("--protocol", required=True)
    freeze_parser.add_argument("--work-dir", required=True)
    freeze_parser.add_argument("--proposed-default", required=True)
    freeze_parser.add_argument("--pass-rule", required=True)
    freeze_parser.set_defaults(handler=freeze_confirmation)

    summary_parser = subparsers.add_parser("summarize", help="reconcile and summarize stored records")
    summary_parser.add_argument("--protocol", required=True)
    summary_parser.add_argument("--work-dir", required=True)
    summary_parser.add_argument("--partition", choices=("development", "confirmation"), required=True)
    summary_parser.add_argument("--output", required=True)
    summary_parser.set_defaults(handler=summarize)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
