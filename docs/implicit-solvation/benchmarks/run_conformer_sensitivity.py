#!/usr/bin/env python3
"""Run and summarize the frozen Route 1 conformer-sensitivity benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import threading
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core import (  # noqa: E402
    canonical_json_bytes,
    load_json,
    load_protocol,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from maple.function.calculator.extra_correction.implicit.openmm_gb import (  # noqa: E402
    KJ_PER_MOL_PER_HARTREE,
    OpenMMGB,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402


KCAL_PER_HARTREE = KJ_PER_MOL_PER_HARTREE / 4.184


def _load_conformer_protocol(path: str | Path) -> tuple[dict[str, Any], str, Path]:
    protocol_path = Path(path).resolve()
    protocol = load_json(protocol_path)
    if protocol.get("schema_version") != 1:
        raise ValueError("Unsupported conformer-sensitivity protocol schema.")
    if protocol.get("source_partition") != "development":
        raise ValueError("Conformer sensitivity must remain development-only.")
    cases = protocol.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Conformer-sensitivity protocol has no frozen cases.")
    case_ids = [case.get("compound_id") for case in cases]
    if any(not case_id for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("Conformer-sensitivity cases require unique compound IDs.")
    fingerprint = sha256_bytes(canonical_json_bytes(protocol))
    return protocol, fingerprint, protocol_path


def _relative_protocol_path(protocol_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (protocol_path.parent / path).resolve()


def _validate_base_evidence(
    protocol: dict[str, Any],
    protocol_path: Path,
    base_work_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    base = protocol["base_evidence"]
    base_protocol_path = _relative_protocol_path(protocol_path, base["protocol"])
    _base_protocol, base_fingerprint = load_protocol(base_protocol_path)
    if base_fingerprint != base["protocol_fingerprint"]:
        raise ValueError("Base FreeSolv protocol fingerprint changed.")

    summary_path = _relative_protocol_path(protocol_path, base["development_summary"])
    if sha256_file(summary_path) != base["development_summary_sha256"]:
        raise ValueError("Base development summary hash changed.")
    summary = load_json(summary_path)
    if (
        summary.get("protocol_fingerprint") != base_fingerprint
        or summary.get("partition") != "development"
    ):
        raise ValueError("Base development summary is not the frozen development evidence.")

    prepared_path = base_work_dir / "prepared.json"
    prepared = load_json(prepared_path)
    if prepared.get("protocol_fingerprint") != base_fingerprint:
        raise ValueError("Prepared FreeSolv workspace targets a different base protocol.")
    candidates = {candidate["compound_id"]: candidate for candidate in prepared["candidates"]}
    for case in protocol["cases"]:
        compound_id = case["compound_id"]
        candidate = candidates.get(compound_id)
        if candidate is None or candidate.get("partition") != "development":
            raise ValueError(f"Frozen conformer case is not in development: {compound_id}")
        for key in ("mol2_sha256", "flexibility_bin"):
            if candidate.get(key) != case.get(key):
                raise ValueError(f"Frozen conformer case metadata changed: {compound_id}/{key}")
        mol2_path = base_work_dir / candidate["mol2_relative_path"]
        if sha256_file(mol2_path) != case["mol2_sha256"]:
            raise ValueError(f"Frozen conformer MOL2 changed: {compound_id}")
    return summary, candidates


def _provider_version(executable: str, pattern: str) -> str:
    completed = subprocess.run(
        [executable, "--version"],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    text = completed.stdout + "\n" + completed.stderr
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not parse provider version from {executable}.")
    return match.group(1)


def _provider_environment(
    protocol: dict[str, Any], crest: str, xtb: str
) -> dict[str, Any]:
    crest_path = shutil.which(crest)
    xtb_path = shutil.which(xtb)
    if crest_path is None:
        raise FileNotFoundError(f"CREST executable not found: {crest}")
    if xtb_path is None:
        raise FileNotFoundError(f"xTB executable not found: {xtb}")
    crest_version = _provider_version(crest_path, r"Version\s+([0-9.]+)")
    xtb_version = _provider_version(xtb_path, r"xtb version\s+([0-9.]+)")
    required = protocol["generator"]["required_versions"]
    if crest_version != required["crest"] or xtb_version != required["xtb"]:
        raise ValueError(
            "Conformer provider version mismatch: "
            f"CREST {crest_version}/{required['crest']}, "
            f"xTB {xtb_version}/{required['xtb']}."
        )
    openmm_version = importlib.metadata.version("openmm")
    required_openmm = protocol["evaluation"]["required_openmm_version"]
    if openmm_version != required_openmm:
        raise ValueError(
            f"Conformer evaluation requires OpenMM {required_openmm}, "
            f"observed {openmm_version}."
        )
    return {
        "crest": {
            "version": crest_version,
            "executable": crest_path,
            "executable_sha256": sha256_file(crest_path),
        },
        "xtb": {
            "version": xtb_version,
            "executable": xtb_path,
            "executable_sha256": sha256_file(xtb_path),
        },
        "openmm_version": openmm_version,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _write_xyz(path: Path, atoms, title: str) -> None:
    lines = [str(len(atoms)), title]
    for symbol, (x, y, z) in zip(
        atoms.get_chemical_symbols(), atoms.get_positions(), strict=True
    ):
        lines.append(f"{symbol:<3s} {x: .10f} {y: .10f} {z: .10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_xyz_ensemble(
    path: Path, expected_symbols: list[str]
) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    frames: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        try:
            atom_count = int(lines[cursor].strip())
        except ValueError as exc:
            raise ValueError(f"Malformed XYZ atom count at line {cursor + 1}.") from exc
        if atom_count != len(expected_symbols):
            raise ValueError("CREST conformer atom count changed.")
        if cursor + atom_count + 1 >= len(lines):
            raise ValueError("Truncated CREST conformer ensemble.")
        try:
            crest_energy = float(lines[cursor + 1].strip().split()[0])
        except (IndexError, ValueError) as exc:
            raise ValueError("CREST conformer comment lacks a numeric energy.") from exc
        symbols: list[str] = []
        positions: list[list[float]] = []
        for line in lines[cursor + 2 : cursor + 2 + atom_count]:
            fields = line.split()
            if len(fields) < 4:
                raise ValueError("Malformed CREST XYZ coordinate row.")
            symbols.append(fields[0])
            positions.append([float(fields[1]), float(fields[2]), float(fields[3])])
        if symbols != expected_symbols:
            raise ValueError("CREST changed atom order or element identity.")
        array = np.asarray(positions, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("CREST conformer contains non-finite coordinates.")
        frames.append(
            {
                "crest_energy_hartree": crest_energy,
                "positions_angstrom": array,
            }
        )
        cursor += atom_count + 2
    if not frames:
        raise ValueError("CREST produced no conformers.")
    return frames


def _load_charge_cache(
    base_work_dir: Path,
    compound_id: str,
    charge_method: str,
    base_fingerprint: str,
    mol2_sha256: str,
) -> tuple[list[float], dict[str, Any]]:
    path = base_work_dir / "charges" / compound_id / f"{charge_method}.json"
    cache = load_json(path)
    if (
        cache.get("protocol_fingerprint") != base_fingerprint
        or cache.get("mol2_sha256") != mol2_sha256
        or cache.get("charge_method") != charge_method
    ):
        raise ValueError(f"Frozen charge cache mismatch: {compound_id}/{charge_method}")
    return list(cache["charges_e"]), dict(cache["provenance"])


def _method_stats(values: list[float], reference: float) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum_kcal_mol": float(array.min()),
        "maximum_kcal_mol": float(array.max()),
        "mean_kcal_mol": float(array.mean()),
        "standard_deviation_kcal_mol": float(array.std()),
        "range_kcal_mol": float(array.max() - array.min()),
        "max_abs_delta_from_reference_kcal_mol": float(
            np.max(np.abs(array - reference))
        ),
    }


def _run_case(
    case: dict[str, Any],
    *,
    protocol: dict[str, Any],
    protocol_fingerprint: str,
    base_summary: dict[str, Any],
    candidate: dict[str, Any],
    base_work_dir: Path,
    output_dir: Path,
    environment: dict[str, Any],
    timeout: float,
) -> tuple[int, int]:
    compound_id = case["compound_id"]
    record_path = output_dir / "records" / f"{compound_id}.json"
    if record_path.is_file():
        return 0, 1

    base_fingerprint = protocol["base_evidence"]["protocol_fingerprint"]
    mol2_path = base_work_dir / candidate["mol2_relative_path"]
    atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
    audit_dir = (
        output_dir
        / "audit"
        / compound_id
        / f"run-{os.getpid()}-{threading.get_ident()}"
    )
    audit_dir.mkdir(parents=True, exist_ok=False)
    input_path = audit_dir / "input.xyz"
    _write_xyz(input_path, atoms, compound_id)

    settings = protocol["generator"]["settings"]
    command = [
        environment["crest"]["executable"],
        input_path.name,
        "--v3",
        "--quick",
        f"--{settings['method']}",
        "--alpb",
        settings["solvent"],
        "--chrg",
        str(settings["charge"]),
        "--uhf",
        str(settings["uhf"]),
        "-T",
        str(settings["threads_per_case"]),
        "-xnam",
        environment["xtb"]["executable"],
        "--ewin",
        str(settings["energy_window_kcal_mol"]),
        "--rthr",
        str(settings["rmsd_threshold_angstrom"]),
        "--ethr",
        str(settings["energy_threshold_kcal_mol"]),
        "--bthr",
        str(settings["rotational_constant_threshold"]),
        "--temp",
        str(settings["temperature_kelvin"]),
    ]
    completed = subprocess.run(
        command,
        cwd=audit_dir,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    stdout_path = audit_dir / "crest.stdout.log"
    stderr_path = audit_dir / "crest.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    command_record = {
        "command": command,
        "returncode": completed.returncode,
        "timeout_seconds": timeout,
    }
    write_json_atomic(audit_dir / "crest.command.json", command_record)

    record: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "conformer-sensitivity-case",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": protocol_fingerprint,
        "base_protocol_fingerprint": base_fingerprint,
        "compound_id": compound_id,
        "name": case["name"],
        "flexibility_bin": case["flexibility_bin"],
        "mol2_sha256": case["mol2_sha256"],
        "environment": environment,
        "audit_dir": str(audit_dir),
        "command": command_record,
    }
    ensemble_path = audit_dir / "crest_conformers.xyz"
    if completed.returncode != 0 or not ensemble_path.is_file():
        record.update(
            status="failure",
            failure={
                "phase": "conformer_generation",
                "exception_class": "RuntimeError",
                "reason": (
                    f"CREST exited with {completed.returncode}"
                    if completed.returncode
                    else "CREST did not produce crest_conformers.xyz"
                ),
            },
        )
        write_json_atomic(record_path, record)
        return 1, 0

    try:
        frames = _read_xyz_ensemble(
            ensemble_path, list(atoms.get_chemical_symbols())
        )
        charges: dict[str, list[float]] = {}
        charge_provenance: dict[str, dict[str, Any]] = {}
        for charge_method in protocol["evaluation"]["charge_methods"]:
            values, provenance = _load_charge_cache(
                base_work_dir,
                compound_id,
                charge_method,
                base_fingerprint,
                case["mol2_sha256"],
            )
            charges[charge_method] = values
            charge_provenance[charge_method] = provenance

        methods: dict[str, Any] = {}
        for charge_method in protocol["evaluation"]["charge_methods"]:
            for gb_model in protocol["evaluation"]["gb_models"]:
                key = f"{charge_method}/{gb_model}"
                base_record_path = (
                    base_work_dir
                    / "records"
                    / "development"
                    / f"{compound_id}__{charge_method}__{gb_model}.json"
                )
                base_record = load_json(base_record_path)
                try:
                    provider = OpenMMGB(
                        atoms,
                        charges[charge_method],
                        model=gb_model,
                        nonpolar=protocol["evaluation"]["nonpolar"],
                        platform=protocol["evaluation"]["openmm_platform"],
                    )
                    conformer_values = []
                    for frame in frames:
                        conformer_atoms = atoms.copy()
                        conformer_atoms.set_positions(frame["positions_angstrom"])
                        result = provider.evaluate(conformer_atoms, need_forces=False)
                        conformer_values.append(
                            {
                                "polar_kcal_mol": (
                                    result.components_hartree["polar"]
                                    * KCAL_PER_HARTREE
                                ),
                                "nonpolar_kcal_mol": (
                                    result.components_hartree["nonpolar"]
                                    * KCAL_PER_HARTREE
                                ),
                                "total_kcal_mol": (
                                    result.energy_hartree * KCAL_PER_HARTREE
                                ),
                            }
                        )
                    if base_record.get("status") != "success":
                        raise ValueError(
                            f"Base development attempt is not successful: {key}"
                        )
                    reference = float(base_record["predicted_kcal_mol"])
                    totals = [value["total_kcal_mol"] for value in conformer_values]
                    methods[key] = {
                        "status": "success",
                        "reference_geometry_kcal_mol": reference,
                        "conformer_values": conformer_values,
                        "statistics": _method_stats(totals, reference),
                        "provider_provenance": provider.provenance,
                    }
                except Exception as exc:
                    methods[key] = {
                        "status": "failure",
                        "failure": {
                            "phase": "gb",
                            "exception_class": type(exc).__name__,
                            "reason": str(exc),
                        },
                        "base_attempt_status": base_record.get("status"),
                    }

        record.update(
            status="success",
            conformer_count=len(frames),
            crest_energy_hartree=[
                frame["crest_energy_hartree"] for frame in frames
            ],
            charge_provenance=charge_provenance,
            methods=methods,
            ensemble_sha256=sha256_file(ensemble_path),
            stdout_sha256=sha256_file(stdout_path),
            stderr_sha256=sha256_file(stderr_path),
            base_development_summary_sha256=protocol["base_evidence"][
                "development_summary_sha256"
            ],
            base_development_attempt_hashes={
                key: base_summary["record_sha256"][
                    f"{compound_id}__{key.replace('/', '__')}"
                ]
                for key in methods
            },
        )
    except Exception as exc:
        record.update(
            status="failure",
            failure={
                "phase": "evaluation",
                "exception_class": type(exc).__name__,
                "reason": str(exc),
            },
        )
    write_json_atomic(record_path, record)
    return 1, 0


def run(args: argparse.Namespace) -> None:
    protocol, fingerprint, protocol_path = _load_conformer_protocol(args.protocol)
    base_work_dir = Path(args.base_work_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    base_summary, candidates = _validate_base_evidence(
        protocol, protocol_path, base_work_dir
    )
    environment = _provider_environment(protocol, args.crest, args.xtb)
    jobs = int(args.jobs)
    if jobs <= 0:
        raise ValueError("--jobs must be positive.")
    timeout = float(args.timeout)
    if timeout <= 0:
        raise ValueError("--timeout must be positive.")

    def process(case: dict[str, Any]) -> tuple[int, int]:
        return _run_case(
            case,
            protocol=protocol,
            protocol_fingerprint=fingerprint,
            base_summary=base_summary,
            candidate=candidates[case["compound_id"]],
            base_work_dir=base_work_dir,
            output_dir=output_dir,
            environment=environment,
            timeout=timeout,
        )

    if jobs == 1:
        counts = [process(case) for case in protocol["cases"]]
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            counts = list(executor.map(process, protocol["cases"]))
    completed = sum(count[0] for count in counts)
    skipped = sum(count[1] for count in counts)
    print(f"Wrote {completed} conformer case records; resumed/skipped {skipped}.")


def _aggregate(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(array.max()),
    }


def summarize(args: argparse.Namespace) -> None:
    protocol, fingerprint, _protocol_path = _load_conformer_protocol(args.protocol)
    output_dir = Path(args.output_dir).resolve()
    expected = [case["compound_id"] for case in protocol["cases"]]
    record_dir = output_dir / "records"
    missing = [case_id for case_id in expected if not (record_dir / f"{case_id}.json").is_file()]
    extra = (
        sorted(path.stem for path in record_dir.glob("*.json") if path.stem not in set(expected))
        if record_dir.exists()
        else []
    )
    if missing or extra:
        raise ValueError(
            f"Conformer record reconciliation failed: missing={len(missing)}, "
            f"extra={len(extra)}."
        )

    records = [load_json(record_dir / f"{case_id}.json") for case_id in expected]
    for record in records:
        if (
            record.get("protocol_fingerprint") != fingerprint
            or record.get("compound_id") not in set(expected)
            or record.get("status") not in {"success", "failure"}
        ):
            raise ValueError("Invalid conformer-sensitivity record.")
    environment_payloads = {
        canonical_json_bytes(record["environment"])
        for record in records
        if record.get("environment")
    }
    if len(environment_payloads) != 1:
        raise ValueError("Conformer records used inconsistent provider environments.")

    method_keys = [
        f"{charge_method}/{gb_model}"
        for charge_method in protocol["evaluation"]["charge_methods"]
        for gb_model in protocol["evaluation"]["gb_models"]
    ]
    flexibility_by_case = {
        case["compound_id"]: case["flexibility_bin"] for case in protocol["cases"]
    }
    methods: dict[str, Any] = {}
    for key in method_keys:
        successful = [
            record
            for record in records
            if record["status"] == "success"
            and record["methods"][key]["status"] == "success"
        ]
        ranges = [
            record["methods"][key]["statistics"]["range_kcal_mol"]
            for record in successful
        ]
        reference_deltas = [
            record["methods"][key]["statistics"][
                "max_abs_delta_from_reference_kcal_mol"
            ]
            for record in successful
        ]
        methods[key] = {
            "expected_case_count": len(records),
            "successful_case_count": len(successful),
            "failure_rate": (len(records) - len(successful)) / len(records),
            "conformer_range_kcal_mol": _aggregate(ranges),
            "max_abs_delta_from_reference_kcal_mol": _aggregate(reference_deltas),
            "cases": {
                record["compound_id"]: record["methods"][key]["statistics"]
                for record in successful
            },
            "failures": [
                {
                    "compound_id": record["compound_id"],
                    **(
                        record.get("failure")
                        if record["status"] == "failure"
                        else record["methods"][key]["failure"]
                    ),
                }
                for record in records
                if record["status"] == "failure"
                or record["methods"][key]["status"] == "failure"
            ],
        }
        methods[key]["flexibility_strata"] = {}
        for label in protocol["selection"]["per_stratum"]:
            stratum_records = [
                record
                for record in records
                if flexibility_by_case[record["compound_id"]] == label
            ]
            stratum_successful = [
                record
                for record in stratum_records
                if record["status"] == "success"
                and record["methods"][key]["status"] == "success"
            ]
            methods[key]["flexibility_strata"][label] = {
                "expected_case_count": len(stratum_records),
                "successful_case_count": len(stratum_successful),
                "failure_rate": (
                    (len(stratum_records) - len(stratum_successful))
                    / len(stratum_records)
                ),
                "conformer_range_kcal_mol": _aggregate(
                    [
                        record["methods"][key]["statistics"]["range_kcal_mol"]
                        for record in stratum_successful
                    ]
                ),
                "max_abs_delta_from_reference_kcal_mol": _aggregate(
                    [
                        record["methods"][key]["statistics"][
                            "max_abs_delta_from_reference_kcal_mol"
                        ]
                        for record in stratum_successful
                    ]
                ),
            }

    summary = {
        "schema_version": 1,
        "artifact_type": "conformer-sensitivity-summary",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "source_partition": "development",
        "base_protocol_fingerprint": protocol["base_evidence"][
            "protocol_fingerprint"
        ],
        "base_development_summary_sha256": protocol["base_evidence"][
            "development_summary_sha256"
        ],
        "case_count": len(records),
        "successful_generator_case_count": sum(
            record["status"] == "success" for record in records
        ),
        "generator_failure_count": sum(
            record["status"] == "failure" for record in records
        ),
        "total_conformer_count": sum(
            record["conformer_count"]
            for record in records
            if record["status"] == "success"
        ),
        "conformer_count_by_case": {
            record["compound_id"]: record["conformer_count"]
            for record in records
            if record["status"] == "success"
        },
        "conformer_count": _aggregate(
            [
                float(record["conformer_count"])
                for record in records
                if record["status"] == "success"
            ]
        ),
        "conformer_count_by_flexibility": {
            label: _aggregate(
                [
                    float(record["conformer_count"])
                    for record in records
                    if record["status"] == "success"
                    and flexibility_by_case[record["compound_id"]] == label
                ]
            )
            for label in protocol["selection"]["per_stratum"]
        },
        "environment": json.loads(next(iter(environment_payloads))),
        "methods": methods,
        "generator_failures": [
            {"compound_id": record["compound_id"], **record["failure"]}
            for record in records
            if record["status"] == "failure"
        ],
        "record_sha256": {
            case_id: sha256_file(record_dir / f"{case_id}.json")
            for case_id in expected
        },
        "interpretation": protocol["interpretation"],
    }
    write_json_atomic(args.output, summary)
    print(f"Wrote conformer-sensitivity summary to {Path(args.output).resolve()}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    run_parser = subparsers.add_parser("run", help="generate and evaluate frozen conformers")
    run_parser.add_argument("--protocol", required=True)
    run_parser.add_argument("--base-work-dir", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--crest", default="crest")
    run_parser.add_argument("--xtb", default="xtb")
    run_parser.add_argument("--jobs", type=int, default=1)
    run_parser.add_argument("--timeout", type=float, default=1800.0)
    run_parser.set_defaults(handler=run)

    summary_parser = subparsers.add_parser("summarize", help="summarize frozen case records")
    summary_parser.add_argument("--protocol", required=True)
    summary_parser.add_argument("--output-dir", required=True)
    summary_parser.add_argument("--output", required=True)
    summary_parser.set_defaults(handler=summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
