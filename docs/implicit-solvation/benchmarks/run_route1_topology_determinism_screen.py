#!/usr/bin/env python3
"""Audit repeated parmchk2 topology construction for Route 1 matrix inputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core import (  # pyright: ignore[reportImplicitRelativeImport]
    artifact_content_sha256,
    canonical_json_bytes,
    command_provenance,
    load_json,
    seal_artifact,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from run_route1_performance import (  # pyright: ignore[reportImplicitRelativeImport]
    load_charge_vector,
)

from maple.function.calculator.extra_correction.implicit.amber_chagb import (
    render_typed_mol2,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader

HEX = frozenset("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    text = str(value).lower()
    return len(text) == 64 and set(text) <= HEX


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object.")
    return cast(Mapping[str, Any], value)


def _as_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be a JSON integer.")
    return value


def _repository_path(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe repository-relative path: {value!r}.")
    return (REPOSITORY_ROOT / path).resolve()


def _recorded_path(value: object, label: str) -> Path:
    path = Path(str(value))
    if not path.parts:
        raise ValueError(f"{label} must not be empty.")
    if path.is_absolute():
        return path.resolve()
    if ".." in path.parts:
        raise ValueError(f"{label} must not escape the repository.")
    return (REPOSITORY_ROOT / path).resolve()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def load_and_validate_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = load_json(path)
    if (
        not isinstance(protocol, dict)
        or protocol.get("protocol_id")
        != "maple-route1-parmchk2-topology-determinism-v1"
    ):
        raise ValueError("Unsupported Route 1 topology-determinism protocol.")
    if _as_int(protocol.get("schema_version"), "schema_version") != 1:
        raise ValueError("Unsupported Route 1 topology-determinism schema.")
    performance_protocol = _repository_path(
        protocol.get("performance_protocol_relative_path")
    )
    charge_manifest = _repository_path(protocol.get("charge_manifest_relative_path"))
    pinned_inputs = (
        (performance_protocol, protocol.get("performance_protocol_sha256")),
        (charge_manifest, protocol.get("charge_manifest_sha256")),
    )
    if any(
        not _is_sha256(expected)
        or not input_path.is_file()
        or sha256_file(input_path) != expected
        for input_path, expected in pinned_inputs
    ):
        raise ValueError("Pinned topology-screen protocol input changed.")

    execution = _require_mapping(protocol.get("execution"), "execution")
    repeat_count = _as_int(execution.get("repeat_count"), "execution.repeat_count")
    if (
        repeat_count < 2
        or execution.get("sequential") is not True
        or execution.get("parmchk2_arguments") != ["-f", "mol2", "-s", "gaff2"]
        or execution.get("determinism_identity")
        != "byte-level SHA256 of molecule.frcmod"
        or execution.get("acceptance_rule")
        != "exactly one observed frcmod SHA256 across all repeats"
    ):
        raise ValueError("Topology-screen execution contract changed.")

    candidates = protocol.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("Topology screen requires exactly two candidates.")
    roles: set[str] = set()
    compound_ids: set[str] = set()
    for raw_candidate in candidates:
        candidate = _require_mapping(raw_candidate, "candidate")
        compound_id = str(candidate.get("compound_id", ""))
        role = str(candidate.get("role", ""))
        if (
            not compound_id
            or compound_id in compound_ids
            or role not in {"rejected_large_candidate", "accepted_large_candidate"}
            or role in roles
        ):
            raise ValueError("Topology-screen candidate identity changed.")
        atom_count = _as_int(candidate.get("atom_count"), f"{compound_id}.atom_count")
        heavy_atom_count = _as_int(
            candidate.get("heavy_atom_count"), f"{compound_id}.heavy_atom_count"
        )
        mol2_path = _repository_path(candidate.get("mol2_relative_path"))
        if (
            atom_count < 1
            or heavy_atom_count < 1
            or heavy_atom_count > atom_count
            or not _is_sha256(candidate.get("mol2_sha256"))
            or not mol2_path.is_file()
            or sha256_file(mol2_path) != candidate["mol2_sha256"]
        ):
            raise ValueError(f"Topology-screen candidate changed: {compound_id}.")
        roles.add(role)
        compound_ids.add(compound_id)
    return protocol, sha256_bytes(canonical_json_bytes(protocol))


def _executable_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _portable_path(path),
        "sha256": sha256_file(path),
    }
    wrapped = path.parent / "wrapped_progs" / path.name
    if wrapped.is_file():
        record["wrapped_program"] = {
            "path": _portable_path(wrapped),
            "sha256": sha256_file(wrapped),
        }
    return record


def _prepare_typed_mol2(
    candidate: Mapping[str, Any],
    *,
    charge_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    source = _repository_path(candidate["mol2_relative_path"])
    atoms = MOL2Reader(str(source), charge=0, mult=1)
    atom_count = _as_int(candidate["atom_count"], "candidate.atom_count")
    if len(atoms) != atom_count:
        raise ValueError(f"Atom count changed for {candidate['compound_id']}.")
    heavy_atom_count = sum(symbol != "H" for symbol in atoms.get_chemical_symbols())
    if heavy_atom_count != _as_int(
        candidate["heavy_atom_count"], "candidate.heavy_atom_count"
    ):
        raise ValueError(f"Heavy-atom count changed for {candidate['compound_id']}.")
    charges, charge_record = load_charge_vector(
        charge_manifest,
        str(candidate["compound_id"]),
        atom_count=atom_count,
    )
    if charge_record.get("source_mol2_sha256") != candidate["mol2_sha256"]:
        raise ValueError(f"Charge provenance changed for {candidate['compound_id']}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_typed_mol2(
            source.read_text(encoding="utf-8"),
            np.asarray(atoms.get_positions(), dtype=np.float64),
            charges,
        ),
        encoding="utf-8",
    )
    return {
        "source_mol2": _portable_path(source),
        "source_mol2_sha256": sha256_file(source),
        "typed_mol2_sha256": sha256_file(output),
        "charge_count": len(charges),
        "charge_source_mol2_sha256": charge_record["source_mol2_sha256"],
    }


def _run_candidate(
    candidate: Mapping[str, Any],
    *,
    charge_manifest: Path,
    parmchk2: Path,
    repeat_count: int,
    work_dir: Path,
) -> dict[str, Any]:
    candidate_dir = work_dir / str(candidate["compound_id"])
    typed_mol2 = candidate_dir / "typed-input.mol2"
    input_record = _prepare_typed_mol2(
        candidate,
        charge_manifest=charge_manifest,
        output=typed_mol2,
    )
    runs = []
    representatives: dict[str, str] = {}
    for index in range(repeat_count):
        repeat_dir = candidate_dir / f"repeat-{index:02d}"
        repeat_dir.mkdir(parents=True, exist_ok=False)
        local_mol2 = repeat_dir / "molecule.mol2"
        shutil.copyfile(typed_mol2, local_mol2)
        command = [
            str(parmchk2),
            "-i",
            local_mol2.name,
            "-f",
            "mol2",
            "-o",
            "molecule.frcmod",
            "-s",
            "gaff2",
        ]
        completed = subprocess.run(
            command,
            cwd=repeat_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"parmchk2 failed for {candidate['compound_id']} repeat {index}: "
                f"{completed.stderr[-1000:]}"
            )
        frcmod = repeat_dir / "molecule.frcmod"
        if not frcmod.is_file():
            raise RuntimeError("parmchk2 did not create molecule.frcmod.")
        digest = sha256_file(frcmod)
        representatives.setdefault(digest, frcmod.read_text(encoding="utf-8"))
        runs.append(
            {
                "repeat_index": index,
                "frcmod_sha256": digest,
                "stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
                "stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
            }
        )
    histogram = dict(sorted(Counter(run["frcmod_sha256"] for run in runs).items()))
    return {
        "compound_id": candidate["compound_id"],
        "name": candidate["name"],
        "role": candidate["role"],
        "input": input_record,
        "repeat_count": repeat_count,
        "runs": runs,
        "frcmod_sha256_histogram": histogram,
        "variant_count": len(histogram),
        "deterministic": len(histogram) == 1,
        "representative_frcmod_by_sha256": {
            digest: representatives[digest] for digest in sorted(representatives)
        },
    }


def _selection_decision(records: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    by_role = {str(record["role"]): record for record in records}
    if set(by_role) != {"accepted_large_candidate", "rejected_large_candidate"}:
        raise ValueError("Topology-screen records do not cover both candidate roles.")
    accepted = by_role["accepted_large_candidate"].get("deterministic") is True
    rejected = by_role["rejected_large_candidate"].get("deterministic") is False
    return {
        "accepted_candidate_passed": accepted,
        "rejected_candidate_failed": rejected,
        "selection_supported": accepted and rejected,
    }


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    protocol_path: str | Path,
    artifact_path: str | Path | None = None,
    expected_content_sha256: str | None = None,
    verify_external_executables: bool = True,
) -> None:
    """Validate a topology-screen artifact and all derivable claims fail-closed."""
    expected_keys = {
        "schema_version",
        "artifact_type",
        "protocol_id",
        "protocol_fingerprint",
        "command_provenance",
        "executable",
        "charge_manifest",
        "records",
        "decision",
        "selection_boundary",
        "content_sha256",
    }
    if set(artifact) != expected_keys:
        raise ValueError("Topology-screen artifact schema is invalid.")
    if (
        _as_int(artifact.get("schema_version"), "artifact.schema_version") != 1
        or artifact.get("artifact_type")
        != "route1-parmchk2-topology-determinism-screen"
    ):
        raise ValueError("Unsupported topology-screen artifact.")

    protocol_file = Path(protocol_path).resolve()
    protocol, protocol_fingerprint = load_and_validate_protocol(protocol_file)
    if (
        artifact.get("protocol_id") != protocol["protocol_id"]
        or artifact.get("protocol_fingerprint") != protocol_fingerprint
    ):
        raise ValueError("Topology-screen artifact uses a different protocol.")
    content_sha256 = artifact.get("content_sha256")
    if not _is_sha256(content_sha256) or content_sha256 != artifact_content_sha256(
        artifact
    ):
        raise ValueError("Topology-screen artifact content hash is invalid.")
    if expected_content_sha256 is not None and (
        not _is_sha256(expected_content_sha256)
        or content_sha256 != expected_content_sha256
    ):
        raise ValueError("Topology-screen artifact differs from the frozen content.")

    command = _require_mapping(
        artifact.get("command_provenance"), "artifact.command_provenance"
    )
    arguments = _require_mapping(
        command.get("arguments"), "artifact.command_provenance.arguments"
    )
    expected_script = (
        "docs/implicit-solvation/benchmarks/run_route1_topology_determinism_screen.py"
    )
    if (
        set(arguments)
        != {"command", "protocol", "ambertools_bin", "work_dir", "output"}
        or arguments.get("command") != "run"
        or command.get("script") != expected_script
        or command.get("script_sha256") != sha256_file(__file__)
        or command.get("arguments_sha256")
        != sha256_bytes(canonical_json_bytes(arguments))
        or not str(command.get("python_executable", "")).strip()
        or command.get("environment_variables") != {}
    ):
        raise ValueError("Topology-screen command provenance is invalid.")
    if (
        _recorded_path(
            arguments.get("protocol"), "artifact.command_provenance.arguments.protocol"
        )
        != protocol_file
    ):
        raise ValueError("Topology-screen command references a different protocol.")
    if (
        artifact_path is not None
        and _recorded_path(
            arguments.get("output"), "artifact.command_provenance.arguments.output"
        )
        != Path(artifact_path).resolve()
    ):
        raise ValueError("Topology-screen command references a different output.")
    work_dir = _recorded_path(
        arguments.get("work_dir"), "artifact.command_provenance.arguments.work_dir"
    )
    if work_dir == REPOSITORY_ROOT:
        raise ValueError("Topology-screen work directory is invalid.")
    ambertools_bin = _recorded_path(
        arguments.get("ambertools_bin"),
        "artifact.command_provenance.arguments.ambertools_bin",
    )
    parmchk2 = ambertools_bin / "parmchk2"

    executable = _require_mapping(artifact.get("executable"), "artifact.executable")
    if set(executable) not in (
        {"path", "sha256"},
        {"path", "sha256", "wrapped_program"},
    ):
        raise ValueError("Topology-screen executable schema is invalid.")
    if _recorded_path(
        executable.get("path"), "artifact.executable.path"
    ) != parmchk2 or not _is_sha256(executable.get("sha256")):
        raise ValueError("Topology-screen parmchk2 provenance is invalid.")
    wrapped = executable.get("wrapped_program")
    if wrapped is not None:
        wrapped_record = _require_mapping(
            wrapped, "artifact.executable.wrapped_program"
        )
        wrapped_path = parmchk2.parent / "wrapped_progs" / parmchk2.name
        if (
            set(wrapped_record) != {"path", "sha256"}
            or _recorded_path(
                wrapped_record.get("path"), "artifact.executable.wrapped_program.path"
            )
            != wrapped_path
            or not _is_sha256(wrapped_record.get("sha256"))
        ):
            raise ValueError("Topology-screen wrapped parmchk2 provenance is invalid.")
    if verify_external_executables:
        if (
            not parmchk2.is_file()
            or not os.access(parmchk2, os.X_OK)
            or sha256_file(parmchk2) != executable["sha256"]
        ):
            raise ValueError("Recorded parmchk2 is unavailable or changed.")
        if wrapped is not None:
            wrapped_record = _require_mapping(
                wrapped, "artifact.executable.wrapped_program"
            )
            wrapped_path = _recorded_path(
                wrapped_record["path"], "artifact.executable.wrapped_program.path"
            )
            if (
                not wrapped_path.is_file()
                or not os.access(wrapped_path, os.X_OK)
                or sha256_file(wrapped_path) != wrapped_record["sha256"]
            ):
                raise ValueError("Recorded wrapped parmchk2 is unavailable or changed.")

    charge_manifest = _require_mapping(
        artifact.get("charge_manifest"), "artifact.charge_manifest"
    )
    expected_charge_manifest = _repository_path(
        protocol["charge_manifest_relative_path"]
    )
    if (
        set(charge_manifest) != {"path", "sha256"}
        or _recorded_path(charge_manifest.get("path"), "artifact.charge_manifest.path")
        != expected_charge_manifest
        or charge_manifest.get("sha256") != protocol["charge_manifest_sha256"]
        or not expected_charge_manifest.is_file()
        or sha256_file(expected_charge_manifest) != charge_manifest["sha256"]
    ):
        raise ValueError("Topology-screen charge-manifest provenance is invalid.")

    raw_records = artifact.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != len(
        protocol["candidates"]
    ):
        raise ValueError("Topology-screen records do not cover the protocol.")
    records = [_require_mapping(record, "artifact.records[]") for record in raw_records]
    by_compound = {str(record.get("compound_id", "")): record for record in records}
    if len(by_compound) != len(records):
        raise ValueError("Topology-screen compound records are duplicated.")
    repeat_count = _as_int(
        protocol["execution"]["repeat_count"], "execution.repeat_count"
    )
    expected_record_keys = {
        "compound_id",
        "name",
        "role",
        "input",
        "repeat_count",
        "runs",
        "frcmod_sha256_histogram",
        "variant_count",
        "deterministic",
        "representative_frcmod_by_sha256",
    }
    for raw_candidate in protocol["candidates"]:
        candidate = _require_mapping(raw_candidate, "protocol.candidates[]")
        compound_id = str(candidate["compound_id"])
        record = by_compound.get(compound_id)
        if record is None or set(record) != expected_record_keys:
            raise ValueError(f"Topology-screen record schema changed: {compound_id}.")
        if (
            record.get("name") != candidate["name"]
            or record.get("role") != candidate["role"]
            or _as_int(
                record.get("repeat_count"),
                f"artifact.records.{compound_id}.repeat_count",
            )
            != repeat_count
        ):
            raise ValueError(f"Topology-screen record metadata changed: {compound_id}.")

        input_record = _require_mapping(
            record.get("input"), f"artifact.records.{compound_id}.input"
        )
        expected_source = _repository_path(candidate["mol2_relative_path"])
        expected_source_sha = candidate["mol2_sha256"]
        if (
            set(input_record)
            != {
                "source_mol2",
                "source_mol2_sha256",
                "typed_mol2_sha256",
                "charge_count",
                "charge_source_mol2_sha256",
            }
            or _recorded_path(
                input_record.get("source_mol2"),
                f"artifact.records.{compound_id}.input.source_mol2",
            )
            != expected_source
            or input_record.get("source_mol2_sha256") != expected_source_sha
            or input_record.get("charge_source_mol2_sha256") != expected_source_sha
            or _as_int(
                input_record.get("charge_count"),
                f"artifact.records.{compound_id}.input.charge_count",
            )
            != _as_int(candidate["atom_count"], f"{compound_id}.atom_count")
            or not _is_sha256(input_record.get("typed_mol2_sha256"))
            or not expected_source.is_file()
            or sha256_file(expected_source) != expected_source_sha
        ):
            raise ValueError(
                f"Topology-screen input provenance changed: {compound_id}."
            )

        raw_runs = record.get("runs")
        if not isinstance(raw_runs, list) or len(raw_runs) != repeat_count:
            raise ValueError(f"Topology-screen repeat coverage changed: {compound_id}.")
        run_hashes: list[str] = []
        for index, raw_run in enumerate(raw_runs):
            run_record = _require_mapping(
                raw_run, f"artifact.records.{compound_id}.runs[{index}]"
            )
            if (
                set(run_record)
                != {
                    "repeat_index",
                    "frcmod_sha256",
                    "stdout_sha256",
                    "stderr_sha256",
                }
                or _as_int(
                    run_record.get("repeat_index"),
                    f"artifact.records.{compound_id}.runs[{index}].repeat_index",
                )
                != index
                or any(
                    not _is_sha256(run_record.get(key))
                    for key in ("frcmod_sha256", "stdout_sha256", "stderr_sha256")
                )
            ):
                raise ValueError(
                    f"Topology-screen repeat record changed: {compound_id}/{index}."
                )
            run_hashes.append(str(run_record["frcmod_sha256"]))

        histogram = _require_mapping(
            record.get("frcmod_sha256_histogram"),
            f"artifact.records.{compound_id}.frcmod_sha256_histogram",
        )
        normalized_histogram: dict[str, int] = {}
        for digest, count in histogram.items():
            if not _is_sha256(digest):
                raise ValueError(
                    f"Topology-screen histogram hash is invalid: {compound_id}."
                )
            normalized_count = _as_int(
                count, f"artifact.records.{compound_id}.histogram.{digest}"
            )
            if normalized_count < 1:
                raise ValueError(
                    f"Topology-screen histogram count is invalid: {compound_id}."
                )
            normalized_histogram[str(digest)] = normalized_count
        expected_histogram = dict(sorted(Counter(run_hashes).items()))
        if normalized_histogram != expected_histogram:
            raise ValueError(
                f"Topology-screen histogram is not reproducible: {compound_id}."
            )

        representatives = _require_mapping(
            record.get("representative_frcmod_by_sha256"),
            f"artifact.records.{compound_id}.representative_frcmod_by_sha256",
        )
        if set(representatives) != set(expected_histogram):
            raise ValueError(
                f"Topology-screen representatives are incomplete: {compound_id}."
            )
        for digest, content in representatives.items():
            if not isinstance(content, str) or sha256_bytes(content.encode()) != digest:
                raise ValueError(
                    f"Topology-screen representative hash changed: {compound_id}."
                )
        variant_count = len(expected_histogram)
        if _as_int(
            record.get("variant_count"),
            f"artifact.records.{compound_id}.variant_count",
        ) != variant_count or record.get("deterministic") is not (variant_count == 1):
            raise ValueError(
                f"Topology-screen determinism decision changed: {compound_id}."
            )

    if set(by_compound) != {
        str(candidate["compound_id"]) for candidate in protocol["candidates"]
    }:
        raise ValueError("Topology-screen records contain an unexpected compound.")
    if artifact.get("decision") != _selection_decision(records):
        raise ValueError("Topology-screen selection decision is not reproducible.")
    if artifact.get("selection_boundary") != protocol["selection_boundary"]:
        raise ValueError("Topology-screen selection boundary changed.")


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol, protocol_fingerprint = load_and_validate_protocol(args.protocol)
    ambertools_bin = Path(args.ambertools_bin).resolve()
    parmchk2 = ambertools_bin / "parmchk2"
    if not parmchk2.is_file() or not os.access(parmchk2, os.X_OK):
        raise FileNotFoundError(parmchk2)
    work_dir = Path(args.work_dir).resolve()
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(
            f"Topology-screen work directory is not empty: {work_dir}"
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    charge_manifest = _repository_path(protocol["charge_manifest_relative_path"])
    repeat_count = _as_int(
        protocol["execution"]["repeat_count"], "execution.repeat_count"
    )
    records = [
        _run_candidate(
            _require_mapping(candidate, "candidate"),
            charge_manifest=charge_manifest,
            parmchk2=parmchk2,
            repeat_count=repeat_count,
            work_dir=work_dir,
        )
        for candidate in protocol["candidates"]
    ]
    decision = _selection_decision(records)
    result = seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-parmchk2-topology-determinism-screen",
            "protocol_id": protocol["protocol_id"],
            "protocol_fingerprint": protocol_fingerprint,
            "command_provenance": command_provenance(
                __file__,
                vars(args),
                repository_root=REPOSITORY_ROOT,
            ),
            "executable": _executable_record(parmchk2),
            "charge_manifest": {
                "path": _portable_path(charge_manifest),
                "sha256": sha256_file(charge_manifest),
            },
            "records": records,
            "decision": decision,
            "selection_boundary": protocol["selection_boundary"],
        }
    )
    validate_artifact(
        result,
        protocol_path=args.protocol,
        artifact_path=args.output,
        expected_content_sha256=result["content_sha256"],
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the repeated topology screen.")
    run_parser.add_argument("--protocol", required=True)
    run_parser.add_argument("--ambertools-bin", required=True)
    run_parser.add_argument("--work-dir", required=True)
    run_parser.add_argument("--output", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a frozen topology-screen artifact."
    )
    validate_parser.add_argument("--protocol", required=True)
    validate_parser.add_argument("--artifact", required=True)
    validate_parser.add_argument("--expected-content-sha256")
    validate_parser.add_argument(
        "--skip-external-executable-check",
        action="store_true",
        help="Validate portable frozen evidence without requiring AmberTools locally.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        result = run(args)
        write_json_atomic(args.output, result)
        print(json.dumps(result, indent=2))
        return
    artifact = load_json(args.artifact)
    if not isinstance(artifact, dict):
        raise TypeError("Topology-screen artifact must be a JSON object.")
    validate_artifact(
        artifact,
        protocol_path=args.protocol,
        artifact_path=args.artifact,
        expected_content_sha256=args.expected_content_sha256,
        verify_external_executables=not args.skip_external_executable_check,
    )
    print(json.dumps({"valid": True, "content_sha256": artifact["content_sha256"]}))


if __name__ == "__main__":
    main()
