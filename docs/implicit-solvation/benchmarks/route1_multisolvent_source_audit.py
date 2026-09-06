#!/usr/bin/env python3
"""Audit a license-controlled MNSol multi-solvent panel without reading labels.

The audit is deliberately a source-and-geometry gate, not a benchmark scorer.
It verifies a user-supplied MNSol-v2012 archive, freezes the selected solvent
panel, and emits aggregate counts only.  Experimental ``DeltaGsolv`` values
are never copied into, or read by, the emitted artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, cast
from zipfile import BadZipFile, ZipFile, ZipInfo

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # pyright: ignore[reportImplicitRelativeImport]


MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_ZIP_MEMBERS = 20_000
MAX_TABLE_BYTES = 8 * 1024 * 1024
MAX_COMPRESSION_RATIO = 100.0
MNSOL_TABLE_NAME = "MNSol_alldata.txt"
MNSOL_REQUIRED_COLUMNS = frozenset(
    {
        "No.",
        "FileHandle",
        "Charge",
        "Solvent",
        "DeltaGsolv",
        "type",
        "eps",
        "n",
        "alpha",
        "beta",
        "gamma",
    }
)
MNSOL_METADATA_COLUMNS = ("FileHandle", "Charge", "Solvent", "type")
ALLOWED_PROTOCOL_KEYS = {
    "schema_version",
    "protocol_id",
    "benchmark_kind",
    "claim_scope",
    "route1_boundary",
    "dataset",
    "panel",
    "evaluation_design",
    "literature",
}


def _as_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be a JSON integer.")
    return int(value)


def _sha256(value: object, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a 64-character SHA256 digest.")
    return digest


def _safe_zip_member(info: ZipInfo) -> None:
    path = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(mode):
        raise ValueError(f"Unsafe ZIP member: {info.filename!r}.")
    if info.file_size > MAX_TABLE_BYTES and path.name == MNSOL_TABLE_NAME:
        raise ValueError("MNSol source table exceeds the configured size limit.")
    if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
        raise ValueError(f"Suspicious ZIP compression ratio: {info.filename!r}.")


def _load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol_obj = json.loads(Path(path).read_text(encoding="utf-8"))
    protocol = cast(dict[str, Any], protocol_obj)
    if not isinstance(protocol, dict):
        raise ValueError("Multi-solvent source protocol must be a JSON object.")
    unexpected = sorted(set(protocol) - ALLOWED_PROTOCOL_KEYS)
    if unexpected:
        raise ValueError("Unexpected protocol fields: " + ", ".join(unexpected))
    required = ALLOWED_PROTOCOL_KEYS - {"literature"}
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError("Missing protocol fields: " + ", ".join(missing))
    if protocol["schema_version"] != 1:
        raise ValueError("Only multi-solvent source protocol schema version 1 is supported.")
    if protocol["benchmark_kind"] != "route1-mnsol-multisolvent-source-audit":
        raise ValueError("Unexpected multi-solvent source protocol benchmark_kind.")
    if not str(protocol["protocol_id"]).strip():
        raise ValueError("Multi-solvent source protocol_id must be non-empty.")

    route = protocol["route1_boundary"]
    if not isinstance(route, dict) or route != {
        "name": "Additive fixed-charge PB/GB implicit solvation",
        "formula": "E_solution(R)=E_MLIP,gas(R)+G_polar(R,q_fixed)+G_nonpolar(R)",
        "gas_phase_mm_energy": False,
        "hydration_label_residual": False,
        "mlip_retraining": False,
        "fixed_charge": "AM1-BCC",
    }:
        raise ValueError("Multi-solvent audit must preserve the Route 1 boundary.")

    dataset = protocol["dataset"]
    if not isinstance(dataset, dict):
        raise ValueError("Multi-solvent protocol requires a dataset object.")
    if dataset.get("version") != "2012":
        raise ValueError("The multi-solvent audit is pinned to MNSol-v2012.")
    if dataset.get("acquisition_policy") != "user-supplied-no-auto-download":
        raise ValueError("MNSol acquisition must remain user-supplied-no-auto-download.")
    archive = dataset.get("archive")
    table = dataset.get("table")
    if not isinstance(archive, dict) or not isinstance(table, dict):
        raise ValueError("Multi-solvent protocol requires archive and table metadata.")
    if archive.get("name") != "MNSolDatabase_v2012.zip":
        raise ValueError("Unexpected MNSol archive name.")
    _sha256(archive.get("sha256"), "dataset.archive.sha256")
    if _as_int(archive.get("size_bytes"), "dataset.archive.size_bytes") <= 0:
        raise ValueError("dataset.archive.size_bytes must be positive.")
    if table.get("name") != MNSOL_TABLE_NAME:
        raise ValueError("Unexpected MNSol table name.")
    _sha256(table.get("sha256"), "dataset.table.sha256")
    if _as_int(table.get("size_bytes"), "dataset.table.size_bytes") <= 0:
        raise ValueError("dataset.table.size_bytes must be positive.")
    if _as_int(table.get("expected_rows"), "dataset.table.expected_rows") <= 0:
        raise ValueError("dataset.table.expected_rows must be positive.")

    panel = protocol["panel"]
    if not isinstance(panel, list) or len(panel) <= 10:
        raise ValueError("The multi-solvent panel must contain more than ten solvents.")
    names: set[str] = set()
    mnsol_names: set[str] = set()
    for entry in panel:
        if not isinstance(entry, dict):
            raise ValueError("Each multi-solvent panel entry must be an object.")
        name = str(entry.get("canonical_name", "")).strip()
        mnsol_name = str(entry.get("mnsol_name", "")).strip()
        if not name or not mnsol_name or name in names or mnsol_name in mnsol_names:
            raise ValueError("Multi-solvent panel names must be non-empty and unique.")
        names.add(name)
        mnsol_names.add(mnsol_name)
        all_records = _as_int(
            entry.get("expected_all_records"), f"panel[{name}].expected_all_records"
        )
        neutral_records = _as_int(
            entry.get("expected_neutral_absolute_records"),
            f"panel[{name}].expected_neutral_absolute_records",
        )
        if all_records < neutral_records or neutral_records < 7:
            raise ValueError(
                f"Panel solvent {name!r} requires at least seven neutral absolute records."
            )
        if not str(entry.get("rationale", "")).strip():
            raise ValueError(f"Panel solvent {name!r} requires a rationale.")

    design = protocol["evaluation_design"]
    if not isinstance(design, dict) or design != {
        "experimental_values_loaded": False,
        "fit_or_tuning_allowed": False,
        "source_rows_redistributed": False,
        "pooled_random_split_is_sufficient": False,
        "per_solvent_reporting_required": True,
        "leave_one_solvent_out_sensitivity_required": True,
    }:
        raise ValueError("Multi-solvent evaluation design must remain label-free and no-fit.")

    return protocol, core.sha256_bytes(core.canonical_json_bytes(protocol))


def _table_member(archive: ZipFile) -> ZipInfo:
    members = archive.infolist()
    if len(members) > MAX_ZIP_MEMBERS:
        raise ValueError("MNSol archive has too many ZIP members.")
    for member in members:
        _safe_zip_member(member)
    tables = [member for member in members if PurePosixPath(member.filename).name == MNSOL_TABLE_NAME]
    if len(tables) != 1:
        raise ValueError("MNSol archive must contain exactly one MNSol_alldata.txt table.")
    return tables[0]


def _geometry_handles(archive: ZipFile) -> frozenset[str]:
    handles: set[str] = set()
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if len(path.parts) >= 2 and path.parts[-2] == "all_solutes" and path.suffix == ".xyz":
            handles.add(path.stem)
    if not handles:
        raise ValueError("MNSol archive contains no all_solutes XYZ geometries.")
    return frozenset(handles)


def _read_table(
    archive: ZipFile,
    member: ZipInfo,
    expected: dict[str, Any],
) -> list[tuple[str, str, str, str]]:
    """Read only MNSol identity fields after integrity-checking the raw table.

    The raw bytes necessarily pass through this function for the pinned table
    digest, but parsed records retain only ``FileHandle``, ``Charge``,
    ``Solvent``, and ``type``.  In particular, ``DeltaGsolv`` is required in
    the header only to identify the official table; its row values are never
    decoded, indexed, retained, or emitted by this source audit.
    """
    if member.file_size != expected["size_bytes"]:
        raise ValueError(
            "MNSol table size mismatch: "
            f"expected {expected['size_bytes']}, observed {member.file_size}."
        )
    with archive.open(member, "r") as handle:
        payload = handle.read(MAX_TABLE_BYTES + 1)
    if len(payload) > MAX_TABLE_BYTES:
        raise ValueError("MNSol table exceeds the configured size limit.")
    observed_sha = core.sha256_bytes(payload)
    if observed_sha != expected["sha256"]:
        raise ValueError(
            "MNSol table SHA256 mismatch: "
            f"expected {expected['sha256']}, observed {observed_sha}."
        )
    lines = payload.splitlines()
    if not lines:
        raise ValueError("MNSol table is empty.")
    try:
        fieldnames = lines[0].decode("utf-8-sig").split("\t")
    except UnicodeDecodeError as exc:
        raise ValueError("MNSol table is not UTF-8 decodable.") from exc
    if len(fieldnames) != len(set(fieldnames)):
        raise ValueError("MNSol table has duplicate header fields.")
    if not MNSOL_REQUIRED_COLUMNS <= set(fieldnames):
        raise ValueError("MNSol table does not expose the required schema columns.")
    metadata_indices: tuple[int, int, int, int] = cast(
        tuple[int, int, int, int],
        tuple(fieldnames.index(column) for column in MNSOL_METADATA_COLUMNS),
    )
    rows: list[tuple[str, str, str, str]] = []
    for row_number, raw_line in enumerate(lines[1:], start=2):
        raw_columns = raw_line.split(b"\t")
        if len(raw_columns) != len(fieldnames):
            raise ValueError(
                f"MNSol table row {row_number} has {len(raw_columns)} columns; "
                f"expected {len(fieldnames)}."
            )
        try:
            record = (
                raw_columns[metadata_indices[0]].decode("utf-8"),
                raw_columns[metadata_indices[1]].decode("utf-8"),
                raw_columns[metadata_indices[2]].decode("utf-8"),
                raw_columns[metadata_indices[3]].decode("utf-8"),
            )
            rows.append(record)
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"MNSol metadata field at row {row_number} is not UTF-8 decodable."
            ) from exc
    if len(rows) != expected["expected_rows"]:
        raise ValueError(
            "MNSol row-count mismatch: "
            f"expected {expected['expected_rows']}, observed {len(rows)}."
        )
    return rows


def audit_archive(
    protocol: dict[str, Any],
    protocol_fingerprint: str,
    archive_path: str | Path,
) -> dict[str, Any]:
    """Return a label-free aggregate audit for the pinned user-supplied archive."""
    source = Path(archive_path)
    if not source.is_file():
        raise FileNotFoundError(f"MNSol archive is not a file: {source}")
    dataset = protocol["dataset"]
    expected_archive = dataset["archive"]
    observed_size = source.stat().st_size
    if observed_size > MAX_ARCHIVE_BYTES:
        raise ValueError("MNSol archive exceeds the configured size limit.")
    if observed_size != expected_archive["size_bytes"]:
        raise ValueError(
            "MNSol archive size mismatch: "
            f"expected {expected_archive['size_bytes']}, observed {observed_size}."
        )
    observed_archive_sha = core.sha256_file(source)
    if observed_archive_sha != expected_archive["sha256"]:
        raise ValueError(
            "MNSol archive SHA256 mismatch: "
            f"expected {expected_archive['sha256']}, observed {observed_archive_sha}."
        )

    try:
        with ZipFile(source) as archive:
            member = _table_member(archive)
            geometries = _geometry_handles(archive)
            rows = _read_table(archive, member, dataset["table"])
    except BadZipFile as exc:
        raise ValueError("MNSol source is not a readable ZIP archive.") from exc

    all_counts: Counter[str] = Counter()
    neutral_absolute_counts: Counter[str] = Counter()
    geometry_handles: dict[str, set[str]] = {
        str(entry["mnsol_name"]): set() for entry in protocol["panel"]
    }
    panel_names = set(geometry_handles)
    for handle, charge, solvent, process_type in rows:
        solvent = solvent.strip()
        if solvent not in panel_names:
            continue
        all_counts[solvent] += 1
        if charge.strip() == "0" and process_type.strip() == "abs":
            handle = handle.strip()
            if not handle or handle not in geometries:
                raise ValueError(
                    f"MNSol neutral absolute row for {solvent!r} lacks a matching XYZ geometry."
                )
            neutral_absolute_counts[solvent] += 1
            geometry_handles[solvent].add(handle)

    panel_result: list[dict[str, Any]] = []
    for entry in protocol["panel"]:
        solvent = str(entry["mnsol_name"])
        observed_all = int(all_counts[solvent])
        observed_neutral = int(neutral_absolute_counts[solvent])
        if observed_all != entry["expected_all_records"]:
            raise ValueError(
                f"MNSol all-record count mismatch for {solvent!r}: "
                f"expected {entry['expected_all_records']}, observed {observed_all}."
            )
        if observed_neutral != entry["expected_neutral_absolute_records"]:
            raise ValueError(
                f"MNSol neutral-absolute count mismatch for {solvent!r}: "
                f"expected {entry['expected_neutral_absolute_records']}, observed {observed_neutral}."
            )
        panel_result.append(
            {
                "canonical_name": entry["canonical_name"],
                "mnsol_name": solvent,
                "all_records": observed_all,
                "neutral_absolute_records": observed_neutral,
                "unique_neutral_geometry_count": len(geometry_handles[solvent]),
                "rationale": entry["rationale"],
            }
        )

    artifact = {
        "artifact_type": "route1-mnsol-multisolvent-source-audit-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": protocol_fingerprint,
        "route1_boundary": protocol["route1_boundary"],
        "evaluation_design": protocol["evaluation_design"],
        "source_archive": {
            "name": expected_archive["name"],
            "sha256": observed_archive_sha,
            "size_bytes": observed_size,
            "source_path_recorded": False,
        },
        "source_table": {
            "name": dataset["table"]["name"],
            "sha256": dataset["table"]["sha256"],
            "size_bytes": dataset["table"]["size_bytes"],
            "expected_rows": dataset["table"]["expected_rows"],
            "experimental_value_columns_loaded": [],
            "metadata_columns_loaded": ["FileHandle", "Charge", "Solvent", "type"],
        },
        "panel": panel_result,
        "counts": {
            "panel_solvents": len(panel_result),
            "panel_neutral_absolute_records": sum(
                entry["neutral_absolute_records"] for entry in panel_result
            ),
            "panel_unique_neutral_geometry_pairs": sum(
                entry["unique_neutral_geometry_count"] for entry in panel_result
            ),
        },
    }
    return core.seal_artifact(artifact)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol, fingerprint = _load_protocol(args.protocol)
    artifact = audit_archive(protocol, fingerprint, args.archive)
    artifact["command_provenance"] = core.command_provenance(
        __file__,
        {
            "protocol": _repo_relative(args.protocol),
            "archive_name": args.archive.name,
            "archive_sha256": core.sha256_file(args.archive),
            "output": _repo_relative(args.output),
        },
        repository_root=REPOSITORY_ROOT,
    )
    artifact = core.seal_artifact(artifact)
    core.write_json_atomic(args.output, artifact)


if __name__ == "__main__":
    main()
