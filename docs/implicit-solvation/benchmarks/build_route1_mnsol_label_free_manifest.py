#!/usr/bin/env python3
"""Build one solvent's label-free MNSol record-identity manifest.

The user-supplied MNSol-v2012 archive is integrity checked against the frozen
source protocol.  Only ``FileHandle``, ``Charge``, ``Solvent``, and ``type``
are decoded from data rows.  ``DeltaGsolv`` is required in the official table
header, so the combined table bytes are necessarily read for integrity and row
framing, but its values are never decoded, interpreted, retained, or emitted.
The full table digest binds source provenance; a separate record identity and
record-set digest intentionally exclude that digest and remain label invariant.

The output contains license-controlled row identities and is intended for a
local ``.omx`` workspace only; it is not a redistributable benchmark artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path, PurePosixPath
from typing import Any, cast
from zipfile import BadZipFile, ZipFile, ZipInfo

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # pyright: ignore[reportImplicitRelativeImport]
import route1_multisolvent_source_audit as source_audit  # pyright: ignore[reportImplicitRelativeImport]


ALLOWED_ROW_COLUMNS = ("FileHandle", "Charge", "Solvent", "type")
FORBIDDEN_VALUE_FIELDS = frozenset(
    {
        "deltagsolv",
        "experimental_kcal_mol",
        "experimental_uncertainty_kcal_mol",
        "signed_error_kcal_mol",
        "absolute_error_kcal_mol",
    }
)
MAX_GEOMETRY_BYTES = 4 * 1024 * 1024


def _require_keys(
    value: object,
    required: set[str],
    field: str,
    *,
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object.")
    keys = set(value)
    allowed = required | (optional or set())
    missing = sorted(required - keys)
    unexpected = sorted(keys - allowed)
    if missing or unexpected:
        raise ValueError(
            f"{field} schema mismatch; missing={missing}, unexpected={unexpected}."
        )
    return cast(dict[str, Any], value)


def _repo_reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.name


def _load_source_audit(
    path: str | Path,
    protocol: dict[str, Any],
    protocol_fingerprint: str,
) -> dict[str, Any]:
    artifact_obj = json.loads(Path(path).read_text(encoding="utf-8"))
    artifact = cast(dict[str, Any], artifact_obj)
    if not isinstance(artifact, dict):
        raise ValueError("MNSol source audit must be a JSON object.")
    _forbid_value_fields(artifact)
    _require_keys(
        artifact,
        {
            "artifact_type",
            "content_sha256",
            "counts",
            "evaluation_design",
            "panel",
            "protocol_fingerprint",
            "protocol_id",
            "route1_boundary",
            "source_archive",
            "source_table",
        },
        "source_audit",
        optional={"command_provenance"},
    )
    if artifact.get("artifact_type") != "route1-mnsol-multisolvent-source-audit-v1":
        raise ValueError("Unexpected MNSol source-audit artifact type.")
    if artifact.get("content_sha256") != core.artifact_content_sha256(artifact):
        raise ValueError("MNSol source-audit content seal is invalid.")
    if (
        artifact.get("protocol_id") != protocol["protocol_id"]
        or artifact.get("protocol_fingerprint") != protocol_fingerprint
    ):
        raise ValueError("MNSol source audit does not bind the selected protocol.")
    if artifact.get("route1_boundary") != protocol["route1_boundary"]:
        raise ValueError("MNSol source audit changed the Route 1 boundary.")
    if artifact.get("evaluation_design") != protocol["evaluation_design"]:
        raise ValueError("MNSol source audit changed the evaluation design.")

    dataset = protocol["dataset"]
    expected_archive = dataset["archive"]
    expected_table = dataset["table"]
    observed_archive = _require_keys(
        artifact.get("source_archive"),
        {"name", "sha256", "size_bytes", "source_path_recorded"},
        "source_audit.source_archive",
    )
    observed_table = _require_keys(
        artifact.get("source_table"),
        {
            "name",
            "sha256",
            "size_bytes",
            "expected_rows",
            "experimental_value_columns_loaded",
            "metadata_columns_loaded",
        },
        "source_audit.source_table",
    )
    if observed_archive != {
        "name": expected_archive["name"],
        "sha256": expected_archive["sha256"],
        "size_bytes": expected_archive["size_bytes"],
        "source_path_recorded": False,
    }:
        raise ValueError("MNSol source-audit archive identity mismatch.")
    if (
        observed_table.get("name") != expected_table["name"]
        or observed_table.get("sha256") != expected_table["sha256"]
        or observed_table.get("size_bytes") != expected_table["size_bytes"]
        or observed_table.get("expected_rows") != expected_table["expected_rows"]
    ):
        raise ValueError("MNSol source-audit table identity mismatch.")
    if observed_table.get("experimental_value_columns_loaded") != []:
        raise ValueError("MNSol source audit is not label-free.")
    if observed_table.get("metadata_columns_loaded") != list(ALLOWED_ROW_COLUMNS):
        raise ValueError("MNSol source audit decoded unexpected row columns.")
    counts = _require_keys(
        artifact.get("counts"),
        {
            "panel_neutral_absolute_records",
            "panel_solvents",
            "panel_unique_neutral_geometry_pairs",
        },
        "source_audit.counts",
    )
    panel = artifact.get("panel")
    if not isinstance(panel, list):
        raise ValueError("source_audit.panel must be a list.")
    panel_fields = {
        "canonical_name",
        "mnsol_name",
        "all_records",
        "neutral_absolute_records",
        "unique_neutral_geometry_count",
        "rationale",
    }
    for index, entry in enumerate(panel):
        _require_keys(entry, panel_fields, f"source_audit.panel[{index}]")
    expected_panel = [
        {
            "canonical_name": entry["canonical_name"],
            "mnsol_name": entry["mnsol_name"],
            "all_records": entry["expected_all_records"],
            "neutral_absolute_records": entry[
                "expected_neutral_absolute_records"
            ],
            "unique_neutral_geometry_count": entry[
                "expected_neutral_absolute_records"
            ],
            "rationale": entry["rationale"],
        }
        for entry in protocol["panel"]
    ]
    if panel != expected_panel:
        raise ValueError(
            "MNSol source-audit panel does not exactly match the frozen protocol."
        )
    if (
        counts["panel_solvents"] != len(panel)
        or counts["panel_neutral_absolute_records"]
        != sum(int(entry["neutral_absolute_records"]) for entry in panel)
        or counts["panel_unique_neutral_geometry_pairs"]
        != sum(int(entry["unique_neutral_geometry_count"]) for entry in panel)
    ):
        raise ValueError("MNSol source-audit aggregate counts are inconsistent.")
    if "command_provenance" in artifact:
        provenance = _require_keys(
            artifact["command_provenance"],
            {
                "arguments",
                "arguments_sha256",
                "environment_variables",
                "python_executable",
                "script",
                "script_sha256",
            },
            "source_audit.command_provenance",
        )
        arguments = _require_keys(
            provenance["arguments"],
            {"archive_name", "archive_sha256", "output", "protocol"},
            "source_audit.command_provenance.arguments",
        )
        if (
            arguments["archive_name"] != expected_archive["name"]
            or arguments["archive_sha256"] != expected_archive["sha256"]
            or not isinstance(arguments["output"], str)
            or not arguments["output"]
            or not isinstance(arguments["protocol"], str)
            or not arguments["protocol"]
            or provenance["arguments_sha256"]
            != core.sha256_bytes(core.canonical_json_bytes(arguments))
            or provenance["environment_variables"] != {}
            or not isinstance(provenance["python_executable"], str)
            or not provenance["python_executable"]
            or not isinstance(provenance["script"], str)
            or not provenance["script"]
            or not isinstance(provenance["script_sha256"], str)
            or len(provenance["script_sha256"]) != 64
        ):
            raise ValueError("MNSol source-audit command provenance is invalid.")
    return artifact


def _read_identity_rows(
    archive: ZipFile,
    member: ZipInfo,
    expected: dict[str, Any],
) -> list[tuple[int, str, str, str, str]]:
    """Read row ordinals and allowlisted metadata without decoding labels."""
    if member.file_size != expected["size_bytes"]:
        raise ValueError("MNSol table size does not match the frozen protocol.")
    with archive.open(member, "r") as handle:
        payload = handle.read(source_audit.MAX_TABLE_BYTES + 1)
    if len(payload) > source_audit.MAX_TABLE_BYTES:
        raise ValueError("MNSol table exceeds the configured size limit.")
    if core.sha256_bytes(payload) != expected["sha256"]:
        raise ValueError("MNSol table SHA256 does not match the frozen protocol.")

    lines = payload.splitlines()
    if not lines:
        raise ValueError("MNSol table is empty.")
    try:
        fieldnames = lines[0].decode("utf-8-sig").split("\t")
    except UnicodeDecodeError as exc:
        raise ValueError("MNSol table header is not UTF-8 decodable.") from exc
    if len(fieldnames) != len(set(fieldnames)):
        raise ValueError("MNSol table has duplicate header fields.")
    if not source_audit.MNSOL_REQUIRED_COLUMNS <= set(fieldnames):
        raise ValueError("MNSol table does not expose the official required schema.")
    indices = tuple(fieldnames.index(column) for column in ALLOWED_ROW_COLUMNS)

    rows: list[tuple[int, str, str, str, str]] = []
    for ordinal, raw_line in enumerate(lines[1:], start=1):
        raw_columns = raw_line.split(b"\t")
        if len(raw_columns) != len(fieldnames):
            raise ValueError(
                f"MNSol data row {ordinal} has {len(raw_columns)} columns; "
                f"expected {len(fieldnames)}."
            )
        try:
            values = tuple(
                raw_columns[index].decode("utf-8") for index in indices
            )
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"MNSol allowlisted metadata is not UTF-8 at data row {ordinal}."
            ) from exc
        rows.append((ordinal, values[0], values[1], values[2], values[3]))
    if len(rows) != expected["expected_rows"]:
        raise ValueError("MNSol table row count does not match the frozen protocol.")
    return rows


def _geometry_members(archive: ZipFile) -> dict[str, ZipInfo]:
    """Map geometry handles and reject ambiguous duplicate members."""
    members: dict[str, ZipInfo] = {}
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if (
            len(path.parts) < 2
            or path.parts[-2] != "all_solutes"
            or path.suffix != ".xyz"
        ):
            continue
        handle = path.stem
        if handle in members:
            raise ValueError(f"Duplicate MNSol XYZ geometry handle: {handle!r}.")
        members[handle] = member
    if not members:
        raise ValueError("MNSol archive contains no all_solutes XYZ geometries.")
    return members


def _geometry_identity(payload: bytes, handle: str) -> dict[str, Any]:
    """Validate MNSol's Gaussian-style XYZ and return no coordinates."""
    if not payload or len(payload) > MAX_GEOMETRY_BYTES:
        raise ValueError(f"MNSol geometry {handle!r} is empty or too large.")
    try:
        lines = [line.strip() for line in payload.decode("utf-8").splitlines()]
    except UnicodeDecodeError as exc:
        raise ValueError(f"MNSol geometry {handle!r} is not UTF-8.") from exc
    nonempty = [line for line in lines if line]
    if len(nonempty) < 3:
        raise ValueError(f"MNSol geometry {handle!r} is truncated.")
    state = nonempty[1].split()
    if len(state) != 2:
        raise ValueError(f"MNSol geometry {handle!r} lacks charge/multiplicity.")
    try:
        declared_charge, multiplicity = (int(token) for token in state)
    except ValueError as exc:
        raise ValueError(
            f"MNSol geometry {handle!r} has invalid charge/multiplicity."
        ) from exc
    if multiplicity <= 0:
        raise ValueError(f"MNSol geometry {handle!r} has invalid multiplicity.")

    atomic_numbers: list[int] = []
    for atom_index, line in enumerate(nonempty[2:], start=1):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(
                f"MNSol geometry {handle!r} atom {atom_index} is malformed."
            )
        try:
            atomic_number = int(fields[0])
            coordinates = tuple(float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(
                f"MNSol geometry {handle!r} atom {atom_index} is non-numeric."
            ) from exc
        if not 1 <= atomic_number <= 118 or not all(
            math.isfinite(value) for value in coordinates
        ):
            raise ValueError(
                f"MNSol geometry {handle!r} atom {atom_index} is invalid."
            )
        atomic_numbers.append(atomic_number)
    if not atomic_numbers:
        raise ValueError(f"MNSol geometry {handle!r} has no atoms.")
    return {
        "geometry_sha256": core.sha256_bytes(payload),
        "atom_count": len(atomic_numbers),
        "element_sequence_sha256": core.sha256_bytes(
            core.canonical_json_bytes(atomic_numbers)
        ),
        "declared_charge": declared_charge,
        "multiplicity": multiplicity,
    }


def _forbid_value_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in FORBIDDEN_VALUE_FIELDS:
                raise ValueError(f"Forbidden value-bearing field: {key!r}.")
            _forbid_value_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _forbid_value_fields(nested)


def build_manifest(
    *,
    protocol_path: str | Path,
    source_audit_path: str | Path,
    archive_path: str | Path,
    canonical_solvent: str,
) -> dict[str, Any]:
    protocol, protocol_fingerprint = source_audit._load_protocol(protocol_path)
    source_evidence = _load_source_audit(
        source_audit_path, protocol, protocol_fingerprint
    )
    panel_matches = [
        entry
        for entry in protocol["panel"]
        if entry["canonical_name"] == canonical_solvent
    ]
    if len(panel_matches) != 1:
        raise ValueError(
            "Requested solvent must match exactly one predeclared canonical name."
        )
    panel_entry = panel_matches[0]
    mnsol_name = str(panel_entry["mnsol_name"])

    source = Path(archive_path)
    if not source.is_file():
        raise FileNotFoundError(f"MNSol archive is not a file: {source}")
    expected_archive = protocol["dataset"]["archive"]
    if (
        source.stat().st_size != expected_archive["size_bytes"]
        or core.sha256_file(source) != expected_archive["sha256"]
    ):
        raise ValueError("MNSol archive identity does not match the frozen protocol.")

    try:
        with ZipFile(source) as archive:
            table_member = source_audit._table_member(archive)
            geometries = _geometry_members(archive)
            rows = _read_identity_rows(
                archive, table_member, protocol["dataset"]["table"]
            )
            records: list[dict[str, Any]] = []
            seen_handles: set[str] = set()
            seen_identities: set[str] = set()
            for ordinal, raw_handle, raw_charge, raw_solvent, raw_type in rows:
                handle = raw_handle.strip()
                charge = raw_charge.strip()
                solvent = raw_solvent.strip()
                process_type = raw_type.strip()
                if solvent != mnsol_name or charge != "0" or process_type != "abs":
                    continue
                if not handle or handle in seen_handles:
                    raise ValueError(
                        "Selected MNSol rows have an empty or duplicate FileHandle."
                    )
                member = geometries.get(handle)
                if member is None:
                    raise ValueError(
                        f"Selected MNSol row {ordinal} lacks geometry {handle!r}."
                    )
                with archive.open(member, "r") as geometry_file:
                    geometry_payload = geometry_file.read(MAX_GEOMETRY_BYTES + 1)
                geometry = _geometry_identity(geometry_payload, handle)
                if geometry["declared_charge"] != 0:
                    raise ValueError(
                        f"MNSol row/geometry charge mismatch for {handle!r}."
                    )
                source_bound_identity_payload = {
                    "source_table_sha256": protocol["dataset"]["table"]["sha256"],
                    "data_row_ordinal": ordinal,
                    "file_handle": handle,
                    "charge": charge,
                    "solvent": solvent,
                    "type": process_type,
                }
                identity_sha256 = core.sha256_bytes(
                    core.canonical_json_bytes(source_bound_identity_payload)
                )
                label_independent_identity_sha256 = core.sha256_bytes(
                    core.canonical_json_bytes(
                        {
                            "dataset_name": protocol["dataset"]["name"],
                            "dataset_version": protocol["dataset"]["version"],
                            "data_row_ordinal": ordinal,
                            "file_handle": handle,
                            "charge": charge,
                            "solvent": solvent,
                            "type": process_type,
                        }
                    )
                )
                if identity_sha256 in seen_identities:
                    raise ValueError("Selected MNSol record identities are not unique.")
                seen_handles.add(handle)
                seen_identities.add(identity_sha256)
                records.append(
                    {
                        "data_row_ordinal": ordinal,
                        "file_handle": handle,
                        "charge": charge,
                        "solvent_mnsol_name": solvent,
                        "solvent_canonical_name": canonical_solvent,
                        "process_type": process_type,
                        "source_table_bound_record_identity_sha256": (
                            identity_sha256
                        ),
                        "label_independent_record_identity_sha256": (
                            label_independent_identity_sha256
                        ),
                        "geometry_member": member.filename,
                        "geometry_sha256": geometry["geometry_sha256"],
                        "atom_count": geometry["atom_count"],
                        "element_sequence_sha256": geometry[
                            "element_sequence_sha256"
                        ],
                        "multiplicity": geometry["multiplicity"],
                    }
                )
    except BadZipFile as exc:
        raise ValueError("MNSol source is not a readable ZIP archive.") from exc

    records.sort(key=lambda record: int(record["data_row_ordinal"]))
    expected_count = int(panel_entry["expected_neutral_absolute_records"])
    audit_matches = [
        entry
        for entry in source_evidence["panel"]
        if entry["canonical_name"] == canonical_solvent
    ]
    if (
        len(records) != expected_count
        or len(audit_matches) != 1
        or audit_matches[0]["neutral_absolute_records"] != expected_count
    ):
        raise ValueError("Selected MNSol denominator does not match frozen evidence.")

    label_independent_record_set = [
        {
            key: value
            for key, value in record.items()
            if key != "source_table_bound_record_identity_sha256"
        }
        for record in records
    ]
    record_set_sha256 = core.sha256_bytes(
        core.canonical_json_bytes(label_independent_record_set)
    )
    artifact: dict[str, Any] = {
        "artifact_type": "route1-mnsol-label-free-record-manifest-v1",
        "claim_scope": (
            "License-controlled local identity/geometry manifest only; no label, "
            "energy, score, endpoint, standard-state, ranking, or accuracy claim."
        ),
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": protocol_fingerprint,
        "route1_boundary": protocol["route1_boundary"],
        "source_audit": {
            "name": Path(source_audit_path).name,
            "file_sha256": core.sha256_file(source_audit_path),
            "content_sha256": source_evidence["content_sha256"],
            "source_path_recorded": False,
        },
        "source_archive": {
            "name": expected_archive["name"],
            "sha256": expected_archive["sha256"],
            "size_bytes": expected_archive["size_bytes"],
            "source_path_recorded": False,
        },
        "source_table": {
            "name": protocol["dataset"]["table"]["name"],
            "sha256": protocol["dataset"]["table"]["sha256"],
            "size_bytes": protocol["dataset"]["table"]["size_bytes"],
            "expected_rows": protocol["dataset"]["table"]["expected_rows"],
            "metadata_columns_loaded": list(ALLOWED_ROW_COLUMNS),
            "experimental_value_columns_loaded": [],
        },
        "selection": {
            "canonical_solvent": canonical_solvent,
            "mnsol_solvent": mnsol_name,
            "charge": "0",
            "process_type": "abs",
            "expected_complete_denominator": expected_count,
        },
        "containment": {
            "raw_combined_table_bytes_read_for_integrity_and_row_framing": True,
            "experimental_value_columns_decoded": [],
            "experimental_values_interpreted": False,
            "experimental_values_used": False,
            "experimental_values_emitted": False,
            "fit_or_tuning_performed": False,
            "energy_or_endpoint_selected": False,
            "intended_storage": "local_omx_only_not_for_redistribution",
        },
        "record_count": len(records),
        "record_set_sha256": record_set_sha256,
        "records": records,
    }
    _forbid_value_fields(artifact)
    return core.seal_artifact(artifact)


def _local_omx_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    local_root = (REPOSITORY_ROOT / ".omx").resolve()
    try:
        relative = resolved.relative_to(local_root)
    except ValueError as exc:
        raise ValueError(
            "MNSol row-level manifests must be written under repository .omx/."
        ) from exc
    if not relative.parts:
        raise ValueError("MNSol manifest output must name a file under .omx/.")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--canonical-solvent", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output_path = _local_omx_output_path(args.output)
    artifact = build_manifest(
        protocol_path=args.protocol,
        source_audit_path=args.source_audit,
        archive_path=args.archive,
        canonical_solvent=args.canonical_solvent,
    )
    payload = {key: value for key, value in artifact.items() if key != "content_sha256"}
    payload["command_provenance"] = core.command_provenance(
        __file__,
        {
            "protocol": _repo_reference(args.protocol),
            "source_audit": args.source_audit.name,
            "source_audit_path_recorded": False,
            "archive_name": args.archive.name,
            "archive_sha256": core.sha256_file(args.archive),
            "archive_path_recorded": False,
            "canonical_solvent": args.canonical_solvent,
            "output": _repo_reference(output_path),
        },
        repository_root=REPOSITORY_ROOT,
    )
    core.write_json_atomic(output_path, core.seal_artifact(payload))


if __name__ == "__main__":
    main()
