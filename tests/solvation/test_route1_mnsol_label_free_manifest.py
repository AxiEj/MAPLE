from __future__ import annotations

import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_STORED, ZipFile

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
SOURCE_PROTOCOL_PATH = BENCHMARK_DIR / "route1_multisolvent_source_protocol.json"

BUILDER_SPEC = importlib.util.spec_from_file_location(
    "route1_mnsol_label_free_manifest",
    BENCHMARK_DIR / "build_route1_mnsol_label_free_manifest.py",
)
assert BUILDER_SPEC is not None
builder = importlib.util.module_from_spec(BUILDER_SPEC)
assert BUILDER_SPEC.loader is not None
BUILDER_SPEC.loader.exec_module(builder)

AUDIT_SPEC = importlib.util.spec_from_file_location(
    "route1_multisolvent_source_audit_for_manifest_test",
    BENCHMARK_DIR / "route1_multisolvent_source_audit.py",
)
assert AUDIT_SPEC is not None
audit = importlib.util.module_from_spec(AUDIT_SPEC)
assert AUDIT_SPEC.loader is not None
AUDIT_SPEC.loader.exec_module(audit)


def _protocol() -> dict[str, Any]:
    protocol = copy.deepcopy(
        json.loads(SOURCE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    )
    for entry in protocol["panel"]:
        entry["expected_all_records"] = 7
        entry["expected_neutral_absolute_records"] = 7
    return protocol


def _xyz(atomic_number: int = 6, coordinate: float = 0.0) -> bytes:
    return (
        "synthetic test geometry\n\n"
        "0 1\n"
        f"{atomic_number:3d} {coordinate:.8f} 0.00000000 0.00000000\n"
    ).encode()


def _write_sources(
    tmp_path: Path,
    *,
    label: bytes = b"-3.25",
    geometry: bytes | None = None,
    duplicate_ethanol_geometry: bool = False,
) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    protocol = _protocol()
    headers = [
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
    ]
    table = io.BytesIO()
    header_buffer = io.StringIO(newline="")
    csv.writer(header_buffer, delimiter="\t", lineterminator="\n").writerow(headers)
    table.write(header_buffer.getvalue().encode())
    records: list[tuple[int, dict[str, Any], str, int]] = []
    ordinal = 0
    for solvent_index, entry in enumerate(protocol["panel"]):
        for sample_index in range(7):
            ordinal += 1
            handle = f"solute_{solvent_index}_{sample_index}"
            prefix = (
                f"{ordinal}\t{handle}\t0\t{entry['mnsol_name']}\t".encode()
            )
            suffix = b"\tabs\t1.0\t1.0\t0.0\t0.0\t0.0\n"
            table.write(prefix + label + suffix)
            records.append((ordinal, entry, handle, sample_index))
    table_bytes = table.getvalue()

    archive_path = tmp_path / "MNSolDatabase_v2012.zip"
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("MNSolDatabase_v2012/MNSol_alldata.txt", table_bytes)
        for _ordinal, entry, handle, sample_index in records:
            payload = (
                geometry
                if (
                    entry["canonical_name"] == "ethanol"
                    and sample_index == 0
                    and geometry is not None
                )
                else _xyz()
            )
            archive.writestr(
                f"MNSolDatabase_v2012/all_solutes/{handle}.xyz",
                payload,
            )
        if duplicate_ethanol_geometry:
            archive.writestr(
                "other/all_solutes/solute_1_0.xyz",
                geometry or _xyz(),
            )

    protocol["dataset"]["table"].update(
        {
            "size_bytes": len(table_bytes),
            "sha256": audit.core.sha256_bytes(table_bytes),
            "expected_rows": len(records),
        }
    )
    protocol["dataset"]["archive"].update(
        {
            "size_bytes": archive_path.stat().st_size,
            "sha256": audit.core.sha256_file(archive_path),
        }
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    loaded, fingerprint = audit._load_protocol(protocol_path)
    source_artifact = audit.audit_archive(loaded, fingerprint, archive_path)
    source_audit_path = tmp_path / "source-audit.json"
    audit.core.write_json_atomic(source_audit_path, source_artifact)
    return protocol_path, source_audit_path, archive_path


def _build(paths: tuple[Path, Path, Path]) -> dict[str, Any]:
    protocol_path, source_audit_path, archive_path = paths
    return builder.build_manifest(
        protocol_path=protocol_path,
        source_audit_path=source_audit_path,
        archive_path=archive_path,
        canonical_solvent="ethanol",
    )


def test_manifest_is_one_solvent_complete_and_label_free(tmp_path: Path) -> None:
    artifact = _build(_write_sources(tmp_path))

    assert artifact["content_sha256"] == audit.core.artifact_content_sha256(
        artifact
    )
    assert artifact["record_count"] == 7
    assert artifact["selection"] == {
        "canonical_solvent": "ethanol",
        "mnsol_solvent": "ethanol",
        "charge": "0",
        "process_type": "abs",
        "expected_complete_denominator": 7,
    }
    assert artifact["source_table"]["experimental_value_columns_loaded"] == []
    assert artifact["source_table"]["metadata_columns_loaded"] == [
        "FileHandle",
        "Charge",
        "Solvent",
        "type",
    ]
    assert artifact["source_archive"]["source_path_recorded"] is False
    assert artifact["source_audit"]["source_path_recorded"] is False
    assert (
        artifact["containment"][
            "raw_combined_table_bytes_read_for_integrity_and_row_framing"
        ]
        is True
    )
    assert artifact["containment"]["experimental_value_columns_decoded"] == []
    assert artifact["containment"]["experimental_values_interpreted"] is False
    assert artifact["containment"]["experimental_values_used"] is False
    assert artifact["containment"]["intended_storage"].startswith("local_omx")
    record = artifact["records"][0]
    assert record["data_row_ordinal"] == 8
    assert record["atom_count"] == 1
    assert len(record["source_table_bound_record_identity_sha256"]) == 64
    assert len(record["label_independent_record_identity_sha256"]) == 64
    assert len(record["element_sequence_sha256"]) == 64
    serialized = json.dumps(artifact)
    assert "-3.25" not in serialized
    assert "DeltaGsolv" not in serialized


def test_record_set_is_label_invariant_but_artifact_binds_source(
    tmp_path: Path,
) -> None:
    first = _build(_write_sources(tmp_path / "first", label=b"-3.25"))
    second = _build(_write_sources(tmp_path / "second", label=b"99.75"))

    assert [
        record["label_independent_record_identity_sha256"]
        for record in first["records"]
    ] == [
        record["label_independent_record_identity_sha256"]
        for record in second["records"]
    ]
    assert [
        record["source_table_bound_record_identity_sha256"]
        for record in first["records"]
    ] != [
        record["source_table_bound_record_identity_sha256"]
        for record in second["records"]
    ]
    assert first["record_set_sha256"] == second["record_set_sha256"]
    assert first["source_table"]["sha256"] != second["source_table"]["sha256"]
    assert first["content_sha256"] != second["content_sha256"]


def test_geometry_change_changes_only_geometry_bound_identity(tmp_path: Path) -> None:
    first = _build(_write_sources(tmp_path / "first", geometry=_xyz(6, 0.0)))
    second = _build(_write_sources(tmp_path / "second", geometry=_xyz(8, 1.0)))

    left = first["records"][0]
    right = second["records"][0]
    assert (
        left["source_table_bound_record_identity_sha256"]
        == right["source_table_bound_record_identity_sha256"]
    )
    assert (
        left["label_independent_record_identity_sha256"]
        == right["label_independent_record_identity_sha256"]
    )
    assert left["geometry_sha256"] != right["geometry_sha256"]
    assert left["element_sequence_sha256"] != right["element_sequence_sha256"]
    assert first["record_set_sha256"] != second["record_set_sha256"]


def test_manifest_does_not_decode_invalid_label_bytes(tmp_path: Path) -> None:
    artifact = _build(_write_sources(tmp_path, label=b"\xff\xfe"))

    assert artifact["record_count"] == 7
    assert artifact["source_table"]["experimental_value_columns_loaded"] == []


def test_manifest_rejects_duplicate_geometry_handles(tmp_path: Path) -> None:
    paths = _write_sources(tmp_path, duplicate_ethanol_geometry=True)

    with pytest.raises(ValueError, match="Duplicate MNSol XYZ geometry handle"):
        _build(paths)


def test_manifest_rejects_malformed_geometry(tmp_path: Path) -> None:
    paths = _write_sources(tmp_path, geometry=b"not a geometry\n")

    with pytest.raises(ValueError, match="truncated"):
        _build(paths)


def test_manifest_rejects_tampered_source_audit_seal(tmp_path: Path) -> None:
    protocol_path, source_audit_path, archive_path = _write_sources(tmp_path)
    source_artifact = json.loads(source_audit_path.read_text())
    source_artifact["counts"]["panel_neutral_absolute_records"] = 0
    source_audit_path.write_text(json.dumps(source_artifact))

    with pytest.raises(ValueError, match="content seal"):
        _build((protocol_path, source_audit_path, archive_path))


def test_manifest_rejects_resealed_source_audit_label_field(
    tmp_path: Path,
) -> None:
    protocol_path, source_audit_path, archive_path = _write_sources(tmp_path)
    source_artifact = json.loads(source_audit_path.read_text())
    source_artifact["DeltaGsolv"] = -1.0
    source_artifact["content_sha256"] = audit.core.artifact_content_sha256(
        source_artifact
    )
    source_audit_path.write_text(json.dumps(source_artifact))

    with pytest.raises(ValueError, match="Forbidden value-bearing field"):
        _build((protocol_path, source_audit_path, archive_path))


def test_manifest_rejects_resealed_incomplete_source_audit_panel(
    tmp_path: Path,
) -> None:
    protocol_path, source_audit_path, archive_path = _write_sources(tmp_path)
    source_artifact = json.loads(source_audit_path.read_text())
    removed = source_artifact["panel"].pop()
    source_artifact["counts"]["panel_solvents"] -= 1
    source_artifact["counts"]["panel_neutral_absolute_records"] -= removed[
        "neutral_absolute_records"
    ]
    source_artifact["counts"]["panel_unique_neutral_geometry_pairs"] -= removed[
        "unique_neutral_geometry_count"
    ]
    source_artifact["content_sha256"] = audit.core.artifact_content_sha256(
        source_artifact
    )
    source_audit_path.write_text(json.dumps(source_artifact))

    with pytest.raises(ValueError, match="exactly match the frozen protocol"):
        _build((protocol_path, source_audit_path, archive_path))


def test_manifest_rejects_resealed_nested_provenance_label(
    tmp_path: Path,
) -> None:
    protocol_path, source_audit_path, archive_path = _write_sources(tmp_path)
    source_artifact = json.loads(source_audit_path.read_text())
    source_artifact["command_provenance"] = {
        "script": "route1_multisolvent_source_audit.py",
        "script_sha256": "a" * 64,
        "python_executable": "python",
        "arguments": {
            "archive_name": "MNSolDatabase_v2012.zip",
            "archive_sha256": source_artifact["source_archive"]["sha256"],
            "output": "source-audit.json",
            "protocol": "protocol.json",
            "experimental_values": [-1.0],
        },
        "arguments_sha256": "b" * 64,
        "environment_variables": {},
    }
    source_artifact["content_sha256"] = audit.core.artifact_content_sha256(
        source_artifact
    )
    source_audit_path.write_text(json.dumps(source_artifact))

    with pytest.raises(ValueError, match="schema mismatch"):
        _build((protocol_path, source_audit_path, archive_path))


def test_output_path_is_restricted_to_repository_omx(tmp_path: Path) -> None:
    accepted = REPOSITORY_ROOT / ".omx/benchmarks/manifest.json"
    assert builder._local_omx_output_path(accepted) == accepted.resolve()

    with pytest.raises(ValueError, match="repository .omx"):
        builder._local_omx_output_path(tmp_path / "manifest.json")


def test_forbidden_value_field_fails_closed() -> None:
    with pytest.raises(ValueError, match="Forbidden value-bearing field"):
        builder._forbid_value_fields({"DeltaGsolv": -1.0})
