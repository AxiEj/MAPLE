from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from zipfile import ZipFile

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import mnsol_dataset as mnsol
import mnsol_pilot


def _row(
    entry: int,
    handle: str,
    solvent: str,
    *,
    charge: int = 0,
    process_type: str = "abs",
    subset: str = "[g]",
) -> list[str]:
    fields = ["0"] * len(mnsol.MNSOL_V2012_HEADER)
    fields[:12] = [
        str(entry),
        handle,
        f"solute-{handle}",
        "H4C1",
        subset,
        str(charge),
        "1",
        "1",
        "1",
        solvent,
        "-1.25",
        process_type,
    ]
    fields[12:20] = ["10.0", "1.3", "0.1", "0.2", "30.0", "0.0", "0.0", "0.04"]
    fields[20:] = ["0.0"] * (len(fields) - 20)
    fields[-1] = "12.5"
    return fields


def _table(rows: list[list[str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(mnsol.MNSOL_V2012_HEADER)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _geometry(handle: str, *, charge: int = 0, multiplicity: int = 1) -> bytes:
    return (
        f"{handle} H4C1 methane m062x_mg3s_geom\n\n"
        f"{charge} {multiplicity}\n"
        "6 0.0 0.0 0.0\n"
        "1 0.6 0.6 0.6\n"
        "1 -0.6 -0.6 0.6\n"
        "1 -0.6 0.6 -0.6\n"
        "1 0.6 -0.6 -0.6\n"
    ).encode("utf-8")


def _disconnected_geometry(handle: str) -> bytes:
    return _geometry(handle).replace(
        b"1 0.6 -0.6 -0.6\n",
        b"1 20.0 20.0 20.0\n",
    )


def _fixture(tmp_path: Path, rows: list[list[str]]) -> tuple[Path, Path]:
    table = _table(rows)
    source = tmp_path / "MNSolDatabase_v2012.zip"
    handles = {row[1] for row in rows}
    geometry_payloads = {
        handle: _geometry(
            handle,
            charge=int(next(row[5] for row in rows if row[1] == handle)),
        )
        for handle in handles
    }
    with ZipFile(source, "w") as archive:
        archive.writestr(
            "MNSolDatabase_v2012/MNSol_alldata.txt",
            table,
        )
        for handle, payload in geometry_payloads.items():
            archive.writestr(
                f"MNSolDatabase_v2012/all_solutes/{handle}.xyz",
                payload,
            )

    protocol = json.loads(
        (BENCHMARK_DIR / "route2-mnsol-protocol-v1.json").read_text(encoding="utf-8")
    )
    absolute_solvents = {row[9] for row in rows if row[11] == "abs"}
    transfer_labels = {row[9] for row in rows if row[11] == "rel"}
    protocol["dataset"]["table_sha256"] = hashlib.sha256(table).hexdigest()
    protocol["dataset"]["normalized_bundle_sha256"] = mnsol._normalized_bundle_sha256(
        table, geometry_payloads
    )
    protocol["dataset"]["expected_record_count"] = len(rows)
    protocol["dataset"]["expected_unique_solutes"] = len(handles)
    protocol["dataset"]["expected_absolute_solvents"] = len(absolute_solvents)
    protocol["dataset"]["expected_transfer_labels"] = len(transfer_labels)
    protocol["domain"]["molecular_mass_da"] = [1.0, 500.0]
    protocol["solvent_panel"] = [
        {
            "canonical_name": f"solvent-{index}",
            "mnsol_name": solvent,
            "neutral_absolute_validation": True,
            "expected_all_records": sum(row[9] == solvent for row in rows),
            "expected_neutral_absolute_records": sum(
                row[9] == solvent and row[11] == "abs" and row[5] == "0" for row in rows
            ),
            "rationale": "fixture",
        }
        for index, solvent in enumerate(sorted(absolute_solvents), start=1)
    ]
    if len(absolute_solvents) != 10:
        raise AssertionError("Fixture helper requires ten absolute solvents.")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    return source, protocol_path


def _refresh_protocol_hashes(source: Path, protocol_path: Path) -> None:
    table, geometries, _, _ = mnsol._read_source(source)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["dataset"]["table_sha256"] = hashlib.sha256(table).hexdigest()
    protocol["dataset"]["normalized_bundle_sha256"] = mnsol._normalized_bundle_sha256(
        table, geometries
    )
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")


def _ten_solvent_rows() -> list[list[str]]:
    return [
        _row(index, f"solute{index:02d}", f"solvent{index:02d}")
        for index in range(1, 11)
    ]


def test_mnsol_loader_reconciles_table_geometries_and_aggregate_scope(tmp_path):
    source, protocol_path = _fixture(tmp_path, _ten_solvent_rows())
    protocol = mnsol.load_mnsol_protocol(protocol_path)
    dataset = mnsol.load_mnsol_v2012(source, protocol)
    manifest = mnsol.build_coverage_manifest(dataset, protocol)

    assert len(dataset.records) == 10
    assert len(dataset.geometries) == 10
    assert manifest["dataset"]["record_count"] == 10
    assert manifest["integrity"] == {
        "table_geometry_identity_reconciled": True,
        "table_geometry_charge_reconciled": True,
        "table_geometry_formula_reconciled": True,
        "geometry_level_tag": "m062x_mg3s_geom",
        "solvent_descriptor_inconsistencies": [],
        "raw_rows_emitted": False,
    }
    assert manifest["initial_neutral_absolute_scope"]["eligible_record_count"] == 10
    assert (
        manifest["initial_neutral_absolute_scope"]["standard_state"]
        == "1M-ideal-gas-to-1M-ideal-solution"
    )
    assert manifest["initial_neutral_absolute_scope"]["temperature_k"] == 298.0
    assert (
        manifest["initial_neutral_absolute_scope"]["required_covalent_component_count"]
        == 1
    )
    assert (
        sum(manifest["partition"]["record_counts"].values())
        == manifest["initial_neutral_absolute_scope"]["eligible_record_count"]
    )
    assert manifest["partition"]["same_solute_cross_solvent_leakage"] is False
    eligible = mnsol.eligible_mnsol_records(dataset, protocol)
    assert len(eligible) == 10
    assert {item.partition for item in eligible} <= {
        "development",
        "confirmation",
    }


def test_mnsol_loader_rejects_header_drift_before_reading_rows(tmp_path):
    rows = _ten_solvent_rows()
    source, protocol_path = _fixture(tmp_path, rows)
    with ZipFile(source, "r") as archive:
        geometry_payloads = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.endswith(".xyz")
        }
    bad_header = list(mnsol.MNSOL_V2012_HEADER)
    bad_header[10] = "ChangedDeltaG"
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(bad_header)
    writer.writerows(rows)
    bad_table = buffer.getvalue().encode("utf-8")
    with ZipFile(source, "w") as archive:
        archive.writestr("MNSolDatabase_v2012/MNSol_alldata.txt", bad_table)
        for name, payload in geometry_payloads.items():
            archive.writestr(name, payload)
    _refresh_protocol_hashes(source, protocol_path)

    protocol = mnsol.load_mnsol_protocol(protocol_path)
    with pytest.raises(ValueError, match="header does not match"):
        mnsol.load_mnsol_v2012(source, protocol)


def test_mnsol_loader_rejects_missing_geometry(tmp_path):
    rows = _ten_solvent_rows()
    source, protocol_path = _fixture(tmp_path, rows)
    with ZipFile(source, "r") as archive:
        members = {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("solute10.xyz")
        }
    with ZipFile(source, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)

    _refresh_protocol_hashes(source, protocol_path)
    protocol = mnsol.load_mnsol_protocol(protocol_path)
    with pytest.raises(ValueError, match="identities do not reconcile"):
        mnsol.load_mnsol_v2012(source, protocol)


def test_mnsol_loader_rejects_geometry_bundle_tampering(tmp_path):
    source, protocol_path = _fixture(tmp_path, _ten_solvent_rows())
    with ZipFile(source, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    target = "MNSolDatabase_v2012/all_solutes/solute10.xyz"
    members[target] = members[target].replace(
        b"1 0.6 -0.6 -0.6\n",
        b"1 0.7 -0.6 -0.6\n",
    )
    with ZipFile(source, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)

    protocol = mnsol.load_mnsol_protocol(protocol_path)
    with pytest.raises(ValueError, match="normalized bundle hash mismatch"):
        mnsol.load_mnsol_v2012(source, protocol)


def test_mnsol_manifest_excludes_disconnected_geometry(tmp_path):
    source, protocol_path = _fixture(tmp_path, _ten_solvent_rows())
    with ZipFile(source, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    target = "MNSolDatabase_v2012/all_solutes/solute10.xyz"
    members[target] = _disconnected_geometry("solute10")
    with ZipFile(source, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    _refresh_protocol_hashes(source, protocol_path)

    protocol = mnsol.load_mnsol_protocol(protocol_path)
    dataset = mnsol.load_mnsol_v2012(source, protocol)
    manifest = mnsol.build_coverage_manifest(dataset, protocol)

    assert manifest["initial_neutral_absolute_scope"]["eligible_record_count"] == 9
    assert manifest["initial_neutral_absolute_scope"]["aggregate_exclusion_counts"] == {
        "disconnected_geometry": 1
    }


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("dataset", "temperature_k", 310.0, "temperature_k"),
        ("dataset", "standard_state", "1atm-to-1M", "standard_state"),
        ("domain", "molecular_charge", 1, "molecular_charge"),
        ("domain", "multiplicity", 3, "multiplicity"),
        ("domain", "process_type", "rel", "process_type"),
    ],
)
def test_mnsol_protocol_rejects_scientific_contract_drift(
    tmp_path, section, field, value, message
):
    payload = json.loads(
        (BENCHMARK_DIR / "route2-mnsol-protocol-v1.json").read_text(encoding="utf-8")
    )
    payload[section][field] = value
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        mnsol.load_mnsol_protocol(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("strategy", "pooled-random-split"),
        ("pooled_random_split_is_not_sufficient", False),
    ],
)
def test_mnsol_protocol_rejects_solvent_generalization_drift(tmp_path, field, value):
    payload = json.loads(
        (BENCHMARK_DIR / "route2-mnsol-protocol-v1.json").read_text(encoding="utf-8")
    )
    payload["partition"]["solvent_generalization"][field] = value
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="MNSol"):
        mnsol.load_mnsol_protocol(path)


def test_mnsol_filehandle_partition_prevents_cross_solvent_leakage(tmp_path):
    rows = _ten_solvent_rows()
    rows[1][1] = rows[0][1]
    rows[1][2] = rows[0][2]
    rows[1][3] = rows[0][3]
    source, protocol_path = _fixture(tmp_path, rows)
    protocol = mnsol.load_mnsol_protocol(protocol_path)
    dataset = mnsol.load_mnsol_v2012(source, protocol)
    manifest = mnsol.build_coverage_manifest(dataset, protocol)

    shared_assignments = {
        mnsol._partition_for_handle(record.geometry_handle, protocol)
        for record in dataset.records
        if record.geometry_handle == "solute01"
    }
    serialized = json.dumps(manifest, sort_keys=True)

    assert len(shared_assignments) == 1
    assert manifest["partition"]["same_solute_cross_solvent_leakage"] is False
    assert "solute01" not in serialized
    assert "solute-solute01" not in serialized
    assert "-1.25" not in serialized
    assert dataset.records[0].raw_row_sha256 not in serialized


def test_mnsol_zip_reader_enforces_uncompressed_size_limit(tmp_path, monkeypatch):
    source, protocol_path = _fixture(tmp_path, _ten_solvent_rows())
    protocol = mnsol.load_mnsol_protocol(protocol_path)
    monkeypatch.setattr(mnsol, "MNSOL_MAX_ZIP_UNCOMPRESSED_BYTES", 1)

    with pytest.raises(ValueError, match="uncompressed-size safety limit"):
        mnsol.load_mnsol_v2012(source, protocol)


def test_mnsol_directory_reader_enforces_geometry_size_limit(tmp_path, monkeypatch):
    source, protocol_path = _fixture(tmp_path, _ten_solvent_rows())
    extracted = tmp_path / "extracted"
    with ZipFile(source, "r") as archive:
        archive.extractall(extracted)
    protocol = mnsol.load_mnsol_protocol(protocol_path)
    monkeypatch.setattr(mnsol, "MNSOL_MAX_GEOMETRY_BYTES", 1)

    with pytest.raises(ValueError, match="geometry .* safety limit"):
        mnsol.load_mnsol_v2012(extracted, protocol)


def test_mnsol_directory_reader_bounds_geometry_name_collection(tmp_path, monkeypatch):
    source, protocol_path = _fixture(tmp_path, _ten_solvent_rows())
    extracted = tmp_path / "extracted"
    with ZipFile(source, "r") as archive:
        archive.extractall(extracted)
    protocol = mnsol.load_mnsol_protocol(protocol_path)
    monkeypatch.setattr(mnsol, "MNSOL_MAX_GEOMETRY_COUNT", 1)

    with pytest.raises(ValueError, match="too many geometries"):
        mnsol.load_mnsol_v2012(extracted, protocol)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation is POSIX-only.")
def test_mnsol_directory_reader_rejects_non_regular_geometry(tmp_path):
    source, protocol_path = _fixture(tmp_path, _ten_solvent_rows())
    extracted = tmp_path / "extracted"
    with ZipFile(source, "r") as archive:
        archive.extractall(extracted)
    geometry = extracted / "MNSolDatabase_v2012" / "all_solutes" / "solute10.xyz"
    geometry.unlink()
    os.mkfifo(geometry)
    protocol = mnsol.load_mnsol_protocol(protocol_path)

    with pytest.raises(ValueError, match="must be a regular file"):
        mnsol.load_mnsol_v2012(extracted, protocol)


def test_mnsol_zip_reader_enforces_compression_ratio_limit(tmp_path, monkeypatch):
    source, protocol_path = _fixture(tmp_path, _ten_solvent_rows())
    protocol = mnsol.load_mnsol_protocol(protocol_path)
    monkeypatch.setattr(mnsol, "MNSOL_MAX_ZIP_COMPRESSION_RATIO", 0.5)

    with pytest.raises(ValueError, match="compression-ratio safety limit"):
        mnsol.load_mnsol_v2012(source, protocol)


def test_tracked_mnsol_protocol_has_ten_neutral_solvents_and_keeps_methanol_honest():
    protocol = mnsol.load_mnsol_protocol(
        BENCHMARK_DIR / "route2-mnsol-protocol-v1.json"
    )
    neutral_panel = [
        item for item in protocol.panel if item.neutral_absolute_validation
    ]
    methanol = next(
        item for item in protocol.panel if item.canonical_name == "methanol"
    )

    assert len(neutral_panel) == 10
    assert methanol.expected_all_records == 80
    assert methanol.expected_neutral_absolute_records == 0
    assert methanol.neutral_absolute_validation is False


def test_mnsol_protocol_rejects_string_boolean(tmp_path):
    payload = json.loads(
        (BENCHMARK_DIR / "route2-mnsol-protocol-v1.json").read_text(encoding="utf-8")
    )
    payload["solvent_panel"][0]["neutral_absolute_validation"] = "false"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be booleans"):
        mnsol.load_mnsol_protocol(path)


def test_mnsol_dataset_module_is_provider_independent():
    source = (BENCHMARK_DIR / "mnsol_dataset.py").read_text(encoding="utf-8")

    assert "extra_correction" not in source
    assert "ddpcm_smd" not in source
    assert "pcmsolver" not in source.lower()


def test_mnsol_pilot_selection_is_unique_and_emits_no_row_data(tmp_path):
    source, protocol_path = _fixture(tmp_path, _ten_solvent_rows())
    protocol = mnsol.load_mnsol_protocol(protocol_path)
    dataset = mnsol.load_mnsol_v2012(source, protocol)

    selection = mnsol_pilot.select_mnsol_pilot(dataset, protocol)
    manifest = mnsol_pilot.build_mnsol_pilot_selection_manifest(
        dataset,
        protocol,
    )
    serialized = json.dumps(manifest, sort_keys=True)

    assert len(selection) == 10
    assert (
        len({item.eligible_record.record.geometry_handle for item in selection}) == 10
    )
    assert all(
        len(item.eligible_record.geometry.atomic_numbers)
        <= mnsol_pilot.PILOT_MAX_ATOM_COUNT
        for item in selection
    )
    assert (
        manifest["selection_policy"]["experimental_value_used_for_selection"] is False
    )
    assert manifest["selection_policy"]["model_output_used_for_selection"] is False
    assert manifest["redistribution_guard"] == {
        "raw_rows_emitted": False,
        "entry_numbers_emitted": False,
        "geometry_handles_emitted": False,
        "solute_names_emitted": False,
        "formulas_emitted": False,
        "coordinates_emitted": False,
        "experimental_values_emitted": False,
    }
    assert "solute01" not in serialized
    assert "solute-solute01" not in serialized
    assert "-1.25" not in serialized
    assert (
        mnsol_pilot.validate_frozen_mnsol_pilot_selection(
            manifest,
            dataset,
            protocol,
        )
        == selection
    )


def test_mnsol_pilot_choice_does_not_change_with_experimental_values(tmp_path):
    rows_a = _ten_solvent_rows()
    rows_b = [row.copy() for row in rows_a]
    for index, row in enumerate(rows_b):
        row[10] = str(1000.0 + index)

    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    source_a, protocol_a_path = _fixture(first, rows_a)
    source_b, protocol_b_path = _fixture(second, rows_b)
    protocol_a = mnsol.load_mnsol_protocol(protocol_a_path)
    protocol_b = mnsol.load_mnsol_protocol(protocol_b_path)
    dataset_a = mnsol.load_mnsol_v2012(source_a, protocol_a)
    dataset_b = mnsol.load_mnsol_v2012(source_b, protocol_b)

    selected_a = mnsol_pilot.select_mnsol_pilot(dataset_a, protocol_a)
    selected_b = mnsol_pilot.select_mnsol_pilot(dataset_b, protocol_b)

    assert [item.eligible_record.record.geometry_handle for item in selected_a] == [
        item.eligible_record.record.geometry_handle for item in selected_b
    ]
    assert [item.selection_score_sha256 for item in selected_a] == [
        item.selection_score_sha256 for item in selected_b
    ]
