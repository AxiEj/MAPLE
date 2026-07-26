from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
PROTOCOL_PATH = BENCHMARK_DIR / "solprop_external_water_audit_protocol.json"
MANIFEST_PATH = BENCHMARK_DIR / "solprop_external_water_source_manifest.json"
ARTIFACT_PATH = BENCHMARK_DIR / "route1-solprop-external-water-audit-2026-07-26.json"
SPEC = importlib.util.spec_from_file_location(
    "run_solprop_external_water_audit",
    BENCHMARK_DIR / "run_solprop_external_water_audit.py",
)
assert SPEC is not None
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def test_protocol_pins_official_archive_and_label_free_source_table():
    protocol, fingerprint = audit._load_protocol(PROTOCOL_PATH)

    assert len(fingerprint) == 64
    assert protocol["dataset"]["expected_archive"] == {
        "name": "SolProp_v1.2.zip",
        "sha256": "f66bb046bc1d5b8471d36436e485d0cb9d95fb7111aa2db4deb85fdbbea6b766",
        "md5": "677d6f107df6fbd0e6d3963e32f06bdb",
        "size_bytes": 268574239,
    }
    assert protocol["dataset"]["source_table"] == {
        "name": "CombiSolv-Exp.xlsx",
        "relative_archive_path": "SolProp_v1.2/Data/CombiSolv-Exp.xlsx",
        "sha256": "a68bb5ba4f120846f8250cb79f18afd6439cebba4e7d418f2e54ba38919d91b7",
        "size_bytes": 466840,
        "expected_rows": 8780,
    }
    assert set(protocol["columns"].values()) == {
        "Source_all",
        "smiles_solvent",
        "smiles_solute",
        "inchi_solute",
    }
    assert not any("dgsolv" in value.lower() for value in protocol["columns"].values())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), 1.9),
        (("dataset", "expected_archive", "size_bytes"), 268_574_239.9),
        (("dataset", "source_table", "expected_rows"), 8_780.9),
        (("dataset", "source_table", "size_bytes"), "466840"),
        (("analysis", "size_bins_heavy_atoms", "small_max"), 8.0),
        (("counts", "candidate_rows"), 561.9),
        (("counts", "candidate_rows"), "561"),
        (("counts", "candidate_rows"), True),
    ],
)
def test_protocol_rejects_non_integer_numeric_fields(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    target = protocol
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    tampered = tmp_path / "protocol.json"
    tampered.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(TypeError, match="JSON integer"):
        audit._load_protocol(tampered)


def test_freesolv_source_detection_and_connectivity_dedup_are_conservative():
    assert audit._source_has_freesolv("[\" 'FreeSolv (entry number=287)'\"]")
    assert audit._source_has_freesolv("[\" 'Abraham Paper (A1)'\", \" 'FreeSolv'\"]")
    assert not audit._source_has_freesolv("[\" 'CompSol Binary'\"]")

    database = json.loads(
        (
            REPOSITORY_ROOT
            / ".omx/benchmarks/neutral-water-freesolv-route1-20260723"
            / "dataset/database.json"
        ).read_text(encoding="utf-8")
    )
    mapping = audit._build_reference_connectivity(database)
    duplicate_groups = [
        compound_ids for compound_ids in mapping.values() if len(compound_ids) > 1
    ]

    assert len(database) == 642
    assert len(mapping) == 639
    assert len(duplicate_groups) == 3
    assert all(len(compound_ids) == 2 for compound_ids in duplicate_groups)


def test_committed_manifest_is_label_free_unique_and_self_hashed():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["content_sha256"] == audit.core.artifact_content_sha256(manifest)
    assert manifest["counts"]["total_rows"] == 8780
    assert manifest["counts"]["water_rows"] == 1153
    assert manifest["counts"]["overlap_connectivity_rows"] == 560
    assert manifest["counts"]["source_has_freesolv_rows"] == 560
    assert manifest["counts"]["source_has_freesolv_mismatch_rows"] == 0
    assert manifest["counts"]["nonoverlap_rows"] == 593
    assert manifest["counts"]["candidate_rows"] == 561
    assert manifest["counts"]["filter_exclusions"] == 32
    assert (
        manifest["counts"]["candidate_small_rows"],
        manifest["counts"]["candidate_medium_rows"],
        manifest["counts"]["candidate_large_rows"],
    ) == (114, 304, 143)
    assert manifest["source_dataset"]["loaded_columns"] == [
        "Source_all",
        "smiles_solvent",
        "smiles_solute",
        "inchi_solute",
    ]
    assert manifest["source_dataset"]["experimental_value_columns_loaded"] == []

    records = manifest["records"]
    assert len(records) == 561
    assert len({record["candidate_id"] for record in records}) == 561
    assert len({record["connectivity_block"] for record in records}) == 561
    assert all(record["source_has_freesolv"] is False for record in records)
    assert all(
        set(record["elements"]) <= {"H", "C", "N", "O", "F", "S", "Cl"}
        and "C" in record["elements"]
        and record["radical_electrons"] == 0
        for record in records
    )
    forbidden = ("dgsolv", "reported_mean", "reported_std", "experimental_value")
    assert all(
        not any(token in key.lower() for token in forbidden)
        for record in records
        for key in record
    )
    assert all(
        record["row_sha256"]
        == audit.core.sha256_bytes(
            audit.core.canonical_json_bytes(
                {key: value for key, value in record.items() if key != "row_sha256"}
            )
        )
        for record in records
    )


def test_summary_artifact_binds_manifest_without_copying_candidate_records():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    assert artifact["content_sha256"] == audit.core.artifact_content_sha256(artifact)
    assert artifact["record_count"] == 561
    assert "records" not in artifact
    assert artifact["source_manifest_sha256"] == audit.core.sha256_file(MANIFEST_PATH)
    assert artifact["source_manifest_content_sha256"] == manifest["content_sha256"]
    assert artifact["source_table"]["experimental_value_columns_loaded"] == []
    assert artifact["source_table"]["sha256"] == (
        "a68bb5ba4f120846f8250cb79f18afd6439cebba4e7d418f2e54ba38919d91b7"
    )
