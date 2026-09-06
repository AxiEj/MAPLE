from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
SPEC = importlib.util.spec_from_file_location(
    "build_route1_exploratory_panel",
    BENCHMARK_DIR / "build_route1_exploratory_panel.py",
)
assert SPEC is not None
panel_builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(panel_builder)


def _record(solvent: str, index: int) -> dict[str, Any]:
    handle = f"{solvent.replace(' ', '_')}_{index:02d}"
    digest = panel_builder.core.sha256_bytes
    return {
        "data_row_ordinal": index + 1,
        "file_handle": handle,
        "charge": "0",
        "solvent_mnsol_name": solvent.replace(" ", ""),
        "solvent_canonical_name": solvent,
        "process_type": "abs",
        "source_table_bound_record_identity_sha256": digest(
            f"source:{solvent}:{index}".encode()
        ),
        "label_independent_record_identity_sha256": digest(
            f"label-free:{solvent}:{index}".encode()
        ),
        "geometry_member": f"all_solutes/{handle}.xyz",
        "geometry_sha256": digest(f"geometry:{solvent}:{index}".encode()),
        "atom_count": 1,
        "element_sequence_sha256": digest(b"[6]"),
        "multiplicity": 1,
    }


def _manifest(solvent: str, count: int) -> dict[str, Any]:
    records = [_record(solvent, index) for index in range(count)]
    artifact: dict[str, Any] = {
        "artifact_type": "route1-mnsol-label-free-record-manifest-v1",
        "claim_scope": "local identity manifest only",
        "protocol_id": "synthetic-protocol",
        "protocol_fingerprint": "a" * 64,
        "route1_boundary": {"charge": "neutral", "process": "absolute"},
        "source_audit": {
            "name": "audit.json",
            "file_sha256": "b" * 64,
            "content_sha256": "c" * 64,
            "source_path_recorded": False,
        },
        "source_archive": {
            "name": "MNSolDatabase_v2012.zip",
            "sha256": "d" * 64,
            "size_bytes": 1,
            "source_path_recorded": False,
        },
        "source_table": {
            "name": "MNSol_alldata.txt",
            "sha256": "e" * 64,
            "size_bytes": 1,
            "expected_rows": 1,
            "metadata_columns_loaded": ["FileHandle", "Charge", "Solvent", "type"],
            "experimental_value_columns_loaded": [],
        },
        "selection": {
            "canonical_solvent": solvent,
            "mnsol_solvent": solvent.replace(" ", ""),
            "charge": "0",
            "process_type": "abs",
            "expected_complete_denominator": count,
        },
        "containment": {
            "experimental_value_columns_decoded": [],
            "experimental_values_interpreted": False,
        },
        "record_count": count,
        "records": records,
    }
    return _reseal_manifest(artifact)


def _reseal_manifest(artifact: dict[str, Any]) -> dict[str, Any]:
    records = artifact["records"]
    label_independent = [
        {
            key: value
            for key, value in record.items()
            if key != "source_table_bound_record_identity_sha256"
        }
        for record in records
    ]
    artifact["record_set_sha256"] = panel_builder.core.sha256_bytes(
        panel_builder.core.canonical_json_bytes(label_independent)
    )
    return panel_builder.core.seal_artifact(artifact)


def test_selection_is_stable_hash_order_and_source_order_invariant() -> None:
    ethanol = _manifest("ethanol", 25)
    water = _manifest("water", 23)
    reordered_ethanol = copy.deepcopy(ethanol)
    reordered_ethanol["records"].reverse()
    _reseal_manifest(reordered_ethanol)

    first = panel_builder.assemble_panel(
        [water, ethanol],
        requested_per_solvent=20,
        canonical_solvent_order=["water", "ethanol"],
    )
    second = panel_builder.assemble_panel(
        [reordered_ethanol, water],
        requested_per_solvent=20,
        canonical_solvent_order=["water", "ethanol"],
    )

    assert [entry["canonical_solvent"] for entry in first["solvents"]] == [
        "water",
        "ethanol",
    ]
    for left, right in zip(first["solvents"], second["solvents"], strict=True):
        assert left["records"] == right["records"]
        keys = [
            (record["label_independent_record_identity_sha256"], record["file_handle"])
            for record in left["records"]
        ]
        assert keys == sorted(keys)


def test_counts_report_full_and_shortfall_without_filler() -> None:
    artifact = panel_builder.assemble_panel(
        [_manifest("water", 21), _manifest("ethanol", 7)],
        requested_per_solvent=20,
        canonical_solvent_order=["water", "ethanol"],
    )

    assert artifact["counts"] == {
        "solvent_count": 2,
        "requested_count": 40,
        "available_count": 28,
        "selected_count": 27,
        "shortfall_count": 13,
        "full_solvent_count": 1,
        "shortfall_solvent_count": 1,
    }
    assert [
        {
            key: entry[key]
            for key in (
                "requested_count",
                "available_count",
                "selected_count",
                "shortfall_count",
            )
        }
        for entry in artifact["solvents"]
    ] == [
        {
            "requested_count": 20,
            "available_count": 21,
            "selected_count": 20,
            "shortfall_count": 0,
        },
        {
            "requested_count": 20,
            "available_count": 7,
            "selected_count": 7,
            "shortfall_count": 13,
        },
    ]


@pytest.mark.parametrize("invalid_count", [0, -1, True, 2.5])
def test_requested_count_must_be_a_positive_integer(invalid_count: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        panel_builder.assemble_panel(
            [_manifest("water", 3)], requested_per_solvent=invalid_count
        )


def test_manifest_count_and_both_seals_are_validated() -> None:
    malformed_count = _manifest("water", 3)
    malformed_count["record_count"] = 2
    malformed_count = panel_builder.core.seal_artifact(malformed_count)
    with pytest.raises(ValueError, match="record count"):
        panel_builder.assemble_panel([malformed_count])

    tampered_content = _manifest("water", 3)
    tampered_content["records"][0]["atom_count"] = 2
    with pytest.raises(ValueError, match="content seal"):
        panel_builder.assemble_panel([tampered_content])

    tampered_record_set = _manifest("water", 3)
    tampered_record_set["record_set_sha256"] = "f" * 64
    panel_builder.core.seal_artifact(tampered_record_set)
    with pytest.raises(ValueError, match="record-set seal"):
        panel_builder.assemble_panel([tampered_record_set])


def test_common_source_validation_includes_first_input_when_order_is_custom() -> None:
    ethanol = _manifest("ethanol", 3)
    water = _manifest("water", 3)
    ethanol["source_archive"]["sha256"] = "f" * 64
    panel_builder.core.seal_artifact(ethanol)

    with pytest.raises(ValueError, match="one source archive"):
        panel_builder.assemble_panel(
            [ethanol, water], canonical_solvent_order=["water", "ethanol"]
        )


@pytest.mark.parametrize("duplicate_field", ["file_handle", "geometry_sha256"])
def test_duplicate_solutes_are_rejected(duplicate_field: str) -> None:
    manifest = _manifest("water", 3)
    manifest["records"][1][duplicate_field] = manifest["records"][0][duplicate_field]
    _reseal_manifest(manifest)

    with pytest.raises(ValueError, match="duplicate solutes"):
        panel_builder.assemble_panel([manifest])


def test_output_is_label_free_and_has_no_accuracy_threshold_or_energy_result() -> None:
    artifact = panel_builder.assemble_panel([_manifest("water", 3)])
    serialized = json.dumps(artifact, sort_keys=True).lower()
    keys: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            keys.update(str(key).lower() for key in value)
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    collect(artifact)
    assert "deltagsolv" not in serialized
    assert not any("threshold" in key for key in keys)
    assert not any(key.endswith("energy_kcal_mol") for key in keys)
    assert artifact["containment"]["experimental_value_columns_decoded"] == []
    assert "no energy" in artifact["claim_scope"]
    assert "accuracy-completion claim" in artifact["claim_scope"]


def test_output_directory_is_restricted_to_repository_omx(tmp_path: Path) -> None:
    accepted = REPOSITORY_ROOT / ".omx/benchmarks/route1-panel"
    assert panel_builder._local_output_directory(accepted) == accepted.resolve()

    with pytest.raises(ValueError, match="repository .omx"):
        panel_builder._local_output_directory(tmp_path / "route1-panel")
