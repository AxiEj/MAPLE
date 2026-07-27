from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT / "docs/implicit-solvation/benchmarks/" "route2-mnsol-pilot-selection-v1.json"
)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def test_mnsol_pilot_selection_is_preregistered_and_redistribution_safe():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    fingerprint = artifact.pop("selection_fingerprint")

    assert artifact["artifact"] == "route2-mnsol-pilot-selection-v1"
    assert artifact["selection_status"] == "preregistered-before-model-run"
    assert artifact["dataset"]["table_sha256"] == (
        "6dba4397764d1ca665c5dac653b9963bd64f784c353311a72c15e42897b90156"
    )
    assert artifact["dataset"]["normalized_bundle_sha256"] == (
        "6465a65a024cd06872cb9812381184ed6e9b1a528adfe12be9d5d43b4aec75d8"
    )
    policy = artifact["selection_policy"]
    assert policy["experimental_value_used_for_selection"] is False
    assert policy["model_output_used_for_selection"] is False
    assert policy["max_atom_count"] == 20
    assert policy["partition_preference"] == [
        "confirmation",
        "development",
    ]
    assert policy["require_distinct_geometry_handles"] is True

    selected = artifact["selected_records"]
    assert len(selected) == 10
    assert len({record["opaque_record_id"] for record in selected}) == 10
    assert [record["canonical_solvent"] for record in selected] == [
        "water",
        "ethanol",
        "acetonitrile",
        "dimethylsulfoxide",
        "dimethylformamide",
        "tetrahydrofuran",
        "chloroform",
        "dichloromethane",
        "toluene",
        "hexane",
    ]
    assert all(record["atom_count"] <= 20 for record in selected)
    assert artifact["redistribution_guard"] == {
        "raw_rows_emitted": False,
        "entry_numbers_emitted": False,
        "geometry_handles_emitted": False,
        "solute_names_emitted": False,
        "formulas_emitted": False,
        "coordinates_emitted": False,
        "experimental_values_emitted": False,
    }
    assert "reports no AIMNet2" in artifact["claim_boundary"]
    assert fingerprint == hashlib.sha256(_canonical_json_bytes(artifact)).hexdigest()
