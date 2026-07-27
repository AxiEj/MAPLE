from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import mnsol_partition  # noqa: E402

ARTIFACT = BENCHMARK_DIR / "route2-mnsol-confirmation-selection-v1.json"


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


def test_confirmation_selection_is_complete_and_redistribution_safe():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    fingerprint = artifact.pop("selection_fingerprint")

    assert artifact["artifact"] == "route2-mnsol-partition-selection-v1"
    assert artifact["partition"] == "confirmation"
    assert artifact["selection_status"] == "frozen-before-partition-run"
    assert artifact["record_count"] == 148
    assert artifact["unique_geometry_count"] == 83
    assert artifact["prior_pilot_overlap"]["record_count"] == 46
    assert artifact["prior_pilot_overlap"]["unique_geometry_count"] == 8
    assert (
        artifact["prior_pilot_overlap"]["no_prior_pilot_geometry_overlap_record_count"]
        == 102
    )
    assert artifact["ordering"]["uses_experimental_values"] is False
    assert artifact["ordering"]["uses_model_outputs"] is False

    selected = artifact["selected_records"]
    assert len(selected) == 148
    assert len({record["opaque_record_id"] for record in selected}) == 148
    assert [record["selection_index"] for record in selected] == list(range(148))
    assert all(record["partition"] == "confirmation" for record in selected)
    assert all(
        record["prior_pilot_geometry_overlap"] is False for record in selected[:102]
    )
    assert all(
        record["prior_pilot_geometry_overlap"] is True for record in selected[102:]
    )
    forbidden = {
        "entry_number",
        "geometry_handle",
        "solute_name",
        "formula",
        "coordinates",
        "experimental_delta_g_kcal_mol",
    }
    assert all(forbidden.isdisjoint(record) for record in selected)
    assert artifact["redistribution_guard"] == {
        "raw_rows_emitted": False,
        "entry_numbers_emitted": False,
        "geometry_handles_emitted": False,
        "solute_names_emitted": False,
        "formulas_emitted": False,
        "coordinates_emitted": False,
        "experimental_values_emitted": False,
    }
    assert fingerprint == hashlib.sha256(_canonical_json_bytes(artifact)).hexdigest()


def test_partition_selection_requires_an_explicit_bounded_index():
    records = tuple(f"record-{index}" for index in range(4))

    assert mnsol_partition.indexed_partition_record(records, 2) == [(2, "record-2")]
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        mnsol_partition.indexed_partition_record(records, 4)
    with pytest.raises(ValueError, match="explicit --record-index"):
        mnsol_partition.indexed_partition_record(records, None)


def test_partition_order_score_does_not_depend_on_experimental_value():
    first = SimpleNamespace(
        canonical_solvent="water",
        record=SimpleNamespace(
            geometry_handle="geometry-17",
            entry_number=42,
            delta_g_kcal_mol=-1.0,
            raw_row_sha256="first-raw-row",
        ),
    )
    changed_experiment = SimpleNamespace(
        canonical_solvent="water",
        record=SimpleNamespace(
            geometry_handle="geometry-17",
            entry_number=42,
            delta_g_kcal_mol=-99.0,
            raw_row_sha256="changed-raw-row",
        ),
    )

    assert mnsol_partition._selection_score(
        first,
        partition="confirmation",
    ) == mnsol_partition._selection_score(
        changed_experiment,
        partition="confirmation",
    )
