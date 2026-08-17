from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import aggregate_mnsol_aimnet2_full_matrix as aggregate  # noqa: E402
import run_mnsol_aimnet2_multisolvent_pilot as worker  # noqa: E402
import run_mnsol_aimnet2_partition as scheduler  # noqa: E402


def _method(total: float, *, experimental: float) -> dict[str, object]:
    error = total - experimental
    return {
        "polarization_energy_hartree": -0.01,
        "polarization_energy_kcal_mol": total - 0.4,
        "smd_cds_energy_kcal_mol": 0.4,
        "total_solvation_kcal_mol": total,
        "signed_error_kcal_mol": error,
        "absolute_error_kcal_mol": abs(error),
        "half_coupling_identity_error_ev": 1.0e-14,
        "runtime_provenance": {},
        "timing_seconds": {"build": 0.1, "solve": 0.2},
    }


def _record(index: int, *, experimental: float = -4.0) -> dict[str, object]:
    return {
        "selection_index": index,
        "canonical_solvent": "water",
        "partition": "development",
        "opaque_record_id": f"opaque-{index}",
        "entry_number": index + 1,
        "geometry_handle": f"geometry-{index}",
        "geometry_sha256": "a" * 64,
        "prior_pilot_geometry_overlap": False,
        "experimental_delta_g_kcal_mol": experimental,
        "atom_count": 5 + index,
        "smd_cds_energy_kcal_mol": 0.4,
        "methods": {
            "ddpcm": _method(-4.5 + index, experimental=experimental),
            "ddcosmo": _method(-4.1 + index, experimental=experimental),
        },
    }


def test_complete_frozen_partition_manifests_cover_653_rows():
    manifests = {
        partition: json.loads(
            (BENCHMARK_DIR / f"route2-mnsol-{partition}-selection-v1.json").read_text(
                encoding="utf-8"
            )
        )
        for partition in aggregate.PARTITIONS
    }
    assert manifests["development"]["record_count"] == 505
    assert manifests["confirmation"]["record_count"] == 148
    assert sum(item["record_count"] for item in manifests.values()) == 653
    assert all(
        item["ordering"]["excludes_records"] is False for item in manifests.values()
    )
    assert all(
        item["ordering"]["uses_experimental_values"] is False
        for item in manifests.values()
    )
    assert all(
        item["ordering"]["uses_model_outputs"] is False for item in manifests.values()
    )


def test_scheduler_accepts_complete_selection_and_requires_source_bound_pair():
    selection = scheduler._selection_contract(
        BENCHMARK_DIR / "route2-mnsol-development-selection-v1.json"
    )
    assert selection["partition"] == "development"
    assert selection["record_count"] == 505

    base = {
        "artifact": scheduler.EXPECTED_ARTIFACT,
        "schema_version": 1,
        "do_not_commit": True,
        "complete_panel": False,
        "run_kind": scheduler.EXPECTED_RUN_KIND,
        "execution_git_head": "b" * 40,
        "selection_fingerprint": selection["selection_fingerprint"],
        "checkpoint": {"sha256": "c" * 64},
    }
    private = {**base, "records": [{"selection_index": 7}]}
    summary = {**base, "selection_indices": [7]}
    assert scheduler._matching_complete(
        private,
        summary,
        index=7,
        head="b" * 40,
        selection=selection,
        checkpoint_sha256="c" * 64,
    )
    summary["selection_indices"] = [8]
    assert not scheduler._matching_complete(
        private,
        summary,
        index=7,
        head="b" * 40,
        selection=selection,
        checkpoint_sha256="c" * 64,
    )


def test_full_matrix_metrics_include_tail_and_cluster_balanced_views():
    records = [_record(0), _record(1, experimental=-5.0)]
    block = aggregate._metric_block(records)
    assert block["record_count"] == 2
    assert block["unique_geometry_count"] == 2
    assert block["methods"]["ddpcm"]["mean_absolute_error_kcal_mol"] == pytest.approx(
        1.0
    )
    assert block["methods"]["ddpcm"]["p95_absolute_error_kcal_mol"] == pytest.approx(
        1.45
    )
    clustered = aggregate._cluster_balanced_metrics(records, "ddpcm")
    assert clustered["unique_geometry_count"] == 2
    assert aggregate._atom_bin(records[0]) == "01-05"
    assert aggregate._atom_bin(records[1]) == "06-10"


def test_shard_validator_recomputes_single_record_public_metrics():
    record = _record(0)
    selection_manifest = {
        "protocol_fingerprint": "d" * 64,
        "selection_fingerprint": "e" * 64,
    }
    base = {
        "artifact": aggregate.SOURCE_ARTIFACT,
        "schema_version": 1,
        "complete_panel": False,
        "do_not_commit": True,
        "run_kind": aggregate.SOURCE_RUN_KIND,
        "execution_git_head": "f" * 40,
        "protocol_fingerprint": "d" * 64,
        "selection_fingerprint": "e" * 64,
        "checkpoint": {"sha256": "1" * 64},
    }
    private = {**base, "records": [record]}
    summary = {
        **base,
        "selection_indices": [0],
        "aggregate_metrics": {
            method: worker._metrics([record], method) for method in aggregate.METHODS
        },
        "paired_method_comparison": worker._paired_method_comparison([record]),
    }
    selected = SimpleNamespace(
        canonical_solvent="water",
        opaque_record_id="opaque-0",
        prior_pilot_geometry_overlap=False,
        eligible_record=SimpleNamespace(
            partition="development",
            record=SimpleNamespace(
                entry_number=1,
                geometry_handle="geometry-0",
                delta_g_kcal_mol=-4.0,
            ),
            geometry=SimpleNamespace(sha256="a" * 64),
        ),
    )
    assert (
        aggregate._validate_shard(
            private,
            summary,
            selected=selected,
            index=0,
            selection_manifest=selection_manifest,
            checkpoint_sha256="1" * 64,
        )
        is record
    )
    summary["aggregate_metrics"]["ddpcm"]["mean_absolute_error_kcal_mol"] += 0.1
    with pytest.raises(ValueError, match="single-row metrics drifted"):
        aggregate._validate_shard(
            private,
            summary,
            selected=selected,
            index=0,
            selection_manifest=selection_manifest,
            checkpoint_sha256="1" * 64,
        )


def test_public_aggregate_guard_rejects_row_level_keys():
    aggregate._assert_public_safe({"aggregate_metrics": {"record_count": 653}})
    with pytest.raises(ValueError, match="leaks private keys"):
        aggregate._assert_public_safe({"nested": {"entry_number": 12}})
