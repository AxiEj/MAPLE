from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release import (
    AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS,
    AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_ARTIFACT_SCHEMA_VERSION,
    AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_CONTRACT_VERSION,
    AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_ARTIFACT_SCHEMA_VERSION,
    canonical_json_sha256,
    summarize_aimnet2_geometry_mediated_pes_panel,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "aimnet2-geometry-mediated-pes-panel-v2-cd8769d4"
)
EXECUTION_HEAD = "cd8769d4cdad439f9aa861429da4069535092c4a"
EXECUTION_TREE = "3a14e042cfc94cfcce892e8ae24ee3af838c18e5"
AGGREGATE_MEASUREMENT_SHA256 = (
    "b0796da42ebd729f0bd33cc3d98ea3a0f0b3711d3d2d3819086b0df5e860e3b4"
)
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
FAILED_MOLECULE_IDS = [
    "methanol",
    "ethanol",
    "acetone",
    "acetonitrile",
    "benzene",
    "methane",
    "trans-butane",
    "dimethyl-ether",
    "formic-acid",
    "acetic-acid",
    "acetaldehyde",
    "acetamide",
    "ethylamine",
    "pyridine",
    "nitromethane",
]
MEASURED_KEYS = (
    "contract_version",
    "panel_asset_sha256",
    "protocol",
    "identity",
    "records",
    "summary",
)


def _checksums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        result[name] = digest
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_compressed(name: str) -> tuple[dict[str, object], bytes]:
    compressed = EVIDENCE / f"{name}.gz"
    raw = gzip.decompress(compressed.read_bytes())
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload, raw


def test_full_v2_bundle_checksums_bind_compressed_and_raw_json():
    compressed = _checksums(EVIDENCE / "SHA256SUMS")
    expected_files = {
        path.name for path in EVIDENCE.iterdir() if path.name != "SHA256SUMS"
    }
    assert set(compressed) == expected_files
    for name, digest in compressed.items():
        assert _sha256((EVIDENCE / name).read_bytes()) == digest

    raw = _checksums(EVIDENCE / "RAW_SHA256SUMS")
    expected_raw = {
        path.name.removesuffix(".gz") for path in EVIDENCE.glob("*.json.gz")
    }
    assert set(raw) == expected_raw
    for name, digest in raw.items():
        assert _sha256(gzip.decompress((EVIDENCE / f"{name}.gz").read_bytes())) == (
            digest
        )


def _validate_run(prefix: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    panel, panel_raw = _load_compressed(f"{prefix}-panel.json")
    assert panel["schema_version"] == (
        AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_ARTIFACT_SCHEMA_VERSION
    )
    assert panel["contract_version"] == (
        AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_CONTRACT_VERSION
    )
    assert panel["status"] == "diagnostic-panel-failed-not-admitted"
    assert panel["execution_git_head"] == EXECUTION_HEAD
    assert panel["execution_git_tree"] == EXECUTION_TREE
    assert panel["capabilities"] == NO_CAPABILITIES
    assert panel["aggregate_measurement_sha256"] == AGGREGATE_MEASUREMENT_SHA256

    raw_shards: list[dict[str, object]] = []
    source_ledgers = []
    input_records = {
        int(record["molecule_index"]): record for record in panel["input_artifacts"]
    }
    assert tuple(input_records) == tuple(range(17))
    for index in range(17):
        shard, shard_raw = _load_compressed(f"{prefix}-shard-{index:02d}.json")
        assert shard["schema_version"] == (
            AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_ARTIFACT_SCHEMA_VERSION
        )
        assert shard["execution_git_head"] == EXECUTION_HEAD
        assert shard["execution_git_tree"] == EXECUTION_TREE
        assert shard["working_tree_clean"] is True
        assert shard["capabilities"] == NO_CAPABILITIES
        assert shard["summary"]["molecule_index"] == index
        assert shard["summary"]["molecule_id"] == (
            AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS[index]
        )
        measured = {key: shard[key] for key in MEASURED_KEYS}
        assert canonical_json_sha256(measured) == shard["measurement_sha256"]
        input_record = input_records[index]
        assert input_record["sha256"] == _sha256(shard_raw)
        assert input_record["measurement_sha256"] == shard["measurement_sha256"]
        source_ledgers.append(shard["source_files_sha256"])
        raw_shards.append(
            {
                "molecule_index": index,
                "records": shard["records"],
            }
        )

    assert all(ledger == source_ledgers[0] for ledger in source_ledgers)
    assert len(source_ledgers[0]) == 133
    recomputed = summarize_aimnet2_geometry_mediated_pes_panel(raw_shards)
    assert recomputed == panel["panel_summary"]
    assert canonical_json_sha256(raw_shards) == AGGREGATE_MEASUREMENT_SHA256
    assert panel["panel_summary"]["failed_molecule_ids"] == FAILED_MOLECULE_IDS
    assert panel["panel_summary"]["diagnostic_gates_passed"] is False
    assert (
        _sha256(panel_raw)
        == _checksums(EVIDENCE / "RAW_SHA256SUMS")[f"{prefix}-panel.json"]
    )
    return panel, raw_shards


def test_full_v2_panel_is_recomputed_and_cold_process_replay_is_exact():
    primary, primary_shards = _validate_run("primary")
    replay, replay_shards = _validate_run("replay")
    assert replay_shards == primary_shards
    assert replay["panel_summary"] == primary["panel_summary"]
    assert [record["measurement_sha256"] for record in replay["input_artifacts"]] == [
        record["measurement_sha256"] for record in primary["input_artifacts"]
    ]
    first_shard, _ = _load_compressed("primary-shard-00.json")
    assert_source_files_match_execution_commit(ROOT, first_shard)


def test_full_v2_negative_boundaries_remain_fail_closed():
    panel, _ = _load_compressed("primary-panel.json")
    summary = panel["panel_summary"]
    assert summary["molecule_count"] == 17
    assert summary["geometry_count"] == 51
    assert summary["directional_record_count"] == 153
    assert summary["directional_sample_count"] == 459
    assert summary["gates"] == {
        "all_continuum_event_guards": False,
        "all_deterministic_replays": True,
        "all_directional_force_fd": False,
        "all_neighbor_cutoff_guards": True,
        "all_reciprocity_metric_charge_gauge_audits": True,
        "all_sphere_tangency_guards": False,
        "all_stationarity_audits": True,
        "all_stencils_same_stratum": True,
    }
    passed = [
        shard["molecule_id"]
        for shard in summary["shard_summaries"]
        if shard["diagnostic_gates_passed"]
    ]
    assert passed == ["water", "hydrogen-peroxide"]
    assert summary["capabilities"] == NO_CAPABILITIES
    assert summary["opt_admitted"] is False
    assert summary["freq_ts_irc_admitted"] is False
    assert summary["md_admitted"] is False
