from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release import (
    AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_PANEL_ARTIFACT_SCHEMA_VERSION,
    AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_PANEL_CONTRACT_VERSION,
    AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_SHARD_ARTIFACT_SCHEMA_VERSION,
    AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS,
    canonical_json_sha256,
    summarize_aimnet2_geometry_mediated_pes_panel,
    summarize_aimnet2_geometry_mediated_pes_shard,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "aimnet2-frozen-charge-water-harmonic-ddpcm-pes-panel-smoothed-97efb08e"
)
RUNTIME_V2_EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "aimnet2-frozen-charge-water-harmonic-ddpcm-pes-panel-27a388a2"
)
EXECUTION_HEAD = "97efb08ecbeff1a30a8876f3f83f01b40e10ad0e"
EXECUTION_TREE = "af0cc177b4965fbed4c967f7bb595ac698ee698e"
AGGREGATE_MEASUREMENT_SHA256 = (
    "a219082dbe88097923fd18b39fff83a613f1a7a02d57d7e126df2e549708c42b"
)
CONTINUUM_KIND = "harmonic-ddpcm-water"
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
FAILED_MOLECULE_IDS = [
    "methanol",
    "methane",
    "dimethyl-ether",
    "acetic-acid",
    "ethylamine",
]
PASSED_MOLECULE_IDS = [
    "water",
    "ethanol",
    "acetone",
    "acetonitrile",
    "benzene",
    "trans-butane",
    "formic-acid",
    "acetaldehyde",
    "acetamide",
    "pyridine",
    "nitromethane",
    "hydrogen-peroxide",
]
MEASURED_KEYS = (
    "contract_version",
    "panel_asset_sha256",
    "protocol",
    "identity",
    "records",
    "summary",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checksums(path: Path) -> dict[str, str]:
    return {
        name: digest
        for digest, name in (
            line.split(maxsplit=1)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    }


def _load(evidence: Path, name: str) -> tuple[dict[str, object], bytes]:
    raw = gzip.decompress((evidence / f"{name}.gz").read_bytes())
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload, raw


def test_smoothed_frozen_charge_water_ddpcm_bundle_checksums_bind_all_json():
    compressed = _checksums(EVIDENCE / "SHA256SUMS")
    expected = {path.name for path in EVIDENCE.iterdir() if path.name != "SHA256SUMS"}
    assert set(compressed) == expected
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
    panel, _ = _load(EVIDENCE, f"{prefix}-panel.json")
    assert panel["schema_version"] == (
        AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_PANEL_ARTIFACT_SCHEMA_VERSION
    )
    assert panel["contract_version"] == (
        AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_PANEL_CONTRACT_VERSION
    )
    assert panel["execution_git_head"] == EXECUTION_HEAD
    assert panel["execution_git_tree"] == EXECUTION_TREE
    assert panel["status"] == "diagnostic-panel-failed-not-admitted"
    assert panel["capabilities"] == NO_CAPABILITIES
    assert panel["aggregate_measurement_sha256"] == AGGREGATE_MEASUREMENT_SHA256

    inputs = {int(item["molecule_index"]): item for item in panel["input_artifacts"]}
    assert tuple(inputs) == tuple(range(17))
    raw_shards = []
    source_ledgers = []
    for index in range(17):
        shard, raw = _load(EVIDENCE, f"{prefix}-shard-{index:02d}.json")
        assert shard["schema_version"] == (
            AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_SHARD_ARTIFACT_SCHEMA_VERSION
        )
        assert shard["execution_git_head"] == EXECUTION_HEAD
        assert shard["execution_git_tree"] == EXECUTION_TREE
        assert shard["working_tree_clean"] is True
        assert shard["continuum_kind"] == CONTINUUM_KIND
        assert shard["capabilities"] == NO_CAPABILITIES
        assert shard["summary"]["molecule_index"] == index
        assert shard["summary"]["molecule_id"] == (
            AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS[index]
        )
        runtime = shard["identity"]["model_runtime"]
        assert runtime["runtime_kind"] == "aimnet-reconstructed-float64-runtime-v3"
        assert runtime["ordinary_forward_role"] == (
            "per-geometry energy-charge-gradient-vjp parity oracle only"
        )
        assert runtime["first_order_ordinary_decomposed_parity_tolerances"] == {
            "charge_absolute_e": 1.0e-10,
            "charge_vjp_absolute_eV_per_A": 1.0e-10,
            "energy_absolute_eV": 1.0e-6,
            "intrinsic_gradient_absolute_eV_per_A": 1.0e-7,
        }
        measured = {key: shard[key] for key in MEASURED_KEYS}
        assert canonical_json_sha256(measured) == shard["measurement_sha256"]
        assert inputs[index]["sha256"] == _sha256(raw)
        assert inputs[index]["measurement_sha256"] == shard["measurement_sha256"]
        assert (
            summarize_aimnet2_geometry_mediated_pes_shard(
                molecule_index=index,
                records=shard["records"],
                continuum_kind=CONTINUUM_KIND,
            )
            == shard["summary"]
        )
        source_ledgers.append(shard["source_files_sha256"])
        raw_shards.append({"molecule_index": index, "records": shard["records"]})

    assert all(ledger == source_ledgers[0] for ledger in source_ledgers)
    assert len(source_ledgers[0]) == 135
    assert canonical_json_sha256(raw_shards) == AGGREGATE_MEASUREMENT_SHA256
    assert (
        summarize_aimnet2_geometry_mediated_pes_panel(
            raw_shards, continuum_kind=CONTINUUM_KIND
        )
        == panel["panel_summary"]
    )
    return panel, raw_shards


def test_smoothed_panel_recomputes_and_clean_process_replay_is_exact():
    primary, primary_shards = _validate_run("primary")
    replay, replay_shards = _validate_run("replay")
    assert replay_shards == primary_shards
    assert replay["panel_summary"] == primary["panel_summary"]
    first, _ = _load(EVIDENCE, "primary-shard-00.json")
    assert_source_files_match_execution_commit(ROOT, first)


def test_smoothed_panel_improves_force_convergence_without_hiding_topology_failures():
    panel, _ = _load(EVIDENCE, "primary-panel.json")
    summary = panel["panel_summary"]
    assert summary["failed_molecule_ids"] == FAILED_MOLECULE_IDS
    assert [
        shard["molecule_id"]
        for shard in summary["shard_summaries"]
        if shard["diagnostic_gates_passed"]
    ] == PASSED_MOLECULE_IDS
    assert all(
        all(
            direction["convergence"]["gate_passed"]
            for geometry in shard["geometry_records"]
            for direction in geometry["directional_force_fd"].values()
        )
        for shard in summary["shard_summaries"]
    )
    assert summary["maximum_directional_absolute_error_eV_per_A"] == pytest.approx(
        5.148876215019804e-4, rel=0.0, abs=1.0e-18
    )
    assert summary["gates"] == {
        "all_continuum_event_guards": False,
        "all_deterministic_replays": True,
        "all_directional_force_fd": False,
        "all_energy_cotangent_kkt_audits": True,
        "all_neighbor_cutoff_guards": True,
        "all_primal_responses_excluded_from_provider_field": True,
        "all_reciprocity_metric_charge_gauge_audits": True,
        "all_sphere_tangency_guards": False,
        "all_stationarity_audits": True,
        "all_stencils_same_stratum": True,
    }
    assert summary["capabilities"] == NO_CAPABILITIES
    assert summary["opt_admitted"] is False
    assert summary["freq_ts_irc_admitted"] is False
    assert summary["md_admitted"] is False

    old_panel, _ = _load(RUNTIME_V2_EVIDENCE, "primary-panel.json")
    assert old_panel["continuum_kind"] == panel["continuum_kind"]
    assert old_panel["contract_version"] == panel["contract_version"]
    assert old_panel["checkpoint"]["sha256"] == panel["checkpoint"]["sha256"]
    assert old_panel["panel_summary"]["molecule_ids"] == summary["molecule_ids"]
    assert len(old_panel["panel_summary"]["failed_molecule_ids"]) == 15
    assert len(summary["failed_molecule_ids"]) == 5
