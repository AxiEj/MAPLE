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
    AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_SHARD_CONTRACT_VERSION,
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
    / "aimnet2-frozen-charge-water-harmonic-ddpcm-pes-panel-27a388a2"
)
EXECUTION_HEAD = "27a388a23ea6cb8c54e96a36938b81eae486c45b"
EXECUTION_TREE = "7c67b5148c3d42853d3f0e1b10efc32b7e1c3d7d"
CHECKPOINT_SHA256 = "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d"
AGGREGATE_MEASUREMENT_SHA256 = (
    "b53dacb5d14c46d621b1900c8734e396ec939b8398a10a28f2cecb450ed16d72"
)
CONTINUUM_KIND = "harmonic-ddpcm-water"
SCALAR_ID = (
    "route2-candidate-aimnet2-frozen-charge-water-"
    "smoothharmonicgalerkin-ddpcm-electrostatic-v1"
)
PROFILE_ID = (
    "route2-profile-candidate-aimnet2-frozen-charge-water-"
    "smoothharmonicgalerkin-ddpcm-electrostatic-v1"
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


def test_frozen_charge_water_ddpcm_bundle_checksums_bind_compressed_and_raw_json():
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
        AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_PANEL_ARTIFACT_SCHEMA_VERSION
    )
    assert panel["contract_version"] == (
        AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_PANEL_CONTRACT_VERSION
    )
    assert panel["status"] == "diagnostic-panel-failed-not-admitted"
    assert panel["continuum_kind"] == CONTINUUM_KIND
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
            AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_SHARD_ARTIFACT_SCHEMA_VERSION
        )
        assert shard["contract_version"] == (
            AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_SHARD_CONTRACT_VERSION
        )
        assert shard["artifact_kind"] == (
            "disabled-aimnet2-reconstructed-float64-frozen-charge-water-"
            "smooth-harmonic-ddpcm-pes-shard"
        )
        assert shard["continuum_kind"] == CONTINUUM_KIND
        assert shard["execution_git_head"] == EXECUTION_HEAD
        assert shard["execution_git_tree"] == EXECUTION_TREE
        assert shard["working_tree_clean"] is True
        assert shard["checkpoint"]["sha256"] == CHECKPOINT_SHA256
        assert shard["capabilities"] == NO_CAPABILITIES
        assert shard["summary"]["molecule_index"] == index
        assert shard["summary"]["molecule_id"] == (
            AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS[index]
        )
        assert shard["identity"]["scalar_id"] == SCALAR_ID
        assert shard["identity"]["profile_id"] == PROFILE_ID
        continuum = shard["protocol"]["continuum"]
        assert continuum["aimnet2_source_evaluation"] == "one-shot-per-geometry"
        assert continuum["continuum_field_supplied_to_aimnet2"] is False
        assert continuum["electronic_scf_iteration"] is False
        assert continuum["solvent"] == "water"
        assert continuum["dielectric"] == pytest.approx(78.355, rel=0.0, abs=0.0)

        measured = {key: shard[key] for key in MEASURED_KEYS}
        assert canonical_json_sha256(measured) == shard["measurement_sha256"]
        recomputed = summarize_aimnet2_geometry_mediated_pes_shard(
            molecule_index=index,
            records=shard["records"],
            continuum_kind=CONTINUUM_KIND,
        )
        assert recomputed == shard["summary"]
        input_record = input_records[index]
        assert input_record["sha256"] == _sha256(shard_raw)
        assert input_record["measurement_sha256"] == shard["measurement_sha256"]
        source_ledgers.append(shard["source_files_sha256"])
        raw_shards.append({"molecule_index": index, "records": shard["records"]})

    assert all(ledger == source_ledgers[0] for ledger in source_ledgers)
    assert len(source_ledgers[0]) == 135
    recomputed = summarize_aimnet2_geometry_mediated_pes_panel(
        raw_shards,
        continuum_kind=CONTINUUM_KIND,
    )
    assert recomputed == panel["panel_summary"]
    assert canonical_json_sha256(raw_shards) == AGGREGATE_MEASUREMENT_SHA256
    assert panel["panel_summary"]["failed_molecule_ids"] == FAILED_MOLECULE_IDS
    assert panel["panel_summary"]["diagnostic_gates_passed"] is False
    assert (
        _sha256(panel_raw)
        == _checksums(EVIDENCE / "RAW_SHA256SUMS")[f"{prefix}-panel.json"]
    )
    return panel, raw_shards


def test_frozen_charge_water_ddpcm_panel_recomputes_and_replays_exactly():
    primary, primary_shards = _validate_run("primary")
    replay, replay_shards = _validate_run("replay")
    assert replay_shards == primary_shards
    assert replay["panel_summary"] == primary["panel_summary"]
    assert [record["measurement_sha256"] for record in replay["input_artifacts"]] == [
        record["measurement_sha256"] for record in primary["input_artifacts"]
    ]
    first_shard, _ = _load_compressed("primary-shard-00.json")
    assert_source_files_match_execution_commit(ROOT, first_shard)


def test_frozen_charge_water_ddpcm_negative_boundaries_remain_fail_closed():
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
        "all_energy_cotangent_kkt_audits": True,
        "all_neighbor_cutoff_guards": True,
        "all_primal_responses_excluded_from_provider_field": True,
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
    assert summary["maximum_directional_absolute_error_eV_per_A"] == pytest.approx(
        6.495762976865826e-4, rel=0.0, abs=1.0e-18
    )
    assert summary["minimum_continuum_event_margin_A"] == pytest.approx(
        0.002235739521015301, rel=0.0, abs=1.0e-18
    )
    assert summary["minimum_sphere_tangency_margin_A"] == pytest.approx(
        0.011178247440158717, rel=0.0, abs=1.0e-18
    )
    assert summary["maximum_kkt_vs_autograd_relative_error"] == pytest.approx(
        6.71025801204701e-14, rel=0.0, abs=1.0e-26
    )
    assert summary["capabilities"] == NO_CAPABILITIES
    assert summary["opt_admitted"] is False
    assert summary["freq_ts_irc_admitted"] is False
    assert summary["md_admitted"] is False
