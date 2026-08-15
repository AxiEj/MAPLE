from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pytest

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release import (
    canonical_json_sha256,
    summarize_aimnet2_geometry_mediated_pes_shard,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = ROOT / "docs" / "route2" / "evidence"
EXECUTION_HEAD = "02c21b52ad3bd2d7539640ed4863e4c1d022563f"
EXECUTION_TREE = "834430397c1ac9c446aa4a76b9088490a931df6a"
CHECKPOINT_SHA256 = "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
MEASURED_KEYS = (
    "contract_version",
    "panel_asset_sha256",
    "protocol",
    "identity",
    "records",
    "summary",
)


@dataclass(frozen=True)
class ShardExpectation:
    directory: str
    molecule_index: int
    molecule_id: str
    measurement_sha256: str
    status: str
    diagnostic_gates_passed: bool
    file_sha256: dict[str, str]
    maximum_directional_absolute_error_eV_per_A: float
    minimum_neighbor_cutoff_margin_A: float
    minimum_continuum_event_margin_A: float
    minimum_sphere_tangency_margin_A: float
    maximum_surface_condition_number: float
    maximum_reciprocity_absolute_error_eV: float
    maximum_charge_fd_absolute_error_eV_per_e: float


SHARDS = (
    ShardExpectation(
        directory="aimnet2-geometry-mediated-pes-water-v2-02c21b52",
        molecule_index=0,
        molecule_id="water",
        measurement_sha256=(
            "3357780ea571d3920fe0637ecbf2d92ce2fd852646b7b0f2c0831ba789f13570"
        ),
        status="diagnostic-gates-passed-not-admitted",
        diagnostic_gates_passed=True,
        file_sha256={
            "measurements.json": (
                "5f34132c0d8fb182590cca382b6dfc982983acbbc29249d8a7566f204dcfe877"
            ),
            "cold-replay.json": (
                "2fcb25c9b618d21add18d54bd2e375d6cd638d2c3bd3a5cbf6a35b8c07db119b"
            ),
            "README.md": (
                "0e6a7d3c4a77ef6f32893fac7595591ca3b1fe447aa83a73b5a5517dba65d734"
            ),
        },
        maximum_directional_absolute_error_eV_per_A=5.866424167422224e-05,
        minimum_neighbor_cutoff_margin_A=3.428049281108132,
        minimum_continuum_event_margin_A=0.11773221132261891,
        minimum_sphere_tangency_margin_A=0.5351277072462719,
        maximum_surface_condition_number=150.593715831781,
        maximum_reciprocity_absolute_error_eV=2.6645352591003757e-15,
        maximum_charge_fd_absolute_error_eV_per_e=2.1084523016412504e-12,
    ),
    ShardExpectation(
        directory="aimnet2-geometry-mediated-pes-methanol-v2-02c21b52",
        molecule_index=1,
        molecule_id="methanol",
        measurement_sha256=(
            "9f0b43679a1cbdadcbae2daccf10b62bfab7cad5987db71c853a5e4f75933f20"
        ),
        status="diagnostic-gates-failed-not-admitted",
        diagnostic_gates_passed=False,
        file_sha256={
            "measurements.json": (
                "9aa6f7c96cae674d2a612e3c536fffde6edfb2209279493440ba6e7b3c4a4912"
            ),
            "cold-replay.json": (
                "2f27779b37de05cd899d24e5ffb90b912e09815027809ff8c04193aa1e6bf78f"
            ),
            "README.md": (
                "3de3af024675ae01d9c9950005e54ddf375334e7616312033a5f0782341bb4ec"
            ),
        },
        maximum_directional_absolute_error_eV_per_A=5.690032751104468e-05,
        minimum_neighbor_cutoff_margin_A=2.1782239740961504,
        minimum_continuum_event_margin_A=0.015511399564898554,
        minimum_sphere_tangency_margin_A=0.04978002374103285,
        maximum_surface_condition_number=604.244462442586,
        maximum_reciprocity_absolute_error_eV=1.1102230246251565e-15,
        maximum_charge_fd_absolute_error_eV_per_e=1.887073830530994e-12,
    ),
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_common_artifact(
    artifact: dict[str, object], expectation: ShardExpectation
) -> dict[str, object]:
    assert artifact["schema_version"] == (
        "route2-aimnet2-geometry-mediated-pes-shard-artifact-v2"
    )
    assert artifact["contract_version"] == (
        "route2-aimnet2-geometry-mediated-pes-shard-contract-v2"
    )
    assert artifact["artifact_kind"] == (
        "disabled-aimnet2-reconstructed-float64-geometry-mediated-"
        "smooth-harmonic-pes-shard"
    )
    assert artifact["status"] == expectation.status
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["execution_git_tree"] == EXECUTION_TREE
    assert artifact["working_tree_clean"] is True
    assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert artifact["aimnet_runtime"] == "reconstructed-python-float64"
    assert artifact["continuum_kind"] == "harmonic-point"
    assert artifact["dtype"] == "float64"
    assert artifact["capabilities"] == NO_CAPABILITIES
    assert artifact["measurement_sha256"] == expectation.measurement_sha256
    assert len(artifact["source_files_sha256"]) == 114

    summary = summarize_aimnet2_geometry_mediated_pes_shard(
        molecule_index=expectation.molecule_index,
        records=artifact["records"],
    )
    assert summary == artifact["summary"]
    measured = {key: artifact[key] for key in MEASURED_KEYS}
    assert canonical_json_sha256(measured) == expectation.measurement_sha256

    assert summary["molecule_index"] == expectation.molecule_index
    assert summary["molecule_id"] == expectation.molecule_id
    assert summary["variant_count"] == 3
    assert summary["directional_record_count"] == 9
    assert summary["directional_sample_count"] == 27
    assert summary["diagnostic_gates_passed"] is (expectation.diagnostic_gates_passed)
    assert summary["capabilities"] == NO_CAPABILITIES
    assert summary["opt_admitted"] is False
    assert summary["freq_ts_irc_admitted"] is False
    assert summary["md_admitted"] is False
    for key in (
        "maximum_directional_absolute_error_eV_per_A",
        "minimum_neighbor_cutoff_margin_A",
        "minimum_continuum_event_margin_A",
        "minimum_sphere_tangency_margin_A",
        "maximum_surface_condition_number",
        "maximum_reciprocity_absolute_error_eV",
        "maximum_charge_fd_absolute_error_eV_per_e",
    ):
        assert summary[key] == pytest.approx(
            getattr(expectation, key), rel=0.0, abs=1.0e-24
        )
    assert summary["maximum_charge_gauge_vjp_norm_eV_per_A"] == 0.0
    return summary


@pytest.mark.parametrize("expectation", SHARDS, ids=lambda value: value.molecule_id)
def test_v2_pes_shard_evidence_is_source_bound_and_recomputed(
    expectation: ShardExpectation,
):
    evidence = EVIDENCE_ROOT / expectation.directory
    primary = _load(evidence / "measurements.json")
    replay = _load(evidence / "cold-replay.json")
    primary_summary = _assert_common_artifact(primary, expectation)
    replay_summary = _assert_common_artifact(replay, expectation)
    assert replay_summary == primary_summary
    for key in (*MEASURED_KEYS, "measurement_sha256"):
        assert replay[key] == primary[key]
    assert replay["exact_command"] != primary["exact_command"]
    assert replay["runtime_seconds"] != primary["runtime_seconds"]
    assert_source_files_match_execution_commit(ROOT, primary)

    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (evidence / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == expectation.file_sha256
    for name, expected in expectation.file_sha256.items():
        assert _sha256(evidence / name) == expected


def test_methanol_v2_failure_is_only_the_frozen_point_shell_event_guard():
    expectation = SHARDS[1]
    artifact = _load(EVIDENCE_ROOT / expectation.directory / "measurements.json")
    summary = _assert_common_artifact(artifact, expectation)
    assert summary["gates"] == {
        "all_continuum_event_guards": False,
        "all_deterministic_replays": True,
        "all_directional_force_fd": False,
        "all_neighbor_cutoff_guards": True,
        "all_reciprocity_metric_charge_gauge_audits": True,
        "all_sphere_tangency_guards": True,
        "all_stationarity_audits": True,
        "all_stencils_same_stratum": True,
    }
    stretched = next(
        record
        for record in summary["geometry_records"]
        if record["variant"] == "bond-stretched"
    )
    assert stretched["minimum_continuum_event_margin_A"] < 0.02
    assert stretched["minimum_sphere_tangency_margin_A"] > 0.02
    for directional in stretched["directional_force_fd"].values():
        assert directional["gate_passed"] is False
        assert directional["convergence"]["gate_passed"] is True
        assert all(
            record["numerical_gate_passed"] is True
            and record["same_model_and_cavity_stratum"] is True
            and record["continuum_event_guard_passed"] is False
            and record["sphere_tangency_guard_passed"] is True
            for record in directional["records"]
        )
