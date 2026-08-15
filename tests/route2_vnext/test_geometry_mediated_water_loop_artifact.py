from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release import (
    canonical_json_sha256,
    summarize_aimnet2_geometry_mediated_water_loop,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "aimnet2-geometry-mediated-water-loop-e0a347f5"
)
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"

EXECUTION_HEAD = "e0a347f5f4e5a93f9b238333964273a6f19cbf02"
EXECUTION_TREE = "db21006bd6996a64ee5bb945eafbe8a57e45ffc7"
CHECKPOINT_SHA256 = "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d"
MEASUREMENT_SHA256 = "5e389819e022dda7a2466d9e472486f45cc9bd6715c9735efefa0bd0b0dad0e3"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILE_SHA256 = {
    "measurements.json": (
        "ecb496767dde7562b35e8b175ff311dc17ed3b59a6ca54c9ad4dc5cab3bc1bb5"
    ),
    "cold-replay.json": (
        "d7e8ac83bca638f4dd09f50132c000a3d97f58007c5cb952e5ffe0bae8da15cb"
    ),
    "README.md": "57e929ea95b74edd1929949d70c8fd01bd9a4f428112e1f2d2f953051f144cdc",
}
MEASURED_KEYS = (
    "contract_version",
    "protocol",
    "identity",
    "forward_records",
    "reverse_records",
    "summary",
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _minimum_segment_bound(summary: dict[str, object], topology: str) -> float:
    values = []
    for traversal_name in ("forward", "reverse"):
        traversal = summary[traversal_name]
        assert isinstance(traversal, dict)
        for guard in traversal["segment_guards"]:
            values.append(guard[topology]["certified_segment_lower_bound_A"])
    return min(values)


def _assert_measurement(artifact: dict[str, object]) -> None:
    assert artifact["schema_version"] == (
        "route2-aimnet2-geometry-mediated-water-loop-artifact-v1"
    )
    assert artifact["artifact_kind"] == (
        "disabled-aimnet2-reconstructed-float64-geometry-mediated-"
        "smooth-harmonic-water-bidirectional-loop"
    )
    assert artifact["status"] == "diagnostic-gates-passed-not-admitted"
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["execution_git_tree"] == EXECUTION_TREE
    assert artifact["working_tree_clean"] is True
    assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert artifact["aimnet_runtime"] == "reconstructed-python-float64"
    assert artifact["continuum_kind"] == "harmonic-point"
    assert artifact["dtype"] == "float64"
    assert artifact["capabilities"] == NO_CAPABILITIES
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
    assert len(artifact["source_files_sha256"]) == 114

    protocol = artifact["protocol"]
    assert protocol["outer_charge_fixed_point"] is False
    assert protocol["fixed_geometry_mutual_polarization"] is False
    assert protocol["independent_forward_reverse_evaluations"] is True
    assert protocol["continuum"]["finite_dielectric_parameterization"] is False

    identity = artifact["identity"]
    assert identity["profile_id"] == (
        "route2-profile-diagnostic-aimnet2-geometry-mediated-"
        "smoothharmonicgalerkin-cpcm-electrostatic-v1"
    )
    assert identity["scalar_id"] == (
        "route2-diagnostic-aimnet2-geometry-mediated-"
        "smoothharmonicgalerkin-cpcm-electrostatic-v1"
    )

    recomputed = summarize_aimnet2_geometry_mediated_water_loop(
        forward_records=artifact["forward_records"],
        reverse_records=artifact["reverse_records"],
    )
    assert recomputed == artifact["summary"]
    measured = {key: artifact[key] for key in MEASURED_KEYS}
    assert canonical_json_sha256(measured) == MEASUREMENT_SHA256

    summary = recomputed
    assert summary["diagnostic_gates_passed"] is True
    assert all(summary["gates"].values())
    assert summary["capabilities"] == NO_CAPABILITIES
    assert summary["opt_admitted"] is False
    assert summary["freq_ts_irc_admitted"] is False
    assert summary["md_admitted"] is False
    assert summary["point_count_per_traversal"] == 17
    assert summary["forward"]["work"]["simpson_work_eV"] == pytest.approx(
        0.00014855129592190874, rel=0.0, abs=1.0e-20
    )
    assert summary["reverse"]["work"]["simpson_work_eV"] == pytest.approx(
        -0.0001485512959219093, rel=0.0, abs=1.0e-20
    )
    assert summary["forward_reverse_work_sum_eV"] == pytest.approx(
        -5.692061405548898e-19, rel=0.0, abs=1.0e-30
    )
    assert summary["maximum_same_coordinate_energy_error_eV"] == 0.0
    assert summary["maximum_same_coordinate_source_error"] == 0.0
    assert summary["maximum_same_coordinate_force_error_eV_per_A"] == 0.0
    assert summary["maximum_stationarity_absolute_residual_eV_per_e"] == pytest.approx(
        5.497741648040659e-15, rel=0.0, abs=1.0e-27
    )
    assert summary["maximum_surface_condition_number"] == pytest.approx(
        113.03902157292583, rel=0.0, abs=1.0e-12
    )
    assert summary["maximum_reciprocity_absolute_error_eV"] == pytest.approx(
        3.1086244689504383e-15, rel=0.0, abs=1.0e-27
    )
    assert summary["maximum_charge_fd_absolute_error_eV_per_e"] == pytest.approx(
        4.294398170401337e-12, rel=0.0, abs=1.0e-24
    )
    assert summary["maximum_source_gradient_half_error_eV_per_source_unit"] == (
        pytest.approx(9.650801630464006e-16, rel=0.0, abs=1.0e-28)
    )
    assert summary["maximum_charge_gauge_vjp_norm_eV_per_A"] == 0.0
    assert summary["minimum_neighbor_cutoff_margin_A"] == pytest.approx(
        3.4355787226946504, rel=0.0, abs=1.0e-15
    )
    assert summary["minimum_point_source_shell_margin_A"] == pytest.approx(
        0.20407200707293016, rel=0.0, abs=1.0e-15
    )
    assert summary["minimum_sphere_tangency_margin_A"] == pytest.approx(
        0.6216205409966026, rel=0.0, abs=1.0e-15
    )
    assert _minimum_segment_bound(summary, "neighbor_cutoff") == pytest.approx(
        3.423985104581346, rel=0.0, abs=1.0e-15
    )
    assert _minimum_segment_bound(summary, "point_source_shell") == pytest.approx(
        0.1924783889596255, rel=0.0, abs=1.0e-15
    )
    assert _minimum_segment_bound(summary, "sphere_tangency") == pytest.approx(
        0.610026922883298, rel=0.0, abs=1.0e-15
    )


def test_water_loop_artifact_recomputes_raw_diagnostics_and_stays_closed():
    primary = _load(PRIMARY)
    _assert_measurement(primary)
    assert_source_files_match_execution_commit(ROOT, primary)


def test_water_loop_cold_replay_is_scientifically_identical():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)
    _assert_measurement(replay)
    for key in (*MEASURED_KEYS, "measurement_sha256"):
        assert replay[key] == primary[key]
    assert replay["exact_command"] != primary["exact_command"]
    assert replay["runtime_seconds"] != primary["runtime_seconds"]


def test_water_loop_evidence_file_hashes_are_frozen():
    assert FILE_SHA256
    for name, expected in FILE_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILE_SHA256
