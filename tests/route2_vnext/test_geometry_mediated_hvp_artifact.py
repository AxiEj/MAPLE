from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release import (
    canonical_json_sha256,
    summarize_aimnet2_geometry_mediated_hvp_water,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "aimnet2-geometry-mediated-hvp-water-2b119022"
)
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"

EXECUTION_HEAD = "2b11902290ed9ccf884410cd30773fa0bb9abcf7"
EXECUTION_TREE = "1dd9124f395e4b2fdd9484ea5a416a0cbb47f564"
CHECKPOINT_SHA256 = "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d"
MEASUREMENT_SHA256 = "01009ee3f5f829da5906cf051f4bc6b110d8c28300ff31467a6dc663943b88a6"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILE_SHA256 = {
    "measurements.json": "17b462ec4f6faa80f164ee960cf1dc46f13d5eb10033beecd7edd12c07622f4f",
    "cold-replay.json": "15fd705bb9d6918f3cd6821af0ec461db317ace3c39a088b8c93be7ed7f20838",
    "README.md": "a908a837be3b808dc47404863626addccd61b9dc5d0c495659dcb9e59362ceb3",
}
MEASURED_KEYS = (
    "contract_version",
    "protocol",
    "identity",
    "admission_boundary",
    "center_record",
    "direction_records",
    "finite_difference_records",
    "summary",
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_measurement(artifact: dict[str, object]) -> None:
    assert artifact["schema_version"] == (
        "route2-aimnet2-geometry-mediated-hvp-water-artifact-v1"
    )
    assert artifact["artifact_kind"] == (
        "disabled-aimnet2-reconstructed-float64-geometry-mediated-"
        "smooth-harmonic-water-complete-hvp"
    )
    assert artifact["status"] == "diagnostic-hvp-gates-passed-not-admitted"
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["execution_git_tree"] == EXECUTION_TREE
    assert artifact["working_tree_clean"] is True
    assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
    assert artifact["capabilities"] == NO_CAPABILITIES
    assert artifact["aimnet_runtime"] == "reconstructed-python-float64"
    assert artifact["continuum_kind"] == "harmonic-point"
    assert artifact["dtype"] == "float64"
    assert len(artifact["source_files_sha256"]) == 115

    protocol = artifact["protocol"]
    assert protocol["outer_charge_fixed_point"] is False
    assert protocol["fixed_geometry_mutual_polarization"] is False
    assert protocol["fixed_source_cotangent_in_contracted_charge_hessian_fd"] is True
    assert protocol["continuum"]["finite_dielectric_parameterization"] is False
    runtime = artifact["identity"]["model_runtime"]
    assert runtime["checkpoint_weights_changed"] is False
    assert runtime["public_hessian"] is False
    assert runtime["public_hvp"] is False
    assert runtime["public_ase_calculator"] is False
    assert runtime["route2_public_ase_admitted"] is False

    boundary = artifact["admission_boundary"]
    assert boundary["continuum_capabilities"] == NO_CAPABILITIES
    assert all(
        value is False
        for key, value in boundary.items()
        if key != "continuum_capabilities"
    )

    recomputed = summarize_aimnet2_geometry_mediated_hvp_water(
        center_record=artifact["center_record"],
        direction_records=artifact["direction_records"],
        finite_difference_records=artifact["finite_difference_records"],
    )
    assert recomputed == artifact["summary"]
    measured = {key: artifact[key] for key in MEASURED_KEYS}
    assert canonical_json_sha256(measured) == MEASUREMENT_SHA256

    summary = recomputed
    assert summary["diagnostic_gates_passed"] is True
    assert all(summary["gates"].values())
    assert summary["capabilities"] == NO_CAPABILITIES
    assert all(value is False for value in summary["workflow_admission"].values())
    assert summary["tier_h_admitted"] is False
    assert summary["freq_ts_irc_admitted"] is False
    assert summary["md_admitted"] is False
    assert summary["finite_difference_error_norms"]["charge_jvp_e_per_A"] == (
        pytest.approx(
            [
                3.524794331122906e-7,
                8.81192833351706e-8,
                2.203106592738926e-8,
            ],
            rel=0.0,
            abs=1.0e-20,
        )
    )
    assert summary["finite_difference_error_norms"][
        "contracted_charge_hessian_eV_per_A2"
    ] == pytest.approx(
        [
            8.140312848327802e-6,
            2.035084244973546e-6,
            5.08769444399133e-7,
        ],
        rel=0.0,
        abs=1.0e-19,
    )
    assert summary["finite_difference_error_norms"]["total_hvp_eV_per_A2"] == (
        pytest.approx(
            [
                5.355905410854544e-4,
                1.4003029983263116e-4,
                3.391279860294818e-5,
            ],
            rel=0.0,
            abs=1.0e-18,
        )
    )
    assert summary["bilinear_symmetry"]["absolute_error_eV_per_A2"] == (
        pytest.approx(3.552713678800501e-15, rel=0.0, abs=1.0e-30)
    )
    assert summary["translation_zero_modes"]["source_jvp_norms_e_per_A"] == [
        0.0,
        0.0,
        0.0,
    ]
    assert summary["translation_zero_modes"]["total_hvp_norms_eV_per_A2"] == [
        0.0,
        0.0,
        0.0,
    ]
    assert summary["model_parity_maxima"]["energy_absolute_error_eV"] == (
        pytest.approx(4.306457412894815e-10, rel=0.0, abs=1.0e-24)
    )
    assert summary["model_parity_maxima"][
        "intrinsic_gradient_max_absolute_error_eV_per_A"
    ] == pytest.approx(1.4042003115832813e-9, rel=0.0, abs=1.0e-23)


def test_hvp_artifact_recomputes_raw_operands_and_stays_closed():
    primary = _load(PRIMARY)
    _assert_measurement(primary)
    assert_source_files_match_execution_commit(ROOT, primary)


def test_hvp_cold_replay_is_scientifically_identical():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)
    _assert_measurement(replay)
    for key in (*MEASURED_KEYS, "measurement_sha256"):
        assert replay[key] == primary[key]
    assert replay["exact_command"] != primary["exact_command"]
    assert replay["runtime_seconds"] != primary["runtime_seconds"]


def test_hvp_evidence_file_hashes_are_frozen():
    for name, expected in FILE_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILE_SHA256
