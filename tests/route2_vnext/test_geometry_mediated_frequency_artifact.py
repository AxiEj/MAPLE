from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release import (
    canonical_json_sha256,
    summarize_aimnet2_geometry_mediated_frequency_water,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "aimnet2-geometry-mediated-frequency-water-f79d5051"
)
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"

EXECUTION_HEAD = "f79d50513a39cb4fd6e77b8d3a80dc4b65aa12f8"
EXECUTION_TREE = "c086a9592d6f1b2c98d58cc506aed0992aa20075"
CHECKPOINT_SHA256 = "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d"
MEASUREMENT_SHA256 = "126327853eb0caadcf5e98b41992030789bbdf0f7ddf23cec9c4c31ed98b99a4"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILE_SHA256 = {
    "measurements.json": "59ecf9aa2e8fde3b76c3f960230fb1cb1270a84b9141aa62eb1eca4fa5c1aac3",
    "cold-replay.json": "13787871457da1b64d699eff09e959d0dacdc4f2f1873c915a2876014ab230d6",
    "README.md": "bb9ed5783b036b973d18ddfec50a46bfa5ad33f12e785f70797e8422f7ba4b97",
}
MEASURED_KEYS = (
    "contract_version",
    "protocol",
    "identity",
    "admission_boundary",
    "search_record",
    "center_record",
    "hessian_vector_records",
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
        "route2-aimnet2-geometry-mediated-frequency-water-artifact-v1"
    )
    assert artifact["artifact_kind"] == (
        "disabled-aimnet2-reconstructed-float64-geometry-mediated-"
        "smooth-harmonic-stationary-water-dense-hessian-frequency"
    )
    assert artifact["status"] == "diagnostic-frequency-gates-passed-not-admitted"
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["execution_git_tree"] == EXECUTION_TREE
    assert artifact["working_tree_clean"] is True
    assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
    assert artifact["capabilities"] == NO_CAPABILITIES
    assert artifact["aimnet_runtime"] == "reconstructed-python-float64"
    assert artifact["continuum_kind"] == "harmonic-point"
    assert artifact["dtype"] == "float64"
    assert len(artifact["source_files_sha256"]) == 128

    protocol = artifact["protocol"]
    assert protocol["outer_charge_fixed_point"] is False
    assert protocol["fixed_geometry_mutual_polarization"] is False
    assert protocol["finite_difference_target"] == "total-scalar-gradient"
    assert protocol["hessian_input_unit"] == "eV/angstrom^2"
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

    recomputed = summarize_aimnet2_geometry_mediated_frequency_water(
        search_record=artifact["search_record"],
        center_record=artifact["center_record"],
        hessian_vector_records=artifact["hessian_vector_records"],
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

    assert summary["search"]["result"]["solution"] == pytest.approx(
        [1.0226804300596355, 1.0226804300596355, 2.0118160254551998],
        rel=0.0,
        abs=1.0e-15,
    )
    assert summary["search"]["result"]["nfev"] == 15
    assert summary["center"]["gradient_norm_eV_per_A"] == pytest.approx(
        5.375312708523219e-12,
        rel=0.0,
        abs=1.0e-27,
    )
    assert summary["center"]["stationarity"]["absolute_residual_eV_per_e"] == (
        pytest.approx(7.961471427347596e-15, rel=0.0, abs=1.0e-29)
    )
    assert summary["dense_hessian"]["symmetry_max_abs_eV_per_A2"] == (
        pytest.approx(1.2434497875801753e-14, rel=0.0, abs=1.0e-29)
    )
    assert summary["finite_difference_error_norms"]["frobenius_eV_per_A2"] == (
        pytest.approx(
            [
                1.4897589772074288e-3,
                3.69929014813758e-4,
                9.454998028117011e-5,
                4.074765566486307e-5,
            ],
            rel=0.0,
            abs=1.0e-18,
        )
    )
    assert summary["central_convergence_ratios"]["maximum_absolute"] == (
        pytest.approx(
            [
                0.24719688704779524,
                0.25254109360729227,
                0.45356974587629545,
            ],
            rel=0.0,
            abs=1.0e-15,
        )
    )
    assert summary["rigid_modes"]["translation_HVP_norms_eV_per_A2"] == (
        pytest.approx(
            [
                7.303296316030796e-16,
                1.6249875634344887e-14,
                1.5302438523914105e-14,
            ],
            rel=0.0,
            abs=1.0e-28,
        )
    )
    assert summary["rigid_modes"]["rotation_HVP_norms_eV_per_A2"] == (
        pytest.approx(
            [
                4.0596458768953666e-9,
                2.7068386143688735e-9,
                6.719144789488478e-9,
            ],
            rel=0.0,
            abs=1.0e-22,
        )
    )
    assert summary["vibrational_analysis"]["frequencies_cm1"] == pytest.approx(
        [1638.2321451446662, 2629.6268042951565, 2831.3349242547506],
        rel=0.0,
        abs=1.0e-10,
    )
    assert summary["vibrational_analysis"][
        "eigenpair_residual_norms_eV_per_A2_amu"
    ] == pytest.approx(
        [4.169988336685166e-15, 5.187741650369041e-10, 9.946295946304222e-15],
        rel=0.0,
        abs=1.0e-23,
    )
    assert summary["model_parity_maxima"]["energy_absolute_error_eV"] == (
        pytest.approx(1.2964846973773092e-9, rel=0.0, abs=1.0e-23)
    )
    assert summary["maximum_source_tangent_residual_e_per_A"] == pytest.approx(
        1.3877787807814457e-17,
        rel=0.0,
        abs=1.0e-31,
    )


def test_frequency_artifact_recomputes_raw_operands_and_stays_closed():
    primary = _load(PRIMARY)
    _assert_measurement(primary)
    assert_source_files_match_execution_commit(ROOT, primary)


def test_frequency_cold_replay_is_scientifically_identical():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)
    _assert_measurement(replay)
    for key in (*MEASURED_KEYS, "measurement_sha256"):
        assert replay[key] == primary[key]
    assert replay["exact_command"] != primary["exact_command"]
    assert replay["runtime_seconds"] != primary["runtime_seconds"]


def test_frequency_evidence_file_hashes_are_frozen():
    for name, expected in FILE_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILE_SHA256
