from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release import canonical_json_sha256
from maple.solvation.release.geometry_mediated import (
    geometry_mediated_admission_decision,
    summarize_geometry_mediated_cartesian_audit,
    summarize_geometry_mediated_directional_audit,
    summarize_geometry_mediated_reciprocity_audit,
    summarize_geometry_mediated_rotation_audit,
)
from maple.solvation.release.geometry_mediated_adaptive import (
    summarize_geometry_mediated_adaptive_directional_audit,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "aimnet2-frozen-charge-water-harmonic-ddpcm-0c19ede4"
)
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"

EXECUTION_HEAD = "0c19ede4b4354e2c75deeea98ea27ba19c374846"
EXECUTION_TREE = "111417ed58787c775bce00b3c1405a4b48db58eb"
CHECKPOINT_SHA256 = "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d"
MEASUREMENT_SHA256 = "29ad72cd84a137b6bc7b9079ce6979000e7dab58983e2c41d6d2e3ce82fa08e5"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILE_SHA256 = {
    "measurements.json": "db2f0a63ddde9b4acc62d3aadc95f1881573b9965efa7c2483b5b8c4145854fe",
    "cold-replay.json": "e5251243fe429430f51bb93a47e72861c17a6f2639aff526981abe1d16285767",
    "README.md": "7143bfc786fbb2737d632fc09fb611b84aed7fb9401cceba9a9007a7447aa391",
}
MEASURED_KEYS = (
    "protocol",
    "geometry",
    "identity",
    "center",
    "deterministic_replay",
    "reciprocity_metric_charge_gauge",
    "stationarity",
    "coordinate_directional",
    "coordinate_directional_adaptive",
    "coordinate_cartesian",
    "rigid_rotation",
    "decision",
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recompute(artifact: dict[str, object]) -> dict[str, object]:
    center = artifact["center"]
    identity = artifact["identity"]
    protocol = artifact["protocol"]
    reciprocity_raw = artifact["reciprocity_metric_charge_gauge"]
    reciprocity = summarize_geometry_mediated_reciprocity_audit(
        reciprocity_raw,
        reaction_field=center["reaction_field"],
    )
    assert reciprocity["gate_passed"] is reciprocity_raw["gate_passed"]
    for name in (
        "maximum_reciprocity_absolute_error_eV",
        "maximum_reciprocity_relative_error",
        "maximum_apply_adjoint_absolute_error_eV",
        "maximum_apply_adjoint_relative_error",
        "maximum_charge_fd_absolute_error_eV_per_e",
        "maximum_charge_fd_relative_error",
        "source_gradient_half_error_eV_per_source_unit",
        "charge_gauge_vjp_norm_eV_per_A",
    ):
        assert reciprocity[name] == pytest.approx(
            reciprocity_raw[name], rel=0.0, abs=1.0e-24
        )

    common = {
        "analytic_gradient_eV_per_A": center["gradient_components_eV_per_A"]["total"],
        "center_model_topology": identity["center_model_topology"],
        "center_continuum_topology": identity["center_continuum_topology"],
    }
    directional = summarize_geometry_mediated_directional_audit(
        **common,
        direction=protocol["coordinate_direction"],
        samples=artifact["coordinate_directional"]["raw"],
        reciprocity_audit=reciprocity_raw,
    )
    assert directional == artifact["coordinate_directional"]["summary"]

    cartesian = summarize_geometry_mediated_cartesian_audit(
        **common,
        samples=artifact["coordinate_cartesian"]["raw_displaced_components"],
        reciprocity_audit=reciprocity_raw,
    )
    assert cartesian == artifact["coordinate_cartesian"]

    adaptive = summarize_geometry_mediated_adaptive_directional_audit(
        **common,
        direction=protocol["coordinate_direction"],
        center_positions_A=artifact["geometry"]["positions_A"],
        adaptive_record=artifact["coordinate_directional_adaptive"]["raw"],
        reciprocity_audit=reciprocity_raw,
    )
    assert adaptive == artifact["coordinate_directional_adaptive"]["summary"]

    rotation = summarize_geometry_mediated_rotation_audit(
        positions_A=artifact["geometry"]["positions_A"],
        base_energy_eV=center["total_energy_eV"],
        base_forces_eV_per_A=center["forces_eV_per_A"],
        base_source=center["source"],
        base_model_topology=identity["center_model_topology"],
        base_continuum_topology=identity["center_continuum_topology"],
        rotation_records=artifact["rigid_rotation"]["raw"],
    )
    assert rotation == artifact["rigid_rotation"]["summary"]

    decision = geometry_mediated_admission_decision(
        deterministic_replay_passed=artifact["deterministic_replay"]["gate_passed"],
        directional_audit=directional,
        cartesian_audit=cartesian,
        rotation_audit=rotation,
        post_solve_residual_available=True,
    )
    decision["adaptive_directional_gate_passed"] = adaptive["gate_passed"]
    decision["local_diagnostic_gates_passed"] = bool(
        decision["local_diagnostic_gates_passed"]
        and decision["adaptive_directional_gate_passed"]
    )
    assert decision == artifact["decision"]
    return {
        "adaptive": adaptive,
        "cartesian": cartesian,
        "decision": decision,
        "directional": directional,
        "reciprocity": reciprocity,
        "rotation": rotation,
    }


def _assert_artifact(artifact: dict[str, object]) -> dict[str, object]:
    assert artifact["schema_version"] == (
        "route2-aimnet2-geometry-mediated-real-stack-canary-v6"
    )
    assert artifact["artifact_kind"] == (
        "disabled-aimnet2-reconstructed-python-float64-geometry-mediated-"
        "harmonic-ddpcm-water-real-stack-canary"
    )
    assert artifact["status"] == "diagnostic-gates-passed-not-admitted"
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["execution_git_tree"] == EXECUTION_TREE
    assert artifact["working_tree_clean"] is True
    assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
    assert artifact["capabilities"] == NO_CAPABILITIES
    assert artifact["aimnet_runtime"] == "reconstructed-python-float64"
    assert artifact["continuum_kind"] == "harmonic-ddpcm-water"
    assert artifact["dtype"] == "float64"
    assert len(artifact["source_files_sha256"]) == 135
    assert artifact["runtime"]["packages"]["torch"] == "2.12.0+cu130"

    identity = artifact["identity"]
    assert identity["scalar_id"] == (
        "route2-candidate-aimnet2-frozen-charge-water-"
        "smoothharmonicgalerkin-ddpcm-electrostatic-v1"
    )
    assert identity["profile_id"] == (
        "route2-profile-candidate-aimnet2-frozen-charge-water-"
        "smoothharmonicgalerkin-ddpcm-electrostatic-v1"
    )
    assert identity["model_provider_id"] == (
        "maple.route2.model.aimnet2-frozen-charge-water-float64.impl.v1"
    )
    assert identity["continuum_provider_id"] == (
        "maple.route2.continuum.aimnet2-frozen-charge-water-"
        "smooth-harmonic-ddpcm.impl.v1"
    )

    model_runtime = artifact["identity"]["model_runtime"]
    assert model_runtime["package_versions"]["aimnet"] == "0.2.0"
    assert model_runtime["checkpoint_weights_changed"] is False
    assert model_runtime["coordinate_dtype"] == "torch.float64"
    assert model_runtime["public_ase_calculator"] is False
    assert model_runtime["public_hessian"] is False
    assert model_runtime["public_hvp"] is False
    assert model_runtime["route2_public_ase_admitted"] is False

    continuum = artifact["protocol"]["continuum"]
    assert continuum["model"] == (
        "water-bound-smooth-weighted-harmonic-finite-dielectric-ddpcm"
    )
    assert continuum["finite_dielectric_parameterization"] is True
    assert continuum["uniform_cosmo_dielectric_energy_scaling"] is False
    assert continuum["dielectric"] == pytest.approx(78.355, rel=0.0, abs=0.0)
    assert continuum["admission_identity"] == (
        "water-bound-frozen-charge-candidate; capabilities-none"
    )
    assert continuum["solvent"] == "water"
    assert continuum["radii_A"] == [1.52, 1.2, 1.2]
    assert continuum["transition_width_A2"] == pytest.approx(0.18)
    assert continuum["surface_lmax"] == 1
    assert continuum["exposure_lmax"] == 2
    assert continuum["exposure_radial_quadrature_order"] == 32
    assert continuum["green_radial_quadrature_order"] == 32
    assert continuum["aimnet2_source_evaluation"] == "one-shot-per-geometry"
    assert continuum["continuum_field_supplied_to_aimnet2"] is False
    assert continuum["electronic_scf_iteration"] is False
    assert continuum["post_solve_residual_available"] is True

    measured = {key: artifact[key] for key in MEASURED_KEYS}
    assert canonical_json_sha256(measured) == MEASUREMENT_SHA256
    recomputed = _recompute(artifact)

    assert artifact["center"]["continuum_energy_eV"] == pytest.approx(
        -0.49914801549225324, rel=0.0, abs=1.0e-15
    )
    adaptive = recomputed["adaptive"]
    assert adaptive["absolute_error_eV_per_A"] == pytest.approx(
        1.8257224045914455e-6, rel=0.0, abs=1.0e-18
    )
    assert adaptive["relative_error"] == pytest.approx(
        2.060202711135806e-6, rel=0.0, abs=1.0e-18
    )
    assert adaptive["scipy_result"]["nfev"] == 7
    assert adaptive["gate_passed"] is True

    directional = recomputed["directional"]
    assert directional["records"][-1]["absolute_error_eV_per_A"] == pytest.approx(
        1.7625409445276574e-6, rel=0.0, abs=1.0e-18
    )
    assert directional["gate_passed"] is True

    cartesian = recomputed["cartesian"]
    assert cartesian["records"][-1]["maximum_error_eV_per_A"] == pytest.approx(
        2.826476747208595e-5, rel=0.0, abs=1.0e-18
    )
    assert cartesian["records"][-1]["rms_error_eV_per_A"] == pytest.approx(
        1.5826934760464932e-5, rel=0.0, abs=1.0e-18
    )
    assert cartesian["convergence"]["observed_first_to_last_order"] == (
        pytest.approx(1.9945636040785466, rel=0.0, abs=1.0e-15)
    )
    assert cartesian["gate_passed"] is True

    rotation = recomputed["rotation"]
    assert (
        max(record["energy_absolute_error_eV"] for record in rotation["records"]) == 0.0
    )
    assert max(
        record["force_covariance_relative_error"] for record in rotation["records"]
    ) == pytest.approx(5.5951040041648464e-11, rel=0.0, abs=1.0e-22)
    assert rotation["base_net_force_norm_eV_per_A"] == pytest.approx(
        4.658887789198649e-11, rel=0.0, abs=1.0e-22
    )
    assert rotation["base_torque_norm_eV"] == pytest.approx(
        3.0449087707327465e-11, rel=0.0, abs=1.0e-22
    )
    assert rotation["gate_passed"] is True

    stationarity = artifact["stationarity"]
    assert stationarity["finite_dielectric_parameterization"] is True
    assert stationarity["gate_passed"] is True
    assert max(stationarity["condition_numbers"].values()) == pytest.approx(
        109.61213248462516, rel=0.0, abs=1.0e-13
    )
    assert stationarity["response_operator_audit"][
        "half_coupling_absolute_error_eV"
    ] == pytest.approx(4.440892098500626e-16, rel=0.0, abs=1.0e-30)

    decision = recomputed["decision"]
    assert decision["tier_f_prerequisites_passed"] is True
    assert decision["local_diagnostic_gates_passed"] is True
    assert all(
        decision[name] is False
        for name in (
            "public_energy_admitted",
            "public_force_admitted",
            "opt_admitted",
            "hessian_freq_ts_irc_admitted",
            "md_admitted",
            "tier_v_mutual_polarization_admitted",
        )
    )
    return recomputed


def test_water_frozen_charge_ddpcm_artifact_is_source_bound_recomputed_and_closed():
    artifact = _load(PRIMARY)
    _assert_artifact(artifact)
    assert_source_files_match_execution_commit(ROOT, artifact)


def test_water_frozen_charge_ddpcm_cold_replay_is_scientifically_identical():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)
    _assert_artifact(replay)
    for key in (*MEASURED_KEYS, "measurement_sha256"):
        assert replay[key] == primary[key]
    assert replay["exact_command"] != primary["exact_command"]
    assert replay["runtime_seconds"] != primary["runtime_seconds"]


def test_water_frozen_charge_ddpcm_adaptive_raw_trace_rejects_energy_tampering():
    artifact = _load(PRIMARY)
    tampered = deepcopy(artifact["coordinate_directional_adaptive"]["raw"])
    tampered["samples"][1]["energy_eV"] += 1.0e-8
    with pytest.raises(ValueError, match="SciPy result disagrees"):
        summarize_geometry_mediated_adaptive_directional_audit(
            analytic_gradient_eV_per_A=artifact["center"][
                "gradient_components_eV_per_A"
            ]["total"],
            direction=artifact["protocol"]["coordinate_direction"],
            center_positions_A=artifact["geometry"]["positions_A"],
            center_model_topology=artifact["identity"]["center_model_topology"],
            center_continuum_topology=artifact["identity"]["center_continuum_topology"],
            adaptive_record=tampered,
            reciprocity_audit=artifact["reciprocity_metric_charge_gauge"],
        )


def test_water_frozen_charge_ddpcm_evidence_file_hashes_are_frozen():
    for name, expected in FILE_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILE_SHA256
