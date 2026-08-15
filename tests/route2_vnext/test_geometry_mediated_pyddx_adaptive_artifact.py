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
    / "aimnet2-geometry-mediated-pyddx-adaptive-water-a816f733"
)
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"

EXECUTION_HEAD = "a816f73365623800a8f1c68058ee34b9fe982eaf"
EXECUTION_TREE = "f8fff2ae6cd0fd0336ba0f0df52ec7361cf30cc9"
CHECKPOINT_SHA256 = "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d"
MEASUREMENT_SHA256 = "6fad81d35b8da5926e0f9117baed0f6e88d295729fd3ab3d56df9016a595061c"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILE_SHA256 = {
    "measurements.json": "d6932956439d55cbe5d547ca23da98744cdc56264a45d0eda1794173218a6ad3",
    "cold-replay.json": "c0045179c5b534c28654c6b2c53e34a141340dabfccc8744ae0bed64aa6db2f5",
    "README.md": "13f15034e088cd691576c3e5c20da583082bc0ec9db22347caa08df83b31f9c4",
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
        post_solve_residual_available=False,
    )
    decision["adaptive_directional_gate_passed"] = adaptive["gate_passed"]
    decision["local_diagnostic_gates_passed"] = bool(
        decision["local_diagnostic_gates_passed"]
        and decision["adaptive_directional_gate_passed"]
    )
    assert decision == artifact["decision"]
    return {
        "directional": directional,
        "cartesian": cartesian,
        "adaptive": adaptive,
        "rotation": rotation,
        "decision": decision,
    }


def _assert_artifact(artifact: dict[str, object]) -> dict[str, object]:
    assert artifact["schema_version"] == (
        "route2-aimnet2-geometry-mediated-real-stack-canary-v5"
    )
    assert artifact["artifact_kind"] == (
        "disabled-aimnet2-reconstructed-python-float64-geometry-mediated-"
        "ddpcm-real-stack-canary"
    )
    assert artifact["status"] == "diagnostic-gates-failed-not-admitted"
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["execution_git_tree"] == EXECUTION_TREE
    assert artifact["working_tree_clean"] is True
    assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
    assert artifact["capabilities"] == NO_CAPABILITIES
    assert artifact["aimnet_runtime"] == "reconstructed-python-float64"
    assert artifact["continuum_kind"] == "ddpcm"
    assert artifact["dtype"] == "float64"
    assert len(artifact["source_files_sha256"]) == 134
    assert artifact["runtime"]["packages"]["pyddx"] == "0.8.0"
    assert artifact["runtime"]["packages"]["scipy"] == "1.17.1"
    assert artifact["identity"]["model_runtime"]["checkpoint_weights_changed"] is False
    assert artifact["stationarity"] is None

    measured = {key: artifact[key] for key in MEASURED_KEYS}
    assert canonical_json_sha256(measured) == MEASUREMENT_SHA256
    recomputed = _recompute(artifact)

    topology = artifact["identity"]["center_continuum_topology"]
    assert topology["cavity_active_node_count"] == 540
    assert topology["minimum_cavity_active_set_clearance_angstrom"] == pytest.approx(
        5.826261813812086e-9, rel=0.0, abs=1.0e-24
    )
    reciprocity = artifact["reciprocity_metric_charge_gauge"]
    assert reciprocity["gate_passed"] is True
    assert reciprocity["maximum_reciprocity_absolute_error_eV"] == pytest.approx(
        7.327471962526033e-14, rel=0.0, abs=1.0e-28
    )
    assert reciprocity["maximum_charge_fd_absolute_error_eV_per_e"] == pytest.approx(
        3.476902099563972e-11, rel=0.0, abs=1.0e-25
    )
    assert reciprocity["charge_gauge_vjp_norm_eV_per_A"] == 0.0

    adaptive = recomputed["adaptive"]
    assert adaptive["scipy_result"] == {
        "success": True,
        "status": 0,
        "derivative_eV_per_A": pytest.approx(
            -0.09163049909943766, rel=0.0, abs=1.0e-15
        ),
        "error_estimate_eV_per_A": pytest.approx(
            4.7343462483695475e-7, rel=0.0, abs=1.0e-21
        ),
        "iterations": 2,
        "nfev": 7,
        "x_A": 0.0,
    }
    assert adaptive["absolute_error_eV_per_A"] == pytest.approx(
        3.190291369137288e-7, rel=0.0, abs=1.0e-21
    )
    assert adaptive["adaptive_convergence_gate_passed"] is True
    assert adaptive["analytic_agreement_gate_passed"] is True
    assert adaptive["topology"]["all_samples_same_stratum"] is False
    assert adaptive["topology"]["all_continuum_event_guards_passed"] is False
    assert adaptive["gate_passed"] is False

    for name in ("directional", "cartesian"):
        summary = recomputed[name]
        assert summary["topology"]["all_continuum_event_margins_available"] is True
        assert summary["topology"]["all_continuum_event_guards_passed"] is False
        assert summary["gate_passed"] is False
    assert recomputed["directional"]["convergence"]["gate_passed"] is True
    assert all(recomputed["cartesian"]["convergence"]["gates"].values())

    rotation = recomputed["rotation"]
    assert rotation["all_rotation_topologies_match"] is False
    assert rotation["continuum_event_guard_applicable"] is True
    assert rotation["all_continuum_event_margins_available"] is True
    assert rotation["gate_passed"] is False

    decision = recomputed["decision"]
    assert decision["post_solve_residual_available"] is False
    assert decision["adaptive_directional_gate_passed"] is False
    assert decision["directional_metric_topology_gate_passed"] is False
    assert decision["cartesian_metric_topology_gate_passed"] is False
    assert decision["rotation_topology_gate_passed"] is False
    assert all(
        decision[name] is False
        for name in (
            "local_diagnostic_gates_passed",
            "tier_f_prerequisites_passed",
            "public_energy_admitted",
            "public_force_admitted",
            "opt_admitted",
            "hessian_freq_ts_irc_admitted",
            "md_admitted",
            "tier_v_mutual_polarization_admitted",
        )
    )
    return recomputed


def test_pyddx_adaptive_artifact_is_source_bound_recomputed_and_closed():
    artifact = _load(PRIMARY)
    _assert_artifact(artifact)
    assert_source_files_match_execution_commit(ROOT, artifact)


def test_pyddx_adaptive_cold_replay_is_scientifically_identical():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)
    _assert_artifact(replay)
    for key in (*MEASURED_KEYS, "measurement_sha256"):
        assert replay[key] == primary[key]
    assert replay["exact_command"] != primary["exact_command"]
    assert replay["runtime_seconds"] != primary["runtime_seconds"]


def test_pyddx_adaptive_raw_trace_rejects_energy_tampering():
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


def test_pyddx_adaptive_evidence_file_hashes_are_frozen():
    for name, expected in FILE_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILE_SHA256
