from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

from maple.solvation.api.profiles import (
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_PROFILE_V1,
    PROFILE_REGISTRY,
)
from maple.solvation.api.scalar_registry import (
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1,
    MACE_MDP_POLAR_HYBRID_HARMONIC_FORCE_ADMISSION_EVIDENCE_ID,
    SCALAR_REGISTRY,
)
from maple.solvation.experimental.mace_mdp_polar_harmonic import (
    ADMITTED_CONTINUUM_SETTINGS,
    ADMITTED_DEVICE,
    ADMITTED_DTYPE,
    ADMITTED_HYBRID_CONFIGURATION_SHA256,
    ADMITTED_HYBRID_PROVENANCE_SHA256,
)

ROOT = Path(__file__).parents[2]
PREREGISTRATION = (
    ROOT / "docs/route2/preregistrations/"
    "mace-mdp-polar-hybrid-harmonic-force-admission-v1.json"
)
RUNNER = (
    ROOT / "tools/route2_release/"
    "run_mace_mdp_polar_hybrid_harmonic_force_admission.py"
)
ADMISSION = (
    ROOT / "docs/route2/evidence/"
    "mace-mdp-polar-hybrid-harmonic-force-admission-replicated-4cf8db40.json"
)


def _preregistration() -> dict[str, object]:
    payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256_bytes(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(payload: object) -> str:
    return _sha256_bytes(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    )


def test_harmonic_force_admission_is_honest_and_narrowly_enabled() -> None:
    payload = _preregistration()
    assert payload["status"] == "frozen-before-full-admission-execution"
    assert payload["target_profile_id"] == (
        EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_PROFILE_V1
    )
    assert payload["target_scalar_id"] == (
        EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1
    )
    assert payload["target_capabilities"] == {
        "E": True,
        "F": True,
        "H": False,
        "V": False,
        "M": False,
    }
    prior = payload["prior_observation_boundary"]
    assert isinstance(prior, dict)
    assert "development-only" in str(prior["scope"])
    assert "no water" in str(prior["scope"])

    profile = PROFILE_REGISTRY[payload["target_profile_id"]]
    scalar = SCALAR_REGISTRY[payload["target_scalar_id"]]
    assert profile.enabled is True
    assert profile.capabilities.energy is True
    assert profile.capabilities.conservative_force is True
    assert profile.capabilities.hessian is False
    assert profile.capabilities.variational_functional is False
    assert profile.capabilities.molecular_dynamics is False
    assert profile.evidence_artifact_ids == (
        MACE_MDP_POLAR_HYBRID_HARMONIC_FORCE_ADMISSION_EVIDENCE_ID,
    )
    assert scalar.enabled is True
    assert scalar.admitted_capabilities == profile.capabilities
    assert scalar.evidence_artifact_ids == profile.evidence_artifact_ids


def test_harmonic_force_admission_freezes_required_force_symmetry_loop_gates() -> None:
    payload = _preregistration()
    runtime = payload["frozen_runtime_contract"]
    panel = payload["force_panel"]
    gates = payload["admission_gates"]
    assert isinstance(runtime, dict)
    assert isinstance(panel, dict)
    assert isinstance(gates, dict)
    assert runtime["force_coarse_step_angstrom"] == 5.0e-4
    assert runtime["force_fine_step_angstrom"] == 2.5e-4
    assert runtime["independent_force_step_angstrom"] == 1.25e-4
    assert runtime["force_maximum_local_error_estimate_ev_per_angstrom"] == 2.0e-4
    assert panel["maximum_benzene_h4_difference_ev_per_angstrom"] == 2.0e-4
    assert panel["maximum_rotation_energy_error_ev"] == 1.0e-7
    assert panel["maximum_rotation_force_relative_error"] == 2.0e-5
    assert panel["maximum_translation_force_relative_error"] == 2.0e-5
    assert panel["maximum_closed_loop_work_abs_ev"] == 2.0e-7
    assert gates["benzene_predecessor_failure_coordinate_passes_h_h2_h4"] is True
    assert gates["translation_energy_and_net_force_gates_pass"] is True
    assert gates["rotation_energy_and_force_covariance_gates_pass"] is True
    assert gates["closed_loop_work_plus_numerical_error_bound_gate_passes"] is True
    assert gates["two_independent_processes_same_measurement_sha256"] is True
    assert gates["accuracy_threshold"] is None
    for key, value in ADMITTED_CONTINUUM_SETTINGS:
        assert runtime[key] == value


def test_harmonic_force_admission_runner_is_parseable_and_binds_exact_profile() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source, filename=str(RUNNER))
    assert "RepositorySnapshot.capture" in source
    assert "measurement_sha256" in source
    assert "candidate-energy-force-gate-passed-awaiting-independent-replay" in source
    assert '"public_capability_admitted": False' in source
    assert "_component_convergence_record" in source
    assert "_closed_loop_record" in source
    assert "_proper_rotation" in source


def test_replicated_admission_binds_two_clean_identical_measurements() -> None:
    admission = json.loads(ADMISSION.read_text(encoding="utf-8"))
    assert admission["artifact_id"] == (
        MACE_MDP_POLAR_HYBRID_HARMONIC_FORCE_ADMISSION_EVIDENCE_ID
    )
    assert admission["status"] == (
        "admitted-experimental-electrostatic-energy-and-numerical-force"
    )
    assert admission["admitted_capabilities"] == {
        "E": True,
        "F": True,
        "H": False,
        "V": False,
        "M": False,
    }
    assert admission["replicate_measurements_equal"] is True
    assert admission["preregistration_sha256"] == _sha256_file(PREREGISTRATION)
    assert admission["admitted_runtime_binding"] == {
        "hybrid_configuration_sha256": ADMITTED_HYBRID_CONFIGURATION_SHA256,
        "hybrid_provenance_sha256": ADMITTED_HYBRID_PROVENANCE_SHA256,
        "dtype": ADMITTED_DTYPE,
        "device": ADMITTED_DEVICE,
        "cavity_radii_profile": "smd-water-coulomb-radii-v1",
    }
    assert len(admission["replicates"]) == 2

    measurements = []
    for replicate in admission["replicates"]:
        path = ROOT / replicate["relative_path"]
        assert _sha256_file(path) == replicate["sha256"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["artifact_id"] == replicate["artifact_id"]
        assert payload["working_tree_clean"] is True
        assert payload["execution_git_head"] == admission["execution_git_head"]
        assert payload["execution_git_tree"] == admission["execution_git_tree"]
        assert payload["status"] == (
            "candidate-energy-force-gate-passed-awaiting-independent-replay"
        )
        assert payload["capabilities"] == {
            "E": False,
            "F": False,
            "H": False,
            "V": False,
            "M": False,
        }
        assert payload["aggregate"]["all_execution_gates_passed"] is True
        measurement = {
            key: payload[key]
            for key in (
                "protocol",
                "benzene_gepol_regression",
                "water_force_symmetry_loop",
                "aggregate",
                "decision",
            )
        }
        assert _canonical_sha256(measurement) == payload["measurement_sha256"]
        measurements.append(measurement)
        for relative, expected in payload["source_files_sha256"].items():
            blob = subprocess.check_output(
                ["git", "show", f"{payload['execution_git_head']}:{relative}"],
                cwd=ROOT,
            )
            assert _sha256_bytes(blob) == expected

    assert measurements[0] == measurements[1]
    assert _canonical_sha256(measurements[0]) == admission["measurement_sha256"]
    assert admission["aggregate"] == measurements[0]["aggregate"]
