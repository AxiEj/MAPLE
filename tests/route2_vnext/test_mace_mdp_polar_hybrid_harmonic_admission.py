from __future__ import annotations

import ast
import json
from pathlib import Path

from maple.solvation.api.profiles import (
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_PROFILE_V1,
    PROFILE_REGISTRY,
)
from maple.solvation.api.scalar_registry import (
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1,
    SCALAR_REGISTRY,
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


def _preregistration() -> dict[str, object]:
    payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_harmonic_force_admission_is_honest_and_fail_closed_before_execution() -> None:
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
    assert profile.enabled is False
    assert profile.capabilities.enabled_tiers == ()
    assert profile.evidence_artifact_ids == ()
    assert scalar.enabled is False
    assert scalar.admitted_capabilities.enabled_tiers == ()
    assert scalar.evidence_artifact_ids == ()


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
