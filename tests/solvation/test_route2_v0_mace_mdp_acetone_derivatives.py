from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"


def test_mace_mdp_derivative_preregistration_stays_coefficient_only():
    preregistration = json.loads(
        (
            BENCHMARKS / "route2-v0-mace-mdp-acetone-derivatives-prereg-v1.json"
        ).read_text(encoding="utf-8")
    )

    assert preregistration["status"] == "frozen-before-execution"
    assert preregistration["hard_constraints"] == {
        "mace_mdp_treated_as_an_energy_or_force_model": False,
        "gto_density_or_pcm_source_constructed": False,
        "continuum_or_pcm_invoked": False,
        "experimental_solvation_labels_read": False,
        "post_training": False,
        "fine_tuning": False,
        "target_fit_or_calibration": False,
        "finite_difference_step_selected_after_execution": False,
        "derivative_rescaling_or_symmetrization": False,
        "legacy_public_route_changed": False,
    }
    assert "cannot certify a force" in preregistration["claim_boundary"]
    assert (
        "cannot infer an energy"
        in preregistration["candidate_identity"]["strict_role_boundary"]
    )


def test_mace_mdp_derivative_preregistration_freezes_full_cartesian_fd_gates():
    preregistration = json.loads(
        (
            BENCHMARKS / "route2-v0-mace-mdp-acetone-derivatives-prereg-v1.json"
        ).read_text(encoding="utf-8")
    )

    protocol = preregistration["finite_difference_protocol"]
    assert protocol["coordinate_step_angstrom"] == 1.0e-4
    assert "all 3N=30 Cartesian" in protocol["coordinates"]
    assert "(3,N,3)" in protocol["analytic_interface"]
    assert preregistration["numerical_gates"] == {
        "derivative_relative_frobenius_max": 1.0e-5,
        "polarizability_derivative_antisymmetry_max": 1.0e-10,
        "translation_relative_max": 1.0e-10,
        "interpretation": preregistration["numerical_gates"]["interpretation"],
    }
    assert (
        "Do not change the finite-difference step"
        in preregistration["numerical_gates"]["interpretation"]
    )


def test_mace_mdp_derivative_preregistration_binds_runner_and_dependencies():
    preregistration = json.loads(
        (
            BENCHMARKS / "route2-v0-mace-mdp-acetone-derivatives-prereg-v1.json"
        ).read_text(encoding="utf-8")
    )
    runner = BENCHMARKS / "run_route2_v0_mace_mdp_acetone_derivatives.py"
    contract = preregistration["execution_contract"]

    assert contract["source_sha256"] == {
        "docs/implicit-solvation/benchmarks/"
        "run_route2_v0_mace_mdp_acetone_derivatives.py": hashlib.sha256(
            runner.read_bytes()
        ).hexdigest()
    }
    inputs = contract["input_sha256"]
    for name in (
        "route2-v0-mace-mdp-acetone-response-v1.json",
        "route2-v0-mace-mdp-atomic-map-acetone-v1.json",
        "route2-v0-qeq-acetone-qm-field-v1.json",
    ):
        assert name in " ".join(inputs)
    assert "/mace/modules/utils.py" in " ".join(inputs)


def test_mace_mdp_derivative_artifact_admits_only_property_coefficients():
    artifact = json.loads(
        (BENCHMARKS / "route2-v0-mace-mdp-acetone-derivatives-v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert artifact["status"] == "pass"
    assert artifact["decision"]["verdict"] == (
        "admit-mace-mdp-property-coordinate-derivatives-only"
    )
    assert (
        artifact["hard_constraints"]["mace_mdp_treated_as_an_energy_or_force_model"]
        is False
    )
    assert artifact["hard_constraints"]["continuum_or_pcm_invoked"] is False
    assert artifact["hard_constraints"]["experimental_solvation_labels_read"] is False
    assert (
        artifact["finite_difference_protocol"]["central_difference_evaluations"] == 60
    )
    assert all(check["passes"] for check in artifact["numerical_checks"].values())
    assert artifact["numerical_checks"]["dipole_derivative_finite_difference"][
        "value"
    ] == pytest.approx(2.2778223475587386e-08)
    assert artifact["numerical_checks"]["polarizability_derivative_finite_difference"][
        "value"
    ] == pytest.approx(3.984714474887568e-08)
    assert (
        "does not create a model energy, force"
        in artifact["decision"]["admission_boundary"]
    )

    runner = BENCHMARKS / "run_route2_v0_mace_mdp_acetone_derivatives.py"
    assert artifact["source_files_sha256"] == {
        "docs/implicit-solvation/benchmarks/"
        "run_route2_v0_mace_mdp_acetone_derivatives.py": hashlib.sha256(
            runner.read_bytes()
        ).hexdigest()
    }
