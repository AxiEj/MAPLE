from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"


def test_mace_mdp_preregistration_freezes_a_coefficient_only_screen():
    preregistration = json.loads(
        (BENCHMARKS / "route2-v0-mace-mdp-acetone-response-prereg-v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert preregistration["status"] == "frozen-before-execution"
    assert preregistration["candidate_identity"]["model_type"] == (
        "DipolePolarizabilityMACE"
    )
    assert preregistration["hard_constraints"] == {
        "mace_mdp_treated_as_an_energy_or_force_model": False,
        "continuum_or_pcm_invoked": False,
        "experimental_solvation_labels_read": False,
        "post_training": False,
        "fine_tuning": False,
        "target_fit_or_calibration": False,
        "response_rescaling": False,
        "eigenvalue_clipping_or_shift": False,
        "raw_tensor_symmetrization_to_pass": False,
        "field_or_threshold_selection_after_execution": False,
        "legacy_public_route_changed": False,
        "no_runtime_qm_in_candidate": True,
    }
    assert "not an energy or force" in preregistration["claim_boundary"]
    assert "cannot establish chemical accuracy" in preregistration["claim_boundary"]


def test_mace_mdp_preregistration_freezes_units_and_rejection_gates():
    preregistration = json.loads(
        (BENCHMARKS / "route2-v0-mace-mdp-acetone-response-prereg-v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert preregistration["unit_conversion"]["eangstrom2_per_volt_to_bohr3"] == (
        97.1736242922823
    )
    numerical = preregistration["numerical_gates"]
    assert numerical["raw_antisymmetry_relative_frobenius_max"] == 1.0e-10
    assert numerical["minimum_canonical_eigenvalue_bohr3_min"] == 0.0
    scientific = preregistration["scientific_falsification_gates"]
    assert scientific == {
        "relative_frobenius_mismatch_max": 0.2,
        "trace_ratio_min": 0.8,
        "trace_ratio_max": 1.2,
        "principal_value_relative_max": 0.3,
        "interpretation": scientific["interpretation"],
    }
    assert "Do not alter the tensor" in preregistration["decision_rule"]


def test_mace_mdp_preregistration_binds_the_runner_and_frozen_qm_input():
    preregistration = json.loads(
        (BENCHMARKS / "route2-v0-mace-mdp-acetone-response-prereg-v1.json").read_text(
            encoding="utf-8"
        )
    )
    contract = preregistration["execution_contract"]
    runner = BENCHMARKS / "run_route2_v0_mace_mdp_acetone_response.py"
    qm_reference = BENCHMARKS / "route2-v0-qeq-acetone-qm-field-v1.json"

    assert contract["source_sha256"] == {
        "docs/implicit-solvation/benchmarks/"
        "run_route2_v0_mace_mdp_acetone_response.py": hashlib.sha256(
            runner.read_bytes()
        ).hexdigest()
    }
    assert contract["input_sha256"] == {
        "docs/implicit-solvation/benchmarks/route2-v0-qeq-acetone-qm-field-v1.json": (
            hashlib.sha256(qm_reference.read_bytes()).hexdigest()
        )
    }
    assert contract["model"]["sha256"] == (
        "126f8d1602549e6fa0df775c701a5119ddeb0e3738202af8e7aa736de6c2b692"
    )


def test_mace_mdp_artifact_admits_only_frozen_response_coefficients():
    artifact = json.loads(
        (BENCHMARKS / "route2-v0-mace-mdp-acetone-response-v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert artifact["status"] == "pass"
    assert artifact["scientific_falsification"]["verdict"] == (
        "admit-frozen-mace-mdp-response-coefficients-only"
    )
    assert (
        artifact["hard_constraints"]["mace_mdp_treated_as_an_energy_or_force_model"]
        is False
    )
    assert artifact["hard_constraints"]["experimental_solvation_labels_read"] is False
    assert artifact["hard_constraints"]["continuum_or_pcm_invoked"] is False
    assert (
        artifact["numerical_checks"]["raw_polarizability_antisymmetry"]["value"]
        < 1.0e-10
    )
    assert artifact["numerical_checks"]["minimum_canonical_eigenvalue"]["value"] > 0.0
    checks = artifact["scientific_falsification"]["checks"]
    assert all(check["passes"] for check in checks.values())
    assert checks["relative_frobenius_mismatch"]["value"] < 0.2
    assert 0.8 < checks["trace_ratio"]["value"] < 1.2
    assert checks["principal_value_relative_max"]["value"] < 0.3
    assert (
        "does not permit an experimental-solvation accuracy panel"
        in artifact["scientific_falsification"]["admission_boundary"]
    )

    runner = BENCHMARKS / "run_route2_v0_mace_mdp_acetone_response.py"
    assert artifact["source_files_sha256"] == {
        "docs/implicit-solvation/benchmarks/"
        "run_route2_v0_mace_mdp_acetone_response.py": hashlib.sha256(
            runner.read_bytes()
        ).hexdigest()
    }
