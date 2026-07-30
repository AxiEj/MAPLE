from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"


def test_atomic_map_preregistration_is_an_identity_audit_not_a_source_claim():
    preregistration = json.loads(
        (BENCHMARKS / "route2-v0-mace-mdp-atomic-map-prereg-v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert preregistration["status"] == "frozen-before-execution"
    assert preregistration["hard_constraints"] == {
        "mace_mdp_treated_as_an_energy_or_force_model": False,
        "model_weights_or_forward_equations_modified": False,
        "readout_hook_changes_model_output": False,
        "gto_width_or_density_projection_selected": False,
        "continuum_or_pcm_invoked": False,
        "experimental_solvation_labels_read": False,
        "post_training": False,
        "fine_tuning": False,
        "target_fit_or_calibration": False,
        "atomic_partition_rescaling_or_clipping": False,
        "legacy_public_route_changed": False,
    }
    assert "does not create an electronic density" in preregistration["claim_boundary"]
    assert (
        "does not yet map" in preregistration["protocol"]["candidate_moment_partition"]
    )


def test_atomic_map_preregistration_freezes_the_exact_reconstruction_gates():
    preregistration = json.loads(
        (BENCHMARKS / "route2-v0-mace-mdp-atomic-map-prereg-v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert preregistration["numerical_gates"] == {
        "public_direct_relative_max": 1.0e-12,
        "atomic_reconstruction_relative_max": 1.0e-12,
        "net_charge_absolute_max_e": 1.0e-12,
        "moment_identity_frobenius_max": 1.0e-12,
        "interpretation": preregistration["numerical_gates"]["interpretation"],
    }
    assert "Do not change the hook" in preregistration["decision_rule"]
    assert (
        "No atomic readout hook has been run"
        in preregistration["preexecution_disclosure"]
    )


def test_atomic_map_preregistration_binds_runner_model_and_implementation():
    preregistration = json.loads(
        (BENCHMARKS / "route2-v0-mace-mdp-atomic-map-prereg-v1.json").read_text(
            encoding="utf-8"
        )
    )
    runner = BENCHMARKS / "run_route2_v0_mace_mdp_atomic_map.py"
    contract = preregistration["execution_contract"]

    assert contract["source_sha256"] == {
        "docs/implicit-solvation/benchmarks/"
        "run_route2_v0_mace_mdp_atomic_map.py": hashlib.sha256(
            runner.read_bytes()
        ).hexdigest()
    }
    assert contract["model"]["sha256"] == (
        "126f8d1602549e6fa0df775c701a5119ddeb0e3738202af8e7aa736de6c2b692"
    )
    inputs = contract["input_sha256"]
    assert (
        inputs[
            "docs/implicit-solvation/benchmarks/"
            "route2-v0-mace-mdp-acetone-response-v1.json"
        ]
        == hashlib.sha256(
            (BENCHMARKS / "route2-v0-mace-mdp-acetone-response-v1.json").read_bytes()
        ).hexdigest()
    )
    assert "/mace/modules/models.py" in " ".join(inputs)
    assert "/mace/calculators/mace.py" in " ".join(inputs)
    assert "/mace/tools/torch_tools.py" in " ".join(inputs)
