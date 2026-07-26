from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-mace-local-field-thermodynamic-canary-v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_thermodynamic_canary_is_source_bound_and_claim_bounded():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact"] == (
        "route2-mace-local-field-thermodynamic-canary-v1"
    )
    assert artifact["schema_version"] == 1
    assert artifact["execution_git_head"] == (
        "c8fe19b35b2ea0c9631891d3b0f999fd2e0fe703"
    )
    assert artifact["checkpoint"]["sha256"] == (
        "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
    )
    identity = artifact["scientific_identity"]
    assert identity["solute_response_model"] == "official MACE-POLAR-1-M"
    assert identity["solute_source"] == "point-multipole-l1"
    assert identity["continuum"] == "pyddx ddPCM"
    assert identity["reaction_field_projector"] == "local-jet"
    assert identity["field_interface"] == "local-potential-gradient-v1"
    assert identity["cds_present_in_public_energy"] is True
    assert identity["cds_used_in_diagnostics"] is False
    assert identity["strict_original_smd_equivalence"] is False
    assert artifact["diagnostic_protocol"]["pass_thresholds"] is None
    assert "sets no scientific pass threshold" in artifact["claim_boundary"]
    assert "exact-GTO energy-only response" in artifact["claim_boundary"]

    for relative, expected in artifact["source_files_sha256"].items():
        assert _sha256(ROOT / relative) == expected

    energy = artifact["public_energy_hartree"]
    np.testing.assert_allclose(
        energy["gas"] + energy["delta_g_solv"],
        energy["combined"],
        rtol=0.0,
        atol=2.0e-14,
    )

    same_root = artifact["same_root"]
    assert same_root["scf_iterations"] == 9
    assert same_root["unmixed_density_residual_inf_e"] <= 2.0e-11
    assert same_root["half_coupling_identity_error_ev"] <= 1.0e-12
    assert abs(same_root["root_density_monopole_sum_e"]) <= 1.0e-12
    assert abs(same_root["response_density_monopole_sum_e"]) <= 1.0e-12

    conjugacy = artifact["diagnostics"]["energy_density_conjugacy"]
    assert conjugacy["field_interface"] == "local-potential-gradient-v1"
    assert conjugacy["lower_defect_candidate"] == "intrinsic"
    assert (
        0.6
        < conjugacy["intrinsic_candidate"][
            "maximum_relative_block_error"
        ]
        < 0.7
    )
    assert (
        conjugacy["coupled_candidate"]["maximum_relative_block_error"]
        > 0.99
    )

    reciprocity = artifact["diagnostics"]["response_reciprocity"]
    assert len(reciprocity) == 3
    assert min(item["absolute_error_ev"] for item in reciprocity) > 2.0e-3
    for item in reciprocity:
        dot_test = item["linearization_adjoint_dot_test"]
        assert dot_test["passed"] is True
        assert (
            dot_test["absolute_error_ev"]
            <= dot_test["tolerance_ev"]
            == 1.0e-10
        )

    stability = artifact["diagnostics"]["response_stability"]
    assert stability["dimension"] == 12
    assert stability["negative_eigenvalue_count"] == 5
    assert stability["near_zero_eigenvalue_count"] == 4
    assert stability["positive_eigenvalue_count"] == 3
    assert stability["passivity_violation_ev"] > 8.0e-3
    assert stability["antisymmetric_frobenius_norm_ev"] > 1.0e-2

    loops = artifact["diagnostics"]["field_loop_work_refinement"]
    assert len(loops) == 2
    area_normalized = [
        item["closed_loop_work_ev"]
        / item["potential_step_scale_ev"] ** 2
        for item in loops
    ]
    np.testing.assert_allclose(
        area_normalized,
        [-0.004161031549973927, -0.00416091134811114],
        rtol=1.0e-12,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        area_normalized[0],
        area_normalized[1],
        rtol=5.0e-5,
        atol=0.0,
    )
    assert artifact["diagnostics"][
        "field_loop_fresh_model_evaluations"
    ] == 6

    timing = artifact["runtime"]["timing_seconds"]
    assert timing["model_load"] > 0.0
    assert timing["public_coupled_root_and_energy"] > 0.0
    assert timing["energy_density_conjugacy"] > 0.0
    assert timing["reciprocity_three_pairs"] > 0.0
    assert timing["stability_twelve_jvps"] > 0.0
    assert timing["total"] > 0.0
