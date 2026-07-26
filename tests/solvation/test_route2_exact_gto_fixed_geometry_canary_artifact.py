from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-mace-exact-gto-fixed-geometry-canary-v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_gto_fixed_geometry_canary_is_source_bound_and_claim_bounded():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact"] == (
        "route2-mace-exact-gto-fixed-geometry-canary-v1"
    )
    assert artifact["schema_version"] == 1
    assert artifact["execution_git_head"] == (
        "408ca3fce8f9831db3cb93882fc2f0e0dc439e85"
    )
    assert artifact["checkpoint"]["sha256"] == (
        "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
    )

    identity = artifact["scientific_identity"]
    assert identity["solute_response_model"] == "official MACE-POLAR-1-M"
    assert identity["solute_source"] == "point-multipole-l1"
    assert identity["continuum"] == "PCMSolver IEFPCM"
    assert identity["reaction_field_projector"] == "exact-gto-v1"
    assert identity["model_field_gauge"] == "atomic-center-mean-zero-v1"
    assert identity["strict_original_smd_equivalence"] is False

    protocol = artifact["diagnostic_protocol"]
    assert protocol["coordinate_derivative_requested"] is False
    assert protocol["scientific_pass_thresholds"] is None
    assert "fixed-geometry derivative algebra" in artifact["claim_boundary"]
    assert "sets no scientific pass threshold" in artifact["claim_boundary"]
    assert "exact-GTO coordinate or gauge derivative" in artifact[
        "claim_boundary"
    ]

    for relative, expected in artifact["source_files_sha256"].items():
        assert _sha256(ROOT / relative) == expected

    energy = artifact["public_energy_hartree"]
    np.testing.assert_allclose(
        energy["gas"] + energy["delta_g_solv"],
        energy["combined"],
        rtol=0.0,
        atol=2.0e-14,
    )

    experiment = artifact["experiment_comparison"]
    np.testing.assert_allclose(
        [
            experiment["predicted_kcal_mol"],
            experiment["experimental_kcal_mol"],
            experiment["experimental_uncertainty_kcal_mol"],
            experiment["signed_error_kcal_mol"],
            experiment["absolute_error_kcal_mol"],
        ],
        [
            -4.68356701455962,
            -3.8,
            0.6,
            -0.8835670145596204,
            0.8835670145596204,
        ],
        rtol=0.0,
        atol=2.0e-14,
    )
    assert "provenance only" in experiment["interpretation"]

    same_root = artifact["same_root"]
    assert same_root["scf_iterations"] == 13
    assert same_root["stored_unmixed_density_residual_inf_e"] <= 1.0e-5
    np.testing.assert_allclose(
        same_root["stored_unmixed_density_residual_inf_e"],
        same_root["fresh_unmixed_density_residual_inf_e"],
        rtol=0.0,
        atol=2.0e-16,
    )
    assert abs(same_root["root_density_monopole_sum_e"]) <= 1.0e-12
    assert abs(same_root["response_density_monopole_sum_e"]) <= 1.0e-12
    assert same_root["public_to_reopened_field_max_abs_ev"] == 0.0
    assert same_root["public_to_reopened_feature_max_abs"] == 0.0

    state = artifact["fixed_geometry_state"]
    assert state["density_shape"] == [10, 4]
    assert state["model_feature_shape"] == [10, 8]
    assert state["cavity_size"] == 516
    assert abs(state["physical_rhs_monopole_sum_ev"]) <= 1.0e-12

    diagnostics = artifact["diagnostics"]
    for name in (
        "continuum_feature_jvp_vjp",
        "mace_feature_density_jvp_vjp",
        "composed_residual_jvp_vjp",
    ):
        assert diagnostics[name]["absolute_error"] <= 1.0e-9

    adjoint = diagnostics["adjoint"]
    assert adjoint["relative_residual"] <= 1.0e-8
    assert adjoint["operator_applications"] == 7
    assert adjoint["residual_callback_count"] == 5

    finite_differences = diagnostics[
        "physical_rhs_directional_finite_difference"
    ]
    best = min(finite_differences, key=lambda item: item["relative_error"])
    assert best["step_density_coefficient"] == 1.0e-3
    assert best["relative_error"] <= 1.0e-6
    assert [
        item["relative_error"] for item in finite_differences
    ] == sorted(
        (item["relative_error"] for item in finite_differences)
    )

    warnings = artifact["pcmsolver_diagnostics"]
    assert warnings["public_native_stderr_warning_count"] == 0
    assert warnings["reopened_native_stderr_warning_count"] == 0
    assert warnings["public_pedra_warning_count"] == 1
    assert warnings["reopened_pedra_warning_count"] == 1
    expected_warning = (
        "** WARNING  ** A very poor tesselation has been chosen. "
        "It is valuable almost only for testing."
    )
    assert warnings["public_pedra_warnings"][0]["message"] == expected_warning
    assert warnings["reopened_pedra_warnings"][0]["message"] == expected_warning
    assert "do not select a different fixed cavity" in warnings[
        "warning_semantics"
    ]
