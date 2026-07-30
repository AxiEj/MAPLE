from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"


def test_gfn2_xtb_field_preregistration_freezes_a_no_runtime_qm_screen():
    preregistration = json.loads(
        (BENCHMARKS / "route2-v0-gfn2-xtb-acetone-field-prereg-v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert preregistration["status"] == "frozen-before-execution"
    assert preregistration["candidate_identity"]["electronic_method"] == "GFN2-xTB"
    assert preregistration["hard_constraints"] == {
        "gas_phase_gfn2_xtb_only": True,
        "xtb_builtin_solvation_disabled": True,
        "experimental_solvation_labels_read": False,
        "post_training": False,
        "fine_tuning": False,
        "target_fit_or_calibration": False,
        "response_rescaling_or_eigenvalue_clipping": False,
        "field_step_or_pair_distance_selection_after_execution": False,
        "legacy_public_route_changed": False,
        "no_runtime_qm_in_candidate": True,
    }
    protocol = preregistration["finite_field_protocol"]
    assert protocol["field_steps_au"] == [3.0e-4, 1.0e-3]
    assert protocol["pair_distance_bohr"] == 200.0
    assert protocol["point_charge_gamma"] == 1.0e8
    assert "asymptotically uniform" in protocol["construction"]
    assert "--cosmo" in " ".join(
        preregistration["candidate_identity"]["excluded_interfaces"]
    )


def test_gfn2_xtb_field_preregistration_keeps_the_qm_comparison_and_rejection_gates():
    preregistration = json.loads(
        (BENCHMARKS / "route2-v0-gfn2-xtb-acetone-field-prereg-v1.json").read_text(
            encoding="utf-8"
        )
    )

    numerical = preregistration["numerical_gates"]
    assert numerical["pair_inhomogeneity_proxy_max"] == 1.0e-3
    assert numerical["response_antisymmetry_relative_frobenius_max"] == 0.03
    assert numerical["energy_dipole_diagonal_relative_max"] == 0.05
    scientific = preregistration["scientific_falsification_gates"]
    assert scientific["relative_frobenius_mismatch_max"] == 0.2
    assert scientific["trace_ratio_min"] == 0.8
    assert scientific["trace_ratio_max"] == 1.2
    assert scientific["principal_value_relative_max"] == 0.3
    assert "Do not run an accuracy panel" in preregistration["decision_rule"]
