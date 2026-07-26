from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = (
    REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/route2-qm-fidelity-v1.json"
)
GAS_CONSTANT_KCAL_MOL_K = 0.00198720425864083
HISTORICAL_RUNTIME_EQUIVALENCE_REFERENCE_HEAD = (
    "4c933135a58051b3ace5b8017cd4e64bf25a84af"
)
HALF_DEGREE_RUNTIME_EQUIVALENCE_REFERENCE_HEAD = (
    "3069ef7fef95134e4a3deac50a51267e376a87c2"
)
CURRENT_RUNTIME_EQUIVALENCE_REFERENCE_HEAD = (
    "564c5ff492b587a17fd8222d1b965304df76a4c6"
)


def _load_baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _free_energy(energies_kcal_mol: list[float], temperature_k: float) -> float:
    minimum = min(energies_kcal_mol)
    thermal = GAS_CONSTANT_KCAL_MOL_K * temperature_k
    return minimum - thermal * math.log(
        sum(math.exp(-(energy - minimum) / thermal) for energy in energies_kcal_mol)
    )


def _summary(errors: list[float]) -> dict[str, float]:
    return {
        "mae_kcal_mol": sum(errors) / len(errors),
        "rmse_kcal_mol": math.sqrt(
            sum(error * error for error in errors) / len(errors)
        ),
        "maximum_absolute_error_kcal_mol": max(errors),
    }


def _assert_sha256(value: str) -> None:
    assert len(value) == 64
    int(value, 16)


def _assert_git_sha(value: str) -> None:
    assert len(value) == 40
    int(value, 16)


def test_qm_fidelity_baseline_recomputes_fixed_conformer_metrics():
    baseline = _load_baseline()
    panel = baseline["fixed_conformer_panel"]
    records = panel["records"]

    assert baseline["schema_version"] == 3
    assert baseline["status"] == "frozen-bounded-pilot"
    assert baseline["route2_profiles"] == [
        "smd-ddpcm-l15-n1202-gaff2-o-mace-kspace40-v1",
        "smd-ddpcm-l15-n1202-v1",
    ]
    runtime_alignment = baseline["runtime_source_alignment"]
    latest_canary_head = runtime_alignment["latest_canary_execution_head"]
    runtime_reference_head = runtime_alignment["runtime_code_reference_git_head"]
    _assert_git_sha(latest_canary_head)
    _assert_git_sha(runtime_reference_head)
    runtime_equivalence_reference_head = runtime_alignment[
        "runtime_equivalence_reference_head"
    ]
    _assert_git_sha(runtime_equivalence_reference_head)
    assert runtime_equivalence_reference_head == CURRENT_RUNTIME_EQUIVALENCE_REFERENCE_HEAD
    assert runtime_alignment["runtime_source_scope"] == "maple/"
    assert runtime_alignment["runtime_code_unchanged_between_canary_heads"] is True
    canary_execution_heads = runtime_alignment["canary_execution_heads"]
    assert canary_execution_heads == {
        "2-acetoxyethyl acetate": "e34abc5bb01be0cfe65e536ca7ec90e7c0ed4e55",
        "acetone": "f3e989245189f742a03a61a19a848df54035c96f",
        "methanol": runtime_reference_head,
    }
    for execution_head in canary_execution_heads.values():
        _assert_git_sha(execution_head)
    assert runtime_alignment[
        "fixed_records_confirmed_by_runtime_equivalent_canary"
    ] == [
        "methanol",
        "acetone",
        "2-acetoxyethyl acetate",
    ]
    assert runtime_alignment["fixed_records_not_rerun"] == []
    force_canary_execution_heads = runtime_alignment["force_canary_execution_heads"]
    assert force_canary_execution_heads == {
        "2-acetoxyethyl acetate": latest_canary_head,
    }
    _assert_git_sha(force_canary_execution_heads["2-acetoxyethyl acetate"])
    force_finite_difference_execution_heads = runtime_alignment[
        "force_finite_difference_execution_heads"
    ]
    assert force_finite_difference_execution_heads == force_canary_execution_heads
    torsion_single_step_execution_heads = runtime_alignment[
        "torsion_single_step_execution_heads"
    ]
    assert torsion_single_step_execution_heads == force_canary_execution_heads
    torsion_second_step_execution_heads = runtime_alignment[
        "torsion_second_step_execution_heads"
    ]
    assert torsion_second_step_execution_heads == force_canary_execution_heads
    assert runtime_alignment[
        "force_evidence_confirmed_by_runtime_equivalent_canary"
    ] == [
        "2-acetoxyethyl acetate source-geometry analytic force",
        "2-acetoxyethyl acetate source-geometry one-component finite-difference pair",
        "2-acetoxyethyl acetate central C-C torsion one-step +/-0.5-degree pair",
        (
            "2-acetoxyethyl acetate central C-C torsion second +/-1.0-degree "
            "pair and two-step refinement trend"
        ),
    ]
    assert runtime_alignment["force_evidence_still_historical"] == [
        (
            "2-acetoxyethyl acetate third-step/asymptotic torsion convergence "
            "and closed-loop panel"
        ),
        "2-propoxyethanol center force",
    ]
    assert (
        runtime_alignment[
            "flexible_panel_runtime_source_equivalence_confirmed"
        ]
        is None
    )
    assert len(records) == panel["summary"]["record_count"] == 3
    assert len({record["compound_id"] for record in records}) == len(records)

    route2_qm_errors = []
    route2_experiment_errors = []
    for record in records:
        assert record["same_geometry"] is True
        assert record["training_membership_confirmed_excluded"] is False
        assert record["observed_timing"]["hardware_normalized"] is False
        assert record["observed_timing"]["speedup_certified"] is False
        evidence = record["source_evidence"]
        _assert_sha256(evidence["sha256"])
        _assert_git_sha(evidence["evidence_git_head"])
        assert evidence["status"] == "comparison-valid"

        route2 = record["route2"]
        qm = record["qm"]
        experiment = record["experiment"]
        differences = record["differences_kcal_mol"]
        route2_components = route2["components_kcal_mol"]
        qm_components = qm["components_kcal_mol"]

        assert sum(route2_components.values()) == pytest.approx(
            route2["delta_g_solv_kcal_mol"], abs=1.0e-9
        )
        assert sum(qm_components.values()) == pytest.approx(
            qm["delta_g_solv_kcal_mol"], abs=1.0e-9
        )
        assert differences["route2_minus_qm"] == pytest.approx(
            route2["delta_g_solv_kcal_mol"] - qm["delta_g_solv_kcal_mol"],
            abs=1.0e-12,
        )
        assert differences["route2_minus_experiment"] == pytest.approx(
            route2["delta_g_solv_kcal_mol"] - experiment["delta_g_solv_kcal_mol"],
            abs=1.0e-12,
        )
        assert differences["qm_minus_experiment"] == pytest.approx(
            qm["delta_g_solv_kcal_mol"] - experiment["delta_g_solv_kcal_mol"],
            abs=1.0e-12,
        )
        assert record["observed_timing"][
            "qm_over_route2_evaluation_ratio"
        ] == pytest.approx(
            qm["gas_plus_solution_seconds"] / route2["evaluation_seconds"],
            abs=1.0e-12,
        )

        component_differences = differences["route2_minus_qm_components"]
        for name in ("delta_e_solute", "u_pol", "g_cds"):
            assert component_differences[name] == pytest.approx(
                route2_components[name] - qm_components[name], abs=1.0e-12
            )
        assert component_differences["delta_g_solv"] == pytest.approx(
            differences["route2_minus_qm"], abs=1.0e-12
        )
        component_sum = sum(
            component_differences[name] for name in ("delta_e_solute", "u_pol", "g_cds")
        )
        assert component_sum == pytest.approx(
            differences["route2_minus_qm"], abs=1.0e-10
        )
        assert component_differences["g_cds"] == pytest.approx(0.0)
        assert (
            component_differences["delta_e_solute"] * component_differences["u_pol"]
            < 0.0
        )

        if record["name"] in canary_execution_heads:
            assert evidence["runtime_source_equivalence_confirmed"] is True
            assert evidence["alignment_status"] == (
                "confirmed-by-runtime-equivalent-energy-canary"
            )
            canary = evidence["runtime_equivalent_energy_canary"]
            assert canary["status"] == "pass"
            assert canary["git_head"] == canary_execution_heads[record["name"]]
            assert evidence["energy_canary_execution_head"] == canary["git_head"]
            assert canary["root_scf_iterations"] == route2["root_scf_iterations"]
            assert canary["evaluation_seconds"] > 0.0
            assert canary["used_for_timing_claim"] is False
            for hash_name in (
                "sha256",
                "force_reference_sha256",
                "mol2_sha256",
                "runner_sha256",
            ):
                _assert_sha256(canary[hash_name])
            assert canary[
                "absolute_difference_from_historical_correction_ev"
            ] == pytest.approx(
                abs(
                    canary["correction_energy_ev"]
                    - canary["historical_correction_energy_ev"]
                ),
                abs=1.0e-18,
            )
        else:
            assert evidence["runtime_source_equivalence_confirmed"] is None
            assert evidence["alignment_status"] == (
                "not-rerun-on-runtime-equivalence-reference"
            )
            assert (
                evidence["runtime_equivalence_reference_head"]
                == runtime_equivalence_reference_head
            )

        route2_qm_errors.append(abs(differences["route2_minus_qm"]))
        route2_experiment_errors.append(abs(differences["route2_minus_experiment"]))

    flexible_record = next(
        record for record in records if record["name"] == "2-acetoxyethyl acetate"
    )
    force_evidence = flexible_record["source_evidence"]
    assert force_evidence["force_alignment_status"] == (
        "confirmed-by-runtime-equivalent-analytic-force-cartesian-finite-difference-"
        "and-two-step-torsion-refinement-canaries"
    )
    assert force_evidence["force_canary_execution_head"] == latest_canary_head
    assert force_evidence["force_runtime_source_equivalence_confirmed"] is True
    force_canary = force_evidence["runtime_equivalent_force_canary"]
    assert force_canary["status"] == "pass"
    assert force_canary["git_head"] == latest_canary_head
    assert force_canary["root_scf_iterations"] == 18
    assert (
        force_canary["direct_runtime_equivalent_finite_difference_performed"]
        is True
    )
    assert force_canary["used_for_timing_claim"] is False
    assert force_canary["pcmsolver_warning_count"] == 0
    assert force_canary["legacy_label_noise_count"] == 0
    assert force_canary["evaluation_seconds"] > 0.0
    for hash_name in (
        "alignment_report_sha256",
        "historical_finite_difference_sha256",
        "historical_force_sha256",
        "mol2_sha256",
        "prepared_sha256",
        "runner_sha256",
        "runtime_equivalent_energy_canary_sha256",
        "sha256",
    ):
        _assert_sha256(force_canary[hash_name])
    assert (
        force_canary[
            "absolute_energy_difference_from_runtime_equivalent_energy_canary_ev"
        ]
        <= 1.0e-12
    )
    assert (
        force_canary["absolute_energy_difference_from_historical_force_canary_ev"]
        <= 1.0e-12
    )
    assert (
        force_canary["maximum_absolute_correction_force_difference_ev_per_angstrom"]
        <= 1.0e-10
    )
    assert (
        force_canary["maximum_absolute_total_force_difference_ev_per_angstrom"]
        <= 1.0e-10
    )
    assert (
        force_canary["adjoint"]["relative_residual"]
        <= force_canary["adjoint"]["relative_tolerance"]
    )
    assert force_canary[
        "cross_head_absolute_error_against_historical_finite_difference_ev_per_angstrom"
    ] == pytest.approx(3.084074560066874e-06)

    force_finite_difference = force_evidence[
        "runtime_equivalent_force_finite_difference"
    ]
    assert force_finite_difference["status"] == "pass"
    assert (
        force_finite_difference["git_head"]
        == force_finite_difference_execution_heads["2-acetoxyethyl acetate"]
    )
    assert (
        force_finite_difference["runtime_equivalence_reference_head"]
        == HISTORICAL_RUNTIME_EQUIVALENCE_REFERENCE_HEAD
    )
    assert force_finite_difference["used_for_timing_claim"] is False
    assert force_finite_difference["pcmsolver_warning_count"] == 0
    assert force_finite_difference["legacy_label_noise_count"] == 0
    assert force_finite_difference["step_angstrom"] == pytest.approx(5.0e-4)
    assert force_finite_difference["total_wall_seconds"] > 0.0
    for hash_name in (
        "analytic_force_canary_sha256",
        "alignment_report_sha256",
        "historical_finite_difference_sha256",
        "mol2_sha256",
        "runner_sha256",
        "sha256",
        "stderr_sha256",
        "time_sha256",
    ):
        _assert_sha256(force_finite_difference[hash_name])
    assert (
        force_finite_difference["analytic_force_canary_sha256"]
        == force_canary["sha256"]
    )
    assert (
        force_finite_difference["historical_finite_difference_sha256"]
        == force_canary["historical_finite_difference_sha256"]
    )
    assert force_finite_difference["mol2_sha256"] == force_canary["mol2_sha256"]
    assert force_finite_difference["selected_component"] == {
        "atom_index_one_based": 2,
        "atom_index_zero_based": 1,
        "axis": "z",
        "axis_index": 2,
    }
    minus = force_finite_difference["minus"]
    plus = force_finite_difference["plus"]
    assert minus["root_scf_iterations"] == plus["root_scf_iterations"] == 18
    assert minus["evaluation_seconds"] > 0.0
    assert plus["evaluation_seconds"] > 0.0
    finite_difference_force = -(
        plus["correction_energy_ev"] - minus["correction_energy_ev"]
    ) / (2.0 * force_finite_difference["step_angstrom"])
    assert force_finite_difference[
        "finite_difference_correction_force_ev_per_angstrom"
    ] == pytest.approx(finite_difference_force, abs=1.0e-12)
    absolute_force_error = abs(
        finite_difference_force
        - force_finite_difference[
            "analytic_correction_force_ev_per_angstrom"
        ]
    )
    assert force_finite_difference[
        "absolute_force_error_ev_per_angstrom"
    ] == pytest.approx(absolute_force_error, abs=1.0e-15)
    assert absolute_force_error <= 2.0e-4
    relative_force_error = absolute_force_error / max(
        abs(
            force_finite_difference[
                "analytic_correction_force_ev_per_angstrom"
            ]
        ),
        1.0e-12,
    )
    assert force_finite_difference["relative_force_error"] == pytest.approx(
        relative_force_error,
        abs=1.0e-15,
    )
    assert relative_force_error <= 5.0e-4
    assert abs(
        force_finite_difference[
            "finite_difference_force_difference_from_historical_ev_per_angstrom"
        ]
    ) <= 1.0e-10

    assert panel["summary"]["route2_vs_qm"] == pytest.approx(
        _summary(route2_qm_errors), abs=1.0e-12
    )
    assert panel["summary"]["route2_vs_experiment"] == pytest.approx(
        _summary(route2_experiment_errors), abs=1.0e-12
    )


def test_qm_fidelity_baseline_recomputes_single_step_torsion_canary():
    baseline = _load_baseline()
    record = next(
        item
        for item in baseline["fixed_conformer_panel"]["records"]
        if item["name"] == "2-acetoxyethyl acetate"
    )
    evidence = record["source_evidence"]
    force_canary = evidence["runtime_equivalent_force_canary"]
    torsion = evidence["runtime_equivalent_torsion_single_step"]
    runtime_alignment = baseline["runtime_source_alignment"]

    assert torsion["status"] == "pass"
    assert (
        torsion["git_head"]
        == evidence["torsion_single_step_execution_head"]
        == runtime_alignment["torsion_single_step_execution_heads"][
            "2-acetoxyethyl acetate"
        ]
        == "d72dfbaa3997b8d449720d07f3dd6dc076775eea"
    )
    assert (
        torsion["runtime_equivalence_reference_head"]
        == HALF_DEGREE_RUNTIME_EQUIVALENCE_REFERENCE_HEAD
    )
    assert torsion["profile"] == record["route2"]["profile"]
    assert torsion["coordinate"] == {
        "axis_bond_indices_one_based": [5, 6],
        "axis_direction_index_zero_based": 5,
        "axis_origin_index_zero_based": 4,
        "definition": (
            "Rigidly rotate the downstream fragment about the central C3-C4 "
            "single-bond axis using the positive right-hand rule."
        ),
        "rotated_fragment_indices_one_based": [
            6,
            7,
            8,
            9,
            10,
            16,
            17,
            18,
            19,
            20,
        ],
        "rotated_fragment_indices_zero_based": [
            5,
            6,
            7,
            8,
            9,
            15,
            16,
            17,
            18,
            19,
        ],
    }
    assert torsion["step_degrees"] == pytest.approx(0.5)
    assert torsion["step_radians"] == pytest.approx(math.radians(0.5), abs=1.0e-18)

    minus = torsion["minus"]
    plus = torsion["plus"]
    for point in (minus, plus):
        assert point["converged"] is True
        assert point["forces_evaluated"] is False
        assert point["root_scf_iterations"] == 18
        assert point["energy_formula_closure_error_hartree"] <= 1.0e-12
        assert point["manifest_position_max_error_angstrom"] <= 1.0e-12

    finite_difference = -(
        plus["correction_energy_ev"] - minus["correction_energy_ev"]
    ) / (2.0 * torsion["step_radians"])
    assert torsion[
        "finite_difference_generalized_force_ev_per_radian"
    ] == pytest.approx(finite_difference, abs=1.0e-14)
    absolute_error = abs(
        finite_difference - torsion["analytic_generalized_force_ev_per_radian"]
    )
    assert torsion[
        "absolute_generalized_force_error_ev_per_radian"
    ] == pytest.approx(absolute_error, abs=1.0e-15)
    relative_error = absolute_error / max(
        abs(torsion["analytic_generalized_force_ev_per_radian"]),
        1.0e-12,
    )
    assert torsion["relative_generalized_force_error"] == pytest.approx(
        relative_error,
        abs=1.0e-15,
    )
    assert absolute_error <= 2.0e-4
    assert relative_error <= 5.0e-3
    assert torsion["symmetric_second_difference_ev"] == pytest.approx(
        plus["correction_energy_ev"]
        + minus["correction_energy_ev"]
        - 2.0 * torsion["center_correction_energy_ev"],
        abs=1.0e-15,
    )
    assert torsion["center_correction_energy_ev"] == pytest.approx(
        force_canary["correction_energy_ev"],
        abs=1.0e-15,
    )

    assert torsion["pcmsolver_warning_count"] == 0
    assert torsion["legacy_label_noise_count"] == 0
    assert torsion["total_wall_seconds"] > 0.0
    assert torsion["used_for_timing_claim"] is False
    for hash_name in (
        "alignment_report_sha256",
        "analytic_force_canary_sha256",
        "finalizer_sha256",
        "historical_torsion_sha256",
        "lock_sha256",
        "mol2_sha256",
        "runner_sha256",
        "sha256",
    ):
        _assert_sha256(torsion[hash_name])
    for path_name, suffix in (
        ("path", ".json"),
        ("alignment_report_path", ".alignment.json"),
        ("analytic_force_canary_path", "2acetoxyethyl-acetate-force-d72dfba.json"),
        ("historical_torsion_path", "torsion_force_fd_2acetoxyethyl_acetate.json"),
        ("lock_path", ".lock.json"),
        ("runner_path", ".runner.py"),
        ("finalizer_path", ".finalizer.py"),
    ):
        assert torsion[path_name].startswith(".omx/benchmarks/")
        assert torsion[path_name].endswith(suffix)
    assert torsion["analytic_force_canary_sha256"] == force_canary["sha256"]
    assert torsion["mol2_sha256"] == force_canary["mol2_sha256"]
    assert (
        torsion["historical_torsion_sha256"]
        == "ada069fd1371baab4f1aa9ae57c26a2ca9a12cdb0bf9c9d84438961657599b0a"
    )
    assert abs(
        torsion[
            "analytic_generalized_force_difference_from_historical_ev_per_radian"
        ]
    ) <= 1.0e-10
    assert abs(
        torsion[
            "finite_difference_generalized_force_difference_from_historical_ev_per_radian"
        ]
    ) <= 1.0e-10
    assert abs(torsion["minus_energy_difference_from_historical_ev"]) <= 1.0e-12
    assert abs(torsion["plus_energy_difference_from_historical_ev"]) <= 1.0e-12

    recovery = torsion["recovery"]
    assert recovery["original_runner_exit"] == 1
    assert "NumPy boolean" in recovery["original_runner_failure"]
    assert recovery["scientific_evaluations_rerun"] is False
    assert "step-size convergence" in torsion["claim_boundary"]
    assert "closed-loop conservativity" in torsion["claim_boundary"]
    assert "second molecule" in torsion["claim_boundary"]
    assert "NVE conservation" in torsion["claim_boundary"]


def test_qm_fidelity_baseline_recomputes_two_step_torsion_refinement():
    baseline = _load_baseline()
    record = next(
        item
        for item in baseline["fixed_conformer_panel"]["records"]
        if item["name"] == "2-acetoxyethyl acetate"
    )
    evidence = record["source_evidence"]
    force_canary = evidence["runtime_equivalent_force_canary"]
    half_step = evidence["runtime_equivalent_torsion_single_step"]
    refinement = evidence["runtime_equivalent_torsion_two_step_refinement"]
    runtime_alignment = baseline["runtime_source_alignment"]

    assert refinement["status"] == "pass"
    assert (
        refinement["git_head"]
        == evidence["torsion_second_step_execution_head"]
        == runtime_alignment["torsion_second_step_execution_heads"][
            "2-acetoxyethyl acetate"
        ]
        == "d72dfbaa3997b8d449720d07f3dd6dc076775eea"
    )
    assert (
        refinement["runtime_equivalence_reference_head"]
        == runtime_alignment["runtime_equivalence_reference_head"]
        == CURRENT_RUNTIME_EQUIVALENCE_REFERENCE_HEAD
    )
    assert refinement["profile"] == record["route2"]["profile"] == half_step["profile"]
    assert refinement["coordinate"] == half_step["coordinate"]
    assert refinement["new_step_degrees"] == pytest.approx(1.0)
    assert refinement["reference_refined_step_degrees"] == pytest.approx(
        half_step["step_degrees"]
    )

    minus = refinement["minus"]
    plus = refinement["plus"]
    for point in (minus, plus):
        assert point["converged"] is True
        assert point["forces_evaluated"] is False
        assert point["root_scf_iterations"] == 18
        assert point["energy_formula_closure_error_hartree"] <= 1.0e-12
        assert point["manifest_position_max_error_angstrom"] <= 1.0e-12

    one_degree_step_radians = math.radians(refinement["new_step_degrees"])
    one_degree_finite_difference = -(
        plus["correction_energy_ev"] - minus["correction_energy_ev"]
    ) / (2.0 * one_degree_step_radians)
    assert refinement["one_degree_finite_difference_ev_per_radian"] == pytest.approx(
        one_degree_finite_difference,
        abs=1.0e-14,
    )
    analytic = refinement["analytic_generalized_force_ev_per_radian"]
    assert analytic == pytest.approx(
        half_step["analytic_generalized_force_ev_per_radian"],
        abs=1.0e-14,
    )
    one_degree_absolute_error = abs(one_degree_finite_difference - analytic)
    assert refinement["one_degree_absolute_error_ev_per_radian"] == pytest.approx(
        one_degree_absolute_error,
        abs=1.0e-15,
    )
    one_degree_relative_error = one_degree_absolute_error / max(abs(analytic), 1.0e-12)
    assert refinement["one_degree_relative_error"] == pytest.approx(
        one_degree_relative_error,
        abs=1.0e-15,
    )
    assert one_degree_absolute_error <= 2.0e-4
    assert one_degree_relative_error <= 5.0e-3

    assert refinement["half_degree_finite_difference_ev_per_radian"] == pytest.approx(
        half_step["finite_difference_generalized_force_ev_per_radian"],
        abs=1.0e-14,
    )
    assert refinement["half_degree_absolute_error_ev_per_radian"] == pytest.approx(
        half_step["absolute_generalized_force_error_ev_per_radian"],
        abs=1.0e-15,
    )
    assert refinement["half_degree_relative_error"] == pytest.approx(
        half_step["relative_generalized_force_error"],
        abs=1.0e-15,
    )
    absolute_error_reduction = (
        one_degree_absolute_error
        - half_step["absolute_generalized_force_error_ev_per_radian"]
    )
    assert refinement["absolute_error_reduction_ev_per_radian"] == pytest.approx(
        absolute_error_reduction,
        abs=1.0e-15,
    )
    assert absolute_error_reduction > 0.0
    refined_over_coarse = (
        half_step["absolute_generalized_force_error_ev_per_radian"]
        / one_degree_absolute_error
    )
    assert refinement["absolute_error_ratio_refined_over_coarse"] == pytest.approx(
        refined_over_coarse,
        abs=1.0e-15,
    )
    step_drift = abs(
        one_degree_finite_difference
        - half_step["finite_difference_generalized_force_ev_per_radian"]
    )
    assert refinement["finite_difference_step_drift_ev_per_radian"] == pytest.approx(
        step_drift,
        abs=1.0e-15,
    )
    assert step_drift <= 1.0e-4

    assert refinement["center_correction_energy_ev"] == pytest.approx(
        force_canary["correction_energy_ev"],
        abs=1.0e-15,
    )
    assert refinement["pcmsolver_warning_count"] == 0
    assert refinement["legacy_label_noise_count"] == 0
    assert refinement["permitted_dependency_warning_count"] == 1
    assert refinement["internal_new_energy_points_seconds"] > 0.0
    assert refinement["total_wall_seconds"] > 0.0
    assert refinement["used_for_timing_claim"] is False

    for hash_name in (
        "alignment_report_sha256",
        "analytic_force_canary_sha256",
        "current_half_step_result_sha256",
        "historical_torsion_sha256",
        "lock_sha256",
        "minus_audit_sha256",
        "minus_manifest_sha256",
        "mol2_sha256",
        "plus_audit_sha256",
        "plus_manifest_sha256",
        "runner_sha256",
        "runner_stderr_sha256",
        "runner_stdout_sha256",
        "runner_time_sha256",
        "sha256",
    ):
        _assert_sha256(refinement[hash_name])
    for path_name, suffix in (
        ("path", ".json"),
        ("alignment_report_path", ".alignment.json"),
        ("analytic_force_canary_path", "2acetoxyethyl-acetate-force-d72dfba.json"),
        (
            "current_half_step_result_path",
            "2acetoxyethyl-acetate-torsion-cc-deg0p5-d72dfba.json",
        ),
        ("historical_torsion_path", "torsion_force_fd_2acetoxyethyl_acetate.json"),
        ("lock_path", ".lock.json"),
        ("minus_audit_path", ".minus.audit.json"),
        ("minus_manifest_path", ".minus.manifest.json"),
        ("plus_audit_path", ".plus.audit.json"),
        ("plus_manifest_path", ".plus.manifest.json"),
        ("runner_path", ".runner.py"),
        ("runner_stderr_path", ".runner.stderr.log"),
        ("runner_stdout_path", ".runner.stdout.log"),
        ("runner_time_path", ".runner.time.txt"),
    ):
        assert refinement[path_name].startswith(".omx/benchmarks/")
        assert refinement[path_name].endswith(suffix)

    assert refinement["analytic_force_canary_sha256"] == force_canary["sha256"]
    assert refinement["current_half_step_result_sha256"] == half_step["sha256"]
    assert refinement["historical_torsion_sha256"] == half_step[
        "historical_torsion_sha256"
    ]
    assert refinement["mol2_sha256"] == force_canary["mol2_sha256"]
    assert abs(
        refinement[
            "analytic_generalized_force_difference_from_historical_ev_per_radian"
        ]
    ) <= 1.0e-10
    assert abs(
        refinement[
            "finite_difference_generalized_force_difference_from_historical_ev_per_radian"
        ]
    ) <= 1.0e-10
    assert abs(refinement["minus_energy_difference_from_historical_ev"]) <= 1.0e-12
    assert abs(refinement["plus_energy_difference_from_historical_ev"]) <= 1.0e-12

    assert "bounded two-step refinement trend" in refinement["interpretation"]
    assert "not an estimate of asymptotic convergence order" in refinement[
        "interpretation"
    ]
    for excluded_claim in (
        "third-step asymptotic regime",
        "closed-loop conservativity",
        "second molecule",
        "NVE conservation",
    ):
        assert excluded_claim in refinement["claim_boundary"]


def test_qm_fidelity_baseline_recomputes_bounded_electronic_ensemble():
    baseline = _load_baseline()
    panel = baseline["flexible_conformer_panel"]
    records = panel["records"]
    summary = panel["summary"]
    temperature = panel["temperature_k"]

    assert panel["geometry_count"] == len(records) == 4
    assert panel["geometry_generation"].startswith(
        "One original source geometry plus three ETKDGv3/MMFF"
    )
    assert panel["predeclared_primary_gate"] == {
        "metric": "mean_absolute_route2_minus_qm_kcal_mol",
        "status": "pass",
        "threshold_kcal_mol": 1.0,
    }
    evidence = panel["source_evidence"]
    _assert_sha256(evidence["sha256"])
    assert evidence["status"] == "complete-valid"
    assert (
        evidence["runtime_equivalence_reference_head"]
        == HISTORICAL_RUNTIME_EQUIVALENCE_REFERENCE_HEAD
    )
    assert evidence["runtime_source_equivalence_confirmed"] is None
    assert evidence["alignment_status"] == (
        "not-rerun-on-runtime-equivalence-reference"
    )
    for evidence_head in evidence["evidence_git_heads"]:
        _assert_git_sha(evidence_head)

    signed_errors = []
    for record in records:
        route2 = record["route2"]
        qm = record["qm"]
        _assert_sha256(record["mol2_sha256"])
        assert sum(route2["components_kcal_mol"].values()) == pytest.approx(
            route2["delta_g_solv_kcal_mol"], abs=1.0e-9
        )
        assert sum(qm["components_kcal_mol"].values()) == pytest.approx(
            qm["delta_g_solv_kcal_mol"], abs=1.0e-9
        )
        assert (
            route2["aqueous_energy_kcal_mol"] - route2["gas_energy_kcal_mol"]
        ) == pytest.approx(route2["delta_g_solv_kcal_mol"], abs=1.0e-9)
        assert (
            qm["aqueous_energy_kcal_mol"] - qm["gas_energy_kcal_mol"]
        ) == pytest.approx(qm["delta_g_solv_kcal_mol"], abs=1.0e-9)
        signed_error = route2["delta_g_solv_kcal_mol"] - qm["delta_g_solv_kcal_mol"]
        assert record["route2_minus_qm_kcal_mol"] == pytest.approx(
            signed_error, abs=1.0e-12
        )
        assert record["absolute_route2_minus_qm_kcal_mol"] == pytest.approx(
            abs(signed_error), abs=1.0e-12
        )
        component_differences = record["route2_minus_qm_components_kcal_mol"]
        for name in ("delta_e_solute", "u_pol", "g_cds"):
            assert component_differences[name] == pytest.approx(
                route2["components_kcal_mol"][name] - qm["components_kcal_mol"][name],
                abs=1.0e-12,
            )
        assert sum(component_differences.values()) == pytest.approx(
            signed_error, abs=1.0e-10
        )
        assert component_differences["g_cds"] == pytest.approx(0.0)
        assert (
            component_differences["delta_e_solute"] * component_differences["u_pol"] < 0
        )
        signed_errors.append(signed_error)

    route2_ensemble = _free_energy(
        [record["route2"]["aqueous_energy_kcal_mol"] for record in records],
        temperature,
    ) - _free_energy(
        [record["route2"]["gas_energy_kcal_mol"] for record in records],
        temperature,
    )
    qm_ensemble = _free_energy(
        [record["qm"]["aqueous_energy_kcal_mol"] for record in records],
        temperature,
    ) - _free_energy(
        [record["qm"]["gas_energy_kcal_mol"] for record in records],
        temperature,
    )

    assert summary["mean_absolute_route2_minus_qm_kcal_mol"] == pytest.approx(
        sum(abs(error) for error in signed_errors) / len(signed_errors),
        abs=1.0e-12,
    )
    assert summary["maximum_absolute_route2_minus_qm_kcal_mol"] == pytest.approx(
        max(abs(error) for error in signed_errors)
    )
    assert summary["mean_signed_route2_minus_qm_kcal_mol"] == pytest.approx(
        sum(signed_errors) / len(signed_errors), abs=1.0e-12
    )
    assert summary["route2_minus_qm_bias_range_kcal_mol"] == pytest.approx(
        max(signed_errors) - min(signed_errors), abs=1.0e-12
    )
    assert summary["route2_bounded_electronic_ensemble_kcal_mol"] == pytest.approx(
        route2_ensemble, abs=1.0e-9
    )
    assert summary["qm_bounded_electronic_ensemble_kcal_mol"] == pytest.approx(
        qm_ensemble, abs=1.0e-9
    )
    assert summary["absolute_ensemble_difference_kcal_mol"] == pytest.approx(
        abs(route2_ensemble - qm_ensemble), abs=1.0e-9
    )
    assert summary[
        "route2_ensemble_absolute_experimental_error_kcal_mol"
    ] == pytest.approx(
        abs(route2_ensemble - summary["experimental_kcal_mol"]), abs=1.0e-9
    )

    gates = panel["certification_gates"]
    primary_gate = panel["predeclared_primary_gate"]
    primary_gate_passed = (
        summary[primary_gate["metric"]] <= primary_gate["threshold_kcal_mol"]
    )
    assert (primary_gate["status"] == "pass") is primary_gate_passed
    assert gates["all_four_locked_geometries_complete"] is True
    assert gates["predeclared_four_geometry_mae_pass"] is primary_gate_passed
    assert gates["complete_conformer_thermochemistry"] is False
    assert gates["solution_phase_geometry_relaxation"] is False
    assert gates["training_membership_confirmed_excluded"] is False


def test_qm_fidelity_baseline_preserves_reference_and_claim_boundaries():
    baseline = _load_baseline()
    diagnostic = baseline["acetone_same_cavity_ao_reference_diagnostic"]

    assert "not CI inputs" in baseline["source_artifact_scope"]
    assert baseline["qm_reference"]["comparison_level"] == (
        "chemical-level same-geometry reference, not operator-identical"
    )
    assert diagnostic["status"] == "acetone-specific-invalid-reference"
    assert diagnostic["production_change_required"] is False
    assert "Do not enlarge Route-2 radii" in diagnostic["forbidden_inference"]
    assert "does not prove" in diagnostic["scope_boundary"]
    invalid_case = diagnostic["numerically_completed_but_physically_invalid_case"]
    assert invalid_case["status"] == "completed-but-physically-invalid"
    assert invalid_case["dd_solvation_energy_hartree"] == pytest.approx(
        -114.16103810006106
    )
    _assert_sha256(diagnostic["source_evidence"]["sha256"])
    assert diagnostic["source_evidence"]["status"] == "source-diagnostic-valid"
    assert (
        diagnostic["source_evidence"]["runtime_equivalence_reference_head"]
        == HISTORICAL_RUNTIME_EQUIVALENCE_REFERENCE_HEAD
    )
    assert diagnostic["source_evidence"]["runtime_source_equivalence_confirmed"] is None
    assert "No vibrational thermochemistry" in baseline["claim_boundary"]
    assert "hardware-normalized speedup" in baseline["claim_boundary"]
