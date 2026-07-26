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
TWO_STEP_RUNTIME_EQUIVALENCE_REFERENCE_HEAD = (
    "564c5ff492b587a17fd8222d1b965304df76a4c6"
)
CURRENT_RUNTIME_EQUIVALENCE_REFERENCE_HEAD = (
    "f7a6dea687a94c365848fc3fc11e7f0e2784ace9"
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

    assert baseline["schema_version"] == 4
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
    torsion_third_step_execution_heads = runtime_alignment[
        "torsion_third_step_execution_heads"
    ]
    assert torsion_third_step_execution_heads == force_canary_execution_heads
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
        (
            "2-acetoxyethyl acetate central C-C torsion third +/-0.25-degree "
            "pair with valid converged energy points"
        ),
    ]
    assert runtime_alignment[
        "coupling_diagnostic_evidence_confirmed_by_runtime_equivalent_canary"
    ] == [
        (
            "2-acetoxyethyl acetate pre-registered energy-only coupling "
            "diagnostic with no force evaluation and active-set-associated "
            "explicit continuum geometry response localization"
        )
    ]
    assert runtime_alignment[
        "force_evidence_failed_pre_registered_validation"
    ] == [
        (
            "2-acetoxyethyl acetate central C-C torsion "
            "1.0/0.5/0.25-degree smooth second-order asymptotic test"
        ),
    ]
    assert runtime_alignment["force_evidence_still_historical"] == [
        "2-acetoxyethyl acetate closed-loop panel",
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
        "and-three-step-torsion-canaries-with-pre-registered-asymptotic-order-fail"
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
        == TWO_STEP_RUNTIME_EQUIVALENCE_REFERENCE_HEAD
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


def test_qm_fidelity_baseline_recomputes_failed_three_step_torsion_test():
    baseline = _load_baseline()
    record = next(
        item
        for item in baseline["fixed_conformer_panel"]["records"]
        if item["name"] == "2-acetoxyethyl acetate"
    )
    evidence = record["source_evidence"]
    force_canary = evidence["runtime_equivalent_force_canary"]
    half_step = evidence["runtime_equivalent_torsion_single_step"]
    one_step = evidence["runtime_equivalent_torsion_two_step_refinement"]
    third_step = evidence[
        "runtime_equivalent_torsion_three_step_asymptotic_test"
    ]
    runtime_alignment = baseline["runtime_source_alignment"]

    assert third_step["status"] == "fail-predeclared-asymptotic-order-gates"
    assert third_step["scientific_points_valid"] is True
    assert third_step["pre_registered_validation_passed"] is False
    assert (
        third_step["git_head"]
        == evidence["torsion_third_step_execution_head"]
        == runtime_alignment["torsion_third_step_execution_heads"][
            "2-acetoxyethyl acetate"
        ]
        == "d72dfbaa3997b8d449720d07f3dd6dc076775eea"
    )
    assert (
        third_step["runtime_equivalence_reference_head"]
        == runtime_alignment["runtime_equivalence_reference_head"]
        == CURRENT_RUNTIME_EQUIVALENCE_REFERENCE_HEAD
    )
    assert third_step["profile"] == one_step["profile"] == half_step["profile"]
    assert third_step["coordinate"] == one_step["coordinate"] == half_step[
        "coordinate"
    ]
    assert third_step["steps_degrees"] == [1.0, 0.5, 0.25]

    minus = third_step["minus"]
    plus = third_step["plus"]
    for point in (minus, plus):
        assert point["converged"] is True
        assert point["forces_evaluated"] is False
        assert point["root_scf_iterations"] == 18
        assert point["energy_formula_closure_error_hartree"] <= 1.0e-12
        assert point["manifest_position_max_error_angstrom"] <= 1.0e-12

    fine_step_radians = math.radians(third_step["fine_step_degrees"])
    fine_finite_difference = -(
        plus["correction_energy_ev"] - minus["correction_energy_ev"]
    ) / (2.0 * fine_step_radians)
    assert third_step["fine_finite_difference_ev_per_radian"] == pytest.approx(
        fine_finite_difference,
        abs=1.0e-14,
    )
    analytic = third_step["analytic_generalized_force_ev_per_radian"]
    assert analytic == pytest.approx(
        one_step["analytic_generalized_force_ev_per_radian"],
        abs=1.0e-14,
    )
    assert analytic == pytest.approx(
        half_step["analytic_generalized_force_ev_per_radian"],
        abs=1.0e-14,
    )
    fine_absolute_error = abs(fine_finite_difference - analytic)
    assert third_step["fine_absolute_error_ev_per_radian"] == pytest.approx(
        fine_absolute_error,
        abs=1.0e-15,
    )
    fine_relative_error = fine_absolute_error / max(abs(analytic), 1.0e-12)
    assert third_step["fine_relative_error"] == pytest.approx(
        fine_relative_error,
        abs=1.0e-15,
    )
    assert fine_absolute_error <= third_step["thresholds"][
        "fine_maximum_absolute_error_ev_per_radian"
    ]
    assert fine_relative_error <= third_step["thresholds"][
        "fine_maximum_relative_error"
    ]

    finite_differences = [
        one_step["one_degree_finite_difference_ev_per_radian"],
        half_step["finite_difference_generalized_force_ev_per_radian"],
        fine_finite_difference,
    ]
    absolute_errors = [abs(value - analytic) for value in finite_differences]
    assert third_step["finite_differences_ev_per_radian"] == pytest.approx(
        finite_differences,
        abs=1.0e-14,
    )
    assert third_step["absolute_errors_ev_per_radian"] == pytest.approx(
        absolute_errors,
        abs=1.0e-15,
    )
    assert absolute_errors[2] < absolute_errors[1] < absolute_errors[0]

    coarse_to_middle_drift = abs(finite_differences[1] - finite_differences[0])
    middle_to_fine_drift = abs(finite_differences[2] - finite_differences[1])
    assert third_step["coarse_to_middle_drift_ev_per_radian"] == pytest.approx(
        coarse_to_middle_drift,
        abs=1.0e-15,
    )
    assert third_step["middle_to_fine_drift_ev_per_radian"] == pytest.approx(
        middle_to_fine_drift,
        abs=1.0e-15,
    )
    assert middle_to_fine_drift > coarse_to_middle_drift
    drift_ratio = coarse_to_middle_drift / middle_to_fine_drift
    assert third_step[
        "drift_ratio_coarse_to_middle_over_middle_to_fine"
    ] == pytest.approx(drift_ratio, abs=1.0e-15)
    observed_order = math.log2(drift_ratio)
    assert third_step["observed_central_difference_order"] == pytest.approx(
        observed_order,
        abs=1.0e-15,
    )
    assert observed_order < third_step["thresholds"][
        "minimum_observed_central_difference_order"
    ]

    second_order_richardson = finite_differences[2] + (
        finite_differences[2] - finite_differences[1]
    ) / 3.0
    assert third_step["second_order_richardson_ev_per_radian"] == pytest.approx(
        second_order_richardson,
        abs=1.0e-15,
    )
    richardson_analytic_error = abs(second_order_richardson - analytic)
    assert third_step[
        "richardson_analytic_error_ev_per_radian"
    ] == pytest.approx(richardson_analytic_error, abs=1.0e-15)

    expected_failed_gates = [
        "fine_drift_within_threshold",
        "finite_difference_drift_decreases",
        "observed_order_consistent_with_central_difference",
    ]
    recomputed_gate_results = {
        "fine_drift_within_threshold": (
            middle_to_fine_drift
            <= third_step["thresholds"][
                "maximum_middle_to_fine_drift_ev_per_radian"
            ]
        ),
        "finite_difference_drift_decreases": (
            middle_to_fine_drift < coarse_to_middle_drift
        ),
        "observed_order_consistent_with_central_difference": (
            third_step["thresholds"][
                "minimum_observed_central_difference_order"
            ]
            <= observed_order
            <= third_step["thresholds"][
                "maximum_observed_central_difference_order"
            ]
        ),
        "second_order_richardson_matches_analytic_threshold": (
            richardson_analytic_error
            <= third_step["thresholds"][
                "maximum_richardson_analytic_error_ev_per_radian"
            ]
        ),
    }
    for gate_name, passed in recomputed_gate_results.items():
        assert third_step["all_gates"][gate_name] is passed
    assert third_step["failed_pre_registered_gates"] == expected_failed_gates
    assert sorted(
        name for name, passed in third_step["all_gates"].items() if not passed
    ) == expected_failed_gates
    assert third_step["all_gates"][
        "absolute_error_decreases_across_all_three_steps"
    ] is True
    assert third_step["all_gates"][
        "second_order_richardson_matches_analytic_threshold"
    ] is True

    components = third_step["component_diagnostic"]
    component_rows = components["generalized_forces_ev_per_radian"]
    component_names = tuple(component_rows["1.0"])
    component_closures = []
    for step in ("1.0", "0.5", "0.25"):
        row = component_rows[step]
        assert tuple(row) == component_names
        component_closures.append(
            row["delta_g_solv"]
            - row["solute_polarization"]
            - row["pcm_polarization"]
            - row["cds"]
            - row["standard_state"]
        )
        component_closures.append(
            row["solvent_intrinsic_minus_gas"] - row["solute_polarization"]
        )
        component_closures.append(
            row["solvent_intrinsic_minus_gas"]
            - row["solvent_intrinsic_mace_energy_ev"]
            + row["gas_mace_energy_ev"]
        )
    recomputed_component_orders = {}
    for component_name in component_names:
        coarse_to_middle_signed = (
            component_rows["0.5"][component_name]
            - component_rows["1.0"][component_name]
        )
        middle_to_fine_signed = (
            component_rows["0.25"][component_name]
            - component_rows["0.5"][component_name]
        )
        stored_drifts = components["signed_refinement_drifts_ev_per_radian"][
            component_name
        ]
        assert stored_drifts["1.0_to_0.5"] == pytest.approx(
            coarse_to_middle_signed,
            abs=1.0e-15,
        )
        assert stored_drifts["0.5_to_0.25"] == pytest.approx(
            middle_to_fine_signed,
            abs=1.0e-15,
        )
        if coarse_to_middle_signed == middle_to_fine_signed == 0.0:
            recomputed_component_orders[component_name] = None
        else:
            recomputed_component_orders[component_name] = math.log2(
                abs(coarse_to_middle_signed / middle_to_fine_signed)
            )
        stored_order = components["observed_orders"][component_name]
        if stored_order is None:
            assert recomputed_component_orders[component_name] is None
        else:
            assert stored_order == pytest.approx(
                recomputed_component_orders[component_name],
                abs=2.0e-10,
            )
    component_orders = components["observed_orders"]
    assert 1.5 <= component_orders["gas_mace_energy_ev"] <= 2.5
    assert 1.5 <= component_orders["cds"] <= 2.5
    assert component_orders["solvent_intrinsic_mace_energy_ev"] < 1.0
    assert component_orders["pcm_polarization"] < 1.0
    assert component_orders["delta_g_solv"] == pytest.approx(
        observed_order,
        abs=2.0e-10,
    )
    maximum_component_closure = max(abs(value) for value in component_closures)
    assert components["closure_maximum_absolute_ev_per_radian"] == pytest.approx(
        maximum_component_closure,
        abs=1.0e-18,
    )
    assert "self-consistent electrostatic coupling block" in components["finding"]
    assert "does not distinguish" in components["causal_boundary"]

    assert third_step["pcmsolver_warning_count"] == 0
    assert third_step["legacy_label_noise_count"] == 0
    assert third_step["permitted_dependency_warning_count"] == 1
    assert third_step["runner_exit"] == 1
    assert "not an execution or serialization failure" in third_step[
        "runner_exit_interpretation"
    ]
    assert third_step["internal_new_energy_points_seconds"] > 0.0
    assert third_step["total_wall_seconds"] > 0.0
    assert third_step["used_for_timing_claim"] is False

    for hash_name in (
        "alignment_report_sha256",
        "analytic_force_canary_sha256",
        "current_half_step_result_sha256",
        "current_one_step_result_sha256",
        "minus_audit_sha256",
        "minus_manifest_sha256",
        "minus_output_sha256",
        "mol2_sha256",
        "plus_audit_sha256",
        "plus_manifest_sha256",
        "plus_output_sha256",
        "pre_result_lock_sha256",
        "runner_exitcode_sha256",
        "runner_sha256",
        "runner_stderr_sha256",
        "runner_stdout_sha256",
        "runner_time_sha256",
        "sha256",
    ):
        _assert_sha256(third_step[hash_name])
    for path_name, suffix in (
        ("path", ".json"),
        ("alignment_report_path", ".alignment.json"),
        ("analytic_force_canary_path", "2acetoxyethyl-acetate-force-d72dfba.json"),
        (
            "current_half_step_result_path",
            "2acetoxyethyl-acetate-torsion-cc-deg0p5-d72dfba.json",
        ),
        (
            "current_one_step_result_path",
            "2acetoxyethyl-acetate-torsion-cc-deg1p0-d72dfba.json",
        ),
        ("minus_audit_path", ".minus.audit.json"),
        ("minus_manifest_path", ".minus.manifest.json"),
        ("minus_output_path", ".minus.out"),
        ("plus_audit_path", ".plus.audit.json"),
        ("plus_manifest_path", ".plus.manifest.json"),
        ("plus_output_path", ".plus.out"),
        ("pre_result_lock_path", ".lock.json"),
        ("runner_exitcode_path", ".runner.exitcode"),
        ("runner_path", ".runner.py"),
        ("runner_stderr_path", ".runner.stderr.log"),
        ("runner_stdout_path", ".runner.stdout.log"),
        ("runner_time_path", ".runner.time.txt"),
    ):
        assert third_step[path_name].startswith(".omx/benchmarks/")
        assert third_step[path_name].endswith(suffix)
    assert third_step["analytic_force_canary_sha256"] == force_canary["sha256"]
    assert third_step["current_half_step_result_sha256"] == half_step["sha256"]
    assert third_step["current_one_step_result_sha256"] == one_step["sha256"]
    assert third_step["mol2_sha256"] == force_canary["mol2_sha256"]

    assert "Root cause remains unresolved" in third_step["interpretation"]
    for excluded_claim in (
        "Do not retune thresholds",
        "closed-loop",
        "second-molecule",
        "optimization",
        "NVE",
        "QM-force",
        "speed certification",
    ):
        assert excluded_claim in third_step["claim_boundary"]


def test_qm_fidelity_baseline_freezes_coupling_root_cause_diagnostic():
    baseline = _load_baseline()
    evidence = baseline["fixed_conformer_panel"]["records"][2]["source_evidence"]
    diagnostic = evidence["runtime_equivalent_coupling_root_cause_diagnostic"]

    assert diagnostic["status"] == "pass-specific-localization"
    assert diagnostic["scientific_diagnostic_valid"] is True
    assert diagnostic["specific_localization_passed"] is True
    assert diagnostic["failed_validity_gates"] == []
    assert diagnostic["runner_exit"] == 0
    assert diagnostic["evaluation_counts"] == {
        "active_set_models": 7,
        "continuum_scalar_solves": 26,
        "force_evaluations": 0,
        "independent_full_root_replicates": 1,
        "mace_polar_state_calls_after_replicate": 24,
        "new_finite_difference_steps": 0,
        "new_geometries": 0,
    }
    assert diagnostic["gates"] == {
        "active_set_mapping_exact": True,
        "all_derivative_decompositions_close": True,
        "all_energy_decompositions_close": True,
        "all_polarization_identities_close": True,
        "all_saved_state_energy_replays_reproducible": True,
        "all_saved_state_gradient_replays_reproducible": True,
        "all_saved_state_potential_replays_reproducible": True,
        "all_source_positions_exact": True,
        "exact_diagnostic_evaluation_counts": True,
        "exactly_one_independent_full_root_replicate": True,
        "independent_density_reproducibility": True,
        "independent_energy_reproducibility": True,
        "independent_full_root_replicate_converged_energy_only": True,
        "independent_position_reproducibility": True,
        "independent_reaction_gradient_reproducibility": True,
        "independent_reaction_potential_reproducibility": True,
        "no_legacy_primary_warning": True,
        "no_pcmsolver_warning": True,
    }
    assert len(diagnostic["gates"]) == 18

    replicate = diagnostic["independent_full_root_replicate"]
    assert replicate == {
        "source_case": "0.25_plus",
        "scf_iterations": 18,
        "maximum_component_energy_difference_ev": 2.220446049250313e-16,
        "maximum_density_difference_e": 2.220446049250313e-16,
        "maximum_reaction_potential_difference_ev": 1.887379141862766e-15,
        "maximum_reaction_gradient_difference_ev_per_angstrom": (
            7.216449660063518e-16
        ),
        "maximum_position_difference_angstrom": 0.0,
        "audit_sha256": (
            "4e5d56271c9c383626a5b9d9cfb5598ab145ca746b9f69d89e48179fce6e640a"
        ),
        "state_sha256": (
            "54f220bbc0c18bd76ba84055d67e2fecf64c3476db9cf2662491d06277c996ff"
        ),
        "manifest_sha256": (
            "7f4723dd6dc117806fc13b82ea03a988da204e7f58073c94bb40330be0f05261"
        ),
    }

    active_set = diagnostic["active_set"]
    assert active_set == {
        "changed_anywhere": True,
        "changed_in_fine_refinement_interval": True,
        "fine_interval_comparisons": {
            "0.5_minus_vs_0.25_minus": {
                "fine_only_count": 8,
                "middle_only_count": 11,
                "symmetric_difference_count": 19,
            },
            "0.5_plus_vs_0.25_plus": {
                "fine_only_count": 16,
                "middle_only_count": 11,
                "symmetric_difference_count": 27,
            },
        },
        "all_sphere_assignments_unique": True,
        "all_lebedev_assignments_unique": True,
        "maximum_sphere_assignment_error_bohr": 1.3322676295501878e-15,
        "maximum_lebedev_direction_match_error": 5.721958498152797e-16,
        "minimum_lebedev_second_neighbor_distance": 0.052522722703240585,
    }

    classification = diagnostic["classification"]
    assert classification == {
        "causal_boundary": (
            "This exact energy decomposition can distinguish density relaxation "
            "from the combined explicit reaction-map/cavity geometry response. "
            "The pyddx 0.8.0 API does not expose a provider-consistent "
            "operator-only versus cavity-only split, so no such finer causal "
            "claim is permitted. Active-set association is not causation and is "
            "authoritative only when the fine refinement interval changes and "
            "frozen density independently meets the locked dominance and "
            "non-smoothness rules."
        ),
        "derivative_noise_bound_ev_per_radian": 5.562154026943398e-11,
        "dominant_frozen_density_component": "pcm_fixed_center_density",
        "dominant_frozen_density_component_is_resolvable_nonsmooth": True,
        "dominant_top_level_component": "frozen_density",
        "dominant_top_level_component_is_resolvable_nonsmooth": True,
        "fine_interval_active_set_changed": True,
        "frozen_density_fine_drift_l1_fractions": {
            "cds": 0.0028438704903923926,
            "fixed_center_field_mace_minus_gas": 0.0023185421817203513,
            "pcm_fixed_center_density": 0.6756445828419462,
            "reaction_map_through_mace": 0.31919300448594107,
        },
        "label": "active-set-associated-explicit-continuum-geometry-response",
        "order_resolution_floor_ev_per_radian": 5.562154026943398e-10,
        "specific_localization": True,
        "top_level_fine_drift_l1_fractions": {
            "density_relaxation": 0.26225618959876046,
            "frozen_density": 0.7377438104012395,
        },
    }
    top_fractions = classification["top_level_fine_drift_l1_fractions"]
    assert sum(top_fractions.values()) == pytest.approx(1.0, abs=1.0e-15)
    frozen_fractions = classification["frozen_density_fine_drift_l1_fractions"]
    assert sum(frozen_fractions.values()) == pytest.approx(1.0, abs=1.0e-15)

    metrics = diagnostic["component_refinement_metrics"]
    observed_orders = {
        name: values["observed_order"]
        for name, values in metrics.items()
        if values["observed_order"] is not None
    }
    assert observed_orders == pytest.approx(
        {
            "cds": 2.000558489904326,
            "density_relaxation": -1.8174182049774574,
            "fixed_center_field_mace_minus_gas": 2.001828710877889,
            "frozen_density": 0.2906561113501319,
            "full_replayed": -0.033842598961002716,
            "pcm_fixed_center_density": 0.42167896313449477,
            "reaction_map_through_mace": 0.5596344306758558,
        },
        abs=1.0e-15,
    )
    assert metrics["standard_state"]["observed_order"] is None
    fine_drifts = {
        name: values["absolute_drift_0p5_to_0p25_ev_per_radian"]
        for name, values in metrics.items()
    }
    assert fine_drifts == pytest.approx(
        {
            "cds": 3.653506120513284e-07,
            "density_relaxation": 1.630272935439908e-05,
            "fixed_center_field_mace_minus_gas": 2.978619483623088e-07,
            "frozen_density": 4.586064371733464e-05,
            "full_replayed": 6.21633730717372e-05,
            "pcm_fixed_center_density": 8.679971985517876e-05,
            "reaction_map_through_mace": 4.1006564801532265e-05,
            "standard_state": 0.0,
        },
        abs=1.0e-15,
    )

    assert diagnostic["decomposition_closure"] == {
        "maximum_derivative_error_ev_per_radian": 6.505213034913027e-19,
        "maximum_energy_error_ev": 0.0,
        "maximum_polarization_identity_error_ev": 4.5852210917018965e-14,
    }
    assert diagnostic["saved_state_replay"] == {
        "center_pcm_energy_difference_ev": 1.213473765915296e-13,
        "center_reaction_gradient_maximum_difference_ev_per_angstrom": (
            6.304401445333951e-13
        ),
        "center_reaction_potential_maximum_difference_ev": 2.0571322423279526e-12,
        "maximum_energy_difference_ev": 6.161737786669619e-14,
        "maximum_reaction_gradient_difference_ev_per_angstrom": (
            5.056510765655275e-13
        ),
        "maximum_reaction_potential_difference_ev": 2.4496515926841766e-12,
    }
    assert diagnostic["warning_counts"] == {
        "pcmsolver": 0,
        "legacy_primary": 0,
        "python_dependency": 73,
        "logging": 1,
    }
    assert diagnostic["total_internal_seconds"] == pytest.approx(
        190.4744086849969,
        abs=1.0e-12,
    )
    assert diagnostic["outer_wall_seconds"] == pytest.approx(192.76, abs=1.0e-12)
    assert diagnostic["used_for_timing_claim"] is False

    expected_hashes = {
        "freeze_manifest_sha256": (
            "df88991e2d3b245d7e81cfc30855f7b0060dec7fee2756ec2cc2d643d18b1c92"
        ),
        "lock_sha256": (
            "c5b1717a9ec33be8ed930fd15198496655bf622ae00f2973b251f8383b346c48"
        ),
        "result_sha256": (
            "c9b05396016c4b96ba1c0973de2c9eaa11891e923e6fe6028b07336975e76798"
        ),
        "run_sentinel_sha256": (
            "c1ca796a8ec2d1253a384af5390cd450bd9047e547e49bc0ef077882b30166b5"
        ),
        "runner_sha256": (
            "71953e2fc3cd374a3dc005d7d16180d9c6a5461143d412233755319f2507a553"
        ),
    }
    assert {
        hash_name: diagnostic[hash_name] for hash_name in expected_hashes
    } == expected_hashes
    all_hashes = list(expected_hashes.values()) + [
        replicate[hash_name]
        for hash_name in ("audit_sha256", "manifest_sha256", "state_sha256")
    ]
    for digest in all_hashes:
        _assert_sha256(digest)

    evidence_root = (
        ".omx/benchmarks/route2-qm-fidelity-current-head-20260726/"
        "2acetoxyethyl-acetate-coupling-root-cause-71953e2f.raw/"
    )
    assert {
        "freeze_manifest_path": diagnostic["freeze_manifest_path"],
        "lock_path": diagnostic["lock_path"],
        "result_path": diagnostic["result_path"],
        "run_sentinel_path": diagnostic["run_sentinel_path"],
        "runner_path": diagnostic["runner_path"],
    } == {
        "freeze_manifest_path": evidence_root + "freeze-manifest.json",
        "lock_path": (
            evidence_root
            + "coupling_root_cause_diagnostic_2acetoxyethyl_acetate.lock.json"
        ),
        "result_path": (
            evidence_root
            + "coupling_root_cause_diagnostic_2acetoxyethyl_acetate.json"
        ),
        "run_sentinel_path": (
            evidence_root
            + "coupling_root_cause_diagnostic_2acetoxyethyl_acetate."
            "run-sentinel.json"
        ),
        "runner_path": (
            evidence_root
            + "run_coupling_root_cause_diagnostic_2acetoxyethyl_acetate.py"
        ),
    }
    assert (
        evidence["coupling_root_cause_diagnostic_execution_head"]
        == diagnostic["execution_git_head"]
        == "d72dfbaa3997b8d449720d07f3dd6dc076775eea"
    )
    assert diagnostic["runtime_equivalence_reference_head"] == (
        "bc0e58b8603a30bd3a0f1cb43b7a2db99e64faa6"
    )
    assert diagnostic["runtime_source_scope"] == "maple/"
    assert diagnostic["profile"] == (
        "smd-ddpcm-l15-n1202-gaff2-o-mace-kspace40-v1"
    )
    assert diagnostic["interpretation"] == (
        "Independent full-root and saved-state replay rule out numerical noise. "
        "The canonical cavity active set changes across both fine refinement "
        "interval sides. Frozen density carries 73.8% of the fine-drift L1 norm; "
        "within it, fixed-center-density PCM carries 67.6% and "
        "reaction-map-through-MACE carries 31.9%. Gas MACE and CDS remain locally "
        "second order. This validly associates the non-h2 drift with explicit "
        "continuum geometry/active-set response, but does not prove an "
        "operator-only or cavity-only cause."
    )
    assert diagnostic["claim_boundary"] == (
        "This one-conformer diagnostic can distinguish numerical "
        "reproducibility, density relaxation, and the combined explicit "
        "reaction-map/cavity geometry response for the exact Route-2 energy "
        "ledger. pyddx 0.8.0 does not expose a provider-consistent operator-only "
        "versus cavity-only split, so the result must not claim that finer "
        "causality. It does not certify global forces, a solution-phase PES, "
        "QM-force fidelity, energy conservation, or speed."
    )


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
