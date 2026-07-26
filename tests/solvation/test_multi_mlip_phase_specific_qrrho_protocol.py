from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
PROTOCOL_PATH = BENCHMARK_DIR / "multi_mlip_phase_specific_qrrho_protocol.json"
SOURCE_MANIFEST_PATH = (
    BENCHMARK_DIR / "multi_mlip_discrete_conformer_source_manifest.json"
)
FAILURE_ARTIFACT_PATH = (
    BENCHMARK_DIR
    / "route1-multi-mlip-phase-specific-selected-minimum-rrho-v6-failure-2026-07-25.json"
)
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from source_compatibility import validate_frozen_source


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_content_sha256(value: dict) -> str:
    payload = {key: item for key, item in value.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def test_qrrho_protocol_freezes_label_blind_source_and_models():
    protocol = _load(PROTOCOL_PATH)
    manifest = _load(SOURCE_MANIFEST_PATH)
    source = protocol["source"]

    assert protocol["status"] == (
        "preregistered-before-energy-evaluation-and-label-scoring"
    )
    assert protocol["protocol_id"] == (
        "maple-route1-multi-mlip-phase-specific-selected-minimum-rrho-v8"
    )
    assert protocol["amendment_from"] == (
        "maple-route1-multi-mlip-phase-specific-selected-minimum-rrho-v7"
    )
    assert "three label-blind v7" in protocol["amendment_reason"]
    assert "missing isinstance guard" in protocol["amendment_reason"]
    assert "before aggregate sealing" in protocol["amendment_reason"]
    assert protocol["source_partition"] == "development"
    assert _sha256(SOURCE_MANIFEST_PATH) == source["manifest_file_sha256"]
    assert manifest["content_sha256"] == source["manifest_content_sha256"]
    assert (
        manifest["label_boundary"]["prepared_state_manifest_contains_labels"] is False
    )
    assert source["labels_forbidden_until"].startswith("The complete 18-model-case")

    selected = []
    used = set()
    for target in source["selection_target_state_counts"]:
        eligible = [
            case
            for case in manifest["cases"]
            if len(case["atomic_numbers"]) <= source["maximum_atom_count"]
            and case["compound_id"] not in used
        ]
        chosen = min(
            eligible,
            key=lambda case: (
                abs(math.log(case["state_count"]) - math.log(target)),
                len(case["atomic_numbers"]),
                case["compound_id"],
            ),
        )
        selected.append(chosen)
        used.add(chosen["compound_id"])
    assert [case["compound_id"] for case in selected] == [
        case["compound_id"] for case in protocol["cases"]
    ]
    assert [case["compound_id"] for case in selected] == [
        "mobley_1952272",
        "mobley_1717215",
        "mobley_8118832",
        "mobley_4463913",
        "mobley_5759258",
        "mobley_1017962",
    ]
    assert len(selected) == source["expected_case_count"] == 6
    assert (
        sum(case["state_count"] for case in selected)
        == source["expected_source_state_count"]
        == 221
    )
    assert (
        [model["name"] for model in protocol["models"]]
        == source["expected_models"]
        == ["maceoff23m", "aimnet2", "ani2x"]
    )

    manifest_models = {model["name"]: model for model in manifest["models"]}
    for model in protocol["models"]:
        assert (
            model["checkpoint_sha256"]
            == manifest_models[model["name"]]["checkpoint_sha256"]
        )
        assert model["maple_output_unit"] == "hartree"


def test_qrrho_protocol_freezes_every_state_file_and_default_route1_endpoint():
    protocol = _load(PROTOCOL_PATH)
    manifest = _load(SOURCE_MANIFEST_PATH)
    manifest_cases = {case["compound_id"]: case for case in manifest["cases"]}

    assert protocol["solvation"] == {
        "charge_method": "am1bcc",
        "charge_mode": "fixed",
        "charge_geometry": "keep",
        "charge_source": "frozen-am1bcc-json",
        "recharge_during_optimization": False,
        "method": "gb",
        "provider": "openmm",
        "polar_model": "obc2",
        "profile": "obc2-mbondi2",
        "nonpolar_model": "ace",
        "platform": "Reference",
        "solvent": "water",
        "same_charge_vector_for_every_geometry_and_mlip_of_a_case": True,
    }
    assert manifest["solvent_endpoint"] == {
        "charge_method": "am1bcc",
        "fixed_charge": True,
        "nonpolar_model": "ace",
        "polar_model": "obc2",
        "same_correction_for_every_gas_mlip": True,
    }
    assert protocol["route"]["gas_phase_mm_energy"] is False
    assert protocol["route"]["hydration_label_residual"] is False
    assert protocol["route"]["mlip_retraining"] is False
    assert protocol["route"]["charge_refitting"] is False

    for case in protocol["cases"]:
        source_case = manifest_cases[case["compound_id"]]
        assert case["source_state_count"] == source_case["state_count"]
        assert case["state_file"] == source_case["state_file"]
        assert case["state_file_sha256"] == source_case["state_file_sha256"]
        assert (
            _sha256(REPOSITORY_ROOT / case["state_file"]) == case["state_file_sha256"]
        )


def test_qrrho_protocol_freezes_the_scientific_implementation():
    protocol = _load(PROTOCOL_PATH)
    implementation = protocol["implementation_freeze"]

    assert implementation["required_versions"] == {
        "ase": "3.27.0",
        "numpy": "2.4.6",
        "openmm": "8.5.2",
        "torch": "2.12.0+cu130",
    }
    assert implementation["required_python"] == "3.11.14"
    assert len(implementation["source_files"]) == 21
    compatibility_modes = []
    for source in implementation["source_files"]:
        compatibility_modes.append(
            validate_frozen_source(
                REPOSITORY_ROOT,
                source["path"],
                source["sha256"],
            )["mode"]
        )
    assert "documented-postexecution-production-safety-change" in compatibility_modes
    assert (
        "docs/implicit-solvation/benchmarks/" "run_multi_mlip_phase_specific_qrrho.py"
    ) in {source["path"] for source in implementation["source_files"]}
    assert (
        "docs/implicit-solvation/benchmarks/" "score_multi_mlip_phase_specific_qrrho.py"
    ) in {source["path"] for source in implementation["source_files"]}
    assert "versioned protocol amendment" in implementation["amendment_policy"]
    assert "label access invalidate" in implementation["amendment_policy"]


def test_qrrho_protocol_pins_exact_external_am1bcc_vectors():
    protocol = _load(PROTOCOL_PATH)
    manifest = _load(SOURCE_MANIFEST_PATH)
    manifest_cases = {case["compound_id"]: case for case in manifest["cases"]}

    assert protocol["solvation"]["charge_source"] == "frozen-am1bcc-json"
    for case in protocol["cases"]:
        source_case = manifest_cases[case["compound_id"]]
        assert case["mol2_file"] == source_case["source"]["mol2"]
        assert case["mol2_file_sha256"] == source_case["source"]["mol2_sha256"]
        assert _sha256(REPOSITORY_ROOT / case["mol2_file"]) == (
            case["mol2_file_sha256"]
        )
        charge_path = REPOSITORY_ROOT / case["charge_file"]
        charge_record = _load(charge_path)
        assert _sha256(charge_path) == case["charge_file_sha256"]
        assert charge_record["charge_method"] == "am1bcc"
        assert charge_record["mol2_sha256"] == case["mol2_file_sha256"]
        assert (
            hashlib.sha256(
                (
                    json.dumps(
                        [float(value) for value in charge_record["charges_e"]],
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    + "\n"
                ).encode("utf-8")
            ).hexdigest()
            == case["charge_vector_sha256"]
        )

    execution = protocol["execution"]
    assert execution["device"] == "cuda:0"
    assert execution["batch_size"] == 16
    assert execution["source_energy_repeat_count"] == 2
    assert execution["maximum_repeat_relative_energy_difference_kcal_mol"] == (0.0002)
    assert execution["record_granularity"] == (
        "one atomically sealed JSON file per model-case"
    )
    assert "shared v8 scientific schema validator" in execution["resume_policy"]
    assert execution["energy_runner_never_reads_experimental_labels"] is True
    provenance = execution["repeat_tolerance_provenance"]
    provenance_path = REPOSITORY_ROOT / provenance["source_artifact"]
    assert _sha256(provenance_path) == provenance["source_artifact_file_sha256"]
    assert provenance["maximum_prior_label_free_repeat_difference_kcal_mol"] == (
        0.00010377602839106714
    )
    assert "no experimental labels" in provenance["selection"]
    sealing = protocol["evidence_sealing"]
    assert sealing["strict_validator_shared_by"] == [
        "resume",
        "seal",
        "post-seal scorer",
    ]
    assert sealing["absolute_parent_or_symlink_escaping_paths_allowed"] is False
    assert sealing["label_access_requires_durable_evidence_validation"] is True
    assert "fixed repository-relative" in sealing["durable_storage"]


def test_qrrho_protocol_requires_complete_phase_forces_and_hessians():
    protocol = _load(PROTOCOL_PATH)
    sampling = protocol["phase_specific_sampling"]
    optimization = protocol["optimization"]
    hessian = protocol["hessian_and_stationary_point"]
    dedup = protocol["optimized_minimum_deduplication"]
    gates = protocol["energy_artifact_gates"]

    assert sampling["evaluate_every_frozen_source_state"] is True
    assert sampling["maximum_seed_count_per_phase_model_case"] == 3
    assert sampling["experimental_labels_used"] is False
    assert sampling["cross_phase_seed_substitution"] is False
    assert sampling["seed_budget_convergence_diagnostic_available"] is False
    assert "does not infer convergence" in sampling["seed_budget_convergence_reason"]
    assert "seed_budget_prefix_diagnostics" not in sampling
    assert "0.125 angstrom" in sampling["seed_diversity"]
    assert optimization["periodic_boundary_conditions"] is False
    assert optimization["constraints_allowed"] is False
    assert optimization["fmax_eV_per_angstrom"] == 0.01
    assert optimization["max_steps"] == 500
    assert "W_AM1-BCC/OBC-II/ACE" in optimization["solution_force"]
    assert hessian["method"] == (
        "central finite difference of the complete phase force"
    )
    assert hessian["cartesian_displacement_angstrom"] == 0.002
    assert hessian["mass_weighted_translation_rotation_projection"] is True
    assert hessian["linearity_moment_ratio_threshold"] == 0.01
    assert "exactly 3N-5 modes" in hessian["vibrational_mode_selection"]
    assert "invalidates the minimum" in hessian["imaginary_mode_policy"]
    assert (
        "No selected negative frequency is replaced" in hessian["imaginary_mode_policy"]
    )
    assert hessian["imaginary_frequency_cutoff_cm1"] == 0.0
    assert hessian["maximum_rigid_mode_leakage_cm1"] == 1.0
    assert hessian["every_unique_minimum_must_be_valid"] is True
    assert (
        hessian[
            "maximum_absolute_raw_hessian_asymmetry_hartree_per_angstrom2"
        ]
        == 0.02
    )
    assert hessian["maximum_relative_raw_asymmetry_frobenius"] == 0.002
    assert "scale-aware Frobenius" in hessian["numerical_quality_gate"]
    recovery = hessian["imaginary_mode_recovery"]
    assert recovery["directions"] == [-1, 1]
    assert recovery["maximum_atom_displacement_angstrom"] == 0.5
    assert recovery["maximum_recovery_cycles"] == 1
    assert recovery["minimum_energy_lowering_kcal_mol"] == 0.001
    qualification = hessian["label_blind_numerical_qualification"]
    qualification_path = REPOSITORY_ROOT / qualification["artifact"]
    qualification_artifact = _load(qualification_path)
    assert _sha256(qualification_path) == qualification["artifact_file_sha256"]
    assert (
        qualification_artifact["content_sha256"]
        == qualification["artifact_content_sha256"]
        == _artifact_content_sha256(qualification_artifact)
    )
    assert qualification["experimental_labels_read"] is False
    assert (
        qualification_artifact["label_boundary"]["experimental_labels_read"]
        is False
    )
    assert hessian["displacement_sensitivity"] == {
        "status": "preregistered-label-blind-preflight",
        "compound_ids": ["mobley_1952272", "mobley_4463913"],
        "model_names": ["maceoff23m", "aimnet2", "ani2x"],
        "phases": ["gas", "solution"],
        "unique_minimum_indices": [0],
        "cartesian_displacements_angstrom": [0.001, 0.002, 0.004],
        "reference_displacement_angstrom": 0.002,
        "maximum_selected_mode_rms_difference_cm1": 25.0,
        "scope_interpretation": (
            "The rigid one-state nitromethane anchor checks plumbing; the flexible "
            "19-state 1-acetoxyethyl acetate case checks a low-frequency "
            "multi-conformer basin. This remains a targeted numerical preflight, "
            "not a proof for every flexible molecule."
        ),
        "failure_action": (
            "A failed displacement sensitivity gate invalidates the affected "
            "model-case and blocks the six-case pilot."
        ),
    }
    assert "Retain the model-case in the denominator" in hessian["failure_action"]
    assert dedup["duplicate_arrival_count_used_as_degeneracy"] is False
    assert dedup["symmetry_degeneracy_inferred"] is False
    assert gates["expected_model_case_count"] == 18
    assert gates["all_model_cases_must_complete"] is True
    assert gates["at_least_one_valid_unique_minimum_per_phase_model_case"] is True
    assert gates["every_selected_seed_must_resolve_as_valid_or_duplicate"] is True
    assert gates["every_unique_minimum_must_be_valid"] is True
    assert gates["prediction_artifact_must_not_contain_experimental_labels"] is True
    assert gates["failed_case_or_model_substitution"] is False
    assert gates["every_model_case_must_pass_shared_semantic_validator"] is True
    assert gates["all_hessians_must_be_repository_contained_and_content_hash_valid"]
    assert gates["durable_raw_evidence_must_be_sealed_before_scoring"] is True


def test_qrrho_protocol_freezes_thermochemistry_and_standard_state():
    protocol = _load(PROTOCOL_PATH)
    thermo = protocol["thermochemistry"]
    estimator = protocol["ensemble_estimator"]

    assert thermo["temperature_kelvin"] == 298.15
    assert thermo["primary"] == {
        "ilowfreq": 0,
        "name": (
            "Local harmonic RRHO primary; low-frequency rotor interpolation is "
            "excluded from the primary to avoid unquantified double counting "
            "with the explicit selected-minimum sum."
        ),
        "omega0_cm1": 100.0,
        "alpha": 4,
        "nu_floor_cm1": 1.0,
    }
    assert [
        (variant["id"], variant["ilowfreq"], variant["omega0_cm1"])
        for variant in thermo["predeclared_sensitivity_variants"]
    ] == [
        ("grimme-entropy-only", 2, 100.0),
        ("qrrho-otlyotov-omega100", 3, 100.0),
        ("qrrho-omega50", 3, 50.0),
        ("qrrho-omega150", 3, 150.0),
    ]
    assert thermo["sensitivity_variants_are_not_primary"] is True
    assert (
        "double count torsional entropy" in thermo["torsional_double_counting_boundary"]
    )
    assert "Q_rot,j/sigma_rot,j" in thermo["rotational_partition_policy"]
    assert [case["rotational_symmetry_number"] for case in protocol["cases"]] == [1] * 6
    assert all(
        "internal methyl" in case["rotational_symmetry_number_basis"]
        for case in protocol["cases"]
    )
    assert thermo["translational_terms_in_partition_sums"] is False
    assert (
        thermo["standard_state"]["transfer_convention"]
        == "1 mol/L gas to 1 mol/L solution, matching FreeSolv"
    )
    assert (
        thermo["standard_state"]["standalone_1atm_to_1M_correction_added_to_prediction"]
        is False
    )
    assert estimator["pure_gas_mlip_as_hydration_baseline"] is False
    assert estimator["estimator_identity"] == "selected-minimum local-RRHO surrogate"
    assert estimator["not_exact_cartesian_endpoint_ratio"] is True
    assert "G_tilde_s,m-G_tilde_g,m" in estimator["hydration_prediction"]
    assert "global orientation-group volume" in estimator["global_factor_cancellation"]


def test_qrrho_protocol_preregisters_falsification_not_promotion():
    protocol = _load(PROTOCOL_PATH)
    scoring = protocol["post_seal_scoring"]
    boundary = protocol["claim_boundary"]

    assert scoring["bootstrap"] == {
        "resamples": 10000,
        "random_seed": 20260725,
        "confidence_interval": "two-sided percentile 95%",
        "resample_same_case_indices_for_baseline_and_qrrho": True,
    }
    assert scoring["script"] == (
        "docs/implicit-solvation/benchmarks/" "score_multi_mlip_phase_specific_qrrho.py"
    )
    assert _sha256(REPOSITORY_ROOT / scoring["script"]) == scoring["script_sha256"]
    assert scoring["positive_model_gate"] == {
        "minimum_point_improvement_kcal_mol": 0.2,
        "bootstrap_lower_bound_must_exceed_kcal_mol": 0.0,
    }
    assert "At least two of three MLIPs" in scoring["route_accuracy_gate"]
    assert (
        scoring["cross_model_direction_gate"][
            "minimum_cases_with_same_nonzero_sign_in_at_least_two_models"
        ]
        == 4
    )
    assert (
        scoring["low_frequency_stability_gate"]["maximum_unstable_cases_per_model"] == 1
    )
    assert scoring["low_frequency_stability_gate"]["all_three_models_must_pass"] is True
    assert boundary["validated_absolute_hydration_free_energy_method"] is False
    assert boundary["public_solvfe_eligible"] is False
    assert boundary["conformer_completeness_established"] is False
    assert boundary["basin_measures_included"] is False
    assert boundary["absolute_hydration_free_energy_estimator_under_test"] is False
    assert boundary["selected_minimum_local_rrho_surrogate_under_test"] is True
    assert boundary["rotational_symmetry_numbers_frozen_per_case"] is True
    assert boundary["posthoc_case_model_seed_or_cutoff_selection_allowed"] is False


def test_v6_failure_is_durable_label_blind_and_cannot_be_rescored():
    protocol = _load(PROTOCOL_PATH)
    artifact = _load(FAILURE_ARTIFACT_PATH)

    assert artifact["content_sha256"] == _artifact_content_sha256(artifact)
    assert artifact["protocol_id"] == (
        "maple-route1-multi-mlip-phase-specific-selected-minimum-rrho-v6"
    )
    assert artifact["status"] == "protocol-falsified-before-label-scoring"
    assert artifact["failure_record"]["status"] == "failure"
    assert "Hessian and stationary-point validation" in (
        artifact["failure_record"]["failure"]["reason"]
    )
    assert set(artifact["label_boundary"].values()) == {False}
    assert artifact["decision"] == {
        "v6_complete_execution_allowed": False,
        "v6_resume_or_record_replacement_allowed": False,
        "v6_sealing_allowed": False,
        "v6_scoring_allowed": False,
        "accuracy_or_public_solvfe_claim_allowed": False,
        "next_step": (
            "A separately versioned, label-blind numerical-Hessian "
            "qualification protocol must be frozen before any new scientific "
            "pilot; v6 must not be silently relaxed or rerun."
        ),
    }

    for evidence in artifact["linked_raw_evidence"].values():
        path = REPOSITORY_ROOT / evidence["path"]
        assert path.is_file()
        assert _sha256(path) == evidence["file_sha256"]

    gate = artifact["frozen_gates"][
        "maximum_hessian_asymmetry_hartree_per_angstrom2"
    ]
    for phase in ("gas", "solution"):
        rows = artifact["post_failure_label_blind_diagnostic"]["phases"][phase]
        reference = next(
            row
            for row in rows
            if row["cartesian_displacement_angstrom"] == 0.002
        )
        assert reference["maximum_raw_asymmetry_hartree_per_angstrom2"] > (
            1000.0 * gate
        )
        assert all(row["raw_asymmetry_gate_passed"] is False for row in rows)
        assert all(
            row["valid_stationary_point_after_symmetrization"] is True
            for row in rows
        )
        assert all(
            row["displacement_sensitivity_gate_passed"] is True for row in rows
        )


def test_v7_preflight_is_durably_invalidated_before_v8_rerun():
    protocol = _load(PROTOCOL_PATH)
    audit = protocol["engineering_amendment_audit"]
    path = REPOSITORY_ROOT / audit["artifact"]
    artifact = _load(path)

    assert _sha256(path) == audit["artifact_file_sha256"]
    assert (
        artifact["content_sha256"]
        == audit["artifact_content_sha256"]
        == _artifact_content_sha256(artifact)
    )
    assert artifact["protocol_id"].endswith("-v7")
    assert artifact["status"] == "invalidated-before-seal-or-label-scoring"
    assert set(artifact["label_boundary"].values()) == {False}
    assert artifact["invalidation"]["v7_resume_allowed"] is False
    assert artifact["invalidation"]["v7_seal_allowed"] is False
    assert artifact["invalidation"]["v7_score_allowed"] is False
    assert audit["v7_records_reused"] is False


def test_qrrho_protocol_cites_primary_literature_for_each_boundary():
    references = _load(PROTOCOL_PATH)["references"]
    dois = {reference["doi"] for reference in references if "doi" in reference}

    assert {
        "10.1002/chem.201200497",
        "10.1002/jcc.27129",
        "10.1021/jp0764384",
        "10.1039/D1SC00621E",
        "10.1039/D1CP05805C",
        "10.1021/jp902968m",
        "10.1021/jp963817g",
        "10.1063/5.0197592",
        "10.1021/acs.jced.7b00104",
        "10.1021/ct3010722",
        "10.1021/jacs.4c17622",
        "10.1021/acs.jpca.2c08023",
    } <= dois
