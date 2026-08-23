from __future__ import annotations

import hashlib
import json
from pathlib import Path

from maple.solvation.api import (
    OPERATIONAL_MACE_MDP_POLAR_HYBRID_SMD_TOTAL_SMOOTH_HARMONIC_DDPCM_PROFILE_V1,
    OPERATIONAL_MACE_MDP_POLAR_HYBRID_SMD_TOTAL_SMOOTH_HARMONIC_DDPCM_V1,
)

ROOT = Path(__file__).parents[2]
PREREGISTRATION = ROOT / (
    "docs/implicit-solvation/benchmarks/"
    "route2-mace-mdp-polar-hybrid-mnsol10-fullsolv-prereg-v1.json"
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hybrid_mnsol10_prereg_is_exposure_aware_and_pre_result() -> None:
    payload = _load(PREREGISTRATION)
    classification = payload["protocol_classification"]
    assert payload["status"] == (
        "prepared-exposure-aware-known-panel-regression-no-hybrid-output-observed"
    )
    assert classification["is_blind_preregistration"] is False
    assert classification["is_exposure_aware_revalidation"] is True
    assert "No MACE-MDP+MACE-POLAR" in classification["unobserved_target"]
    assert len(classification["known_prior_evidence"]) == 2
    assert "must execute all ten" in classification["non_transfer_rule"]
    ledger = classification["prior_observation_ledger"]
    assert len(ledger) == 11
    assert {item["role"] for item in ledger} >= {
        "public-aggregate-observed",
        "private-row-level-observed-local-only",
        "public-hybrid-electrostatic-observed",
        "public-hybrid-water-force-observed",
        "public-hybrid-water-hvp-observed",
    }
    for item in ledger:
        path = ROOT / item["path"]
        if item["path"].startswith(".omx/"):
            assert item["role"] == "private-row-level-observed-local-only"
            if path.is_file():
                assert _sha256(path) == item["file_sha256"]
        else:
            assert path.is_file()
            assert _sha256(path) == item["file_sha256"]
    normalized = json.dumps(payload, sort_keys=True).lower()
    for forbidden in (
        '"records"',
        '"aggregate_metrics"',
        '"predicted_delta_g_kcal_mol"',
        '"signed_error_kcal_mol"',
        '"status": "pass"',
    ):
        assert forbidden not in normalized


def test_hybrid_mnsol10_dataset_membership_and_provenance_are_exact() -> None:
    payload = _load(PREREGISTRATION)
    dataset = payload["dataset_contract"]
    for role in ("protocol", "selection"):
        path = ROOT / dataset[f"{role}_path"]
        assert _sha256(path) == dataset[f"{role}_sha256"]
    selection = _load(ROOT / dataset["selection_path"])
    assert dataset["selection_fingerprint"] == selection["selection_fingerprint"]
    assert dataset["record_count"] == len(selection["selected_records"]) == 10
    assert (
        dataset["solvent_count"]
        == len({row["canonical_solvent"] for row in selection["selected_records"]})
        == 10
    )
    assert dataset["partition_counts"] == {"confirmation": 8, "development": 2}
    assert dataset["row_level_output"] == "private-below-.omx-no-redistribution"


def test_hybrid_mnsol10_source_receiver_and_energy_ledger_are_not_conflated() -> None:
    payload = _load(PREREGISTRATION)
    identity = payload["target_identity"]
    assert identity["profile_id"] == (
        OPERATIONAL_MACE_MDP_POLAR_HYBRID_SMD_TOTAL_SMOOTH_HARMONIC_DDPCM_PROFILE_V1
    )
    assert identity["scalar_id"] == (
        OPERATIONAL_MACE_MDP_POLAR_HYBRID_SMD_TOTAL_SMOOTH_HARMONIC_DDPCM_V1
    )
    model = payload["model_contract"]
    nonpolar = payload["nonpolar_contract"]
    assert "G_hybrid_ddPCM" in identity["exact_solvation_scalar"]
    assert "+G_SMD_CDS" in identity["exact_solvation_scalar"]
    assert "cancels" in identity["gas_energy_cancellation"]
    assert (
        "field-conditioned raw MACE-POLAR energy difference"
        in identity["excluded_energy_terms"]
    )
    assert "exterior point" in model["permanent_source_kernel"]
    assert "sigma=1.5-A Gaussian" in model["induced_source_kernel"]
    assert "eight-channel" in model["receiver_space"]
    assert nonpolar == {
        "provider_id": "maple.route2.solvent-term.pyscf-smd-cds.impl.v1",
        "model": "PySCF-2.13.1 SMD-CDS",
        "coefficient_scale": 1.0,
        "fit_or_calibration": False,
        "combination_rule": "strict additive G_hybrid_ddPCM + G_SMD_CDS",
    }


def test_hybrid_mnsol10_numerics_are_inherited_without_target_tuning() -> None:
    payload = _load(PREREGISTRATION)
    continuum = payload["continuum_contract"]
    provenance = continuum["numerics_provenance"]
    pure_prereg = ROOT / provenance["hybrid_four_case_preregistration_path"]
    assert _sha256(pure_prereg) == provenance["hybrid_four_case_preregistration_sha256"]
    assert {
        key: continuum[key]
        for key in (
            "transition_width_angstrom2",
            "surface_lmax",
            "partition_lmax",
            "partition_radial_quadrature_order",
            "source_radial_quadrature_order",
            "double_layer_radial_quadrature_order",
            "receiver_radial_quadrature_order",
        )
    } == {
        "transition_width_angstrom2": 0.08,
        "surface_lmax": 3,
        "partition_lmax": 6,
        "partition_radial_quadrature_order": 96,
        "source_radial_quadrature_order": 128,
        "double_layer_radial_quadrature_order": 128,
        "receiver_radial_quadrature_order": 128,
    }
    assert "do not transfer" in provenance["rule"]
    assert "not this profile identity" in provenance["pure_mnsol10_role"]

    root = payload["root_contract"]
    root_provenance = root["root_semantics_provenance"]
    hybrid_prereg = ROOT / root_provenance["hybrid_four_case_preregistration_path"]
    assert (
        _sha256(hybrid_prereg)
        == root_provenance["hybrid_four_case_preregistration_sha256"]
    )
    assert root["method"] == "anderson"
    assert root["starts"] == ["cold-zero", "deterministic-wide-0.05"]
    assert root["wide_initial_amplitude"] == 0.05
    assert root["maximum_actual_residual_norm"] == 1.0e-10
    assert root["maximum_permanent_charge_error_e"] == 1.0e-8
    assert root["maximum_induced_charge_error_e"] == 1.0e-8
    assert root["maximum_combined_charge_error_e"] == 1.0e-8


def test_hybrid_mnsol10_decision_rule_is_hard_and_nonadaptive() -> None:
    payload = _load(PREREGISTRATION)
    decision = payload["decision_rule"]
    assert decision["maximum_mae_kcal_mol"] == 1.5
    assert decision["primary_metric"] == (
        "mean_absolute_error_kcal_mol over all ten frozen records"
    )
    assert decision["required_gates"] == [
        "complete_exact_frozen_panel",
        "ten_distinct_solvents",
        "all_cold_wide_roots_pass",
        "all_actual_root_residuals_pass",
        "all_permanent_and_induced_charge_checks_pass",
        "mae_at_most_1_5_kcal_mol",
    ]
    assert "Do not change membership" in decision["failure"]
    prohibitions = payload["post_execution_prohibitions"]
    assert set(prohibitions.values()) == {False}
    for boundary in (
        "complete 148-row confirmation",
        "broad MNSol generalization",
        "distorted PES coverage",
        "Hessian/FREQ",
        "MD/NVE",
        "strict Tier V",
    ):
        assert boundary in payload["claim_boundary"]


def test_hybrid_mnsol10_execution_boundary_is_split_and_deterministic() -> None:
    execution = _load(PREREGISTRATION)["execution_contract"]
    assert "no experimental target" in execution["label_boundary"]
    assert "one host-user-global no-replace claim" in execution["one_time_claim"]
    assert "immutable semantic stage" in execution["one_time_claim"]
    assert "never reopens" in execution["one_time_claim"]
    assert "complete-or-typed-failure" in execution["prediction_terminal"]
    assert "model-free and continuum-free" in execution["scoring_boundary"]
    assert "separately retryable" in execution["public_projection"]
    assert "fsync(file)" in execution["publication_durability"]
    assert "fsync(parent directory)" in execution["publication_durability"]
    assert "aborted-unresolved" in execution["orphan_recovery"]
    assert "boot_id" in execution["orphan_recovery"]
    assert execution["evidence_class"] == "exposure-aware-known-panel-regression"
    assert execution["required_environment"] == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
    }


def test_hybrid_mnsol10_verified_pro_stop_repair_approve_chain_is_bound() -> None:
    contract = _load(PREREGISTRATION)["verified_pro_audit"]
    audit_path = ROOT / contract["path"]
    assert _sha256(audit_path) == contract["file_sha256"]
    audit = _load(audit_path)
    assert audit["artifact_id"] == contract["artifact_id"]
    assert audit["status"] == ("verified-pro-final-approve-before-scientific-execution")
    assert audit["scientific_output_observed_before_final_approval"] is False
    assert [item["verdict"] for item in audit["rounds"]] == [
        "STOP",
        "STOP",
        "APPROVE",
    ]
    final = audit["rounds"][-1]
    assert final["prompt_sha256"] == contract["final_prompt_sha256"]
    assert final["response_sha256"] == contract["final_response_sha256"]
    assert final["end_marker"] == contract["terminal_marker"]
    for item in audit["rounds"]:
        for role in ("prompt", "mode", "submission", "response", "uia"):
            path = ROOT / item[f"{role}_path"]
            if path.is_file():
                assert _sha256(path) == item[f"{role}_sha256"]
    final_response_path = ROOT / final["response_path"]
    if final_response_path.is_file():
        final_response = final_response_path.read_text(encoding="utf-8")
        assert "MAPLE HYBRID ACCURACY FINAL REPAIR AUDIT" in final_response
        assert "APPROVE" in final_response
        assert final_response.rstrip().endswith(
            "MAPLE HYBRID ACCURACY FINAL REPAIR COMPLETE\nResponse actions"
        )
    final_mode_path = ROOT / final["mode_path"]
    final_submission_path = ROOT / final["submission_path"]
    if final_mode_path.is_file() and final_submission_path.is_file():
        final_mode = final_mode_path.read_text(encoding="utf-8-sig")
        final_submission = final_submission_path.read_text(encoding="utf-8-sig")
        assert "composer=Pro" in final_mode
        assert "power=Pro, 5 of 5." in final_mode
        assert "pro_thinking_observed=True" in final_submission
