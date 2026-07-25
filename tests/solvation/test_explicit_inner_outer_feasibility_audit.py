from __future__ import annotations

import hashlib
import json
from pathlib import Path

AUDIT_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/implicit-solvation/benchmarks"
    / "route1-explicit-inner-implicit-outer-feasibility-audit-2026-07-25.json"
)
PREBUILT_PROTOCOL_PATH = AUDIT_PATH.parent / "prebuilt_inner_outer_protocol.json"
PREBUILT_SMOKE_PATH = (
    AUDIT_PATH.parent / "route1-prebuilt-inner-outer-smoke-2026-07-24.json"
)
DOCUMENTATION_DIR = AUDIT_PATH.parent.parent


def _load_audit() -> dict:
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def test_explicit_inner_outer_audit_has_a_self_consistent_fingerprint():
    audit = _load_audit()
    recorded = audit.pop("content_sha256")
    payload = json.dumps(
        audit,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()

    assert hashlib.sha256(payload).hexdigest() == recorded
    assert audit["schema_version"] == 1
    assert audit["protocol"]["protocol_id"] == (
        "route1-explicit-inner-implicit-outer-feasibility-v1"
    )
    assert all(
        len(fingerprint) == 64 for fingerprint in audit["source_fingerprints"].values()
    )


def test_repository_prebuilt_evidence_matches_the_audited_source_fingerprints():
    fingerprints = _load_audit()["source_fingerprints"]

    assert hashlib.sha256(PREBUILT_PROTOCOL_PATH.read_bytes()).hexdigest() == (
        fingerprints["prebuilt_inner_outer_protocol_sha256"]
    )
    assert hashlib.sha256(PREBUILT_SMOKE_PATH.read_bytes()).hexdigest() == (
        fingerprints["prebuilt_inner_outer_smoke_sha256"]
    )


def test_prebuilt_runtime_is_a_potential_not_an_absolute_free_energy():
    audit = _load_audit()
    current = audit["current_route1_runtime"]
    completion = audit["absolute_free_energy_completion_gate"]

    assert current["thermodynamic_quantity"] == (
        "fixed-shell cluster-continuum configurational potential"
    )
    assert not current["absolute_solvation_free_energy_claim"]
    assert current["multi_mlip_engineering_smoke_models"] == [
        "maceoff23m",
        "aimnet2",
        "ani2x",
    ]
    assert len(completion["terms_missing_from_prebuilt_runtime"]) == 4
    completion_boolean_gates = {
        key: value for key, value in completion.items() if isinstance(value, bool)
    }
    assert completion_boolean_gates == {
        "requires_cluster_conformer_and_size_ensemble": True,
        "requires_gas_and_solution_free_energies_not_only_minimum_energies": True,
        "requires_pure_solvent_reference": True,
        "requires_standard_state_accounting": True,
    }


def test_qcg_closes_a_supermolecular_cycle_but_not_with_arbitrary_mlip():
    audit = _load_audit()
    qcg = audit["candidate_workflows"]["crest_qcg"]
    source = qcg["current_upstream_source_audit"]

    assert qcg["computes_a_supermolecular_solvation_free_energy"]
    assert qcg["generates_solute_solvent_and_reference_solvent_ensembles"]
    assert qcg["includes_thermochemical_and_conformational_terms"]
    assert qcg["published_method_family"] == "GFN-xTB/GFN-FF"
    assert qcg["remote_head_verified_equal"]
    assert source["generic_backend_exists_for_non_qcg_crest_calculations"]
    assert not source["generic_backend_is_wired_into_qcg"]
    assert source["qcg_files_containing_generic_backend_token"] == 0
    assert source["qcg_direct_xtb_single_point_and_optimization"]
    assert source["qcg_frequency_path_calls_xtb_hess_or_ohess"]
    assert not qcg["arbitrary_gas_mlip_end_to_end_validation"]
    assert not qcg["neutral_aqueous_freesolv_product_validation"]


def test_febiss_is_site_selection_not_a_free_energy_engine():
    audit = _load_audit()
    febiss = audit["candidate_workflows"]["febiss"]

    assert febiss["requires_existing_explicit_solvent_trajectory_and_topology"]
    assert febiss["uses_gist_to_rank_solvent_sites"]
    assert febiss["writes_selected_microsolvated_structures"]
    assert febiss["remote_head_verified_equal"]
    assert not febiss["computes_complete_absolute_solvation_free_energy"]
    assert febiss["role"] == "optional inner-shell proposal preprocessor only"


def test_no_automatic_product_path_is_admitted_without_multimlip_gates():
    audit = _load_audit()
    admission = audit["multi_mlip_admission_gate"]
    decision = audit["overall_decision"]

    boolean_gates = {
        key: value for key, value in admission.items() if isinstance(value, bool)
    }
    assert boolean_gates == {
        "charge_spin_element_and_size_compatibility_required": True,
        "energy_zero_cancellation_requires_atom_conserving_cycle": True,
        "force_or_hessian_support_required_for_sampling_or_rrho": True,
        "intermolecular_cluster_domain_validation_required": True,
        "outer_fixed_charge_provider_must_cover_cluster_and_reference_species": True,
        "prospective_neutral_aqueous_accuracy_gate_required": True,
        (
            "pure_solvent_and_solute_cluster_ensembles_must_use_a_consistent_"
            "target_hamiltonian"
        ): True,
        "reference_to_target_overlap_must_fail_closed": True,
        "same_target_mlip_must_cover_all_cycle_species": True,
    }
    assert "atom-conserving" in admission["important_cancellation"]
    assert "end-to-end wall time" in admission["measured_cost_comparator"]

    decision_flags = {
        key: decision[key]
        for key in (
            "arbitrary_mlip_compatibility_claim_allowed",
            "automatic_inner_shell_runtime_added",
            "faster_than_mm_claim_allowed",
            "freesolv_accuracy_claim_allowed",
            "new_dependency_added",
            "paper_based_reimplementation_planned",
            "prebuilt_runtime_classification",
            "prebuilt_runtime_retained",
            "status",
        )
    }
    assert decision_flags == {
        "arbitrary_mlip_compatibility_claim_allowed": False,
        "automatic_inner_shell_runtime_added": False,
        "faster_than_mm_claim_allowed": False,
        "freesolv_accuracy_claim_allowed": False,
        "new_dependency_added": False,
        "paper_based_reimplementation_planned": False,
        "prebuilt_runtime_classification": ("experimental fixed-shell potential only"),
        "prebuilt_runtime_retained": True,
        "status": "no_admissible_neutral_small_molecule_product_path",
    }


def test_route1_docs_preserve_the_explicit_inner_outer_admission_boundary():
    specification = (DOCUMENTATION_DIR / "ROUTE1_PRODUCT_SPEC.md").read_text(
        encoding="utf-8"
    )
    formulas = (DOCUMENTATION_DIR / "FORMULAS_AND_REFERENCES.md").read_text(
        encoding="utf-8"
    )
    validation = (DOCUMENTATION_DIR / "VALIDATION_STATUS.md").read_text(
        encoding="utf-8"
    )
    benchmark = (AUDIT_PATH.parent / "README.md").read_text(encoding="utf-8")

    assert (
        "Automatic explicit-inner / implicit-outer feasibility boundary"
        in specification
    )
    assert "no_admissible_neutral_small_molecule_product_path" in specification
    assert AUDIT_PATH.name in specification

    assert "10.1021/acs.jctc.2c00239" in formulas
    assert "CREST's generic backend is not wired into QCG" in formulas
    assert "FEBISS is an optional proposal preprocessor" in formulas
    assert AUDIT_PATH.name in formulas

    assert "CREST's generic backend is not" in validation
    assert "wired into QCG" in validation
    assert "FEBISS is an optional proposal preprocessor" in validation
    assert "no automatic runtime or FreeSolv selector was added" in validation

    assert AUDIT_PATH.name in benchmark
    assert "proposal preprocessor only" in benchmark
    assert "No automatic explicit-inner runtime" in benchmark
    assert "faster-than-MM claim was added" in benchmark
