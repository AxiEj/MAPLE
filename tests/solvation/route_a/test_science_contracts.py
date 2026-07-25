from __future__ import annotations

import hashlib
import json
import math
import copy

from .conftest import DOCS_DIR, FIXTURE_DIR, PROJECT_ROOT, load_json
from .validators import canonical_sha256, derive_reference_cycle_gates


PROTOCOL_PATH = DOCS_DIR / "protocol-v1.json"
CLAIM_PATH = DOCS_DIR / "claim-contract-v1.json"
FAILURE_PATH = DOCS_DIR / "failure-contract-v1.json"
STATE_PATH = DOCS_DIR / "state-contract-v1.json"
RESTRAINT_PATH = DOCS_DIR / "restraint-contract-v1.json"
OUTER_ADAPTER_PATH = DOCS_DIR / "outer-adapter-contract-v1.json"
REFERENCE_CYCLE_PATH = DOCS_DIR / "reference-cycle-ablation-contract-v1.json"
OUTER_PARITY_PATH = FIXTURE_DIR / "outer" / "route2_acetone_component_parity_v1.json"
OUTER_FALSIFIER_MANIFEST = (
    FIXTURE_DIR / "outer" / "route2_outer_falsifier_manifest_v1.json"
)
REFERENCE_EVIDENCE_PATH = FIXTURE_DIR / "reference_cycle" / "valid_ablation_evidence_v1.json"
PREREGISTRATION_PATH = (
    FIXTURE_DIR / "reference_cycle" / "preregistration_registry_v1.json"
)


def _canonical_sha256(path) -> str:
    payload = json.loads(path.read_text())
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _outer_parity_evidence_valid(contract: dict, fixture: dict) -> bool:
    source = fixture.get("source", {})
    if source.get("route2_commit") != contract["route2_parity_source"]["commit"]:
        return False
    if contract["route2_parity_source"]["clean_tree_required"] and (
        source.get("working_tree_clean") is not True
        or source.get("development_dirty_tree_override") is not False
    ):
        return False

    for name, hash_field, hash_kind in [
        ("canary_artifact", "canonical_sha256", "canonical"),
        ("provider_audit_artifact", "canonical_sha256", "canonical"),
        ("native_stderr_artifact", "raw_sha256", "raw"),
        ("pedra_artifact", "raw_sha256", "raw"),
    ]:
        artifact = source.get(name, {})
        path = PROJECT_ROOT / artifact.get("path", "")
        if not path.is_file():
            return False
        if hash_kind == "canonical":
            actual = _canonical_sha256(path)
        else:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != artifact.get(hash_field):
            return False

    canary = load_json(PROJECT_ROOT / source["canary_artifact"]["path"])
    provider_audit = load_json(
        PROJECT_ROOT / source["provider_audit_artifact"]["path"]
    )
    if (
        canary.get("git_head") != source["route2_commit"]
        or canary.get("working_tree_clean") is not True
        or canary.get("development_dirty_tree_override") is not False
        or provider_audit.get("schema_version")
        != contract["audit_schema"]["route2_result_schema_version"]
    ):
        return False

    warning_policy = fixture.get("adapter_warning_policy", {})
    warning_counts = fixture.get("warning_counts", {})
    if (
        warning_policy.get("native_and_pedra_warnings_fatal") is not True
        or warning_policy.get(
            "source_reported_pedra_selection_policy_is_authoritative"
        )
        is not False
        or warning_counts != {"native_stderr": 0, "pedra": 0}
    ):
        return False
    diagnostics = canary["record"]["pcmsolver_diagnostics"]
    if (
        diagnostics["native_stderr_warning_count"]
        != warning_counts["native_stderr"]
        or diagnostics["pedra_warning_count"] != warning_counts["pedra"]
    ):
        return False

    return canary["record"]["components_kcal_mol"] == fixture[
        "components_kcal_mol"
    ]


def test_standard_state_density_has_explicit_units_and_dimensionless_log_argument():
    protocol = load_json(PROTOCOL_PATH)
    for standard_state in [protocol["domain"]["standard_state"], protocol["standard_state"]]:
        c0 = standard_state["C_number_0_per_A3"]
        rho_number = standard_state["rho_W_number_per_A3"]
        rho_ratio = standard_state["rho_W_over_C0"]
        water_molar = standard_state["water_bulk_concentration_molar"]
        solution_molar = standard_state["solution_reference_concentration_molar"]

        assert c0 == 0.000602214076
        assert rho_number == 0.03332953803622
        assert rho_ratio == 55.345
        assert math.isclose(
            rho_number,
            water_molar * c0 / solution_molar,
            rel_tol=0.0,
            abs_tol=1e-14,
        )
        assert math.isclose(rho_ratio, rho_number / c0, rel_tol=0.0, abs_tol=1e-12)

    formula = protocol["thermodynamics"]["formulas"]["qct_n_entry"]
    assert "ln(rho_W_over_C0)" in formula
    assert "ln(rho_W_number_per_A3)" not in formula


def test_restraint_contract_freezes_measures_hamiltonians_and_zero_release_v1():
    contract = load_json(RESTRAINT_PATH)

    cluster = contract["measures"]["cluster"]
    water = contract["measures"]["water_reference"]
    assert cluster["measure_id"] == "nonperiodic-solute-com-reduced-v1"
    assert water["measure_id"] == "nonperiodic-water-com-reduced-v1"
    assert cluster["boundary_conditions"] == {"pbc": False}
    assert water["boundary_conditions"] == {"pbc": False}
    assert cluster["removed_degrees_of_freedom"] == ["global translation by solute COM"]
    assert water["removed_degrees_of_freedom"] == ["global translation by water COM"]
    assert "solute rotation" in cluster["retained_degrees_of_freedom"]
    assert "water relative translations" in cluster["retained_degrees_of_freedom"]
    assert "water rotations" in cluster["retained_degrees_of_freedom"]
    assert "water rotation" in water["retained_degrees_of_freedom"]
    assert cluster["atom_list"] == "one connected solute X plus n complete labeled waters"
    assert water["atom_list"] == "one complete labeled water"

    restraints = contract["restraints"]
    assert restraints["R_0"]["formula"] == "R_0(q)=0"
    assert restraints["R_W"]["formula"] == "R_W(q)=0"
    assert restraints["R_n"]["formula"] == (
        "R_n(q)=sum_{j=1..n} 0.5*k*max(d_s(O_j)-lambda_s,0)^2"
    )
    assert "all solute atoms" in restraints["R_n"]["signed_distance"]
    assert restraints["R_n"]["membership"] == (
        "n labeled complete waters; oxygen alone determines shell membership"
    )
    assert restraints["R_n"]["effective_volume"] == (
        "V_eff,i=<integral exp(-beta*r_i(q_X,r_O)) d^3r_O>_(pi_base,i-1)"
    )
    assert restraints["R_n"]["effective_volume_is_hard_indicator_volume"] is False
    assert "average the per-base-frame translational integral before applying" in (
        restraints["R_n"]["estimator"]
    )
    assert set(restraints["R_n"]["forbidden_estimators"]) == {
        "hard-indicator geometric volume",
        "exp of an average log-volume",
        "interacting-endpoint frame average",
    }
    assert contract["alchemical_edge"]["H_D_i"].endswith("+R_(i-1)+r_i")
    assert contract["alchemical_edge"]["H_P_i"] == "U_OMOL_vac(XW_i)+R_i"
    assert contract["alchemical_edge"]["auxiliary_core_release_belongs_to"] == (
        "DeltaG_alch_labeled(i)"
    )
    core = contract["alchemical_edge"]["repulsive_core"]
    assert core["contract_id"] == "finite-gaussian-pair-core-v1"
    assert core["formula"] == "U_rep(r)=epsilon_eV*exp(-(r/sigma_A)^2)"
    assert core["force_formula"] == "F_i=(2*U_rep/sigma_A^2)*(r_i-r_j)"
    assert core["epsilon_eV"] > 0.0
    assert core["sigma_A"] > 0.0
    assert core["coincident_pair_barrier_eV"] == core["epsilon_eV"]
    assert any("minimum is atom coincidence" in form for form in core["forbidden_forms"])
    many_body = contract["alchemical_edge"]["many_body_interaction_contract"]
    assert many_body["pairwise_cross_sum_substitution_allowed"] is False
    assert "U_model(XW_i" in many_body["formula"]
    assert "irreducible cross-fragment three-body" in many_body["test_requirement"]

    release = contract["release"]
    assert release["definition"] == (
        "DeltaG_release(i)=F_target_i_interacting-F_biased_i_interacting"
    )
    assert release["v1_policy"] == (
        "no additional placement or orientation restraint beyond R_n"
    )
    assert release["identity"] == "F_target_i_interacting=F_biased_i_interacting"
    assert release["value_kcal_per_mol"] == 0.0
    assert release["applies_to_every_insertion_edge"] is True

    identity = contract["outer_endpoint_identity"]
    assert identity["required_equal_fields"] == [
        "atom_list_hash",
        "restraint_hash",
        "measure_id",
        "boundary_conditions",
    ]
    assert identity["mismatch_failure_code"] == "STATE_SPACE_MISMATCH"


def test_outer_adapter_is_an_additive_delta_once_and_not_a_second_total_energy():
    contract = load_json(OUTER_ADAPTER_PATH)
    formula = contract["energy_contract"]["delta_outer_kcal"]
    assert formula == (
        "Hartree_to_kcal_mol*((E_intrinsic_Polar[V_reac]-"
        "E_intrinsic_Polar[0])/Hartree_eV+E_PCM_pol_Ha+G_CDS_Ha)"
    )
    assert contract["energy_contract"]["target_hamiltonian"] == (
        "U_target_kcal(q)=U_OMOL_vac_kcal(q)+delta_outer_kcal(q)"
    )
    assert contract["energy_contract"]["pcm_polarization"] == (
        "E_PCM_pol_Ha=0.5*dot(MEP_Ha_per_e,ASC_e)"
    )
    assert contract["energy_contract"]["apply_outer_delta_exactly_once"] is True
    assert contract["energy_contract"]["intrinsic_energy_definition"] == (
        "Polar ML energy with explicit density-potential work removed, "
        "evaluated at converged reaction field and at zero field"
    )
    assert contract["energy_contract"]["forbidden_compositions"] == [
        "U_OMOL_vac plus a complete Route2 solvated total energy",
        "a second PCM polarization term",
        "a second CDS term",
        "a second half-coupling term",
    ]

    representation = contract["polar_representation"]
    assert representation["required_field"] == "density_coefficients"
    assert representation["shape"] == "(N,4)"
    assert representation["interpretation"] == (
        "learned atom-centered coefficients used to evaluate coarse point-multipole MEP"
    )
    assert representation["not_claimed_as"] == "quantum-mechanical electron density"
    assert representation["charge_fallback_allowed"] is False

    endpoints = contract["endpoint_identity"]
    assert endpoints["required_equal_fields"] == [
        "atom_list_hash",
        "restraint_hash",
        "measure_id",
        "boundary_conditions",
    ]
    assert contract["sampler_boundary"]["off24_role"] == "periodic discovery and proposal only"
    assert contract["sampler_boundary"]["off24_frames_are_production_rows"] is False
    assert contract["sampler_boundary"]["periodic_to_nonperiodic_direct_reweighting_allowed"] is False
    assert contract["scf_contract"] == {
        "maximum_iterations": 50,
        "mixing": 0.5,
        "density_tolerance_e": 1e-05,
        "energy_tolerance_eV": 1e-05,
        "density_coefficients_required": True,
        "fail_closed_on_nonconvergence": True,
    }
    assert contract["provider"]["reuse_policy"] == (
        "extract a shared stateless provider from Route 2; do not copy its implementation"
    )
    assert contract["pcm_response"]["external_mep_response_contract_version"] == 1
    assert contract["cavity_profile"]["minimum_added_sphere_radius_angstrom"] == 1.0
    assert contract["cavity_profile"]["native_and_pedra_warnings_fatal"] is True
    assert contract["cds_profile"]["sasa_probe_radius_angstrom"] == 0.4
    assert contract["cds_profile"]["sasa_grid_points_per_atom"] == 5810
    assert contract["audit_schema"]["route2_result_schema_version"] == 5


def test_outer_adapter_real_route2_component_parity_fixture():
    contract = load_json(OUTER_ADAPTER_PATH)
    fixture = load_json(OUTER_PARITY_PATH)
    components = fixture["components_kcal_mol"]

    assert math.isclose(
        components["electrostatic"],
        components["solute_polarization"] + components["pcm_polarization"],
        rel_tol=0.0,
        abs_tol=2e-15,
    )
    assert math.isclose(
        components["delta_g_solv"],
        components["electrostatic"] + components["cds"] + components["standard_state"],
        rel_tol=0.0,
        abs_tol=2e-15,
    )
    assert fixture["warning_counts"] == {"native_stderr": 0, "pedra": 0}
    assert _outer_parity_evidence_valid(contract, fixture)
    assert fixture["source"]["working_tree_clean"] is True
    assert fixture["source"]["development_dirty_tree_override"] is False
    assert fixture["adapter_warning_policy"][
        "native_and_pedra_warnings_fatal"
    ] is True
    assert fixture["adapter_warning_policy"][
        "source_reported_pedra_selection_policy_is_authoritative"
    ] is False
    assert fixture["source"]["route2_commit"] == contract["route2_parity_source"]["commit"]
    assert contract["provider_config"]["profile"] == fixture["provider_config"]["profile"]
    assert contract["provider_config"]["response"] == fixture["provider_config"]["response"]
    assert contract["audit_schema"]["parity_fixture"] == (
        "tests/solvation/route_a/fixtures/outer/route2_acetone_component_parity_v1.json"
    )
    assert contract["audit_schema"]["parity_fixture_canonical_sha256"] == (
        _canonical_sha256(OUTER_PARITY_PATH)
    )
    assert all(
        len(value) == 64
        for value in contract["route2_parity_source"]["source_sha256"].values()
    )

    dirty = copy.deepcopy(fixture)
    dirty["source"]["working_tree_clean"] = False
    assert not _outer_parity_evidence_valid(contract, dirty)

    warned = copy.deepcopy(fixture)
    warned["warning_counts"]["pedra"] = 1
    assert not _outer_parity_evidence_valid(contract, warned)

    tampered = copy.deepcopy(fixture)
    tampered["source"]["canary_artifact"]["canonical_sha256"] = "0" * 64
    assert not _outer_parity_evidence_valid(contract, tampered)


def test_reference_cycle_ablation_cannot_be_bypassed_for_final_scientific_status():
    ablation = load_json(REFERENCE_CYCLE_PATH)
    claim = load_json(CLAIM_PATH)
    failure = load_json(FAILURE_PATH)

    assert ablation["cycles"] == [
        "sequential-monomer-water-reference-v1",
        "solvent-cluster-reference-v1",
    ]
    assert ablation["selection_policy"]["final_holdout_may_select_cycle"] is False
    assert ablation["selection_policy"]["ablation_is_sensitivity_evidence_not_reselection"] is True
    assert ablation["selection_policy"]["preregistered_cycle_required"] is True
    assert ablation["selection_policy"]["numeric_agreement_threshold"] is None
    assert ablation["comparability"]["required_same_fields"] == [
        "frozen_holdout_hash",
        "n_by_molecule_hash",
        "nonreference_controls_hash",
    ]
    assert set(ablation["manifest_required_fields"]).issuperset(
        {
            "frozen_holdout_hash",
            "n_by_molecule_hash",
            "nonreference_controls_hash",
            "preregistered_selection_policy_hash",
            "per_molecule_both_cycle_results",
            "artifact_hashes",
            "adjudication_hash",
        }
    )

    required_fields = {
        term["field"]
        for row in claim["status_lattice"]
        if row["status"] == "scientifically-validated"
        for term in row["required"]
    }
    expected_gates = set(ablation["claim_gate_fields"])
    assert expected_gates == {
        "reference_cycle_ablation_completed",
        "reference_cycle_ablation_same_frozen_holdout",
        "reference_cycle_ablation_same_n_by_molecule",
        "reference_cycle_ablation_nonreference_controls_locked",
        "reference_cycle_ablation_both_cycles_complete",
        "reference_cycle_ablation_artifact_verified",
    }
    assert expected_gates.issubset(required_fields)
    assert ablation["gate_derivation"]["direct_boolean_gate_input_allowed"] is False
    assert claim["evidence_derivation"]["direct_boolean_gate_input_allowed"] is False
    derived_gates = set(claim["evidence_derivation"]["derived_gate_fields"])
    assert expected_gates.issubset(derived_gates)
    assert {
        "frozen_absolute_accuracy_passed",
        "material_partition_bias_passed",
    }.issubset(derived_gates)

    failure_codes = {row["code"] for row in failure["catalog"]["failure_codes"]}
    assert "REFERENCE_CYCLE_ABLATION_MISSING_OR_NONCOMPARABLE" in failure_codes


def _valid_ablation_evidence(contract):
    evidence = load_json(REFERENCE_EVIDENCE_PATH)
    assert {row["cycle_id"] for row in evidence["cycles"]} == set(contract["cycles"])
    return evidence


def test_reference_cycle_evidence_is_derived_and_rejects_spoofed_boolean_gates():
    contract = load_json(REFERENCE_CYCLE_PATH)
    evidence = _valid_ablation_evidence(contract)
    trusted = load_json(PREREGISTRATION_PATH)
    assert derive_reference_cycle_gates(
        contract,
        evidence,
        trusted,
    ) is None
    assert derive_reference_cycle_gates(
        contract,
        evidence,
        trusted,
        allow_test_evidence=True,
    ) == evidence["spoofed_boolean_gates"]
    assert trusted["evidence_class"] == (
        contract["trusted_preregistration_root"][
            "test_fixture_evidence_class"
        ]
    )

    mutations = []
    missing_cycle = copy.deepcopy(evidence)
    missing_cycle["cycles"].pop()
    mutations.append(missing_cycle)

    divergent_n_artifact = copy.deepcopy(evidence)
    divergent_n_artifact["n_by_molecule"]["m1"] = 9
    mutations.append(divergent_n_artifact)

    divergent_control_artifact = copy.deepcopy(evidence)
    divergent_control_artifact["nonreference_controls"]["temperature_k"] = 310.0
    mutations.append(divergent_control_artifact)

    false_open_order = copy.deepcopy(evidence)
    false_open_order["holdout_lock"]["opened_for_scoring_after_lock"] = False
    false_open_order["holdout_lock_hash"] = canonical_sha256(false_open_order["holdout_lock"])
    mutations.append(false_open_order)

    undeclared_primary = copy.deepcopy(evidence)
    undeclared_primary["selection"]["selected_primary_cycle"] = "undeclared-cycle"
    undeclared_primary["selection_hash"] = canonical_sha256(undeclared_primary["selection"])
    undeclared_primary["holdout_lock"]["development_selection_artifact_hash"] = undeclared_primary["selection_hash"]
    undeclared_primary["holdout_lock_hash"] = canonical_sha256(undeclared_primary["holdout_lock"])
    undeclared_primary["primary_cycle"] = "undeclared-cycle"
    mutations.append(undeclared_primary)

    wrong_holdout = copy.deepcopy(evidence)
    wrong_holdout["cycles"][0]["frozen_holdout_hash"] = "9" * 64
    wrong_holdout["cycles"][1]["frozen_holdout_hash"] = "9" * 64
    mutations.append(wrong_holdout)

    failed_cycle = copy.deepcopy(evidence)
    failed_cycle["cycles"][0]["failure_count"] = 1
    mutations.append(failed_cycle)

    bad_result_hash = copy.deepcopy(evidence)
    bad_result_hash["cycles"][0]["per_molecule_results"][0]["value"] = 999.0
    mutations.append(bad_result_hash)

    for mutation in mutations:
        assert derive_reference_cycle_gates(
            contract,
            mutation,
            trusted,
            allow_test_evidence=True,
        ) is None

    forged_root = copy.deepcopy(trusted)
    forged_root["selected_primary_cycle"] = contract["cycles"][1]
    assert derive_reference_cycle_gates(
        contract,
        evidence,
        forged_root,
        allow_test_evidence=True,
    ) is None

def test_new_science_contracts_are_hash_bound_by_protocol_and_state_contract():
    protocol = load_json(PROTOCOL_PATH)
    expected = {
        "restraint": RESTRAINT_PATH,
        "outer_adapter": OUTER_ADAPTER_PATH,
        "reference_cycle_ablation": REFERENCE_CYCLE_PATH,
    }
    for key, path in expected.items():
        contract = load_json(path)
        top = protocol[f"{key}_contract"]["reference"]
        artifact = protocol["artifact_references"][f"{key}_contract"]
        assert protocol[f"{key}_contract"]["code"] == contract["contract_name"]
        assert top["path"] == artifact["path"]
        assert top["hash_kind"] == artifact["hash_kind"] == "canonical_sha256"
        assert top["hash"] == artifact["sha256"] == _canonical_sha256(path)

    state = load_json(STATE_PATH)
    invariants = state["objects"]["QCTStateResult"]["invariants"]
    assert (
        "release_total == sum(release_steps) == 0.0 for restraint-contract-v1"
        in invariants
    )
    assert (
        "outer endpoint pairs have identical atom_list_hash, restraint_hash, "
        "measure_id, and boundary_conditions"
    ) in invariants
    assert "outer_cluster_endpoint_pair" in state["objects"]["QCTStateResult"]["required_ledger_terms"]
    assert "water_outer_endpoint_pair" in state["objects"]["QCTStateResult"]["required_ledger_terms"]


def test_real_xw_outer_falsifier_is_hash_complete_and_keeps_blocker_open():
    manifest = load_json(OUTER_FALSIFIER_MANIFEST)
    observations = manifest["observations"]

    for artifact in manifest["artifacts"].values():
        path = PROJECT_ROOT / artifact["path"]
        assert path.is_file()
        assert artifact["canonical_sha256"] == _canonical_sha256(path)

    raw = load_json(
        PROJECT_ROOT / manifest["artifacts"]["x_xw_raw"]["path"]
    )
    robustness = load_json(
        PROJECT_ROOT
        / manifest["artifacts"]["xw2_robustness_raw"]["path"]
    )
    area_scan = load_json(
        PROJECT_ROOT
        / manifest["artifacts"]["cavity_area_scan_partial"]["path"]
    )

    assert raw["source"]["git_head"] == manifest["source_contract"][
        "route2_commit"
    ]
    assert raw["source"]["working_tree_clean"] is True
    assert [record["label"] for record in raw["records"]] == ["W", "X", "XW1"]
    assert raw["failures"][0]["label"] == "XW2"
    assert "PCMSolver emitted a warning" in raw["failures"][0]["message"]
    assert observations["xw2_provider_evaluated"] == 9
    assert observations["xw2_warning_free"] == robustness["counts"][
        "warning-free"
    ] == 6
    assert observations["xw2_provider_failed"] == robustness["counts"][
        "provider-failed"
    ] == 3
    assert area_scan["aborted_candidate"]["area_bohr2"] == 1.5
    best = next(
        row
        for row in area_scan["summary"]
        if row["area_bohr2"] == observations["best_completed_area_bohr2"]
    )
    assert best["passed"] == observations["best_completed_area_passed"] == 9
    assert best["total"] == observations["best_completed_area_total"] == 13
    assert manifest["claim_boundary"]["hydration_free_energy"] is False
    assert manifest["claim_boundary"]["profile_change_approved"] is False
    assert "open production blocker" in manifest["decision"]
