from __future__ import annotations

import hashlib
import json
from pathlib import Path

from maple.solvation.release.admission import (
    CLAIM_BOUNDARY_ID,
    EXPOSURE_AWARE_PROTOCOL_LABEL,
    H1_V2_CAVITY_PROFILE_ID,
    H1_V2_CONTINUUM_PROFILE_ID,
    H1_V2_DISPLACEMENT_POLICY,
    H1_V2_EXACT_SCALAR,
    H1_V2_EXCLUDED_COMPONENTS,
    H1_V2_FORCE_DERIVATIVE,
    H1_V2_INCLUDED_COMPONENTS,
    H1_V2_INDUCED_SOURCE_KERNEL,
    H1_V2_LONG_RANGE_EVALUATOR_ID,
    H1_V2_ENDPOINT_REPLAY_SCOPE,
    H1_V2_PERMANENT_SOURCE_KERNEL,
    H1_V2_PROFILE_ID,
    H1_V2_RADII_PROVIDER_ID,
    H1_V2_RECEIVER_KERNEL,
    H1_V2_RECEIVER_SPACE_CONTRACT_SHA256,
    H1_V2_RECEIVER_SPACE_ID,
    H1_V2_ROOT_METHOD,
    H1_V2_SCALAR_ID,
    H1_V2_SECOND_START,
    H1_V2_SOURCE_COEFFICIENT_BASIS_ID,
    H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
    H1_V2_STATE_ID,
    H1_V2_TOPOLOGY_POLICY,
    REQUIRED_DOMAIN_GUARDS,
    REQUIRED_NON_ADMISSIONS,
    REQUIRED_RUNTIME_GUARDS,
    V1_ADMISSION_GATE_NAMES,
    verified_pro_schema_amendment_contract,
)
from maple.solvation.release.evidence import canonical_json_sha256


ROOT = Path(__file__).parents[2]
PREREG_PATH = ROOT / (
    "docs/route2/preregistrations/"
    "mace-mdp-polar-hybrid-harmonic-ef-revalidation-v2.json"
)
V1_PATH = ROOT / (
    "docs/route2/preregistrations/"
    "mace-mdp-polar-hybrid-harmonic-force-admission-v1.json"
)
PANEL_PATH = ROOT / (
    "docs/implicit-solvation/benchmarks/"
    "route2-gto-pcm-energy-projection-four-prereg-v1.json"
)
LINEAGE_PATH = ROOT / (
    "docs/route2/evidence/"
    "mace-mdp-polar-hybrid-pre-h1-lineage-observation-20260823.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_is_prepared_but_not_frozen_before_h1_identity_exists() -> None:
    protocol = _load(PREREG_PATH)
    classification = protocol["protocol_classification"]

    assert protocol["schema_version"] == (
        "route2-exposure-aware-retrospective-ef-revalidation-protocol-v2"
    )
    assert protocol["status"] == (
        "prepared-exposure-aware-retrospective-revalidation-awaiting-clean-"
        "commit-content-addressed-freeze-before-h1-seal"
    )
    assert classification["label"] == EXPOSURE_AWARE_PROTOCOL_LABEL
    assert classification["is_blind_preregistration"] is False
    assert classification["is_retrospective_revalidation"] is True
    assert classification["is_frozen"] is False
    assert classification["h1_identity_exists"] is False
    language = " ".join(
        [
            protocol["status"],
            classification["freeze_boundary"],
            classification["purpose"],
        ]
    ).lower()
    assert "exposure-aware" in language
    assert "retrospective revalidation" in language
    assert "pre-h1 development checkout" in language
    assert "prepared but not frozen" in language
    assert "future clean committed protocol byte sha256" in language
    assert "creates the h1 identity" in language


def test_v2_target_ids_science_guards_and_ef_only_boundary_match_code() -> None:
    protocol = _load(PREREG_PATH)
    identity = protocol["prospective_h1_identity_contract"]
    science = protocol["prospective_h1_scientific_settings"]

    assert {key: identity[key] for key in (
        "profile_id", "scalar_id", "state_id", "claim_boundary_id"
    )} == {
        "profile_id": H1_V2_PROFILE_ID,
        "scalar_id": H1_V2_SCALAR_ID,
        "state_id": H1_V2_STATE_ID,
        "claim_boundary_id": CLAIM_BOUNDARY_ID,
    }
    assert "do not constitute an H1 identity" in identity["lifecycle"]
    assert "valid computation seal" in identity["lifecycle"]
    assert protocol["target_capabilities"] == {
        "E": True,
        "F": True,
        "H": False,
        "V": False,
        "M": False,
    }
    assert science == {
        "continuum": {
            "profile_id": H1_V2_CONTINUUM_PROFILE_ID,
            "transition_width_angstrom2": 0.18,
            "surface_lmax": 1,
            "exposure_lmax": 2,
            "exposure_radial_quadrature_order": 32,
            "source_radial_quadrature_order": 32,
            "green_radial_quadrature_order": 32,
        },
        "cavity": {
            "profile_id": H1_V2_CAVITY_PROFILE_ID,
            "radii_provider_id": H1_V2_RADII_PROVIDER_ID,
        },
        "source_receiver": {
            "permanent_source_kernel": H1_V2_PERMANENT_SOURCE_KERNEL,
            "induced_source_kernel": H1_V2_INDUCED_SOURCE_KERNEL,
            "receiver_kernel": H1_V2_RECEIVER_KERNEL,
            "long_range_evaluator_id": H1_V2_LONG_RANGE_EVALUATOR_ID,
            "source_space_id": H1_V2_SOURCE_COEFFICIENT_BASIS_ID,
            "source_space_contract_sha256": H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
            "receiver_space_id": H1_V2_RECEIVER_SPACE_ID,
            "receiver_space_contract_sha256": H1_V2_RECEIVER_SPACE_CONTRACT_SHA256,
        },
        "energy_ledger": {
            "exact_scalar": H1_V2_EXACT_SCALAR,
            "included_components": list(H1_V2_INCLUDED_COMPONENTS),
            "excluded_components": list(H1_V2_EXCLUDED_COMPONENTS),
        },
        "root_algorithm": {
            "method": H1_V2_ROOT_METHOD,
            "tolerance_ev": 1.0e-10,
            "maximum_iterations": 40,
            "second_start": H1_V2_SECOND_START,
            "total_charge_tolerance_e": 1.0e-8,
            "multi_start_field_tolerance_ev": 2.0e-9,
            "multi_start_energy_tolerance_ev": 1.0e-10,
            "evidence_scope": H1_V2_ENDPOINT_REPLAY_SCOPE,
        },
        "force_stencil": {
            "derivative": H1_V2_FORCE_DERIVATIVE,
            "coarse_step_angstrom": 5.0e-4,
            "fine_step_angstrom": 2.5e-4,
            "independent_step_angstrom": 1.25e-4,
            "maximum_local_error_ev_per_angstrom": 2.0e-4,
            "topology_policy": H1_V2_TOPOLOGY_POLICY,
            "displacement_policy": H1_V2_DISPLACEMENT_POLICY,
        },
    }
    assert protocol["runtime_guards"] == list(REQUIRED_RUNTIME_GUARDS)
    assert protocol["domain_guards"] == list(REQUIRED_DOMAIN_GUARDS)
    boundary = protocol["claim_boundary"]
    assert boundary["id"] == CLAIM_BOUNDARY_ID
    assert boundary["non_admissions"] == list(REQUIRED_NON_ADMISSIONS)
    boundary_language = " ".join(
        [boundary["candidate_admission"], boundary["protocol_does_not_admit"]]
    ).lower()
    assert "only" in boundary_language
    assert "e" in boundary_language and "f" in boundary_language
    assert "admits no capability" in boundary_language


def test_every_inherited_v1_field_is_exact() -> None:
    protocol = _load(PREREG_PATH)
    v1 = _load(V1_PATH)
    inherited = protocol["inherited_v1_contract"]
    inherited_keys = (
        "target_capabilities",
        "exact_scalar",
        "included_components",
        "excluded_components",
        "predecessor_failure",
        "prior_observation_boundary",
        "parent_panel",
        "frozen_runtime_contract",
        "force_panel",
        "admission_gates",
        "decision_rule",
        "claim_boundary",
        "fit_calibration_or_case_selection",
    )

    assert set(inherited) == set(inherited_keys)
    for key in inherited_keys:
        assert inherited[key] == v1[key], key
    assert tuple(inherited["admission_gates"])[0:-1] == V1_ADMISSION_GATE_NAMES

    runtime = inherited["frozen_runtime_contract"]
    science = protocol["prospective_h1_scientific_settings"]
    assert science["continuum"]["transition_width_angstrom2"] == runtime[
        "transition_width_angstrom2"
    ]
    for field in (
        "surface_lmax",
        "exposure_lmax",
        "exposure_radial_quadrature_order",
        "source_radial_quadrature_order",
        "green_radial_quadrature_order",
    ):
        assert science["continuum"][field] == runtime[field]
    assert science["root_algorithm"] == {
        "method": runtime["root_method"],
        "tolerance_ev": runtime["root_tolerance_ev"],
        "maximum_iterations": runtime["maximum_root_iterations"],
        "second_start": runtime["second_start"],
        "total_charge_tolerance_e": runtime["total_charge_tolerance_e"],
        "multi_start_field_tolerance_ev": runtime[
            "multi_start_field_tolerance_ev"
        ],
        "multi_start_energy_tolerance_ev": runtime[
            "multi_start_energy_tolerance_ev"
        ],
        "evidence_scope": H1_V2_ENDPOINT_REPLAY_SCOPE,
    }
    assert science["force_stencil"] == {
        "derivative": runtime["force_derivative"],
        "coarse_step_angstrom": runtime["force_coarse_step_angstrom"],
        "fine_step_angstrom": runtime["force_fine_step_angstrom"],
        "independent_step_angstrom": runtime["independent_force_step_angstrom"],
        "maximum_local_error_ev_per_angstrom": runtime[
            "force_maximum_local_error_estimate_ev_per_angstrom"
        ],
        "topology_policy": runtime["force_topology_policy"],
        "displacement_policy": runtime["force_displacement_policy"],
    }


def test_parent_v1_and_four_record_panel_are_content_addressed() -> None:
    protocol = _load(PREREG_PATH)
    bindings = protocol["parent_bindings"]
    panel = _load(PANEL_PATH)

    assert bindings["v1_preregistration"] == {
        "artifact_id": (
            "route2-mace-mdp-polar-hybrid-harmonic-force-admission-prereg-v1"
        ),
        "relative_path": (
            "docs/route2/preregistrations/"
            "mace-mdp-polar-hybrid-harmonic-force-admission-v1.json"
        ),
        "sha256": _sha256(V1_PATH),
    }
    panel_binding = bindings["v1_parent_panel"]
    assert panel_binding["artifact_id"] == panel["artifact_id"]
    assert panel_binding["relative_path"] == (
        "docs/implicit-solvation/benchmarks/"
        "route2-gto-pcm-energy-projection-four-prereg-v1.json"
    )
    assert panel_binding["sha256"] == _sha256(PANEL_PATH)
    assert panel_binding["record_count"] == len(panel["records"]) == 4
    assert panel_binding["record_order"] == panel["method"]["record_order"]
    assert panel_binding["record_order"] == [
        record["compound_id"] for record in panel["records"]
    ]


def test_h0_exposure_inventory_binds_two_known_passes_and_aggregate() -> None:
    protocol = _load(PREREG_PATH)
    exposure = protocol["exposure_inventory"]

    assert exposure["historical_execution_git_head"].startswith("4cf8db40")
    assert exposure["known_pass_count"] == 2
    assert [item["label"] for item in exposure["historical_replicates"]] == [
        "a",
        "b",
    ]
    assert all(
        item["observed_outcome"] == "candidate-energy-force-gate-passed"
        for item in exposure["historical_replicates"]
    )
    expected = {
        "docs/route2/evidence/"
        "mace-mdp-polar-hybrid-harmonic-force-admission-replicate-a-4cf8db40.json": (
            "1a248e06218c0cb968e87992a7444923a246064c3fa7fa7362c00cf394587589"
        ),
        "docs/route2/evidence/"
        "mace-mdp-polar-hybrid-harmonic-force-admission-replicate-b-4cf8db40.json": (
            "3c393008037697526f06731e3de0e983f740e5cc3f0a02b5ad2dc54347c7e415"
        ),
        "docs/route2/evidence/"
        "mace-mdp-polar-hybrid-harmonic-force-admission-replicated-4cf8db40.json": (
            "33b5eae63396219dea2e8c062352dbc1eb05eb52329ee6a6b854deaf756eea77"
        ),
    }
    records = [*exposure["historical_replicates"], exposure["historical_aggregate"]]
    assert {item["relative_path"]: item["sha256"] for item in records} == expected
    for relative_path, expected_sha in expected.items():
        assert _sha256(ROOT / relative_path) == expected_sha
    assert "pre-H1 development checkout" in exposure["transfer_rule"]
    assert "future H1 identity" in exposure["transfer_rule"]
    assert "H1 revalidation authors" not in exposure["disclosure"]
    cavity = exposure["cavity_identity_disclosure"]
    assert "legacy frozen projection-result" in cavity["historical_v1_benzene_input"]
    assert "canonical smd-water-coulomb-radii-v1" in cavity[
        "prospective_h1_benzene_input"
    ]
    assert "not the H1 continuum input" in cavity["projection_result_role_in_h1"]


def test_tracked_lineage_report_is_self_hashed_and_fully_bound() -> None:
    audit = _load(PREREG_PATH)["historical_lineage_audit"]
    report = _load(LINEAGE_PATH)
    expected_paths = {
        "maple/function/calculator/extra_correction/implicit/continuum_response.py",
        "maple/function/calculator/extra_correction/implicit/route2_derivative.py",
        (
            "maple/function/calculator/extra_correction/implicit/"
            "route2_electronic_model.py"
        ),
        "maple/function/calculator/extra_correction/implicit/route2_engine.py",
        "maple/solvation/api/__init__.py",
        "maple/solvation/api/profiles.py",
        "maple/solvation/api/scalar_registry.py",
        "maple/solvation/api/state_registry.py",
        "maple/solvation/continuum/__init__.py",
        "maple/solvation/continuum/functional.py",
        "maple/solvation/continuum/harmonic_exposure.py",
        "maple/solvation/continuum/harmonic_torch_functional.py",
        "maple/solvation/continuum/harmonic_torch_primitives.py",
        "maple/solvation/coupling/separated_operators.py",
        "maple/solvation/coupling/state_equation.py",
        "maple/solvation/derivatives/__init__.py",
        "maple/solvation/derivatives/scalar_finite_difference.py",
        "maple/solvation/experimental/__init__.py",
        "maple/solvation/experimental/mace_mdp_polar_harmonic.py",
        "maple/solvation/models/mace_mdp.py",
        "maple/solvation/models/mace_mdp_polar_hybrid.py",
        "maple/solvation/models/mace_polar_separated.py",
    }

    assert audit["report_relative_path"] == LINEAGE_PATH.relative_to(ROOT).as_posix()
    assert audit["raw_file_sha256"] == _sha256(LINEAGE_PATH)
    assert audit["raw_file_sha256"] == (
        "c50f41353edec90a73def498c3b3c36940a9ff2094a3afad16577666a6385032"
    )
    report_without_self_hash = dict(report)
    internal_hash = report_without_self_hash.pop("report_sha256")
    assert internal_hash == canonical_json_sha256(report_without_self_hash)
    assert audit["report_sha256"] == internal_hash
    assert audit["schema_version"] == report["schema_version"]

    source = report["evidence_artifact"]
    assert audit["source_evidence_relative_path"] == source["path"]
    assert audit["source_evidence_sha256"] == source["sha256"]
    assert _sha256(ROOT / source["path"]) == source["sha256"]
    assert audit["repository_snapshot"] == {
        key: report["repository"][key]
        for key in ("head", "tree", "dirty", "status_porcelain")
    }
    assert audit["repository_snapshot"]["head"] == (
        "4cf8db400de6081d34db0e0426d0c2fc5c8f3d5c"
    )
    assert audit["repository_snapshot"]["tree"] == (
        "ac96348b6e7f937f1028aa39f48b89a7f76c30a9"
    )
    assert audit["repository_snapshot"]["dirty"] is True
    assert audit["repository_snapshot"]["status_porcelain"]

    expected_counts = {
        "total": 131,
        "matching": 109,
        "drifted": 22,
        "missing": 0,
    }
    assert audit["counts"] == report["counts"] == expected_counts
    assert len(report["matching"]) == expected_counts["matching"]
    assert all(isinstance(path, str) and path for path in report["matching"])
    assert len(set(report["matching"])) == len(report["matching"])
    assert report["missing"] == []
    assert len(report["drifted"]) == expected_counts["drifted"]
    assert all(
        set(record) == {"path", "expected_sha256", "actual_sha256"}
        and len(record["expected_sha256"]) == 64
        and len(record["actual_sha256"]) == 64
        for record in report["drifted"]
    )
    assert audit["drifted_paths"] == [
        record["path"] for record in report["drifted"]
    ]
    assert set(audit["drifted_paths"]) == expected_paths
    assert not expected_paths.intersection(report["matching"])
    assert "cannot be transferred" in audit["conclusion"]
    assert "pre-H1 development checkout" in audit["conclusion"]


def test_execution_order_is_fail_closed_without_results_or_post_run_freedom() -> None:
    protocol = _load(PREREG_PATH)
    assert protocol["frozen_execution_sequence"] == [
        "protocol",
        "computation_seal",
        "cold_replicate_a",
        "cold_replicate_b",
        "mechanical_aggregator",
        "data_only_admission_overlay",
    ]
    invariants = protocol["sequence_invariants"]
    assert all(
        value is True
        for key, value in invariants.items()
        if key != "failure_capabilities"
    )
    assert set(invariants["failure_capabilities"].values()) == {False}
    prohibitions = protocol["post_freeze_prohibitions"]
    assert all(value is False for key, value in prohibitions.items() if key != "policy")
    policy = prohibitions["policy"].lower()
    for phrase in (
        "no fit",
        "calibration",
        "case selection",
        "post-run protocol change",
    ):
        assert phrase in policy

    execution = protocol["execution_state"]
    assert execution["computation_seal"] is None
    assert execution["cold_replicate_a"] is None
    assert execution["cold_replicate_b"] is None
    assert execution["aggregate_result"] is None
    assert execution["admission_overlay"] is None
    current_capabilities = execution[
        "capabilities_currently_admitted_by_this_protocol"
    ]
    assert set(current_capabilities.values()) == {False}
    assert invariants["failure_retains_negative_evidence"] is True


def test_verified_pro_amendment_is_content_bound_and_archive_only() -> None:
    protocol = _load(PREREG_PATH)
    amendment = protocol["verified_pro_schema_amendment"]
    assert amendment == verified_pro_schema_amendment_contract()
    binding = amendment["verified_pro_audit"]
    audit_path = ROOT / binding["relative_path"]
    audit = _load(audit_path)
    assert _sha256(audit_path) == binding["raw_file_sha256"]
    audit_payload = dict(audit)
    internal = audit_payload.pop("audit_sha256")
    assert internal == binding["audit_sha256"]
    assert internal == canonical_json_sha256(audit_payload)
    for role in ("prompt", "response", "mode_verification"):
        record = audit[role]
        assert _sha256(ROOT / record["relative_path"]) == record["sha256"]
    response = (ROOT / audit["response"]["relative_path"]).read_text(
        encoding="utf-8"
    )
    assert "**APPROVE**" in response
    assert binding["terminal_marker"] in response
    mode_lines = (ROOT / audit["mode_verification"]["relative_path"]).read_text(
        encoding="utf-8"
    ).splitlines()
    assert "composer=Pro" in mode_lines
    assert "power=Pro, 5 of 5." in mode_lines
    policy = amendment["historical_h0_archive_policy"]
    assert policy["historical_rich_preimage"] == "non-identifiable-do-not-fabricate"
    assert policy["synthetic_rich_golden"] == "adapter-compatibility-only"
    assert policy["admission_transfer_to_h1"] is False
