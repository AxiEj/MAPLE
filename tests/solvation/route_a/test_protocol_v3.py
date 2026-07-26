from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from maple.function.dispatcher.solvfe.protocol import (
    ProtocolIntegrityError,
    RouteAProtocol,
)

from .conftest import DOCS_DIR, PROJECT_ROOT


PROTOCOL_PATH = DOCS_DIR / "protocol-v3.json"
SCHEMA_PATH = DOCS_DIR / "protocol-schema-v3.json"
CONDITIONING_PATH = DOCS_DIR / "conditioning-contract-v3.json"
PACKING_PATH = DOCS_DIR / "packing-contract-v3.json"
LEDGER_PATH = DOCS_DIR / "ledger-contract-v3.json"
THERMODYNAMICS_PATH = DOCS_DIR / "thermodynamics-v3.md"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_v3_is_self_consistent_and_binds_all_scientific_artifacts():
    protocol = RouteAProtocol.load(
        PROTOCOL_PATH,
        project_root=PROJECT_ROOT,
    )

    assert protocol.content_hash == protocol.data["protocol_sha256"]
    assert protocol.data["protocol_version"] == "3.1.0"
    assert protocol.data["status"] == (
        "research-contract-implemented-production-validation-pending"
    )
    assert protocol.data["supersedes"]["protocol_version"] == "2.0.0"
    assert protocol.verified_artifact_count == 5


def test_protocol_v3_uses_one_measure_and_remains_fail_closed():
    protocol = _load(PROTOCOL_PATH)
    conditioning = protocol["conditioning_measure"]
    packing = protocol["packing"]
    association = protocol["fixed_n_association"]
    status = protocol["scientific_status"]
    ledger = protocol["soft_occupancy_and_closure"]

    assert conditioning["member_weight"] == (
        "b=1/(1+exp((d_s-lambda_s)/R))"
    )
    assert conditioning["nonmember_weight"] == "1-b"
    assert conditioning["same_membership_hash_required_for_all_terms"] is True
    assert conditioning["hard_empty_indicator_role"] == "diagnostic-only"
    assert conditioning["surface_coordinate"] == (
        "d_s=-tau*ln sum_i exp(-d_i/tau)"
    )
    assert conditioning["surface_hard_min_sensitivity_required"] is True
    assert (
        conditioning["pre_v3_1_nearest_surface_artifacts_compatible"] is False
    )
    assert conditioning["active_occupancy_max_minimum"] == 1
    assert conditioning["active_support_must_include_n0_and_n1"] is True
    assert conditioning["full_support_zero_tail_allowed"] is True
    assert packing["target_state_index_explicit"] is True
    assert packing["full_field_state_index_explicit"] is True
    assert packing["row_position_assumptions_allowed"] is False
    assert packing["water_hamiltonian_status"] == "not-frozen"
    assert (
        packing["real_replica_sampling_allowed_before_water_hamiltonian_freeze"]
        is False
    )
    assert association[
        "same_membership_and_solute_measure_as_packing_required"
    ] is True
    assert association["hard_conditioned_partition_function_claim_allowed"] is False
    assert ledger["master_equation"] == (
        "mu_A=-RT ln(p0_soft)-RT ln sum_n exp[-beta A_n]"
    )
    assert ledger["fixed_n_cross_check"] == (
        "mu_n=A_n+RT ln[x_tilde(n)/p0_soft]"
    )
    assert ledger["density_volume_factor_count_per_edge"] == 1
    assert ledger["factorial_count_per_row"] == 1
    assert ledger["shared_bootstrap_canonical_column_order_required"] is True
    assert ledger["nonzero_cross_block_covariance_propagation_required"] is True
    assert ledger["finite_oracle_boundary_adapter_derived_and_source_bound"] is True
    assert ledger["implementation_status"] == (
        "runtime-and-enumeration-closure-implemented-real-sampling-pending"
    )
    assert status["public_solvfe_runtime"] == "fail-closed"
    assert status["route_a_more_accurate_than_route2_claim_allowed"] is False
    assert status["next_missing_stage"] == (
        "freeze-bulk-water-hamiltonian-and-run-real-v3-packing-occupancy-cluster-replicas"
    )


def test_v3_contracts_point_only_to_existing_runtime_primitives():
    artifacts = (
        _load(CONDITIONING_PATH),
        _load(PACKING_PATH),
        _load(LEDGER_PATH),
    )

    for artifact in artifacts:
        for implementation in artifact["runtime_implementations"].values():
            relative_path, separator, symbol = implementation.partition("::")
            assert separator == "::"
            assert symbol
            assert (PROJECT_ROOT / relative_path).is_file()


def test_v3_contracts_bind_reviewed_active_support_and_provenance_repairs():
    conditioning = _load(CONDITIONING_PATH)
    packing = _load(PACKING_PATH)
    ledger = _load(LEDGER_PATH)

    assert conditioning["contract_version"] == "3.1.0"
    assert "surface_smoothing_angstrom" in conditioning[
        "same_measure_bindings"
    ]
    assert conditioning["scientific_boundaries"][
        "pre_v3_1_nearest_surface_artifacts_compatible"
    ] is False
    assert conditioning["membership"][
        "periodic_cut_locus_continuity_required"
    ] is True
    assert "relative log-weight tolerance 50" in conditioning[
        "membership"
    ]["periodic_image_adapter"]
    assert packing["contract_version"] == "3.1.0"
    assert "log-sum-exp smooth union" in packing["schedule"]["surface_measure"]
    assert "minimum-image cut-locus switching" in packing[
        "schedule"
    ]["surface_measure"]
    assert "nmax>=1" in packing["schedule"]["active_occupancy_max"]
    assert "exactly zero" in packing["schedule"]["full_support_zero_tail_allowed"]
    assert packing["scientific_boundaries"]["production_nmax_zero_allowed"] is False

    exact = ledger["exact_vs_approximate"]
    assert (
        exact["placeholder_or_caller_selected_finite_boundary_adapter_allowed"]
        is False
    )
    assert (
        exact["finite_boundary_adapter_bound_into_source_and_hamiltonian_bridge_hashes"]
        is True
    )
    covariance = ledger["covariance"]
    assert "canonical labeled ledger order" in covariance[
        "shared_bootstrap_column_order"
    ]
    assert "must alter propagated ledger uncertainty" in covariance[
        "nonzero_cross_block_propagation"
    ]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["conditioning_measure"].__setitem__(
            "nonmember_weight", "independent-field"
        ),
        lambda value: value["conditioning_measure"].__setitem__(
            "same_membership_hash_required_for_all_terms", False
        ),
        lambda value: value["conditioning_measure"].__setitem__(
            "active_occupancy_max_minimum", 0
        ),
        lambda value: value["packing"].__setitem__(
            "row_position_assumptions_allowed", True
        ),
        lambda value: value["packing"].__setitem__(
            "water_hamiltonian_status", "silently-inherited"
        ),
        lambda value: value["fixed_n_association"].__setitem__(
            "hard_conditioned_partition_function_claim_allowed", True
        ),
        lambda value: value["soft_occupancy_and_closure"].__setitem__(
            "density_volume_factor_count_per_edge", 2
        ),
        lambda value: value["soft_occupancy_and_closure"].__setitem__(
            "factorial_count_per_row", 2
        ),
        lambda value: value["soft_occupancy_and_closure"].__setitem__(
            "nonzero_cross_block_covariance_propagation_required", False
        ),
        lambda value: value["soft_occupancy_and_closure"].__setitem__(
            "finite_oracle_boundary_adapter_derived_and_source_bound", False
        ),
        lambda value: value["outer_continuum"].__setitem__(
            "frame_deletion_allowed", True
        ),
        lambda value: value["scientific_status"].__setitem__(
            "route_a_more_accurate_than_route2_claim_allowed", True
        ),
    ],
)
def test_protocol_v3_schema_rejects_weakened_scientific_contract(mutator):
    mutated = copy.deepcopy(_load(PROTOCOL_PATH))
    mutator(mutated)

    with pytest.raises(Exception):
        Draft202012Validator(_load(SCHEMA_PATH)).validate(mutated)


def test_protocol_v3_thermodynamics_documents_measure_and_claim_boundaries():
    text = THERMODYNAMICS_PATH.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "one complementary membership definition everywhere" in normalized
    assert "p0_soft = <exp(-beta U_empty)>_0" in text
    assert "neither the first row nor the last row" in text
    assert "hard empty-shell count may be reported only as a diagnostic" in text
    assert "MACE-OFF24 or MACE-OMOL cannot enter this row silently" in text
    assert "do not promote the fixed-`n=1` acetone value" in normalized
    assert "Public `#solvfe` therefore remains fail-closed" in text
    assert "mu_A = -RT ln(p0_soft)" in text
    assert "No `p_tilde(n)` factor is then added to this row" in text
    assert "Deliberately applying" in text
    assert "Production packing and occupancy artifacts require `nmax >= 1`" in text
    assert "nonzero cross-block covariance" in text
    assert "finite periodic boundary adapter is likewise canonical" in text
    assert "force switch at weighted Voronoi seams" in text
    assert "minimum-image" in text
    assert "Cut-locus finite differences and NVE crossings" in text
    assert "Pre-v3.1 nearest-center artifacts are incompatible" in text


def test_protocol_loader_rejects_v3_hash_drift(tmp_path: Path):
    protocol = _load(PROTOCOL_PATH)
    protocol["packing"]["row_position_assumptions_allowed"] = True
    drifted = tmp_path / "protocol-v3.json"
    drifted.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ProtocolIntegrityError, match="self hash mismatch"):
        RouteAProtocol.load(drifted, project_root=PROJECT_ROOT)
