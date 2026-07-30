from __future__ import annotations

import json
from pathlib import Path

import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_rism_state_source import (
    V0_MOLECULAR_RISM_BULK_STATE_REQUIRED_PROPERTIES,
    V0_MOLECULAR_RISM_BULK_STATE_SOURCE_CONSTRUCTION,
    V0_MOLECULAR_RISM_BULK_STATE_SOURCE_STATUS,
    Route2V0MolecularRismBulkStateSource,
    load_route2_v0_molecular_rism_bulk_state_source,
    parse_route2_v0_molecular_rism_bulk_state_source,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_rism_molecular_source import (
    Route2V0Rism1dInput,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
DCM_STATE_SOURCE = (
    BENCHMARKS
    / "route2-v0-molecular-rism-state-sources"
    / "dichloromethane-scm-adf-3drism-v1.json"
)
DCM_KH_AUDIT = (
    BENCHMARKS / "route2-v0-dcm-kh-source-frozen-feasibility-audit-v1.json"
)


def _property_source(name: str) -> dict[str, str]:
    return {
        "property": name,
        "origin": "upstream_model_validation",
        "document_url": "https://example.org/molecular-rism-state.pdf",
        "document_sha256": "a" * 64,
        "source_locator": f"Table 1, {name}",
        "retrieved_utc": "2026-07-30T00:00:00Z",
    }


def _payload() -> dict[str, object]:
    state = {
        "temperature_kelvin": 298.15,
        "pressure_bar": 1.0,
        "molecular_number_density_angstrom3": 0.033,
        "static_dielectric_constant": 20.0,
        "isothermal_compressibility_pa_inverse": 4.0e-10,
        "surface_tension_newton_per_meter": 0.025,
    }
    return {
        "protocol_id": V0_MOLECULAR_RISM_BULK_STATE_SOURCE_CONSTRUCTION,
        "schema_version": 1,
        "status": V0_MOLECULAR_RISM_BULK_STATE_SOURCE_STATUS,
        "solvent_id": "fixture-solvent",
        "model": {
            "identifier": "fixture-all-atom-v1",
            "source_sha256": "b" * 64,
        },
        "state": state,
        "property_sources": [_property_source(name) for name in state],
        "no_target_policy": {
            "post_training": False,
            "fine_tuning": False,
            "experimental_solvation_fit": False,
            "map_or_uq_calibration": False,
            "target_solvation_labels_used": False,
        },
        "claim_boundary": (
            "This source binds a fixed-charge molecular-RISM state to one "
            "molecular model, but it is not a molecular liquid functional."
        ),
        "not_claimed": [
            "A scalar state record does not determine finite-k direct correlations.",
            "This source does not admit a total solvation free-energy endpoint.",
        ],
    }


def _rism_input(*, dielectric: float = 20.0) -> Route2V0Rism1dInput:
    return Route2V0Rism1dInput(
        theory="DRISM",
        closure="PSE3",
        radial_point_count=1024,
        radial_spacing_angstrom=0.025,
        temperature_kelvin=298.15,
        component_count=1,
        dielectric_constant=dielectric,
        molecular_number_density_angstrom3=0.033,
        coulomb_smear_angstrom=1.0,
        residual_tolerance=1.0e-12,
        self_test=-1,
        output_list="xc",
        maximum_steps=10000,
        model="fixture.mdl",
    )


def test_molecular_rism_state_is_content_addressed_and_has_no_optical_field():
    source = parse_route2_v0_molecular_rism_bulk_state_source(
        json.dumps(_payload())
    )

    assert isinstance(source, Route2V0MolecularRismBulkStateSource)
    assert source.construction == V0_MOLECULAR_RISM_BULK_STATE_SOURCE_CONSTRUCTION
    assert source.status == V0_MOLECULAR_RISM_BULK_STATE_SOURCE_STATUS
    assert source.solvent_id == "fixture-solvent"
    assert source.model_source_sha256 == "b" * 64
    assert source.is_molecular_liquid_asset is False
    assert "optical_dielectric_constant" not in (
        V0_MOLECULAR_RISM_BULK_STATE_REQUIRED_PROPERTIES
    )
    assert not hasattr(source, "optical_dielectric_constant")

    source.verify_model_source_digest("b" * 64)
    with pytest.raises(ValueError, match="model source digest"):
        source.verify_model_source_digest("c" * 64)
    assert source.property_source_for("surface_tension_newton_per_meter").origin == (
        "upstream_model_validation"
    )

    structure_factor = source.number_structure_factor_zero_mode
    assert structure_factor > 0.0
    assert source.number_channel_direct_correlation_zero_mode_angstrom3 == (
        pytest.approx(
            (1.0 - 1.0 / structure_factor)
            / source.molecular_number_density_angstrom3
        )
    )


def test_molecular_rism_state_crosschecks_only_rism_observables():
    source = parse_route2_v0_molecular_rism_bulk_state_source(
        json.dumps(_payload())
    )

    source.verify_rism1d_state(_rism_input())

    with pytest.raises(ValueError, match="static dielectric"):
        source.verify_rism1d_state(_rism_input(dielectric=20.1))


def test_molecular_rism_state_rejects_optical_proxy_target_learning_and_bad_sources():
    payload = _payload()
    state = payload["state"]
    assert isinstance(state, dict)
    state["optical_dielectric_constant"] = 1.8
    with pytest.raises(ValueError, match="liquid state keys differ"):
        parse_route2_v0_molecular_rism_bulk_state_source(json.dumps(payload))

    payload = _payload()
    policy = payload["no_target_policy"]
    assert isinstance(policy, dict)
    policy["fine_tuning"] = True
    with pytest.raises(ValueError, match="policy flags must be false"):
        parse_route2_v0_molecular_rism_bulk_state_source(json.dumps(payload))

    payload = _payload()
    property_sources = payload["property_sources"]
    assert isinstance(property_sources, list)
    property_sources[-1]["property"] = "optical_dielectric_constant"
    with pytest.raises(ValueError, match="Unsupported molecular-RISM"):
        parse_route2_v0_molecular_rism_bulk_state_source(json.dumps(payload))

    payload = _payload()
    property_sources = payload["property_sources"]
    assert isinstance(property_sources, list)
    property_sources.pop()
    with pytest.raises(ValueError, match="must cover exactly"):
        parse_route2_v0_molecular_rism_bulk_state_source(json.dumps(payload))


def test_checked_in_dichloromethane_state_is_source_only_and_model_bound():
    source = load_route2_v0_molecular_rism_bulk_state_source(DCM_STATE_SOURCE)
    payload = json.loads(DCM_STATE_SOURCE.read_text(encoding="utf-8"))

    assert source.solvent_id == "dichloromethane"
    assert source.model_source_sha256 == (
        "efacc15b6d9fed5ba28be58916f66e76b1087149ffe7b2e261f98f4a913343ed"
    )
    assert source.temperature_kelvin == pytest.approx(298.15)
    assert source.pressure_bar == pytest.approx(1.01325)
    assert source.molecular_number_density_angstrom3 == pytest.approx(
        0.009338466243871424
    )
    assert source.static_dielectric_constant == pytest.approx(8.93)
    assert source.isothermal_compressibility_pa_inverse == pytest.approx(
        1.0264001973846534e-08
    )
    assert source.surface_tension_newton_per_meter == pytest.approx(0.0272)
    assert source.is_molecular_liquid_asset is False
    assert {record.property_name for record in source.property_sources} == (
        V0_MOLECULAR_RISM_BULK_STATE_REQUIRED_PROPERTIES
    )
    assert all(
        value is False for value in payload["no_target_policy"].values()
    )
    assert "not a finite-k molecular susceptibility" in source.claim_boundary
    assert "optical dielectric constant" in " ".join(source.not_claimed)


def test_molecular_rism_preregistration_preserves_state_domain_separation():
    preregistration = json.loads(
        (
            BENCHMARKS / "route2-v0-molecular-rism-state-prereg-v1.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (BENCHMARKS / "route2-v0-solvent-asset-inventory-v1.json").read_text(
            encoding="utf-8"
        )
    )
    admission = json.loads(
        (
            BENCHMARKS / "route2-v0-structured-solvent-admission-v1.json"
        ).read_text(encoding="utf-8")
    )

    assert set(preregistration["required_state_fields"]) == (
        V0_MOLECULAR_RISM_BULK_STATE_REQUIRED_PROPERTIES
    )
    assert preregistration["hard_constraints"]["optical_dielectric_proxy_for_molecular_rism"] is False
    assert preregistration["current_inventory"]["physical_molecular_rism_assets"] == []
    assert preregistration["current_inventory"]["source_only_molecular_rism_state_records"] == [
        "route2-v0-molecular-rism-state-sources/dichloromethane-scm-adf-3drism-v1.json"
    ]
    assert "does not identify a molecular site/orientational" in preregistration[
        "construction"
    ]["zero_mode_constraint"]

    contract = inventory["molecular_rism_bulk_state_source_contract"]
    assert contract["source_only_state_records"] == [
        "route2-v0-molecular-rism-state-sources/dichloromethane-scm-adf-3drism-v1.json"
    ]
    assert contract["physical_molecular_rism_assets"] == []
    assert "epsilon(infinity)" in contract["state_domain_boundary"]

    foundation = admission["implemented_foundation"][
        "molecular_rism_bulk_state_source_registry"
    ]
    assert foundation["module"].endswith("route2_v0_molecular_rism_state_source")
    assert "fixed-charge" in foundation["capability"]
    assert "C_ab(k)" in foundation["source_boundary"]


def test_source_frozen_dcm_kh_probe_is_rejected_without_parameter_search():
    audit = json.loads(DCM_KH_AUDIT.read_text(encoding="utf-8"))

    assert audit["protocol_id"] == "route2-v0-dcm-kh-source-frozen-feasibility-audit-v1"
    assert audit["status"] == (
        "rejected-source-frozen-kh-drism-numerical-probe-not-a-liquid-asset"
    )
    assert "not a finite-k liquid asset" in audit["claim_boundary"]

    binding = audit["source_binding"]
    assert binding["molecular_state_record"] == (
        "route2-v0-molecular-rism-state-sources/"
        "dichloromethane-scm-adf-3drism-v1.json"
    )
    assert binding["generated_mdl_sha256"] == (
        "efacc15b6d9fed5ba28be58916f66e76b1087149ffe7b2e261f98f4a913343ed"
    )
    assert len(binding["rism1d_binary_sha256"]) == 64

    source_input = audit["source_frozen_input"]
    assert source_input["theory"] == "DRISM"
    assert source_input["closure"] == "KH"
    assert source_input["radial_point_count"] == 16384
    assert source_input["radial_spacing_angstrom"] == pytest.approx(0.025)
    assert source_input["residual_tolerance"] == pytest.approx(1.0e-12)
    assert source_input["maximum_steps"] == 10000
    assert source_input["mdiis_nvec"] == 20
    assert source_input["mdiis_del"] == pytest.approx(0.3)
    state = load_route2_v0_molecular_rism_bulk_state_source(DCM_STATE_SOURCE)
    assert binding["generated_mdl_sha256"] == state.model_source_sha256
    assert source_input["molecular_density_molar"] == pytest.approx(
        state.molecular_number_density_angstrom3 / 6.02214076e-4
    )

    observed = audit["observed_numerics"]
    assert "auditor-terminated" in observed["termination"]
    assert observed["minimum_residual"] > source_input["residual_tolerance"]
    assert observed["cycle_start_step"] == 27
    assert observed["complete_cycle_count"] == 689
    cycle = observed["residual_cycle"]
    witness = observed["cycle_witness_steps"]
    assert [item["residual"] for item in witness[-6:]] == cycle * 2

    assert all(value is False for value in audit["no_target_policy"].values())
    assert "does not authorize a mixing sweep" in audit["decision"]
    assert "closure switch chosen from solvation error" in audit["decision"]
    assert "emitted no XVV/Cvv output" in audit["decision"]

    preregistration = json.loads(
        (
            BENCHMARKS / "route2-v0-molecular-rism-state-prereg-v1.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (BENCHMARKS / "route2-v0-solvent-asset-inventory-v1.json").read_text(
            encoding="utf-8"
        )
    )
    audit_name = DCM_KH_AUDIT.name
    assert audit_name in preregistration["current_inventory"][
        "rejected_source_frozen_numerical_audits"
    ]
    assert audit_name in inventory["molecular_rism_bulk_state_source_contract"][
        "rejected_source_frozen_numerical_audits"
    ]
