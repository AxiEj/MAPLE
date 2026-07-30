from __future__ import annotations

import copy
import json

import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_bulk_liquid_state_source import (
    V0_BULK_LIQUID_STATE_REQUIRED_PROPERTIES,
    V0_BULK_LIQUID_STATE_SOURCE_CONSTRUCTION,
    V0_BULK_LIQUID_STATE_SOURCE_STATUS,
    Route2V0BulkLiquidStateSource,
    parse_route2_v0_bulk_liquid_state_source,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_rism_molecular_source import (
    Route2V0Rism1dInput,
)


def _property_source(name: str) -> dict[str, str]:
    return {
        "property": name,
        "origin": "upstream_model_validation",
        "document_url": "https://example.org/model-state.pdf",
        "document_sha256": "a" * 64,
        "source_locator": f"Table 1, {name}",
        "retrieved_utc": "2026-07-30T00:00:00Z",
    }


def _payload() -> dict:
    return {
        "protocol_id": V0_BULK_LIQUID_STATE_SOURCE_CONSTRUCTION,
        "schema_version": 1,
        "status": V0_BULK_LIQUID_STATE_SOURCE_STATUS,
        "solvent_id": "fixture-solvent",
        "model": {
            "identifier": "fixture-all-atom-v1",
            "source_sha256": "b" * 64,
        },
        "state": {
            "temperature_kelvin": 298.15,
            "pressure_bar": 1.0,
            "molecular_number_density_angstrom3": 0.033,
            "static_dielectric_constant": 20.0,
            "optical_dielectric_constant": 1.8,
            "isothermal_compressibility_pa_inverse": 4.0e-10,
            "surface_tension_newton_per_meter": 0.025,
        },
        "property_sources": [
            _property_source(name)
            for name in (
                "temperature_kelvin",
                "pressure_bar",
                "molecular_number_density_angstrom3",
                "static_dielectric_constant",
                "optical_dielectric_constant",
                "isothermal_compressibility_pa_inverse",
                "surface_tension_newton_per_meter",
            )
        ],
        "no_target_policy": {
            "post_training": False,
            "fine_tuning": False,
            "experimental_solvation_fit": False,
            "map_or_uq_calibration": False,
            "target_solvation_labels_used": False,
        },
        "claim_boundary": (
            "This source binds independently declared pure-liquid state values "
            "to one molecular model, but it is not a molecular liquid functional."
        ),
        "not_claimed": [
            "A static state record does not determine finite-k direct correlations.",
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


def test_bulk_state_source_is_content_addressed_and_remains_source_only():
    source = parse_route2_v0_bulk_liquid_state_source(json.dumps(_payload()))

    assert isinstance(source, Route2V0BulkLiquidStateSource)
    assert source.construction == V0_BULK_LIQUID_STATE_SOURCE_CONSTRUCTION
    assert source.status == V0_BULK_LIQUID_STATE_SOURCE_STATUS
    assert source.solvent_id == "fixture-solvent"
    assert source.model_source_sha256 == "b" * 64
    assert source.is_molecular_liquid_asset is False
    source.verify_model_source_digest("b" * 64)
    with pytest.raises(ValueError, match="model source digest"):
        source.verify_model_source_digest("c" * 64)
    assert source.property_source_for("surface_tension_newton_per_meter").origin == (
        "upstream_model_validation"
    )

    structure_factor = source.number_structure_factor_zero_mode
    assert structure_factor > 0.0
    assert (
        source.number_channel_direct_correlation_zero_mode_angstrom3
        == pytest.approx(
            (1.0 - 1.0 / structure_factor) / source.molecular_number_density_angstrom3
        )
    )


def test_bulk_state_source_crosschecks_only_the_rism_state_it_can_observe():
    source = parse_route2_v0_bulk_liquid_state_source(json.dumps(_payload()))

    source.verify_rism1d_state(_rism_input())

    with pytest.raises(ValueError, match="dielectric"):
        source.verify_rism1d_state(_rism_input(dielectric=20.1))


def test_bulk_state_source_rejects_target_learning_or_incomplete_property_provenance():
    payload = _payload()
    payload["no_target_policy"]["fine_tuning"] = True
    with pytest.raises(ValueError, match="policy flags must be false"):
        parse_route2_v0_bulk_liquid_state_source(json.dumps(payload))

    payload = _payload()
    payload["property_sources"] = payload["property_sources"][:-1]
    with pytest.raises(ValueError, match="must cover exactly"):
        parse_route2_v0_bulk_liquid_state_source(json.dumps(payload))

    payload = _payload()
    payload["property_sources"][0]["document_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="SHA-256"):
        parse_route2_v0_bulk_liquid_state_source(json.dumps(payload))


def test_bulk_state_source_rejects_nonphysical_or_inconsistent_macroscopic_state():
    payload = _payload()
    payload["state"]["optical_dielectric_constant"] = 20.1
    with pytest.raises(ValueError, match="(?i)optical dielectric"):
        parse_route2_v0_bulk_liquid_state_source(json.dumps(payload))

    payload = copy.deepcopy(_payload())
    payload["state"]["isothermal_compressibility_pa_inverse"] = 0.0
    with pytest.raises(ValueError, match="compressibility"):
        parse_route2_v0_bulk_liquid_state_source(json.dumps(payload))


def test_bulk_state_preregistration_cannot_promote_scalars_to_a_liquid_asset():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    benchmarks = root / "docs/implicit-solvation/benchmarks"
    preregistration = json.loads(
        (benchmarks / "route2-v0-bulk-liquid-state-prereg-v1.json").read_text(
            encoding="utf-8"
        )
    )
    inventory = json.loads(
        (benchmarks / "route2-v0-solvent-asset-inventory-v1.json").read_text(
            encoding="utf-8"
        )
    )
    admission = json.loads(
        (benchmarks / "route2-v0-structured-solvent-admission-v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert preregistration["construction"]["source_construction"] == (
        V0_BULK_LIQUID_STATE_SOURCE_CONSTRUCTION
    )
    assert preregistration["construction"]["source_status"] == (
        V0_BULK_LIQUID_STATE_SOURCE_STATUS
    )
    assert set(preregistration["required_state_fields"]) == (
        V0_BULK_LIQUID_STATE_REQUIRED_PROPERTIES
    )
    assert preregistration["current_inventory"]["physical_state_source_records"] == []
    assert preregistration["hard_constraints"] == {
        "official_checkpoint_unmodified": True,
        "post_training": False,
        "fine_tuning": False,
        "experimental_solvation_fit": False,
        "map_or_uq_calibration": False,
        "target_solvation_labels_used": False,
        "error_driven_property_selection": False,
        "dielectric_only_total_free_energy": False,
    }
    assert (
        "delta C_ab(k)"
        in preregistration["construction"]["finite_k_nonidentifiability"]
    )

    source_contract = inventory["bulk_liquid_state_source_contract"]
    assert source_contract["physical_state_source_records"] == []
    assert source_contract["protocol"] == "route2-v0-bulk-liquid-state-prereg-v1.json"
    assert "finite-k direct correlations" in source_contract["not_an_admitted_asset"]

    solvent_contract = admission["solvent_asset_contract"]
    assert "bulk-state certificate" in " ".join(
        solvent_contract["required_before_target_scoring"]
    )
    foundation = admission["implemented_foundation"][
        "bulk_liquid_state_source_registry"
    ]
    assert foundation["module"].endswith("route2_v0_bulk_liquid_state_source")
    assert "No physical state source record" in foundation["current_inventory"]
    assert "C_ab(k)" in foundation["source_boundary"]
