from __future__ import annotations

import copy
import json

import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_bulk_liquid_state_source import (
    V0_BULK_LIQUID_STATE_SOURCE_CONSTRUCTION,
    V0_BULK_LIQUID_STATE_SOURCE_STATUS,
    parse_route2_v0_bulk_liquid_state_source,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_lorentz_dielectric_source import (
    V0_LORENTZ_DIELECTRIC_SOURCE_CONSTRUCTION,
    V0_LORENTZ_DIELECTRIC_SOURCE_STATUS,
    Route2V0LorentzNonlocalDielectricSource,
    parse_route2_v0_lorentz_nonlocal_dielectric_source,
)


def _bulk_payload() -> dict:
    property_names = (
        "temperature_kelvin",
        "pressure_bar",
        "molecular_number_density_angstrom3",
        "static_dielectric_constant",
        "optical_dielectric_constant",
        "isothermal_compressibility_pa_inverse",
        "surface_tension_newton_per_meter",
    )
    return {
        "protocol_id": V0_BULK_LIQUID_STATE_SOURCE_CONSTRUCTION,
        "schema_version": 1,
        "status": V0_BULK_LIQUID_STATE_SOURCE_STATUS,
        "solvent_id": "lorentz-fixture",
        "model": {
            "identifier": "lorentz-fixture-all-atom-v1",
            "source_sha256": "a" * 64,
        },
        "state": {
            "temperature_kelvin": 298.15,
            "pressure_bar": 1.0,
            "molecular_number_density_angstrom3": 0.025,
            "static_dielectric_constant": 30.0,
            "optical_dielectric_constant": 1.8,
            "isothermal_compressibility_pa_inverse": 5.0e-10,
            "surface_tension_newton_per_meter": 0.03,
        },
        "property_sources": [
            {
                "property": name,
                "origin": "upstream_model_validation",
                "document_url": "https://example.org/lorentz-state.pdf",
                "document_sha256": "b" * 64,
                "source_locator": f"Table 1, {name}",
                "retrieved_utc": "2026-07-30T00:00:00Z",
            }
            for name in property_names
        ],
        "no_target_policy": {
            "post_training": False,
            "fine_tuning": False,
            "experimental_solvation_fit": False,
            "map_or_uq_calibration": False,
            "target_solvation_labels_used": False,
        },
        "claim_boundary": "Fixture state only; never a liquid endpoint.",
        "not_claimed": ["It has no finite-wavevector molecular correlation."],
    }


def _response_payload() -> dict:
    return {
        "protocol_id": V0_LORENTZ_DIELECTRIC_SOURCE_CONSTRUCTION,
        "schema_version": 1,
        "status": V0_LORENTZ_DIELECTRIC_SOURCE_STATUS,
        "solvent_id": "lorentz-fixture",
        "model": {
            "identifier": "lorentz-fixture-all-atom-v1",
            "source_sha256": "a" * 64,
        },
        "response": {
            "static_dielectric_constant": 30.0,
            "optical_dielectric_constant": 1.8,
            "orientational_correlation_length_bohr": 0.65,
        },
        "correlation_length_source": {
            "origin": "ab_initio",
            "document_url": "https://example.org/lorentz-correlation.pdf",
            "document_sha256": "c" * 64,
            "source_locator": "Figure 3, longitudinal response fit",
            "retrieved_utc": "2026-07-30T00:00:00Z",
        },
        "no_target_policy": {
            "post_training": False,
            "fine_tuning": False,
            "experimental_solvation_fit": False,
            "map_or_uq_calibration": False,
            "target_solvation_labels_used": False,
        },
        "claim_boundary": "Fixture response only; never a liquid endpoint.",
        "not_claimed": ["It has no cavity or molecular liquid functional."],
    }


def test_lorentz_response_source_is_content_addressed_and_state_bound():
    bulk = parse_route2_v0_bulk_liquid_state_source(json.dumps(_bulk_payload()))
    response = parse_route2_v0_lorentz_nonlocal_dielectric_source(
        json.dumps(_response_payload())
    )

    assert isinstance(response, Route2V0LorentzNonlocalDielectricSource)
    assert response.construction == V0_LORENTZ_DIELECTRIC_SOURCE_CONSTRUCTION
    assert response.status == V0_LORENTZ_DIELECTRIC_SOURCE_STATUS
    assert response.is_molecular_liquid_asset is False
    assert response.orientational_correlation_length_bohr == pytest.approx(0.65)
    assert response.correlation_length_source.origin == "ab_initio"
    response.verify_bulk_liquid_state_source(bulk)


def test_lorentz_response_source_rejects_target_learning_and_incomplete_provenance():
    payload = _response_payload()
    payload["no_target_policy"]["fine_tuning"] = True
    with pytest.raises(ValueError, match="policy flags must be false"):
        parse_route2_v0_lorentz_nonlocal_dielectric_source(json.dumps(payload))

    payload = _response_payload()
    del payload["correlation_length_source"]["document_sha256"]
    with pytest.raises(ValueError, match="keys differ"):
        parse_route2_v0_lorentz_nonlocal_dielectric_source(json.dumps(payload))

    payload = _response_payload()
    payload["correlation_length_source"]["document_url"] = "http://example.org/x"
    with pytest.raises(ValueError, match="HTTPS"):
        parse_route2_v0_lorentz_nonlocal_dielectric_source(json.dumps(payload))


def test_lorentz_response_source_rejects_nonphysical_or_mismatched_state():
    payload = _response_payload()
    payload["response"]["orientational_correlation_length_bohr"] = 0.0
    with pytest.raises(ValueError, match="correlation length"):
        parse_route2_v0_lorentz_nonlocal_dielectric_source(json.dumps(payload))

    payload = _response_payload()
    payload["response"]["optical_dielectric_constant"] = 30.1
    with pytest.raises(ValueError, match="optical dielectric"):
        parse_route2_v0_lorentz_nonlocal_dielectric_source(json.dumps(payload))

    response = parse_route2_v0_lorentz_nonlocal_dielectric_source(
        json.dumps(_response_payload())
    )
    bulk_payload = copy.deepcopy(_bulk_payload())
    bulk_payload["state"]["static_dielectric_constant"] = 29.9
    bulk = parse_route2_v0_bulk_liquid_state_source(json.dumps(bulk_payload))
    with pytest.raises(ValueError, match="dielectric limits disagree"):
        response.verify_bulk_liquid_state_source(bulk)

    bulk_payload = copy.deepcopy(_bulk_payload())
    bulk_payload["model"]["source_sha256"] = "d" * 64
    bulk = parse_route2_v0_bulk_liquid_state_source(json.dumps(bulk_payload))
    with pytest.raises(ValueError, match="model identity disagree"):
        response.verify_bulk_liquid_state_source(bulk)


def test_lorentz_response_source_allows_only_the_unidentified_constant_limit():
    payload = _response_payload()
    payload["response"] = {
        "static_dielectric_constant": 1.8,
        "optical_dielectric_constant": 1.8,
        "orientational_correlation_length_bohr": None,
    }
    payload["correlation_length_source"] = None
    response = parse_route2_v0_lorentz_nonlocal_dielectric_source(
        json.dumps(payload)
    )
    assert response.orientational_correlation_length_bohr is None
    assert response.correlation_length_source is None

    payload = _response_payload()
    payload["response"] = {
        "static_dielectric_constant": 1.8,
        "optical_dielectric_constant": 1.8,
        "orientational_correlation_length_bohr": 0.65,
    }
    with pytest.raises(ValueError, match="unidentifiable"):
        parse_route2_v0_lorentz_nonlocal_dielectric_source(json.dumps(payload))

    payload = _response_payload()
    payload["response"] = {
        "static_dielectric_constant": 1.8,
        "optical_dielectric_constant": 1.8,
        "orientational_correlation_length_bohr": None,
    }
    with pytest.raises(ValueError, match="must not carry a correlation-length source"):
        parse_route2_v0_lorentz_nonlocal_dielectric_source(json.dumps(payload))
