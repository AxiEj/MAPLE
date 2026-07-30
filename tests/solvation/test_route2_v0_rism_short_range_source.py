from __future__ import annotations

import json

import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_rism_short_range_source import (
    V0_RISM_SHORT_RANGE_DERIVATION,
    V0_RISM_SHORT_RANGE_SOURCE_CONSTRUCTION,
    V0_RISM_SHORT_RANGE_SOURCE_SCOPE,
    parse_route2_v0_rism_short_range_source,
)


def _payload() -> dict:
    return {
        "construction": V0_RISM_SHORT_RANGE_SOURCE_CONSTRUCTION,
        "source_scope": V0_RISM_SHORT_RANGE_SOURCE_SCOPE,
        "derivation": V0_RISM_SHORT_RANGE_DERIVATION,
        "closure": "PSE3",
        "temperature_kelvin": 298.0,
        "pressure_bar": 1.0,
        "coulomb_tail_start_angstrom": 6.0,
        "coulomb_tail_tolerance_dimensionless": 1.0e-12,
        "target_solvation_labels_used": False,
        "source_sha256": {
            "site_model": "1" * 64,
            "rism1d_input": "2" * 64,
            "xvv": "3" * 64,
            "cvv": "4" * 64,
            "thermodynamic_output": "5" * 64,
            "provenance_statement": "6" * 64,
        },
    }


def test_short_range_source_parses_strict_solvent_side_certificate():
    source = parse_route2_v0_rism_short_range_source(json.dumps(_payload()))

    assert source.derivation == V0_RISM_SHORT_RANGE_DERIVATION
    assert source.source_scope == "solvent-side"
    assert source.source_sha256["cvv"] == "4" * 64
    with pytest.raises(TypeError):
        source.source_sha256["cvv"] = "0" * 64


def test_short_range_source_rejects_target_labels_and_mace_fields():
    payload = _payload()
    payload["target_solvation_labels_used"] = True
    with pytest.raises(ValueError, match="exclude target labels"):
        parse_route2_v0_rism_short_range_source(json.dumps(payload))

    payload = _payload()
    payload["mace_checkpoint"] = "not-a-solvent-side-field"
    with pytest.raises(ValueError, match="extra=.*mace_checkpoint"):
        parse_route2_v0_rism_short_range_source(json.dumps(payload))


def test_short_range_source_rejects_incomplete_or_invalid_hashes():
    payload = _payload()
    del payload["source_sha256"]["cvv"]
    with pytest.raises(ValueError, match="missing=.*cvv"):
        parse_route2_v0_rism_short_range_source(json.dumps(payload))

    payload = _payload()
    payload["source_sha256"]["cvv"] = "not-a-hash"
    with pytest.raises(ValueError, match="lowercase digest"):
        parse_route2_v0_rism_short_range_source(json.dumps(payload))
