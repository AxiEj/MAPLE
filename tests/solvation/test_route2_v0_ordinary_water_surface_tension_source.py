from __future__ import annotations

import json
from pathlib import Path

import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_ordinary_water_surface_tension_source import (
    V0_ORDINARY_WATER_SURFACE_TENSION_SOURCE_CONSTRUCTION,
    V0_ORDINARY_WATER_SURFACE_TENSION_SOURCE_STATUS,
    Route2V0OrdinaryWaterSurfaceTensionSource,
    load_route2_v0_ordinary_water_surface_tension_source,
    parse_route2_v0_ordinary_water_surface_tension_source,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "docs/implicit-solvation/benchmarks"
    / "route2-v0-ordinary-water-surface-tension-iapws-r1-76-2014-v1.json"
)


def test_iapws_ordinary_water_surface_tension_source_is_exact_and_source_only():
    source = load_route2_v0_ordinary_water_surface_tension_source(SOURCE)

    assert isinstance(source, Route2V0OrdinaryWaterSurfaceTensionSource)
    assert source.construction == V0_ORDINARY_WATER_SURFACE_TENSION_SOURCE_CONSTRUCTION
    assert source.status == V0_ORDINARY_WATER_SURFACE_TENSION_SOURCE_STATUS
    assert source.temperature_kelvin == pytest.approx(298.0)
    assert source.critical_temperature_kelvin == pytest.approx(647.096)
    assert source.prefactor_newton_per_meter == pytest.approx(0.2358)
    assert source.exponent == pytest.approx(1.256)
    assert source.linear_coefficient == pytest.approx(-0.625)
    assert source.reduced_temperature_distance == pytest.approx(0.539481004364113)
    assert source.surface_tension_newton_per_meter == pytest.approx(0.07199532948823899)
    assert source.surface_tension_hartree_per_bohr2 > 0.0
    assert source.is_molecular_liquid_asset is False
    assert source.is_molecular_rism_state_source is False
    assert source.property_source.origin == "independent_measurement"
    assert source.property_source.document_sha256 == (
        "77d07ae4c6d473806b73f982d4c031b927db02a5718bd3c25cd7a03c3321587d"
    )
    assert "cSPC/E" in " ".join(source.not_claimed)


def test_iapws_surface_tension_source_rejects_formula_or_claim_drift():
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload["surface_tension_newton_per_meter"] = 0.072
    with pytest.raises(ValueError, match="does not reproduce"):
        parse_route2_v0_ordinary_water_surface_tension_source(json.dumps(payload))

    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload["status"] = "physical-liquid-admitted"
    with pytest.raises(ValueError, match="remain source-only"):
        parse_route2_v0_ordinary_water_surface_tension_source(json.dumps(payload))

    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload["no_target_policy"]["experimental_solvation_fit"] = True
    with pytest.raises(ValueError, match="must be false"):
        parse_route2_v0_ordinary_water_surface_tension_source(json.dumps(payload))

    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload["formula"]["exponent"] = 1.25
    with pytest.raises(ValueError, match="constants changed"):
        parse_route2_v0_ordinary_water_surface_tension_source(json.dumps(payload))

    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload["property_source"]["document_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source identity changed"):
        parse_route2_v0_ordinary_water_surface_tension_source(json.dumps(payload))

    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload["claim_boundary"] = "source-only but otherwise unspecified"
    with pytest.raises(ValueError, match="claim boundary changed"):
        parse_route2_v0_ordinary_water_surface_tension_source(json.dumps(payload))
