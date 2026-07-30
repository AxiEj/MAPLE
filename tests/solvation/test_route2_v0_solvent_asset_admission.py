from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_solvent_asset import (
    V0_DEFAULT_SOLVENT_IDS,
    V0_FROZEN_SOLVENT_ASSET_CONSTRUCTION,
    load_route2_v0_frozen_solvent_registry,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
ADMISSION = BENCHMARKS / "route2-v0-structured-solvent-admission-v1.json"
INVENTORY = BENCHMARKS / "route2-v0-solvent-asset-inventory-v1.json"
CSPCE_CANDIDATE = BENCHMARKS / "route2-v0-solvent-assets/water-cspce-pse3/manifest.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_solvent_asset_registry_is_provenance_only_and_no_fit():
    protocol = _json(ADMISSION)
    contract = protocol["solvent_asset_contract"]
    registry = protocol["implemented_foundation"]["frozen_solvent_asset_registry"]

    assert V0_FROZEN_SOLVENT_ASSET_CONSTRUCTION in contract["registry_protocol"]
    assert (
        "zero physical default-solvent assets" in contract["current_inventory_policy"]
    )
    assert registry["module"].endswith("route2_v0_solvent_asset")
    assert "SHA-256" in registry["capability"]
    assert "canonical rigid molecular geometry" in registry["capability"]
    assert "XVV MTV" in registry["capability"]
    assert (
        "MNSol, FreeSolv, development, confirmation, and blind"
        in registry["capability"]
    )
    assert registry["inventory_artifact"] == INVENTORY.name
    assert (
        "does not supply the MACE-native short-range"
        in registry["not_a_physical_liquid_backend"]
    )


def test_current_solvent_inventory_fails_closed_until_all_assets_exist():
    inventory = _json(INVENTORY)

    assert inventory["status"] == "inventory-only-zero-admitted-physical-default-assets"
    assert tuple(inventory["required_default_solvents"]) == V0_DEFAULT_SOLVENT_IDS
    assert inventory["admitted_physical_default_assets"] == []
    assert inventory["missing_default_assets"] == list(V0_DEFAULT_SOLVENT_IDS)
    assert inventory["available_parser_control"]["model"] == "cSPCE"
    assert (
        "Neither this legacy control nor the separate source-complete candidate"
        in inventory["available_parser_control"]["not_an_admitted_asset"]
    )
    candidates = inventory["source_complete_candidate_assets"]
    assert [candidate["solvent"] for candidate in candidates] == ["water"]
    assert candidates[0]["manifest"] == (
        "route2-v0-solvent-assets/water-cspce-pse3/manifest.json"
    )
    assert "not-production-or-accuracy-admitted" in candidates[0]["status"]
    assert inventory["hard_constraints"] == {
        "post_training": False,
        "fine_tuning": False,
        "experimental_solvation_fit": False,
        "map_or_uq_calibration": False,
        "amber_gaff_or_am1bcc_solute_substitution": False,
    }
    assert "Total-free-energy execution is rejected" in inventory["execution_policy"]


def test_v0_aq_liquid_provider_audit_does_not_convert_local_tools_into_a_proxy():
    inventory = _json(INVENTORY)
    audit = inventory["v0_aq_l_provider_audit"]

    assert "host-local availability audit" in audit["scope"]
    jdftx = audit["jdftx_classical_dft"]
    assert jdftx["checked_executables"] == ["jdftx", "jdftx_gpu"]
    assert jdftx["status"] == "not-installed-on-this-host-path"
    assert "separately authorized, version-pinned installation" in jdftx["consequence"]
    amber = audit["ambertools_rism"]
    assert amber["runtime"].startswith("AmberTools 26.0")
    assert "--pdb, --prmtop, and --xvv" in amber["stock_rism3d_requirement"]
    assert "cannot be relabelled as a MACE-native" in amber["stock_rism3d_requirement"]
    assert "only the cSPCE bulk control" in amber["bundled_bulk_models"]
    assert "not an admissible V0-AQ-L physical endpoint" in amber["consequence"]
    assert "No local executable plus admitted eleven-solvent" in audit["decision"]
    assert "GAFF/AM1-BCC" in audit["decision"]


def test_checked_in_cspce_source_candidate_is_exact_but_not_panel_admitted():
    registry = load_route2_v0_frozen_solvent_registry(CSPCE_CANDIDATE)
    water = registry.asset_for("water")

    assert water.model_identifier == "AmberTools26-cSPCE-PSE3-298K-source-candidate"
    assert water.closure == "PSE3"
    assert water.temperature_kelvin == pytest.approx(298.0)
    assert water.file_for("site_model").sha256 == (
        "15464c0c53e7a2e774438316853c7381dcd6ab5fa9b098c29d73bf3c8479b0c1"
    )
    assert water.short_range_source.target_solvation_labels_used is False
    assert water.generation_source.status == (
        "source-complete-candidate-not-production-or-accuracy-admitted"
    )
    assert len(water.generation_source.runs) == 2
    assert {run.primary_iterations for run in water.generation_source.runs} == {130}
    assert {
        run.temperature_derivative_iterations for run in water.generation_source.runs
    } == {36}
    assert (
        max(run.primary_final_residual for run in water.generation_source.runs)
        < water.generation_source.residual_tolerance
    )
    assert (
        max(
            run.temperature_derivative_final_residual
            for run in water.generation_source.runs
        )
        < water.generation_source.residual_tolerance
    )
    assert water.generation_source.normalized_xvv_sha256 == (
        "b0047215f6b62d53edf72e0c674f8954402259d957f1ffe7375245e29e8a1c80"
    )
    assert water.thermodynamic_source.sm_identity_relative_residual == pytest.approx(
        -3.1105111541542655e-12
    )
    np.testing.assert_array_equal(
        water.molecular_source.site_charges_e,
        water.molecular_reference.site_charges_e,
    )
    assert float(np.sum(water.molecular_source.site_charges_e)) == 0.0
    assert abs(float(np.sum(water.molecular_source.serialized_site_charges_e))) < 2e-9
    assert (
        water.bulk_direct_correlation.coulomb_tail_residual(
            minimum_radius_angstrom=water.coulomb_tail_start_angstrom
        )
        < water.coulomb_tail_tolerance_dimensionless
    )
    registry.verify_integrity()

    with pytest.raises(ValueError, match="missing default assets"):
        registry.require_default_solvent_panel()
