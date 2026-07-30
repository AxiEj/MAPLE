from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_all_atom_solvent_model_source import (
    V0_ALL_ATOM_SOLVENT_MODEL_SOURCE_STATUS,
    load_route2_v0_all_atom_solvent_model_source,
)
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
MODEL_SOURCES = BENCHMARKS / "route2-v0-solvent-model-sources"


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
    source_only = inventory["all_atom_model_source_candidates"]
    assert [candidate["solvent"] for candidate in source_only] == [
        "chloroform",
        "dichloromethane",
    ]
    assert all(
        candidate["status"] == V0_ALL_ATOM_SOLVENT_MODEL_SOURCE_STATUS
        for candidate in source_only
    )
    assert all(
        candidate["source_record"].endswith(".json") for candidate in source_only
    )
    assert all(candidate["generated_mdl"].endswith(".mdl") for candidate in source_only)
    assert all(
        "lacks independently source-bound bulk density/dielectric"
        in candidate["not_an_admitted_asset"]
        for candidate in source_only
    )
    for candidate in source_only:
        source_record = BENCHMARKS / candidate["source_record"]
        generated_mdl = BENCHMARKS / candidate["generated_mdl"]
        assert (
            hashlib.sha256(source_record.read_bytes()).hexdigest()
            == candidate["source_record_sha256"]
        )
        assert (
            hashlib.sha256(generated_mdl.read_bytes()).hexdigest()
            == candidate["generated_mdl_sha256"]
        )
    assert inventory["hard_constraints"] == {
        "post_training": False,
        "fine_tuning": False,
        "experimental_solvation_fit": False,
        "map_or_uq_calibration": False,
        "amber_gaff_or_am1bcc_solute_substitution": False,
    }
    assert "Total-free-energy execution is rejected" in inventory["execution_policy"]


def test_source_only_all_atom_models_remain_below_frozen_liquid_asset_admission():
    protocol = _json(ADMISSION)
    foundation = protocol["implemented_foundation"][
        "all_atom_solvent_model_source_registry"
    ]

    assert foundation["module"].endswith("route2_v0_all_atom_solvent_model_source")
    assert "real-element atomic number and matching mass" in foundation["capability"]
    assert "not frozen liquid assets" in foundation["current_inventory"]
    assert "do not choose a bulk density" in foundation["not_a_physical_liquid_backend"]
    assert (
        "two SCM 3D-RISM table-derived"
        in protocol["solvent_asset_contract"]["current_inventory_policy"]
    )

    for stem, solvent_id in (
        ("chloroform-scm-adf-3drism-v1", "chloroform"),
        ("dichloromethane-scm-adf-3drism-v1", "dichloromethane"),
    ):
        source = load_route2_v0_all_atom_solvent_model_source(
            MODEL_SOURCES / f"{stem}.json"
        )
        assert source.solvent_id == solvent_id
        assert source.status == V0_ALL_ATOM_SOLVENT_MODEL_SOURCE_STATUS
        assert source.to_amber_mdl_text() == (MODEL_SOURCES / f"{stem}.mdl").read_text(
            encoding="utf-8"
        )


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
