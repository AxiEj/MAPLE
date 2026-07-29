from __future__ import annotations

import json
from pathlib import Path

from maple.function.calculator.extra_correction.implicit.route2_v0_solvent_asset import (
    V0_DEFAULT_SOLVENT_IDS,
    V0_FROZEN_SOLVENT_ASSET_CONSTRUCTION,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
ADMISSION = BENCHMARKS / "route2-v0-structured-solvent-admission-v1.json"
INVENTORY = BENCHMARKS / "route2-v0-solvent-asset-inventory-v1.json"


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
        "does not define the MACE-native short-range"
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
        "lacks a schema-v2 hash-locked canonical molecular reference"
        in inventory["available_parser_control"]["not_an_admitted_asset"]
    )
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
