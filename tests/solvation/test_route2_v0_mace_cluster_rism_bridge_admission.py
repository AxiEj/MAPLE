from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADMISSION = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-v0-structured-solvent-admission-v1.json"
)


def test_asset_bound_mace_cluster_rism_bridge_is_explicitly_no_fit_and_nonphysical():
    protocol = json.loads(ADMISSION.read_text(encoding="utf-8"))
    foundation = protocol["implemented_foundation"]
    bridge = foundation["mace_cluster_rism_asset_bound_bridge_control"]

    assert bridge["module"].endswith("route2_v0_mace_cluster_rism_bridge")
    assert "source-SMEAR energy-conjugate RISM kernel" in bridge["capability"]
    assert "checkpoint-locked zero-field MACE cluster" in bridge["capability"]
    assert "asset canonical molecular reference" in bridge["capability"]
    assert "XVV multiplicity" in bridge["capability"]
    assert "MACE envelope-force" in bridge["capability"]
    assert (
        "MACE field-conditioned response density" in bridge["hybrid_reference_boundary"]
    )
    assert "does not claim" in bridge["hybrid_reference_boundary"]
    assert "scalar dielectric constant" in bridge["custom_solvent_boundary"]
    assert "total solvation free energy" in bridge["not_a_physical_liquid_backend"]
    assert "moving-cavity" in bridge["not_a_physical_liquid_backend"]
    assert "accuracy result" in bridge["not_a_physical_liquid_backend"]
    assert any(
        "asset-bound zero-field-MACE/RISM bridge" in item
        for item in protocol["validation_sequence"]
    )
