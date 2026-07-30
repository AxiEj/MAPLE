from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREREG = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-v0-weighted-density-bridge-prereg-v1.json"
)


def _protocol() -> dict:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def test_weighted_density_bridge_preregistration_locks_the_no_fit_boundary():
    protocol = _protocol()

    assert protocol["protocol_id"] == "route2-v0-weighted-density-bridge-prereg-v1"
    assert protocol["status"] == "preregistered-before-pure-solvent-bridge-asset"
    construction = protocol["construction"]
    assert construction["name"] == "route2-v0-molecular-weighted-density-bridge-v1"
    assert "Omega_HNC" in construction["base_scalar"]
    assert "D^T" in construction["centre_projection"]
    assert "periodic-even" in construction["weighted_density"]
    assert "(P_HNC - P_s) / rho_bulk^3" in construction["cubic_constraint"]
    assert "exact derivatives" in construction["response_rule"]

    assert protocol["hard_constraints"] == {
        "official_checkpoint_unmodified": True,
        "post_training": False,
        "fine_tuning": False,
        "experimental_solvation_fit": False,
        "map_or_uq_calibration": False,
        "error_driven_cavity_or_dispersion_adjustment": False,
        "pc_plus_or_posthoc_volume_correction": False,
        "per_record_bridge_or_kernel_selection": False,
        "legacy_public_route_changed": False,
    }


def test_weighted_density_bridge_requires_pure_solvent_assets_not_solute_labels():
    protocol = _protocol()
    anchors = protocol["pure_solvent_anchor_contract"]

    assert (
        "matching molecular-HNC vacuum-limit pressure" in anchors["cubic_coefficient"]
    )
    assert "surface tension" in anchors["quartic_coefficient"]
    assert (
        "cannot be optimized against solvation values" in anchors["quartic_coefficient"]
    )
    assert "content-addressed" in anchors["weighted_density_kernel"]
    assert anchors["physical_bridge_assets"] == []
    requirements = " ".join(anchors["required_certificate_contents"])
    assert "all-atom solvent model" in requirements
    assert "no MNSol, FreeSolv" in requirements

    custom = protocol["custom_solvent_contract"]
    assert "insufficient" in custom["dielectric_only"]
    total_requirements = " ".join(custom["required_before_total_free_energy"])
    assert "all-atom" in total_requirements
    assert "surface tension" in total_requirements
    assert "MACE-native" in total_requirements

    sequence = " ".join(protocol["validation_sequence"])
    assert "exact discrete adjoint pairing" in sequence
    assert "same-functional vacuum-limit pressure" in sequence
    assert "Hessian reciprocity" in sequence
    assert "external-blind maximum-error gates" in sequence
    assert "post-hoc PC+" in protocol["decision_rule"]
