from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ADMISSION = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-v0-structured-solvent-admission-v1.json"
)


def _protocol() -> dict:
    return json.loads(ADMISSION.read_text(encoding="utf-8"))


def test_structured_solvent_admission_locks_the_no_training_boundary():
    protocol = _protocol()

    assert protocol["protocol_id"] == "route2-v0-structured-solvent-admission-v1"
    assert protocol["status"] == "preregistered-before-implementation"
    assert protocol["construction"]["name"] == (
        "route2-v0-fixed-density-structured-solvent-v1"
    )

    assert protocol["hard_constraints"] == {
        "official_checkpoint_unmodified": True,
        "post_training": False,
        "fine_tuning": False,
        "experimental_solvation_fit": False,
        "map_or_uq_calibration": False,
        "density_projection": False,
        "response_tempering_or_eigenvalue_clipping": False,
        "error_driven_cavity_or_dispersion_adjustment": False,
        "per_record_closure_or_model_selection": False,
        "smd_cds_in_no_fit_core": False,
        "amber_gaff_or_am1bcc_solute_substitution": False,
        "legacy_public_route_changed": False,
    }


def test_structured_solvent_admission_preserves_the_mace_source_boundary():
    protocol = _protocol()
    construction = protocol["construction"]
    equivalence = protocol["source_equivalence_contract"]

    assert "unmodified zero-reaction-field MACE-POLAR" in construction[
        "electronic_state"
    ]
    assert "gaussian_multipole_potential" in construction["electrostatic_source"]
    assert "nonnegative promolecular" in construction["short_range_source"]
    assert "does not replace phi_c0^G" in construction["short_range_source"]
    assert "MACE-native all-space grid potential" in equivalence[
        "required_grid_interface"
    ]
    assert "forbidden" in equivalence["ambertools_stock_endpoint"]
    assert "GAFF, AM1-BCC" in equivalence["ambertools_stock_endpoint"]
    assert "separate research control only" in equivalence["route1_all_site_rism"]
    assert "forbidden" in equivalence["smd_cds_combination"]


def test_structured_solvent_admission_requires_a_frozen_liquid_asset_and_review():
    protocol = _protocol()
    liquid = protocol["structured_solvent_contract"]
    solvent = protocol["solvent_asset_contract"]
    benchmark = protocol["benchmark_and_review_contract"]

    assert liquid["primary_closure"].startswith("Kovalenko-Hirata")
    assert "thermodynamic pressure correction" in liquid["pressure_correction"]
    assert "diagnostic-only" in liquid["pc_plus_policy"]
    assert solvent["minimum_registered_solvent_count"] == 11
    assert len(solvent["registered_default_solvents"]) == 11
    assert "methanol" in solvent["registered_default_solvents"]
    assert any("bulk susceptibility" in item for item in solvent["required_before_target_scoring"])
    assert "insufficient" in solvent["custom_solvent_rule"]
    assert benchmark["target_solvation_labels_read_before_implementation"] is False
    assert "external final blind dataset" in benchmark["external_blind_manifest"]
    assert "strictly below 1.5 kcal/mol" in benchmark["accuracy_gate"]
    assert "independently review source equivalence" in benchmark["independent_review"]
