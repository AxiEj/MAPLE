from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
PREREG = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-v0-molecular-cubic-wda-prereg-v1.json"
)


def _protocol() -> dict[str, Any]:
    payload = json.loads(PREREG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Cubic-WDA preregistration must be a JSON object.")
    return cast(dict[str, Any], payload)


def test_molecular_cubic_wda_preregistration_locks_the_common_scalar_form():
    protocol = _protocol()

    assert protocol["protocol_id"] == "route2-v0-molecular-cubic-wda-prereg-v1"
    assert protocol["status"] == (
        "preregistered-control-before-physical-cubic-wda-admission"
    )
    assert protocol["literature"]["doi"] == "10.1063/5.0057506"
    construction = protocol["construction"]
    assert construction["name"] == "molecular-cubic-wda-2021"
    assert "Omega_HNC" in construction["base_scalar"]
    assert "D^T q" in construction["centre_projection"]
    assert "rho_bar-rho_bulk" in construction["bridge_scalar"]
    assert "Gaussian" in construction["gaussian_kernel"]
    assert "S_NN(0)" in construction["compressibility_constraint"]
    assert "P_HNC-P_s" in construction["pressure_cross_check"]
    assert (
        "stationary pure-liquid planar-interface"
        in construction["surface_tension_constraint"]
    )
    assert "exact D^T K_sigma" in construction["response_rule"]


def test_molecular_cubic_wda_preregistration_excludes_response_and_label_repairs():
    protocol = _protocol()

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
        "electrostatic_excess_chemical_potential_bridge_not_silently_enabled": True,
    }
    admission = protocol["physical_admission_contract"]
    assert admission["current_physical_assets"] == []
    assert (
        "cannot establish physical-liquid admission"
        in admission["direct_constructor_status"]
    )
    required = " ".join(admission["required_before_target_solute_execution"])
    assert "all-atom solvent model" in required
    assert "orientational liquid correlation" in required
    assert "isothermal compressibility" in required
    assert "stationary planar-interface" in required
    assert "MACE-native" in required
    forbidden = " ".join(admission["forbidden_inputs"])
    assert "MNSol" in forbidden
    assert "FreeSolv" in forbidden
    assert "post-training" in forbidden

    electrostatic = protocol["electrostatic_bridge_boundary"]
    assert electrostatic["status"] == "deferred"
    assert electrostatic["response_or_ledger_patch_forbidden"] is True
    assert "multi-solvent falsification" in electrostatic["reason"]


def test_molecular_cubic_wda_preregistration_keeps_the_historical_outlier_gate():
    protocol = _protocol()
    sequence = " ".join(protocol["validation_sequence"])

    assert "Hessian-vector action" in sequence
    assert "source-bound physical pure-liquid certificate" in sequence
    assert "route2-v0-historical-freesolv10-regression-v1" in sequence
    assert "ethyl acetate" in sequence
    assert "strictly below 1.5 kcal/mol" in sequence
    assert "11-solvent development" in sequence
    assert "external blind" in sequence
    assert "post-training" in protocol["decision_rule"]
