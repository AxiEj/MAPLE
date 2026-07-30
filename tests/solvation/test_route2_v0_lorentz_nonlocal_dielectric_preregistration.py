from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    / "route2-v0-lorentz-nonlocal-dielectric-prereg-v1.json"
)


def _protocol() -> dict[str, Any]:
    payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Lorentz nonlocal-dielectric preregistration must be an object.")
    return cast(dict[str, Any], payload)


def test_lorentz_preregistration_locks_one_passive_common_response_scalar():
    protocol = _protocol()

    assert protocol["protocol_id"] == "route2-v0-lorentz-nonlocal-dielectric-prereg-v1"
    assert protocol["status"] == (
        "preregistered-custom-solvent-electrostatic-control-before-liquid-admission"
    )
    assert protocol["literature"][0]["doi"] == "10.1103/PhysRevE.94.012114"
    construction = protocol["construction"]
    assert "epsilon_infinity" in construction["spectrum"]
    assert "lambda_s" in construction["spectrum"]
    assert "epsilon_s(0)=epsilon_0" in construction["limits"]
    assert "minimize F_or" in construction["orientational_scalar"]
    assert "delta G_pol/d rho = V_reac" in construction["derivative"]
    assert "cannot be inferred" in construction["correlation_length_rule"]


def test_lorentz_preregistration_preserves_the_no_fit_and_no_total_endpoint_boundary():
    protocol = _protocol()

    assert protocol["hard_constraints"] == {
        "official_checkpoint_unmodified": True,
        "post_training": False,
        "fine_tuning": False,
        "experimental_solvation_fit": False,
        "map_or_uq_calibration": False,
        "error_driven_correlation_length_or_radius_selection": False,
        "dielectric_only_total_free_energy": False,
        "legacy_public_route_changed": False,
    }
    source = protocol["source_contract"]
    assert source["current_physical_state_records"] == []
    required = " ".join(source["required_before_physical_custom_electrostatic_use"])
    assert "static dielectric" in required
    assert "optical dielectric" in required
    assert "finite-wavevector dielectric" in required
    forbidden = " ".join(source["forbidden_inputs"])
    assert "FreeSolv" in forbidden
    assert "MNSol" in forbidden
    assert "post-training" in forbidden


def test_lorentz_preregistration_cannot_bypass_the_historical_or_multisolvent_gates():
    protocol = _protocol()
    validation = " ".join(protocol["validation_sequence"])

    assert "same reaction scalar" in validation
    assert "fixed-charge molecular-RISM inputs" in validation
    assert "route2-v0-historical-freesolv10-regression-v1" in validation
    assert "ethyl acetate" in validation
    assert "11-or-more solvent strata" in validation
    assert "strictly below 1.5 kcal/mol" in validation
    assert "fine-tuning" in protocol["decision_rule"]
