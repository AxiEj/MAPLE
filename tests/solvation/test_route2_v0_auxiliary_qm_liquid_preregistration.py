from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-v0-auxiliary-qm-liquid-prereg-v1.json"
)
THEORY = ROOT / "docs/implicit-solvation/ROUTE2_V0_AQ_THEORY.md"


def _protocol() -> dict:
    return json.loads(PREREGISTRATION.read_text(encoding="utf-8"))


def test_auxiliary_qm_liquid_preregistration_locks_the_no_training_boundary():
    protocol = _protocol()

    assert protocol["protocol_id"] == "route2-v0-auxiliary-qm-liquid-prereg-v1"
    assert protocol["status"] == "preregistered-before-auxiliary-execution"
    assert protocol["construction"]["name"] == (
        "route2-v0-auxiliary-qm-liquid-difference-v1"
    )
    assert protocol["hard_constraints"] == {
        "official_checkpoint_unmodified": True,
        "post_training": False,
        "fine_tuning": False,
        "experimental_solvation_fit": False,
        "map_or_uq_calibration": False,
        "mace_density_relabelling_or_projection": False,
        "mace_field_response_or_feature_feedback": False,
        "response_tempering_or_eigenvalue_clipping": False,
        "extra_pcm_half_coupling_outside_auxiliary_difference": False,
        "smd_cds_in_no_fit_core": False,
        "error_driven_cavity_dispersion_or_solvent_property_adjustment": False,
        "per_record_or_per_solvent_model_selection_from_error": False,
        "amber_gaff_or_am1bcc_solute_substitution": False,
        "legacy_public_route_changed": False,
    }


def test_auxiliary_qm_liquid_ledger_preserves_mace_gas_and_subtracts_aux_gas():
    construction = _protocol()["construction"]

    assert "unmodified official MACE-POLAR checkpoint" in construction["gas_mace_term"]
    assert "A_aux^sol(R;s)-A_aux^gas(R)" in construction["required_energy_ledger"]
    assert "mandatory" in construction["required_energy_ledger"]
    assert (
        "neither MACE density coefficients" in construction["auxiliary_state_boundary"]
    )
    assert (
        "must not be injected into MACE field features"
        in construction["auxiliary_state_boundary"]
    )
    assert "no SMD CDS" in construction["v0_aq_e_control"]
    assert "total-solvation claim" in construction["v0_aq_e_control"]


def test_auxiliary_qm_liquid_contract_requires_one_stationary_scalar():
    stationarity = _protocol()["stationarity_and_force_contract"]

    assert "0.5 sigma^T A_R sigma" in stationarity["pcm_control_lagrangian"]
    assert "A_R sigma+B_R n=0" in stationarity["pcm_control_stationarity"]
    assert "PCM-SCF" in stationarity["pcm_control_energy"]
    assert "gas-SCF" in stationarity["pcm_control_energy"]
    assert (
        "implementation-specific isolated PCM component"
        in stationarity["pcm_control_energy"]
    )
    assert "No legacy MACE field-response derivative" in stationarity["derivative_rule"]
    assert "insufficient for a force/PES claim" in stationarity["force_gate"]


def test_auxiliary_qm_liquid_contract_fails_closed_without_physical_liquid_assets():
    protocol = _protocol()
    liquid = protocol["liquid_asset_contract"]

    assert liquid["minimum_registered_solvent_count"] == 11
    assert any(
        "site model" in item for item in liquid["required_before_target_scoring"]
    )
    assert any(
        "standard-state convention" in item
        for item in liquid["required_before_target_scoring"]
    )
    assert (
        "rejects V0-AQ-L total-free-energy execution" in liquid["dielectric_only_rule"]
    )
    assert (
        "frozen independently sourced liquid assets"
        in liquid["upstream_family_policy"]["joint_or_classical_dft"]
    )
    assert (
        "jointly stationary auxiliary QM--liquid scalar"
        in liquid["upstream_family_policy"]["qm_3drism_scf_or_mdft"]
    )
    assert "fitted dispersion contribution" in liquid["upstream_family_policy"]["salsa"]
    assert "No physical 11-solvent" in liquid["current_inventory"]


def test_auxiliary_qm_liquid_theory_keeps_electronic_control_distinct_from_total_solvent():
    theory = THEORY.read_text(encoding="utf-8")
    normalized_theory = " ".join(theory.split())

    assert "auxiliary quantum--liquid variational-difference contract" in theory
    assert "V0-AQ-E" in theory
    assert "V0-AQ-L" in theory
    assert "not a total solvation free energy" in normalized_theory
    assert "not a MACE density" in normalized_theory
    assert "The gas subtraction is mandatory" in theory
    assert "post-result radius/scale selection" in theory
