from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "docs/route2/evidence/vqm24-mdp-p13-passive-response-pilot-20260827"
O2 = ROOT / "docs/route2/evidence/vqm24-mdp-p13-a3-o2-lbfgs-20260827"
MODES = ROOT / "docs/route2/evidence/vqm24-sector-balanced-p13-response-modes-20260827"
PRO = ROOT / "docs/route2/evidence/mdp-polar-p13-postpilot-pro-20260827"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p13_response_pilot_preserves_failed_predictive_gates() -> None:
    payload = json.loads((PILOT / "result.json").read_text())
    assert payload["status"] == "fail-development-response-pilot"
    assert payload["selected_candidate_for_grouped_cv"] is None
    assert payload["claim_boundary"]["validation_or_blind_formula_opened"] is False
    a2 = payload["candidates"]["A2"]
    a3 = payload["candidates"]["A3"]
    assert a2["gate_passed"] is False
    assert a3["gate_passed"] is False
    assert a3["aggregate"]["mean_audit_mep_relative"] == pytest.approx(
        0.11432149320211824, abs=0.0
    )
    assert a3["relative_score_improvement_over_previous"] == pytest.approx(
        0.12846684869447064, abs=0.0
    )
    assert a3["aggregate"]["maximum_induced_charge_residual_e"] < 3.0e-17
    assert a3["aggregate"]["maximum_source_point_reciprocity_relative"] < 3.0e-16
    assert a3["aggregate"][
        "maximum_source_point_symmetric_eigenvalue_hartree_per_e2"
    ] < 0.0


def test_p13_o2_closure_stops_without_plateau_or_gate_reopening() -> None:
    payload = json.loads((O2 / "result.json").read_text())
    assert payload["status"] == "fail-a3-o2-no-plateau"
    assert payload["backend_preflight"]["passed"] is True
    assert payload["backend_preflight"]["maximum_radial_source_absolute"] < 2.0e-16
    assert payload["backend_preflight"]["parameter_gradient_relative"] < 3.0e-14
    optimization = payload["optimization"]
    assert optimization["accepted_outer_steps"] == 500
    assert optimization["objective_evaluations"] == 1567
    assert optimization["plateau_reached"] is False
    assert optimization["termination"] == "hard-step-cap"
    assert optimization["best_objective"]["total"] < optimization[
        "initial_objective"
    ]["total"]
    assert min(optimization["plateau_block_relative_improvements"]) > 0.01
    assert payload["metrics"] is None
    assert payload["aggregate"] is None
    assert payload["gates"] is None
    assert payload["decision"] == {
        "A3_selected_for_grouped_cv": False,
        "A4_authorized": False,
        "cross_factor_C_authorized": False,
        "maple_capability_admitted": False,
        "validation_or_blind_opened": False,
    }


def test_sector_balanced_modes_are_target_free_full_rank_and_content_bound() -> None:
    payload = json.loads((MODES / "manifest.json").read_text())
    assert payload["status"] == "pass-target-independent-mode-generation"
    assert payload["aggregate"] == {
        "maximum_condition_number": pytest.approx(4.365610169321945, abs=0.0),
        "maximum_radial_fraction": pytest.approx(0.5042872499274398, abs=0.0),
        "minimum_greedy_residual_norm": pytest.approx(0.8787705168111166, abs=0.0),
        "minimum_radial_fraction": pytest.approx(0.48635839299783895, abs=0.0),
        "pass_count": 32,
        "record_count": 32,
    }
    assert payload["claim_boundary"]["qm_response_or_energy_read"] is False
    assert payload["claim_boundary"]["model_prediction_read"] is False
    for record in payload["records"]:
        assert record["status"] == "pass"
        assert record["metrics"]["balanced_rank"] == 12
        assert record["metrics"]["target_used"] is False
        path = Path(record["modes_path"])
        evidence_path = MODES / record["record_id"] / path.name
        assert _sha256(evidence_path) == record["modes_sha256"]


def test_postpilot_pro_manifest_binds_genuine_pro_and_local_decision() -> None:
    manifest = json.loads((PRO / "manifest.json").read_text())
    assert manifest["verified_model"] == "Pro, 5 of 5."
    assert manifest["pro_generation_observed"] is True
    assert manifest["prompt_exact"] is True
    assert manifest["answer_chars"] == 26290
    for name, digest in manifest["files_sha256"].items():
        assert _sha256(PRO / name) == digest

