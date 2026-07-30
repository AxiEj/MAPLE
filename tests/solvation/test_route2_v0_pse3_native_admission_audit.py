from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from maple.function.calculator.extra_correction.implicit.route2_v0_mace_cluster_rism_bridge import (
    V0_MOLECULAR_HNC_REQUIRED_RISM_CLOSURE,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
AUDIT = BENCHMARKS / "route2-v0-pse3-native-admission-audit-v1.json"
CLOSURE_IDENTITY = BENCHMARKS / "route2-v0-molecular-hnc-closure-identity-v1.json"
THEORY = ROOT / "docs/implicit-solvation/ROUTE2_V0_PSE_N_ADMISSION_AUDIT_20260730.md"


JsonObject = dict[str, object]


def _object(value: object, *, name: str) -> JsonObject:
    assert isinstance(value, dict), f"{name} must be an object"
    raw = cast(dict[object, object], value)
    assert all(isinstance(key, str) for key in raw), f"{name} has a non-text key"
    return cast(JsonObject, raw)


def _text(value: object, *, name: str) -> str:
    assert isinstance(value, str), f"{name} must be text"
    return value


def _number(value: object, *, name: str) -> float:
    assert isinstance(value, (int, float)), f"{name} must be numeric"
    return float(value)


def _strings(value: object, *, name: str) -> list[str]:
    assert isinstance(value, list), f"{name} must be a list"
    raw = cast(list[object], value)
    assert all(isinstance(item, str) for item in raw), f"{name} must contain text"
    return cast(list[str], raw)


def _json(path: Path) -> JsonObject:
    value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    return _object(value, name=str(path))


def test_native_pse3_audit_locks_the_site_state_and_no_hybrid_boundary():
    audit = _json(AUDIT)

    assert _text(audit["protocol_id"], name="protocol_id") == (
        "route2-v0-pse3-native-admission-audit-v1"
    )
    assert _text(audit["status"], name="status") == (
        "reject-current-mace-molecular-hnc-pse3-shortcut"
    )
    assert "not a 3D-RISM execution" in _text(
        audit["claim_boundary"], name="claim_boundary"
    )

    native = _object(audit["native_pse_n_identity"], name="native_pse_n_identity")
    assert "u_alpha(r), h_alpha(r), c_alpha(r)" in _text(
        native["closure_variables"], name="closure_variables"
    )
    assert "sum_{j=0}^n" in _text(native["closure"], name="closure")
    assert "converged native PSE-n solution" in _text(
        native["matched_on_shell_chemical_potential"],
        name="matched_on_shell_chemical_potential",
    )
    assert "bulk Cvv alone" in _text(native["meaning"], name="meaning")

    current = _object(
        audit["current_route2_state_space"], name="current_route2_state_space"
    )
    assert "whole-molecule" in _text(current["mace_source"], name="mace_source")
    assert "not a solute PSE3 solution" in _text(current["mismatch"], name="mismatch")
    assert _strings(current["prohibited_shortcuts"], name="prohibited_shortcuts") == [
        "insert a PSE3 Cvv into the molecular-HNC quadratic scalar",
        "add the PSE3 on-shell correction after molecular-HNC stationarity",
        "decompose a whole MACE cluster energy into site potentials",
        "select a PSE order, PC/PC+, UC, NgB, or another correction from target-solvation errors",
    ]

    assert _object(audit["hard_constraints"], name="hard_constraints") == {
        "post_training": False,
        "fine_tuning": False,
        "experimental_solvation_fit": False,
        "map_or_uq_calibration": False,
        "posthoc_pse3_correction": False,
        "pse3_into_hnc_hybrid": False,
        "closure_selected_from_target_error": False,
    }


def test_pse3_audit_preserves_the_historical_maximum_error_gate_and_closure_gate():
    audit = _json(AUDIT)
    accuracy = _object(
        audit["external_accuracy_context"], name="external_accuracy_context"
    )
    closure = _object(_json(CLOSURE_IDENTITY)["pse_n_boundary"], name="pse_n_boundary")

    assert (
        _number(
            accuracy["reported_direct_pse3_mue_kcal_per_mol"],
            name="reported_direct_pse3_mue_kcal_per_mol",
        )
        == 19.693
    )
    assert (
        _number(
            accuracy["reported_direct_pse3_rmse_kcal_per_mol"],
            name="reported_direct_pse3_rmse_kcal_per_mol",
        )
        == 20.473
    )
    assert "not the Route-2 historical FreeSolv10 panel" in _text(
        accuracy["direct_cspce_3drism_pse3_study_scope"],
        name="direct_cspce_3drism_pse3_study_scope",
    )
    assert "all-record <1.5 kcal/mol gate" in _text(
        accuracy["interpretation"], name="interpretation"
    )
    assert "route2-v0-pse3-native-admission-audit-v1.json" == _text(
        closure["native_admission_audit"], name="native_admission_audit"
    )
    assert "whole-molecule MACE cluster potential" in _text(
        closure["reason_rejected_by_current_bridge"],
        name="reason_rejected_by_current_bridge",
    )
    assert "invented site potentials" in _text(
        closure["future_rule"], name="future_rule"
    )
    assert V0_MOLECULAR_HNC_REQUIRED_RISM_CLOSURE == "HNC"

    theory = THEORY.read_text(encoding="utf-8")
    assert "## Why it cannot consume the present MACE source" in theory
    assert "all records < 1.5" in theory
    assert "DOI:10.1063/1.3041709" in theory
    assert "DOI:10.1088/0953-8984/28/34/344002" in theory
