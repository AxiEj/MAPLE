from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

AUDIT_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/implicit-solvation/benchmarks"
    / "route1-gbmv2-feasibility-audit-2026-07-25.json"
)
DOCUMENTATION_DIR = AUDIT_PATH.parent.parent


def _load_audit() -> dict:
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def test_gbmv2_audit_has_a_self_consistent_fingerprint():
    audit = _load_audit()
    recorded = audit.pop("content_sha256")
    payload = json.dumps(
        audit,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()

    assert hashlib.sha256(payload).hexdigest() == recorded
    assert audit["schema_version"] == 1
    assert audit["protocol"]["protocol_id"] == "route1-gbmv2-feasibility-v1"
    assert all(
        len(fingerprint) == 64 for fingerprint in audit["source_fingerprints"].values()
    )


def test_gbmv2_is_a_physical_am1bcc_compatible_candidate():
    audit = _load_audit()
    candidate = audit["gbmv2"]
    evidence = candidate["small_molecule_evidence"]
    original = evidence["knight_brooks_2011"]
    held_out = evidence["vanommeslaeghe_et_al_2013"]

    assert candidate["route1_formula_compatible"]
    assert candidate["learned_solvent_correction"] is False
    assert candidate["gas_phase_mm_energy_in_product_potential"] is False
    assert candidate["published_force_support"]["analytic_first_derivatives"]
    assert original["charge_model"] == "AM1-BCC"
    assert original["molecule_count"] == 499
    assert original["sampling"]["trajectory_length_ns"] == pytest.approx(10.5)
    assert original["sampling"]["analysis"] == "BAR"
    assert original["gbmv2_metrics"]["aue_kcal_per_mol"] == pytest.approx(1.14)
    assert original["gbmv2_metrics"]["rmse_kcal_per_mol"] == pytest.approx(1.60)
    assert held_out["non_cgenff_375"]["aue_kcal_per_mol"] == pytest.approx(1.24)
    assert held_out["non_cgenff_375"]["r_squared"] == pytest.approx(0.758)


def test_scientific_promise_does_not_bypass_runtime_and_license_gates():
    audit = _load_audit()
    availability = audit["runtime_availability"]
    decision = audit["overall_decision"]

    assert availability["academic_charmm"]["registration_required"]
    assert availability["academic_charmm"]["permitted_use"] == (
        "academic, government, and nonprofit"
    )
    assert not availability["public_openmm"]["gbmv2_string_present"]
    assert not availability["public_openmm"]["gbmv2_force_class_present"]
    assert availability["local_environment"]["charmm_executable_found"] is False
    assert availability["local_environment"]["openmm_gbmv2_symbol_found"] is False
    assert decision["status"] == "scientifically_promising_runtime_blocked"
    assert not decision["new_runtime_provider_added"]
    assert not decision["free_solv_product_screen_opened"]


def test_route1_docs_preserve_the_gbmv2_admission_boundary():
    specification = (DOCUMENTATION_DIR / "ROUTE1_PRODUCT_SPEC.md").read_text(
        encoding="utf-8"
    )
    formulas = (DOCUMENTATION_DIR / "FORMULAS_AND_REFERENCES.md").read_text(
        encoding="utf-8"
    )
    validation = (DOCUMENTATION_DIR / "VALIDATION_STATUS.md").read_text(
        encoding="utf-8"
    )
    benchmark = (AUDIT_PATH.parent / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(
        (specification + formulas + validation + benchmark).split()
    )

    assert "GBMV2 physical-provider feasibility boundary" in specification
    assert "10.1002/jcc.21876" in formulas
    assert "10.1002/jcc.26133" in formulas
    assert "scientifically promising" in normalized
    assert "public OpenMM" in normalized
    assert "No GBMV2 runtime provider" in benchmark
