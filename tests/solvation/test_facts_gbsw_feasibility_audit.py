from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

AUDIT_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/implicit-solvation/benchmarks"
    / "route1-facts-gbsw-feasibility-audit-2026-07-25.json"
)
DOCUMENTATION_DIR = AUDIT_PATH.parent.parent


def _load_audit() -> dict:
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def test_facts_gbsw_audit_has_a_self_consistent_fingerprint():
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
    assert audit["protocol"]["protocol_id"] == "route1-facts-gbsw-feasibility-v1"
    assert all(
        len(fingerprint) == 64 for fingerprint in audit["source_fingerprints"].values()
    )


def test_gbsw_is_physical_but_does_not_claim_faster_than_bare_mm():
    audit = _load_audit()
    gbsw = audit["models"]["gbsw"]
    original = gbsw["small_molecule_evidence"]["knight_brooks_2011"]
    held_out = gbsw["small_molecule_evidence"]["vanommeslaeghe_et_al_2013"]

    assert gbsw["route1_formula_compatible"]
    assert gbsw["learned_solvent_correction"] is False
    assert gbsw["gas_phase_mm_energy_in_product_potential"] is False
    assert gbsw["published_force_support"]["complete_solvation_energy_and_forces"]
    assert original["charge_model"] == "AM1-BCC"
    assert original["optimized_profile_metrics"]["aue_kcal_per_mol"] == pytest.approx(
        1.20
    )
    assert original["optimized_profile_metrics"]["rmse_kcal_per_mol"] == pytest.approx(
        1.52
    )
    assert held_out["non_cgenff_375"]["aue_kcal_per_mol"] == pytest.approx(1.33)
    assert held_out["non_cgenff_375"]["r_squared"] == pytest.approx(0.701)
    assert gbsw["published_speed_scope"]["relative_to_vacuum"] == (
        "about 4x slower"
    )
    assert gbsw["published_speed_scope"]["relative_to_gbmv"] == (
        "about 2-3x faster"
    )
    assert not gbsw["published_speed_scope"]["faster_than_bare_mm_claim"]


def test_facts_accuracy_and_small_molecule_typing_limits_are_explicit():
    audit = _load_audit()
    facts = audit["models"]["facts"]
    original = facts["small_molecule_evidence"]["knight_brooks_2011"]
    held_out = facts["small_molecule_evidence"]["vanommeslaeghe_et_al_2013"]
    applicability = facts["applicability"]

    assert facts["route1_formula_compatible"]
    assert facts["learned_solvent_correction"] is False
    assert original["optimized_profile_metrics"]["aue_kcal_per_mol"] == pytest.approx(
        1.25
    )
    assert original["optimized_profile_metrics"]["rmse_kcal_per_mol"] == pytest.approx(
        1.80
    )
    assert held_out["non_cgenff_375"]["aue_kcal_per_mol"] == pytest.approx(1.42)
    assert held_out["non_cgenff_375"]["r_squared"] == pytest.approx(0.628)
    assert applicability["official_parameters_derived_for_protein_atoms_only"]
    assert applicability["unknown_radii_use_tavw_interpolation"]
    assert not facts["published_speed_scope"]["faster_than_bare_mm_claim"]


def test_runtime_barrier_prevents_provider_admission():
    audit = _load_audit()
    runtime = audit["runtime_availability"]
    decision = audit["overall_decision"]

    assert runtime["academic_charmm"]["registration_required"]
    assert runtime["academic_charmm"]["gbsw_openmm_plugin_in_distribution"]
    assert not runtime["public_openmm"]["gbsw_string_present"]
    assert not runtime["public_openmm"]["facts_string_present"]
    assert runtime["local_environment"]["charmm_executable_found"] is False
    assert runtime["local_environment"]["openmm_gbsw_symbol_found"] is False
    assert decision["priority_order"] == ["GBMV2", "GBSW", "FACTS"]
    assert decision["status"] == "no_new_provider"
    assert not decision["new_runtime_provider_added"]
    assert not decision["paper_based_reimplementation_planned"]


def test_route1_docs_preserve_facts_gbsw_admission_boundary():
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

    assert "FACTS/GBSW physical-provider feasibility boundary" in specification
    assert "10.1002/jcc.10321" in formulas
    assert "10.1002/jcc.20832" in formulas
    assert "about four times slower than vacuum" in normalized
    assert "No FACTS or GBSW runtime provider" in benchmark
