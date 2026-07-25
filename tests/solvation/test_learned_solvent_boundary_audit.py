from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

AUDIT_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/implicit-solvation/benchmarks"
    / "route1-learned-solvent-boundary-audit-2026-07-25.json"
)
DOCUMENTATION_DIR = AUDIT_PATH.parent.parent


def _load_audit() -> dict:
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def test_learned_solvent_boundary_audit_has_a_self_consistent_fingerprint():
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
    assert audit["protocol"]["protocol_id"] == (
        "route1-learned-solvent-boundary-v1"
    )
    assert all(
        len(fingerprint) == 64 for fingerprint in audit["source_fingerprints"].values()
    )


def test_runtime_feasibility_does_not_admit_gnnis_to_route1():
    audit = _load_audit()
    route = audit["route_contract"]
    gnnis = audit["gnnis"]
    smoke = gnnis["local_runtime_feasibility_smoke"]

    assert route["default_charge"] == "AM1-BCC"
    assert not route["gas_mlip_retraining"]
    assert not route["chemistry_specific_learned_solvent_correction"]
    assert gnnis["source_findings"]["released_runtime_returns_solvent_energy_only"]
    assert gnnis["source_findings"][
        "force_is_negative_autograd_gradient_of_reported_scalar_energy"
    ]
    assert smoke["energy_kj_per_mol"] == pytest.approx(
        smoke["polar_kj_per_mol"] + smoke["nonpolar_kj_per_mol"]
    )
    assert smoke["single_coordinate_fd"]["relative_error"] < 0.003
    assert gnnis["license"]["repository_classification"] == "MIT-0"
    assert gnnis["license"]["setup_classifier"] == "MIT License"
    assert gnnis["route1_baseline_decision"] == "excluded"
    assert not audit["overall_decision"]["new_runtime_provider_added"]


def test_qm_gnnis_delta_transfer_remains_outside_route1():
    audit = _load_audit()
    qm_gnnis = audit["qm_gnnis"]
    findings = qm_gnnis["source_findings"]

    assert findings["released_delta_class"] == (
        "GNN3_Multisolvent_embedding_run_multiple_Delta"
    )
    assert findings["additional_qm_training_required"] is False
    assert findings["experimental_data_used_to_train_correction"] is False
    assert "G_GNNIS-G_GB-Neck2" in findings["published_combination"]
    assert findings["published_runtime_host"] == (
        "ASE SumCalculator combining the Torch delta with ORCA/CPCM"
    )
    assert qm_gnnis["route1_baseline_decision"] == "excluded"
    assert not audit["overall_decision"]["gnnis_or_qm_gnnis_in_route1_baseline"]


def test_source_claims_are_linked_to_pinned_paths_and_line_ranges():
    audit = _load_audit()
    evidence = audit["source_evidence"]
    fingerprints = audit["source_fingerprints"]

    target = evidence["gnnis_training_target"]
    assert target["commit"] == audit["gnnis"]["commit"]
    assert target["path"] == "Simulation/Simulator.py"
    assert target["line_range"] == [1506, 1519]
    assert target["sha256"] == fingerprints["gnnis_label_source_sha256"]

    loss = evidence["gnnis_force_only_loss"]
    assert loss["path"] == "MachineLearning/GNN_Loss_Functions.py"
    assert loss["line_range"] == [9, 11]
    assert loss["sha256"] == fingerprints["gnnis_loss_source_sha256"]

    delta = evidence["qm_gnnis_delta_class"]
    assert delta["path"] == "MachineLearning/GNN_Models.py"
    assert delta["line_range"] == [947, 1065]
    assert delta["sha256"] == fingerprints["gnnis_model_source_sha256"]

    runtime = evidence["qm_gnnis_runtime_sum"]
    assert runtime["commit"] == audit["qm_gnnis"]["commit"]
    assert runtime["path"] == "implicitml/calculator.py"
    assert runtime["line_range"] == [149, 165]
    assert runtime["sha256"] == fingerprints["qm_gnnis_calculator_source_sha256"]

    paper = evidence["qm_gnnis_paper_correction_and_scope"]
    assert paper["line_ranges"] == [[9, 11], [27, 29], [76, 90], [100, 112]]
    assert paper["sha256"] == fingerprints["qm_gnnis_pmc_text_sha256"]


def test_route1_docs_exclude_learned_solvent_candidates_for_the_right_reason():
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

    assert "Learned-solvent boundary: GNNIS and QM-GNNIS" in specification
    assert "mean explicit-solvent solute force minus the vacuum OpenFF" in normalized
    assert "G_GNNIS-G_GB-Neck2" in normalized
    assert (
        "=E_{\\mathrm{QM,CPCM}}\n"
        "+\\left(G_{\\mathrm{GNNIS}}-G_{\\mathrm{GB\\mbox{-}Neck2}}\\right)"
        in specification
    )
    assert "not a global smoothness, force-conservativity, or MD" in normalized
    assert "sums the Torch delta with ORCA/CPCM" in normalized
    assert "No dependency, runtime provider" in benchmark
    assert "10.1039/D4SC02432J" in formulas
    assert "10.1021/acs.jctc.5c00728" in formulas
