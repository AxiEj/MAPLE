from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "tests/solvation/run_independent_obc2_candidate_cell.py"
SPEC = importlib.util.spec_from_file_location("obc2_candidate_cell", HARNESS)
candidate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(candidate)


def test_candidate_evidence_label_cannot_claim_installed_qualification():
    artifact = candidate.seal(
        {
            "artifact_type": "independent-obc2-candidate-exact-cell-v1",
            "qualification_stage": "candidate qualification, not installed qualification",
            "nonpolar": ["ace", "none"],
        }
    )
    payload = dict(artifact)
    content_sha256 = payload.pop("content_sha256")
    import hashlib
    assert hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest() == content_sha256


def test_candidate_harness_uses_only_the_single_private_monkeypatch_gate():
    source = HARNESS.read_text(encoding="utf-8")
    assert "module._admit_independent_obc2_analytic_cell = exact_test_gate" in source
    assert "module._admit_independent_obc2_analytic_cell = original" in source
    assert "os.environ" not in source
    assert "provider=qualified" not in source


def test_candidate_hessian_uses_frozen_full_force_fd_ladder_and_thresholds():
    source = HARNESS.read_text(encoding="utf-8")
    assert "for step in (1.0e-6, 2.0e-6, 5.0e-6)" in source
    assert "numerical_hessian_from_atoms" in source
    assert "rms <= 5.0e-6 and maximum <= 2.0e-5" in source
    assert '"hessian_force_fd_ladder": fd_ladder' in source


def test_candidate_composition_and_direct_hvp_forbid_double_or_dense_hessian():
    source = HARNESS.read_text(encoding="utf-8")
    assert "hessian - (gas_hessian + solvent_hessian)" in source
    assert '"hessian_composition_max_hartree_per_angstrom2"' in source
    assert "direct HVP materialized a dense Hessian" in source
    assert '"direct_hvp_dense_hessian_forbidden": True' in source


def test_candidate_harness_binds_clean_unchanged_qualification_source_snapshot():
    source = HARNESS.read_text(encoding="utf-8")
    assert "qualification.source_snapshot(require_clean=True)" in source
    assert '"candidate_source_snapshot": source_before' in source
    assert "source_before != source_after" in source
