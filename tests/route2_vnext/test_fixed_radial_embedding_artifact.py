from __future__ import annotations

import hashlib
import json
from pathlib import Path

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release.evidence import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/route2/evidence/mace-fixed-radial-source-terminal-7c5fc16b"
RUN1 = EVIDENCE / "run1.json"
RUN2 = EVIDENCE / "run2.json"
EXECUTION_HEAD = "7c5fc16b5bcb98221a97d5663329941ee74fe4f3"
MEASUREMENT_SHA256 = "77cd7ab8a845820a78d0236395411ace5c68fd3568c27e999e0a15705291f7bc"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
JSON_SHA256 = {
    "run1.json": "16416b006cdfb33ae1d83e9759e095b900a49bdf1ac36dae1854f2925982032b",
    "run2.json": "ce21ce4fa0ae57444573c77d576ac9fed63987385e3315e279a2a9c79bd689ad",
}


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measurement(artifact: dict[str, object]) -> dict[str, object]:
    return {
        key: artifact[key]
        for key in (
            "protocol",
            "candidate_grid",
            "selection",
            "selected_embedding",
            "static_mep_records",
            "static_mep_gate",
            "matched_pcmsolver_records",
            "matched_pcmsolver_gate",
            "decision",
        )
    }


def test_fixed_radial_source_fails_terminal_matched_pcm_gate() -> None:
    artifact = _load(RUN1)
    assert artifact["schema_version"] == (
        "route2-mace-fixed-radial-embedding-terminal-v1"
    )
    assert artifact["status"] == "fixed-radial-embedding-fails-pcmsolver-energy-gate"
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["working_tree_clean"] is True
    assert artifact["capabilities"] == NO_CAPABILITIES
    assert artifact["selected_embedding"] == {
        "candidate_id": "pure-sigma-0p75",
        "configuration_sha256": (
            "2477a79a7331d3b01f56697aeeac6f7e2c559bedb13e7aeb26175bee634394f3"
        ),
        "sigmas_angstrom": [0.75],
        "weights": [1.0],
    }
    assert artifact["selection"]["relative_training_improvement"] == (
        0.8934497834489507
    )
    assert artifact["static_mep_gate"]["gate_passed"] is True
    assert artifact["static_mep_gate"]["selected_heldout_objective"] == (
        0.02034296928817235
    )
    pcm_gate = artifact["matched_pcmsolver_gate"]
    assert pcm_gate == {
        "case_pass_count": 0,
        "executed": True,
        "gate_passed": False,
        "maximum_energy_absolute_error_kcal_per_mol": 4.887705463072207,
    }
    errors = {
        record["compound_id"]: record["polarization_energy_absolute_error_kcal_per_mol"]
        for record in artifact["matched_pcmsolver_records"]
    }
    assert errors == {
        "mobley_3034976": 3.59096053261316,
        "mobley_3053621": 1.8285303218119144,
        "mobley_352111": 4.887705463072207,
        "mobley_3867265": 2.107752938986818,
    }
    assert artifact["decision"] == {
        "additional_radial_patches_authorized": False,
        "fixed_radial_source_profile_passed": False,
        "phi0_ledger_decomposition_authorized": False,
        "phi1_delta_ledger_decomposition_authorized": False,
        "public_capability_admitted": False,
        "selected_profile_may_be_frozen": False,
        "transition_to_scalar_first_or_independent_variational_polarization": True,
    }
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
    assert canonical_json_sha256(_measurement(artifact)) == MEASUREMENT_SHA256
    assert_source_files_match_execution_commit(ROOT, artifact)


def test_fixed_radial_terminal_measurement_replays_exactly() -> None:
    first = _load(RUN1)
    second = _load(RUN2)
    assert _measurement(first) == _measurement(second)
    for artifact in (first, second):
        assert artifact["execution_git_head"] == EXECUTION_HEAD
        assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
        assert artifact["capabilities"] == NO_CAPABILITIES
        assert_source_files_match_execution_commit(ROOT, artifact)
    for name, expected in JSON_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
