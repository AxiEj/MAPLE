from __future__ import annotations

import hashlib
import json
from pathlib import Path

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release.evidence import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "mace-original-source-farfield-four-9c1d4fb9"
)
RUN1 = EVIDENCE / "run1.json"
RUN2 = EVIDENCE / "run2.json"

EXECUTION_HEAD = "9c1d4fb961a7d1b06372d6dcd86c165e582394c9"
MEASUREMENT_SHA256 = "dc9cce0f1ced9c34af1ab67bdae329d4521f2e67b780a1adcad6b7dd645d97d8"
PREREGISTRATION_SHA256 = (
    "a1d6bd40e3801a6e79056b1796f2fadf8527777a78172353a4b152418ab3141b"
)
CHECKPOINT_SHA256 = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
PYSCF_PYTHON_SHA256 = "7d68ade3cc070c3cd72717ea617a0565bd7fb9eb67cd8ed468dad9ac207add8d"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
JSON_SHA256 = {
    "run1.json": "771d800632c9597dc23141fad97add5ccdd2009f5f924bb7fe74f79e57022633",
    "run2.json": "a8e62a01975794401fd3830c0cfe785032ac7f2a803918c9affb88318d7d61ab",
}


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measurement(artifact: dict[str, object]) -> dict[str, object]:
    return {
        key: artifact[key] for key in ("protocol", "records", "aggregate", "decision")
    }


def test_original_source_far_field_gate_authorizes_only_one_research_branch():
    artifact = _load(RUN1)
    assert artifact["schema_version"] == (
        "route2-mace-original-source-farfield-four-v1"
    )
    assert artifact["status"] == ("far-field-pass-radial-embedding-research-authorized")
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["working_tree_clean"] is True
    assert artifact["capabilities"] == NO_CAPABILITIES
    assert artifact["external_assets"]["mace_checkpoint"]["sha256"] == (
        CHECKPOINT_SHA256
    )
    assert artifact["external_assets"]["pyscf_python"]["sha256"] == (
        PYSCF_PYTHON_SHA256
    )
    assert (
        artifact["external_assets"]["far_field_preregistration"]["sha256"]
        == PREREGISTRATION_SHA256
    )
    assert artifact["decision"] == {
        "far_field_necessary_precondition_passed": True,
        "fixed_radial_embedding_research_authorized": True,
        "original_four_channel_quantitative_pcm_source_admitted": False,
        "public_capability_admitted": False,
        "radial_embedding_authorization_scope": (
            "one separately named fixed symmetry-preserving research profile; "
            "held-out cavity-near-field evidence still mandatory"
        ),
        "scalar_first_or_independent_polarization_branch_required": False,
        "separated_phi0_ledger_evaluation_authorized": False,
        "separated_phi1_delta_ledger_evaluation_authorized": False,
    }

    assert artifact["aggregate"] == {
        "case_pass_count": 4,
        "maximum_dipole_relative_l2_error": 0.09466400522274811,
        "maximum_far_field_shell_relative_l2_error": 0.14262072643100593,
        "maximum_quadrupole_relative_frobenius_error": 0.35842617172295177,
        "record_count": 4,
    }
    records = artifact["records"]
    assert [record["compound_id"] for record in records] == [
        "mobley_3034976",
        "mobley_3053621",
        "mobley_352111",
        "mobley_3867265",
    ]
    assert all(record["case_passed"] is True for record in records)
    assert all(record["comparison"]["charge_gate_passed"] is True for record in records)
    assert all(record["comparison"]["dipole_gate_passed"] is True for record in records)
    assert all(
        record["comparison"]["quadrupole_gate_passed"] is True for record in records
    )
    assert all(
        record["comparison"]["far_field_mep_gate_passed"] is True for record in records
    )
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
    assert canonical_json_sha256(_measurement(artifact)) == MEASUREMENT_SHA256
    assert_source_files_match_execution_commit(ROOT, artifact)


def test_original_source_far_field_panel_replays_exact_measurement():
    first = _load(RUN1)
    second = _load(RUN2)
    assert _measurement(second) == _measurement(first)
    for artifact in (first, second):
        assert artifact["execution_git_head"] == EXECUTION_HEAD
        assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
        assert artifact["capabilities"] == NO_CAPABILITIES
        assert_source_files_match_execution_commit(ROOT, artifact)
    for name, expected in JSON_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
