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
    / "mace-original-source-pcmsolver-four-1d40c93b"
)
RUN1 = EVIDENCE / "run1.json"
RUN2 = EVIDENCE / "run2.json"

EXECUTION_HEAD = "1d40c93bad67d6f4b4153f9bb8c7a981c391e3bf"
MEASUREMENT_SHA256 = "ff40c7c426cbcb59ab28378f29a1ed5675da98d4063fba20423b0247d997649b"
CHECKPOINT_SHA256 = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
PCMSOLVER_SHA256 = "296b6f34a03789943ae8823b3790f36357c50c896497f21374ac16fe5a6c43a3"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
JSON_SHA256 = {
    "run1.json": "fb6f5f175348b51b99f55147f5df75782f8b2ccbbb86ba16aaae7c2fdb0f0748",
    "run2.json": "26a9aa7da8899331f554b7b238bd7729d926e0a992504210005f061df457288b",
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


def test_original_source_fails_matched_qm_pcmsolver_physical_gate():
    artifact = _load(RUN1)
    assert artifact["schema_version"] == (
        "route2-mace-original-source-pcmsolver-four-v1"
    )
    assert artifact["status"] == "original-source-fails-frozen-four-case-gate"
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["working_tree_clean"] is True
    assert artifact["capabilities"] == NO_CAPABILITIES
    assert artifact["external_assets"]["mace_checkpoint"]["sha256"] == (
        CHECKPOINT_SHA256
    )
    assert artifact["external_assets"]["pcmsolver_library"]["sha256"] == (
        PCMSOLVER_SHA256
    )
    assert artifact["decision"] == {
        "original_four_channel_quantitative_pcm_source_admitted": False,
        "public_capability_admitted": False,
        "radial_embedding_precondition_status": (
            "far-field test-charge and quadrupole panel not yet complete"
        ),
        "radial_embedding_research_authorized": False,
        "separated_phi0_ledger_evaluation_authorized": False,
        "separated_phi1_delta_ledger_evaluation_authorized": False,
    }

    aggregate = artifact["aggregate"]
    assert aggregate["record_count"] == 4
    assert aggregate["case_pass_count"] == 0
    assert aggregate["mean_polarization_energy_absolute_error_kcal_per_mol"] == (
        7.4612812780353766
    )
    assert aggregate["maximum_polarization_energy_absolute_error_kcal_per_mol"] == (
        11.82831871091405
    )
    assert aggregate["maximum_surface_mep_area_weighted_relative_l2_error"] == (
        1.0341564506467698
    )

    records = artifact["records"]
    assert [record["compound_id"] for record in records] == [
        "mobley_3034976",
        "mobley_3053621",
        "mobley_352111",
        "mobley_3867265",
    ]
    assert all(record["charge_gate_passed"] is True for record in records)
    assert all(record["fixed_source_energy_gate_passed"] is False for record in records)
    assert all(record["case_passed"] is False for record in records)
    assert (
        min(record["surface_mep_area_weighted_relative_l2_error"] for record in records)
        > 0.62
    )
    assert (
        min(
            record["polarization_energy_absolute_error_kcal_per_mol"]
            for record in records
        )
        > 2.8
    )
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
    assert canonical_json_sha256(_measurement(artifact)) == MEASUREMENT_SHA256
    assert_source_files_match_execution_commit(ROOT, artifact)


def test_original_source_panel_replays_exact_scientific_measurement():
    first = _load(RUN1)
    second = _load(RUN2)
    for artifact in (first, second):
        assert artifact["execution_git_head"] == EXECUTION_HEAD
        assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
        assert artifact["capabilities"] == NO_CAPABILITIES
        assert_source_files_match_execution_commit(ROOT, artifact)
    assert _measurement(second) == _measurement(first)
    for name, expected in JSON_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
