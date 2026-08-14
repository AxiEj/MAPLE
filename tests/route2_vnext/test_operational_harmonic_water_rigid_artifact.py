from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "operational-analytic-harmonic-rigid-water-bea47120"
)
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"
HEAD = "bea4712067e9b0eb32d3b9476c6d839c64122c32"
CHECKPOINT = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
MEASUREMENT = "d8802924939ad40814eb81986aef4e76a76ba03fde88880597b5724b8f45480c"
CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILES = {
    "measurements.json": (
        "b66e9fe9333c9b75aab1ec43e2111f6f1cf2d4c87a8e39a41ea24f1639c03ed0"
    ),
    "cold-replay.json": (
        "0864b3743323bfcfa891a8c9ae16bb533c7996543c5b0221a6759fe649e5cd41"
    ),
    "README.md": "72753176e1dff207ec73eaa61369e4eb82607250d61cb78e2b0ced11372145b9",
}


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _assert_artifact(artifact: dict[str, object]) -> None:
    assert artifact["schema_version"] == (
        "route2-operational-analytic-harmonic-rigid-panel-shard-v1"
    )
    assert artifact["status"] == "rigid-panel-shard-passed-not-admitted"
    assert artifact["execution_git_head"] == HEAD
    assert artifact["working_tree_clean"] is True
    assert artifact["checkpoint"]["sha256"] == CHECKPOINT
    assert artifact["capabilities"] == CAPABILITIES
    assert artifact["measurement_sha256"] == MEASUREMENT
    assert_source_files_match_execution_commit(ROOT, artifact)

    contract = artifact["contract"]
    assert contract["asset_sha256"] == (
        "ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3"
    )
    assert contract["shard_start"] == 0
    assert contract["shard_stop"] == 1
    assert contract["shard_molecule_ids"] == ["water"]
    assert contract["rotation_count"] == 3
    assert contract["harmonic_configuration"]["laboratory_fixed_surface_grid"] is False
    assert artifact["decision"] == {
        "all_shard_gates_passed": True,
        "legacy_laboratory_grid_rehabilitated": False,
        "public_energy_admitted": False,
        "public_force_admitted": False,
        "tier_v_admitted": False,
    }

    molecule = artifact["molecules"][0]
    assert len(artifact["molecules"]) == 1
    assert molecule["molecule_id"] == "water"
    assert molecule["chemical_formula"] == "H2O"
    assert molecule["all_gates_passed"] is True
    assert molecule["base"]["iterations"] == 23
    assert molecule["base"]["primal_residual"] == pytest.approx(
        3.545566114861273e-13, rel=0.0, abs=1.0e-27
    )
    assert molecule["base"]["adjoint_residual"] == pytest.approx(
        2.5836982065329233e-15, rel=0.0, abs=1.0e-28
    )
    assert molecule["cold_warm"]["gate_passed"] is True
    assert molecule["cold_warm"]["source_relative_difference"] == 0.0
    assert molecule["cold_warm"]["field_relative_difference"] == 0.0
    assert molecule["cold_warm"]["energy_absolute_difference_eV"] == 0.0
    assert molecule["scalar_identity"]["gate_passed"] is True

    rigid = molecule["rigid_symmetry"]
    assert rigid["all_gates_passed"] is True
    assert all(rigid["gates"].values())
    assert rigid["translation"]["energy_abs_eV"] == 0.0
    assert rigid["translation"]["force_difference_norm_eV_per_A"] == 0.0
    assert rigid["permutation"]["energy_abs_eV"] == 0.0
    assert rigid["maximum_rotation_energy_abs_eV"] == pytest.approx(
        2.0313564164098352e-9, rel=0.0, abs=1.0e-21
    )
    assert rigid["maximum_rotation_force_covariance_relative"] == pytest.approx(
        5.7058772273700395e-8, rel=0.0, abs=1.0e-20
    )
    assert rigid["maximum_rotation_source_covariance_relative"] == pytest.approx(
        1.9864466151380228e-9, rel=0.0, abs=1.0e-21
    )
    assert rigid["maximum_primal_residual"] == pytest.approx(
        4.368923247736517e-13, rel=0.0, abs=1.0e-26
    )
    assert rigid["maximum_adjoint_residual"] == pytest.approx(
        2.6042185552828183e-15, rel=0.0, abs=1.0e-28
    )
    assert molecule["field_covariance"]["maximum_relative"] == pytest.approx(
        4.337470603753039e-9, rel=0.0, abs=1.0e-21
    )

    measured = {key: artifact[key] for key in ("contract", "molecules", "decision")}
    assert canonical_json_sha256(measured) == MEASUREMENT


def test_water_rigid_artifacts_are_bound_and_replay_exactly():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)
    _assert_artifact(primary)
    _assert_artifact(replay)
    for key in ("contract", "molecules", "decision", "measurement_sha256"):
        assert replay[key] == primary[key]
    assert replay["runtime_seconds"] != primary["runtime_seconds"]


def test_water_rigid_evidence_hashes_are_frozen():
    for name, expected in FILES.items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILES
