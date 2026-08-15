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
    / "operational-analytic-harmonic-rigid-thiophene-b8c7ac21"
)
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"
HEAD = "b8c7ac21aee8c0d8301bdc6ad0ddbe2f6cc38d84"
CHECKPOINT = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
MEASUREMENT = "db104947eb42b0802723a101a3fd02dace2f57002b9075c6f748052d7e7763f7"
CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILES = {
    "measurements.json": (
        "675cf5e72104f391cab65bb1ea4c46d294971b0d821174cce97cd8d994e63149"
    ),
    "cold-replay.json": (
        "b3d2f614b883843d6597be4d9ea8d84a2a19bed221fe47229371269979ca6f22"
    ),
    "README.md": "9a0213539dfc2c5cef97dcf0acc16fb8028be3a0fae3017dcf05431c68078749",
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
    assert contract["shard_start"] == 17
    assert contract["shard_stop"] == 18
    assert contract["shard_molecule_ids"] == ["thiophene"]
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
    assert molecule["molecule_id"] == "thiophene"
    assert molecule["chemical_formula"] == "C4H4S"
    assert molecule["all_gates_passed"] is True
    assert molecule["base"]["iterations"] == 22
    assert molecule["base"]["primal_residual"] == pytest.approx(
        8.072873845760511e-13, rel=0.0, abs=1.0e-26
    )
    assert molecule["base"]["adjoint_residual"] == pytest.approx(
        1.029613925086936e-14, rel=0.0, abs=1.0e-27
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
    assert rigid["permutation"]["energy_abs_eV"] == 0.0
    assert rigid["maximum_rotation_energy_abs_eV"] == pytest.approx(
        2.561137080192566e-8, rel=0.0, abs=1.0e-20
    )
    assert rigid["maximum_rotation_force_covariance_relative"] == pytest.approx(
        4.0470329051080785e-8, rel=0.0, abs=1.0e-20
    )
    assert rigid["maximum_rotation_source_covariance_relative"] == pytest.approx(
        7.683791133801178e-9, rel=0.0, abs=1.0e-21
    )
    assert rigid["maximum_primal_residual"] == pytest.approx(
        8.072902860752413e-13, rel=0.0, abs=1.0e-26
    )
    assert rigid["maximum_adjoint_residual"] == pytest.approx(
        1.0305199637545521e-14, rel=0.0, abs=1.0e-27
    )
    assert molecule["field_covariance"]["maximum_relative"] == pytest.approx(
        1.1699202622620264e-8, rel=0.0, abs=1.0e-20
    )

    measured = {key: artifact[key] for key in ("contract", "molecules", "decision")}
    assert canonical_json_sha256(measured) == MEASUREMENT


def test_thiophene_rigid_artifacts_are_bound_and_replay_exactly():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)
    _assert_artifact(primary)
    _assert_artifact(replay)
    for key in ("contract", "molecules", "decision", "measurement_sha256"):
        assert replay[key] == primary[key]
    assert replay["runtime_seconds"] != primary["runtime_seconds"]


def test_thiophene_rigid_evidence_hashes_are_frozen():
    for name, expected in FILES.items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILES
