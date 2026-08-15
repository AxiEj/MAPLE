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
    / "operational-analytic-harmonic-rigid-chloroform-920c1b0f"
)
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"
HEAD = "920c1b0f6474c0d201e02549288c3e45437a5907"
CHECKPOINT = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
MEASUREMENT = "99020fbf5d3a893fd3820e682004aaaf4ea0fb7cade9bb7df74550aa9ba921dd"
CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILES = {
    "measurements.json": (
        "6b7b0f050e78d0bddf5b5419240f964daa6548c7984d01cddbc6d02b57872062"
    ),
    "cold-replay.json": (
        "7ce5e5e4fb3b8a06334675173653eed8f3101054f734deafa0567b49f471c564"
    ),
    "README.md": "ba9789edfa954da3288da0fe4c04c38ed8fdcdbedb95f4e971637f89bc57fb86",
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
    assert contract["shard_start"] == 19
    assert contract["shard_stop"] == 20
    assert contract["shard_molecule_ids"] == ["chloroform"]
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
    assert molecule["molecule_id"] == "chloroform"
    assert molecule["chemical_formula"] == "CHCl3"
    assert molecule["all_gates_passed"] is True
    assert molecule["base"]["iterations"] == 22
    assert molecule["base"]["primal_residual"] == pytest.approx(
        5.341749550461511e-13, rel=0.0, abs=1.0e-26
    )
    assert molecule["base"]["adjoint_residual"] == pytest.approx(
        6.164335186956882e-14, rel=0.0, abs=1.0e-27
    )
    assert molecule["cold_warm"]["gate_passed"] is True
    assert molecule["cold_warm"]["source_relative_difference"] == 0.0
    assert molecule["cold_warm"]["field_relative_difference"] == 0.0
    assert molecule["cold_warm"]["energy_absolute_difference_eV"] == 0.0
    assert molecule["scalar_identity"]["gate_passed"] is True
    assert molecule["scalar_identity"]["absolute_error_eV"] == 0.0
    assert molecule["scalar_identity"]["missing_radial_block_max_abs"] == 0.0

    rigid = molecule["rigid_symmetry"]
    assert rigid["all_gates_passed"] is True
    assert all(rigid["gates"].values())
    assert rigid["translation"]["energy_abs_eV"] == 0.0
    assert rigid["permutation"]["energy_abs_eV"] == 0.0
    assert rigid["maximum_rotation_energy_abs_eV"] == pytest.approx(
        3.2669049687683582e-9, rel=0.0, abs=1.0e-21
    )
    assert rigid["maximum_rotation_force_covariance_relative"] == pytest.approx(
        2.1676643473015314e-8, rel=0.0, abs=1.0e-20
    )
    assert rigid["maximum_rotation_source_covariance_relative"] == pytest.approx(
        5.141216133607704e-9, rel=0.0, abs=1.0e-21
    )
    assert rigid["maximum_primal_residual"] == pytest.approx(
        9.019087642060526e-13, rel=0.0, abs=1.0e-26
    )
    assert rigid["maximum_adjoint_residual"] == pytest.approx(
        6.174559832555903e-14, rel=0.0, abs=1.0e-27
    )
    assert molecule["field_covariance"]["maximum_relative"] == pytest.approx(
        5.0708117686345365e-8, rel=0.0, abs=1.0e-20
    )

    measured = {key: artifact[key] for key in ("contract", "molecules", "decision")}
    assert canonical_json_sha256(measured) == MEASUREMENT


def test_chloroform_rigid_artifacts_are_bound_and_replay_exactly():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)
    _assert_artifact(primary)
    _assert_artifact(replay)
    for key in ("contract", "molecules", "decision", "measurement_sha256"):
        assert replay[key] == primary[key]
    assert replay["runtime_seconds"] != primary["runtime_seconds"]


def test_chloroform_rigid_evidence_hashes_are_frozen():
    for name, expected in FILES.items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILES
