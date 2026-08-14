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
    / "operational-analytic-harmonic-rigid-acetonitrile-cf43e050"
)
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"
HEAD = "cf43e050f2c7abfa7f864a516dbe5fe71fac2bf5"
CHECKPOINT = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
MEASUREMENT = "2f01bb45ffa23f1dba09ee15123e71dbae0f742b607ff231758423487cfde616"
CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILES = {
    "measurements.json": (
        "3b3c168582f5da7f6d2b06dc19c995477a24802e9e073c38732fb2d765910c52"
    ),
    "cold-replay.json": (
        "7d822548a0aadc09846a9c782d5bda7a21963b4461b18bc3470cbe0814fa7eaf"
    ),
    "README.md": "4bcd83e1e101c7d8acdc979fb98e6097d26c449b3e4fe1bf63e098e9c3bb5e37",
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
    assert contract["shard_start"] == 4
    assert contract["shard_stop"] == 5
    assert contract["shard_molecule_ids"] == ["acetonitrile"]
    assert contract["rotation_count"] == 3
    assert contract["root"] == {
        "method": "anderson",
        "tolerance": 1.0e-12,
        "max_iterations": 160,
        "damping": 0.7,
        "history": 6,
    }
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
    assert molecule["molecule_id"] == "acetonitrile"
    assert molecule["chemical_formula"] == "C2H3N"
    assert molecule["all_gates_passed"] is True
    assert (
        molecule["identity"]["model_release_contract"][
            "structural_so3_equivariance_admitted"
        ]
        is True
    )
    assert molecule["base"]["iterations"] == 23
    assert molecule["base"]["primal_residual"] == pytest.approx(
        4.5044820892005636e-13, rel=0.0, abs=1.0e-27
    )
    assert molecule["base"]["adjoint_residual"] == pytest.approx(
        9.575193078134051e-14, rel=0.0, abs=1.0e-27
    )
    assert molecule["cold_warm"]["gate_passed"] is True
    assert molecule["cold_warm"]["source_relative_difference"] == 0.0
    assert molecule["cold_warm"]["field_relative_difference"] == 0.0
    assert molecule["cold_warm"]["energy_absolute_difference_eV"] == 0.0
    assert molecule["scalar_identity"]["gate_passed"] is True

    rigid = molecule["rigid_symmetry"]
    assert rigid["all_gates_passed"] is True
    assert all(rigid["gates"].values())
    assert rigid["maximum_rotation_energy_abs_eV"] == pytest.approx(
        1.7384991224389523e-8, rel=0.0, abs=1.0e-20
    )
    assert rigid["maximum_rotation_force_covariance_relative"] == pytest.approx(
        1.7558870386345935e-8, rel=0.0, abs=1.0e-20
    )
    assert rigid["maximum_rotation_source_covariance_relative"] == pytest.approx(
        3.395228677265498e-9, rel=0.0, abs=1.0e-21
    )
    assert rigid["maximum_primal_residual"] == pytest.approx(
        7.579717824709418e-13, rel=0.0, abs=1.0e-26
    )
    assert rigid["maximum_adjoint_residual"] == pytest.approx(
        9.575338705685621e-14, rel=0.0, abs=1.0e-27
    )
    assert molecule["field_covariance"]["maximum_relative"] == pytest.approx(
        8.92400703128298e-9, rel=0.0, abs=1.0e-21
    )
    measured = {key: artifact[key] for key in ("contract", "molecules", "decision")}
    assert canonical_json_sha256(measured) == MEASUREMENT


def test_acetonitrile_rigid_artifacts_are_bound_and_replay_exactly():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)
    _assert_artifact(primary)
    _assert_artifact(replay)
    for key in ("contract", "molecules", "decision", "measurement_sha256"):
        assert replay[key] == primary[key]
    assert replay["runtime_seconds"] != primary["runtime_seconds"]


def test_acetonitrile_rigid_evidence_hashes_are_frozen():
    for name, expected in FILES.items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILES
