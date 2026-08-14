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
    / "operational-analytic-harmonic-rigid-acetone-371a2b60"
)
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"

EXECUTION_HEAD = "371a2b605b09bd88286555360b72a5c34baf4f8f"
CHECKPOINT_SHA256 = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
MEASUREMENT_SHA256 = "38efae6ff40ac225b6f1992bcfb00a62249ac695a607bbdb35a7915b3212d387"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILE_SHA256 = {
    "measurements.json": (
        "faec8db8b875ffad20001d74bb02842add026919be4f77336db442567b5947c5"
    ),
    "cold-replay.json": (
        "5d42af54f3cec4504f373f70c9d725243c8521c122aab94da31de74dcd1355c1"
    ),
    "README.md": "ab01a3c63c95d60a8f4cda7fc78431a1bce3e31fba936d314584eb0d789d9622",
}


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_measurement(artifact: dict[str, object]) -> None:
    assert artifact["schema_version"] == (
        "route2-operational-analytic-harmonic-rigid-panel-shard-v1"
    )
    assert artifact["artifact_kind"] == (
        "disabled-operational-analytic-harmonic-rigid-panel-shard"
    )
    assert artifact["status"] == "rigid-panel-shard-passed-not-admitted"
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["working_tree_clean"] is True
    assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert artifact["capabilities"] == NO_CAPABILITIES
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
    assert_source_files_match_execution_commit(ROOT, artifact)

    contract = artifact["contract"]
    assert contract["asset_sha256"] == (
        "ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3"
    )
    assert contract["shard_start"] == 3
    assert contract["shard_stop"] == 4
    assert contract["shard_molecule_ids"] == ["acetone"]
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
    assert len(artifact["molecules"]) == 1
    molecule = artifact["molecules"][0]
    assert molecule["molecule_id"] == "acetone"
    assert molecule["chemical_formula"] == "C3H6O"
    assert molecule["all_gates_passed"] is True
    assert (
        molecule["identity"]["model_release_contract"][
            "structural_so3_equivariance_admitted"
        ]
        is True
    )

    base = molecule["base"]
    assert base["iterations"] == 23
    assert base["primal_residual"] == pytest.approx(
        4.0773181220552224e-13, rel=0.0, abs=1.0e-27
    )
    assert base["adjoint_residual"] == pytest.approx(
        8.13212155028115e-16, rel=0.0, abs=1.0e-29
    )
    assert molecule["cold_warm"]["gate_passed"] is True
    assert molecule["cold_warm"]["source_relative_difference"] == 0.0
    assert molecule["cold_warm"]["field_relative_difference"] == 0.0
    assert molecule["cold_warm"]["energy_absolute_difference_eV"] == 0.0

    scalar = molecule["scalar_identity"]
    assert scalar["gate_passed"] is True
    assert scalar["absolute_error_eV"] == 0.0
    assert scalar["missing_radial_block_max_abs"] == pytest.approx(
        1.3877787807814457e-17, rel=0.0, abs=1.0e-30
    )

    rigid = molecule["rigid_symmetry"]
    assert rigid["all_gates_passed"] is True
    assert all(rigid["gates"].values())
    assert rigid["base_net_force_norm_eV_per_A"] == pytest.approx(
        3.291406376790657e-16, rel=0.0, abs=1.0e-29
    )
    assert rigid["base_torque_norm_eV"] == pytest.approx(
        2.223966914868445e-16, rel=0.0, abs=1.0e-29
    )
    assert rigid["translation"]["energy_abs_eV"] == 0.0
    assert rigid["permutation"]["energy_abs_eV"] == 0.0
    assert rigid["maximum_rotation_energy_abs_eV"] == pytest.approx(
        6.936716090422124e-9, rel=0.0, abs=1.0e-21
    )
    assert rigid["maximum_rotation_force_covariance_relative"] == pytest.approx(
        2.0250564002461316e-8, rel=0.0, abs=1.0e-20
    )
    assert rigid["maximum_rotation_source_covariance_relative"] == pytest.approx(
        6.218784064316717e-9, rel=0.0, abs=1.0e-21
    )
    assert rigid["maximum_primal_residual"] == pytest.approx(
        4.1500973478519087e-13, rel=0.0, abs=1.0e-26
    )
    assert rigid["maximum_adjoint_residual"] == pytest.approx(
        8.625906011240059e-16, rel=0.0, abs=1.0e-29
    )
    assert len(rigid["topology_hashes"]) == 1

    field = molecule["field_covariance"]
    assert field["gate_passed"] is True
    assert field["maximum_relative"] == pytest.approx(
        3.212899614059117e-8, rel=0.0, abs=1.0e-20
    )

    measured = {key: artifact[key] for key in ("contract", "molecules", "decision")}
    assert canonical_json_sha256(measured) == MEASUREMENT_SHA256


def test_acetone_rigid_canary_is_bound_and_fail_closed():
    _assert_measurement(_load(PRIMARY))


def test_acetone_rigid_cold_replay_is_scientifically_identical():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)
    _assert_measurement(primary)
    _assert_measurement(replay)
    for key in ("contract", "molecules", "decision", "measurement_sha256"):
        assert replay[key] == primary[key]
    assert replay["runtime_seconds"] != primary["runtime_seconds"]


def test_acetone_rigid_evidence_file_hashes_are_frozen():
    for name, expected in FILE_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILE_SHA256
