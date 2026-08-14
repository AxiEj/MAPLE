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
    / "operational-analytic-harmonic-rigid-ethanol-538f9f4d"
)
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"

EXECUTION_HEAD = "538f9f4d350c4f2f0ed0519a59dbae4858df75a7"
CHECKPOINT_SHA256 = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
MEASUREMENT_SHA256 = "575420ae56c17f8dc37d28536ba86c9cba0f13559552f6afdc7934c4a2945937"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILE_SHA256 = {
    "measurements.json": (
        "8e23ff06d4571c3b72ebb12e49ef8be3808a12cf697da3ce40f9018d0a2a1030"
    ),
    "cold-replay.json": (
        "b259a5fd0d0fbe7f239467fb42180be8519b0c0c68146e230ccffbb580c79696"
    ),
    "README.md": "f86065f2eb5fcb446cf47b378d9ccde8aaddf20d949fb5d70304cc62c8873afa",
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
    assert contract["contract_version"] == (
        "route2-operational-analytic-harmonic-rigid-panel-v1"
    )
    assert contract["asset_sha256"] == (
        "ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3"
    )
    assert contract["shard_start"] == 2
    assert contract["shard_stop"] == 3
    assert contract["shard_molecule_ids"] == ["ethanol"]
    assert contract["rotation_count"] == 3
    assert contract["root"] == {
        "method": "anderson",
        "tolerance": 1.0e-12,
        "max_iterations": 160,
        "damping": 0.7,
        "history": 6,
    }
    assert contract["harmonic_configuration"]["laboratory_fixed_surface_grid"] is False
    assert contract["scalar_id"] == (
        "route2-operational-macepolar-analytic-gaussian-multipole-"
        "smoothharmonicgalerkin-cpcm-v1"
    )

    assert artifact["decision"] == {
        "all_shard_gates_passed": True,
        "legacy_laboratory_grid_rehabilitated": False,
        "public_energy_admitted": False,
        "public_force_admitted": False,
        "tier_v_admitted": False,
    }
    assert len(artifact["molecules"]) == 1
    molecule = artifact["molecules"][0]
    assert molecule["molecule_id"] == "ethanol"
    assert molecule["chemical_formula"] == "C2H6O"
    assert molecule["all_gates_passed"] is True
    identity = molecule["identity"]
    assert (
        identity["model_release_contract"]["structural_so3_equivariance_admitted"]
        is True
    )
    assert len(identity["continuum_topology_sha256"]) == 64

    base = molecule["base"]
    assert base["iterations"] == 23
    assert base["primal_residual"] == pytest.approx(
        3.3676767929859406e-13, rel=0.0, abs=1.0e-27
    )
    assert base["adjoint_residual"] == pytest.approx(
        3.548825119053039e-13, rel=0.0, abs=1.0e-26
    )
    assert base["energy_eV"] == pytest.approx(-4218.493133014895, rel=0.0, abs=1.0e-12)

    replay = molecule["cold_warm"]
    assert replay["numerically_equivalent"] is True
    assert replay["source_relative_difference"] == 0.0
    assert replay["field_relative_difference"] == 0.0
    assert replay["energy_absolute_difference_eV"] == 0.0
    assert replay["warm_iterations"] == 0
    assert replay["gate_passed"] is True

    scalar = molecule["scalar_identity"]
    assert scalar["gate_passed"] is True
    assert scalar["absolute_error_eV"] == pytest.approx(
        1.3877787807814457e-17, rel=0.0, abs=1.0e-30
    )
    assert scalar["missing_radial_block_max_abs"] == 0.0

    rigid = molecule["rigid_symmetry"]
    assert rigid["all_gates_passed"] is True
    assert all(rigid["gates"].values())
    assert rigid["base_net_force_norm_eV_per_A"] == pytest.approx(
        5.815852873680767e-17, rel=0.0, abs=1.0e-29
    )
    assert rigid["base_torque_norm_eV"] == pytest.approx(
        2.3096049378557194e-10, rel=0.0, abs=1.0e-22
    )
    assert rigid["translation"]["energy_abs_eV"] == 0.0
    assert rigid["translation"]["force_difference_norm_eV_per_A"] == pytest.approx(
        1.1801983333760825e-14, rel=0.0, abs=1.0e-27
    )
    assert rigid["permutation"]["energy_abs_eV"] == 0.0
    assert rigid["permutation"]["force_covariance_relative"] == pytest.approx(
        4.981661729142136e-16, rel=0.0, abs=1.0e-28
    )
    assert rigid["maximum_rotation_energy_abs_eV"] == pytest.approx(
        1.9072103896178305e-9, rel=0.0, abs=1.0e-21
    )
    assert rigid["maximum_rotation_force_covariance_relative"] == pytest.approx(
        4.7353129839838976e-8, rel=0.0, abs=1.0e-20
    )
    assert rigid["maximum_rotation_source_covariance_relative"] == pytest.approx(
        3.2020656091357397e-9, rel=0.0, abs=1.0e-21
    )
    assert rigid["maximum_primal_residual"] == pytest.approx(
        7.856985340851677e-13, rel=0.0, abs=1.0e-26
    )
    assert rigid["maximum_adjoint_residual"] == pytest.approx(
        3.548967219884553e-13, rel=0.0, abs=1.0e-26
    )
    assert len(rigid["topology_hashes"]) == 1

    field = molecule["field_covariance"]
    assert field["gate_passed"] is True
    assert field["maximum_relative"] == pytest.approx(
        2.3729140073528977e-9, rel=0.0, abs=1.0e-21
    )

    measured = {key: artifact[key] for key in ("contract", "molecules", "decision")}
    assert canonical_json_sha256(measured) == MEASUREMENT_SHA256


def test_ethanol_rigid_canary_is_bound_and_fail_closed():
    _assert_measurement(_load(PRIMARY))


def test_ethanol_rigid_cold_replay_is_scientifically_identical():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)
    _assert_measurement(primary)
    _assert_measurement(replay)
    for key in ("contract", "molecules", "decision", "measurement_sha256"):
        assert replay[key] == primary[key]
    assert replay["runtime_seconds"] != primary["runtime_seconds"]


def test_ethanol_rigid_evidence_file_hashes_are_frozen():
    for name, expected in FILE_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILE_SHA256
