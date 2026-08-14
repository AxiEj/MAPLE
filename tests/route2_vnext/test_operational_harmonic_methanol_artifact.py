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
    / "operational-analytic-harmonic-rigid-methanol-93c98598"
)
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"

EXECUTION_HEAD = "93c985980756a590b761839043e38cb9f1107df7"
CHECKPOINT_SHA256 = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
MEASUREMENT_SHA256 = "2bd5b02b6376faa39789cd9b081985f6805903a9ce227044842bf57af666ccfe"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILE_SHA256 = {
    "measurements.json": (
        "d9022f228f0838c732e1f33fb85972d0380465984d4cb62787265b4953f64334"
    ),
    "cold-replay.json": (
        "e2a5f3c8b7656bb77f60d78eb7f1f31aeeac71c69b5a093b1ef1ad66b5ff5647"
    ),
    "README.md": "3e24c0a1f48bc71242e69381babdc317d4894e6a8c8ce3cf97f896571191c2cf",
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
    assert contract["shard_start"] == 1
    assert contract["shard_stop"] == 2
    assert contract["shard_molecule_ids"] == ["methanol"]
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
    assert molecule["molecule_id"] == "methanol"
    assert molecule["chemical_formula"] == "CH4O"
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
        3.1258015520278824e-13, rel=0.0, abs=1.0e-27
    )
    assert base["adjoint_residual"] == pytest.approx(
        6.934609168311564e-16, rel=0.0, abs=1.0e-29
    )
    assert base["energy_eV"] == pytest.approx(-3148.8354478311776, rel=0.0, abs=1.0e-12)

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
        3.469446951953614e-18, rel=0.0, abs=1.0e-30
    )
    assert scalar["missing_radial_block_max_abs"] == pytest.approx(
        5.692061405548898e-19, rel=0.0, abs=1.0e-31
    )

    rigid = molecule["rigid_symmetry"]
    assert rigid["all_gates_passed"] is True
    assert all(rigid["gates"].values())
    assert rigid["base_net_force_norm_eV_per_A"] == pytest.approx(
        1.3455014484288012e-16, rel=0.0, abs=1.0e-29
    )
    assert rigid["base_torque_norm_eV"] == pytest.approx(
        5.441308376097357e-10, rel=0.0, abs=1.0e-22
    )
    assert rigid["translation"]["energy_abs_eV"] == 0.0
    assert rigid["translation"]["force_difference_norm_eV_per_A"] == pytest.approx(
        5.702249140124941e-15, rel=0.0, abs=1.0e-27
    )
    assert rigid["permutation"]["energy_abs_eV"] == 0.0
    assert rigid["permutation"]["force_covariance_relative"] == pytest.approx(
        5.953525952799556e-16, rel=0.0, abs=1.0e-28
    )
    assert rigid["maximum_rotation_energy_abs_eV"] == pytest.approx(
        2.8617250791285187e-9, rel=0.0, abs=1.0e-21
    )
    assert rigid["maximum_rotation_force_covariance_relative"] == pytest.approx(
        5.000913075374153e-8, rel=0.0, abs=1.0e-20
    )
    assert rigid["maximum_rotation_source_covariance_relative"] == pytest.approx(
        2.592598170552474e-9, rel=0.0, abs=1.0e-21
    )
    assert rigid["maximum_primal_residual"] == pytest.approx(
        5.6774011402604e-13, rel=0.0, abs=1.0e-26
    )
    assert rigid["maximum_adjoint_residual"] == pytest.approx(
        7.006299331053965e-16, rel=0.0, abs=1.0e-29
    )
    assert len(rigid["topology_hashes"]) == 1

    field = molecule["field_covariance"]
    assert field["gate_passed"] is True
    assert field["maximum_relative"] == pytest.approx(
        4.269092690650618e-9, rel=0.0, abs=1.0e-21
    )

    measured = {key: artifact[key] for key in ("contract", "molecules", "decision")}
    assert canonical_json_sha256(measured) == MEASUREMENT_SHA256


def test_methanol_rigid_canary_is_bound_and_fail_closed():
    _assert_measurement(_load(PRIMARY))


def test_methanol_rigid_cold_replay_is_scientifically_identical():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)
    _assert_measurement(primary)
    _assert_measurement(replay)
    for key in ("contract", "molecules", "decision", "measurement_sha256"):
        assert replay[key] == primary[key]
    assert replay["runtime_seconds"] != primary["runtime_seconds"]


def test_methanol_rigid_evidence_file_hashes_are_frozen():
    for name, expected in FILE_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILE_SHA256
