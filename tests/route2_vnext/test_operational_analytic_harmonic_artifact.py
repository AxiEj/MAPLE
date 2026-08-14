from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release.evidence import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "operational-analytic-harmonic-water-fa6f0200"
)
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"

EXECUTION_HEAD = "fa6f02001b17356734bb801597a31cfe22027fae"
CHECKPOINT_SHA256 = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
MEASUREMENT_SHA256 = "586024051151be3e3c63171e73bab34b6ce43459d4232c8070daadcd204b4547"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILE_SHA256 = {
    "measurements.json": (
        "15e1fa2b911e0c89130fd0d6df2ee2a807c579b09c4e6796d83bb0b3995ec19a"
    ),
    "cold-replay.json": (
        "b0f1989e15e20bc2f1dbe51724580b3bf7327cdad5de6e674afa93b232325256"
    ),
    "README.md": "2f07e6a2df202453a8ecb449498420c7b4a13bac8e1c27c8032fb2de234b9e7e",
}


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_measurement(artifact: dict[str, object]) -> None:
    assert artifact["schema_version"] == (
        "route2-operational-analytic-original-source-harmonic-water-"
        "real-checkpoint-canary-v1"
    )
    assert artifact["artifact_kind"] == (
        "disabled-real-checkpoint-operational-analytic-original-source-"
        "harmonic-scalar-canary"
    )
    assert artifact["status"] == "operational-canary-passed-not-admitted"
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["working_tree_clean"] is True
    assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert artifact["capabilities"] == NO_CAPABILITIES
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
    assert_source_files_match_execution_commit(ROOT, artifact)

    identity = artifact["identity"]
    assert identity["long_range_evaluator_profile"] == (
        "graph-longrange-analytic-gaussian-multipole-realspace-v1"
    )
    assert identity["long_range_structural_so3_equivariance_admitted"] is True
    assert identity["scalar_id"] == (
        "route2-operational-macepolar-analytic-gaussian-multipole-"
        "smoothharmonicgalerkin-cpcm-v1"
    )

    assert artifact["decision"] == {
        "cold_warm_root_gate_passed": True,
        "continuum_coefficient_architecture_is_so3_equivariant": True,
        "continuum_scalar_identity_gate_passed": True,
        "field_conditioned_model_energy_difference_included": False,
        "net_force_gate_passed": True,
        "operational_canary_passed": True,
        "public_energy_admitted": False,
        "public_force_admitted": False,
        "rigid_rotation_gate_passed": True,
        "source_model_identity": (
            "original-four-channel-density-head-embedded-first-radial-block"
        ),
        "stationary_root_converged": True,
        "three_step_force_directional_gate_passed": True,
        "tier_v_admitted": False,
        "torque_gate_passed": True,
    }

    root = artifact["center_root_replay"]
    assert root["contract"] == "route2-root-equivalence-v1"
    assert root["gate_passed"] is True
    assert root["numerically_equivalent"] is True
    assert root["source_relative_difference"] == 0.0
    assert root["energy_abs_difference_eV"] == 0.0
    cold = root["cold"]
    warm = root["warm"]
    assert cold["iterations"] == 31
    assert warm["iterations"] == 0
    assert cold["actual_unmixed_residual_norm"] == pytest.approx(
        1.7379968031740547e-10, rel=0.0, abs=1.0e-24
    )
    assert cold["total_energy_eV"] == pytest.approx(
        -2079.8811744282916, rel=0.0, abs=1.0e-12
    )
    assert cold["continuum_energy_eV"] == pytest.approx(
        0.5 * cold["coupling_energy_eV"], rel=2.0e-15, abs=2.0e-17
    )
    source = np.asarray(cold["source"], dtype=float)
    assert float(np.sum(source[:, :2])) == pytest.approx(0.0, abs=2.0e-15)
    np.testing.assert_array_equal(source[:, (1, 5, 6, 7)], 0.0)

    scalar = artifact["scalar_identity"]
    assert scalar["gate_passed"] is True
    assert scalar["absolute_error_eV"] == 0.0
    assert scalar["missing_radial_block_max_abs"] == 0.0

    force = artifact["force_directional_derivative"]
    assert force["admitted"] is False
    assert force["all_steps_passed"] is True
    assert force["net_force_gate_passed"] is True
    assert force["torque_gate_passed"] is True
    assert force["net_force_norm_eV_per_A"] == pytest.approx(
        2.7755575615628914e-17, rel=0.0, abs=1.0e-30
    )
    assert force["torque_norm_eV"] == pytest.approx(
        1.6752969150823915e-10, rel=0.0, abs=1.0e-22
    )
    steps = force["directional_steps"]
    assert [record["step_A"] for record in steps] == [5.0e-4, 2.0e-4, 1.0e-4]
    assert all(record["gate_passed"] is True for record in steps)
    assert [record["absolute_error_eV_per_A"] for record in steps] == pytest.approx(
        [
            3.5913535214593395e-6,
            5.477295019762352e-7,
            8.616094082647408e-8,
        ],
        rel=0.0,
        abs=1.0e-19,
    )

    rotation = artifact["rigid_rotation"]
    assert rotation["gate_passed"] is True
    continuum = rotation["continuum_structural_check"]
    assert continuum["energy_absolute_error_eV"] == 0.0
    assert continuum["gradient_max_absolute_error_eV_per_A"] == pytest.approx(
        1.8214596497756474e-17, rel=0.0, abs=1.0e-29
    )
    operational = rotation["operational_scalar_check"]
    assert operational["energy_absolute_error_eV"] == pytest.approx(
        2.7694113668985665e-9, rel=0.0, abs=1.0e-21
    )
    assert operational["source_relative_error"] == pytest.approx(
        8.025998472808204e-9, rel=0.0, abs=1.0e-21
    )
    assert operational["field_relative_error"] == pytest.approx(
        2.3417317587664313e-9, rel=0.0, abs=1.0e-21
    )
    assert operational["force_relative_error"] == pytest.approx(
        7.857995561759406e-8, rel=0.0, abs=1.0e-20
    )
    assert operational["force_max_absolute_error_eV_per_A"] == pytest.approx(
        2.2821200651446105e-8, rel=0.0, abs=1.0e-20
    )

    measured = {
        key: artifact[key]
        for key in (
            "protocol",
            "geometry",
            "identity",
            "center_root_replay",
            "scalar_identity",
            "force_directional_derivative",
            "rigid_rotation",
            "decision",
        )
    }
    assert canonical_json_sha256(measured) == MEASUREMENT_SHA256


def test_operational_analytic_harmonic_canary_is_bound_and_fail_closed():
    _assert_measurement(_load(PRIMARY))


def test_operational_analytic_harmonic_cold_replay_is_scientifically_identical():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)
    _assert_measurement(primary)
    _assert_measurement(replay)
    for key in (
        "protocol",
        "geometry",
        "identity",
        "center_root_replay",
        "scalar_identity",
        "force_directional_derivative",
        "rigid_rotation",
        "decision",
        "measurement_sha256",
    ):
        assert replay[key] == primary[key]
    assert replay["runtime_seconds"] != primary["runtime_seconds"]


def test_operational_analytic_harmonic_evidence_file_hashes_are_frozen():
    assert FILE_SHA256
    for name, expected in FILE_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILE_SHA256
