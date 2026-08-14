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
    / "variational-analytic-harmonic-water-50809803"
)
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"

EXECUTION_HEAD = "508098034faaad28beceb5d7ee6243aa04bddca1"
CHECKPOINT_SHA256 = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
MEASUREMENT_SHA256 = "7ce9e9c07f40552ea513e0bbd4f29f4e4a5aa6887d7b5888c5647756525fe500"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILE_SHA256 = {
    "measurements.json": (
        "c75ebdee104244937c74dc13f49640e5c393a0c742ab1515b9e0e9336f7e1328"
    ),
    "cold-replay.json": (
        "2a72a853f0c0b07c510b2afe1cf3802182209c3974b7bd63d837be34207a74d7"
    ),
    "README.md": ("59d6506e8c933ee6043faf5212269f2f810408a0c41c7b0919619e706e8561c7"),
}


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_measurement(artifact: dict[str, object]) -> None:
    assert artifact["schema_version"] == (
        "route2-variational-analytic-gaussian-multipole-harmonic-water-"
        "real-checkpoint-canary-v1"
    )
    assert artifact["artifact_kind"] == (
        "disabled-real-checkpoint-analytic-gaussian-multipole-"
        "changed-source-harmonic-common-scalar-canary"
    )
    assert artifact["status"] == "same-scalar-canary-passed-not-admitted"
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
    assert identity["long_range_symmetry_contract_id"] == (
        "analytic-isotropic-gaussian-multipole-pair-kernel-so3-structural-v1"
    )
    assert identity["long_range_structural_so3_equivariance_admitted"] is True
    assert identity["scalar_id"] == (
        "route2-variational-macepolar-analytic-gaussian-multipole-"
        "energygradient-smoothharmonicgalerkin-cpcm-v1"
    )

    assert artifact["decision"] == {
        "cold_warm_root_gate_passed": True,
        "continuum_coefficient_architecture_is_so3_equivariant": True,
        "full_common_scalar_global_so3_admitted": False,
        "model_long_range_structural_so3_equivariance_admitted": True,
        "original_density_head_role": "zero-field-anchor-and-diagnostic-only",
        "public_force_admitted": False,
        "rigid_rotation_gate_passed": True,
        "same_scalar_real_checkpoint_canary_passed": True,
        "source_model_identity": (
            "changed-complete-eight-channel-field-energy-gradient-effective-source"
        ),
        "stationary_root_converged": True,
        "three_step_envelope_directional_gate_passed": True,
        "tier_v_admitted": False,
    }

    root = artifact["center_root_replay"]
    assert root["contract"] == "route2-root-equivalence-v1"
    assert root["gate_passed"] is True
    assert root["numerically_equivalent"] is True
    assert root["source_l2_difference"] == 0.0
    assert root["source_relative_difference"] == 0.0
    assert root["energy_abs_difference_eV"] == 0.0
    cold = root["cold"]
    warm = root["warm"]
    assert cold["iterations"] == 31
    assert warm["iterations"] == 0
    assert cold["actual_unmixed_residual_norm"] == pytest.approx(
        1.720697910222882e-10, rel=0.0, abs=1.0e-24
    )
    assert cold["total_energy_eV"] == pytest.approx(
        -2079.8808921464733, rel=0.0, abs=1.0e-12
    )
    assert cold["continuum_energy_eV"] == pytest.approx(
        0.5 * cold["coupling_energy_eV"], rel=2.0e-15, abs=2.0e-17
    )
    source = np.asarray(cold["source"], dtype=float)
    assert float(np.sum(source[:, :2])) == pytest.approx(0.0, abs=2.0e-15)
    assert np.linalg.norm(source[:, (1, 5, 6, 7)]) > 1.0e-3

    envelope = artifact["envelope_directional_derivative"]
    assert envelope["admitted"] is False
    assert envelope["all_steps_passed"] is True
    steps = envelope["directional_steps"]
    assert [record["step_A"] for record in steps] == [5.0e-4, 2.0e-4, 1.0e-4]
    assert all(record["gate_passed"] is True for record in steps)
    assert [record["absolute_error_eV_per_A"] for record in steps] == pytest.approx(
        [
            1.5331551746211591e-7,
            2.5758885538462728e-8,
            6.432123125788003e-9,
        ],
        rel=0.0,
        abs=1.0e-20,
    )

    rotation = artifact["rigid_rotation"]
    assert "solve_seconds" not in rotation
    assert artifact["rotation_solve_seconds"] > 0.0
    assert rotation["gate_passed"] is True
    continuum = rotation["continuum_structural_check"]
    assert continuum["energy_absolute_error_eV"] == 0.0
    assert continuum["gradient_max_absolute_error_eV_per_A"] == pytest.approx(
        7.806255641895632e-18, rel=0.0, abs=1.0e-30
    )
    common = rotation["common_stationary_check"]
    assert common["energy_absolute_error_eV"] == pytest.approx(
        2.799424692057073e-9, rel=0.0, abs=1.0e-21
    )
    assert common["source_relative_error"] == pytest.approx(
        7.89339570981596e-9, rel=0.0, abs=1.0e-21
    )
    assert common["gradient_relative_error"] == pytest.approx(
        8.198584915195558e-8, rel=0.0, abs=1.0e-20
    )

    measured = {
        key: artifact[key]
        for key in (
            "protocol",
            "geometry",
            "identity",
            "center_root_replay",
            "envelope_directional_derivative",
            "rigid_rotation",
            "decision",
        )
    }
    assert canonical_json_sha256(measured) == MEASUREMENT_SHA256


def test_analytic_harmonic_common_scalar_canary_is_bound_and_fail_closed():
    _assert_measurement(_load(PRIMARY))


def test_analytic_harmonic_cold_replay_has_identical_scientific_measurement():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)
    _assert_measurement(primary)
    _assert_measurement(replay)
    for key in (
        "protocol",
        "geometry",
        "identity",
        "center_root_replay",
        "envelope_directional_derivative",
        "rigid_rotation",
        "decision",
        "measurement_sha256",
    ):
        assert replay[key] == primary[key]
    assert replay["runtime_seconds"] != primary["runtime_seconds"]


def test_analytic_harmonic_evidence_file_hashes_are_frozen():
    for name, expected in FILE_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILE_SHA256
