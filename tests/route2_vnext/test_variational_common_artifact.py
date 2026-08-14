from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release.evidence import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs" / "route2" / "evidence" / "variational-common-water-576550e9"
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"
EXECUTION_HEAD = "576550e9cfe1011519c7bd93e7ee256972ff2560"
CHECKPOINT_SHA256 = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
MEASUREMENT_SHA256 = "62d43c63868270fc74254cf0ddbc182b33af0d158c3fbca1ef6493cdf52ee9fc"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILE_SHA256 = {
    "measurements.json": (
        "ca9083365c4c83b0cd8dce2eeba9d54c3d1b6153a2055b060d80bdfcb17ced3f"
    ),
    "runner.stdout.log": (
        "d3c1c181eecba0bf9615b21bde33cc4a79a6752c1d51ef85888b0a459bf3a055"
    ),
    "runner.stderr.log": (
        "23bb6c158b2504926eec42412b5476be78040e5edd4e221ef34d880ab3494d65"
    ),
    "cold-replay.json": (
        "1b2fce85486c85e07646d1f93f99ca381bb74656d0a6a006925acb7e6cc6c5c2"
    ),
    "cold-replay.stdout.log": (
        "5247a1513690c6e176e8f0bf46cd705076e27e6cd4633e3dc4bfdd5ee10dc6c1"
    ),
    "cold-replay.stderr.log": (
        "2e9d30ea7a2c68d907324e8e21c949ceb899d5eccb31f12ffe20f1e420a92706"
    ),
    "README.md": ("201299cbce05d16cea3a93bffe8e7138525a6bff05121c0889786430e1c40f3f"),
}


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_measurement(artifact: dict[str, object]) -> None:
    assert artifact["schema_version"] == (
        "route2-variational-common-water-real-checkpoint-canary-v1"
    )
    assert artifact["artifact_kind"] == (
        "disabled-real-checkpoint-changed-source-common-scalar-canary"
    )
    assert artifact["status"] == "same-scalar-canary-passed-not-admitted"
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["working_tree_clean"] is True
    assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert artifact["capabilities"] == NO_CAPABILITIES
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
    assert_source_files_match_execution_commit(ROOT, artifact)

    decision = artifact["decision"]
    assert decision == {
        "cold_warm_root_gate_passed": True,
        "global_so3_guarantee": False,
        "original_density_head_role": "zero-field-anchor-and-diagnostic-only",
        "public_force_admitted": False,
        "same_scalar_real_checkpoint_canary_passed": True,
        "source_model_identity": (
            "changed-complete-eight-channel-field-energy-gradient-effective-source"
        ),
        "stationary_root_converged": True,
        "three_step_envelope_directional_gate_passed": True,
        "tier_v_admitted": False,
    }

    replay = artifact["center_root_replay"]
    assert replay["contract"] == "route2-root-equivalence-v1"
    assert replay["gate_passed"] is True
    assert replay["numerically_equivalent"] is True
    assert replay["source_l2_difference"] == 0.0
    assert replay["source_relative_difference"] == 0.0
    assert replay["energy_abs_difference_eV"] == 0.0
    cold = replay["cold"]
    warm = replay["warm"]
    assert cold["converged"] is True
    assert cold["iterations"] == 32
    assert warm["iterations"] == 0
    assert cold["actual_unmixed_residual_norm"] == pytest.approx(
        1.104397210579806e-10, rel=0.0, abs=1.0e-24
    )
    assert cold["continuum_energy_eV"] == pytest.approx(
        0.5 * cold["coupling_energy_eV"], rel=2.0e-15, abs=2.0e-17
    )
    assert cold["total_energy_eV"] == pytest.approx(
        cold["model_energy_eV"]
        - cold["coupling_energy_eV"]
        + cold["continuum_energy_eV"],
        rel=0.0,
        abs=2.0e-13,
    )
    source = np.asarray(cold["source"], dtype=float)
    assert np.linalg.norm(source[:, (1, 5, 6, 7)]) > 1.0e-3
    assert float(np.sum(source[:, :2])) == pytest.approx(0.0, abs=2.0e-15)

    envelope = artifact["envelope_directional_derivative"]
    assert envelope["admitted"] is False
    assert envelope["all_steps_passed"] is True
    steps = envelope["directional_steps"]
    assert [record["step_A"] for record in steps] == [5.0e-4, 2.0e-4, 1.0e-4]
    assert all(record["gate_passed"] is True for record in steps)
    assert max(record["absolute_error_eV_per_A"] for record in steps) == pytest.approx(
        2.9895738047208686e-6, rel=0.0, abs=1.0e-18
    )
    assert min(record["absolute_error_eV_per_A"] for record in steps) == pytest.approx(
        4.157446151231703e-9, rel=0.0, abs=1.0e-20
    )

    measured = {
        key: artifact[key]
        for key in (
            "protocol",
            "geometry",
            "identity",
            "center_root_replay",
            "envelope_directional_derivative",
            "decision",
        )
    }
    assert canonical_json_sha256(measured) == MEASUREMENT_SHA256


def test_real_checkpoint_common_scalar_canary_is_source_bound_and_fail_closed():
    _assert_measurement(_load(PRIMARY))


def test_real_checkpoint_common_scalar_cold_process_replay_is_exact():
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
        "decision",
        "measurement_sha256",
    ):
        assert replay[key] == primary[key]


def test_common_scalar_evidence_files_and_runtime_warnings_are_preserved():
    for name, expected in FILE_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected

    recorded = {}
    for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        recorded[name] = digest
    assert recorded == FILE_SHA256

    for name in ("runner.stderr.log", "cold-replay.stderr.log"):
        log = (EVIDENCE / name).read_text(encoding="utf-8")
        assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" in log
        assert "converting models to float64" in log
        assert "Exit status: 0" in log
    for name in ("runner.stdout.log", "cold-replay.stdout.log"):
        log = (EVIDENCE / name).read_text(encoding="utf-8")
        assert "Cuequivariance acceleration will be disabled" in log
        assert '"same_scalar_real_checkpoint_canary_passed": true' in log
        assert '"global_so3_guarantee": false' in log
        assert '"tier_v_admitted": false' in log
