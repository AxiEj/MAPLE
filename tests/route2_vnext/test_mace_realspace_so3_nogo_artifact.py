from __future__ import annotations

import hashlib
import json
from pathlib import Path

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release.evidence import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs" / "route2" / "evidence" / "mace-realspace-so3-nogo-6da676cd"
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"

EXECUTION_HEAD = "6da676cdef27c9f1d72b8fa84a49be326e13adb2"
CHECKPOINT_SHA256 = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
MEASUREMENT_SHA256 = "8774134e8dfdbf540f62a17b6fafbd2778f61b2ffb42af6d79d951c3265112dd"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILE_SHA256 = {
    "measurements.json": "e2f39dc903cbc9e2e8b6c4d2b1a58fa20dfbebabf15e77a0739cddd0cba10408",
    "cold-replay.json": "b7c3992d9591ea45e58cafec9d5c81ebbf7d00e1b3504aa15198a7c55b5935d8",
    "runner.log": "2f1f89eb7e4f0265b532e253e2c63ea4a25b332ac218f9989fd8982e50c3dd7f",
    "cold-replay.log": "c1adc060e14c3134f19cbdf5846d1f8a6ab663875a6d5f22fe11bfaa1d4f49b6",
    "README.md": "b3ee0e008744d5073b74016b8779c199df738a4f973735df3aef9369f81643bb",
}
EXTERNAL_SOURCE_SHA256 = {
    "mace_polar_model": "5ce5372251097f9d6fd17f69f6c63738a6d32278683586522fc70be4d9010e06",
    "gto_electrostatic_features": "57f953dc15d3176b89a2dece143617a005d2989f1743ddd0b33fad66be5af698",
    "realspace_finite_difference_features": (
        "2cf7e098f4960490e6e9a5868ca1ea01ca23beb56ff8ce2cd0a5fe8615aec268"
    ),
    "gto_electrostatic_energy": "81f2d72baad84f4022f40631a8e75bf33c4c3b2108bc665ba0569a33749bc5e5",
    "realspace_finite_difference_energy": (
        "2cf7e098f4960490e6e9a5868ca1ea01ca23beb56ff8ce2cd0a5fe8615aec268"
    ),
}


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_checkpoint_model_so3_counterexample_is_decisive_and_fail_closed():
    artifact = _load(PRIMARY)
    assert artifact["schema_version"] == "route2-mace-molecular-realspace-so3-nogo-v1"
    assert artifact["artifact_kind"] == (
        "disabled-real-checkpoint-model-long-range-so3-counterexample"
    )
    assert artifact["status"] == (
        "model-long-range-so3-counterexample-detected-not-admitted"
    )
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["working_tree_clean"] is True
    assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert artifact["capabilities"] == NO_CAPABILITIES
    assert artifact["decision"] == {
        "current_molecular_realspace_model_global_so3_admitted": False,
        "eight_channel_field_transform_exonerated_by_zero_field": True,
        "first_broken_operator": (
            "graph_longrange.realspace_electrostatics."
            "RealSpaceFiniteDifferenceElectrostaticFeatures"
        ),
        "full_anchored_model_scalar_so3_counterexample_detected": True,
        "isolated_energy_operator_so3_counterexample_detected": True,
        "isolated_feature_operator_so3_counterexample_detected": True,
        "public_force_admitted": False,
        "route2_continuum_exonerated_by_model_only_counterexample": True,
        "tier_v_admitted": False,
    }

    full = artifact["full_model_zero_field_rotation"]
    assert full["anchored_field_energy_eV"]["numerically_equal"] is False
    assert full["anchored_field_energy_eV"]["absolute_difference"] > 8.0e-5
    assert full["energy_gradient_source_covariance"]["relative_l2"] > 1.5e-4
    assert full["original_density_covariance"]["relative_l2"] > 1.5e-4
    gradient = full["fixed_field_coordinate_gradient_covariance_eV_per_A"]
    assert gradient["relative_l2"] > 1.2e-3
    assert gradient["maximum_absolute"] > 3.0e-4

    components = full["checkpoint_energy_components_eV"]
    assert abs(components["interaction_energy"]["signed_difference"]) < 2.0e-9
    assert abs(components["electron_energy"]["signed_difference"]) > 4.0e-5
    assert abs(components["electrostatic_energy"]["signed_difference"]) > 4.5e-5

    isolated = artifact["isolated_upstream_realspace_operators"]
    feature = isolated["feature_operator_covariance"]
    assert feature["numerically_covariant"] is False
    assert feature["relative_l2"] > 2.0e-2
    assert feature["maximum_absolute"] > 1.5e-2
    energy = isolated["energy_operator_rotation"]
    assert energy["numerically_equal"] is False
    assert abs(energy["signed_difference"]) > 5.0e-5
    assert isolated["feature_stencil"]["laboratory_fixed_cartesian_axes"] is True
    assert isolated["energy_stencil"]["laboratory_fixed_cartesian_axes"] is True
    assert isolated["feature_stencil"]["continuous_so3_orbit_closed"] is False
    assert isolated["energy_stencil"]["continuous_so3_orbit_closed"] is False

    sources = artifact["structural_evidence"]["external_python_sources"]
    assert {name: value["sha256"] for name, value in sources.items()} == (
        EXTERNAL_SOURCE_SHA256
    )
    measured = {
        key: artifact[key]
        for key in (
            "protocol",
            "structural_evidence",
            "full_model_zero_field_rotation",
            "isolated_upstream_realspace_operators",
            "decision",
        )
    }
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
    assert canonical_json_sha256(measured) == MEASUREMENT_SHA256
    assert_source_files_match_execution_commit(ROOT, artifact)


def test_model_so3_counterexample_cold_replay_and_file_hashes_are_exact():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)
    for artifact in (primary, replay):
        assert artifact["execution_git_head"] == EXECUTION_HEAD
        assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
        assert artifact["capabilities"] == NO_CAPABILITIES
        assert_source_files_match_execution_commit(ROOT, artifact)
    for key in (
        "protocol",
        "structural_evidence",
        "full_model_zero_field_rotation",
        "isolated_upstream_realspace_operators",
        "decision",
        "measurement_sha256",
    ):
        assert replay[key] == primary[key]

    for name, expected in FILE_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILE_SHA256


def test_model_so3_counterexample_logs_preserve_runtime_warnings_and_decision():
    for name in ("runner.log", "cold-replay.log"):
        log = (EVIDENCE / name).read_text(encoding="utf-8")
        assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" in log
        assert "converting models to float64" in log
        assert "Cuequivariance acceleration will be disabled" in log
        assert '"full_anchored_model_scalar_so3_counterexample_detected": true' in log
        assert '"tier_v_admitted": false' in log
