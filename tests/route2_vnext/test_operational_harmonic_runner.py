from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    ROOT
    / "tools"
    / "route2_release"
    / "run_operational_analytic_harmonic_water_canary.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "route2_operational_analytic_harmonic_runner", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(RUNNER.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(RUNNER.parent))
    return module


def test_help_is_checkpoint_free_and_all_capabilities_remain_closed():
    result = subprocess.run(
        (sys.executable, str(RUNNER), "--help"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout
    assert "--device" in result.stdout
    assert "--output" in result.stdout
    assert "does not admit Route-2 E/F/H/V/M" in result.stdout

    text = RUNNER.read_text(encoding="utf-8")
    assert "NO_CAPABILITIES" in text
    assert '"field_conditioned_model_energy_difference_included": False' in text
    assert '"public_energy_admitted": False' in text
    assert '"public_force_admitted": False' in text
    assert '"tier_v_admitted": False' in text
    assert "original-four-channel-density-head-embedded-first-radial-block" in text
    assert "MACEPolarVariationalFieldEnergy" not in text


def test_protocol_freezes_root_fd_rotation_and_source_identity_gates():
    runner = _load_runner()
    assert runner.SCHEMA_VERSION.endswith("-v1")
    assert runner.ROOT_OPTIONS.tolerance == pytest.approx(2.0e-10)
    assert runner.ROOT_OPTIONS.max_iterations == 120
    assert runner.FD_STEPS_ANGSTROM == (5.0e-4, 2.0e-4, 1.0e-4)
    assert runner.MISSING_RADIAL_BLOCK_ABSOLUTE_TOLERANCE == pytest.approx(1.0e-8)
    assert runner.ROTATION_ENERGY_ABSOLUTE_TOLERANCE_EV == pytest.approx(1.0e-8)
    assert runner.ROTATION_FORCE_RELATIVE_TOLERANCE == pytest.approx(2.0e-6)
    assert runner.SURFACE_LMAX == 1
    assert runner.EXPOSURE_LMAX == 2
    rotation = runner._rotation()
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=2.0e-15)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=2.0e-15)


def test_directional_gate_requires_absolute_and_relative_thresholds():
    runner = _load_runner()
    passed = runner._directional_error(-0.1, -0.100000001)
    assert passed["gate_passed"] is True
    absolute_only = runner._directional_error(1.0e-6, 2.0e-6)
    assert absolute_only["absolute_error_eV_per_A"] < (
        runner.DIRECTIONAL_ABSOLUTE_TOLERANCE_EV_PER_A
    )
    assert absolute_only["relative_error"] > runner.DIRECTIONAL_RELATIVE_TOLERANCE
    assert absolute_only["gate_passed"] is False


def test_measurement_hash_input_excludes_every_wall_time():
    runner = _load_runner()
    records = {
        name: {"kind": name}
        for name in (
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
    measured = runner._measurement_record(**records)
    assert measured == records
    assert "runtime_seconds" not in measured
    assert "timings_seconds" not in measured
