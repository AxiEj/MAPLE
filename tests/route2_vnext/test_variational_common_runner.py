from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools" / "route2_release" / "run_variational_common_water_canary.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "route2_variational_common_runner", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_help_is_checkpoint_free_and_keeps_every_capability_closed():
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
    assert '"global_so3_guarantee": False' in text
    assert '"tier_v_admitted": False' in text
    assert '"public_force_admitted": False' in text
    assert "laboratory-fixed 194-point" in text


def test_directional_gate_requires_both_absolute_and_relative_thresholds():
    runner = _load_runner()
    passed = runner._directional_error(-0.1039157969, -0.1039157928)
    assert passed["gate_passed"] is True

    absolute_only = runner._directional_error(1.0e-6, 2.0e-6)
    assert absolute_only["absolute_error_eV_per_A"] < (
        runner.DIRECTIONAL_ABSOLUTE_TOLERANCE_EV_PER_A
    )
    assert absolute_only["relative_error"] > runner.DIRECTIONAL_RELATIVE_TOLERANCE
    assert absolute_only["gate_passed"] is False

    relative_only = runner._directional_error(1.0, 1.0006)
    assert relative_only["relative_error"] < runner.DIRECTIONAL_RELATIVE_TOLERANCE
    assert relative_only["absolute_error_eV_per_A"] > (
        runner.DIRECTIONAL_ABSOLUTE_TOLERANCE_EV_PER_A
    )
    assert relative_only["gate_passed"] is False


def test_protocol_freezes_three_steps_root_tolerance_and_one_direction():
    runner = _load_runner()
    assert runner.SCHEMA_VERSION.endswith("-v1")
    assert runner.FD_STEPS_ANGSTROM == (5.0e-4, 2.0e-4, 1.0e-4)
    assert runner.ROOT_OPTIONS.tolerance == pytest.approx(2.0e-10)
    direction = runner._direction((3, 3))
    np.testing.assert_allclose(direction.sum(axis=0), 0.0, rtol=0.0, atol=1.0e-15)
    assert float((direction**2).sum()) == pytest.approx(1.0, abs=1.0e-15)
