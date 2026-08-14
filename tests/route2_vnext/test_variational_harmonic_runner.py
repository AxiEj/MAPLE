from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools" / "route2_release" / "run_variational_harmonic_water_canary.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "route2_variational_harmonic_runner", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_help_is_checkpoint_free_and_keeps_capabilities_closed():
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
    assert "--model-evaluator-profile" in result.stdout
    assert "graph-longrange-analytic-gaussian-multipole-realspace-v1" in result.stdout
    assert "--output" in result.stdout
    assert "does not admit Route-2 E/F/H/V/M" in result.stdout

    text = RUNNER.read_text(encoding="utf-8")
    assert "NO_CAPABILITIES" in text
    assert '"continuum_coefficient_architecture_is_so3_equivariant": True' in text
    assert '"full_common_scalar_global_so3_admitted": False' in text
    assert '"tier_v_admitted": False' in text
    assert '"public_force_admitted": False' in text
    assert "no laboratory-fixed cavity grid" in text
    assert "ANALYTIC_SCHEMA_VERSION" in text
    assert "runtime/analytic_gaussian_multipole.py" in text


def test_protocol_freezes_harmonic_orders_root_fd_and_rotation_gates():
    runner = _load_runner()
    assert runner.SCHEMA_VERSION.endswith("-v1")
    assert runner.FD_STEPS_ANGSTROM == (5.0e-4, 2.0e-4, 1.0e-4)
    assert runner.ROOT_OPTIONS.tolerance == pytest.approx(2.0e-10)
    assert runner.SURFACE_LMAX == 1
    assert runner.EXPOSURE_LMAX == 2
    assert runner.TRANSITION_WIDTH_ANGSTROM2 == pytest.approx(0.18)
    assert runner.EXPOSURE_RADIAL_QUADRATURE_ORDER == 32
    assert runner.SOURCE_RADIAL_QUADRATURE_ORDER == 32
    assert runner.GREEN_RADIAL_QUADRATURE_ORDER == 32
    rotation = runner._rotation()
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=2.0e-15)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=2.0e-15)
    assert runner._rotation() == pytest.approx(rotation, abs=0.0)


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
