from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools" / "route2_release" / "run_mace_realspace_so3_nogo.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("route2_mace_so3_nogo", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_is_checkpoint_free_and_capabilities_remain_closed():
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
    assert '"tier_v_admitted": False' in text
    assert '"public_force_admitted": False' in text
    assert '"eight_channel_field_transform_exonerated_by_zero_field": True' in text
    assert '"route2_continuum_exonerated_by_model_only_counterexample": True' in text


def test_rotation_and_numerical_zero_contract_are_fixed():
    runner = _load_runner()
    rotation = runner._rotation()
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=2.0e-15)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=2.0e-15)
    np.testing.assert_array_equal(rotation, runner._rotation())
    assert runner.ROTATION_SEED == 20260815
    assert runner.ROUNDOFF_MULTIPLIER == 4096.0
    assert runner.SCHEMA_VERSION.endswith("-v1")

    near = runner._scalar_rotation_record(2000.0, 2000.0 + 1.0e-10)
    assert near["numerically_equal"] is True
    material = runner._scalar_rotation_record(2000.0, 2000.0 + 1.0e-5)
    assert material["numerically_equal"] is False


def test_covariance_record_does_not_hide_material_vector_defect():
    runner = _load_runner()
    exact = runner._covariance_record(np.eye(3), np.eye(3))
    assert exact["numerically_covariant"] is True

    rotated_wrong = np.eye(3).copy()
    rotated_wrong[0, 1] = 1.0e-6
    defect = runner._covariance_record(rotated_wrong, np.eye(3))
    assert defect["numerically_covariant"] is False
    assert defect["maximum_absolute"] == pytest.approx(1.0e-6)
