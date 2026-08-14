from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np

from maple.solvation.release import load_pes_panel

ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    ROOT
    / "tools"
    / "route2_release"
    / "run_operational_analytic_harmonic_rigid_panel.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "route2_operational_analytic_harmonic_rigid_runner", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(RUNNER.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(RUNNER.parent))
    return module


def test_help_is_checkpoint_and_torch_runtime_free():
    script = f"""
import builtins
import runpy
import sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] == 'torch':
        raise AssertionError('runner imported torch before parsing --help')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
sys.path.insert(0, {str(RUNNER.parent)!r})
sys.argv = [{str(RUNNER)!r}, '--help']
runpy.run_path({str(RUNNER)!r}, run_name='__main__')
"""
    result = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--molecule-start" in result.stdout
    assert "--molecule-stop" in result.stdout
    assert "cannot admit Route-2 E/F/H/V/M" in " ".join(result.stdout.split())


def test_panel_contract_reuses_the_frozen_twenty_molecule_asset():
    runner = _load_runner()
    molecules = load_pes_panel()
    assert len(molecules) == runner.PES_PANEL_MOLECULE_COUNT == 20
    assert [item.molecule_id for item in molecules[:6]] == [
        "water",
        "methanol",
        "ethanol",
        "acetone",
        "acetonitrile",
        "benzene",
    ]
    assert runner.PRIMAL_TOLERANCE == 1.0e-12
    assert runner.FIELD_COVARIANCE_RELATIVE_TOLERANCE == 1.0e-4
    assert runner.SCALAR_IDENTITY_ABSOLUTE_TOLERANCE_EV == 1.0e-10
    assert runner.MISSING_RADIAL_BLOCK_ABSOLUTE_TOLERANCE == 1.0e-8
    assert runner.NO_CAPABILITIES == {
        "E": False,
        "F": False,
        "H": False,
        "V": False,
        "M": False,
    }


def test_runner_is_original_source_operational_only_and_times_are_not_measured():
    text = RUNNER.read_text(encoding="utf-8")
    assert "MACEPolarVariationalFieldEnergy" not in text
    assert "build_system_with_model" in text
    assert '"legacy_laboratory_grid_rehabilitated": False' in text
    assert '"public_energy_admitted": False' in text
    assert '"public_force_admitted": False' in text
    assert '"tier_v_admitted": False' in text
    measurement_start = text.index("measurements = {")
    measurement_end = text.index("repository.assert_unchanged()", measurement_start)
    measurement_block = text[measurement_start:measurement_end]
    assert "runtime_seconds" not in measurement_block
    assert "time.perf_counter" not in measurement_block


def test_relative_error_is_rotation_scale_independent():
    runner = _load_runner()
    values = np.asarray([[1.0, -2.0], [0.5, 3.0]])
    assert runner._relative(values, values) == 0.0
    assert runner._relative(7.0 * values, 7.0 * (values + 1.0e-8)) < 1.0e-8
