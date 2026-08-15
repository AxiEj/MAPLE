from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def test_vnext_numeric_modules_import_when_optional_model_runtimes_are_blocked():
    script = r"""
import sys

for name in ("torch", "mace", "aimnet2calc", "pyscf"):
    sys.modules[name] = None

import maple.function.calculator.extra_correction as optional_corrections
import maple.function.read as read_package
from maple.function.read.filereader.mol2_reader import MOL2Reader
import maple.solvation.continuum.fixed_topology_cpcm
import maple.solvation.continuum.atomic_l1_pyddx
import maple.solvation.coupling.exact_gto
import maple.solvation.coupling.gaussian_multipole_derivatives
import maple.solvation.coupling.geometry_mediated
import maple.solvation.models.aimnet2
import maple.solvation.release.geometry_mediated

assert "torch" not in optional_corrections.__dict__
assert "GBSA" not in optional_corrections.__dict__
assert "QEqTorch" not in optional_corrections.__dict__
assert "InputReader" not in read_package.__dict__
assert MOL2Reader.__name__ == "MOL2Reader"
"""
    result = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_legacy_gbsa_exports_remain_lazy_and_fail_only_when_requested():
    script = r"""
import sys

sys.modules["torch"] = None
import maple.function.calculator.extra_correction as optional_corrections

assert "GBSA" not in optional_corrections.__dict__
try:
    optional_corrections.GBSA
except ModuleNotFoundError as exc:
    assert "torch" in str(exc)
else:
    raise AssertionError("GBSA must require Torch when the symbol is requested")
"""
    result = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_input_orchestration_export_remains_lazy_and_fails_only_when_requested():
    script = r"""
import sys

sys.modules["torch"] = None
import maple.function.read as read_package

assert "InputReader" not in read_package.__dict__
try:
    read_package.InputReader
except ModuleNotFoundError as exc:
    assert "torch" in str(exc)
else:
    raise AssertionError("InputReader must require Torch when requested")
"""
    result = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
