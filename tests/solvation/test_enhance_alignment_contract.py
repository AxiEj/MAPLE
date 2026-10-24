"""Public framework regressions for the upstream-first Route 4 merge."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from maple.function.calculator.calculator_base import CalcABC
from maple.function.calculator.set_calculator import (
    SetCalculator,
    _BUILTIN_NAME_TO_MODULE,
)
from maple.function.read.command_control import CommandControl


class _VacuumCalculator(CalcABC):
    MODEL_NAMES = ("alignment-vacuum",)
    OPTION_KEYS = ()

    def __init__(self, *, device, model, implicit, solvent, execution_task=None):
        super().__init__()
        self.execution_task = execution_task


def test_engine_context_reaches_calculator(tmp_path, monkeypatch):
    setter = SetCalculator(
        "cpu",
        "alignment-vacuum",
        str(tmp_path / "run.out"),
        task="sp",
        charge_options={"source": "mol2"},
    )
    monkeypatch.setattr(
        setter, "_discover_calculator_class", lambda _: _VacuumCalculator
    )
    calc = setter.set_calculator()
    assert calc.execution_task == "sp"
    assert setter.charge_options == {"source": "mol2"}


def test_no_card_gpu_rejected_before_checkpoint_resolution(tmp_path, monkeypatch):
    setter = SetCalculator("cuda", "alignment-vacuum", str(tmp_path / "run.out"))
    monkeypatch.setattr(
        setter, "_discover_calculator_class", lambda _: _VacuumCalculator
    )

    def must_not_resolve(*args):
        pytest.fail("Accelerator rejection must precede checkpoint resolution")

    monkeypatch.setattr(setter, "_resolve_model_path", must_not_resolve)
    with pytest.raises(ValueError, match="no frozen no-loss"):
        setter.set_calculator()


def test_upstream_fennol_identity_is_preserved():
    for name in ("fennol", "fennix-bio1s", "fennix-bio1m"):
        assert _BUILTIN_NAME_TO_MODULE[name] == (
            "maple.function.calculator.fennol._fennol_calculator"
        )


def test_upstream_ion_ff_identifier_is_not_cast_to_integer():
    assert CommandControl._auto_cast("12_6", "ion_ff") == "12_6"


def test_public_cli_accepts_postmerge_engine_context(tmp_path):
    plugin = tmp_path / "alignment_plugin.py"
    plugin.write_text(
        "from maple.function.calculator.calculator_base import CalcABC, register_calculator\n"
        "@register_calculator\n"
        "class AlignmentCalculator(CalcABC):\n"
        "    MODEL_NAMES = ('alignment-test',)\n"
        "    MODEL_ENERGY_UNIT = 'hartree'\n"
        "    OPTION_KEYS = ()\n"
        "    implemented_properties = ['energy', 'free_energy']\n"
        "    def __init__(self, device, model, implicit, solvent, execution_task=None):\n"
        "        super().__init__()\n"
        "        assert execution_task == 'sp'\n"
        "    def calculate(self, atoms=None, properties=None, system_changes=None):\n"
        "        super().calculate(atoms, properties, system_changes)\n"
        "        self._finalize_results(atoms, energy=1.25)\n"
    )
    inp = tmp_path / "run.inp"
    inp.write_text(
        "#model=alignment-test(module=alignment_plugin)\n#device=cpu\n#sp\n\n"
        "0 1\nH 0 0 0\nH 0 0 0.74\n"
    )
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(tmp_path), str(root)))}
    completed = subprocess.run(
        [sys.executable, "-m", "maple.main", str(inp)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = inp.with_suffix(".out").read_text()
    assert completed.returncode == 0, completed.stderr + output
    assert "1.250000" in output
