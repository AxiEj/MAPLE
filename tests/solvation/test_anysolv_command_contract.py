from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from maple.function.calculator.calculator_base import CalcABC
from maple.function.calculator.extra_correction import implicit
from maple.function.calculator.extra_correction.implicit.result import SolvationResult
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.read.command_control import CommandControl

SOLVENT = "#solv(implicit=water,method=anysolv,experimental=true,source_root=/official/anysolv)"


def test_anysolv_header_retains_explicit_vacuum_choice():
    control = CommandControl.from_settings(
        ["#model=aimnet2", "#sp(verbose=0)", SOLVENT]
    )
    assert control.get("model") == "aimnet2"
    assert control.get("solv")["source_root"] == "/official/anysolv"


@pytest.mark.parametrize(
    "path_setting",
    ["source_root=", "source_root=123", "model_path=", "model_path=123"],
)
def test_anysolv_header_rejects_empty_or_non_string_paths(path_setting):
    solvent = (
        "#solv(implicit=water,method=anysolv,experimental=true,"
        f"source_root=/official/anysolv,{path_setting})"
    )
    if path_setting.startswith("source_root="):
        solvent = solvent.replace("source_root=/official/anysolv,", "")
    with pytest.raises(ValueError, match="non-empty path string"):
        CommandControl.from_settings(["#model=aimnet2", "#sp", solvent])


@pytest.mark.parametrize("field", ["source_root", "model_path"])
def test_anysolv_options_cannot_leak_to_other_solvation_methods(field):
    with pytest.raises(ValueError, match="Unknown.*solvation|Unknown.*parameter"):
        CommandControl.from_settings(
            ["#model=aimnet2", "#sp", f"#solv(explicit=water,{field}=/unused)"]
        )


@pytest.mark.parametrize(
    "task", ["#opt", "#freq", "#scan", "#md", "#ts", "#irc", "#sp(verbose=1)"]
)
def test_anysolv_derivative_tasks_rejected_before_runtime(task):
    with pytest.raises(ValueError, match="scalar.*only"):
        CommandControl.from_settings(["#model=aimnet2", task, SOLVENT])


@pytest.mark.parametrize(
    "solvent, message",
    [
        (SOLVENT.replace(",experimental=true", ""), "experimental=true"),
        (SOLVENT.replace(",source_root=/official/anysolv", ""), "source_root"),
        (SOLVENT.replace("implicit=water", "implicit=none"), "implicit=<solvent>"),
        (SOLVENT[:-1] + ",explicit=water)", "does not accept"),
    ],
)
def test_anysolv_header_rejects_incomplete_or_double_solvation(solvent, message):
    with pytest.raises(ValueError, match=message):
        CommandControl.from_settings(["#model=aimnet2", "#sp", solvent])


def test_factory_uses_one_shared_correction_and_contains_eager_forces(
    tmp_path, monkeypatch
):
    calls = {"base": 0, "correction": 0}

    class Vacuum(CalcABC):
        __module__ = "maple.function.calculator.aimnet._aimnet2_calculator"
        MODEL_NAMES = ("aimnet2",)
        OPTION_KEYS = ()
        MODEL_ENERGY_UNIT = "hartree"
        SUPPORTS_CHARGE_MULT = True
        implemented_properties = ["energy", "free_energy", "forces"]

        def __init__(self, device, model, implicit, solvent):
            super().__init__()
            assert implicit == solvent == "none"

        def calculate(self, atoms=None, properties=None, system_changes=None):
            super().calculate(atoms, properties, system_changes)
            calls["base"] += 1
            self._finalize_results(atoms, energy=1.0, forces=np.ones((len(atoms), 3)))

    class Correction:
        supported_properties = {"energy"}

        def __init__(self, **kwargs):
            assert kwargs["solvent"] == "water"

        def evaluate(self, atoms, *, need_forces=False, calculator=None):
            assert not need_forces
            calls["correction"] += 1
            return SolvationResult(
                0.25, quantity_kind="geometry_level_energy_correction"
            )

    monkeypatch.setattr(implicit, "AnySolvBackend", Correction)
    atoms = Atoms(
        "H2", positions=[[0, 0, 0], [0, 0, 0.74]], info={"charge": 0, "mult": 1}
    )
    setter = SetCalculator(
        "cpu",
        "aimnet2",
        str(tmp_path / "run.out"),
        atoms=atoms,
        implicit="anysolv",
        solvent="water",
        task="sp",
        solvation_options={"experimental": True, "source_root": "/official/anysolv"},
    )
    monkeypatch.setattr(setter, "_discover_calculator_class", lambda _: Vacuum)
    atoms.calc = setter.set_calculator()
    assert atoms.get_potential_energy() == 1.25
    assert atoms.get_potential_energy() == 1.25
    assert calls == {"base": 1, "correction": 1}
    assert "forces" not in atoms.calc.results
    with pytest.raises(PropertyNotImplementedError, match="forces"):
        atoms.get_forces()


def test_factory_does_not_wrap_non_calcabc_uma(tmp_path, monkeypatch):
    class NonCalcABC:
        MODEL_NAMES = ("uma",)
        OPTION_KEYS = ()
        SUPPORTED_HESSIAN_MODES = ()

    setter = SetCalculator(
        "cpu",
        "uma",
        str(tmp_path / "run.out"),
        implicit="anysolv",
        solvent="water",
        task="sp",
        solvation_options={"experimental": True, "source_root": "/official/anysolv"},
    )
    monkeypatch.setattr(setter, "_discover_calculator_class", lambda _: NonCalcABC)
    with pytest.raises(ValueError, match="vacuum CalcABC"):
        setter.set_calculator()


@pytest.mark.parametrize(
    ("solvation_options", "message"),
    [
        ({"experimental": True, "source_root": ""}, "source_root"),
        ({"experimental": True, "source_root": 123}, "source_root"),
        ({"experimental": True, "source_root": None}, "source_root"),
        (
            {"experimental": True, "source_root": "/source", "model_path": None},
            "model_path",
        ),
        (
            {"experimental": True, "source_root": "/source", "model_path": ""},
            "model_path",
        ),
        (
            {"experimental": True, "source_root": "/source", "model_path": 123},
            "model_path",
        ),
    ],
)
def test_direct_factory_rejects_empty_or_non_string_paths(
    tmp_path, solvation_options, message
):
    setter = SetCalculator(
        "cpu",
        "aimnet2",
        str(tmp_path / "run.out"),
        implicit="anysolv",
        solvent="water",
        task="sp",
        solvation_options=solvation_options,
    )
    with pytest.raises(ValueError, match=message):
        setter.set_calculator()
