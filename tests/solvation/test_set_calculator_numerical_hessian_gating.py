from __future__ import annotations

import pytest

from maple.function.calculator.calculator_base import CalcABC
from maple.function.calculator.model_capabilities import (
    ModelCapabilities,
    ModelProvenanceCard,
)
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.engine import engine
from maple.function.read.filereader.mol2_reader import MOL2Reader


def _atoms_with_mol2(water_mol2):
    return MOL2Reader(str(water_mol2), charge=0, mult=1)


def _build_fake_setter(
    tmp_path, atoms, model_options, solvation_options, charge_options
):
    output = tmp_path / "route4.out"
    return SetCalculator(
        device="cpu",
        model="route4-dummy-model",
        output=str(output),
        atoms=atoms,
        d4=False,
        implicit="gb",
        solvent="water",
        model_options=model_options,
        solvation_options=solvation_options,
        charge_options=charge_options,
    )


class DummyCalculator(CalcABC):
    MODEL_NAMES = ("route4-dummy-model",)
    SUPPORTED_HESSIAN_MODES = ("numerical",)
    OPTION_KEYS = ()

    @classmethod
    def build_kwargs_from_options(
        cls, model, model_options, *, resolved_model_path=None
    ):
        return {}

    def __init__(self, device, model, implicit, solvent, **kwargs):
        super().__init__()
        self.device = device
        self.model = model
        self.implicit = implicit
        self.solvent = solvent
        self.hessian = None


class FakeCorrection:
    supported_properties = {"energy", "forces"}
    conservative_forces = True
    energy_reference = "relative"

    def __init__(self, *args, **kwargs):
        self._solvent_inputs = dict(kwargs)


def _approved_model_card():
    return ModelProvenanceCard(
        model_id="route4-dummy-model",
        version="test",
        capabilities=ModelCapabilities(
            energy=True,
            forces=True,
            conservative_forces=True,
            hessian="finite_difference",
        ),
        payload={},
    )


def _forbidden_model_card():
    return ModelProvenanceCard(
        model_id="route4-dummy-model",
        version="test",
        capabilities=ModelCapabilities(
            energy=True,
            forces=True,
            conservative_forces=True,
            hessian="none",
        ),
        payload={},
    )


def test_set_calculator_accepts_numerical_gb_only_when_capability_card_allows(
    tmp_path,
    water_mol2,
    monkeypatch,
):
    atoms = _atoms_with_mol2(water_mol2)
    setter = _build_fake_setter(
        tmp_path,
        atoms=atoms,
        model_options={"hessian": "numerical"},
        solvation_options={"experimental": True, "method": "gb"},
        charge_options={"mode": "fixed", "source": "dummy"},
    )

    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit.ImplicitSolvationCorrection",
        FakeCorrection,
    )
    monkeypatch.setattr(
        "maple.function.calculator.set_calculator.SetCalculator._discover_calculator_class",
        lambda self, name: DummyCalculator,
    )
    monkeypatch.setattr(
        "maple.function.calculator.set_calculator.load_model_provenance_card",
        lambda model_name, root: _approved_model_card(),
    )

    calc = setter.set_calculator()
    assert calc.frequency_type == "effective_solution_pmf"


def test_set_calculator_rejects_numerical_gb_without_conservative_support(
    tmp_path,
    water_mol2,
    monkeypatch,
):
    atoms = _atoms_with_mol2(water_mol2)
    setter = _build_fake_setter(
        tmp_path,
        atoms=atoms,
        model_options={"hessian": "numerical"},
        solvation_options={"experimental": True, "method": "gb"},
        charge_options={"mode": "fixed", "source": "dummy"},
    )

    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit.ImplicitSolvationCorrection",
        FakeCorrection,
    )
    monkeypatch.setattr(
        "maple.function.calculator.set_calculator.SetCalculator._discover_calculator_class",
        lambda self, name: DummyCalculator,
    )
    monkeypatch.setattr(
        "maple.function.calculator.set_calculator.load_model_provenance_card",
        lambda model_name, root: _forbidden_model_card(),
    )

    with pytest.raises(ValueError, match="Frequency analysis requires"):
        setter.set_calculator()


def test_set_calculator_rejects_gb_analytic_hessian(tmp_path, water_mol2, monkeypatch):
    atoms = _atoms_with_mol2(water_mol2)
    setter = _build_fake_setter(
        tmp_path,
        atoms=atoms,
        model_options={"hessian": "analytic"},
        solvation_options={"experimental": True, "method": "gb"},
        charge_options={"mode": "fixed", "source": "dummy"},
    )

    monkeypatch.setattr(
        "maple.function.calculator.set_calculator.SetCalculator._discover_calculator_class",
        lambda self, name: DummyCalculator,
    )
    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit.ImplicitSolvationCorrection",
        FakeCorrection,
    )
    monkeypatch.setattr(
        "maple.function.calculator.set_calculator.load_model_provenance_card",
        lambda model_name, root: _approved_model_card(),
    )

    with pytest.raises(ValueError, match="numerical Hessian only"):
        setter.set_calculator()


def test_set_calculator_rejects_forbidden_task_before_model_loading(tmp_path):
    setter = SetCalculator(
        device="cpu",
        model="aimnet2-cpcms-v2",
        output=str(tmp_path / "route4.out"),
        task="md",
    )

    with pytest.raises(ValueError, match="forbids task 'md'"):
        setter.set_calculator()


def test_engine_passes_command_task_to_set_calculator(tmp_path, monkeypatch):
    captured = {}

    class _CommandControl(dict):
        task = "freq"

    class _FakeSetter:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def set_calculator(self):
            return object()

    monkeypatch.setattr(
        "maple.function.calculator.SetClaculator",
        _FakeSetter,
    )
    runner = engine()
    runner.output = str(tmp_path / "route4.out")
    runner.atoms = None
    runner.commandcontrol = _CommandControl()
    runner.model_options = {}
    runner.d4 = False
    runner._mlp_initiator("aimnet2-cpcms-v2", "cpu")

    assert captured["task"] == "freq"
