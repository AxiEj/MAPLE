import importlib

import numpy as np
import torch
import pytest
from ase import Atoms

from maple.function.calculator.aimnet._aimnet2_official_pbc_calculator import (
    AIMNet2OfficialPBCCalculator,
    _load_aimnet2_classes,
)
from maple.function.calculator.set_calculator import (
    MODEL_HESSIAN_SUPPORT,
    SetClaculator,
    model_supports_pbc_md,
    model_supports_stress,
)
from maple.function.dispatcher.frequency.frequency import MWFrequency
from maple.function.read.command_control import CommandControl


def test_command_control_accepts_aimnet_coulomb_options(tmp_path):
    output = tmp_path / "parse.out"

    control = CommandControl.from_settings(
        [
            "#model=aimnet2(coulomb=DSF,cutoff=12.5,dsf_alpha=0.3)",
        ],
        output_path=str(output),
    )

    assert control.params["model"] == "aimnet2"
    assert control.params["model_options"] == {
        "coulomb": "dsf",
        "cutoff": 12.5,
        "dsf_alpha": 0.3,
    }


def test_command_control_rejects_unknown_aimnet_option(tmp_path):
    output = tmp_path / "parse.out"

    with pytest.raises(ValueError, match="Unsupported AIMNet2 option"):
        CommandControl.from_settings(
            [
                "#model=aimnet2(pme_cutoff=12)",
            ],
            output_path=str(output),
        )


def test_command_control_rejects_unknown_aimnet_coulomb_method(tmp_path):
    output = tmp_path / "parse.out"

    with pytest.raises(ValueError, match="Unsupported AIMNet2 Coulomb method"):
        CommandControl.from_settings(
            [
                "#model=aimnet2(coulomb=pme)",
            ],
            output_path=str(output),
        )


def test_command_control_preserves_explicit_aimnet_pbc_model_name(tmp_path):
    output = tmp_path / "parse.out"

    control = CommandControl.from_settings(
        [
            "#model=aimnet2-pbc(coulomb=pme,ewald_accuracy=1e-6)",
            "#pbc(8,8,8)",
        ],
        output_path=str(output),
    )

    assert control.params["model"] == "aimnet2-pbc"
    assert control.params["model_options"] == {
        "coulomb": "pme",
        "ewald_accuracy": 1e-6,
    }
    assert control.params["pbc"] == [8.0, 8.0, 8.0, 90.0, 90.0, 90.0]


def test_command_control_rejects_unknown_aimnet_pbc_option(tmp_path):
    output = tmp_path / "parse.out"

    with pytest.raises(ValueError, match="Unsupported AIMNet2 PBC option"):
        CommandControl.from_settings(
            [
                "#model=aimnet2-pbc(pme_cutoff=12)",
            ],
            output_path=str(output),
        )


def test_set_calculator_rejects_unknown_aimnet_option(tmp_path):
    with pytest.raises(ValueError, match="Unsupported AIMNet2 option"):
        SetClaculator(
            torch.device("cpu"),
            "aimnet2",
            str(tmp_path / "factory.out"),
            model_options={"pme_cutoff": 12},
        ).set_calculator()


def test_aimnet_pbc_capabilities_are_explicit_model_contracts():
    assert not model_supports_pbc_md("aimnet2")
    assert not model_supports_stress("aimnet2")
    assert model_supports_pbc_md("aimnet2-pbc")
    assert model_supports_stress("aimnet2-pbc")
    assert MODEL_HESSIAN_SUPPORT["aimnet2-pbc"] == ()


def test_set_calculator_passes_aimnet_options(monkeypatch, tmp_path):
    captured = {}

    class FakeAIMNet2Calculator:
        supported_hessian_modes = ("analytic", "numerical")

        def __init__(self, **kwargs):
            captured.update(kwargs)

    from maple.function.calculator.aimnet import _aimnet2_calculator as aimnet_module

    monkeypatch.setattr(aimnet_module, "AIMNet2Calculator", FakeAIMNet2Calculator)
    monkeypatch.setattr(SetClaculator, "_ensure_model_file", lambda self, model_name: tmp_path / "aimnet2.pt")

    calculator = SetClaculator(
        torch.device("cpu"),
        "aimnet2",
        str(tmp_path / "factory.out"),
        model_options={"coulomb": "dsf", "cutoff": 12.5, "dsf_alpha": 0.3},
    ).set_calculator()

    assert calculator is not None
    assert captured["model"] == "aimnet2"
    assert captured["coulomb_method"] == "dsf"
    assert captured["cutoff"] == 12.5
    assert captured["dsf_alpha"] == 0.3


def test_set_calculator_rejects_hessian_for_aimnet_pbc(tmp_path):
    with pytest.raises(ValueError, match="does not support Hessian modes"):
        SetClaculator(
            torch.device("cpu"),
            "aimnet2-pbc",
            str(tmp_path / "factory.out"),
            model_options={"hessian": "numerical"},
        ).set_calculator()


def test_aimnet_pbc_missing_dependency_error(monkeypatch):
    def fake_import_module(name):
        if name == "aimnet.calculators":
            raise ImportError("missing aimnet")
        return importlib.import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError) as excinfo:
        _load_aimnet2_classes("aimnet2-pbc")
    assert 'pip install -e ".[pbc-aimnet]"' in str(excinfo.value)


def test_aimnet_pbc_adapter_converts_energy_forces_but_not_stress():
    ev_per_hartree = 27.211386245988
    captured = {}

    class FakeBaseCalculator:
        def __init__(self, model, device=None):
            captured["model"] = model
            captured["device"] = device

        def set_lrcoulomb_method(self, method, cutoff=15.0, dsf_alpha=0.2, ewald_accuracy=1e-6):
            captured["coulomb"] = {
                "method": method,
                "cutoff": cutoff,
                "dsf_alpha": dsf_alpha,
                "ewald_accuracy": ewald_accuracy,
            }

    class FakeASECalculator:
        def __init__(self, base_calc):
            self.base_calc = base_calc
            self.results = {}

        def calculate(self, atoms=None, properties=None, system_changes=None):
            self.results = {
                "energy": ev_per_hartree,
                "forces": np.ones((len(atoms), 3)) * ev_per_hartree,
                "free_energy": 2.0 * ev_per_hartree,
                "stress": np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
            }

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[8.0, 8.0, 8.0], pbc=True)
    calculator = AIMNet2OfficialPBCCalculator(
        torch.device("cpu"),
        "aimnet2nse-pbc",
        coulomb_method="pme",
        cutoff=13.0,
        dsf_alpha=0.25,
        ewald_accuracy=1e-5,
        base_calculator_cls=FakeBaseCalculator,
        ase_calculator_cls=FakeASECalculator,
    )

    calculator.calculate(atoms, properties=["energy", "forces", "stress"])

    assert captured["model"] == "aimnet2-nse"
    assert captured["coulomb"] == {
        "method": "pme",
        "cutoff": 13.0,
        "dsf_alpha": 0.25,
        "ewald_accuracy": 1e-5,
    }
    assert calculator.results["energy"] == pytest.approx(1.0)
    assert calculator.results["free_energy"] == pytest.approx(2.0)
    assert np.allclose(calculator.results["forces"], np.ones((1, 3)))
    assert np.allclose(calculator.results["stress"], [1.0, 2.0, 3.0, 0.1, 0.2, 0.3])


def test_frequency_fails_fast_when_hessian_modes_empty(tmp_path):
    class NoHessianCalculator:
        supported_hessian_modes = ()
        maple_model_name = "aimnet2-pbc"

        def get_hessian(self, atoms):
            raise AssertionError("get_hessian should not be called")

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = NoHessianCalculator()
    job = MWFrequency(output=str(tmp_path / "freq.out"), atoms=atoms)

    with pytest.raises(RuntimeError, match="does not support Hessian/frequency workflows yet"):
        job.get_hessian()
