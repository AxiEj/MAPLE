import importlib

import numpy as np
import pytest
import torch
from ase import Atoms

from maple.function.calculator.mace._mace_official_pbc_calculator import (
    MACEOfficialPBCCalculator,
    _load_mace_mp,
)
from maple.function.calculator.set_calculator import (
    MODEL_HESSIAN_SUPPORT,
    SetClaculator,
    model_supports_pbc_md,
    model_supports_stress,
)
from maple.function.dispatcher.frequency.frequency import MWFrequency
from maple.function.read.command_control import CommandControl


def test_command_control_preserves_explicit_mace_pbc_model_name(tmp_path):
    output = tmp_path / "parse.out"

    control = CommandControl.from_settings(
        [
            "#model=mace-mh-pbc(default_dtype=FLOAT64,dispersion=true,head=OMAT_PBE,foundation=mh-1)",
            "#pbc(8,9,10)",
        ],
        output_path=str(output),
    )

    assert control.params["model"] == "mace-mh-pbc"
    assert control.params["model_options"] == {
        "default_dtype": "float64",
        "dispersion": True,
        "head": "omat_pbe",
        "foundation": "mh-1",
    }
    assert control.params["pbc"] == [8.0, 9.0, 10.0, 90.0, 90.0, 90.0]


def test_command_control_rejects_unknown_mace_pbc_option(tmp_path):
    output = tmp_path / "parse.out"

    with pytest.raises(ValueError, match="Unsupported MACE PBC option"):
        CommandControl.from_settings(
            [
                "#model=mace-mp-pbc(coulomb=dsf)",
            ],
            output_path=str(output),
        )


def test_command_control_rejects_invalid_mace_pbc_dtype(tmp_path):
    output = tmp_path / "parse.out"

    with pytest.raises(ValueError, match="Unsupported MACE PBC default_dtype"):
        CommandControl.from_settings(
            [
                "#model=mace-mp-pbc(default_dtype=float16)",
            ],
            output_path=str(output),
        )


def test_mace_pbc_capabilities_are_explicit_model_contracts():
    assert not model_supports_pbc_md("maceoff23m")
    assert not model_supports_stress("maceoff23m")
    assert model_supports_pbc_md("mace-mp-pbc")
    assert model_supports_stress("mace-mp-pbc")
    assert MODEL_HESSIAN_SUPPORT["mace-mp-pbc"] == ()


def test_set_calculator_passes_mace_pbc_options(monkeypatch, tmp_path):
    captured = {}

    class FakeMACEOfficialPBCCalculator:
        supported_hessian_modes = ()

        def __init__(self, **kwargs):
            captured.update(kwargs)

    from maple.function.calculator.mace import _mace_official_pbc_calculator as mace_module

    monkeypatch.setattr(mace_module, "MACEOfficialPBCCalculator", FakeMACEOfficialPBCCalculator)

    calculator = SetClaculator(
        torch.device("cpu"),
        "mace-mh-pbc",
        str(tmp_path / "factory.out"),
        model_options={
            "foundation": "mh-1",
            "default_dtype": "float64",
            "dispersion": True,
            "head": "omat_pbe",
        },
    ).set_calculator()

    assert calculator is not None
    assert captured["model"] == "mace-mh-pbc"
    assert captured["foundation"] == "mh-1"
    assert captured["default_dtype"] == "float64"
    assert captured["dispersion"] is True
    assert captured["head"] == "omat_pbe"
    assert calculator.maple_pbc_md_supported is True
    assert calculator.maple_stress_supported is True


def test_set_calculator_rejects_hessian_for_mace_pbc(tmp_path):
    with pytest.raises(ValueError, match="does not support Hessian modes"):
        SetClaculator(
            torch.device("cpu"),
            "mace-mp-pbc",
            str(tmp_path / "factory.out"),
            model_options={"hessian": "numerical"},
        ).set_calculator()


def test_mace_pbc_missing_dependency_error(monkeypatch):
    original_import_module = importlib.import_module

    def fake_import_module(name):
        if name == "mace.calculators":
            raise ImportError("missing mace")
        return original_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError) as excinfo:
        _load_mace_mp("mace-mp-pbc")
    assert 'pip install -e ".[pbc-mace]"' in str(excinfo.value)


def test_mace_pbc_missing_mace_mp_attribute_error(monkeypatch):
    class FakeCalculators:
        pass

    original_import_module = importlib.import_module

    def fake_import_module(name):
        if name == "mace.calculators":
            return FakeCalculators()
        return original_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError) as excinfo:
        _load_mace_mp("mace-mp-pbc")
    assert 'pip install -e ".[pbc-mace]"' in str(excinfo.value)


def test_mace_pbc_adapter_converts_energy_forces_but_not_stress():
    ev_per_hartree = 27.211386245988
    captured = {}

    class FakeOfficialCalculator:
        def __init__(self):
            self.results = {}

        def calculate(self, atoms=None, properties=None, system_changes=None):
            captured["properties"] = list(properties)
            self.results = {
                "energy": ev_per_hartree,
                "forces": np.ones((len(atoms), 3)) * ev_per_hartree,
                "free_energy": 2.0 * ev_per_hartree,
                "stress": np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
            }

    def fake_mace_mp(**kwargs):
        captured["factory_kwargs"] = kwargs
        return FakeOfficialCalculator()

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[8.0, 8.0, 8.0], pbc=True)
    calculator = MACEOfficialPBCCalculator(
        torch.device("cpu"),
        "mace-mh-pbc",
        default_dtype="float64",
        dispersion=True,
        mace_mp_factory=fake_mace_mp,
    )

    calculator.calculate(atoms, properties=["energy", "forces", "stress"])

    assert captured["factory_kwargs"] == {
        "model": "mh-1",
        "device": "cpu",
        "default_dtype": "float64",
        "dispersion": True,
        "head": "omat_pbe",
    }
    assert captured["properties"] == ["energy", "forces", "stress"]
    assert calculator.results["energy"] == pytest.approx(1.0)
    assert calculator.results["free_energy"] == pytest.approx(2.0)
    assert np.allclose(calculator.results["forces"], np.ones((1, 3)))
    assert np.allclose(calculator.results["stress"], [1.0, 2.0, 3.0, 0.1, 0.2, 0.3])


def test_frequency_fails_fast_for_mace_pbc_when_hessian_modes_empty(tmp_path):
    class NoHessianCalculator:
        supported_hessian_modes = ()
        maple_model_name = "mace-mp-pbc"

        def get_hessian(self, atoms):
            raise AssertionError("get_hessian should not be called")

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = NoHessianCalculator()
    job = MWFrequency(output=str(tmp_path / "freq.out"), atoms=atoms)

    with pytest.raises(RuntimeError, match="does not support Hessian/frequency workflows yet"):
        job.get_hessian()


def test_mace_pbc_cutoff_minimum_image_violation_is_rejected(tmp_path):
    """MACE backend with extracted cutoff > L/2 must be rejected."""
    import maple.function.calculator.set_calculator as sc

    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)

    class FakeBackend:
        supported_hessian_modes = ()
        maple_neighbor_cutoff = 6.0  # >= 10/2

    orig = sc.SetClaculator._build_calculator

    def fake_build(self):
        return FakeBackend()

    sc.SetClaculator._build_calculator = fake_build
    try:
        with pytest.raises(ValueError, match="Minimum-image"):
            sc.SetClaculator(
                torch.device("cpu"), "mace-mp-pbc", str(tmp_path / "out"), atoms=atoms
            ).set_calculator()
    finally:
        sc.SetClaculator._build_calculator = orig


def test_mace_pbc_cutoff_none_does_not_raise(tmp_path):
    """When maple_neighbor_cutoff is None (introspection failed), gate is silently skipped."""
    import maple.function.calculator.set_calculator as sc

    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)

    class FakeBackend:
        supported_hessian_modes = ()
        maple_neighbor_cutoff = None

    orig = sc.SetClaculator._build_calculator

    def fake_build(self):
        return FakeBackend()

    sc.SetClaculator._build_calculator = fake_build
    try:
        sc.SetClaculator(
            torch.device("cpu"), "mace-mp-pbc", str(tmp_path / "out"), atoms=atoms
        ).set_calculator()
    finally:
        sc.SetClaculator._build_calculator = orig
