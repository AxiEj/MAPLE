import sys
import types

import pytest
import torch

from maple.function.calculator.mace._mace_official_pbc_calculator import (
    MACEOfficialPBCCalculator,
)
from maple.function.calculator.mace.options import MACE_PBC_MODELS, validate_mace_pbc_options
from maple.function.calculator.set_calculator import SetClaculator


def test_mace_pbc_factory_dispatches_to_official_class(monkeypatch, tmp_path):
    _install_mace_stub(monkeypatch)

    for model in MACE_PBC_MODELS:
        setter = SetClaculator(
            device=torch.device("cpu"),
            model=model,
            output=str(tmp_path / f"{model}.out"),
        )

        assert isinstance(setter.set_calculator(), MACEOfficialPBCCalculator)


def test_unknown_mace_pbc_foundation_raises():
    with pytest.raises(ValueError, match="mace-mp-giant.*Supported foundations"):
        validate_mace_pbc_options({"foundation": "mace-mp-giant"})


def test_mace_pbc_capability_true_and_hessian_false():
    assert MACEOfficialPBCCalculator.maple_pbc_md_supported is True
    assert MACEOfficialPBCCalculator.supported_hessian_modes == ()


class DummyMACECalculator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.results = {"energy": 1.0}

    def calculate(self, atoms=None, properties=None, system_changes=None):
        return None


def _install_mace_stub(monkeypatch):
    package = types.ModuleType("mace")
    package.__path__ = []
    calculators = types.ModuleType("mace.calculators")
    calculators.mace_mp = lambda **kwargs: DummyMACECalculator(**kwargs)
    monkeypatch.setitem(sys.modules, "mace", package)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)
