import sys
import types

import numpy as np
import pytest
import torch
from ase import Atoms

from maple.function.calculator._ase_unit_contract import ASE_STRESS_UNIT, EV2HARTREE
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
    assert MACEOfficialPBCCalculator.maple_stress_unit == ASE_STRESS_UNIT
    assert MACEOfficialPBCCalculator.supported_hessian_modes == ()


def test_mace_pbc_converts_energy_and_forces_but_not_stress():
    stress = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    forces = np.array([[2.0, 3.0, 4.0]])
    official = DummyMACECalculator(
        energy=5.0,
        forces=forces,
        stress=stress,
    )
    calc = MACEOfficialPBCCalculator(
        device=torch.device("cpu"),
        model="mace-mp-pbc-small",
        mace_mp_factory=lambda **kwargs: official,
    )

    calc.calculate(
        Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True),
        properties=["energy", "forces", "stress"],
    )

    assert calc.results["energy"] == pytest.approx(5.0 * EV2HARTREE)
    assert calc.results["forces"] == pytest.approx(forces * EV2HARTREE)
    assert calc.results["stress"] == pytest.approx(stress)


class DummyMACECalculator:
    def __init__(self, energy=1.0, forces=None, stress=None, **kwargs):
        self.kwargs = kwargs
        self.results = {"energy": energy}
        if forces is not None:
            self.results["forces"] = forces
        if stress is not None:
            self.results["stress"] = stress

    def calculate(self, atoms=None, properties=None, system_changes=None):
        return None


def _install_mace_stub(monkeypatch):
    package = types.ModuleType("mace")
    package.__path__ = []
    calculators = types.ModuleType("mace.calculators")
    calculators.mace_mp = lambda **kwargs: DummyMACECalculator(**kwargs)
    monkeypatch.setitem(sys.modules, "mace", package)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)
