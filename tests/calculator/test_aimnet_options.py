import sys
import types

import numpy as np
import pytest
import torch
from ase import Atoms

from maple.function.calculator._ase_unit_contract import ASE_STRESS_UNIT, EV2HARTREE
from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator
from maple.function.calculator.aimnet._aimnet2_official_pbc_calculator import (
    AIMNET2_SHORT_RANGE_CUTOFF_A,
    AIMNet2OfficialPBCCalculator,
)
from maple.function.calculator.aimnet.options import validate_aimnet_options
from maple.function.calculator.set_calculator import SetClaculator


def _setter(tmp_path, model_options=None):
    return SetClaculator(
        device=torch.device("cpu"),
        model="aimnet2",
        output=str(tmp_path / "maple.out"),
        model_options=model_options,
    )


def test_known_aimnet_coulomb_option_passes_through(tmp_path):
    setter = _setter(tmp_path, {"coulomb": "ewald", "cutoff": 12.5})

    assert setter._validated_aimnet_options()["coulomb"] == "ewald"
    assert setter._validated_aimnet_options()["cutoff"] == 12.5


def test_unknown_aimnet_option_raises_helpful_error():
    with pytest.raises(ValueError, match="pme_cutof.*Supported options"):
        validate_aimnet_options({"pme_cutof": 9.0})


@pytest.mark.parametrize("model_options", [None, {}])
def test_empty_aimnet_options_are_noop(model_options):
    assert validate_aimnet_options(model_options) == {}


def test_aimnet_pbc_factory_dispatches_to_official_class(monkeypatch, tmp_path):
    _install_aimnet_stub(monkeypatch)

    for model in ("aimnet2-pbc", "aimnet2nse-pbc"):
        setter = SetClaculator(
            device=torch.device("cpu"),
            model=model,
            output=str(tmp_path / f"{model}.out"),
        )

        assert isinstance(setter.set_calculator(), AIMNet2OfficialPBCCalculator)


def test_aimnet_pbc_capability_declared_without_changing_legacy_class():
    assert AIMNet2OfficialPBCCalculator.maple_pbc_md_supported is True
    assert AIMNet2OfficialPBCCalculator.maple_stress_unit == ASE_STRESS_UNIT
    assert getattr(AIMNet2Calculator, "maple_pbc_md_supported", False) is False


def _build_pbc_calc(coulomb_method, cutoff, pme_cutoff=None):
    return AIMNet2OfficialPBCCalculator(
        device=torch.device("cpu"),
        coulomb_method=coulomb_method,
        cutoff=cutoff,
        pme_cutoff=pme_cutoff,
        base_calculator_cls=DummyAIMNetBase,
        ase_calculator_cls=lambda base: DummyOfficialASECalculator(base),
    )


@pytest.mark.parametrize(
    ("coulomb_method", "cutoff", "expected_neighbor_cutoff", "expected_short_range_cutoff"),
    [
        # DSF is a real cutoff-based method: the public cutoff participates in
        # the MIC bound, but the 5 Å AEV short-range descriptor is the floor.
        ("dsf", 15.0, 15.0, 15.0),
        ("dsf", 12.0, 12.0, 12.0),
        ("dsf", 3.0, AIMNET2_SHORT_RANGE_CUTOFF_A, 3.0),
        # Ewald / PME ignore the public ``cutoff`` argument at runtime, so the
        # only cutoff MAPLE can honestly gate against is the AEV short range.
        ("ewald", 15.0, AIMNET2_SHORT_RANGE_CUTOFF_A, None),
        ("pme", 15.0, AIMNET2_SHORT_RANGE_CUTOFF_A, None),
    ],
)
def test_aimnet_pbc_effective_cutoff_matches_coulomb_method(
    coulomb_method, cutoff, expected_neighbor_cutoff, expected_short_range_cutoff
):
    calc = _build_pbc_calc(coulomb_method, cutoff)
    assert calc.neighbor_cutoff_A == pytest.approx(expected_neighbor_cutoff)
    # The long-range method's public cutoff is preserved verbatim for the
    # manifest, separate from the effective local cutoff used at admission.
    assert calc.lrcoulomb_method == coulomb_method
    assert calc.lrcoulomb_cutoff_A == pytest.approx(cutoff)
    assert calc.long_range_coulomb_cutoff_A == pytest.approx(cutoff)
    assert calc.local_descriptor_cutoff_A == pytest.approx(AIMNET2_SHORT_RANGE_CUTOFF_A)
    if expected_short_range_cutoff is None:
        assert calc.short_range_realspace_cutoff_A is None
    else:
        assert calc.short_range_realspace_cutoff_A == pytest.approx(expected_short_range_cutoff)


def test_aimnet_pbc_rejects_unsupported_coulomb_method():
    with pytest.raises(ValueError, match="Unsupported AIMNet2 PBC Coulomb method"):
        _build_pbc_calc("simple", 15.0)


def test_aimnet_pbc_converts_energy_and_forces_but_not_stress():
    stress = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    forces = np.array([[2.0, 3.0, 4.0]])
    calc = AIMNet2OfficialPBCCalculator(
        device=torch.device("cpu"),
        base_calculator_cls=DummyAIMNetBase,
        ase_calculator_cls=lambda base: DummyOfficialASECalculator(
            base,
            {
                "energy": 5.0,
                "forces": forces,
                "stress": stress,
            },
        ),
    )

    calc.calculate(
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        properties=["energy", "forces", "stress"],
    )

    assert calc.results["energy"] == pytest.approx(5.0 * EV2HARTREE)
    assert calc.results["forces"] == pytest.approx(forces * EV2HARTREE)
    assert calc.results["stress"] == pytest.approx(stress)


class DummyAIMNetBase:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.lrcoulomb_call = None

    def set_lrcoulomb_method(self, method, **kwargs):
        self.lrcoulomb_call = (method, kwargs)


class DummyOfficialASECalculator:
    def __init__(self, base, results=None):
        self.base = base
        self.results = results or {"energy": 1.0}

    def calculate(self, atoms=None, properties=None, system_changes=None):
        return None


def _install_aimnet_stub(monkeypatch):
    package = types.ModuleType("aimnet")
    package.__path__ = []
    calculators = types.ModuleType("aimnet.calculators")
    calculators.AIMNet2Calculator = DummyAIMNetBase
    calculators.AIMNet2ASE = DummyOfficialASECalculator
    monkeypatch.setitem(sys.modules, "aimnet", package)
    monkeypatch.setitem(sys.modules, "aimnet.calculators", calculators)
