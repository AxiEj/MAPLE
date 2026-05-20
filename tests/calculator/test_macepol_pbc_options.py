import sys
import types

import numpy as np
import pytest
import torch
from ase import Atoms

from maple.function.calculator._ase_unit_contract import ASE_STRESS_UNIT
from maple.function.calculator.mace._macepol_official_pbc_calculator import (
    MACEPolOfficialPBCCalculator,
)
from maple.function.calculator.mace._macepol_calculator import MACEPolCalculator
from maple.function.calculator.mace.options import (
    MACEPOL_PBC_MODELS,
    validate_macepol_pbc_options,
)
from maple.function.calculator.set_calculator import SetClaculator


def test_macepol_pbc_factory_dispatches_to_official_class(monkeypatch, tmp_path):
    _install_mace_polar_stub(monkeypatch)

    for model, upstream_model in MACEPOL_PBC_MODELS.items():
        setter = SetClaculator(
            device=torch.device("cpu"),
            model=model,
            output=str(tmp_path / f"{model}.out"),
        )

        calculator = setter.set_calculator()

        assert isinstance(calculator, MACEPolOfficialPBCCalculator)
        assert calculator._official_calculator.kwargs["model"] == upstream_model


def test_macepol_pbc_default_spin_is_multiplicity_one():
    official = DummyMACECalculator()
    calculator = MACEPolOfficialPBCCalculator(
        device=torch.device("cpu"),
        model="macepol-pbc-small",
        mace_polar_factory=lambda **kwargs: official,
    )
    atoms = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        cell=[14.0, 14.0, 14.0],
        pbc=True,
    )

    calculator.calculate(atoms, properties=["energy", "forces"])

    assert official.seen_atoms.info["spin"] == 1.0


def test_macepol_pbc_spin_charge_and_external_field_normalization():
    official = DummyMACECalculator()
    calculator = MACEPolOfficialPBCCalculator(
        device=torch.device("cpu"),
        model="macepol-pbc-small",
        mace_polar_factory=lambda **kwargs: official,
    )
    atoms = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        cell=[14.0, 14.0, 14.0],
        pbc=True,
    )
    atoms.info["mult"] = 3
    atoms.info["charge"] = -1

    calculator.calculate(atoms, properties=["energy", "forces"])

    assert official.seen_atoms is not atoms
    assert official.seen_atoms.info["spin"] == 3.0
    assert official.seen_atoms.info["charge"] == -1.0
    np.testing.assert_allclose(official.seen_atoms.info["external_field"], np.zeros(3))
    assert atoms.info.get("spin") is None
    assert "external_field" not in atoms.info


def test_macepol_pbc_accepts_uniform_legacy_per_atom_external_field():
    official = DummyMACECalculator()
    calculator = MACEPolOfficialPBCCalculator(
        device=torch.device("cpu"),
        model="macepol-pbc-small",
        mace_polar_factory=lambda **kwargs: official,
    )
    atoms = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        cell=[14.0, 14.0, 14.0],
        pbc=True,
    )
    atoms.info["external_field"] = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]])

    calculator.calculate(atoms, properties=["energy", "forces"])

    np.testing.assert_allclose(official.seen_atoms.info["external_field"], [0.0, 0.0, 0.1])


def test_macepol_pbc_rejects_nonuniform_per_atom_external_field():
    calculator = MACEPolOfficialPBCCalculator(
        device=torch.device("cpu"),
        model="macepol-pbc-small",
        mace_polar_factory=lambda **kwargs: DummyMACECalculator(),
    )
    atoms = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        cell=[14.0, 14.0, 14.0],
        pbc=True,
    )
    atoms.info["external_field"] = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])

    with pytest.raises(ValueError, match="external_field"):
        calculator.calculate(atoms, properties=["energy", "forces"])


def test_macepol_pbc_reader_spin_does_not_override_multiplicity():
    official = DummyMACECalculator()
    calculator = MACEPolOfficialPBCCalculator(
        device=torch.device("cpu"),
        model="macepol-pbc-small",
        mace_polar_factory=lambda **kwargs: official,
    )
    atoms = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        cell=[14.0, 14.0, 14.0],
        pbc=True,
    )
    atoms.info["mult"] = 3
    atoms.info["spin"] = 1.0

    calculator.calculate(atoms, properties=["energy", "forces"])

    assert official.seen_atoms.info["spin"] == 3.0


def test_macepol_pbc_adapter_specific_spin_override_takes_precedence():
    official = DummyMACECalculator()
    calculator = MACEPolOfficialPBCCalculator(
        device=torch.device("cpu"),
        model="macepol-pbc-small",
        mace_polar_factory=lambda **kwargs: official,
    )
    atoms = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        cell=[14.0, 14.0, 14.0],
        pbc=True,
    )
    atoms.info["mult"] = 3
    atoms.info["spin"] = 1.0
    atoms.info["macepol_spin"] = 0.5

    calculator.calculate(atoms, properties=["energy", "forces"])

    assert official.seen_atoms.info["spin"] == 0.5
    assert "macepol_spin" not in official.seen_atoms.info


def test_unknown_macepol_pbc_dtype_raises():
    with pytest.raises(ValueError, match="bfloat16.*Supported values"):
        validate_macepol_pbc_options({"default_dtype": "bfloat16"})


def test_macepol_pbc_capability_true_and_hessian_false():
    assert MACEPolOfficialPBCCalculator.maple_pbc_md_supported is True
    assert MACEPolOfficialPBCCalculator.maple_stress_supported is True
    assert MACEPolOfficialPBCCalculator.maple_stress_unit == ASE_STRESS_UNIT
    assert MACEPolOfficialPBCCalculator.supported_hessian_modes == ()


def test_macepol_pbc_returns_finite_voigt_stress():
    official = DummyMACECalculator()
    calculator = MACEPolOfficialPBCCalculator(
        device=torch.device("cpu"),
        model="macepol-pbc-small",
        mace_polar_factory=lambda **kwargs: official,
    )
    atoms = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        cell=[14.0, 14.0, 14.0],
        pbc=True,
    )

    calculator.calculate(atoms, properties=["energy", "forces", "stress"])

    assert calculator.results["stress"].shape == (6,)
    assert np.all(np.isfinite(calculator.results["stress"]))


def test_legacy_macepol_keeps_traced_model_spin_as_mult_minus_one():
    calc = object.__new__(MACEPolCalculator)
    calc.device = torch.device("cpu")
    calc.dtype = torch.float32
    calc.r_max = 5.0
    calc.atomic_numbers = [1, 8]
    atoms = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
    )
    atoms.info["charge"] = -1
    atoms.info["mult"] = 3

    inputs = calc._build_inputs(atoms)

    total_charge = inputs[8]
    total_spin = inputs[9]
    assert total_charge.item() == pytest.approx(-1.0)
    assert total_spin.item() == pytest.approx(2.0)


class DummyMACECalculator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.models = [types.SimpleNamespace(r_max=torch.tensor(3.0))]
        self.results = {
            "energy": 1.0,
            "forces": np.zeros((2, 3)),
            "stress": np.zeros(6),
        }
        self.seen_atoms = None
        self.seen_properties = None

    def calculate(self, atoms=None, properties=None, system_changes=None):
        self.seen_atoms = atoms
        self.seen_properties = properties
        return None


def _install_mace_polar_stub(monkeypatch):
    package = types.ModuleType("mace")
    package.__path__ = []
    calculators = types.ModuleType("mace.calculators")
    calculators.mace_polar = lambda **kwargs: DummyMACECalculator(**kwargs)
    monkeypatch.setitem(sys.modules, "mace", package)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)
