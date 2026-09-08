"""Controlled PES setters must invalidate ASE's result cache."""
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import all_changes

from maple.function.calculator.calculator_base import CalcABC


def test_ani_d4_assignment_invalidates_energy_and_force_cache():
    torch = pytest.importorskip('torch')
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    calc = object.__new__(ANICalculator)
    CalcABC.__init__(calc)
    calc.device = torch.device('cpu')
    calc.dtype = torch.float64
    calc.implicit_solv_init('none', 'none')
    calc.d4 = False
    calc.model = lambda species, coordinates: ((coordinates**2).sum(),)
    calc.dftd4 = lambda species, coordinates: 0.25 * (coordinates**2).sum()
    atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
    atoms.calc = calc
    assert atoms.get_potential_energy() == 1.0
    np.testing.assert_allclose(atoms.get_forces()[1], [-2, 0, 0])
    calc.d4 = True
    assert atoms.get_potential_energy() == 1.25
    np.testing.assert_allclose(atoms.get_forces()[1], [-2.5, 0, 0])
    calc.d4 = False
    assert atoms.get_potential_energy() == 1.0


def test_aimnet_coulomb_setter_invalidates_results():
    torch = pytest.importorskip('torch')
    from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator

    class Probe(AIMNet2Calculator):
        def __init__(self):
            CalcABC.__init__(self)
            self.model = torch.nn.Module()
            self.model.lrcoulomb = torch.nn.Module()
            self.model.lrcoulomb.method = 'simple'
            self.model.lrcoulomb.dsf_rc = 15.0
            self.model.lrcoulomb.dsf_alpha = 0.2
            self._set_lrcoulomb_method('simple')
            self.calls = 0

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            atoms = CalcABC.calculate(self, atoms, properties, system_changes)
            assert atoms is not None
            self.calls += 1
            value = 1.0 if self.model.lrcoulomb.method == 'simple' else 2.0
            self.results = {'energy': value, 'forces': np.full((len(atoms), 3), value)}

    atoms = Atoms('H2')
    atoms.calc = Probe()
    assert atoms.get_potential_energy() == 1.0
    atoms.calc._set_lrcoulomb_method('dsf', cutoff=10.0)
    assert atoms.get_potential_energy() == 2.0
    assert atoms.calc.calls == 2
    np.testing.assert_array_equal(atoms.get_forces(), 2.0)
    atoms.calc._set_lrcoulomb_method('simple')
    assert atoms.get_potential_energy() == 1.0


@pytest.mark.parametrize('key,value', [('charge', 1), ('mult', 3)])
def test_electronic_metadata_changes_still_invalidate_cache(key, value):
    class StateModel(CalcABC):
        SUPPORTS_CHARGE_MULT = True

        def __init__(self):
            super().__init__()
            self.implemented_properties = ['energy']

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            atoms = super().calculate(atoms, properties, system_changes)
            assert atoms is not None
            self.results = {'energy': float(atoms.info.get('charge', 0) + atoms.info.get('mult', 1))}

    atoms = Atoms('H2')
    atoms.calc = StateModel()
    assert atoms.get_potential_energy() == 1.0
    atoms.info[key] = value
    expected = value + 1 if key == 'charge' else value
    assert atoms.get_potential_energy() == expected
