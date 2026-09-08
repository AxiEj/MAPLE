"""Requested electronic state must be supported by the actual energy model."""
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import all_changes

from maple.function.calculator.calculator_base import CalcABC
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.dispatcher.frequency.frequency import MWFrequency


class NeutralModel(CalcABC):
    MODEL_ENERGY_UNIT = 'hartree'

    def __init__(self):
        super().__init__()
        self.implemented_properties = ['energy', 'forces']
        self.calls = 0
        self.solvent_correction = None

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        atoms = super().calculate(atoms, properties, system_changes)
        assert atoms is not None
        self.calls += 1
        self.results = {'energy': 0.0, 'forces': np.zeros((len(atoms), 3))}


@pytest.mark.parametrize('charge,mult', [(1, 1), (0, 3)])
def test_factory_rejects_unsupported_state_before_model_construction(tmp_path, charge, mult):
    atoms = Atoms('H2', positions=[[0, 0, 0], [0.8, 0, 0]])
    atoms.info.update(charge=charge, mult=mult)
    factory = SetCalculator('cpu', 'neutral', str(tmp_path / 'out'), atoms=atoms)
    with pytest.raises(ValueError, match='electronic state|charge/multiplicity'):
        factory._validate_against_class(NeutralModel)
    assert atoms.info == {'charge': charge, 'mult': mult}


@pytest.mark.parametrize('charge,mult', [(1, 1), (0, 3)])
def test_direct_backend_rejects_unsupported_state(charge, mult):
    atoms = Atoms('H2', positions=[[0, 0, 0], [0.8, 0, 0]])
    atoms.info.update(charge=charge, mult=mult)
    atoms.calc = NeutralModel()
    with pytest.raises(ValueError, match='electronic state|charge/multiplicity'):
        atoms.get_potential_energy()
    assert atoms.calc.calls == 0


def test_model_bound_thermochemistry_cannot_add_unsupported_spin_entropy(tmp_path):
    atoms = Atoms('H2', positions=[[0, 0, 0], [0.8, 0, 0]])
    atoms.calc = NeutralModel()
    atoms.get_potential_energy()
    atoms.info['mult'] = 3
    job = MWFrequency(str(tmp_path / 'frequency.out'), atoms, ilowfreq=0)
    with pytest.raises(ValueError, match='electronic state|charge/multiplicity'):
        job.compute_thermo(np.array([1000.0]))


def test_neutral_singlet_still_works():
    atoms = Atoms('H2')
    atoms.calc = NeutralModel()
    assert atoms.get_potential_energy() == 0.0


def test_direct_ani_hvp_rejects_unsupported_state_before_forward():
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    atoms = Atoms('H2')
    atoms.info['mult'] = 3
    calculator = object.__new__(ANICalculator)
    with pytest.raises(ValueError, match='electronic state'):
        calculator.get_hvp(atoms, np.zeros(6))


def test_fennol_partial_capability_keeps_charge_but_rejects_spin(tmp_path):
    from maple.function.calculator.fennol._fennol_calculator import FeNNolCalculator

    atoms = Atoms('H2', positions=[[0, 0, 0], [0.8, 0, 0]])
    atoms.info.update(charge=1, mult=1)
    factory = SetCalculator('cpu', 'fennol', str(tmp_path/'out'), atoms=atoms)
    factory._validate_against_class(FeNNolCalculator)
    atoms.info.update(charge=0, mult=3)
    with pytest.raises(ValueError, match='multiplicity|mult|electronic state'):
        factory._validate_against_class(FeNNolCalculator)
    atoms.calc = object.__new__(FeNNolCalculator)
    with pytest.raises(ValueError, match='multiplicity|mult|electronic state'):
        MWFrequency(str(tmp_path/'freq'), atoms).compute_thermo(np.array([1000.0]))


def test_uma_task_capability_is_used_by_factory_and_thermochemistry(tmp_path):
    pytest.importorskip('fairchem.core')
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    class MaterialUMA(UMACalculator):
        @property
        def task_name(self):
            return 'omat'

    atoms = Atoms('H2', positions=[[0, 0, 0], [0.8, 0, 0]])
    atoms.info.update(charge=0, mult=3)
    factory = SetCalculator('cpu', 'uma', str(tmp_path/'out'), atoms=atoms,
                            model_options={'task': 'omat'})
    with pytest.raises(ValueError, match='electronic state|charge/spin'):
        factory._validate_against_class(UMACalculator)
    atoms.calc = object.__new__(MaterialUMA)
    with pytest.raises(ValueError, match='electronic state|charge/spin'):
        MWFrequency(str(tmp_path/'freq'), atoms).compute_thermo(np.array([1000.0]))
    factory.model_options['task'] = 'omol'
    factory._validate_against_class(UMACalculator)


@pytest.mark.parametrize('charge', [0, 1])
def test_fennol_hvp_uses_the_shared_partial_state_contract(charge, monkeypatch):
    pytest.importorskip('torch')
    from maple.function.calculator.fennol._fennol_calculator import FeNNolCalculator

    class Runtime:
        energy_unit = 'hartree'

        def __init__(self):
            self.calls = []

        def hvp(self, atoms, total_charge, vector):
            self.calls.append(total_charge)
            return vector, np.zeros_like(vector), total_charge

    atoms = Atoms('H2')
    atoms.info.update(charge=charge, mult=1)
    calculator = object.__new__(FeNNolCalculator)
    runtime = Runtime()
    monkeypatch.setattr(calculator, "runtime", runtime, raising=False)
    vector = np.ones(6)
    hvp, _forces, energy = calculator.get_hvp(atoms, vector)
    np.testing.assert_array_equal(hvp.numpy(), vector)
    assert energy.item() == charge
    assert runtime.calls == [charge]
    atoms.info['mult'] = 3
    with pytest.raises(ValueError, match='multiplicity|mult'):
        calculator.get_hvp(atoms, vector)
    assert runtime.calls == [charge]
