"""Factory input contract delegates supported Cartesian preparation once."""
from ase import Atoms
import pytest
from maple.function.calculator.set_calculator import SetCalculator


def factory(tmp_path, *, source='maple', pbc=False):
    atoms = Atoms('OH2', positions=[[0, 0, 0], [.9572, 0, 0], [-.24, .9273, 0]], pbc=pbc)
    atoms.info.update(charge=0, mult=1)
    return SetCalculator('cpu', 'ani2x', str(tmp_path / 'job'), atoms=atoms,
                         implicit='gb', solvent='water',
                         charge_options={'source': source},
                         solvation_options={'implicit': 'water', 'method': 'gb',
                                            'model': 'obc2', 'experimental': True})


def test_cartesian_maple_charge_passes_factory_without_preparing_or_mutating(tmp_path):
    configured = factory(tmp_path)
    before = set(configured.atoms.arrays)
    configured._validate_solvent_config()
    assert set(configured.atoms.arrays) == before
    assert 'mol2' not in configured.atoms.info
    assert not list(tmp_path.iterdir())


def test_cartesian_fixed_mol2_charge_is_not_implicitly_replaced(tmp_path):
    with pytest.raises(ValueError, match='MOL2'):
        factory(tmp_path, source='mol2')._validate_solvent_config()


def test_periodic_cartesian_rejected_before_preparation(tmp_path):
    with pytest.raises(ValueError, match='non-periodic'):
        factory(tmp_path, pbc=True)._validate_solvent_config()
