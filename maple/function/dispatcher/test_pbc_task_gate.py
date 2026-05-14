import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator.set_calculator import validate_pbc_capabilities


class EnergyForcesCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))


class StressCalculator(EnergyForcesCalculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, stress):
        super().__init__()
        self._stress = np.asarray(stress, dtype=float)
        self.maple_pbc_md_supported = True
        self.maple_stress_supported = True

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["stress"] = self._stress.copy()


def _periodic_atoms(calc):
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True)
    atoms.calc = calc
    return atoms


@pytest.mark.parametrize("jobtype", ["opt", "scan", "ts", "irc"])
def test_pbc_rejects_nonperiodic_backend_in_all_force_tasks(jobtype):
    atoms = _periodic_atoms(EnergyForcesCalculator())
    atoms.calc.maple_model_name = "aimnet2"  # legacy, not PBC-capable
    with pytest.raises(ValueError, match="periodic"):
        validate_pbc_capabilities(atoms, jobtype)


def test_pbc_capability_gate_passes_for_pbc_backend():
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))
    validate_pbc_capabilities(atoms, "opt")  # must not raise


def test_pbc_capability_gate_skipped_for_nonperiodic_atoms():
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], pbc=False)
    atoms.calc = EnergyForcesCalculator()
    validate_pbc_capabilities(atoms, "opt")  # no raise
