import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.md.ensemble.npt import NPT
from maple.function.dispatcher.md.ensemble.nve import NVE
from maple.function.dispatcher.md.ensemble.nvt import NVT


class EnergyForcesCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, *, pbc_capable: bool, stress_capable: bool = False):
        super().__init__()
        self.maple_model_name = "fake-pbc" if pbc_capable else "fake-cluster"
        self.maple_pbc_md_supported = pbc_capable
        self.maple_stress_supported = stress_capable

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))


class StressCalculator(EnergyForcesCalculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, stress):
        super().__init__(pbc_capable=True, stress_capable=True)
        self._stress = np.asarray(stress, dtype=float)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["stress"] = self._stress.copy()


def _periodic_atoms(calc: Calculator) -> Atoms:
    atoms = Atoms(
        "He",
        positions=[[0.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.calc = calc
    return atoms


def test_npt_rejects_missing_stress_at_startup(tmp_path):
    atoms = _periodic_atoms(EnergyForcesCalculator(pbc_capable=True, stress_capable=True))

    with pytest.raises(ValueError, match="stress tensor"):
        NPT(output=str(tmp_path / "npt.out"), atoms=atoms, paras={"steps": 0, "verbose": 0})


@pytest.mark.parametrize(
    "stress",
    [
        [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
        [np.inf, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
)
def test_npt_rejects_nonfinite_stress_at_startup(stress, tmp_path):
    atoms = _periodic_atoms(StressCalculator(stress))

    with pytest.raises(ValueError, match="non-finite stress"):
        NPT(output=str(tmp_path / "npt.out"), atoms=atoms, paras={"steps": 0, "verbose": 0})


def test_npt_accepts_finite_stress_at_startup(tmp_path):
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))

    NPT(output=str(tmp_path / "npt.out"), atoms=atoms, paras={"steps": 0, "verbose": 0})


@pytest.mark.parametrize("ensemble_cls", [NVE, NVT])
def test_nve_nvt_accept_periodic_pbc_capable_calculator(ensemble_cls, tmp_path):
    atoms = _periodic_atoms(EnergyForcesCalculator(pbc_capable=True))

    ensemble_cls(
        output=str(tmp_path / f"{ensemble_cls.__name__.lower()}.out"),
        atoms=atoms,
        paras={"steps": 0, "verbose": 0},
    )


@pytest.mark.parametrize("ensemble_cls", [NVE, NVT])
def test_nve_nvt_reject_periodic_non_pbc_calculator(ensemble_cls, tmp_path):
    atoms = _periodic_atoms(EnergyForcesCalculator(pbc_capable=False))

    with pytest.raises(ValueError, match=f"{ensemble_cls.__name__.upper()} with PBC"):
        ensemble_cls(
            output=str(tmp_path / f"{ensemble_cls.__name__.lower()}.out"),
            atoms=atoms,
            paras={"steps": 0, "verbose": 0},
        )
