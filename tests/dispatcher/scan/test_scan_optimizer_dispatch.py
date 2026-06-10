import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.scan import Scan


class FakeCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, *, pbc_capable: bool):
        super().__init__()
        self.maple_model_name = "fake-pbc" if pbc_capable else "fake-cluster"
        self.maple_pbc_md_supported = pbc_capable
        self.maple_stress_supported = False

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))


def _periodic_atoms(*, pbc_capable: bool) -> Atoms:
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.calc = FakeCalculator(pbc_capable=pbc_capable)
    return atoms


def test_scan_reused_optimization_path_fires_pbc_gate(tmp_path):
    scan = Scan(
        output=str(tmp_path / "scan.out"),
        atoms=_periodic_atoms(pbc_capable=True),
        method="lbfgs",
        constraints=[[1, 2, 0.1, 1]],
        params={"mode": "relaxed"},
    )

    with pytest.raises(ValueError, match="OPT with PBC"):
        scan._run_optimizer(_periodic_atoms(pbc_capable=False))
