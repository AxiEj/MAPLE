import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator.set_calculator import SetClaculator
from maple.function.dispatcher.irc import IRC
from maple.function.dispatcher.optimization import Optimization
from maple.function.dispatcher.scan import Scan
from maple.function.dispatcher.ts import TransitionState


class FakeCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, *, pbc_capable: bool):
        super().__init__()
        self.maple_model_name = "fake-pbc" if pbc_capable else "fake-cluster"
        self.maple_pbc_md_supported = pbc_capable
        self.maple_stress_supported = pbc_capable

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))
        self.results["stress"] = np.zeros(6)


def _atoms(*, pbc: bool, pbc_capable: bool) -> Atoms:
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=pbc,
    )
    atoms.calc = FakeCalculator(pbc_capable=pbc_capable)
    return atoms


def _construct_task(task: str, atoms: Atoms, output: str):
    if task == "opt":
        return Optimization(params={"method": "lbfgs"}, output=output, atoms=atoms)
    if task == "scan":
        return Scan(
            output=output,
            atoms=atoms,
            method="lbfgs",
            constraints=[[1, 2, 0.1, 1]],
            params={"mode": "rigid"},
        )
    if task == "ts":
        return TransitionState(output=output, atoms=atoms, params={"method": "prfo"}, method="prfo")
    if task == "irc":
        return IRC(params={"method": "hpc"}, output=output, atoms=atoms, method="hpc")
    raise AssertionError(f"unknown task {task}")


@pytest.mark.parametrize("task", ["opt", "scan", "ts", "irc"])
def test_periodic_tasks_reject_non_pbc_calculator(task, tmp_path):
    atoms = _atoms(pbc=True, pbc_capable=False)

    with pytest.raises(ValueError, match=f"{task.upper()} with PBC"):
        _construct_task(task, atoms, str(tmp_path / f"{task}.out"))


@pytest.mark.parametrize("task", ["opt", "scan", "ts", "irc"])
def test_periodic_tasks_accept_pbc_capable_calculator(task, tmp_path):
    atoms = _atoms(pbc=True, pbc_capable=True)

    _construct_task(task, atoms, str(tmp_path / f"{task}.out"))


@pytest.mark.parametrize("task", ["opt", "scan", "ts", "irc"])
def test_nonperiodic_tasks_skip_pbc_capability_gate(task, tmp_path):
    atoms = _atoms(pbc=False, pbc_capable=False)

    _construct_task(task, atoms, str(tmp_path / f"{task}.out"))


def _set_fake_pbc_calculator(monkeypatch, tmp_path, cutoff, cell=None):
    class FakePBCBackend:
        supported_hessian_modes = ()
        neighbor_cutoff_A = cutoff

    def fake_build(self):
        return FakePBCBackend()

    monkeypatch.setattr(SetClaculator, "_build_calculator", fake_build)
    atoms = Atoms(
        "Cu",
        positions=[[0.0, 0.0, 0.0]],
        cell=cell if cell is not None else [3.6, 3.6, 3.6],
        pbc=True,
    )
    return SetClaculator(
        torch.device("cpu"),
        "aimnet2-pbc",
        str(tmp_path / "cutoff.out"),
        atoms=atoms,
    ).set_calculator()


def test_pbc_neighbor_cutoff_inside_minimum_image_radius_passes(monkeypatch, tmp_path):
    _set_fake_pbc_calculator(monkeypatch, tmp_path, 1.0)


def test_pbc_neighbor_cutoff_beyond_minimum_image_radius_raises(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="Minimum-image convention"):
        _set_fake_pbc_calculator(monkeypatch, tmp_path, 5.0)


def test_pbc_neighbor_cutoff_uses_triclinic_shortest_lattice_vector(monkeypatch, tmp_path):
    skew_cell = [
        [10.0, 0.0, 0.0],
        [9.0, 1.0, 0.0],
        [0.0, 0.0, 10.0],
    ]

    with pytest.raises(ValueError, match="shortest periodic lattice vector = 1.414 A"):
        _set_fake_pbc_calculator(monkeypatch, tmp_path, 1.0, cell=skew_cell)


def test_pbc_neighbor_cutoff_none_is_silent_skip(monkeypatch, tmp_path):
    _set_fake_pbc_calculator(monkeypatch, tmp_path, None)
