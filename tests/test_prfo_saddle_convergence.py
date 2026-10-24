"""A small force alone cannot establish a first-order saddle."""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from maple.function.dispatcher.frequency.frequency import HARTREE_AMU_ANGSTROM2_TO_CM1
from maple.function.dispatcher.ts.algorithm.PRFO import (
    PRFO,
    _mass_weighted_optimization_basis,
)


@pytest.mark.parametrize(
    "frequencies,accepted",
    [
        ([100.0, 200.0, 300.0], False),
        ([-5.0, 200.0, 300.0], False),
        ([-100.0, 200.0, 300.0], True),
        ([-100.0, -200.0, 300.0], False),
    ],
)
def test_prfo_requires_exactly_one_material_negative_mode(
    tmp_path, frequencies, accepted
):
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    basis, rank = _mass_weighted_optimization_basis(atoms, project_rigid_modes=True)
    assert rank == 6
    freq = np.asarray(frequencies)
    eigenvalues = np.sign(freq) * (freq / HARTREE_AMU_ANGSTROM2_TO_CM1) ** 2
    mass = np.repeat(np.sqrt(atoms.get_masses()), 3)
    hessian = (mass[:, None] * (basis @ np.diag(eigenvalues) @ basis.T)) * mass[None, :]
    calls = []

    def fresh_hessian(current):
        calls.append(current.get_positions().copy())
        return hessian

    atoms.calc = SimpleNamespace(get_hessian=fresh_hessian)
    optimizer = PRFO(
        str(tmp_path / "index.out"), atoms, paras={"project_rigid_modes": True}
    )
    assert optimizer.check_saddle_curvature(atoms) is accepted
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], atoms.get_positions())
    assert optimizer.last_saddle_curvature is not None
    assert optimizer.last_saddle_curvature["negative_mode_count"] == np.count_nonzero(
        freq < -10
    )


def test_prfo_nonfinite_final_hessian_cannot_certify_convergence(tmp_path):
    atoms = Atoms("H", positions=[[0, 0, 0]])
    atoms.calc = SimpleNamespace(get_hessian=lambda current: np.full((3, 3), np.nan))
    optimizer = PRFO(str(tmp_path / "bad.out"), atoms)
    with pytest.raises(ValueError, match="finite"):
        optimizer.check_saddle_curvature(atoms)


def test_prfo_constrained_or_asymmetric_final_hessian_is_rejected(tmp_path):
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    hessian = np.eye(9)
    hessian[0, 1] = 0.1
    atoms.calc = SimpleNamespace(get_hessian=lambda current: hessian)
    optimizer = PRFO(str(tmp_path / "reject.out"), atoms)
    with pytest.raises(ValueError, match="asymmetry"):
        optimizer.check_saddle_curvature(atoms)
    atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(ValueError, match="constraints"):
        optimizer.check_saddle_curvature(atoms)


class Quadratic(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, positions, hessian):
        super().__init__()
        self.reference = np.array(positions, copy=True)
        self._hessian_matrix = np.asarray(hessian)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        displacement = (self.atoms.positions - self.reference).reshape(-1)
        gradient = self._hessian_matrix @ displacement
        energy = float(displacement @ gradient / 2)
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": -gradient.reshape(-1, 3),
        }

    def get_hessian(self, atoms):
        return self._hessian_matrix.copy()


@pytest.mark.parametrize("index_one", [False, True])
def test_real_prfo_run_accepts_only_an_index_one_stationary_point(tmp_path, index_one):
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    basis, _ = _mass_weighted_optimization_basis(atoms, project_rigid_modes=True)
    mass = np.repeat(np.sqrt(atoms.get_masses()), 3)
    modes = np.diag([-1.0 if index_one else 1.0, 2.0, 3.0])
    hessian = mass[:, None] * (basis @ modes @ basis.T) * mass[None, :]
    atoms.calc = Quadratic(atoms.positions, hessian)
    output = tmp_path / "minimum.out"
    PRFO(
        str(output),
        atoms,
        paras={
            "max_iter": 1,
            "project_rigid_modes": True,
            "trust_radius": 0.02,
            "trust_min": 0.02,
        },
    ).run()
    text = output.read_text()
    assert ("Normal Termination" in text) is index_one
    assert f"Final saddle curvature: {int(index_one)}" in text
    if not index_one:
        assert "Maximum Iterations Reached" in text
