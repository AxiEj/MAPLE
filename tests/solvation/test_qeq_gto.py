from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.charge.qeq import QEqTorch
from maple.function.calculator.extra_correction.implicit.charges import QEqGTO


def test_qeq_gto_conserves_declared_charge():
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    qeq = QEqGTO()
    charges = qeq.solve(atoms, total_charge=0)
    assert abs(float(charges.sum())) < 1.0e-10
    assert np.isfinite(charges).all()


def test_qeq_gto_kernel_matches_cheq_reference_integral():
    qeq = QEqGTO()
    value = qeq.coulomb_kernel(1.128, 0.759, 2, 0.669, 2, lambda_scale=0.4913)
    assert np.isclose(value, 5.837573337, atol=1.0e-8)


def test_qeq_gto_water_locks_full_hydrogen_screening_scf():
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.9572, 0, 0], [-0.239987, 0.927297, 0]])
    charges = QEqGTO(tolerance=1.0e-12).solve(atoms, total_charge=0)
    expected = np.array([-0.42670396, 0.21337354, 0.21333043])
    assert np.allclose(charges, expected, rtol=0.0, atol=1.0e-8)


def test_qeq_gto_methane_locks_full_hydrogen_screening_scf():
    atoms = Atoms(
        "CH4",
        positions=[
            [0.0, 0.0, 0.0],
            [0.629118, 0.629118, 0.629118],
            [-0.629118, -0.629118, 0.629118],
            [-0.629118, 0.629118, -0.629118],
            [0.629118, -0.629118, -0.629118],
        ],
    )
    charges = QEqGTO(tolerance=1.0e-12).solve(atoms, total_charge=0)
    expected = np.array(
        [-0.15648224, 0.03912056, 0.03912056, 0.03912056, 0.03912056]
    )
    assert np.allclose(charges, expected, rtol=0.0, atol=1.0e-8)


def test_qeq_gto_hydrogen_hardness_is_charge_dependent():
    atoms = Atoms("HF", positions=[[0, 0, 0], [0.917, 0, 0]])
    self_consistent = QEqGTO(tolerance=1.0e-10).solve(atoms, total_charge=0)
    simplified = QEqGTO(tolerance=1.0e-10, hydrogen_scf=False).solve(
        atoms, total_charge=0
    )
    assert self_consistent[0] > 0.0
    assert not np.allclose(self_consistent, simplified, atol=1.0e-3)


def test_qeq_gto_rebuilds_hydrogen_diagonal_and_pair_screening():
    atoms = Atoms("OH", positions=[[0, 0, 0], [0.97, 0, 0]])
    solver = QEqGTO()
    _, neutral = solver.interaction_matrix(atoms, np.array([0.0, 0.0]))
    _, charged = solver.interaction_matrix(atoms, np.array([-0.2, 0.2]))
    assert charged[1, 1] != neutral[1, 1]
    assert charged[0, 1] != neutral[0, 1]


def test_legacy_qeq_torch_delegates_to_the_canonical_solver():
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.9572, 0, 0], [-0.239987, 0.927297, 0]])
    expected = QEqGTO().solve(atoms, total_charge=0)
    actual = QEqTorch()(atoms, total_charge=0).numpy()
    assert np.allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_qeq_gto_fixed_charge_force_matches_coordinate_finite_difference():
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.9572, 0, 0], [-0.239987, 0.927297, 0]])
    solver = QEqGTO(tolerance=1.0e-12)
    charges = solver.solve(atoms, total_charge=0)
    _, forces = solver.energy_and_forces_ev_angstrom(atoms, charges)
    h = 1.0e-5
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions[1, 0] += h
    minus.positions[1, 0] -= h
    finite_difference = -(
        solver.energy_ev(plus, charges) - solver.energy_ev(minus, charges)
    ) / (2.0 * h)
    assert np.isclose(forces[1, 0], finite_difference, atol=1.0e-8)


def test_cqeq_variational_gradient_and_charge_constraint_are_consistent():
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.9572, 0, 0], [-0.239987, 0.927297, 0]])
    solver = QEqGTO(tolerance=1.0e-10)
    extra_hessian = np.diag([-0.2, -0.1, -0.1])
    charges = solver.solve_variational(atoms, extra_hessian=extra_hessian)
    gradient = solver.charge_gradient_ev(atoms, charges, extra_hessian)
    direction = np.array([1.0, -0.5, -0.5])
    h = 1.0e-6

    def total_energy(q):
        return solver.energy_ev(atoms, q) + 0.5 * q @ extra_hessian @ q

    finite_difference = (
        total_energy(charges + h * direction)
        - total_energy(charges - h * direction)
    ) / (2.0 * h)
    assert abs(float(charges.sum())) < 1.0e-10
    assert abs(float(gradient @ direction)) < 2.0e-6
    assert abs(finite_difference) < 2.0e-6
    assert solver.last_variational_min_eigenvalue > 0.0


def test_qeq_fixed_provider_freezes_reference_charges():
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    qeq = QEqGTO()
    reference = qeq.solve(atoms, total_charge=0)
    moved = atoms.copy()
    moved.positions[1, 0] += 0.3
    assert not np.allclose(qeq.solve(moved, total_charge=0), reference)
    moved.set_initial_charges(reference)
    assert np.allclose(moved.get_initial_charges(), reference)


@pytest.mark.parametrize(
    "charges",
    [np.array([0.0]), np.array([0.0, np.nan])],
)
def test_qeq_energy_rejects_invalid_charge_arrays(charges):
    atoms = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    solver = QEqGTO()

    with pytest.raises(ValueError, match="charges"):
        solver.energy_ev(atoms, charges)
