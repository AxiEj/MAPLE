from __future__ import annotations

import importlib

import numpy as np
import pytest
from ase import Atoms


def _module():
    return importlib.import_module("maple.function.free_energy.switching")


class _HarmonicEvaluator:
    def __init__(self, force_constant: float, offset_hartree: float):
        self.force_constant = float(force_constant)
        self.offset_hartree = float(offset_hartree)
        self.calls = 0

    def evaluate(self, atoms, *, need_forces: bool):
        self.calls += 1
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        energy = self.offset_hartree + 0.5 * self.force_constant * float(
            np.sum(positions**2)
        )
        return {
            "energy_hartree": energy,
            "forces_hartree_per_angstrom": (
                -self.force_constant * positions if need_forces else None
            ),
        }


def test_linear_hamiltonian_calculator_interpolates_energy_and_force():
    module = _module()
    atoms = Atoms("H", positions=[[0.2, -0.1, 0.3]])
    reference = _HarmonicEvaluator(0.5, 1.0)
    target = _HarmonicEvaluator(1.5, 5.0)
    calculator = module.LinearHamiltonianCalculator(
        reference,
        target,
        coupling=0.25,
        target_energy_offset_hartree=4.0,
    )
    atoms.calc = calculator

    observed_energy = atoms.get_potential_energy()
    observed_forces = atoms.get_forces()
    reference_energy = 1.0 + 0.25 * np.sum(atoms.positions**2)
    aligned_target_energy = 1.0 + 0.75 * np.sum(atoms.positions**2)
    expected_energy = 0.75 * reference_energy + 0.25 * aligned_target_energy
    expected_forces = 0.75 * (-0.5 * atoms.positions) + 0.25 * (-1.5 * atoms.positions)

    assert observed_energy == pytest.approx(expected_energy)
    assert observed_forces == pytest.approx(expected_forces)
    assert calculator.endpoint_energy_gap_hartree == pytest.approx(
        aligned_target_energy - reference_energy
    )
    assert calculator.operation_counts == {
        "reference_energy_force_evaluations": 1,
        "target_energy_force_evaluations": 1,
    }

    calculator.set_coupling(0.75)
    assert atoms.get_potential_energy() == pytest.approx(
        0.25 * reference_energy + 0.75 * aligned_target_energy
    )
    assert atoms.get_forces() == pytest.approx(
        0.25 * (-0.5 * atoms.positions) + 0.75 * (-1.5 * atoms.positions)
    )
    assert reference.calls == 1
    assert target.calls == 1


def test_constant_gap_switch_has_exact_work_and_one_endpoint_call_per_step():
    module = _module()
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    reference = _HarmonicEvaluator(0.0, 2.0)
    target = _HarmonicEvaluator(0.0, 8.0)
    calculator = module.LinearHamiltonianCalculator(
        reference,
        target,
        coupling=0.0,
        target_energy_offset_hartree=5.0,
    )
    atoms.calc = calculator
    schedule = np.linspace(0.0, 1.0, 6)

    result = module.run_linear_nonequilibrium_switch(
        atoms,
        calculator,
        lambda_schedule=schedule,
        temperature_kelvin=298.15,
        timestep_fs=0.1,
        friction_per_fs=0.0,
        seed=901,
        initial_velocities_au=np.full((1, 3), 1.0e-5),
    )

    assert result["direction"] == "forward"
    assert result["work_hartree"] == pytest.approx(1.0, abs=1.0e-14)
    assert result["work_increments_hartree"] == pytest.approx([0.2] * 5)
    assert result["operation_counts"] == {
        "reference_energy_force_evaluations": 6,
        "target_energy_force_evaluations": 6,
    }
    assert np.linalg.norm(result["final_positions_angstrom"]) > 0.0


def test_switch_schedule_must_be_monotonic_and_span_endpoints():
    module = _module()
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calculator = module.LinearHamiltonianCalculator(
        _HarmonicEvaluator(0.0, 0.0),
        _HarmonicEvaluator(0.0, 0.0),
        coupling=0.0,
    )
    atoms.calc = calculator

    with pytest.raises(ValueError, match="span either 0 to 1 or 1 to 0"):
        module.run_linear_nonequilibrium_switch(
            atoms,
            calculator,
            lambda_schedule=[0.0, 0.5],
            temperature_kelvin=298.15,
            timestep_fs=0.1,
            friction_per_fs=0.001,
            seed=1,
        )

    with pytest.raises(ValueError, match="strictly monotonic"):
        module.run_linear_nonequilibrium_switch(
            atoms,
            calculator,
            lambda_schedule=[0.0, 0.5, 0.4, 1.0],
            temperature_kelvin=298.15,
            timestep_fs=0.1,
            friction_per_fs=0.001,
            seed=1,
        )
