from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.md.ensemble.nve import NVE
from maple.function.dispatcher.md.ensemble.nvt import NVT
from maple.function.dispatcher.md.utils import (
    VELOCITY_REPR_STANDARD,
    calculate_kinetic_energy,
    restore_last_valid_md_state,
)


class _ZeroCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": -1.25,
            "forces": np.zeros((len(atoms), 3)),
        }


class _ConstantForceCalculator(_ZeroCalculator):
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["forces"] = np.full((len(atoms), 3), 0.01)


class _SilentLogger:
    def start_simulation(self, **kwargs):
        return None

    def log_main(self, lines):
        return None

    def log_step(self, **kwargs):
        return None

    def end_simulation(self, **kwargs):
        return None


def test_nve_progress_uses_t0_baseline_and_constant_memory():
    atoms = Atoms(
        "OH2",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        calculator=_ZeroCalculator(),
    )
    job = NVE.__new__(NVE)
    job.atoms = atoms
    job.params = SimpleNamespace(
        steps=3,
        timestep=0.1,
        temperature=0.0,
        remove_com_every=0,
        remove_angular_every=0,
        rst_every=100,
    )
    job.logger = _SilentLogger()

    initial_positions = atoms.get_positions().copy()
    velocities = np.zeros((3, 3))
    job._run_simulation(velocities, n_steps=3)

    progress = job._workflow_progress
    assert progress["completed_steps"] == 3
    assert progress["initial_total_energy_hartree"] == -1.25
    assert progress["maximum_total_energy_drift_hartree"] == 0.0
    assert progress["last_valid_step"]["step"] == 3
    assert np.array_equal(atoms.get_positions(), initial_positions)
    assert not hasattr(job, "_workflow_trace")


def test_vrescale_progress_compares_conserved_energy_to_t0():
    atoms = Atoms(
        "OH2",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        calculator=_ZeroCalculator(),
    )
    job = NVT.__new__(NVT)
    job.atoms = atoms
    job.params = SimpleNamespace(
        steps=2,
        timestep=0.1,
        temperature=0.0,
        thermostat="v-rescale",
        remove_com_every=0,
        remove_angular_every=0,
        rst_every=100,
    )
    job.logger = _SilentLogger()
    job.thermostat = SimpleNamespace(
        timestep=1.0,
        apply=lambda velocities: (velocities, 0.0),
    )
    job._runtime_n_dof = 9
    job._runtime_dof_description = "test"
    job._rng = np.random.default_rng(7)

    velocities = np.zeros((3, 3))
    job._run_simulation(
        velocities,
        velocity_representation=VELOCITY_REPR_STANDARD,
        n_steps=2,
    )

    progress = job._workflow_progress
    assert progress["completed_steps"] == 2
    assert progress["initial_conserved_energy_hartree"] == -1.25
    assert progress["maximum_conserved_energy_drift_hartree"] == 0.0
    assert progress["last_valid_step"]["step"] == 2
    assert not hasattr(job, "_workflow_trace")


def test_failed_attempt_restores_independent_last_valid_snapshot():
    atoms = Atoms("H2", positions=[[9, 9, 9], [8, 8, 8]])
    progress = {
        "last_valid_step": {
            "step": 4,
            "positions_angstrom": [[0, 0, 0], [0.74, 0, 0]],
            "velocities_bohr_per_au": [[0, 0, 0], [0, 0, 0]],
            "velocity_representation": VELOCITY_REPR_STANDARD,
        }
    }
    restore_last_valid_md_state(atoms, progress)
    assert np.array_equal(atoms.positions, [[0, 0, 0], [0.74, 0, 0]])
    assert np.array_equal(atoms.arrays["velocities"], np.zeros((2, 3)))


def test_langevin_progress_uses_standard_synchronized_thermodynamics_at_t0():
    atoms = Atoms(
        "OH2",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        calculator=_ConstantForceCalculator(),
    )
    job = NVT.__new__(NVT)
    job.atoms = atoms
    job.params = SimpleNamespace(
        steps=1,
        timestep=0.1,
        temperature=0.0,
        thermostat="langevin",
        remove_com_every=0,
        remove_angular_every=0,
        rst_every=100,
    )
    job.logger = _SilentLogger()
    job.thermostat = SimpleNamespace(timestep=4.1341374575751, apply=lambda v: v)
    job._runtime_n_dof = 9
    job._runtime_dof_description = "test"
    job._rng = np.random.default_rng(7)
    velocities = np.full((3, 3), 1e-4)
    expected_t0 = calculate_kinetic_energy(atoms, velocities) - 1.25

    job._run_simulation(
        velocities,
        velocity_representation=VELOCITY_REPR_STANDARD,
        n_steps=1,
    )

    progress = job._workflow_progress
    assert np.isclose(progress["initial_total_energy_hartree"], expected_t0)
    assert progress["thermodynamic_velocity_representation"] == "standard_synchronized"
    assert progress["last_valid_step"]["velocity_representation"] != (
        progress["thermodynamic_velocity_representation"]
    )
