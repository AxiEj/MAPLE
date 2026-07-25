from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.solvfe.alchemy import (
    FragmentPartition,
    GaussianRepulsiveCore,
    ManyBodyInteractionEvaluator,
)
from maple.function.dispatcher.solvfe.sampling import (
    AlchemicalSampleSet,
    AlchemicalSchedule,
    AlchemicalWindowRunner,
    WindowRunConfig,
)
from maple.function.dispatcher.solvfe.shell import (
    FlatBottomSurfaceRestraint,
)


class _AdditiveHarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(atoms.positions, dtype=float)
        self.results = {
            "energy": 0.5 * float(np.sum(positions**2)),
            "forces": -positions,
        }


def test_initial_schedule_has_shared_endpoints_exactly_once():
    schedule = AlchemicalSchedule.initial(points_per_leg=3)

    assert len(schedule.states) == 7
    assert schedule.states[0].scales == (0.0, 0.0)
    assert schedule.states[-1].scales == (0.0, 1.0)
    assert sum(state.scales == (1.0, 0.0) for state in schedule.states) == 1
    assert sum(state.scales == (1.0, 1.0) for state in schedule.states) == 1
    assert len(set(schedule.labels)) == len(schedule.labels)
    assert schedule.content_hash


def test_basis_samples_generate_dimensionless_reduced_potentials():
    schedule = AlchemicalSchedule.initial(points_per_leg=2)
    basis = np.array(
        [
            [10.0, -2.0, 3.0, 0.5],
            [11.0, -1.0, 4.0, 0.5],
            [12.0, -3.0, 2.0, 0.5],
            [13.0, -4.0, 1.0, 0.5],
        ]
    )
    samples = AlchemicalSampleSet.create(
        schedule=schedule,
        N_k=(1, 1, 1, 1),
        frame_ids=("f0", "f1", "f2", "f3"),
        positions_angstrom=np.zeros((4, 4, 3)),
        basis_ev=basis,
        atom_list_hash="a" * 64,
        restraint_hash="b" * 64,
        measure_id="nonperiodic-solute-com-reduced-v1",
        boundary_conditions="nonperiodic",
    )
    table = samples.reduced_potential_table(beta_ev_inverse=2.0)

    expected_state_0 = 2.0 * (basis[:, 0] + basis[:, 3])
    expected_state_p = 2.0 * (
        basis[:, 0] + basis[:, 1] + basis[:, 3]
    )
    assert table.u_kn[0] == pytest.approx(expected_state_0)
    assert table.u_kn[-1] == pytest.approx(expected_state_p)
    assert table.N_k == (1, 1, 1, 1)


def test_window_runner_generates_grouped_finite_restartable_samples(tmp_path):
    atoms = Atoms(
        "COH2",
        positions=[
            [0.0, 0.0, 0.0],
            [2.9, 0.0, 0.0],
            [3.86, 0.0, 0.0],
            [2.66, 0.93, 0.0],
        ],
    )
    partition = FragmentPartition.from_atoms(
        atoms,
        solute_indices=(0,),
        water_indices=(1, 2, 3),
    )
    evaluator = ManyBodyInteractionEvaluator(
        calculator=_AdditiveHarmonicCalculator(),
        partition=partition,
    )
    restraint = FlatBottomSurfaceRestraint(
        solute_indices=(0,),
        water_oxygen_indices=(1,),
        water_atom_indices=(1, 2, 3),
        vdw_radii_angstrom={6: 1.70},
        lambda_s_angstrom=1.50,
        force_constant_ev_per_angstrom2=2.0,
        shell_boundary_id="toy",
        measure_id="nonperiodic-solute-com-reduced-v1",
    )
    runner = AlchemicalWindowRunner(
        evaluator=evaluator,
        repulsive_core=GaussianRepulsiveCore(
            epsilon_ev=0.48,
            sigma_angstrom=0.85,
        ),
        restraint=restraint,
        schedule=AlchemicalSchedule.initial(points_per_leg=2),
        config=WindowRunConfig(
            temperature_k=50.0,
            timestep_fs=0.1,
            friction_per_fs=0.1,
            equilibration_steps=2,
            production_steps=4,
            sample_interval=2,
            seed=20260725,
        ),
    )

    samples = runner.run(atoms)
    samples.write(tmp_path / "samples")
    resumed = AlchemicalSampleSet.load(tmp_path / "samples")

    assert samples.N_k == (2, 2, 2, 2)
    assert resumed.content_hash == samples.content_hash
    assert resumed.basis_ev.flags.writeable is False
    assert np.all(np.isfinite(resumed.basis_ev))
    assert resumed.reduced_potential_table(
        beta_ev_inverse=1.0
    ).u_kn.shape == (4, 8)
