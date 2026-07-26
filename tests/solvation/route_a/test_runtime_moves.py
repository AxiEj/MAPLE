from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.solvfe.alchemy import (
    AlchemicalState,
    FragmentPartition,
    GaussianRepulsiveCore,
    ManyBodyInteractionEvaluator,
    SequentialInsertionCalculator,
)
from maple.function.dispatcher.solvfe.moves import (
    RigidBodyMetropolis,
    RigidBodyMoveConfig,
)
from maple.function.dispatcher.solvfe.shell import FlatBottomSurfaceRestraint


class _MustNotEvaluate:
    def calculate(self, *args, **kwargs):
        raise AssertionError("D-state rigid moves must not evaluate the MLIP")


class _PairCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(atoms.positions, dtype=float)
        energy = 0.0
        forces = np.zeros_like(positions)
        for left in range(len(atoms)):
            for right in range(left + 1, len(atoms)):
                displacement = positions[left] - positions[right]
                energy += 0.5 * float(np.dot(displacement, displacement))
                forces[left] -= displacement
                forces[right] += displacement
        self.results = {"energy": energy, "forces": forces}


def _system():
    atoms = Atoms(
        "COH2",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.75, 3.55],
            [0.0, -0.75, 3.55],
        ],
    )
    partition = FragmentPartition.from_atoms(
        atoms,
        solute_indices=(0,),
        water_indices=(1, 2, 3),
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
    return atoms, partition, restraint


def test_decoupled_rigid_moves_mix_translation_and_orientation_without_mlip():
    atoms, partition, restraint = _system()
    evaluator = ManyBodyInteractionEvaluator(
        calculator=_MustNotEvaluate(),
        partition=partition,
    )
    config = RigidBodyMoveConfig(
        water_groups=((1, 2, 3),),
        equilibration_attempts=0,
        attempts_per_sample=0,
        translation_step_angstrom=1.0,
        rotation_step_radians=1.0,
    )
    mover = RigidBodyMetropolis(
        evaluator=evaluator,
        repulsive_core=GaussianRepulsiveCore(0.48, 0.85),
        restraint=restraint,
        state=AlchemicalState.endpoint("D"),
        temperature_k=298.15,
        config=config,
    )
    initial_positions = atoms.positions.copy()
    initial_oh = initial_positions[2] - initial_positions[1]
    statistics = mover.run(
        atoms,
        attempts=200,
        rng=np.random.default_rng(20260725),
    )
    final_oh = atoms.positions[2] - atoms.positions[1]

    assert statistics.accepted > 0
    assert statistics.translation_attempts > 0
    assert statistics.rotation_attempts > 0
    assert np.linalg.norm(atoms.positions[1] - initial_positions[1]) > 0.5
    assert np.dot(initial_oh, final_oh) / (
        np.linalg.norm(initial_oh) * np.linalg.norm(final_oh)
    ) < 0.99
    assert np.allclose(
        np.linalg.norm(atoms.positions[2:4] - atoms.positions[1], axis=1),
        np.linalg.norm(initial_positions[2:4] - initial_positions[1], axis=1),
    )


def test_rigid_move_groups_must_cover_exact_alchemical_water_partition():
    atoms, partition, restraint = _system()
    evaluator = ManyBodyInteractionEvaluator(
        calculator=_MustNotEvaluate(),
        partition=partition,
    )
    with pytest.raises(ValueError, match="exactly one rigid OHH water"):
        RigidBodyMoveConfig(
            water_groups=((1, 2),),
            equilibration_attempts=0,
            attempts_per_sample=0,
            translation_step_angstrom=1.0,
            rotation_step_radians=1.0,
        )

    config = RigidBodyMoveConfig(
        water_groups=((0, 1, 2),),
        equilibration_attempts=0,
        attempts_per_sample=0,
        translation_step_angstrom=1.0,
        rotation_step_radians=1.0,
    )
    with pytest.raises(
        ValueError,
        match="partition all alchemical water indices",
    ):
        RigidBodyMetropolis(
            evaluator=evaluator,
            repulsive_core=GaussianRepulsiveCore(0.48, 0.85),
            restraint=restraint,
            state=AlchemicalState.endpoint("D"),
            temperature_k=298.15,
            config=config,
        )


def test_rigid_move_config_rejects_multiple_water_groups():
    with pytest.raises(ValueError, match="exactly one rigid OHH water"):
        RigidBodyMoveConfig(
            water_groups=((1, 2, 3), (4, 5, 6)),
            equilibration_attempts=0,
            attempts_per_sample=0,
            translation_step_angstrom=1.0,
            rotation_step_radians=1.0,
        )


def test_rigid_move_rejects_non_ohh_alchemical_fragment():
    atoms = Atoms(
        "COHHe",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.75, 3.55],
            [0.0, -0.75, 3.55],
        ],
    )
    partition = FragmentPartition.from_atoms(
        atoms,
        solute_indices=(0,),
        water_indices=(1, 2, 3),
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
    mover = RigidBodyMetropolis(
        evaluator=ManyBodyInteractionEvaluator(
            calculator=_MustNotEvaluate(),
            partition=partition,
        ),
        repulsive_core=GaussianRepulsiveCore(0.48, 0.85),
        restraint=restraint,
        state=AlchemicalState.endpoint("D"),
        temperature_k=298.15,
        config=RigidBodyMoveConfig(
            water_groups=((1, 2, 3),),
            equilibration_attempts=0,
            attempts_per_sample=0,
            translation_step_angstrom=1.0,
            rotation_step_radians=1.0,
        ),
    )

    with pytest.raises(ValueError, match="exactly one O and two H"):
        mover.run(atoms, attempts=1, rng=np.random.default_rng(7))


def test_single_water_move_energy_difference_matches_full_hamiltonian():
    atoms, partition, restraint = _system()
    evaluator = ManyBodyInteractionEvaluator(
        calculator=_PairCalculator(),
        partition=partition,
    )
    core = GaussianRepulsiveCore(0.48, 0.85)
    state = AlchemicalState("R_TO_I", 0.4)
    mover = RigidBodyMetropolis(
        evaluator=evaluator,
        repulsive_core=core,
        restraint=restraint,
        state=state,
        temperature_k=298.15,
        config=RigidBodyMoveConfig(
            water_groups=((1, 2, 3),),
            equilibration_attempts=0,
            attempts_per_sample=0,
            translation_step_angstrom=1.0,
            rotation_step_radians=1.0,
        ),
    )
    proposal = atoms.copy()
    proposal.positions[[1, 2, 3]] += [0.3, -0.2, 0.1]

    calculator = SequentialInsertionCalculator(
        evaluator=evaluator,
        repulsive_core=core,
        restraint=restraint,
        state=state,
    )
    current_full = atoms.copy()
    current_full.calc = calculator
    current_energy = current_full.get_potential_energy()
    proposal_full = proposal.copy()
    proposal_full.calc = calculator
    proposal_energy = proposal_full.get_potential_energy()

    assert (
        mover._relative_energy_ev(proposal)
        - mover._relative_energy_ev(atoms)
    ) == pytest.approx(proposal_energy - current_energy, abs=1.0e-12)
