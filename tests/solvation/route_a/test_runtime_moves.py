from __future__ import annotations

import numpy as np
from ase import Atoms

from maple.function.dispatcher.solvfe.alchemy import (
    AlchemicalState,
    FragmentPartition,
    GaussianRepulsiveCore,
    ManyBodyInteractionEvaluator,
)
from maple.function.dispatcher.solvfe.moves import (
    RigidBodyMetropolis,
    RigidBodyMoveConfig,
)
from maple.function.dispatcher.solvfe.shell import FlatBottomSurfaceRestraint


class _MustNotEvaluate:
    def calculate(self, *args, **kwargs):
        raise AssertionError("D-state rigid moves must not evaluate the MLIP")


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
    config = RigidBodyMoveConfig(
        water_groups=((1, 2),),
        equilibration_attempts=0,
        attempts_per_sample=0,
        translation_step_angstrom=1.0,
        rotation_step_radians=1.0,
    )

    try:
        RigidBodyMetropolis(
            evaluator=evaluator,
            repulsive_core=GaussianRepulsiveCore(0.48, 0.85),
            restraint=restraint,
            state=AlchemicalState.endpoint("D"),
            temperature_k=298.15,
            config=config,
        )
    except ValueError as exc:
        assert "partition all alchemical water indices" in str(exc)
    else:
        raise AssertionError("Incomplete water group must fail closed.")
