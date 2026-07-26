from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.solvfe.alchemy import (
    AlchemicalState,
    AtomListMismatch,
    FragmentPartition,
    GaussianRepulsiveCore,
    ManyBodyInteractionEvaluator,
    SequentialInsertionCalculator,
)


class TaggedManyBodyCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        if hasattr(self, "call_sizes"):
            self.call_sizes.append(len(atoms))
        positions = np.asarray(atoms.positions, dtype=float)
        tags = np.asarray(atoms.get_tags(), dtype=int)
        energy = 0.0
        forces = np.zeros_like(positions)

        for left in range(len(atoms)):
            for right in range(left + 1, len(atoms)):
                displacement = positions[left] - positions[right]
                if tags[left] == tags[right]:
                    coefficient = 0.2 if tags[left] == 0 else 0.3
                else:
                    coefficient = 0.5
                pair_energy = 0.5 * coefficient * np.dot(
                    displacement, displacement
                )
                pair_force = -coefficient * displacement
                energy += pair_energy
                forces[left] += pair_force
                forces[right] -= pair_force

        # A genuine irreducible solute-water-water term: it exists only in the
        # combined system and cannot be reconstructed from isolated fragments.
        solute = np.flatnonzero(tags == 0)
        water = np.flatnonzero(tags == 1)
        if len(solute) and len(water) >= 2:
            anchor, first, second = solute[0], water[0], water[1]
            a = positions[first] - positions[anchor]
            b = positions[second] - positions[anchor]
            dot = float(np.dot(a, b))
            coefficient = 0.07
            three_body = coefficient * dot * dot
            grad_a = 2.0 * coefficient * dot * b
            grad_b = 2.0 * coefficient * dot * a
            energy += three_body
            forces[first] -= grad_a
            forces[second] -= grad_b
            forces[anchor] += grad_a + grad_b

        self.results = {"energy": float(energy), "forces": forces}


def _system():
    atoms = Atoms(
        "COH2",
        positions=[
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [2.4, 0.2, 0.0],
            [2.8, 0.9, 0.0],
        ],
        tags=[0, 0, 1, 1],
    )
    atoms.info.update(charge=0, mult=1)
    partition = FragmentPartition.from_atoms(
        atoms,
        solute_indices=(0, 1),
        water_indices=(2, 3),
    )
    evaluator = ManyBodyInteractionEvaluator(
        calculator=TaggedManyBodyCalculator(),
        partition=partition,
    )
    core = GaussianRepulsiveCore(epsilon_ev=0.48, sigma_angstrom=0.85)
    return atoms, partition, evaluator, core


def test_sequential_stages_preserve_many_body_endpoint_identities():
    atoms, partition, evaluator, core = _system()
    values = {}
    for stage in ("D", "R", "I", "P"):
        calc = SequentialInsertionCalculator(
            evaluator=evaluator,
            repulsive_core=core,
            state=AlchemicalState.endpoint(stage),
        )
        probe = atoms.copy()
        probe.calc = calc
        values[stage] = {
            "energy": probe.get_potential_energy(),
            "forces": probe.get_forces(),
            "components": dict(calc.results["components"]),
        }

    assert values["D"]["components"]["full_scale"] == 0.0
    assert values["D"]["components"]["repulsive_scale"] == 0.0
    assert values["R"]["energy"] > values["D"]["energy"]
    assert values["I"]["energy"] == pytest.approx(
        values["P"]["energy"]
        + values["R"]["energy"]
        - values["D"]["energy"],
        abs=1.0e-12,
    )
    assert values["P"]["components"]["total_ev"] == pytest.approx(
        values["P"]["components"]["combined_ev"],
        abs=1.0e-12,
    )
    assert values["P"]["components"]["interaction_ev"] != 0.0
    np.testing.assert_allclose(
        values["P"]["forces"],
        values["P"]["components"]["combined_forces_ev_per_angstrom"],
    )


def test_continuous_legs_share_exact_D_R_I_P_endpoints():
    _, _, evaluator, core = _system()
    endpoints = [
        AlchemicalState("D_TO_R", 0.0),
        AlchemicalState("D_TO_R", 1.0),
        AlchemicalState("R_TO_I", 0.0),
        AlchemicalState("R_TO_I", 1.0),
        AlchemicalState("I_TO_P", 0.0),
        AlchemicalState("I_TO_P", 1.0),
    ]
    scales = [state.scales for state in endpoints]
    assert scales == [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ]

    with pytest.raises(ValueError):
        AlchemicalState("R_TO_I", 1.01)


def test_atom_reordering_or_transmutation_fails_before_evaluation():
    atoms, partition, evaluator, core = _system()
    calculator = SequentialInsertionCalculator(
        evaluator=evaluator,
        repulsive_core=core,
        state=AlchemicalState.endpoint("P"),
    )
    reordered = atoms[[1, 0, 2, 3]]
    reordered.calc = calculator
    with pytest.raises(AtomListMismatch, match="ATOM_LIST_MISMATCH"):
        reordered.get_potential_energy()


def test_gaussian_core_is_finite_and_separating_near_overlap():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    partition = FragmentPartition.from_atoms(
        atoms,
        solute_indices=(0,),
        water_indices=(1,),
    )
    core = GaussianRepulsiveCore(epsilon_ev=0.48, sigma_angstrom=0.85)
    energy, forces = core.evaluate(atoms, partition)

    assert np.isfinite(energy)
    assert np.all(np.isfinite(forces))
    displacement = atoms.positions[0] - atoms.positions[1]
    relative_force = forces[0] - forces[1]
    assert energy > 0.0
    assert np.dot(relative_force, displacement) > 0.0
    assert core.content_hash


def test_gaussian_core_rejects_exact_zero_force_coincidence():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0]] * 2)
    partition = FragmentPartition.from_atoms(
        atoms,
        solute_indices=(0,),
        water_indices=(1,),
    )

    with pytest.raises(RuntimeError, match="REPULSIVE_CORE_COINCIDENT"):
        GaussianRepulsiveCore(
            epsilon_ev=0.48,
            sigma_angstrom=0.85,
        ).evaluate(atoms, partition)


def test_decoupled_dynamics_defers_combined_pass_until_sample_boundary():
    atoms, _, evaluator, core = _system()
    evaluator.calculator.call_sizes = []
    calculator = SequentialInsertionCalculator(
        evaluator=evaluator,
        repulsive_core=core,
        state=AlchemicalState.endpoint("D"),
        capture_full_basis=False,
    )
    probe = atoms.copy()
    probe.calc = calculator

    probe.get_forces()

    assert evaluator.calculator.call_sizes == [2, 2]
    assert calculator.results["components"]["interaction_ev"] == 0.0

    full_basis = calculator.evaluate_full_basis(probe)

    assert evaluator.calculator.call_sizes[-3:] == [2, 2, 4]
    assert full_basis["interaction_ev"] != 0.0
    assert full_basis["total_ev"] == pytest.approx(
        calculator.results["components"]["total_ev"]
    )
