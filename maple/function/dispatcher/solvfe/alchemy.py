from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from .protocol import canonical_sha256


class AtomListMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class FragmentPartition:
    solute_indices: tuple[int, ...]
    water_indices: tuple[int, ...]
    atom_list_hash: str

    @classmethod
    def from_atoms(
        cls,
        atoms: Atoms,
        *,
        solute_indices: tuple[int, ...],
        water_indices: tuple[int, ...],
    ) -> "FragmentPartition":
        solute = tuple(int(value) for value in solute_indices)
        water = tuple(int(value) for value in water_indices)
        cls._validate_indices(len(atoms), solute, water)
        return cls(
            solute_indices=solute,
            water_indices=water,
            atom_list_hash=cls._hash(atoms, solute, water),
        )

    @staticmethod
    def _validate_indices(
        atom_count: int,
        solute: tuple[int, ...],
        water: tuple[int, ...],
    ) -> None:
        combined = solute + water
        if not solute or not water:
            raise ValueError("Alchemy requires non-empty solute and water fragments.")
        if len(set(combined)) != len(combined):
            raise ValueError("Alchemy fragment atom lists overlap or contain duplicates.")
        if set(combined) != set(range(atom_count)):
            raise ValueError(
                "Alchemy fragment atom lists must partition the complete atom list."
            )

    @staticmethod
    def _hash(
        atoms: Atoms,
        solute: tuple[int, ...],
        water: tuple[int, ...],
    ) -> str:
        return canonical_sha256(
            {
                "atomic_numbers": [int(value) for value in atoms.numbers],
                "solute_indices": list(solute),
                "water_indices": list(water),
                "combined_indices": list(range(len(atoms))),
            }
        )

    def validate(self, atoms: Atoms) -> None:
        self._validate_indices(
            len(atoms),
            self.solute_indices,
            self.water_indices,
        )
        actual = self._hash(
            atoms,
            self.solute_indices,
            self.water_indices,
        )
        if actual != self.atom_list_hash:
            raise AtomListMismatch(
                "ATOM_LIST_MISMATCH: atom order, identity, or fragment "
                "partition differs from the frozen alchemical state."
            )


@dataclass(frozen=True)
class AlchemicalState:
    leg: str
    lambda_value: float

    _LEGS = frozenset({"D_TO_R", "R_TO_I", "I_TO_P"})

    def __post_init__(self) -> None:
        if self.leg not in self._LEGS:
            raise ValueError(f"Unknown alchemical leg: {self.leg}.")
        if (
            isinstance(self.lambda_value, bool)
            or not math.isfinite(float(self.lambda_value))
            or not 0.0 <= float(self.lambda_value) <= 1.0
        ):
            raise ValueError("Alchemical lambda must be finite and in [0, 1].")

    @property
    def scales(self) -> tuple[float, float]:
        value = float(self.lambda_value)
        if self.leg == "D_TO_R":
            return value, 0.0
        if self.leg == "R_TO_I":
            return 1.0, value
        return 1.0 - value, 1.0

    @classmethod
    def endpoint(cls, stage: str) -> "AlchemicalState":
        endpoints = {
            "D": ("D_TO_R", 0.0),
            "R": ("D_TO_R", 1.0),
            "I": ("R_TO_I", 1.0),
            "P": ("I_TO_P", 1.0),
        }
        try:
            leg, value = endpoints[stage.upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown D/R/I/P endpoint: {stage}.") from exc
        return cls(leg, value)


@dataclass(frozen=True)
class GaussianRepulsiveCore:
    epsilon_ev: float
    sigma_angstrom: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.epsilon_ev, bool)
            or not math.isfinite(float(self.epsilon_ev))
            or self.epsilon_ev <= 0.0
        ):
            raise ValueError("Repulsive epsilon must be finite and positive.")
        if (
            isinstance(self.sigma_angstrom, bool)
            or not math.isfinite(float(self.sigma_angstrom))
            or self.sigma_angstrom <= 0.0
        ):
            raise ValueError("Repulsive sigma must be finite and positive.")

    def evaluate(
        self,
        atoms: Atoms,
        partition: FragmentPartition,
    ) -> tuple[float, np.ndarray]:
        partition.validate(atoms)
        positions = np.asarray(atoms.positions, dtype=float)
        forces = np.zeros_like(positions)
        energy = 0.0
        sigma_squared = float(self.sigma_angstrom) ** 2
        for solute_index in partition.solute_indices:
            for water_index in partition.water_indices:
                displacement = (
                    positions[solute_index] - positions[water_index]
                )
                distance_squared = float(np.dot(displacement, displacement))
                pair_energy = float(self.epsilon_ev) * math.exp(
                    -distance_squared / sigma_squared
                )
                pair_force = (
                    2.0 * pair_energy / sigma_squared * displacement
                )
                energy += pair_energy
                forces[solute_index] += pair_force
                forces[water_index] -= pair_force
        return energy, forces


@dataclass(frozen=True)
class InteractionEvaluation:
    base_ev: float
    combined_ev: float
    interaction_ev: float
    base_forces_ev_per_angstrom: np.ndarray
    combined_forces_ev_per_angstrom: np.ndarray
    interaction_forces_ev_per_angstrom: np.ndarray


class ManyBodyInteractionEvaluator:
    """Compute the exact MLIP cross interaction by whole-minus-fragments.

    This scalar/force subtraction retains all irreducible many-body terms in
    the whole XW supermolecule; no pair-energy decomposition is assumed.
    """

    def __init__(
        self,
        *,
        calculator: Calculator,
        partition: FragmentPartition,
    ) -> None:
        self.calculator = calculator
        self.partition = partition

    def _evaluate(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        probe = atoms.copy()
        probe.info["charge"] = 0
        probe.info["mult"] = 1
        probe.info["spin"] = 1.0
        probe.calc = self.calculator
        return (
            float(probe.get_potential_energy()),
            np.asarray(probe.get_forces(), dtype=float),
        )

    def evaluate(
        self,
        atoms: Atoms,
        *,
        need_combined: bool = True,
    ) -> InteractionEvaluation:
        self.partition.validate(atoms)
        solute = atoms[list(self.partition.solute_indices)]
        water = atoms[list(self.partition.water_indices)]
        solute_energy, solute_forces = self._evaluate(solute)
        water_energy, water_forces = self._evaluate(water)

        base_forces = np.zeros((len(atoms), 3), dtype=float)
        base_forces[list(self.partition.solute_indices)] = solute_forces
        base_forces[list(self.partition.water_indices)] = water_forces
        base_energy = solute_energy + water_energy

        if need_combined:
            combined_energy, combined_forces = self._evaluate(atoms)
            interaction_energy = combined_energy - base_energy
            interaction_forces = combined_forces - base_forces
        else:
            combined_energy = base_energy
            combined_forces = base_forces.copy()
            interaction_energy = 0.0
            interaction_forces = np.zeros_like(base_forces)
        return InteractionEvaluation(
            base_ev=base_energy,
            combined_ev=combined_energy,
            interaction_ev=interaction_energy,
            base_forces_ev_per_angstrom=base_forces,
            combined_forces_ev_per_angstrom=combined_forces,
            interaction_forces_ev_per_angstrom=interaction_forces,
        )


class SequentialInsertionCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        *,
        evaluator: ManyBodyInteractionEvaluator,
        repulsive_core: GaussianRepulsiveCore,
        state: AlchemicalState,
        restraint: Any | None = None,
        capture_full_basis: bool = False,
    ) -> None:
        super().__init__()
        self.evaluator = evaluator
        self.repulsive_core = repulsive_core
        self.state = state
        self.restraint = restraint
        self.capture_full_basis = bool(capture_full_basis)

    def _evaluate_components(
        self,
        atoms: Atoms,
        *,
        need_combined: bool,
    ) -> tuple[dict[str, Any], np.ndarray]:
        repulsive_scale, full_scale = self.state.scales
        interaction = self.evaluator.evaluate(
            atoms,
            need_combined=need_combined,
        )
        repulsive_energy, repulsive_forces = self.repulsive_core.evaluate(
            atoms,
            self.evaluator.partition,
        )
        restraint_energy = 0.0
        restraint_forces = np.zeros((len(atoms), 3), dtype=float)
        if self.restraint is not None:
            restraint_energy, restraint_forces = self.restraint.evaluate(atoms)
            restraint_energy = float(restraint_energy)
            restraint_forces = np.asarray(restraint_forces, dtype=float)
            if restraint_forces.shape != (len(atoms), 3):
                raise ValueError(
                    "Alchemy restraint forces must have shape (N, 3)."
                )

        total_energy = (
            interaction.base_ev
            + full_scale * interaction.interaction_ev
            + repulsive_scale * repulsive_energy
            + restraint_energy
        )
        total_forces = (
            interaction.base_forces_ev_per_angstrom
            + full_scale * interaction.interaction_forces_ev_per_angstrom
            + repulsive_scale * repulsive_forces
            + restraint_forces
        )
        components = {
            "base_ev": interaction.base_ev,
            "combined_ev": interaction.combined_ev,
            "interaction_ev": interaction.interaction_ev,
            "repulsive_ev": repulsive_energy,
            "restraint_ev": restraint_energy,
            "repulsive_scale": repulsive_scale,
            "full_scale": full_scale,
            "total_ev": total_energy,
            "base_forces_ev_per_angstrom": (
                interaction.base_forces_ev_per_angstrom.copy()
            ),
            "combined_forces_ev_per_angstrom": (
                interaction.combined_forces_ev_per_angstrom.copy()
            ),
            "interaction_forces_ev_per_angstrom": (
                interaction.interaction_forces_ev_per_angstrom.copy()
            ),
            "repulsive_forces_ev_per_angstrom": repulsive_forces.copy(),
        }
        return components, total_forces

    def evaluate_full_basis(self, atoms: Atoms) -> dict[str, Any]:
        """Evaluate the complete cross-state basis at one sampled geometry.

        Decoupled MD steps do not need the combined-system OMOL pass for their
        forces.  MBAR still needs that value at every retained sample, so the
        runner calls this method only at the sampling boundary.
        """

        components, _ = self._evaluate_components(
            atoms,
            need_combined=True,
        )
        return components

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        assert atoms is not None
        _, full_scale = self.state.scales
        components, total_forces = self._evaluate_components(
            atoms,
            need_combined=self.capture_full_basis or full_scale != 0.0,
        )
        self.results = {
            "energy": float(components["total_ev"]),
            "free_energy": float(components["total_ev"]),
            "forces": total_forces,
            "components": components,
        }
