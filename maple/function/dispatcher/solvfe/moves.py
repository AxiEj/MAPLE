from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from ase import Atoms, units

from .alchemy import (
    AlchemicalState,
    FragmentPartition,
    GaussianRepulsiveCore,
    ManyBodyInteractionEvaluator,
)
from .protocol import canonical_sha256


@dataclass(frozen=True)
class RigidBodyMoveConfig:
    """Symmetric rigid-water Metropolis moves for relative shell mixing."""

    water_groups: tuple[tuple[int, ...], ...]
    equilibration_attempts: int
    attempts_per_sample: int
    translation_step_angstrom: float
    rotation_step_radians: float
    translation_probability: float = 0.5

    def __post_init__(self) -> None:
        if (
            not self.water_groups
            or any(not group for group in self.water_groups)
            or any(
                isinstance(index, (bool, np.bool_))
                or int(index) != index
                or int(index) < 0
                for group in self.water_groups
                for index in group
            )
        ):
            raise ValueError(
                "Rigid-body water groups require non-empty nonnegative "
                "integer atom indices."
            )
        flattened = tuple(
            int(index) for group in self.water_groups for index in group
        )
        if len(set(flattened)) != len(flattened):
            raise ValueError("Rigid-body water groups overlap.")
        for name, value in (
            ("equilibration_attempts", self.equilibration_attempts),
            ("attempts_per_sample", self.attempts_per_sample),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer.")
        for name, value in (
            ("translation_step_angstrom", self.translation_step_angstrom),
            ("rotation_step_radians", self.rotation_step_radians),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        probability = float(self.translation_probability)
        if (
            isinstance(self.translation_probability, (bool, np.bool_))
            or not math.isfinite(probability)
            or not 0.0 < probability < 1.0
        ):
            raise ValueError(
                "translation_probability must be strictly between zero and one."
            )

    @property
    def content_hash(self) -> str:
        return canonical_sha256(
            {
                "contract_id": "rigid-water-metropolis-v1",
                "water_groups": [list(group) for group in self.water_groups],
                "equilibration_attempts": int(self.equilibration_attempts),
                "attempts_per_sample": int(self.attempts_per_sample),
                "translation_step_angstrom": float(
                    self.translation_step_angstrom
                ),
                "rotation_step_radians": float(
                    self.rotation_step_radians
                ),
                "translation_probability": float(
                    self.translation_probability
                ),
                "proposal_contract": (
                    "symmetric uniform-ball translation or symmetric "
                    "axis-angle rotation about the selected water COM"
                ),
            }
        )


@dataclass(frozen=True)
class MoveStatistics:
    attempts: int
    accepted: int
    translation_attempts: int
    translation_accepted: int
    rotation_attempts: int
    rotation_accepted: int

    @property
    def acceptance_fraction(self) -> float:
        return 0.0 if self.attempts == 0 else self.accepted / self.attempts

    def as_dict(self) -> dict[str, float | int]:
        return {
            "attempts": self.attempts,
            "accepted": self.accepted,
            "acceptance_fraction": self.acceptance_fraction,
            "translation_attempts": self.translation_attempts,
            "translation_accepted": self.translation_accepted,
            "rotation_attempts": self.rotation_attempts,
            "rotation_accepted": self.rotation_accepted,
        }


class RigidBodyMetropolis:
    """Metropolis kernel that preserves each explicit water's geometry.

    Translation and rotation proposals are symmetric, so the acceptance ratio
    contains only the target Hamiltonian difference.  Fragment internal energy
    cancels exactly for a rigid move; only the cross interaction, auxiliary
    core, and shell restraint need evaluation.
    """

    def __init__(
        self,
        *,
        evaluator: ManyBodyInteractionEvaluator,
        repulsive_core: GaussianRepulsiveCore,
        restraint,
        state: AlchemicalState,
        temperature_k: float,
        config: RigidBodyMoveConfig,
    ) -> None:
        self.evaluator = evaluator
        self.repulsive_core = repulsive_core
        self.restraint = restraint
        self.state = state
        self.temperature_k = float(temperature_k)
        self.config = config
        self._validate_partition(evaluator.partition)
        if (
            not math.isfinite(self.temperature_k)
            or self.temperature_k <= 0.0
        ):
            raise ValueError("Rigid-body Metropolis temperature must be positive.")
        self.beta_ev_inverse = 1.0 / (units.kB * self.temperature_k)

    def _validate_partition(self, partition: FragmentPartition) -> None:
        movable = {
            int(index)
            for group in self.config.water_groups
            for index in group
        }
        if movable != set(partition.water_indices):
            raise ValueError(
                "Rigid-body water groups must partition all alchemical water "
                "indices exactly."
            )

    @staticmethod
    def _unit_vector(rng: np.random.Generator) -> np.ndarray:
        while True:
            vector = rng.normal(size=3)
            norm = float(np.linalg.norm(vector))
            if norm > 0.0:
                return vector / norm

    def _translated(
        self,
        atoms: Atoms,
        group: tuple[int, ...],
        rng: np.random.Generator,
    ) -> Atoms:
        proposal = atoms.copy()
        direction = self._unit_vector(rng)
        radius = float(self.config.translation_step_angstrom) * (
            float(rng.random()) ** (1.0 / 3.0)
        )
        proposal.positions[list(group)] += radius * direction
        return proposal

    def _rotated(
        self,
        atoms: Atoms,
        group: tuple[int, ...],
        rng: np.random.Generator,
    ) -> Atoms:
        proposal = atoms.copy()
        indices = list(group)
        coordinates = np.asarray(proposal.positions[indices], dtype=float)
        masses = np.asarray(proposal.get_masses()[indices], dtype=float)
        center = np.average(coordinates, axis=0, weights=masses)
        axis = self._unit_vector(rng)
        angle = float(rng.uniform(
            -float(self.config.rotation_step_radians),
            float(self.config.rotation_step_radians),
        ))
        cross = np.asarray(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        rotation = (
            np.eye(3) * math.cos(angle)
            + (1.0 - math.cos(angle)) * np.outer(axis, axis)
            + math.sin(angle) * cross
        )
        proposal.positions[indices] = (
            (coordinates - center) @ rotation.T + center
        )
        return proposal

    def _relative_energy_ev(self, atoms: Atoms) -> float:
        repulsive_scale, full_scale = self.state.scales
        interaction_ev = 0.0
        if full_scale != 0.0:
            interaction_ev = self.evaluator.evaluate(
                atoms,
                need_combined=True,
            ).interaction_ev
        repulsive_ev = self.repulsive_core.evaluate(
            atoms,
            self.evaluator.partition,
        )[0]
        restraint_ev = float(self.restraint.evaluate(atoms)[0])
        value = (
            full_scale * interaction_ev
            + repulsive_scale * repulsive_ev
            + restraint_ev
        )
        if not math.isfinite(value):
            raise RuntimeError(
                "RIGID_MOVE_NONFINITE: relative move energy is non-finite."
            )
        return value

    def run(
        self,
        atoms: Atoms,
        *,
        attempts: int,
        rng: np.random.Generator,
    ) -> MoveStatistics:
        if (
            isinstance(attempts, (bool, np.bool_))
            or int(attempts) != attempts
            or int(attempts) < 0
        ):
            raise ValueError("Rigid move attempts must be nonnegative.")
        attempts = int(attempts)
        if attempts == 0:
            return MoveStatistics(0, 0, 0, 0, 0, 0)

        current_energy = self._relative_energy_ev(atoms)
        accepted = 0
        translation_attempts = 0
        translation_accepted = 0
        rotation_attempts = 0
        rotation_accepted = 0
        for _ in range(attempts):
            group = self.config.water_groups[
                int(rng.integers(len(self.config.water_groups)))
            ]
            translation = (
                float(rng.random()) < self.config.translation_probability
            )
            if translation:
                translation_attempts += 1
                proposal = self._translated(atoms, group, rng)
            else:
                rotation_attempts += 1
                proposal = self._rotated(atoms, group, rng)
            proposal_energy = self._relative_energy_ev(proposal)
            log_acceptance = -self.beta_ev_inverse * (
                proposal_energy - current_energy
            )
            if log_acceptance >= 0.0 or math.log(float(rng.random())) < (
                log_acceptance
            ):
                atoms.set_positions(proposal.positions)
                current_energy = proposal_energy
                accepted += 1
                if translation:
                    translation_accepted += 1
                else:
                    rotation_accepted += 1

        return MoveStatistics(
            attempts=attempts,
            accepted=accepted,
            translation_attempts=translation_attempts,
            translation_accepted=translation_accepted,
            rotation_attempts=rotation_attempts,
            rotation_accepted=rotation_accepted,
        )
