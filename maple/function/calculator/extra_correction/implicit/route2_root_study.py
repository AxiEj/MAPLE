"""Predeclared multi-start fixed-point agreement evidence for Route 2.

Convergence from one damped SCF trajectory is not evidence that the selected
root is locally unique.  This module compares independently converged starts
at one geometry and one declared energy ledger.  It deliberately reports a
local numerical agreement only; it is not a global bifurcation proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Mapping, Protocol

import numpy as np


NOMINAL_ROUTE2_SCF_REASON = "nominal-density-and-energy-v1"


class _RootState(Protocol):
    """Minimal immutable coupled-state surface required for an agreement test."""

    atomic_numbers: np.ndarray
    positions_angstrom: np.ndarray
    density_coefficients: np.ndarray
    initial_density_label: str
    initial_density_sha256: str
    history: tuple
    scf_convergence: Mapping[str, object]


@dataclass(frozen=True)
class Route2MultiStartRootPolicy:
    """Predeclared tolerances for a local force-admission root study."""

    required_start_count: int = 3
    maximum_pairwise_density_difference: float = 1.0e-9
    maximum_pairwise_energy_difference_ev: float = 1.0e-10

    def __post_init__(self) -> None:
        if self.required_start_count < 2:
            raise ValueError("A root-agreement study requires at least two starts.")
        for value in (
            self.maximum_pairwise_density_difference,
            self.maximum_pairwise_energy_difference_ev,
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("Root-agreement tolerances must be finite and positive.")

    def as_dict(self) -> dict[str, int | float]:
        return {
            "required_start_count": self.required_start_count,
            "maximum_pairwise_density_difference": (
                self.maximum_pairwise_density_difference
            ),
            "maximum_pairwise_energy_difference_ev": (
                self.maximum_pairwise_energy_difference_ev
            ),
        }


@dataclass(frozen=True)
class Route2RootStudyMember:
    """A compact, immutable audit row for one independently seeded solve."""

    start_key: str
    initial_density_label: str
    initial_density_sha256: str
    converged_nominally: bool
    iterations: int
    declared_scalar_energy_ev: float

    def __post_init__(self) -> None:
        if not self.start_key.strip() or not self.initial_density_label.strip():
            raise ValueError("Root-study start identifiers must be non-empty.")
        if len(self.initial_density_sha256) != 64:
            raise ValueError("Root-study initial-density digest must be canonical.")
        if self.iterations <= 0:
            raise ValueError("Root-study iterations must be positive.")
        if not math.isfinite(self.declared_scalar_energy_ev):
            raise ValueError("Root-study scalar energy must be finite.")

    def as_dict(self) -> dict[str, object]:
        return {
            "start_key": self.start_key,
            "initial_density_label": self.initial_density_label,
            "initial_density_sha256": self.initial_density_sha256,
            "converged_nominally": self.converged_nominally,
            "iterations": self.iterations,
            "declared_scalar_energy_ev": self.declared_scalar_energy_ev,
        }


@dataclass(frozen=True)
class Route2MultiStartRootAgreement:
    """Machine-readable evidence that multiple starts reached one local root."""

    policy: Route2MultiStartRootPolicy
    members: tuple[Route2RootStudyMember, ...]
    maximum_pairwise_density_difference: float | None
    maximum_pairwise_energy_difference_ev: float | None
    agreed: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.members) == 0:
            raise ValueError("A root-study result must retain its members.")
        for value in (
            self.maximum_pairwise_density_difference,
            self.maximum_pairwise_energy_difference_ev,
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError("Root-study pairwise metrics must be finite.")
        if self.agreed != (not self.failure_reasons):
            raise ValueError("Root-study decision must match its failure reasons.")

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy.as_dict(),
            "members": [member.as_dict() for member in self.members],
            "maximum_pairwise_density_difference": (
                self.maximum_pairwise_density_difference
            ),
            "maximum_pairwise_energy_difference_ev": (
                self.maximum_pairwise_energy_difference_ev
            ),
            "agreed": self.agreed,
            "failure_reasons": list(self.failure_reasons),
            "scope": (
                "same-geometry local multi-start agreement; not a global "
                "fixed-point uniqueness proof"
            ),
        }


def _validated_energy(energy_ev: float, *, start_key: str) -> float:
    value = float(energy_ev)
    if not math.isfinite(value):
        raise ValueError(f"{start_key}: declared scalar energy must be finite.")
    return value


def evaluate_route2_multi_start_root_agreement(
    states: Mapping[str, _RootState],
    *,
    declared_scalar_energies_ev: Mapping[str, float],
    policy: Route2MultiStartRootPolicy = Route2MultiStartRootPolicy(),
) -> Route2MultiStartRootAgreement:
    """Compare independently seeded nominal roots at one nuclear geometry.

    ``states`` must contain every energy key exactly once.  A state carries its
    actual projected seed label/digest, which prevents a warm start from being
    silently counted as an independent gas start.
    """

    state_items = tuple(states.items())
    state_keys = tuple(key for key, _state in state_items)
    if len(state_keys) != len(set(state_keys)):
        raise ValueError("Root-study start keys must be unique.")
    if set(state_keys) != set(declared_scalar_energies_ev):
        raise ValueError("Root-study states and declared scalar energies differ.")

    members: list[Route2RootStudyMember] = []
    failures: list[str] = []
    densities: dict[str, np.ndarray] = {}
    energies: dict[str, float] = {}
    reference_numbers: np.ndarray | None = None
    reference_positions: np.ndarray | None = None
    seen_seed_digests: set[str] = set()

    for key, state in state_items:
        start_key = str(key)
        if not start_key.strip():
            raise ValueError("Root-study start keys must be non-empty strings.")
        numbers = np.asarray(state.atomic_numbers, dtype=int)
        positions = np.asarray(state.positions_angstrom, dtype=float)
        density = np.asarray(state.density_coefficients, dtype=float)
        if numbers.ndim != 1 or numbers.size == 0:
            raise ValueError(f"{start_key}: atomic numbers are invalid.")
        if positions.shape != (numbers.size, 3) or not np.all(np.isfinite(positions)):
            raise ValueError(f"{start_key}: positions are invalid.")
        if density.shape != (numbers.size, 4) or not np.all(np.isfinite(density)):
            raise ValueError(f"{start_key}: root density is invalid.")
        if reference_numbers is None:
            reference_numbers = np.array(numbers, copy=True)
            reference_positions = np.array(positions, copy=True)
        elif not np.array_equal(numbers, reference_numbers) or not np.array_equal(
            positions, reference_positions
        ):
            failures.append("multi-start-states-do-not-share-one-geometry")
        reason = str(state.scf_convergence.get("reason", ""))
        converged_nominally = reason == NOMINAL_ROUTE2_SCF_REASON
        if not converged_nominally:
            failures.append(f"{start_key}:root-is-not-nominal")
        digest = str(state.initial_density_sha256)
        if digest in seen_seed_digests:
            failures.append("multi-start-seeds-are-not-distinct")
        seen_seed_digests.add(digest)
        energy = _validated_energy(declared_scalar_energies_ev[start_key], start_key=start_key)
        members.append(
            Route2RootStudyMember(
                start_key=start_key,
                initial_density_label=str(state.initial_density_label),
                initial_density_sha256=digest,
                converged_nominally=converged_nominally,
                iterations=len(state.history),
                declared_scalar_energy_ev=energy,
            )
        )
        densities[start_key] = np.array(density, copy=True)
        energies[start_key] = energy

    if len(state_items) < policy.required_start_count:
        failures.append("insufficient-independent-root-starts")

    density_differences: list[float] = []
    energy_differences: list[float] = []
    for left, right in combinations(state_keys, 2):
        density_differences.append(
            float(np.max(np.abs(densities[left] - densities[right])))
        )
        energy_differences.append(abs(energies[left] - energies[right]))
    maximum_density = max(density_differences, default=None)
    maximum_energy = max(energy_differences, default=None)
    if (
        maximum_density is not None
        and maximum_density > policy.maximum_pairwise_density_difference
    ):
        failures.append("multi-start-root-density-disagreement")
    if (
        maximum_energy is not None
        and maximum_energy > policy.maximum_pairwise_energy_difference_ev
    ):
        failures.append("multi-start-root-energy-disagreement")

    # Keep the serialized evidence concise and deterministic even when more
    # than one row detects the same structural issue.
    unique_failures = tuple(dict.fromkeys(failures))
    return Route2MultiStartRootAgreement(
        policy=policy,
        members=tuple(members),
        maximum_pairwise_density_difference=maximum_density,
        maximum_pairwise_energy_difference_ev=maximum_energy,
        agreed=not unique_failures,
        failure_reasons=unique_failures,
    )


__all__ = [
    "NOMINAL_ROUTE2_SCF_REASON",
    "Route2MultiStartRootAgreement",
    "Route2MultiStartRootPolicy",
    "Route2RootStudyMember",
    "evaluate_route2_multi_start_root_agreement",
]
