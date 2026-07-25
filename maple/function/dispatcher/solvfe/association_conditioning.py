from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.units import kB
from scipy.stats import qmc

from .membership import (
    SoftCutoffMembership,
    membership_surface_identity_hash,
    nearest_surface_geometry,
)
from .protocol import canonical_sha256


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _indices(
    values: tuple[int, ...],
    *,
    name: str,
) -> tuple[int, ...]:
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise ValueError(f"{name} must contain integer atom indices.")
    result = tuple(int(value) for value in values)
    if (
        not result
        or len(set(result)) != len(result)
        or any(value < 0 for value in result)
    ):
        raise ValueError(
            f"{name} must contain unique non-negative atom indices."
        )
    return result


@dataclass(frozen=True)
class SoftAssociationEvaluation:
    energy_ev: float
    forces_ev_per_angstrom: np.ndarray
    signed_distances_angstrom: np.ndarray
    memberships: np.ndarray
    log_member_weight: float


@dataclass(frozen=True)
class SoftEffectiveVolumeEstimate:
    volume_angstrom3: float
    replicate_standard_error_angstrom3: float
    tail_upper_bound_angstrom3: float
    replicate_estimates_angstrom3: tuple[float, ...]
    samples_per_replicate: int
    integration_box_angstrom: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    membership_definition_hash: str
    observation_volume_hash: str
    boundary_adapter_hash: str
    solute_geometry_hash: str
    content_hash: str


@dataclass(frozen=True)
class SoftMembershipSurfaceRestraint:
    """Nonperiodic labeled-water association field ``-kT sum ln b_j``."""

    solute_indices: tuple[int, ...]
    water_oxygen_indices: tuple[int, ...]
    water_atom_indices: tuple[int, ...]
    membership: SoftCutoffMembership
    temperature_k: float
    conditioning_measure_id: str
    solute_measure_hash: str
    solute_atom_map_hash: str

    def __post_init__(self) -> None:
        solute_indices = _indices(
            self.solute_indices,
            name="solute_indices",
        )
        oxygen_indices = _indices(
            self.water_oxygen_indices,
            name="water_oxygen_indices",
        )
        water_indices = _indices(
            self.water_atom_indices,
            name="water_atom_indices",
        )
        if set(solute_indices) & set(water_indices):
            raise ValueError(
                "Association solute and water atom indices must be disjoint."
            )
        if not set(oxygen_indices).issubset(water_indices):
            raise ValueError(
                "Every associated oxygen must belong to water_atom_indices."
            )
        if not isinstance(self.membership, SoftCutoffMembership):
            raise ValueError(
                "membership must be a SoftCutoffMembership definition."
            )
        if (
            isinstance(self.temperature_k, (bool, np.bool_))
            or not math.isfinite(float(self.temperature_k))
            or float(self.temperature_k) <= 0.0
        ):
            raise ValueError("temperature_k must be finite and positive.")
        if (
            not isinstance(self.conditioning_measure_id, str)
            or not self.conditioning_measure_id
        ):
            raise ValueError(
                "conditioning_measure_id must be a non-empty string."
            )
        for name in ("solute_measure_hash", "solute_atom_map_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 hash.")

        object.__setattr__(self, "solute_indices", solute_indices)
        object.__setattr__(
            self,
            "water_oxygen_indices",
            oxygen_indices,
        )
        object.__setattr__(self, "water_atom_indices", water_indices)

    @property
    def beta_ev_inverse(self) -> float:
        return 1.0 / (kB * float(self.temperature_k))

    @property
    def membership_definition_hash(self) -> str:
        return self.membership.content_hash

    @property
    def observation_volume_hash(self) -> str:
        return self.membership_surface_hash

    @property
    def membership_surface_hash(self) -> str:
        return membership_surface_identity_hash(
            solute_indices=self.solute_indices,
            solute_atom_map_hash=self.solute_atom_map_hash,
            solute_measure_hash=self.solute_measure_hash,
            membership_definition_hash=self.membership_definition_hash,
        )

    @property
    def boundary_adapter_hash(self) -> str:
        return canonical_sha256(
            {
                "contract_id": "membership-boundary-adapter-v3",
                "membership_surface_hash": self.membership_surface_hash,
                "boundary_conditions": "nonperiodic-cluster",
                "distance": "nonperiodic nearest atom-sphere signed distance",
            }
        )

    @property
    def content_hash(self) -> str:
        return canonical_sha256(
            {
                "contract_id": "soft-membership-association-restraint-v3",
                "observation_volume_hash": self.observation_volume_hash,
                "membership_surface_hash": self.membership_surface_hash,
                "boundary_adapter_hash": self.boundary_adapter_hash,
                "conditioning_measure_id": self.conditioning_measure_id,
                "water_oxygen_indices": list(self.water_oxygen_indices),
                "water_atom_indices": list(self.water_atom_indices),
                "temperature_k": float(self.temperature_k),
                "potential": "sum_j[-kT ln b_j]",
                "boundary_conditions": "nonperiodic",
            }
        )

    def _validate_atoms(self, atoms: Atoms) -> None:
        if bool(np.any(atoms.get_pbc())):
            raise ValueError(
                "Soft-membership association is strictly nonperiodic."
            )
        indices = (
            self.solute_indices
            + self.water_oxygen_indices
            + self.water_atom_indices
        )
        if min(indices) < 0 or max(indices) >= len(atoms):
            raise ValueError(
                "ATOM_LIST_MISMATCH: association indices do not match atoms."
            )
        missing_radii = sorted(
            {
                int(atoms.numbers[index])
                for index in self.solute_indices
                if int(atoms.numbers[index])
                not in self.membership.vdw_radii_angstrom
            }
        )
        if missing_radii:
            raise ValueError(
                "ASSOCIATION_RADIUS_MISSING: no membership radius for atomic "
                "number(s) "
                + ", ".join(str(value) for value in missing_radii)
            )
        if any(
            int(atoms.numbers[index]) != 8
            for index in self.water_oxygen_indices
        ):
            raise ValueError(
                "water_oxygen_indices must identify oxygen atoms."
            )

    def _solute_geometry(
        self,
        atoms: Atoms,
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(
            atoms.positions[list(self.solute_indices)],
            dtype=float,
        )
        radii = np.asarray(
            [
                self.membership.vdw_radii_angstrom[
                    int(atoms.numbers[index])
                ]
                for index in self.solute_indices
            ],
            dtype=float,
        )
        return positions, radii

    def evaluate_details(self, atoms: Atoms) -> SoftAssociationEvaluation:
        self._validate_atoms(atoms)
        solute_positions, radii = self._solute_geometry(atoms)
        oxygen_positions = np.asarray(
            atoms.positions[list(self.water_oxygen_indices)],
            dtype=float,
        )
        geometry = nearest_surface_geometry(
            point_positions_angstrom=oxygen_positions,
            center_positions_angstrom=solute_positions,
            center_radii_angstrom=radii,
        )
        fields = self.membership.evaluate_signed_distances(
            geometry.signed_distances_angstrom,
            temperature_k=self.temperature_k,
        )

        forces = np.zeros((len(atoms), 3), dtype=float)
        for local_oxygen, oxygen_index in enumerate(
            self.water_oxygen_indices
        ):
            nearest_local = int(
                geometry.nearest_center_indices[local_oxygen]
            )
            solute_index = self.solute_indices[nearest_local]
            oxygen_force = (
                -fields.member_derivative_ev_per_angstrom[local_oxygen]
                * geometry.unit_vectors_center_to_point[local_oxygen]
            )
            forces[oxygen_index] += oxygen_force
            forces[solute_index] -= oxygen_force

        energy = float(np.sum(fields.member_potential_ev))
        forces.setflags(write=False)
        return SoftAssociationEvaluation(
            energy_ev=energy,
            forces_ev_per_angstrom=forces,
            signed_distances_angstrom=(
                geometry.signed_distances_angstrom
            ),
            memberships=fields.membership,
            log_member_weight=float(
                -energy * self.beta_ev_inverse
            ),
        )

    def evaluate(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        result = self.evaluate_details(atoms)
        return result.energy_ev, result.forces_ev_per_angstrom

    def effective_translational_volume(
        self,
        atoms: Atoms,
        *,
        sobol_power: int = 16,
        replicates: int = 8,
        seed: int = 20260725,
        tail_log_tolerance: float = 32.0,
        chunk_size: int = 65536,
    ) -> SoftEffectiveVolumeEstimate:
        """Integrate ``b[d(r)]`` for one labeled decoupled water oxygen."""

        self._validate_atoms(atoms)
        for name, value, minimum in (
            ("sobol_power", sobol_power, 4),
            ("replicates", replicates, 2),
            ("chunk_size", chunk_size, 1),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) < minimum
            ):
                raise ValueError(
                    f"{name} must be an integer of at least {minimum}."
                )
        if (
            isinstance(tail_log_tolerance, (bool, np.bool_))
            or not math.isfinite(float(tail_log_tolerance))
            or float(tail_log_tolerance) <= 0.0
        ):
            raise ValueError(
                "tail_log_tolerance must be finite and positive."
            )

        solute_positions, radii = self._solute_geometry(atoms)
        hard_radii = radii + float(self.membership.lambda_s_angstrom)
        tail_extent = (
            float(tail_log_tolerance)
            * float(self.membership.softness_angstrom)
        )
        lower = np.min(
            solute_positions - hard_radii[:, None],
            axis=0,
        ) - tail_extent
        upper = np.max(
            solute_positions + hard_radii[:, None],
            axis=0,
        ) + tail_extent
        box_volume = float(np.prod(upper - lower))

        estimates: list[float] = []
        for replicate in range(int(replicates)):
            engine = qmc.Sobol(
                d=3,
                scramble=True,
                seed=int(seed) + replicate,
            )
            points = qmc.scale(
                engine.random_base2(int(sobol_power)),
                lower,
                upper,
            )
            weight_sum = 0.0
            for start in range(0, len(points), int(chunk_size)):
                block = points[start : start + int(chunk_size)]
                distances = np.linalg.norm(
                    block[:, None, :] - solute_positions[None, :, :],
                    axis=2,
                )
                signed = np.min(
                    distances - radii[None, :],
                    axis=1,
                )
                weights = self.membership.evaluate_signed_distances(
                    signed,
                    temperature_k=self.temperature_k,
                ).membership
                weight_sum += float(np.sum(weights))
            estimates.append(box_volume * weight_sum / len(points))

        estimate_array = np.asarray(estimates, dtype=float)
        standard_error = float(
            np.std(estimate_array, ddof=1) / math.sqrt(len(estimate_array))
        )
        softness = float(self.membership.softness_angstrom)
        tail_exponential = math.exp(
            -float(tail_log_tolerance)
        )
        tail_bound = float(
            sum(
                4.0
                * math.pi
                * tail_exponential
                * (
                    softness * (radius + tail_extent) ** 2
                    + 2.0
                    * softness**2
                    * (radius + tail_extent)
                    + 2.0 * softness**3
                )
                for radius in hard_radii
            )
        )
        solute_geometry_hash = canonical_sha256(
            {
                "atomic_numbers": [
                    int(atoms.numbers[index])
                    for index in self.solute_indices
                ],
                "positions_angstrom": solute_positions.tolist(),
            }
        )
        preimage = {
            "contract_id": "soft-membership-effective-volume-v3",
            "membership_definition_hash": self.membership_definition_hash,
            "observation_volume_hash": self.observation_volume_hash,
            "boundary_adapter_hash": self.boundary_adapter_hash,
            "solute_geometry_hash": solute_geometry_hash,
            "sobol_power": int(sobol_power),
            "replicates": int(replicates),
            "seed": int(seed),
            "tail_log_tolerance": float(tail_log_tolerance),
            "replicate_estimates_angstrom3": [
                float(value) for value in estimate_array
            ],
            "tail_upper_bound_angstrom3": tail_bound,
        }
        return SoftEffectiveVolumeEstimate(
            volume_angstrom3=float(np.mean(estimate_array)),
            replicate_standard_error_angstrom3=standard_error,
            tail_upper_bound_angstrom3=tail_bound,
            replicate_estimates_angstrom3=tuple(
                float(value) for value in estimate_array
            ),
            samples_per_replicate=2 ** int(sobol_power),
            integration_box_angstrom=(
                tuple(float(value) for value in lower),
                tuple(float(value) for value in upper),
            ),
            membership_definition_hash=self.membership_definition_hash,
            observation_volume_hash=self.observation_volume_hash,
            boundary_adapter_hash=self.boundary_adapter_hash,
            solute_geometry_hash=solute_geometry_hash,
            content_hash=canonical_sha256(preimage),
        )


__all__ = [
    "SoftAssociationEvaluation",
    "SoftEffectiveVolumeEstimate",
    "SoftMembershipSurfaceRestraint",
]
