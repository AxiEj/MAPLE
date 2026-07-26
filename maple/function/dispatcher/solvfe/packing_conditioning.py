from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic
from ase.units import kB

from .membership import (
    PERIODIC_IMAGE_LOG_TOLERANCE,
    PERIODIC_IMAGE_MAX_SHELLS,
    SoftCutoffMembership,
    membership_surface_identity_hash,
    smooth_surface_geometry,
    soft_occupancy_weights,
)
from .packing_contract import PackingSchedule
from .protocol import canonical_sha256


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def atom_index_map_hash(
    *,
    role: str,
    indices: tuple[int, ...],
) -> str:
    canonical_indices = _indices(indices, name=f"{role}_indices")
    return canonical_sha256(
        {
            "contract_id": "runtime-atom-index-map-v3",
            "role": role,
            "indices": list(canonical_indices),
        }
    )


def periodic_cell_hash(
    cell_angstrom: np.ndarray,
    pbc: np.ndarray,
) -> str:
    cell = np.asarray(cell_angstrom, dtype=float)
    periodic = np.asarray(pbc, dtype=bool)
    if (
        cell.shape != (3, 3)
        or not np.all(np.isfinite(cell))
        or abs(float(np.linalg.det(cell))) <= 1.0e-12
        or periodic.shape != (3,)
        or not bool(np.all(periodic))
    ):
        raise ValueError(
            "Periodic cell identity requires a finite nonsingular 3D cell."
        )
    return canonical_sha256(
        {
            "contract_id": "periodic-cell-v3",
            "cell_angstrom": cell.tolist(),
            "pbc": periodic.tolist(),
        }
    )


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
class SoftCavityEvaluation:
    energy_ev: float
    forces_ev_per_angstrom: np.ndarray
    hard_empty: bool
    signed_distances_angstrom: np.ndarray
    memberships: np.ndarray
    nonmemberships: np.ndarray
    soft_occupancy_weights: np.ndarray
    log_empty_weight: float


@dataclass(frozen=True)
class GeometryConditionedCavity:
    """Soft empty-shell field attached to a sampled solute configuration.

    The zero-bias ensemble is the independent product of an isolated-solute
    conformational measure and a pure-water measure.  The cavity field is the
    only solute--water coupling in this packing calculation.
    """

    solute_indices: tuple[int, ...]
    membership: SoftCutoffMembership
    temperature_k: float
    conditioning_measure_id: str
    solute_measure_hash: str
    solute_atom_map_hash: str
    active_occupancy_max: int

    def __post_init__(self) -> None:
        solute_indices = _indices(
            self.solute_indices,
            name="solute_indices",
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
        if (
            isinstance(self.active_occupancy_max, (bool, np.bool_))
            or int(self.active_occupancy_max) != self.active_occupancy_max
            or int(self.active_occupancy_max) < 1
        ):
            raise ValueError(
                "active_occupancy_max must include n=0 and at least n=1."
            )
        object.__setattr__(self, "solute_indices", solute_indices)
        object.__setattr__(
            self,
            "active_occupancy_max",
            int(self.active_occupancy_max),
        )

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
                "contract_id": "periodic-membership-boundary-adapter-v5",
                "membership_surface_hash": self.membership_surface_hash,
                "boundary_conditions": "periodic-3d",
                "distance": (
                    "symmetric periodic-image log-sum-exp smooth "
                    "atom-sphere signed distance"
                ),
                "image_log_tolerance": PERIODIC_IMAGE_LOG_TOLERANCE,
                "maximum_image_shells": PERIODIC_IMAGE_MAX_SHELLS,
            }
        )

    @property
    def content_hash(self) -> str:
        return canonical_sha256(
            {
                "contract_id": "geometry-conditioned-soft-cavity-v5",
                "observation_volume_hash": self.observation_volume_hash,
                "membership_surface_hash": self.membership_surface_hash,
                "boundary_adapter_hash": self.boundary_adapter_hash,
                "conditioning_measure_id": self.conditioning_measure_id,
                "temperature_k": float(self.temperature_k),
                "active_occupancy_max": self.active_occupancy_max,
                "potential": "sum_j[-kT ln(1-b_j)]",
            }
        )

    def evaluate(
        self,
        atoms: Atoms,
        *,
        oxygen_indices: tuple[int, ...],
    ) -> SoftCavityEvaluation:
        oxygen_indices = _indices(
            oxygen_indices,
            name="oxygen_indices",
        )
        atom_count = len(atoms)
        if any(
            index >= atom_count
            for index in (*self.solute_indices, *oxygen_indices)
        ):
            raise ValueError(
                "Cavity atom indices must be valid for the supplied atoms."
            )
        if set(self.solute_indices) & set(oxygen_indices):
            raise ValueError(
                "Solute and solvent-oxygen indices must be disjoint."
            )
        pbc = np.asarray(atoms.get_pbc(), dtype=bool)
        cell = np.asarray(atoms.cell.array, dtype=float)
        if (
            not bool(np.all(pbc))
            or cell.shape != (3, 3)
            or not np.all(np.isfinite(cell))
            or abs(float(np.linalg.det(cell))) <= 1.0e-12
        ):
            raise ValueError(
                "Geometry-conditioned packing requires a finite nonsingular "
                "cell and three-dimensional PBC."
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
                "PACKING_RADIUS_MISSING: no membership radius for atomic "
                "number(s) "
                + ", ".join(str(value) for value in missing_radii)
            )
        if any(int(atoms.numbers[index]) != 8 for index in oxygen_indices):
            raise ValueError(
                "oxygen_indices must identify oxygen atoms."
            )
        if self.active_occupancy_max > len(oxygen_indices):
            raise ValueError(
                "active_occupancy_max cannot exceed the number of periodic "
                "solvent molecules."
            )

        solute_positions = np.asarray(
            atoms.positions[list(self.solute_indices)],
            dtype=float,
        )
        oxygen_positions = np.asarray(
            atoms.positions[list(oxygen_indices)],
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
        geometry = smooth_surface_geometry(
            point_positions_angstrom=oxygen_positions,
            center_positions_angstrom=solute_positions,
            center_radii_angstrom=radii,
            surface_smoothing_angstrom=(
                self.membership.surface_smoothing_angstrom
            ),
            cell_angstrom=cell,
            pbc=pbc,
        )
        fields = self.membership.evaluate_signed_distances(
            geometry.signed_distances_angstrom,
            temperature_k=self.temperature_k,
        )

        forces = np.zeros((atom_count, 3), dtype=float)
        for local_oxygen, oxygen_index in enumerate(oxygen_indices):
            center_gradients = geometry.center_gradient_vectors[local_oxygen]
            derivative = float(
                fields.empty_derivative_ev_per_angstrom[local_oxygen]
            )
            oxygen_force = (
                -derivative * np.sum(center_gradients, axis=0)
            )
            forces[oxygen_index] += oxygen_force
            for local_solute, solute_index in enumerate(self.solute_indices):
                forces[solute_index] += (
                    derivative * center_gradients[local_solute]
                )

        energy = float(np.sum(fields.empty_potential_ev))
        log_empty_weight = -energy / (kB * float(self.temperature_k))
        occupancy = soft_occupancy_weights(
            fields.membership,
            active_occupancy_max=self.active_occupancy_max,
        )
        forces.setflags(write=False)
        return SoftCavityEvaluation(
            energy_ev=energy,
            forces_ev_per_angstrom=forces,
            hard_empty=bool(
                np.all(
                    geometry.signed_distances_angstrom
                    > float(self.membership.lambda_s_angstrom)
                )
            ),
            signed_distances_angstrom=(
                geometry.signed_distances_angstrom
            ),
            memberships=fields.membership,
            nonmemberships=fields.nonmembership,
            soft_occupancy_weights=occupancy,
            log_empty_weight=float(log_empty_weight),
        )


class ProductMeasurePackingCalculator(Calculator):
    """Evaluate ``U_X(q_X) + U_W(R_W) + s U_empty(q_X, R_W)``."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        *,
        solute_calculator: Calculator,
        solvent_calculator: Calculator,
        cavity: GeometryConditionedCavity,
        schedule: PackingSchedule,
        state_index: int,
        solvent_indices: tuple[int, ...],
        oxygen_indices: tuple[int, ...],
    ) -> None:
        super().__init__()
        solvent_indices = _indices(
            solvent_indices,
            name="solvent_indices",
        )
        oxygen_indices = _indices(
            oxygen_indices,
            name="oxygen_indices",
        )
        if set(cavity.solute_indices) & set(solvent_indices):
            raise ValueError(
                "Solute and solvent atom partitions must be disjoint."
            )
        if not set(oxygen_indices).issubset(solvent_indices):
            raise ValueError(
                "oxygen_indices must be a subset of solvent_indices."
            )
        if (
            isinstance(state_index, (bool, np.bool_))
            or int(state_index) != state_index
            or not 0 <= int(state_index) < len(schedule.states)
        ):
            raise ValueError(
                "state_index must identify one packing schedule state."
            )
        expected_pairs = (
            (
                "conditioning-measure",
                schedule.conditioning_measure_id,
                cavity.conditioning_measure_id,
            ),
            (
                "membership-definition",
                schedule.membership_definition_hash,
                cavity.membership_definition_hash,
            ),
            (
                "observation-volume",
                schedule.observation_volume_hash,
                cavity.observation_volume_hash,
            ),
            (
                "boundary-adapter",
                schedule.boundary_adapter_hash,
                cavity.boundary_adapter_hash,
            ),
            (
                "solute-measure",
                schedule.solute_measure_hash,
                cavity.solute_measure_hash,
            ),
            (
                "solute-atom-map",
                schedule.solute_atom_map_hash,
                cavity.solute_atom_map_hash,
            ),
            (
                "active-occupancy-max",
                schedule.active_occupancy_max,
                cavity.active_occupancy_max,
            ),
        )
        for label, scheduled, observed in expected_pairs:
            if scheduled != observed:
                raise ValueError(
                    f"Packing schedule/cavity {label} mismatch."
                )
        solvent_hamiltonian_hash = getattr(
            solvent_calculator,
            "maple_hamiltonian_hash",
            None,
        )
        if solvent_hamiltonian_hash != schedule.water_hamiltonian_hash:
            raise ValueError(
                "Packing solvent calculator Hamiltonian identity mismatch."
            )
        if schedule.oxygen_atom_map_hash != atom_index_map_hash(
            role="water-oxygen",
            indices=oxygen_indices,
        ):
            raise ValueError(
                "Packing schedule oxygen atom-map mismatch."
            )
        if not math.isclose(
            float(schedule.temperature_k),
            float(cavity.temperature_k),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Packing schedule/cavity temperature mismatch."
            )

        self.solute_calculator = solute_calculator
        self.solvent_calculator = solvent_calculator
        self.cavity = cavity
        self.schedule = schedule
        self.state_index = int(state_index)
        self.solvent_indices = solvent_indices
        self.oxygen_indices = oxygen_indices

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError(
                "Product-measure packing requires an Atoms object."
            )
        assigned = set(self.cavity.solute_indices) | set(
            self.solvent_indices
        )
        expected = set(range(len(atoms)))
        if assigned != expected:
            missing = sorted(expected - assigned)
            extra = sorted(assigned - expected)
            raise RuntimeError(
                "PACKING_PARTITION_INVALID: solute and solvent partitions "
                f"must cover every atom exactly once; missing={missing}, "
                f"extra={extra}."
            )
        pbc = np.asarray(atoms.get_pbc(), dtype=bool)
        if not bool(np.all(pbc)):
            raise RuntimeError(
                "PACKING_BOUNDARY_INVALID: product-measure packing "
                "requires three-dimensional PBC."
            )
        observed_cell_hash = periodic_cell_hash(
            np.asarray(atoms.cell.array, dtype=float),
            pbc,
        )
        if observed_cell_hash != self.schedule.cell_hash:
            raise RuntimeError(
                "PACKING_CELL_IDENTITY_MISMATCH: runtime cell differs from "
                "the scheduled periodic cell."
            )

        solute = atoms[list(self.cavity.solute_indices)].copy()
        anchor = np.asarray(
            atoms.positions[self.cavity.solute_indices[0]],
            dtype=float,
        )
        displacements = (
            np.asarray(
                atoms.positions[list(self.cavity.solute_indices)],
                dtype=float,
            )
            - anchor
        )
        mic_vectors, _ = find_mic(
            displacements,
            np.asarray(atoms.cell.array, dtype=float),
            pbc=pbc,
        )
        solute.positions[:] = anchor + mic_vectors
        solute.set_pbc(False)
        solute.calc = self.solute_calculator
        solvent = atoms[list(self.solvent_indices)].copy()
        solvent.calc = self.solvent_calculator

        solute_energy = float(solute.get_potential_energy())
        solvent_energy = float(solvent.get_potential_energy())
        solute_forces = np.asarray(solute.get_forces(), dtype=float)
        solvent_forces = np.asarray(solvent.get_forces(), dtype=float)
        bias = self.cavity.evaluate(
            atoms,
            oxygen_indices=self.oxygen_indices,
        )
        state = self.schedule.states[self.state_index]
        scale = float(state.bias_scale)

        forces = np.zeros((len(atoms), 3), dtype=float)
        forces[list(self.cavity.solute_indices)] += solute_forces
        forces[list(self.solvent_indices)] += solvent_forces
        forces += scale * bias.forces_ev_per_angstrom
        energy = solute_energy + solvent_energy + scale * bias.energy_ev
        if (
            not math.isfinite(energy)
            or forces.shape != (len(atoms), 3)
            or not np.all(np.isfinite(forces))
        ):
            raise RuntimeError(
                "PACKING_PRODUCT_MEASURE_NONFINITE: energy or forces "
                "are invalid."
            )

        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
            "components": {
                "solute_ev": solute_energy,
                "solvent_ev": solvent_energy,
                "full_bias_ev": bias.energy_ev,
                "bias_scale": scale,
                "total_ev": energy,
                "cross_solute_solvent_energy_ev": 0.0,
                "hard_empty_diagnostic": bias.hard_empty,
                "log_empty_weight": bias.log_empty_weight,
                "soft_occupancy_weights": bias.soft_occupancy_weights,
                "cavity_hash": self.cavity.content_hash,
                "observation_volume_hash": (
                    self.cavity.observation_volume_hash
                ),
                "membership_definition_hash": (
                    self.cavity.membership_definition_hash
                ),
                "conditioning_measure_id": (
                    self.cavity.conditioning_measure_id
                ),
                "schedule_hash": self.schedule.content_hash,
                "water_hamiltonian_hash": (
                    self.schedule.water_hamiltonian_hash
                ),
                "oxygen_atom_map_hash": (
                    self.schedule.oxygen_atom_map_hash
                ),
                "cell_hash": self.schedule.cell_hash,
                "state_label": state.label,
            },
        }


__all__ = [
    "GeometryConditionedCavity",
    "ProductMeasurePackingCalculator",
    "SoftCavityEvaluation",
    "atom_index_map_hash",
    "periodic_cell_hash",
]
