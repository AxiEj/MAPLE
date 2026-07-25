from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic

from .protocol import canonical_sha256


@dataclass(frozen=True)
class CavityBiasEvaluation:
    energy_ev: float
    forces_ev_per_angstrom: np.ndarray
    empty: bool
    signed_distances_angstrom: np.ndarray
    penetration_angstrom: np.ndarray


@dataclass(frozen=True)
class GhostObservationCavity:
    """Fixed-geometry soft cavity for a pure-water packing calculation.

    The centers are external ghost sites and are not passed to the water
    Hamiltonian. This object implements one conditional solute geometry only;
    a flexible-solute packing calculation must average over a separately
    frozen solute measure rather than silently treating one geometry as exact.
    """

    centers_angstrom: np.ndarray
    radii_angstrom: np.ndarray
    lambda_s_angstrom: float
    force_constant_ev_per_angstrom2: float
    measure_id: str

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers_angstrom, dtype=float)
        radii = np.asarray(self.radii_angstrom, dtype=float)
        if (
            centers.ndim != 2
            or centers.shape[1:] != (3,)
            or len(centers) == 0
            or radii.shape != (len(centers),)
            or not np.all(np.isfinite(centers))
            or not np.all(np.isfinite(radii))
            or np.any(radii <= 0.0)
        ):
            raise ValueError(
                "Ghost cavity centers/radii must be finite arrays with "
                "shapes (N, 3) and (N,), and radii must be positive."
            )
        for name, value, allow_zero in (
            ("lambda_s_angstrom", self.lambda_s_angstrom, True),
            (
                "force_constant_ev_per_angstrom2",
                self.force_constant_ev_per_angstrom2,
                False,
            ),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not math.isfinite(float(value))
                or (float(value) < 0.0 if allow_zero else float(value) <= 0.0)
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be finite and {qualifier}.")
        if not isinstance(self.measure_id, str) or not self.measure_id:
            raise ValueError("measure_id must be a non-empty string.")

        immutable_centers = np.array(centers, copy=True, order="C")
        immutable_radii = np.array(radii, copy=True, order="C")
        immutable_centers.setflags(write=False)
        immutable_radii.setflags(write=False)
        object.__setattr__(self, "centers_angstrom", immutable_centers)
        object.__setattr__(self, "radii_angstrom", immutable_radii)

    @property
    def content_hash(self) -> str:
        return canonical_sha256(
            {
                "contract_id": "fixed-ghost-soft-cavity-v1",
                "centers_angstrom": self.centers_angstrom.tolist(),
                "radii_angstrom": self.radii_angstrom.tolist(),
                "lambda_s_angstrom": float(self.lambda_s_angstrom),
                "force_constant_ev_per_angstrom2": float(
                    self.force_constant_ev_per_angstrom2
                ),
                "measure_id": self.measure_id,
                "potential": (
                    "0.5*k*sum_j(max(0,lambda_s-min_i("
                    "|O_j-R_i|_MIC-r_i))**2)"
                ),
            }
        )

    def evaluate(
        self,
        atoms: Atoms,
        *,
        oxygen_indices: tuple[int, ...],
    ) -> CavityBiasEvaluation:
        indices = tuple(int(value) for value in oxygen_indices)
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(index < 0 or index >= len(atoms) for index in indices)
        ):
            raise ValueError(
                "oxygen_indices must contain unique valid atom indices."
            )
        pbc = np.asarray(atoms.get_pbc(), dtype=bool)
        if not bool(np.all(pbc)):
            raise ValueError(
                "Pure-water packing bias requires three-dimensional PBC."
            )
        cell = np.asarray(atoms.cell.array, dtype=float)
        if (
            cell.shape != (3, 3)
            or not np.all(np.isfinite(cell))
            or abs(float(np.linalg.det(cell))) <= 1.0e-12
        ):
            raise ValueError(
                "Pure-water packing bias requires a finite nonsingular cell."
            )

        forces = np.zeros((len(atoms), 3), dtype=float)
        signed_distances: list[float] = []
        penetrations: list[float] = []
        energy = 0.0
        force_constant = float(self.force_constant_ev_per_angstrom2)
        for oxygen_index in indices:
            displacements = (
                np.asarray(atoms.positions[oxygen_index], dtype=float)
                - self.centers_angstrom
            )
            mic_vectors, distances = find_mic(
                displacements,
                cell,
                pbc=pbc,
            )
            signed_by_center = distances - self.radii_angstrom
            nearest = int(np.argmin(signed_by_center))
            signed_distance = float(signed_by_center[nearest])
            penetration = max(
                0.0,
                float(self.lambda_s_angstrom) - signed_distance,
            )
            signed_distances.append(signed_distance)
            penetrations.append(penetration)
            if penetration == 0.0:
                continue
            distance = float(distances[nearest])
            if distance <= 1.0e-12:
                raise RuntimeError(
                    "PACKING_BIAS_SINGULAR: water oxygen coincides with a "
                    "ghost cavity center."
                )
            energy += 0.5 * force_constant * penetration**2
            forces[oxygen_index] += (
                force_constant
                * penetration
                * np.asarray(mic_vectors[nearest], dtype=float)
                / distance
            )

        signed = np.asarray(signed_distances, dtype=float)
        penetration_array = np.asarray(penetrations, dtype=float)
        return CavityBiasEvaluation(
            energy_ev=float(energy),
            forces_ev_per_angstrom=forces,
            empty=bool(np.all(signed > float(self.lambda_s_angstrom))),
            signed_distances_angstrom=signed,
            penetration_angstrom=penetration_array,
        )


class CavityBiasedCalculator(Calculator):
    """Add one scaled ghost-cavity bias to a periodic water calculator."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        *,
        base_calculator: Calculator,
        cavity: GhostObservationCavity,
        oxygen_indices: tuple[int, ...],
        scale: float,
    ) -> None:
        super().__init__()
        if (
            isinstance(scale, (bool, np.bool_))
            or not math.isfinite(float(scale))
            or not 0.0 <= float(scale) <= 1.0
        ):
            raise ValueError("Cavity bias scale must be finite and in [0, 1].")
        self.base_calculator = base_calculator
        self.cavity = cavity
        self.oxygen_indices = tuple(int(value) for value in oxygen_indices)
        self.scale = float(scale)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("Cavity-biased packing requires an Atoms object.")
        probe = atoms.copy()
        probe.calc = self.base_calculator
        base_energy = float(probe.get_potential_energy())
        base_forces = np.asarray(probe.get_forces(), dtype=float)
        bias = self.cavity.evaluate(
            atoms,
            oxygen_indices=self.oxygen_indices,
        )
        total_energy = base_energy + self.scale * bias.energy_ev
        total_forces = (
            base_forces + self.scale * bias.forces_ev_per_angstrom
        )
        if (
            not math.isfinite(total_energy)
            or total_forces.shape != (len(atoms), 3)
            or not np.all(np.isfinite(total_forces))
        ):
            raise RuntimeError(
                "PACKING_BIAS_NONFINITE: biased energy or forces are invalid."
            )
        self.results = {
            "energy": total_energy,
            "free_energy": total_energy,
            "forces": total_forces,
            "components": {
                "base_ev": base_energy,
                "full_bias_ev": bias.energy_ev,
                "bias_scale": self.scale,
                "total_ev": total_energy,
                "empty": bias.empty,
                "cavity_hash": self.cavity.content_hash,
            },
        }


__all__ = [
    "CavityBiasEvaluation",
    "CavityBiasedCalculator",
    "GhostObservationCavity",
]
