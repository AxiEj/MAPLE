from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from ase import Atoms
from ase.units import kB
from scipy.special import erfc
from scipy.stats import qmc

from .protocol import canonical_sha256


@dataclass(frozen=True)
class EffectiveVolumeEstimate:
    volume_angstrom3: float
    replicate_standard_error_angstrom3: float
    tail_upper_bound_angstrom3: float
    replicate_estimates_angstrom3: tuple[float, ...]
    samples_per_replicate: int
    integration_box_angstrom: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ]


@dataclass(frozen=True)
class FlatBottomSurfaceRestraint:
    """Oxygen-membership shell wall on the all-solute-atom vdW surface."""

    solute_indices: tuple[int, ...]
    water_oxygen_indices: tuple[int, ...]
    vdw_radii_angstrom: Mapping[int, float]
    lambda_s_angstrom: float
    force_constant_ev_per_angstrom2: float
    shell_boundary_id: str
    measure_id: str
    water_atom_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        solute = tuple(int(index) for index in self.solute_indices)
        oxygens = tuple(int(index) for index in self.water_oxygen_indices)
        water_atoms = (
            oxygens
            if self.water_atom_indices is None
            else tuple(int(index) for index in self.water_atom_indices)
        )
        if not solute or not oxygens or not water_atoms:
            raise ValueError(
                "Surface restraint requires solute atoms and complete labeled waters."
            )
        if len(set(solute)) != len(solute):
            raise ValueError("Surface restraint solute indices contain duplicates.")
        if len(set(oxygens)) != len(oxygens):
            raise ValueError(
                "Surface restraint water oxygen indices contain duplicates."
            )
        if len(set(water_atoms)) != len(water_atoms):
            raise ValueError("Surface restraint water atom indices contain duplicates.")
        if set(solute).intersection(water_atoms):
            raise ValueError(
                "Surface restraint solute and water atom indices overlap."
            )
        if not set(oxygens).issubset(water_atoms):
            raise ValueError(
                "Every restrained oxygen must belong to the labeled water atom list."
            )

        radii = {
            int(number): float(radius)
            for number, radius in self.vdw_radii_angstrom.items()
        }
        if not radii or any(
            number <= 0
            or not math.isfinite(radius)
            or radius <= 0.0
            for number, radius in radii.items()
        ):
            raise ValueError(
                "Surface restraint vdW radii must be finite positive values."
            )
        for name, value in (
            ("lambda_s_angstrom", self.lambda_s_angstrom),
            (
                "force_constant_ev_per_angstrom2",
                self.force_constant_ev_per_angstrom2,
            ),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        for name, value in (
            ("shell_boundary_id", self.shell_boundary_id),
            ("measure_id", self.measure_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")

        object.__setattr__(self, "solute_indices", solute)
        object.__setattr__(self, "water_oxygen_indices", oxygens)
        object.__setattr__(self, "water_atom_indices", water_atoms)
        object.__setattr__(
            self,
            "vdw_radii_angstrom",
            MappingProxyType(radii),
        )

    @classmethod
    def from_protocol(
        cls,
        *,
        solute_indices: tuple[int, ...],
        water_oxygen_indices: tuple[int, ...],
        vdw_radii_angstrom: Mapping[int, float],
        lambda_s_angstrom: float,
        shell_boundary_id: str,
        measure_id: str,
        temperature_k: float,
        buffer_height_kbt: float,
        buffer_width_angstrom: float,
        water_atom_indices: tuple[int, ...] | None = None,
    ) -> "FlatBottomSurfaceRestraint":
        for name, value in (
            ("temperature_k", temperature_k),
            ("buffer_height_kbt", buffer_height_kbt),
            ("buffer_width_angstrom", buffer_width_angstrom),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        force_constant = (
            2.0
            * float(buffer_height_kbt)
            * kB
            * float(temperature_k)
            / float(buffer_width_angstrom) ** 2
        )
        return cls(
            solute_indices=solute_indices,
            water_oxygen_indices=water_oxygen_indices,
            water_atom_indices=water_atom_indices,
            vdw_radii_angstrom=vdw_radii_angstrom,
            lambda_s_angstrom=lambda_s_angstrom,
            force_constant_ev_per_angstrom2=force_constant,
            shell_boundary_id=shell_boundary_id,
            measure_id=measure_id,
        )

    @property
    def content_hash(self) -> str:
        return canonical_sha256(
            {
                "contract_id": "flat-bottom-surface-restraint-v1",
                "shell_boundary_id": self.shell_boundary_id,
                "lambda_s_A": float(self.lambda_s_angstrom),
                "k_eV_per_A2": float(
                    self.force_constant_ev_per_angstrom2
                ),
                "solute_atom_indices": list(self.solute_indices),
                "water_oxygen_indices": list(self.water_oxygen_indices),
                "labeled_water_atom_indices": list(self.water_atom_indices),
                "vdw_radii_A": {
                    str(number): radius
                    for number, radius in sorted(
                        self.vdw_radii_angstrom.items()
                    )
                },
                "measure_id": self.measure_id,
                "boundary_conditions": "nonperiodic",
            }
        )

    def _validate_atoms(self, atoms: Atoms) -> None:
        indices = (
            self.solute_indices
            + self.water_oxygen_indices
            + self.water_atom_indices
        )
        if min(indices) < 0 or max(indices) >= len(atoms):
            raise ValueError(
                "ATOM_LIST_MISMATCH: restraint indices do not match the Atoms object."
            )
        missing = sorted(
            {
                int(atoms.numbers[index])
                for index in self.solute_indices
                if int(atoms.numbers[index]) not in self.vdw_radii_angstrom
            }
        )
        if missing:
            raise ValueError(
                "RESTRAINT_RADIUS_MISSING: no vdW radius for atomic number(s) "
                + ", ".join(str(value) for value in missing)
            )
        if any(int(atoms.numbers[index]) != 8 for index in self.water_oxygen_indices):
            raise ValueError(
                "ATOM_LIST_MISMATCH: water_oxygen_indices must identify oxygen atoms."
            )

    def evaluate(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        self._validate_atoms(atoms)
        positions = np.asarray(atoms.positions, dtype=float)
        if not np.all(np.isfinite(positions)):
            raise ValueError("RESTRAINT_GEOMETRY_INVALID: non-finite coordinates.")
        forces = np.zeros((len(atoms), 3), dtype=float)
        energy = 0.0
        solute_positions = positions[list(self.solute_indices)]
        solute_radii = np.asarray(
            [
                self.vdw_radii_angstrom[int(atoms.numbers[index])]
                for index in self.solute_indices
            ],
            dtype=float,
        )
        force_constant = float(self.force_constant_ev_per_angstrom2)
        shell_boundary = float(self.lambda_s_angstrom)

        for oxygen_index in self.water_oxygen_indices:
            displacements = positions[oxygen_index] - solute_positions
            distances = np.linalg.norm(displacements, axis=1)
            surface_distances = distances - solute_radii
            closest_local = int(np.argmin(surface_distances))
            excess = float(surface_distances[closest_local] - shell_boundary)
            if excess <= 0.0:
                continue
            distance = float(distances[closest_local])
            if distance <= 1.0e-12:
                raise ValueError(
                    "RESTRAINT_SINGULAR_GEOMETRY: closest solute-water "
                    "distance is zero outside the flat-bottom region."
                )
            solute_index = self.solute_indices[closest_local]
            energy += 0.5 * force_constant * excess**2
            oxygen_force = (
                -force_constant
                * excess
                * displacements[closest_local]
                / distance
            )
            forces[oxygen_index] += oxygen_force
            forces[solute_index] -= oxygen_force
        return float(energy), forces

    def effective_translational_volume(
        self,
        atoms: Atoms,
        *,
        beta_ev_inverse: float,
        sobol_power: int = 16,
        replicates: int = 8,
        seed: int = 20260725,
        tail_sigma: float = 9.0,
        chunk_size: int = 65536,
    ) -> EffectiveVolumeEstimate:
        """Integrate exp(-beta R) over the inserted-water O translation.

        Independent scrambled Sobol replicates quantify integration noise. The
        finite integration box includes ``tail_sigma / sqrt(alpha)`` beyond
        every flat surface. The returned tail bound is a conservative sum of
        the omitted single-atom Gaussian tails.
        """

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
        for name, value in (
            ("beta_ev_inverse", beta_ev_inverse),
            ("tail_sigma", tail_sigma),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive.")

        solute_positions = np.asarray(
            atoms.positions[list(self.solute_indices)],
            dtype=float,
        )
        radii = np.asarray(
            [
                self.vdw_radii_angstrom[int(atoms.numbers[index])]
                for index in self.solute_indices
            ],
            dtype=float,
        )
        alpha = (
            0.5
            * float(beta_ev_inverse)
            * float(self.force_constant_ev_per_angstrom2)
        )
        tail_extent = float(tail_sigma) / math.sqrt(alpha)
        flat_radii = radii + float(self.lambda_s_angstrom)
        lower = np.min(
            solute_positions - flat_radii[:, None],
            axis=0,
        ) - tail_extent
        upper = np.max(
            solute_positions + flat_radii[:, None],
            axis=0,
        ) + tail_extent
        widths = upper - lower
        box_volume = float(np.prod(widths))

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
                signed_distance = np.min(distances - radii[None, :], axis=1)
                excess = np.maximum(
                    signed_distance - float(self.lambda_s_angstrom),
                    0.0,
                )
                weight_sum += float(np.sum(np.exp(-alpha * excess**2)))
            estimates.append(box_volume * weight_sum / len(points))

        estimate_array = np.asarray(estimates, dtype=float)
        standard_error = float(
            np.std(estimate_array, ddof=1) / math.sqrt(len(estimate_array))
        )
        tail_bound = float(
            sum(
                self._single_center_tail_upper_bound(
                    flat_radius=float(flat_radius),
                    alpha=alpha,
                    tail_extent=tail_extent,
                )
                for flat_radius in flat_radii
            )
        )
        return EffectiveVolumeEstimate(
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
        )

    @staticmethod
    def _single_center_tail_upper_bound(
        *,
        flat_radius: float,
        alpha: float,
        tail_extent: float,
    ) -> float:
        root_alpha = math.sqrt(alpha)
        exponential = math.exp(-alpha * tail_extent**2)
        complementary = float(erfc(root_alpha * tail_extent))
        integral_0 = (
            math.sqrt(math.pi)
            / (2.0 * root_alpha)
            * complementary
        )
        integral_1 = exponential / (2.0 * alpha)
        integral_2 = (
            tail_extent * exponential / (2.0 * alpha)
            + math.sqrt(math.pi)
            / (4.0 * alpha**1.5)
            * complementary
        )
        return (
            4.0
            * math.pi
            * (
                flat_radius**2 * integral_0
                + 2.0 * flat_radius * integral_1
                + integral_2
            )
        )
