"""Exact GTO surface map for the frozen Route-2 V0 atomic-IP response basis.

The atomic independent-particle asset stores transition-density AO matrices,
not a fitted radial surrogate.  This module evaluates their electronic
potential at a declared continuum surface and exposes its mathematical
transpose in the same coefficient/dual pairing:

``v_surface = B x`` and ``f_reaction = B.T q_surface``.

``B`` includes the electron-charge sign.  It is therefore the source map for
``delta n = sum_m x_m tau_m`` and
``delta V_e(s) = -sum_m x_m int tau_m(r)/|r-s| dr``.  The optional PySCF
builder is lazy because the persisted table is deliberately usable without a
runtime quantum-chemistry dependency.  It is a fixed-geometry source/dual
primitive, not a cavity, electronic functional, force, or solvation model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .route2_v0_atomic_independent_particle_response import (
    Route2V0AtomicIndependentParticleBaseline,
    Route2V0AtomicIndependentParticleResponseTable,
)


V0_ATOMIC_INDEPENDENT_PARTICLE_SURFACE_CONSTRUCTION = (
    "route2-v0-atomic-independent-particle-exact-gto-surface-v1"
)
BOHR_ANGSTROM = 0.5291772105638411


def _immutable_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (
        (shape is not None and array.shape != shape)
        or not np.all(np.isfinite(array))
    ):
        expected = "a finite array" if shape is None else f"shape {shape}"
        raise ValueError(f"{name} must be finite with {expected}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _positions(values: np.ndarray, *, atom_count: int) -> np.ndarray:
    return _immutable_array(
        values,
        name="atom_positions_angstrom",
        shape=(atom_count, 3),
    )


def _surface_points(values: np.ndarray) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        raise ValueError("surface_points_bohr must have finite shape (n_surface, 3).")
    return _immutable_array(points, name="surface_points_bohr", shape=points.shape)


def _finite_vector(values: np.ndarray, *, name: str, length: int) -> np.ndarray:
    return _immutable_array(values, name=name, shape=(length,))


def _source_operator_from_pyscf(
    *,
    response_table: Route2V0AtomicIndependentParticleResponseTable,
    baseline: Route2V0AtomicIndependentParticleBaseline,
    atom_positions_angstrom: np.ndarray,
    surface_points_bohr: np.ndarray,
) -> np.ndarray:
    """Build the exact electron-potential matrix with the frozen atomic AO data."""

    try:
        from pyscf import gto
    except ImportError as error:
        raise RuntimeError(
            "PySCF is required to build a new exact atomic-IP surface operator. "
            "Loading the frozen response table itself does not require PySCF."
        ) from error
    # Retain the frozen source-runner multiplication order so a regenerated
    # operator can be compared byte-for-byte with its recorded source matrix.
    positions_bohr = atom_positions_angstrom * (1.0 / BOHR_ANGSTROM)
    molecules: dict[int, Any] = {}
    transition_densities: dict[int, np.ndarray] = {}
    for number in sorted(set(int(value) for value in baseline.atomic_numbers)):
        response = response_table.response_for_atomic_number(number)
        molecule = gto.M(
            atom=f"{response.symbol} 0 0 0",
            basis=response.basis,
            charge=0,
            spin=response.spin_2s,
            verbose=0,
        )
        if molecule.nao_nr() != response.mo_coefficients.shape[0]:
            raise RuntimeError(
                "The PySCF AO layout disagrees with the frozen atomic-IP asset."
            )
        molecules[number] = molecule
        transition_densities[number] = response.transition_density_matrices()
    operator = np.empty(
        (len(surface_points_bohr), baseline.coefficient_count),
        dtype=float,
    )
    for atom_index, coefficient_slice in enumerate(baseline.coefficient_slices):
        number = int(baseline.atomic_numbers[atom_index])
        molecule = molecules[number]
        transitions = transition_densities[number]
        for point_index, point in enumerate(surface_points_bohr):
            with molecule.with_rinv_origin(point - positions_bohr[atom_index]):
                inverse_distance = molecule.intor("int1e_rinv")
            # Transition matrices are electron-number densities.  The physical
            # electrostatic potential contains the electron-charge minus sign.
            operator[point_index, coefficient_slice] = -np.einsum(
                "mij,ij->m",
                transitions,
                inverse_distance,
                optimize=True,
            )
    if not np.all(np.isfinite(operator)):
        raise RuntimeError("Atomic-IP surface operator contains non-finite values.")
    return operator


@dataclass(frozen=True)
class AtomicIndependentParticleSurfaceCoupling:
    """One fixed-geometry exact linear source/dual map for V0 atomic-IP modes."""

    baseline: Route2V0AtomicIndependentParticleBaseline
    atom_positions_angstrom: np.ndarray
    surface_points_bohr: np.ndarray
    surface_operator_hartree_per_e_per_coefficient: np.ndarray
    construction: str = V0_ATOMIC_INDEPENDENT_PARTICLE_SURFACE_CONSTRUCTION

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, Route2V0AtomicIndependentParticleBaseline):
            raise TypeError("baseline must be a frozen atomic independent-particle sum.")
        if self.construction != V0_ATOMIC_INDEPENDENT_PARTICLE_SURFACE_CONSTRUCTION:
            raise ValueError("Unsupported atomic independent-particle surface map.")
        positions = _positions(
            self.atom_positions_angstrom,
            atom_count=len(self.baseline.atomic_numbers),
        )
        points = _surface_points(self.surface_points_bohr)
        operator = _immutable_array(
            self.surface_operator_hartree_per_e_per_coefficient,
            name="surface_operator_hartree_per_e_per_coefficient",
            shape=(len(points), self.baseline.coefficient_count),
        )
        object.__setattr__(self, "atom_positions_angstrom", positions)
        object.__setattr__(self, "surface_points_bohr", points)
        object.__setattr__(
            self,
            "surface_operator_hartree_per_e_per_coefficient",
            operator,
        )

    @classmethod
    def from_pyscf(
        cls,
        *,
        response_table: Route2V0AtomicIndependentParticleResponseTable,
        atomic_numbers: np.ndarray,
        atom_positions_angstrom: np.ndarray,
        surface_points_bohr: np.ndarray,
    ) -> "AtomicIndependentParticleSurfaceCoupling":
        """Construct an exact AO-integral source matrix for one geometry/surface."""

        if not isinstance(response_table, Route2V0AtomicIndependentParticleResponseTable):
            raise TypeError("response_table must be a frozen atomic-IP response table.")
        baseline = response_table.assemble(atomic_numbers)
        positions = _positions(
            atom_positions_angstrom,
            atom_count=len(baseline.atomic_numbers),
        )
        points = _surface_points(surface_points_bohr)
        operator = _source_operator_from_pyscf(
            response_table=response_table,
            baseline=baseline,
            atom_positions_angstrom=positions,
            surface_points_bohr=points,
        )
        return cls(
            baseline=baseline,
            atom_positions_angstrom=positions,
            surface_points_bohr=points,
            surface_operator_hartree_per_e_per_coefficient=operator,
        )

    @property
    def atom_count(self) -> int:
        return len(self.baseline.atomic_numbers)

    @property
    def coefficient_count(self) -> int:
        return self.baseline.coefficient_count

    @property
    def surface_point_count(self) -> int:
        return len(self.surface_points_bohr)

    def coefficients(self, values: np.ndarray, *, name: str = "response_coefficients") -> np.ndarray:
        """Validate a response amplitude in the frozen neutral coefficient space."""

        return _finite_vector(values, name=name, length=self.coefficient_count)

    def surface_potential(self, coefficients: np.ndarray) -> np.ndarray:
        """Return the induced electronic surface MEP ``B @ x``."""

        return self.surface_operator_hartree_per_e_per_coefficient @ self.coefficients(
            coefficients
        )

    def surface_to_coefficient_dual(self, surface_charge_e: np.ndarray) -> np.ndarray:
        """Return the exact transpose ``B.T @ q`` in the declared pairing."""

        charge = _finite_vector(
            surface_charge_e,
            name="surface_charge_e",
            length=self.surface_point_count,
        )
        return self.surface_operator_hartree_per_e_per_coefficient.T @ charge

    def source_duality_error(
        self,
        coefficients: np.ndarray,
        surface_charge_e: np.ndarray,
    ) -> float:
        """Return the exact-forward/exact-transpose pairing residual."""

        amplitude = self.coefficients(coefficients)
        charge = _finite_vector(
            surface_charge_e,
            name="surface_charge_e",
            length=self.surface_point_count,
        )
        return abs(
            float(self.surface_potential(amplitude) @ charge)
            - float(amplitude @ self.surface_to_coefficient_dual(charge))
        )


__all__ = [
    "AtomicIndependentParticleSurfaceCoupling",
    "BOHR_ANGSTROM",
    "V0_ATOMIC_INDEPENDENT_PARTICLE_SURFACE_CONSTRUCTION",
]
