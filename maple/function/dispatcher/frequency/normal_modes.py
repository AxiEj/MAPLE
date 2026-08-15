"""Unit-explicit Cartesian Hessian and vibrational-subspace primitives.

The public frequency driver historically mixed two separate operations:
mass-weighting and removal of rigid translations/rotations.  This module keeps
the mathematics in one small, calculator-independent boundary:

``H_mw = M^{-1/2} H_cart M^{-1/2}``

Rigid displacements are constructed in the same mass-weighted coordinate
space before the vibrational block is diagonalized.  The input unit is
explicitly ASE's public ``eV / angstrom**2`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

ELECTRONVOLT_J = 1.602176634e-19
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
ANGSTROM_M = 1.0e-10
SPEED_OF_LIGHT_CM_PER_S = 2.99792458e10

EV_PER_ANGSTROM2_AMU_TO_WAVENUMBER_CM1 = math.sqrt(
    ELECTRONVOLT_J / (ATOMIC_MASS_UNIT_KG * ANGSTROM_M**2)
) / (2.0 * math.pi * SPEED_OF_LIGHT_CM_PER_S)


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.array(result, copy=True)
    result.setflags(write=False)
    return result


def _masses_positions(
    masses_amu: object,
    positions_angstrom: object,
) -> tuple[np.ndarray, np.ndarray]:
    masses = np.asarray(masses_amu, dtype=float)
    positions = np.asarray(positions_angstrom, dtype=float)
    if (
        masses.ndim != 1
        or masses.size == 0
        or not np.all(np.isfinite(masses))
        or np.any(masses <= 0.0)
    ):
        raise ValueError("masses_amu must be a finite, positive one-dimensional array.")
    if positions.shape != (masses.size, 3) or not np.all(np.isfinite(positions)):
        raise ValueError(
            "positions_angstrom must be finite with shape (len(masses_amu), 3)."
        )
    return np.array(masses, copy=True), np.array(positions, copy=True)


@dataclass(frozen=True, slots=True)
class RigidBodySubspaces:
    """Orthogonal rigid and vibrational bases in mass-weighted coordinates."""

    translation_basis_mass_weighted: np.ndarray
    rotation_basis_mass_weighted: np.ndarray
    vibrational_basis_mass_weighted: np.ndarray
    rotation_singular_values_sqrt_amu_angstrom: np.ndarray
    is_linear: bool

    @property
    def atom_count(self) -> int:
        return self.translation_basis_mass_weighted.shape[0] // 3

    @property
    def translation_rank(self) -> int:
        return self.translation_basis_mass_weighted.shape[1]

    @property
    def rotation_rank(self) -> int:
        return self.rotation_basis_mass_weighted.shape[1]

    @property
    def rigid_rank(self) -> int:
        return self.translation_rank + self.rotation_rank

    @property
    def vibrational_rank(self) -> int:
        return self.vibrational_basis_mass_weighted.shape[1]


@dataclass(frozen=True, slots=True)
class VibrationalAnalysis:
    """Mass-weighted vibrational eigensystem for one Cartesian Hessian."""

    subspaces: RigidBodySubspaces
    hessian_symmetry_max_abs_eV_per_A2: float
    mass_weighted_hessian_eV_per_A2_amu: np.ndarray
    vibrational_hessian_eV_per_A2_amu: np.ndarray
    eigenvalues_eV_per_A2_amu: np.ndarray
    frequencies_cm1: np.ndarray
    modes_mass_weighted: np.ndarray
    modes_cartesian_per_sqrt_amu: np.ndarray


def rigid_body_subspaces(
    masses_amu: object,
    positions_angstrom: object,
    *,
    rotation_rank_relative_tolerance: float = 1.0e-10,
) -> RigidBodySubspaces:
    """Build translation, rotation, and vibrational bases.

    Coordinates are first shifted to the center of mass.  Translation and
    rotation displacements are then multiplied by ``sqrt(mass)`` so all
    orthogonalization occurs in the same coordinate space as ``H_mw``.
    """

    masses, positions = _masses_positions(masses_amu, positions_angstrom)
    tolerance = float(rotation_rank_relative_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rotation_rank_relative_tolerance must be positive.")

    atom_count = masses.size
    coordinate_count = 3 * atom_count
    square_root_masses = np.sqrt(masses)
    total_mass = float(np.sum(masses))
    centered = (
        positions
        - np.sum(positions * masses[:, None], axis=0, keepdims=True) / total_mass
    )

    translations = np.zeros((coordinate_count, 3), dtype=float)
    for axis in range(3):
        translations[axis::3, axis] = square_root_masses / math.sqrt(total_mass)

    raw_rotations = np.zeros((coordinate_count, 3), dtype=float)
    axes = np.eye(3)
    for axis, unit_axis in enumerate(axes):
        displacement = np.cross(np.broadcast_to(unit_axis, centered.shape), centered)
        raw_rotations[:, axis] = (square_root_masses[:, None] * displacement).reshape(
            -1
        )
    raw_rotations -= translations @ (translations.T @ raw_rotations)

    rotation_left, singular_values, _ = np.linalg.svd(
        raw_rotations,
        full_matrices=False,
    )
    scale = float(singular_values[0]) if singular_values.size else 0.0
    rotation_rank = int(
        np.count_nonzero(singular_values > max(scale * tolerance, 1.0e-14))
    )
    if atom_count == 1:
        rotation_rank = 0
    if rotation_rank not in (0, 2, 3):
        raise ValueError(
            "geometry has an unsupported rigid-rotation rank; expected 0, 2, or 3."
        )
    rotations = rotation_left[:, :rotation_rank]

    rigid = np.concatenate((translations, rotations), axis=1)
    gram_error = float(np.max(np.abs(rigid.T @ rigid - np.eye(rigid.shape[1]))))
    if gram_error > 5.0e-12:
        raise RuntimeError("rigid mass-weighted basis lost orthonormality.")
    complete_basis, _ = np.linalg.qr(rigid, mode="complete")
    vibrations = complete_basis[:, rigid.shape[1] :]

    translations = _readonly(
        translations,
        shape=(coordinate_count, 3),
        name="translation basis",
    )
    rotations = _readonly(
        rotations,
        shape=(coordinate_count, rotation_rank),
        name="rotation basis",
    )
    vibrations = _readonly(
        vibrations,
        shape=(coordinate_count, coordinate_count - rigid.shape[1]),
        name="vibrational basis",
    )
    singular_values = _readonly(
        singular_values,
        shape=(3,),
        name="rotation singular values",
    )
    return RigidBodySubspaces(
        translation_basis_mass_weighted=translations,
        rotation_basis_mass_weighted=rotations,
        vibrational_basis_mass_weighted=vibrations,
        rotation_singular_values_sqrt_amu_angstrom=singular_values,
        is_linear=(atom_count > 1 and rotation_rank == 2),
    )


def mass_weighted_basis_to_cartesian(
    basis_mass_weighted: object,
    masses_amu: object,
    *,
    normalize_columns: bool = False,
) -> np.ndarray:
    """Map mass-weighted displacement columns back to Cartesian coordinates."""

    masses = np.asarray(masses_amu, dtype=float)
    basis = np.asarray(basis_mass_weighted, dtype=float)
    if (
        masses.ndim != 1
        or masses.size == 0
        or np.any(masses <= 0.0)
        or not np.all(np.isfinite(masses))
    ):
        raise ValueError("masses_amu must be finite and positive.")
    if basis.ndim != 2 or basis.shape[0] != 3 * masses.size:
        raise ValueError("basis_mass_weighted must have shape (3N, K).")
    result = basis / np.repeat(np.sqrt(masses), 3)[:, None]
    if normalize_columns and result.shape[1]:
        norms = np.linalg.norm(result, axis=0)
        if np.any(norms <= np.finfo(float).tiny):
            raise ValueError("basis contains a singular Cartesian direction.")
        result = result / norms[None, :]
    result = np.array(result, copy=True)
    result.setflags(write=False)
    return result


def analyze_cartesian_hessian(
    hessian_eV_per_A2: object,
    masses_amu: object,
    positions_angstrom: object,
    *,
    symmetry_tolerance_eV_per_A2: float = 1.0e-8,
) -> VibrationalAnalysis:
    """Project a symmetric ASE-unit Hessian into the vibrational subspace."""

    masses, positions = _masses_positions(masses_amu, positions_angstrom)
    coordinate_count = 3 * masses.size
    hessian = np.asarray(hessian_eV_per_A2, dtype=float)
    if hessian.shape != (coordinate_count, coordinate_count) or not np.all(
        np.isfinite(hessian)
    ):
        raise ValueError(
            f"hessian_eV_per_A2 must be finite with shape "
            f"({coordinate_count}, {coordinate_count})."
        )
    tolerance = float(symmetry_tolerance_eV_per_A2)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("symmetry_tolerance_eV_per_A2 must be non-negative.")
    symmetry_error = float(np.max(np.abs(hessian - hessian.T)))
    if symmetry_error > tolerance:
        raise ValueError(
            "Cartesian Hessian is not symmetric within the requested tolerance."
        )
    hessian = 0.5 * (hessian + hessian.T)

    inverse_square_root_mass = 1.0 / np.sqrt(np.repeat(masses, 3))
    hessian_mass_weighted = (
        inverse_square_root_mass[:, None] * hessian * inverse_square_root_mass[None, :]
    )
    subspaces = rigid_body_subspaces(masses, positions)
    vibrational_basis = subspaces.vibrational_basis_mass_weighted
    vibrational_hessian = (
        vibrational_basis.T @ hessian_mass_weighted @ vibrational_basis
    )
    vibrational_hessian = 0.5 * (vibrational_hessian + vibrational_hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(vibrational_hessian)
    modes_mass_weighted = vibrational_basis @ eigenvectors
    modes_cartesian = inverse_square_root_mass[:, None] * modes_mass_weighted
    frequencies = (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * EV_PER_ANGSTROM2_AMU_TO_WAVENUMBER_CM1
    )

    return VibrationalAnalysis(
        subspaces=subspaces,
        hessian_symmetry_max_abs_eV_per_A2=symmetry_error,
        mass_weighted_hessian_eV_per_A2_amu=_readonly(
            hessian_mass_weighted,
            shape=(coordinate_count, coordinate_count),
            name="mass-weighted Hessian",
        ),
        vibrational_hessian_eV_per_A2_amu=_readonly(
            vibrational_hessian,
            shape=(subspaces.vibrational_rank, subspaces.vibrational_rank),
            name="vibrational Hessian",
        ),
        eigenvalues_eV_per_A2_amu=_readonly(
            eigenvalues,
            shape=(subspaces.vibrational_rank,),
            name="vibrational eigenvalues",
        ),
        frequencies_cm1=_readonly(
            frequencies,
            shape=(subspaces.vibrational_rank,),
            name="vibrational frequencies",
        ),
        modes_mass_weighted=_readonly(
            modes_mass_weighted,
            shape=(coordinate_count, subspaces.vibrational_rank),
            name="mass-weighted modes",
        ),
        modes_cartesian_per_sqrt_amu=_readonly(
            modes_cartesian,
            shape=(coordinate_count, subspaces.vibrational_rank),
            name="Cartesian modes",
        ),
    )


__all__ = [
    "EV_PER_ANGSTROM2_AMU_TO_WAVENUMBER_CM1",
    "RigidBodySubspaces",
    "VibrationalAnalysis",
    "analyze_cartesian_hessian",
    "mass_weighted_basis_to_cartesian",
    "rigid_body_subspaces",
]
