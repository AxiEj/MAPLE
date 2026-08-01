"""Frozen exterior surface probes and statistics for Route-2 source oracles.

This module is deliberately independent of a continuum solve.  It constructs
an SMD-radius exterior probe set from geometry alone, then supplies a
quadrature-weighted MEP discrepancy.  It is useful for rejecting an inadequate
permanent source before any PCM or experimental-solvation calculation; it does
not define a cavity, a PCM discretisation, or an energy functional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.units import Bohr


# Six axis, twelve edge, and eight cube-corner directions.  This deterministic
# octahedral/cubic 26-point shell is invariant under signed coordinate-axis
# permutations.  It is a source-oracle probe grid, not a Lebedev quadrature or
# a continuum surface discretisation.
_OCTAHEDRAL_CUBIC_26_DIRECTIONS = np.asarray(
    [
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
        *((sx / np.sqrt(2.0), sy / np.sqrt(2.0), 0.0)
          for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)),
        *((sx / np.sqrt(2.0), 0.0, sz / np.sqrt(2.0))
          for sx in (-1.0, 1.0) for sz in (-1.0, 1.0)),
        *((0.0, sy / np.sqrt(2.0), sz / np.sqrt(2.0))
          for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)),
        *((sx / np.sqrt(3.0), sy / np.sqrt(3.0), sz / np.sqrt(3.0))
          for sx in (-1.0, 1.0)
          for sy in (-1.0, 1.0)
          for sz in (-1.0, 1.0)),
    ],
    dtype=float,
)


@dataclass(frozen=True)
class Route2ExteriorProbeSurface:
    """One geometry-defined, exterior MEP probe shell.

    ``surface_points_bohr`` and weights are already filtered to points outside
    every *expanded* atom sphere.  The filter depends only on frozen geometry,
    radii, and clearance; it cannot use MACE, QM, continuum, or experimental
    values to select a point.
    """

    surface_points_bohr: np.ndarray
    quadrature_weights: np.ndarray
    parent_atom_indices: np.ndarray
    candidate_count: int
    clearance_angstrom: float

    def __post_init__(self) -> None:
        points = np.asarray(self.surface_points_bohr, dtype=float)
        weights = np.asarray(self.quadrature_weights, dtype=float)
        parents = np.asarray(self.parent_atom_indices, dtype=int)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError("surface_points_bohr must have nonempty shape (n, 3).")
        if not np.all(np.isfinite(points)):
            raise ValueError("surface_points_bohr must be finite.")
        if weights.shape != (points.shape[0],) or np.any(weights <= 0.0):
            raise ValueError("quadrature_weights must be positive with one value per point.")
        if not np.all(np.isfinite(weights)):
            raise ValueError("quadrature_weights must be finite.")
        if parents.shape != (points.shape[0],) or np.any(parents < 0):
            raise ValueError("parent_atom_indices must identify every retained point.")
        candidate_count = int(self.candidate_count)
        if candidate_count < points.shape[0] or candidate_count <= 0:
            raise ValueError("candidate_count must bound the retained point count.")
        clearance = float(self.clearance_angstrom)
        if not np.isfinite(clearance) or clearance <= 0.0:
            raise ValueError("clearance_angstrom must be positive and finite.")
        object.__setattr__(self, "surface_points_bohr", np.array(points, copy=True))
        object.__setattr__(self, "quadrature_weights", np.array(weights, copy=True))
        object.__setattr__(self, "parent_atom_indices", np.array(parents, copy=True))
        self.surface_points_bohr.setflags(write=False)
        self.quadrature_weights.setflags(write=False)
        self.parent_atom_indices.setflags(write=False)

    @property
    def retained_point_count(self) -> int:
        return int(self.surface_points_bohr.shape[0])


def build_route2_smd_exterior_probe_surface(
    atom_positions_angstrom: np.ndarray,
    coulomb_radii_angstrom: np.ndarray,
    *,
    clearance_angstrom: float,
) -> Route2ExteriorProbeSurface:
    """Build a deterministic external MEP shell from SMD sphere radii.

    Every atom contributes the same 26 candidate directions at radius
    ``r_A + clearance``.  A candidate is retained exactly when it lies outside
    every other atom's *same-clearance* sphere, with a fixed roundoff tolerance.
    This makes the potential probe relevant to the stated SMD geometry without
    turning it into a continuum matrix or a parameter-selection device.
    """

    positions = np.asarray(atom_positions_angstrom, dtype=float)
    radii = np.asarray(coulomb_radii_angstrom, dtype=float)
    if positions.ndim != 2 or positions.shape[0] == 0 or positions.shape[1] != 3:
        raise ValueError("atom_positions_angstrom must have nonempty shape (n_atoms, 3).")
    if not np.all(np.isfinite(positions)):
        raise ValueError("atom_positions_angstrom must be finite.")
    if radii.shape != (positions.shape[0],) or not np.all(np.isfinite(radii)):
        raise ValueError("coulomb_radii_angstrom must have one finite value per atom.")
    if np.any(radii <= 0.0):
        raise ValueError("coulomb_radii_angstrom must be positive.")
    clearance = float(clearance_angstrom)
    if not np.isfinite(clearance) or clearance <= 0.0:
        raise ValueError("clearance_angstrom must be positive and finite.")

    atom_count = positions.shape[0]
    parent_indices = np.repeat(np.arange(atom_count, dtype=int), len(_OCTAHEDRAL_CUBIC_26_DIRECTIONS))
    directions = np.tile(_OCTAHEDRAL_CUBIC_26_DIRECTIONS, (atom_count, 1))
    expanded_radii = radii + clearance
    candidate_points_angstrom = (
        positions[parent_indices]
        + expanded_radii[parent_indices, None] * directions
    )
    distances = np.linalg.norm(
        candidate_points_angstrom[:, None, :] - positions[None, :, :],
        axis=2,
    )
    # Own sphere is reached exactly, while other spheres must not penetrate the
    # fixed exterior shell.  The tolerance is only floating-point geometry
    # hygiene; it is not a result-dependent cutoff.
    mask = np.all(distances + 2.0e-12 >= expanded_radii[None, :], axis=1)
    if int(np.count_nonzero(mask)) < atom_count:
        raise RuntimeError(
            "The frozen exterior probe shell retained fewer points than atoms; "
            "geometry is too interpenetrating for this source diagnostic."
        )
    retained_points_bohr = candidate_points_angstrom[mask] / Bohr
    retained_parents = parent_indices[mask]
    # Each candidate represents the same solid angle.  A global normalization
    # is intentionally unnecessary for ratios but keeps weighted RMSE in the
    # native potential units.
    weights = np.full(np.count_nonzero(mask), 4.0 * np.pi / len(_OCTAHEDRAL_CUBIC_26_DIRECTIONS))
    return Route2ExteriorProbeSurface(
        surface_points_bohr=retained_points_bohr,
        quadrature_weights=weights,
        parent_atom_indices=retained_parents,
        candidate_count=int(candidate_points_angstrom.shape[0]),
        clearance_angstrom=clearance,
    )


def route2_weighted_surface_mep_discrepancy(
    candidate_hartree_per_e: np.ndarray,
    reference_hartree_per_e: np.ndarray,
    quadrature_weights: np.ndarray,
) -> dict[str, float]:
    """Return scale-explicit static-MEP discrepancy metrics.

    The primary relative L2 measure is weighted by the frozen exterior shell;
    it is not a PCM reaction-energy norm and therefore carries no claim of a
    continuum energy or a variational electronic functional.
    """

    candidate = np.asarray(candidate_hartree_per_e, dtype=float)
    reference = np.asarray(reference_hartree_per_e, dtype=float)
    weights = np.asarray(quadrature_weights, dtype=float)
    if candidate.ndim != 1 or reference.shape != candidate.shape:
        raise ValueError("candidate/reference MEP arrays must be one-dimensional and matched.")
    if weights.shape != candidate.shape or np.any(weights <= 0.0):
        raise ValueError("quadrature_weights must be positive and matched to the MEP arrays.")
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(weights)):
        raise ValueError("MEP arrays and weights must be finite.")
    difference = candidate - reference
    weighted_error_sq = float(np.dot(weights, difference * difference))
    weighted_reference_sq = float(np.dot(weights, reference * reference))
    total_weight = float(np.sum(weights))
    if weighted_reference_sq <= 0.0 or total_weight <= 0.0:
        raise ValueError("Reference MEP must have nonzero weighted norm.")
    reference_inf = float(np.max(np.abs(reference)))
    if reference_inf <= 0.0:
        raise ValueError("Reference MEP must have nonzero infinity norm.")
    return {
        "weighted_relative_l2": float(np.sqrt(weighted_error_sq / weighted_reference_sq)),
        "weighted_rmse_hartree_per_e": float(np.sqrt(weighted_error_sq / total_weight)),
        "maximum_absolute_error_hartree_per_e": float(np.max(np.abs(difference))),
        "relative_infinity_norm": float(np.max(np.abs(difference)) / reference_inf),
    }


__all__ = [
    "Route2ExteriorProbeSurface",
    "build_route2_smd_exterior_probe_surface",
    "route2_weighted_surface_mep_discrepancy",
]
