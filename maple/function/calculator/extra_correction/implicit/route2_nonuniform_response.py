"""Geometry-only localized-field probes for Route-2 response oracles.

The production continuum samples a spatially nonuniform reaction potential;
a molecular ``3 x 3`` uniform-field polarizability is therefore not a complete
response test.  This module selects well-separated external point-charge
locations from an already frozen exterior probe shell and constructs their
exact atom-centred potential/gradient jets.  It contains no continuum solve,
experimental label, fitted parameter, or model-specific response.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.units import Bohr

from .gto_density import cartesian_multipoles


@dataclass(frozen=True)
class Route2LocalizedPointChargeModes:
    """A fixed set of exterior unit point-charge perturbations."""

    source_points_bohr: np.ndarray
    source_surface_indices: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.source_points_bohr, dtype=float)
        indices = np.asarray(self.source_surface_indices, dtype=int)
        if (
            points.ndim != 2
            or points.shape[0] == 0
            or points.shape[1] != 3
            or not np.all(np.isfinite(points))
        ):
            raise ValueError(
                "Localized point-charge sources must have finite shape (n_modes, 3)."
            )
        if indices.shape != (points.shape[0],) or np.any(indices < 0):
            raise ValueError(
                "Localized point-charge source indices must identify every mode."
            )
        if len(set(int(value) for value in indices)) != len(indices):
            raise ValueError("Localized point-charge source indices must be unique.")
        object.__setattr__(self, "source_points_bohr", np.array(points, copy=True))
        object.__setattr__(
            self,
            "source_surface_indices",
            np.array(indices, copy=True),
        )
        self.source_points_bohr.setflags(write=False)
        self.source_surface_indices.setflags(write=False)

    @property
    def mode_count(self) -> int:
        return int(self.source_points_bohr.shape[0])


def select_farthest_exterior_point_charge_modes(
    surface_points_bohr: np.ndarray,
    atom_positions_angstrom: np.ndarray,
    *,
    mode_count: int,
) -> Route2LocalizedPointChargeModes:
    """Select a deterministic maximin subset from a frozen exterior shell.

    The seed is the shell point farthest from the nuclear-geometry centroid.
    Every later point maximizes its minimum squared distance to the selected
    set.  ``numpy.argmax`` supplies a stable lowest-index tie break.  Selection
    uses geometry only: no MACE, QM, continuum, or experimental value can
    affect a mode.
    """

    points = np.asarray(surface_points_bohr, dtype=float)
    positions = np.asarray(atom_positions_angstrom, dtype=float)
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("surface_points_bohr must have finite nonempty shape (n, 3).")
    if (
        positions.ndim != 2
        or positions.shape[0] == 0
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError(
            "atom_positions_angstrom must have finite nonempty shape (n_atoms, 3)."
        )
    if isinstance(mode_count, bool) or int(mode_count) != mode_count:
        raise TypeError("mode_count must be an integer.")
    count = int(mode_count)
    if count <= 0 or count > points.shape[0]:
        raise ValueError("mode_count must be positive and no larger than the shell.")

    centroid_bohr = np.mean(positions, axis=0) / Bohr
    seed_scores = np.sum((points - centroid_bohr[None, :]) ** 2, axis=1)
    selected = [int(np.argmax(seed_scores))]
    minimum_distance_squared = np.sum(
        (points - points[selected[0]][None, :]) ** 2,
        axis=1,
    )
    minimum_distance_squared[selected[0]] = -np.inf
    while len(selected) < count:
        next_index = int(np.argmax(minimum_distance_squared))
        selected.append(next_index)
        distance_squared = np.sum(
            (points - points[next_index][None, :]) ** 2,
            axis=1,
        )
        minimum_distance_squared = np.minimum(
            minimum_distance_squared,
            distance_squared,
        )
        minimum_distance_squared[np.asarray(selected, dtype=int)] = -np.inf

    indices = np.asarray(selected, dtype=int)
    return Route2LocalizedPointChargeModes(
        source_points_bohr=points[indices],
        source_surface_indices=indices,
    )


def localized_point_charge_atom_jets_hartree(
    atom_positions_angstrom: np.ndarray,
    source_points_bohr: np.ndarray,
) -> np.ndarray:
    """Return unit-charge atom jets ``[V,dV/dx,dV/dy,dV/dz]``.

    Potentials are Hartree/e per unit external elementary charge and gradients
    are Hartree/(e bohr) per unit charge.  Each source is a Coulomb point charge
    with its zero fixed at infinity, matching the perturbation used by the QM
    finite-field helper.
    """

    positions = np.asarray(atom_positions_angstrom, dtype=float) / Bohr
    sources = np.asarray(source_points_bohr, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[0] == 0
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("Atom positions must have finite nonempty shape (n_atoms, 3).")
    if (
        sources.ndim != 2
        or sources.shape[0] == 0
        or sources.shape[1] != 3
        or not np.all(np.isfinite(sources))
    ):
        raise ValueError("Source points must have finite nonempty shape (n_modes, 3).")

    displacement = sources[:, None, :] - positions[None, :, :]
    radius = np.linalg.norm(displacement, axis=2)
    if np.any(radius <= 1.0e-12):
        raise ValueError("An external point charge coincides with an atomic centre.")
    potential = 1.0 / radius
    gradient = displacement / radius[:, :, None] ** 3
    jets = np.concatenate((potential[:, :, None], gradient), axis=2)
    if not np.all(np.isfinite(jets)):
        raise RuntimeError("Localized point-charge atom jets are non-finite.")
    return jets


def molecular_dipole_response_from_density_coefficients(
    atom_positions_angstrom: np.ndarray,
    density_response_coefficients: np.ndarray,
) -> np.ndarray:
    """Contract one or more MACE l<=1 density responses into total dipoles.

    ``density_response_coefficients`` may have shape ``(n_atoms, 4)`` or
    ``(n_modes, n_atoms, 4)``.  The result is in e Angstrom per perturbation
    unit.  Fixed nuclei do not contribute to a density derivative.
    """

    positions = np.asarray(atom_positions_angstrom, dtype=float)
    coefficients = np.asarray(density_response_coefficients, dtype=float)
    squeeze = coefficients.ndim == 2
    if squeeze:
        coefficients = coefficients[None, :, :]
    if (
        positions.ndim != 2
        or positions.shape[0] == 0
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("Atom positions must have finite nonempty shape (n_atoms, 3).")
    if (
        coefficients.ndim != 3
        or coefficients.shape[1:] != (positions.shape[0], 4)
        or not np.all(np.isfinite(coefficients))
    ):
        raise ValueError(
            "Density responses must have shape (n_modes, n_atoms, 4) or (n_atoms, 4)."
        )

    dipoles = []
    for response in coefficients:
        charges, atom_dipoles = cartesian_multipoles(response)
        dipoles.append(
            np.sum(charges[:, None] * positions, axis=0)
            + np.sum(atom_dipoles, axis=0)
        )
    result = np.asarray(dipoles, dtype=float)
    return result[0] if squeeze else result


def weighted_response_matrix_discrepancy(
    candidate: np.ndarray,
    reference: np.ndarray,
    surface_weights: np.ndarray,
) -> dict[str, object]:
    """Compare surface-response columns under one frozen positive metric."""

    left = np.asarray(candidate, dtype=float)
    right = np.asarray(reference, dtype=float)
    weights = np.asarray(surface_weights, dtype=float)
    if (
        left.ndim != 2
        or right.shape != left.shape
        or left.shape[0] == 0
        or left.shape[1] == 0
    ):
        raise ValueError("Response matrices must have the same nonempty 2-D shape.")
    if weights.shape != (left.shape[0],) or np.any(weights <= 0.0):
        raise ValueError("Surface weights must be positive and match response rows.")
    if not (
        np.all(np.isfinite(left))
        and np.all(np.isfinite(right))
        and np.all(np.isfinite(weights))
    ):
        raise ValueError("Response matrices and weights must be finite.")

    difference = left - right
    weighted_error_by_mode = np.einsum(
        "s,sm,sm->m",
        weights,
        difference,
        difference,
    )
    weighted_reference_by_mode = np.einsum(
        "s,sm,sm->m",
        weights,
        right,
        right,
    )
    if np.any(weighted_reference_by_mode <= 0.0):
        raise ValueError("Every reference response mode must have nonzero weighted norm.")
    relative_by_mode = np.sqrt(
        weighted_error_by_mode / weighted_reference_by_mode
    )
    total_reference = float(np.sum(weighted_reference_by_mode))
    return {
        "weighted_relative_frobenius": float(
            np.sqrt(float(np.sum(weighted_error_by_mode)) / total_reference)
        ),
        "weighted_relative_by_mode": relative_by_mode.tolist(),
        "maximum_weighted_relative_mode": float(np.max(relative_by_mode)),
        "weighted_rmse_by_mode": np.sqrt(
            weighted_error_by_mode / float(np.sum(weights))
        ).tolist(),
    }


__all__ = [
    "Route2LocalizedPointChargeModes",
    "localized_point_charge_atom_jets_hartree",
    "molecular_dipole_response_from_density_coefficients",
    "select_farthest_exterior_point_charge_modes",
    "weighted_response_matrix_discrepancy",
]
