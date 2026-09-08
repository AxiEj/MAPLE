"""Mass-weighted rigid-body geometry classification."""

import numpy as np
from ase import Atoms


def mass_weighted_rigid_basis(atoms: Atoms) -> np.ndarray:
    """Return an orthonormal rigid-motion basis for the system geometry."""
    masses = np.asarray(atoms.get_masses(), dtype=float)
    positions = np.asarray(atoms.get_positions(), dtype=float)
    n_atoms = len(atoms)
    include_rotations = not np.any(atoms.get_pbc())
    candidates = np.zeros((3 * n_atoms, 6 if include_rotations else 3))

    sqrt_masses = np.sqrt(masses)
    for atom_index, sqrt_mass in enumerate(sqrt_masses):
        block = slice(3 * atom_index, 3 * atom_index + 3)
        candidates[block, :3] = sqrt_mass * np.eye(3)

    if include_rotations and n_atoms:
        center = np.average(positions, axis=0, weights=masses)
        axes = np.eye(3)
        for atom_index, (position, sqrt_mass) in enumerate(
            zip(positions - center, sqrt_masses)
        ):
            block = slice(3 * atom_index, 3 * atom_index + 3)
            for axis_index, axis in enumerate(axes):
                candidates[block, 3 + axis_index] = (
                    sqrt_mass * np.cross(axis, position)
                )

    left_vectors, singular_values, _ = np.linalg.svd(candidates, full_matrices=False)
    if not singular_values.size:
        return np.empty((3 * n_atoms, 0))
    tolerance = (
        max(candidates.shape)
        * np.finfo(candidates.dtype).eps
        * singular_values[0]
    )
    return left_vectors[:, singular_values > tolerance]


def rotational_dof(atoms: Atoms) -> int:
    """Return the rigid rotational dimension: 0, 2, or 3."""
    if np.any(atoms.get_pbc()):
        return 0
    return max(mass_weighted_rigid_basis(atoms).shape[1] - 3, 0)


def is_linear_geometry(atoms: Atoms) -> bool:
    """Return whether the geometry has exactly two rigid rotations."""
    return rotational_dof(atoms) == 2
