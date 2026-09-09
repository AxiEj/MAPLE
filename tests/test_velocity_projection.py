"""Regression tests for the shared mass-weighted rigid-motion projector."""

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from maple.function.dispatcher.md.utils import (
    apply_runtime_motion_projection,
    initialize_velocities,
    remove_rigid_body_rotation,
)
from maple.function.utility.rigid_body import (
    mass_weighted_rigid_basis,
    project_rigid_body_velocities,
)


def _near_linear_triatomic(offset: float, *, rotated: bool = False) -> Atoms:
    atoms = Atoms(
        "HOH",
        positions=[[-1.0, 0.0, 0.0], [0.0, offset, 0.0], [1.0, 0.0, 0.0]],
        masses=[1.0, 16.0, 1.0],
    )
    if rotated:
        rotation = np.array([[1, -2, 2], [2, 2, 1], [-2, 1, 2]], dtype=float) / 3
        atoms.positions[:] = atoms.positions @ rotation.T
    return atoms


def _mass_weighted(atoms: Atoms, velocities: np.ndarray) -> np.ndarray:
    return (np.sqrt(atoms.get_masses())[:, None] * velocities).reshape(-1)


def _weak_rotation_velocity(atoms: Atoms) -> np.ndarray:
    rigid = mass_weighted_rigid_basis(atoms)
    weighted = rigid[:, -1]
    return weighted.reshape((-1, 3)) / np.sqrt(atoms.get_masses())[:, None]


@pytest.mark.parametrize("offset", [1e-6, 1e-8, 1e-9])
@pytest.mark.parametrize("rotated", [False, True])
def test_near_linear_rigid_velocity_is_removed(offset: float, rotated: bool) -> None:
    atoms = _near_linear_triatomic(offset, rotated=rotated)
    assert mass_weighted_rigid_basis(atoms).shape[1] == 6
    velocity = _weak_rotation_velocity(atoms)

    projected = remove_rigid_body_rotation(atoms, velocity)

    residual = np.vdot(_mass_weighted(atoms, projected), _mass_weighted(atoms, projected))
    initial = np.vdot(_mass_weighted(atoms, velocity), _mass_weighted(atoms, velocity))
    assert residual / initial < 1e-24


@pytest.mark.parametrize("rotated", [False, True])
def test_rotation_projection_is_idempotent_and_preserves_com(rotated: bool) -> None:
    atoms = _near_linear_triatomic(1e-9, rotated=rotated)
    rng = np.random.default_rng(17)
    velocity = rng.normal(size=(len(atoms), 3))
    masses = atoms.get_masses()
    com_before = np.average(velocity, axis=0, weights=masses)

    once = remove_rigid_body_rotation(atoms, velocity)
    twice = remove_rigid_body_rotation(atoms, once)

    np.testing.assert_allclose(twice, once, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(
        np.average(once, axis=0, weights=masses), com_before, atol=2e-14, rtol=0.0
    )


@pytest.mark.parametrize("rotated", [False, True])
def test_initialization_and_runtime_use_same_rigid_subspace(rotated: bool) -> None:
    atoms = _near_linear_triatomic(1e-9, rotated=rotated)
    basis = mass_weighted_rigid_basis(atoms)

    initialized = initialize_velocities(
        atoms,
        temperature=300.0,
        remove_com=True,
        remove_angular=True,
        target_n_dof=3,
        rng=np.random.default_rng(23),
    )
    runtime, action = apply_runtime_motion_projection(
        atoms,
        np.random.default_rng(29).normal(size=(len(atoms), 3)),
        step=1,
        remove_angular_every=1,
    )

    assert action == "angular"
    np.testing.assert_allclose(basis.T @ _mass_weighted(atoms, initialized), 0.0, atol=2e-14)
    np.testing.assert_allclose(basis.T @ _mass_weighted(atoms, runtime), 0.0, atol=2e-14)


def test_periodic_and_fixed_atom_runtime_semantics_are_preserved() -> None:
    velocity = np.arange(9, dtype=float).reshape(3, 3)
    periodic = _near_linear_triatomic(1e-9)
    periodic.set_pbc(True)
    unchanged, action = apply_runtime_motion_projection(
        periodic, velocity, step=1, remove_angular_every=1
    )
    assert action == "none"
    np.testing.assert_array_equal(unchanged, velocity)

    anchored = _near_linear_triatomic(1e-9)
    anchored.set_constraint(FixAtoms(indices=[0]))
    frozen, action = apply_runtime_motion_projection(
        anchored, velocity, step=1, remove_angular_every=1
    )
    assert action == "none"
    np.testing.assert_array_equal(frozen[0], 0.0)
    np.testing.assert_array_equal(frozen[1:], velocity[1:])


@pytest.mark.parametrize("offset", [0.0, 1e-6, 1e-8, 1e-9])
def test_retained_rigid_basis_is_jointly_orthonormal(offset: float) -> None:
    atoms = _near_linear_triatomic(offset, rotated=True)
    basis = mass_weighted_rigid_basis(atoms)
    rank = 5 if offset == 0.0 else 6
    assert basis.shape == (9, rank)
    np.testing.assert_allclose(basis.T @ basis, np.eye(rank), atol=2e-14, rtol=0)
    projector = np.eye(9) - basis @ basis.T
    np.testing.assert_allclose(projector @ projector, projector, atol=2e-14, rtol=0)
    # The first three columns must still span translation, not just any 3D space.
    translations = np.tile(np.eye(3), (3, 1)) * np.repeat(np.sqrt(atoms.get_masses()), 3)[:, None]
    translations /= np.sqrt(atoms.get_masses().sum())
    np.testing.assert_allclose(
        basis[:, :3] @ basis[:, :3].T, translations @ translations.T,
        atol=2e-14, rtol=0,
    )


def test_joint_basis_under_seeded_rigid_transforms_and_permutations() -> None:
    rng = np.random.default_rng(83)
    for offset in (0.0, 1e-6, 1e-8, 1e-9):
        for _ in range(30):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            rotation[:, 0] *= np.linalg.det(rotation)
            atoms = _near_linear_triatomic(offset)
            atoms.positions[:] = atoms.positions @ rotation.T + rng.uniform(-2, 2, 3)
            atoms = atoms[rng.permutation(len(atoms))]
            basis = mass_weighted_rigid_basis(atoms)
            rank = 5 if offset == 0.0 else 6
            assert basis.shape[1] == rank
            np.testing.assert_allclose(basis.T @ basis, np.eye(rank), atol=2e-14, rtol=0)
            velocities = rng.normal(size=(len(atoms), 3))
            once = project_rigid_body_velocities(
                atoms, velocities, remove_translation=True, remove_rotation=True,
            )
            twice = project_rigid_body_velocities(
                atoms, once, remove_translation=True, remove_rotation=True,
            )
            np.testing.assert_allclose(once, twice, atol=2e-14, rtol=0)
            rotation_only = remove_rigid_body_rotation(atoms, velocities)
            np.testing.assert_allclose(
                np.average(rotation_only, axis=0, weights=atoms.get_masses()),
                np.average(velocities, axis=0, weights=atoms.get_masses()),
                atol=2e-14, rtol=0,
            )
