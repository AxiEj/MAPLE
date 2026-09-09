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
from maple.function.utility.rigid_body import mass_weighted_rigid_basis


def _near_linear_triatomic(offset: float) -> Atoms:
    atoms = Atoms(
        "HOH",
        positions=[[-1.0, 0.0, 0.0], [0.0, offset, 0.0], [1.0, 0.0, 0.0]],
        masses=[1.0, 16.0, 1.0],
    )
    return atoms


def _mass_weighted(atoms: Atoms, velocities: np.ndarray) -> np.ndarray:
    return (np.sqrt(atoms.get_masses())[:, None] * velocities).reshape(-1)


def _weak_rotation_velocity(atoms: Atoms) -> np.ndarray:
    rigid = mass_weighted_rigid_basis(atoms)
    weighted = rigid[:, -1]
    return weighted.reshape((-1, 3)) / np.sqrt(atoms.get_masses())[:, None]


@pytest.mark.parametrize("offset", [1e-6, 1e-8, 1e-9])
def test_near_linear_rigid_velocity_is_removed(offset: float) -> None:
    atoms = _near_linear_triatomic(offset)
    assert mass_weighted_rigid_basis(atoms).shape[1] == 6
    velocity = _weak_rotation_velocity(atoms)

    projected = remove_rigid_body_rotation(atoms, velocity)

    residual = np.vdot(_mass_weighted(atoms, projected), _mass_weighted(atoms, projected))
    initial = np.vdot(_mass_weighted(atoms, velocity), _mass_weighted(atoms, velocity))
    assert residual / initial < 1e-24


def test_rotation_projection_is_idempotent_and_preserves_com() -> None:
    atoms = _near_linear_triatomic(1e-9)
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


def test_initialization_and_runtime_use_same_rigid_subspace() -> None:
    atoms = _near_linear_triatomic(1e-9)
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
