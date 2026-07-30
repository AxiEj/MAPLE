from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_atomic_displacement_response import (
    BOHR_ANGSTROM,
    Route2V0AtomicDisplacementResponseTable,
    atomic_induced_dipoles,
)


def _table() -> Route2V0AtomicDisplacementResponseTable:
    return Route2V0AtomicDisplacementResponseTable(
        radial_grid_bohr=np.asarray([0.0, 1.0, 2.0, 8.0]),
        enclosed_electrons_by_atomic_number={
            1: np.asarray([0.0, 0.4, 0.8, 1.0]),
            6: np.asarray([0.0, 2.0, 4.0, 6.0]),
        },
        table_sha256="a" * 64,
        manifest_sha256="b" * 64,
    )


def test_atomic_partition_preserves_the_molecular_induced_dipole():
    weights = np.stack((0.25 * np.eye(3), 0.75 * np.eye(3)))
    molecular = np.asarray([0.7, -1.2, 0.4])

    atomic = atomic_induced_dipoles(weights, molecular)

    np.testing.assert_allclose(np.sum(atomic, axis=0), molecular, atol=1.0e-15)


def test_atomic_displacement_source_has_the_exact_exterior_dipole_limit():
    table = _table()
    molecular = np.asarray([1.0, 0.0, 0.0])
    point = np.asarray([[100.0, 0.0, 0.0]])

    potential = table.induced_potential(
        point,
        np.asarray([1]),
        np.zeros((1, 3)),
        np.eye(3)[None, :, :],
        molecular,
    )

    np.testing.assert_allclose(potential, np.asarray([1.0e-4]), atol=1.0e-15)


def test_atomic_displacement_source_is_translation_covariant_and_linear():
    table = _table()
    points = np.asarray([[7.0, -3.0, 2.0], [4.0, 5.0, -6.0]])
    positions = np.asarray([[0.2, -0.3, 0.4], [1.1, 0.7, -0.5]])
    numbers = np.asarray([1, 6])
    weights = np.stack((0.4 * np.eye(3), 0.6 * np.eye(3)))
    molecular = np.asarray([0.8, -0.4, 0.3])

    reference = table.induced_potential(
        points,
        numbers,
        positions,
        weights,
        molecular,
    )
    shifted = table.induced_potential(
        points + np.asarray([0.7, -0.4, 0.2]) / BOHR_ANGSTROM,
        numbers,
        positions + np.asarray([0.7, -0.4, 0.2]),
        weights,
        molecular,
    )
    doubled = table.induced_potential(
        points,
        numbers,
        positions,
        weights,
        2.0 * molecular,
    )

    np.testing.assert_allclose(shifted, reference, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(doubled, 2.0 * reference, rtol=0.0, atol=2.0e-15)


def test_atomic_displacement_source_rejects_nonconserving_partitions_and_unknown_atoms():
    table = _table()
    with pytest.raises(ValueError, match="do not preserve"):
        atomic_induced_dipoles(
            np.stack((0.4 * np.eye(3), 0.4 * np.eye(3))),
            np.asarray([1.0, 0.0, 0.0]),
        )

    with pytest.raises(ValueError, match="no frozen reference"):
        table.induced_potential(
            np.asarray([[3.0, 0.0, 0.0]]),
            np.asarray([8]),
            np.zeros((1, 3)),
            np.eye(3)[None, :, :],
            np.asarray([1.0, 0.0, 0.0]),
        )


def test_atomic_displacement_table_rejects_nonphysical_enclosed_electron_data():
    with pytest.raises(ValueError, match="monotone"):
        Route2V0AtomicDisplacementResponseTable(
            radial_grid_bohr=np.asarray([0.0, 1.0, 2.0]),
            enclosed_electrons_by_atomic_number={1: np.asarray([0.0, 1.2, 1.0])},
            table_sha256="a" * 64,
            manifest_sha256="b" * 64,
        )
