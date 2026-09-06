from __future__ import annotations

import numpy as np
import pytest
from ase.units import Bohr

from maple.solvation.reference.exterior_probe_partition import (
    DENSE_EXTERIOR_AUDIT_ROTATION,
    DENSE_EXTERIOR_LEBEDEV_POINTS,
    build_dense_exterior_probe_partition,
)


def _water():
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.927297, 0.0]]
    )
    radii = np.asarray([1.52, 1.20, 1.20])
    return positions, radii


def test_dense_exterior_probe_has_disjoint_overdetermined_fit_and_audit_parts() -> None:
    pytest.importorskip("pyscf")
    positions, radii = _water()
    probes = build_dense_exterior_probe_partition(
        positions,
        radii,
        clearance_angstrom=1.0,
    )

    fit = probes.mask("fit")
    audit = probes.mask("audit")
    assert not np.any(fit & audit)
    assert np.all(fit | audit)
    assert np.count_nonzero(fit) > 8 * len(positions)
    assert np.count_nonzero(audit) > 8 * len(positions)
    assert probes.candidate_count_per_partition == (
        len(positions) * DENSE_EXTERIOR_LEBEDEV_POINTS
    )
    points_angstrom = probes.surface_points_bohr * Bohr
    expanded = radii + 1.0
    distances = np.linalg.norm(
        points_angstrom[:, None, :] - positions[None, :, :],
        axis=2,
    )
    assert np.all(distances + 2.1e-12 >= expanded[None, :])
    assert not probes.surface_points_bohr.flags.writeable
    assert not probes.partition_indices.flags.writeable


def test_dense_exterior_probe_is_translation_covariant() -> None:
    pytest.importorskip("pyscf")
    positions, radii = _water()
    translation = np.asarray([0.41, -0.29, 0.18])
    reference = build_dense_exterior_probe_partition(
        positions,
        radii,
        clearance_angstrom=1.0,
    )
    shifted = build_dense_exterior_probe_partition(
        positions + translation,
        radii,
        clearance_angstrom=1.0,
    )

    np.testing.assert_allclose(
        shifted.surface_points_bohr,
        reference.surface_points_bohr + translation / Bohr,
        rtol=0.0,
        atol=3.0e-15,
    )
    np.testing.assert_array_equal(
        shifted.parent_atom_indices,
        reference.parent_atom_indices,
    )
    np.testing.assert_array_equal(
        shifted.partition_indices,
        reference.partition_indices,
    )
    np.testing.assert_allclose(
        shifted.quadrature_weights,
        reference.quadrature_weights,
        rtol=0.0,
        atol=0.0,
    )


def test_dense_exterior_audit_rotation_is_a_generic_proper_rotation() -> None:
    rotation = DENSE_EXTERIOR_AUDIT_ROTATION

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=3.0e-16)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=6.0e-16)
    assert not np.allclose(np.abs(rotation), np.eye(3), atol=1.0e-14)
