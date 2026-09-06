from __future__ import annotations

import numpy as np
from ase.units import Hartree

from maple.solvation.coupling.exact_gto import (
    FixedSurfaceGeometry,
    MACEPolarRadialGTOCoupling,
)
from maple.solvation.coupling.gaussian_quadrupole import (
    gaussian_traceless_quadrupole_surface_operator,
)
from maple.solvation.reference.sector_balanced_response_modes import (
    select_sector_balanced_response_modes,
)


def _rotation() -> np.ndarray:
    axis = np.asarray([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    angle = 0.731
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


def _operators(positions: np.ndarray, points: np.ndarray):
    radial = MACEPolarRadialGTOCoupling().surface_operator(
        FixedSurfaceGeometry(positions, points)
    ) / Hartree
    l2 = gaussian_traceless_quadrupole_surface_operator(
        points_bohr=points,
        centers_angstrom=positions,
        sigma_angstrom=1.5,
    )
    return radial, l2


def test_sector_balanced_modes_preserve_prefix_rank_and_ignore_targets() -> None:
    generator = np.random.default_rng(20260827)
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [1.2, -0.3, 0.2], [-0.5, 1.1, -0.1]]
    )
    directions = generator.normal(size=(48, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    points = directions * generator.uniform(5.0, 7.0, size=(48, 1))
    radial, l2 = _operators(positions, points)
    result = select_sector_balanced_response_modes(
        radial_field_operator=radial,
        l2_field_operator=l2,
        fit_surface_indices=np.arange(40, dtype=np.int64),
        frozen_prefix_indices=np.asarray([2, 11, 17, 31], dtype=np.int64),
        mode_count=12,
    )

    assert result.source_surface_indices[:4] == (2, 11, 17, 31)
    assert len(result.source_surface_indices) == 12
    assert result.balanced_rank == 12
    assert result.target_used is False
    assert result.balanced_condition_number < 1.0e8
    assert 0.1 < result.radial_fraction_of_selected_balanced_norm < 0.9


def test_sector_balanced_mode_indices_are_rotation_and_atom_permutation_invariant() -> None:
    generator = np.random.default_rng(19)
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [1.1, -0.2, 0.4], [-0.3, 1.0, 0.2]]
    )
    directions = generator.normal(size=(52, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    points = directions * generator.uniform(5.2, 7.3, size=(52, 1))
    fit = np.arange(44, dtype=np.int64)
    prefix = np.asarray([3, 9, 21, 37], dtype=np.int64)
    radial, l2 = _operators(positions, points)
    reference = select_sector_balanced_response_modes(
        radial_field_operator=radial,
        l2_field_operator=l2,
        fit_surface_indices=fit,
        frozen_prefix_indices=prefix,
        mode_count=12,
    )

    rotation = _rotation()
    radial_rotated, l2_rotated = _operators(
        positions @ rotation.T,
        points @ rotation.T,
    )
    rotated = select_sector_balanced_response_modes(
        radial_field_operator=radial_rotated,
        l2_field_operator=l2_rotated,
        fit_surface_indices=fit,
        frozen_prefix_indices=prefix,
        mode_count=12,
    )
    permutation = np.asarray([2, 0, 1])
    radial_permuted, l2_permuted = _operators(positions[permutation], points)
    permuted = select_sector_balanced_response_modes(
        radial_field_operator=radial_permuted,
        l2_field_operator=l2_permuted,
        fit_surface_indices=fit,
        frozen_prefix_indices=prefix,
        mode_count=12,
    )

    assert rotated.source_surface_indices == reference.source_surface_indices
    assert permuted.source_surface_indices == reference.source_surface_indices
    np.testing.assert_allclose(
        rotated.balanced_singular_values,
        reference.balanced_singular_values,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        permuted.balanced_singular_values,
        reference.balanced_singular_values,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
