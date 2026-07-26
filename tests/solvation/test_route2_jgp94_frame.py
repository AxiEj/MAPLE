from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

BENCHMARK_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "implicit-solvation"
    / "benchmarks"
)
if str(BENCHMARK_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIRECTORY))

from route2_jgp94_frame import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    body_dipoles,
    build_jgp94_frame,
    jgp94_frame_vjp,
)


POSITIONS = np.asarray(
    [
        [-1.4, -0.2, 0.3],
        [-0.1, 0.7, -0.4],
        [1.2, -0.5, 0.2],
        [0.4, 1.5, 0.8],
    ],
    dtype=float,
)
NUCLEAR_CHARGES = np.asarray([6.0, 8.0, 7.0, 1.0], dtype=float)
DIPOLES = np.asarray(
    [
        [0.3, -0.1, 0.2],
        [-0.4, 0.5, 0.1],
        [0.2, 0.1, -0.3],
        [-0.1, -0.2, 0.4],
    ],
    dtype=float,
)


def _loss(
    positions: np.ndarray,
    dipoles: np.ndarray,
    body_position_cotangent: np.ndarray,
    body_dipole_cotangent: np.ndarray,
    *,
    reference_orientation: np.ndarray,
) -> float:
    frame = build_jgp94_frame(
        positions,
        NUCLEAR_CHARGES,
        minimum_relative_eigengap=0.01,
        reference_orientation=reference_orientation,
        minimum_reference_axis_overlap=0.999,
    )
    return float(
        np.vdot(body_position_cotangent, frame.body_positions)
        + np.vdot(body_dipole_cotangent, body_dipoles(dipoles, frame))
    )


def test_jgp94_frame_vjp_matches_all_coordinate_and_dipole_finite_differences():
    frame = build_jgp94_frame(
        POSITIONS,
        NUCLEAR_CHARGES,
        minimum_relative_eigengap=0.01,
    )
    random = np.random.default_rng(20260726)
    body_position_cotangent = random.normal(size=POSITIONS.shape)
    body_dipole_cotangent = random.normal(size=DIPOLES.shape)
    analytic = jgp94_frame_vjp(
        frame,
        NUCLEAR_CHARGES,
        DIPOLES,
        body_position_cotangent,
        body_dipole_cotangent,
    )

    step = 1.0e-5
    position_finite_difference = np.empty_like(POSITIONS)
    for atom_index in range(POSITIONS.shape[0]):
        for axis_index in range(3):
            displaced = []
            for sign in (-1.0, 1.0):
                coordinates = POSITIONS.copy()
                coordinates[atom_index, axis_index] += sign * step
                displaced.append(
                    _loss(
                        coordinates,
                        DIPOLES,
                        body_position_cotangent,
                        body_dipole_cotangent,
                        reference_orientation=frame.orientation,
                    )
                )
            position_finite_difference[atom_index, axis_index] = (
                displaced[1] - displaced[0]
            ) / (2.0 * step)

    dipole_finite_difference = np.empty_like(DIPOLES)
    for atom_index in range(DIPOLES.shape[0]):
        for axis_index in range(3):
            displaced = []
            for sign in (-1.0, 1.0):
                dipoles = DIPOLES.copy()
                dipoles[atom_index, axis_index] += sign * step
                displaced.append(
                    _loss(
                        POSITIONS,
                        dipoles,
                        body_position_cotangent,
                        body_dipole_cotangent,
                        reference_orientation=frame.orientation,
                    )
                )
            dipole_finite_difference[atom_index, axis_index] = (
                displaced[1] - displaced[0]
            ) / (2.0 * step)

    np.testing.assert_allclose(
        analytic.position_cotangent,
        position_finite_difference,
        atol=2.0e-8,
        rtol=2.0e-8,
    )
    np.testing.assert_allclose(
        analytic.dipole_cotangent,
        dipole_finite_difference,
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    np.testing.assert_allclose(
        np.sum(analytic.position_cotangent, axis=0),
        0.0,
        atol=2.0e-14,
        rtol=0.0,
    )
    assert np.linalg.norm(analytic.orientation_position_cotangent) > 1.0e-3


def test_jgp94_frame_uses_a_proper_local_signed_axis_gauge():
    frame = build_jgp94_frame(
        POSITIONS,
        NUCLEAR_CHARGES,
        minimum_relative_eigengap=0.01,
    )
    signed_reference = frame.orientation @ np.diag([-1.0, -1.0, 1.0])
    aligned = build_jgp94_frame(
        POSITIONS,
        NUCLEAR_CHARGES,
        minimum_relative_eigengap=0.01,
        reference_orientation=signed_reference,
        minimum_reference_axis_overlap=0.999999999999,
    )

    np.testing.assert_allclose(
        aligned.orientation,
        signed_reference,
        atol=1.0e-14,
        rtol=0.0,
    )
    assert aligned.reference_axis_overlaps is not None
    np.testing.assert_allclose(
        aligned.reference_axis_overlaps,
        1.0,
        atol=1.0e-14,
        rtol=0.0,
    )
    assert np.linalg.det(aligned.orientation) == pytest.approx(1.0)


def test_jgp94_frame_fails_closed_at_an_eigenvalue_degeneracy():
    linear_positions = np.asarray(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=float,
    )

    with pytest.raises(RuntimeError, match="nondegenerate eigengap"):
        build_jgp94_frame(
            linear_positions,
            np.ones(2),
            minimum_relative_eigengap=0.01,
        )
