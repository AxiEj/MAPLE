from __future__ import annotations

from pathlib import Path

import numpy as np

from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
    load_atomic_reference_density_asset,
)
from maple.solvation.coupling.stockholder_partition import (
    PromolecularStockholderPartition,
)


ROOT = Path(__file__).resolve().parents[2]


def _provider() -> PromolecularStockholderPartition:
    asset = load_atomic_reference_density_asset(
        table_path=ROOT / (
            "docs/implicit-solvation/benchmarks/"
            "route2-rhodrop-atomic-reference-gaussian-mixture-v1.npz"
        ),
        manifest_path=ROOT / (
            "docs/implicit-solvation/benchmarks/"
            "route2-rhodrop-atomic-reference-gaussian-mixture-v1.json"
        ),
    )
    return PromolecularStockholderPartition(asset)


def _geometry():
    numbers = np.asarray([8, 1, 1], dtype=np.int64)
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.45, 1.75, 0.0]],
        dtype=np.float64,
    )
    points = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [1.0, -0.2, 0.4],
            [-0.8, 0.9, -0.1],
            [8.0, -3.0, 2.0],
            [1.0e4, -2.0e4, 3.0e4],
        ],
        dtype=np.float64,
    )
    return numbers, positions, points


def test_stockholder_partition_is_normalized_and_far_field_stable() -> None:
    provider = _provider()
    numbers, positions, points = _geometry()

    result = provider.evaluate(
        points_bohr=points,
        atomic_numbers=numbers,
        positions_bohr=positions,
    )

    assert np.all(np.isfinite(result.weights))
    assert np.all(result.weights >= 0.0)
    np.testing.assert_allclose(
        np.sum(result.weights, axis=1),
        1.0,
        rtol=0.0,
        atol=4.0e-15,
    )
    np.testing.assert_allclose(
        np.sum(result.weight_spatial_gradients_bohr_inv, axis=1),
        0.0,
        rtol=0.0,
        atol=2.0e-13,
    )


def test_stockholder_spatial_gradient_and_vjps_match_finite_difference() -> None:
    provider = _provider()
    numbers, positions, points = _geometry()
    points = points[:4]
    result = provider.evaluate(
        points_bohr=points,
        atomic_numbers=numbers,
        positions_bohr=positions,
    )
    generator = np.random.default_rng(20260824)
    point_direction = generator.normal(size=points.shape)
    nuclear_direction = generator.normal(size=positions.shape)
    cotangent = generator.normal(size=result.weights.shape)
    step = 2.0e-6

    plus_points = provider.evaluate(
        points_bohr=points + step * point_direction,
        atomic_numbers=numbers,
        positions_bohr=positions,
    ).weights
    minus_points = provider.evaluate(
        points_bohr=points - step * point_direction,
        atomic_numbers=numbers,
        positions_bohr=positions,
    ).weights
    point_fd = (plus_points - minus_points) / (2.0 * step)
    point_jvp = np.einsum(
        "max,mx->ma",
        result.weight_spatial_gradients_bohr_inv,
        point_direction,
        optimize=True,
    )
    np.testing.assert_allclose(point_jvp, point_fd, rtol=2.0e-8, atol=2.0e-9)

    plus_nuclear = provider.evaluate(
        points_bohr=points,
        atomic_numbers=numbers,
        positions_bohr=positions + step * nuclear_direction,
    ).weights
    minus_nuclear = provider.evaluate(
        points_bohr=points,
        atomic_numbers=numbers,
        positions_bohr=positions - step * nuclear_direction,
    ).weights
    nuclear_fd = (plus_nuclear - minus_nuclear) / (2.0 * step)
    nuclear_vjp = result.nuclear_vjp(cotangent)
    np.testing.assert_allclose(
        float(np.vdot(cotangent, nuclear_fd)),
        float(np.vdot(nuclear_vjp, nuclear_direction)),
        rtol=3.0e-8,
        atol=3.0e-9,
    )
    point_vjp = result.point_vjp(cotangent)
    np.testing.assert_allclose(
        float(np.vdot(cotangent, point_fd)),
        float(np.vdot(point_vjp, point_direction)),
        rtol=3.0e-8,
        atol=3.0e-9,
    )


def test_stockholder_partition_is_rotation_translation_and_permutation_covariant() -> None:
    provider = _provider()
    numbers, positions, points = _geometry()
    points = points[:4]
    base = provider.evaluate(
        points_bohr=points,
        atomic_numbers=numbers,
        positions_bohr=positions,
    )
    angle = 0.731
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    shift = np.asarray([2.3, -1.4, 0.7])
    moved = provider.evaluate(
        points_bohr=points @ rotation.T + shift,
        atomic_numbers=numbers,
        positions_bohr=positions @ rotation.T + shift,
    )
    np.testing.assert_allclose(moved.weights, base.weights, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        moved.weight_spatial_gradients_bohr_inv,
        base.weight_spatial_gradients_bohr_inv @ rotation.T,
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    permutation = np.asarray([2, 0, 1])
    permuted = provider.evaluate(
        points_bohr=points,
        atomic_numbers=numbers[permutation],
        positions_bohr=positions[permutation],
    )
    np.testing.assert_allclose(
        permuted.weights,
        base.weights[:, permutation],
        rtol=0.0,
        atol=2.0e-15,
    )
