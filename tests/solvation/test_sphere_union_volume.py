from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.sphere_union_volume import (
    volume_and_gradient,
)


def _sphere_volume(radius: float) -> float:
    return 4.0 * math.pi * radius**3 / 3.0


def _two_sphere_union(r1: float, r2: float, distance: float) -> float:
    overlap = (
        math.pi
        * (r1 + r2 - distance) ** 2
        * (distance**2 + 2.0 * distance * (r1 + r2) - 3.0 * (r1 - r2) ** 2)
        / (12.0 * distance)
    )
    return _sphere_volume(r1) + _sphere_volume(r2) - overlap


def test_one_sphere_has_analytic_volume_and_zero_center_gradient():
    result = volume_and_gradient([[3.2, -1.7, 0.4]], [1.8])

    assert result.volume == pytest.approx(_sphere_volume(1.8), rel=2e-10)
    assert result.gradient == pytest.approx(np.zeros((1, 3)), abs=2e-10)
    assert result.diagnostics["divergence_volume_error"] < 1e-9


def test_disjoint_and_contained_spheres_have_expected_volume():
    disjoint = volume_and_gradient([[0, 0, 0], [4, 0, 0]], [1.0, 1.5])
    contained = volume_and_gradient([[0, 0, 0], [0.2, 0, 0]], [2.0, 0.5])

    assert disjoint.volume == pytest.approx(
        _sphere_volume(1.0) + _sphere_volume(1.5), rel=2e-10
    )
    assert contained.volume == pytest.approx(_sphere_volume(2.0), rel=2e-10)
    assert disjoint.diagnostics["disjoint_pairs"] == 1
    assert contained.diagnostics["contained_pairs"] == 1
    assert contained.gradient[1] == pytest.approx(np.zeros(3), abs=2e-10)


@pytest.mark.parametrize(
    ("r1", "r2", "distance"),
    [(1.4, 1.4, 1.1), (1.9, 1.2, 1.3)],
)
def test_two_overlapping_spheres_match_closed_form(r1, r2, distance):
    result = volume_and_gradient([[0, 0, 0], [distance, 0, 0]], [r1, r2])

    assert result.volume == pytest.approx(_two_sphere_union(r1, r2, distance), rel=3e-9)
    assert result.gradient[:, 1:] == pytest.approx(np.zeros((2, 2)), abs=2e-9)
    assert result.gradient.sum(axis=0) == pytest.approx(np.zeros(3), abs=2e-9)


def test_all_center_derivatives_match_finite_difference():
    positions = np.array(
        [[-0.8, 0.1, -0.3], [0.7, -0.2, 0.4], [0.1, 1.0, 0.2], [-0.2, -0.5, 1.1]]
    )
    radii = np.array([1.35, 1.1, 0.95, 0.8])
    result = volume_and_gradient(positions, radii, rtol=2e-9, atol=2e-10)
    step = 2e-5
    finite_difference = np.empty_like(positions)
    for atom in range(len(radii)):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            finite_difference[atom, axis] = (
                volume_and_gradient(plus, radii, rtol=2e-9, atol=2e-10).volume
                - volume_and_gradient(minus, radii, rtol=2e-9, atol=2e-10).volume
            ) / (2.0 * step)

    assert result.gradient == pytest.approx(finite_difference, rel=3e-5, abs=3e-5)
    assert result.gradient.sum(axis=0) == pytest.approx(np.zeros(3), abs=2e-8)


def test_rigid_motion_and_permutation_covariance():
    positions = np.array([[-0.4, 0.2, 0.1], [0.8, -0.3, 0.5], [0.0, 0.9, -0.6]])
    radii = np.array([1.2, 1.0, 0.85])
    axis = np.array([1.0, -2.0, 0.7])
    axis /= np.linalg.norm(axis)
    angle = 0.61
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = (
        np.eye(3) * math.cos(angle)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * cross
    )
    reference = volume_and_gradient(positions, radii)
    moved = volume_and_gradient(positions @ rotation.T + [10.0, -7.0, 2.5], radii)
    order = np.array([2, 0, 1])
    permuted = volume_and_gradient(positions[order], radii[order])

    assert moved.volume == pytest.approx(reference.volume, rel=2e-10)
    assert moved.gradient == pytest.approx(reference.gradient @ rotation.T, abs=2e-8)
    assert permuted.volume == pytest.approx(reference.volume, rel=2e-10)
    assert permuted.gradient == pytest.approx(reference.gradient[order], abs=2e-8)


def test_external_and_internal_tangencies_are_classified():
    external = volume_and_gradient([[0, 0, 0], [2, 0, 0]], [1.0, 1.0])
    internal = volume_and_gradient([[0, 0, 0], [1, 0, 0]], [2.0, 1.0])

    assert external.volume == pytest.approx(2.0 * _sphere_volume(1.0), rel=2e-10)
    assert internal.volume == pytest.approx(_sphere_volume(2.0), rel=2e-10)
    assert external.diagnostics["external_tangent_pairs"] == 1
    assert internal.diagnostics["internal_tangent_pairs"] == 1


def test_tolerance_refinement_is_stable_and_reports_error():
    positions = [[0, 0, 0], [1.0, 0.2, 0.1], [-0.3, 0.9, 0.4]]
    radii = [1.3, 1.1, 0.9]
    coarse = volume_and_gradient(positions, radii, rtol=1e-6, atol=1e-8)
    fine = volume_and_gradient(positions, radii, rtol=1e-10, atol=1e-12)

    assert coarse.volume == pytest.approx(fine.volume, rel=2e-6, abs=2e-7)
    assert coarse.gradient == pytest.approx(fine.gradient, rel=2e-5, abs=2e-6)
    assert fine.quadrature_error >= 0.0
    assert fine.diagnostics["evaluations"] > 0
    assert fine.diagnostics["quadrature_status"] == 0
    assert fine.diagnostics["quadrature_interval_count"] > 0
    assert fine.diagnostics["breakpoint_count"] >= 2 * len(radii)


def test_buried_breakpoint_pruning_preserves_volume_and_gradient():
    positions = np.array(
        [[-0.8, 0.1, -0.3], [0.7, -0.2, 0.4], [0.1, 1.0, 0.2], [-0.2, -0.5, 1.1]]
    )
    radii = np.array([1.35, 1.1, 0.95, 0.8])
    pruned = volume_and_gradient(positions, radii, prune_buried_breakpoints=True)
    audit = volume_and_gradient(positions, radii, prune_buried_breakpoints=False)

    assert pruned.volume == pytest.approx(audit.volume, rel=2e-10, abs=2e-10)
    assert pruned.gradient == pytest.approx(audit.gradient, rel=2e-9, abs=2e-9)
    assert (
        pruned.diagnostics["breakpoint_count"] < audit.diagnostics["breakpoint_count"]
    )
    assert pruned.diagnostics["buried_breakpoint_pruning"] is True
    assert audit.diagnostics["buried_breakpoint_pruning"] is False


def test_real_23_sphere_pruning_preserves_result_and_reduces_work():
    path = (
        Path(__file__).resolve().parents[2]
        / ".omx/benchmarks/route1-numerical-repair-20260905/reference"
        / "nonpolar-parameters.json"
    )
    if not path.is_file():
        pytest.skip("local audited 23-sphere reference artifact is unavailable")
    payload = json.loads(path.read_text())
    positions = np.asarray(payload["positions_angstrom"])
    radii = np.asarray(payload["rmin_angstrom"]) + 1.3

    started = time.perf_counter()
    pruned = volume_and_gradient(positions, radii, prune_buried_breakpoints=True)
    pruned_seconds = time.perf_counter() - started
    started = time.perf_counter()
    audit = volume_and_gradient(positions, radii, prune_buried_breakpoints=False)
    audit_seconds = time.perf_counter() - started

    assert pruned.volume == pytest.approx(568.80852220818, rel=2e-11)
    assert pruned.volume == pytest.approx(audit.volume, rel=2e-11, abs=2e-10)
    assert pruned.gradient == pytest.approx(audit.gradient, rel=2e-8, abs=2e-8)
    assert pruned.diagnostics["breakpoint_count"] < (
        audit.diagnostics["breakpoint_count"] / 5
    )
    assert pruned.diagnostics["evaluations"] < audit.diagnostics["evaluations"]
    # Timings are evidence, not a flaky pass/fail threshold.
    assert pruned_seconds > 0.0
    assert audit_seconds > 0.0


def test_unconverged_quadrature_fails_closed_at_subdivision_limit():
    positions = [[0, 0, 0], [1.0, 0.2, 0.1], [-0.3, 0.9, 0.4]]
    radii = [1.3, 1.1, 0.9]

    with pytest.raises(RuntimeError, match=r"quadrature failed \(status 1\)"):
        volume_and_gradient(positions, radii, rtol=1e-12, atol=1e-14, limit=1)


@pytest.mark.parametrize(
    ("positions", "radii", "exception", "message"),
    [
        ([[0, 0]], [1.0], ValueError, "shape"),
        ([[0, 0, 0]], [0.0], ValueError, "positive"),
        ([[0, 0, 0]], [math.inf], ValueError, "finite"),
        ([[0, 0, 0], [0, 0, 0]], [1.0, 1.0], ValueError, "duplicate"),
        ([[True, 0, 0]], [1.0], TypeError, "positions"),
        ([[0, 0, 0]], [False], TypeError, "radii"),
    ],
)
def test_invalid_or_degenerate_geometry_fails_closed(
    positions, radii, exception, message
):
    with pytest.raises(exception, match=message):
        volume_and_gradient(positions, radii)


@pytest.mark.parametrize(
    ("keyword", "value", "exception", "message"),
    [
        ("rtol", True, TypeError, "rtol"),
        ("rtol", "not-a-number", ValueError, "rtol"),
        ("atol", math.nan, ValueError, "atol"),
        ("limit", True, TypeError, "limit"),
        ("limit", 1.5, TypeError, "limit"),
        ("prune_buried_breakpoints", "yes", TypeError, "prune"),
    ],
)
def test_malformed_quadrature_controls_fail_closed(keyword, value, exception, message):
    with pytest.raises(exception, match=message):
        volume_and_gradient([[0, 0, 0]], [1.0], **{keyword: value})
