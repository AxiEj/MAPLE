"""Local rolling-probe intersections; not a completed multiatom SES."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.torch_continuum_ses_geometry import (
    contact_exposed_azimuth,
    pair_probe_contact_circle,
    pair_probe_exposed_arcs,
    triple_probe_centers,
)


def test_pair_contact_circle_is_on_both_expanded_spheres():
    positions = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64)
    intrinsic = torch.tensor([2.12, 2.12], dtype=torch.float64)
    circle = pair_probe_contact_circle(positions, intrinsic)
    expanded = intrinsic + 0.88
    assert torch.allclose(
        circle.center, torch.tensor([1.5, 0.0, 0.0], dtype=torch.float64)
    )
    assert abs(circle.radius.item() - math.sqrt(9.0 - 1.5**2)) < 1e-14
    assert torch.allclose(
        circle.axis, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    )
    for direction in (
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
    ):
        point = circle.center + circle.radius * direction
        assert (
            torch.max(
                torch.abs(torch.linalg.vector_norm(point - positions, dim=1) - expanded)
            )
            < 1e-13
        )


def test_triple_centers_match_equilateral_sphere_intersection():
    height = 3.0 * math.sqrt(3.0) / 2.0
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.5, height, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    intrinsic = torch.full((3,), 2.12, dtype=torch.float64)
    result = triple_probe_centers(positions, intrinsic, (0, 1, 2))
    assert result.centers.shape == (2, 3)
    assert result.locally_exposed.tolist() == [True, True]
    expected = torch.tensor(
        [[1.5, height / 3.0, -math.sqrt(6.0)], [1.5, height / 3.0, math.sqrt(6.0)]],
        dtype=torch.float64,
    )
    assert torch.max(torch.abs(result.centers - expected)) < 1e-12
    for center in result.centers:
        assert (
            torch.max(
                torch.abs(torch.linalg.vector_norm(center - positions, dim=1) - 3.0)
            )
            < 1e-12
        )
    gradient = torch.autograd.grad(result.centers[1, 2], positions, create_graph=True)[
        0
    ]
    hvp = torch.autograd.grad(gradient.square().sum(), positions)[0]
    assert torch.isfinite(gradient).all() and torch.isfinite(hvp).all()


def test_fourth_sphere_buries_only_one_triple_probe_center():
    height = 3.0 * math.sqrt(3.0) / 2.0
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [1.5, height, 0.0],
            [1.5, height / 3.0, math.sqrt(6.0) + 1.0],
        ],
        dtype=torch.float64,
    )
    intrinsic = torch.tensor([2.12, 2.12, 2.12, 1.12], dtype=torch.float64)
    result = triple_probe_centers(positions, intrinsic, (0, 1, 2))
    assert result.locally_exposed.tolist() == [True, False]


def test_nonintersecting_and_degenerate_triples_fail_closed():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.5, 3.0, 0.0]],
        dtype=torch.float64,
    )
    small = torch.full((3,), 0.5, dtype=torch.float64)
    empty = triple_probe_centers(positions, small, (0, 1, 2))
    assert empty.centers.shape == (0, 3)
    collinear = positions.clone()
    collinear[2] = torch.tensor([6.0, 0.0, 0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="collinear|degenerate"):
        triple_probe_centers(
            collinear, torch.full((3,), 2.12, dtype=torch.float64), (0, 1, 2)
        )


def test_triple_center_rotation_covariance_and_atom_permutation():
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [2.9, -0.1, 0.4], [1.2, 2.4, -0.2]],
        dtype=torch.float64,
    )
    intrinsic = torch.tensor([2.12, 1.88, 2.10], dtype=torch.float64)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    base = triple_probe_centers(positions, intrinsic, (0, 1, 2))
    turned = triple_probe_centers(positions @ rotation.T + 4.0, intrinsic, (0, 1, 2))
    permuted = triple_probe_centers(
        positions[[1, 2, 0]], intrinsic[[1, 2, 0]], (0, 1, 2)
    )
    assert base.centers.shape == (2, 3)
    for candidate in turned.centers:
        assert (
            torch.min(
                torch.linalg.vector_norm(
                    base.centers @ rotation.T + 4.0 - candidate, dim=1
                )
            )
            < 1e-12
        )
    for candidate in permuted.centers:
        assert (
            torch.min(torch.linalg.vector_norm(base.centers - candidate, dim=1)) < 1e-12
        )


def test_pair_probe_arc_is_clipped_by_third_expanded_sphere():
    height = 3.0 * math.sqrt(3.0) / 2.0
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.5, height, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    intrinsic = torch.full((3,), 2.12, dtype=torch.float64)
    arcs = pair_probe_exposed_arcs(positions, intrinsic, (0, 1))
    assert arcs.intervals.shape == (2, 2)
    angular_length = (arcs.intervals[:, 1] - arcs.intervals[:, 0]).sum()
    expected = 2.0 * math.pi - 2.0 * math.acos(1.0 / 3.0)
    assert abs(angular_length.item() - expected) < 1e-12
    centers = []
    for start, stop in arcs.intervals:
        angle = 0.5 * (start + stop)
        center = arcs.circle.center + arcs.circle.radius * (
            torch.cos(angle) * arcs.basis_u + torch.sin(angle) * arcs.basis_v
        )
        centers.append(center)
        assert torch.linalg.vector_norm(center - positions[2]) > intrinsic[2] + 0.88
    gradient = torch.autograd.grad(angular_length, positions)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[2].abs().max() > 0.0
    triple = triple_probe_centers(positions.detach(), intrinsic, (0, 1, 2))
    boundary_angles = (arcs.intervals[0, 1], arcs.intervals[1, 0])
    boundary_centers = torch.stack(
        [
            arcs.circle.center
            + arcs.circle.radius
            * (torch.cos(angle) * arcs.basis_u + torch.sin(angle) * arcs.basis_v)
            for angle in boundary_angles
        ]
    )
    for center in triple.centers:
        assert (
            torch.min(
                torch.linalg.vector_norm(boundary_centers.detach() - center, dim=1)
            )
            < 1e-12
        )


def test_pair_probe_arc_full_and_buried_limits():
    positions = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64)
    intrinsic = torch.tensor([2.12, 2.12], dtype=torch.float64)
    full = pair_probe_exposed_arcs(positions, intrinsic, (0, 1))
    assert full.intervals.shape == (1, 2)
    assert torch.allclose(
        full.intervals[0], torch.tensor([0.0, 2.0 * math.pi], dtype=torch.float64)
    )
    with_burier = torch.cat(
        (positions, torch.tensor([[1.5, 0.0, 0.0]], dtype=torch.float64))
    )
    buried = pair_probe_exposed_arcs(
        with_burier, torch.tensor([2.12, 2.12, 2.0], dtype=torch.float64), (0, 1)
    )
    assert buried.intervals.shape == (0, 2)


def test_atomic_contact_azimuth_matches_exact_single_occlusion_cap():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    intrinsic = torch.tensor([2.12, 2.12], dtype=torch.float64)
    intervals = contact_exposed_azimuth(
        positions, intrinsic, owner=0, polar_cosine=positions.new_tensor(0.0)
    )
    assert intervals.shape == (1, 2)
    assert (
        torch.max(
            torch.abs(
                intervals[0]
                - positions.new_tensor([math.pi / 3.0, 5.0 * math.pi / 3.0])
            )
        )
        < 1e-12
    )
    width = intervals[0, 1] - intervals[0, 0]
    gradient = torch.autograd.grad(width, positions)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[1].abs().max() > 0.0


def test_atomic_contact_azimuth_full_and_two_cap_cases():
    single = contact_exposed_azimuth(
        torch.zeros((1, 3), dtype=torch.float64),
        torch.tensor([2.12], dtype=torch.float64),
        owner=0,
        polar_cosine=torch.tensor(0.0, dtype=torch.float64),
    )
    assert single.shape == (1, 2)
    assert torch.allclose(
        single[0], torch.tensor([0.0, 2.0 * math.pi], dtype=torch.float64)
    )
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    intervals = contact_exposed_azimuth(
        positions,
        torch.full((3,), 2.12, dtype=torch.float64),
        owner=0,
        polar_cosine=torch.tensor(0.0, dtype=torch.float64),
    )
    assert intervals.shape == (2, 2)
    assert (
        abs((intervals[:, 1] - intervals[:, 0]).sum().item() - 2.0 * math.pi / 3.0)
        < 1e-12
    )


def test_vertical_contact_cap_endpoint_is_a_slice_event():
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=torch.float64)
    intrinsic = torch.tensor([2.12, 2.12], dtype=torch.float64)
    intervals = contact_exposed_azimuth(
        positions,
        intrinsic,
        owner=0,
        polar_cosine=torch.tensor(0.5, dtype=torch.float64),
    )
    assert intervals.shape == (1, 2)
