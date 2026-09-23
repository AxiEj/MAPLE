"""Sharp PBSA SAV volume and derivative canaries for the unregistered Torch slice."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.sphere_union_volume import (
    volume_and_gradient,
)
from maple.function.calculator.extra_correction.implicit.torch_continuum_sav import (
    cavity_from_rmin,
    sphere_union_volume,
)


def test_isolated_sphere_volume_and_zero_force():
    positions = torch.tensor(
        [[0.4, -0.2, 0.7]], dtype=torch.float64, requires_grad=True
    )
    radius = torch.tensor([2.3], dtype=torch.float64)
    result = sphere_union_volume(positions, radius, atol=1e-11, rtol=1e-11)
    expected = 4.0 * math.pi * radius[0].item() ** 3 / 3.0
    assert abs(result.volume_angstrom3.item() - expected) < 1e-9
    gradient = torch.autograd.grad(result.volume_angstrom3, positions)[0]
    assert torch.max(torch.abs(gradient)).item() < 1e-8


def test_overlapping_pair_exact_formula():
    first, second, distance = 1.4, 1.1, 1.5
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=torch.float64
    )
    radii = torch.tensor([first, second], dtype=torch.float64)
    intersection = (
        math.pi
        * (first + second - distance) ** 2
        * (distance**2 + 2 * distance * (first + second) - 3 * (first - second) ** 2)
        / (12 * distance)
    )
    expected = 4 * math.pi * (first**3 + second**3) / 3 - intersection
    result = sphere_union_volume(positions, radii, atol=1e-10, rtol=1e-10)
    assert abs(result.volume_angstrom3.item() - expected) < 1e-8


@pytest.mark.parametrize("distance,expected", [(4.0, "sum"), (0.2, "outer")])
def test_disjoint_and_contained_sphere_limits(distance, expected):
    radii = torch.tensor([1.6, 1.1], dtype=torch.float64)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=torch.float64
    )
    volume = sphere_union_volume(positions, radii).volume_angstrom3.item()
    target = (
        4.0
        * math.pi
        / 3.0
        * (radii[0].item() ** 3 + (radii[1].item() ** 3 if expected == "sum" else 0.0))
    )
    assert abs(volume - target) < 1e-8


def test_three_spheres_agree_with_independent_slicing_and_shape_derivative():
    xyz = np.array([[0.0, 0.0, 0.0], [1.5, 0.1, 0.2], [0.2, 1.7, -0.1]])
    radii = np.array([1.5, 1.4, 1.25])
    reference = volume_and_gradient(xyz, radii, atol=1e-11, rtol=1e-10)
    positions = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
    result = sphere_union_volume(
        positions, torch.tensor(radii, dtype=torch.float64), atol=1e-9, rtol=1e-9
    )
    gradient = torch.autograd.grad(result.volume_angstrom3, positions)[0]
    assert abs(result.volume_angstrom3.item() - reference.volume) < 2e-7
    assert np.max(np.abs(gradient.detach().numpy() - reference.gradient)) < 2e-6
    assert np.linalg.norm(gradient.detach().numpy().sum(axis=0)) < 1e-7


def test_rigid_transform_preserves_volume_and_rotates_gradient():
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.1, 0.2], [0.2, 1.7, -0.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    radii = torch.tensor([1.5, 1.4, 1.25], dtype=torch.float64)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    baseline = sphere_union_volume(xyz, radii)
    rotated = sphere_union_volume(xyz @ rotation.T + 4.0, radii)
    base_gradient = torch.autograd.grad(
        baseline.volume_angstrom3, xyz, retain_graph=True
    )[0]
    rotated_gradient = torch.autograd.grad(rotated.volume_angstrom3, xyz)[0]
    assert (
        abs(baseline.volume_angstrom3.item() - rotated.volume_angstrom3.item()) < 2e-8
    )
    assert torch.max(torch.abs(base_gradient - rotated_gradient)).item() < 2e-6


def test_rotation_mixing_slice_axis_preserves_volume_and_vector_force():
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.1, 0.2], [0.2, 1.7, -0.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    radii = torch.tensor([1.5, 1.4, 1.25], dtype=torch.float64)
    rotation = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    turned = (xyz.detach() @ rotation.T + 3.0).requires_grad_()
    base_energy = sphere_union_volume(xyz, radii).volume_angstrom3
    turned_energy = sphere_union_volume(turned, radii).volume_angstrom3
    base_gradient = torch.autograd.grad(base_energy, xyz)[0]
    turned_gradient = torch.autograd.grad(turned_energy, turned)[0]
    assert abs(base_energy.item() - turned_energy.item()) < 1e-8
    assert (
        torch.max(torch.abs(turned_gradient - base_gradient @ rotation.T)).item() < 1e-5
    )
    step = 1e-4
    plus = xyz.detach().clone()
    minus = xyz.detach().clone()
    plus[1, 1] += step
    minus[1, 1] -= step
    numerical = (
        sphere_union_volume(plus, radii).volume_angstrom3.item()
        - sphere_union_volume(minus, radii).volume_angstrom3.item()
    ) / (2 * step)
    assert abs(base_gradient[1, 1].item() - numerical) < 2e-5


def test_cavity_uses_volume_and_frozen_coefficient():
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    rmin = torch.tensor([1.9069], dtype=torch.float64)
    result = cavity_from_rmin(positions, rmin, atol=1e-11, rtol=1e-11)
    radius = rmin.item() + 1.3
    expected = 0.0378 * (4.0 * math.pi * radius**3 / 3.0) - 0.5692
    assert abs(result.energy_kcal_mol.item() - expected) < 1e-9
    assert abs(result.volume_angstrom3.item() - 4.0 * math.pi * radius**3 / 3.0) < 1e-9


def test_tangency_is_not_silently_smoothed():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=torch.float64)
    radii = torch.tensor([1.4, 1.1], dtype=torch.float64)
    with pytest.raises(ValueError, match="tangen"):
        sphere_union_volume(xyz, radii)


def test_cavity_kernel_never_invokes_amber_process(monkeypatch):
    import subprocess

    def forbidden(*_args, **_kwargs):
        raise AssertionError("external process forbidden")

    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(subprocess, "run", forbidden)
    result = cavity_from_rmin(
        torch.zeros((1, 3), dtype=torch.float64),
        torch.tensor([1.9069], dtype=torch.float64),
    )
    assert math.isfinite(result.energy_kcal_mol.item())


def test_cavity_rejects_negative_unshifted_lj_radius():
    with pytest.raises(ValueError, match="rmin"):
        cavity_from_rmin(
            torch.zeros((1, 3), dtype=torch.float64),
            torch.tensor([-1.0], dtype=torch.float64),
        )


def test_sav_gradient_has_no_undefined_masked_ops():
    for coords, radii in (
        ([[0.0, 0.0, 0.0]], [1.5]),
        ([[0.0, 0.0, 0.0], [1.5, 0.25, -0.1]], [1.5, 1.4]),
    ):
        positions = torch.tensor(coords, dtype=torch.float64, requires_grad=True)
        with torch.autograd.detect_anomaly():
            energy = sphere_union_volume(
                positions, torch.tensor(radii, dtype=torch.float64)
            ).volume_angstrom3
            gradient = torch.autograd.grad(energy, positions)[0]
        assert torch.isfinite(gradient).all()


def test_sav_two_site_hvp_graph_is_finite_without_capability_claim():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.25, -0.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    radii = torch.tensor([1.5, 1.4], dtype=torch.float64)
    direction = torch.tensor([[0.2, -0.1, 0.3], [-0.3, 0.4, -0.2]], dtype=torch.float64)
    energy = sphere_union_volume(positions, radii).volume_angstrom3
    gradient = torch.autograd.grad(energy, positions, create_graph=True)[0]
    hvp = torch.autograd.grad((gradient * direction).sum(), positions)[0]
    assert torch.isfinite(hvp).all()
