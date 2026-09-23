"""Frozen sigma-split PBSA dispersion on its own SAS boundary."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.sphere_union_dispersion import (
    dispersion_energy_and_gradient,
)
from maple.function.calculator.extra_correction.implicit.torch_continuum_dispersion import (
    dispersion_from_rmin_epsilon,
)


def _numpy_reference(xyz, rmin, epsilon):
    sigma = (rmin + 1.7683) * 2.0 ** (-1.0 / 6.0)
    mixed_epsilon = np.sqrt(epsilon * 0.1520)
    return dispersion_energy_and_gradient(
        xyz,
        rmin + 0.557,
        sigma,
        mixed_epsilon,
        0.03333 * 1.129,
        phi_order=64,
        atol=1e-10,
        rtol=1e-9,
    )


def test_one_site_exact_sigma_volume_limit_and_zero_force():
    x = torch.tensor([[0.4, -0.2, 0.7]], dtype=torch.float64, requires_grad=True)
    rmin = torch.tensor([1.9069], dtype=torch.float64)
    epsilon = torch.tensor([0.1078], dtype=torch.float64)
    result = dispersion_from_rmin_epsilon(x, rmin, epsilon, atol=1e-10, rtol=1e-10)
    sigma = (rmin.item() + 1.7683) * 2.0 ** (-1.0 / 6.0)
    mixed = math.sqrt(epsilon.item() * 0.1520)
    assert rmin.item() + 0.557 < sigma
    expected = -32.0 * math.pi * (0.03333 * 1.129) * mixed * sigma**3 / 9.0
    assert abs(result.energy_kcal_mol.item() - expected) < 1e-8
    gradient = torch.autograd.grad(result.energy_kcal_mol, x)[0]
    assert gradient.abs().max().item() < 1e-8


def test_two_sites_agree_with_independent_surface_reference_and_gradient():
    xyz = np.array([[0.0, 0.0, 0.0], [1.5, 0.25, -0.1]])
    rmin = np.array([1.9069, 1.7683])
    epsilon = np.array([0.1078, 0.1520])
    reference = _numpy_reference(xyz, rmin, epsilon)
    positions = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
    result = dispersion_from_rmin_epsilon(
        positions,
        torch.tensor(rmin, dtype=torch.float64),
        torch.tensor(epsilon, dtype=torch.float64),
        atol=1e-8,
        rtol=1e-8,
    )
    gradient = torch.autograd.grad(result.energy_kcal_mol, positions)[0]
    assert abs(result.energy_kcal_mol.item() - reference.energy) < 2e-5
    assert np.max(np.abs(gradient.detach().numpy() - reference.gradient)) < 2e-4
    assert np.linalg.norm(gradient.detach().numpy().sum(axis=0)) < 1e-5


def test_dispersion_kernel_never_invokes_amber_process(monkeypatch):
    import subprocess

    def forbidden(*_args, **_kwargs):
        raise AssertionError("external process forbidden")

    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(subprocess, "run", forbidden)
    result = dispersion_from_rmin_epsilon(
        torch.zeros((1, 3), dtype=torch.float64),
        torch.tensor([1.9069], dtype=torch.float64),
        torch.tensor([0.1078], dtype=torch.float64),
    )
    assert math.isfinite(result.energy_kcal_mol.item())


def test_dispersion_rejects_duplicate_sas_spheres():
    with pytest.raises(ValueError, match="duplicate|coincident|tangen"):
        dispersion_from_rmin_epsilon(
            torch.zeros((2, 3), dtype=torch.float64),
            torch.tensor([1.7, 1.7], dtype=torch.float64),
            torch.tensor([0.1, 0.1], dtype=torch.float64),
        )


@pytest.mark.parametrize("count", [92, 400])
def test_dispersion_rejects_oversize_chunk_before_allocation(count):
    positions = torch.zeros((count, 3), dtype=torch.float64)
    positions[:, 0] = torch.arange(count, dtype=torch.float64) * 10.0
    with pytest.raises(RuntimeError, match="geometry-and-chunk memory budget"):
        dispersion_from_rmin_epsilon(
            positions,
            torch.full((count,), 1.7, dtype=torch.float64),
            torch.full((count,), 0.1, dtype=torch.float64),
        )


def test_dispersion_rejects_unqualified_large_ad_graph():
    positions = torch.zeros((7, 3), dtype=torch.float64, requires_grad=True)
    with pytest.raises(RuntimeError, match="AD is limited to six sites"):
        dispersion_from_rmin_epsilon(
            positions,
            torch.full((7,), 1.7, dtype=torch.float64),
            torch.full((7,), 0.1, dtype=torch.float64),
        )


def test_dispersion_gradient_has_no_undefined_masked_ops():
    for coords, radii, epsilons in (
        ([[0.0, 0.0, 0.0]], [1.9069], [0.1078]),
        ([[0.0, 0.0, 0.0], [1.5, 0.25, -0.1]], [1.9069, 1.7683], [0.1078, 0.152]),
    ):
        positions = torch.tensor(coords, dtype=torch.float64, requires_grad=True)
        with torch.autograd.detect_anomaly():
            energy = dispersion_from_rmin_epsilon(
                positions,
                torch.tensor(radii, dtype=torch.float64),
                torch.tensor(epsilons, dtype=torch.float64),
            ).energy_kcal_mol
            gradient = torch.autograd.grad(energy, positions)[0]
        assert torch.isfinite(gradient).all()


def test_dispersion_full_rotation_and_own_scalar_difference():
    coordinates = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.25, -0.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    radii = torch.tensor([1.9069, 1.7683], dtype=torch.float64)
    epsilons = torch.tensor([0.1078, 0.1520], dtype=torch.float64)
    rotation = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )

    def energy(positions):
        return dispersion_from_rmin_epsilon(positions, radii, epsilons).energy_kcal_mol

    turned = (coordinates.detach() @ rotation.T + 3.0).requires_grad_()
    base_energy = energy(coordinates)
    turned_energy = energy(turned)
    base_gradient = torch.autograd.grad(base_energy, coordinates)[0]
    turned_gradient = torch.autograd.grad(turned_energy, turned)[0]
    assert abs(base_energy.item() - turned_energy.item()) < 1e-7
    assert (
        torch.max(torch.abs(turned_gradient - base_gradient @ rotation.T)).item() < 1e-5
    )

    step = 1e-4
    plus = coordinates.detach().clone()
    minus = coordinates.detach().clone()
    plus[1, 1] += step
    minus[1, 1] -= step
    numerical = (energy(plus).item() - energy(minus).item()) / (2 * step)
    assert abs(base_gradient[1, 1].item() - numerical) < 2e-5


def test_dispersion_two_site_hvp_graph_is_finite_without_capability_claim():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.25, -0.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    radii = torch.tensor([1.9069, 1.7683], dtype=torch.float64)
    epsilons = torch.tensor([0.1078, 0.1520], dtype=torch.float64)
    direction = torch.tensor([[0.2, -0.1, 0.3], [-0.3, 0.4, -0.2]], dtype=torch.float64)
    energy = dispersion_from_rmin_epsilon(positions, radii, epsilons).energy_kcal_mol
    gradient = torch.autograd.grad(energy, positions, create_graph=True)[0]
    hvp = torch.autograd.grad((gradient * direction).sum(), positions)[0]
    assert torch.isfinite(hvp).all()
