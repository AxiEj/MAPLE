"""Ownership contract for shared Torch quadrature rules."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.torch_continuum_dispersion import (
    dispersion_from_rmin_epsilon,
)
from maple.function.calculator.extra_correction.implicit.torch_continuum_sav import (
    sphere_union_volume,
)
from maple.function.calculator.extra_correction.implicit.torch_sphere_union_geometry import (
    legendre_rule,
)


def test_callers_cannot_poison_cached_quadrature_or_later_consumers():
    device = "cpu"
    positions = torch.zeros((1, 3), dtype=torch.float64)
    radius = torch.tensor([1.7], dtype=torch.float64)
    rmin = torch.tensor([1.9069], dtype=torch.float64)
    epsilon = torch.tensor([0.1078], dtype=torch.float64)
    baseline_sav = sphere_union_volume(positions, radius).volume_angstrom3.item()
    baseline_dispersion = dispersion_from_rmin_epsilon(
        positions, rmin, epsilon
    ).energy_kcal_mol.item()

    for order in (16, 24, 32, 48):
        nodes, weights = legendre_rule(order, device)
        nodes.fill_(123.0)
        weights.zero_()

        fresh_nodes, fresh_weights = legendre_rule(order, device)
        assert fresh_nodes.data_ptr() != nodes.data_ptr()
        assert fresh_weights.data_ptr() != weights.data_ptr()
        assert torch.all(fresh_nodes.abs() < 1.0)
        assert abs(fresh_weights.sum().item() - 2.0) < 1e-14

    sav = sphere_union_volume(positions, radius)
    assert abs(sav.volume_angstrom3.item() - 4.0 * math.pi * 1.7**3 / 3.0) < 1e-9
    assert sav.volume_angstrom3.item() == baseline_sav

    dispersion = dispersion_from_rmin_epsilon(positions, rmin, epsilon)
    assert math.isfinite(dispersion.energy_kcal_mol.item())
    assert dispersion.energy_kcal_mol.item() == baseline_dispersion
