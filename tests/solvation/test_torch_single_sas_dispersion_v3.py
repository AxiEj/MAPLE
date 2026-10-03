"""Exact single-cover SAS dispersion v3 scalar and derivative tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.torch_continuum_dispersion import (
    WATER_DENSITY_PER_ANGSTROM3,
)
from maple.function.calculator.extra_correction.implicit.torch_single_sas_dispersion_v3 import (
    KERNEL_IDENTITY,
    _single_receiver_surface_integral,
    single_sas_dispersion_from_rmin_epsilon,
)


def _angular_reference(displacement, radius, sigma, mixed_epsilon):
    distance = float(np.linalg.norm(displacement))
    raw_b = 4.0 * mixed_epsilon * sigma**6
    raw_a = raw_b * sigma**6
    inside = -raw_b / (3.0 * sigma**3) + raw_a / (9.0 * sigma**9)

    def integrand(mu):
        q2 = radius**2 + distance**2 + 2.0 * radius * distance * mu
        q = math.sqrt(q2)
        radial = (
            -raw_b / (3.0 * q**6) + raw_a / (9.0 * q**12)
            if q >= sigma
            else inside / q**3
        )
        normal_dot = radius + distance * mu
        return 2.0 * math.pi * radius**2 * radial * normal_dot

    return quad(integrand, -1.0, 1.0, epsabs=1e-13, epsrel=1e-13, points=[0.0])[0]


@pytest.mark.parametrize(
    ("displacement", "radius", "sigma", "label"),
    [
        ([0.0, 0.0, 0.0], 1.0, 2.0, "all_inner"),
        ([0.0, 0.0, 0.0], 2.0, 1.0, "all_outer"),
        ([0.2, -0.1, 0.3], 1.0, 2.0, "all_inner"),
        ([0.2, -0.1, 0.3], 2.0, 1.0, "all_outer"),
        ([0.8, 0.1, -0.2], 1.3, 1.0, "crossing"),
    ],
)
def test_receiver_branches_match_independent_angular_quadrature(
    displacement, radius, sigma, label
):
    vector = torch.tensor(displacement, dtype=torch.float64, requires_grad=True)
    radius_tensor = torch.tensor(radius, dtype=torch.float64)
    sigma_tensor = torch.tensor(sigma, dtype=torch.float64)
    epsilon_tensor = torch.tensor(0.12, dtype=torch.float64)
    actual, branch = _single_receiver_surface_integral(
        vector, radius_tensor, sigma_tensor, epsilon_tensor
    )
    expected = _angular_reference(np.array(displacement), radius, sigma, 0.12)
    assert branch == label
    assert actual.item() == pytest.approx(expected, rel=2e-12, abs=2e-12)
    gradient = torch.autograd.grad(actual, vector, create_graph=True)[0]
    hessian = torch.stack(
        [
            torch.autograd.grad(gradient[index], vector, retain_graph=True)[0]
            for index in range(3)
        ]
    )
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(hessian).all()


def test_d_zero_inner_is_graph_connected_with_exact_zero_derivatives():
    displacement = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    value, branch = _single_receiver_surface_integral(
        displacement,
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(2.0, dtype=torch.float64),
        torch.tensor(0.12, dtype=torch.float64),
    )
    gradient = torch.autograd.grad(value, displacement, create_graph=True)[0]
    hvp = torch.autograd.grad(gradient.sum(), displacement)[0]
    assert branch == "all_inner"
    assert torch.equal(gradient, torch.zeros_like(gradient))
    assert torch.equal(hvp, torch.zeros_like(hvp))


def test_d_zero_outer_has_proved_isotropic_hessian():
    displacement = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    radius = torch.tensor(2.0, dtype=torch.float64)
    sigma = torch.tensor(1.0, dtype=torch.float64)
    mixed = torch.tensor(0.12, dtype=torch.float64)
    value, branch = _single_receiver_surface_integral(
        displacement, radius, sigma, mixed
    )
    gradient = torch.autograd.grad(value, displacement, create_graph=True)[0]
    hessian = torch.stack(
        [
            torch.autograd.grad(gradient[index], displacement, retain_graph=True)[0]
            for index in range(3)
        ]
    )
    raw_b = 4.0 * mixed * sigma**6
    raw_a = raw_b * sigma**6
    coefficient = (
        -8.0 * math.pi * raw_b / radius**5 + 16.0 * math.pi * raw_a / radius**11
    )
    assert branch == "all_outer"
    torch.testing.assert_close(
        hessian, coefficient * torch.eye(3, dtype=torch.float64), rtol=1e-13, atol=1e-13
    )


def test_exceptional_d_zero_sigma_point_rejects_before_norm_derivatives():
    with pytest.raises(ValueError, match="R=sigma"):
        _single_receiver_surface_integral(
            torch.zeros(3, dtype=torch.float64, requires_grad=True),
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(0.12, dtype=torch.float64),
        )


def test_wide_crossing_uses_stable_inner_base_and_matches_independent_derivatives(
    monkeypatch,
):
    from maple.function.calculator.extra_correction.implicit import (
        torch_single_sas_dispersion_v3 as module,
    )

    def forbidden_outer(*_args, **_kwargs):
        raise AssertionError("crossing branch constructed the unstable all-outer base")

    monkeypatch.setattr(module, "_all_outer_surface_integral", forbidden_outer)
    displacement = torch.tensor(
        [2.352173939598834, 0.0, 0.0], dtype=torch.float64, requires_grad=True
    )
    raw, branch = module._single_receiver_surface_integral(
        displacement,
        torch.tensor(2.987151768716868, dtype=torch.float64),
        torch.tensor(2.9454050475891993, dtype=torch.float64),
        torch.tensor(0.15023883121426318, dtype=torch.float64),
    )
    energy = WATER_DENSITY_PER_ANGSTROM3 * raw
    gradient = torch.autograd.grad(energy, displacement, create_graph=True)[0]
    hessian = torch.stack(
        [
            torch.autograd.grad(gradient[index], displacement, retain_graph=True)[0]
            for index in range(3)
        ]
    )
    assert branch == "crossing"
    assert energy.item() == pytest.approx(-1.4041496881836126, abs=5e-12)
    torch.testing.assert_close(
        gradient,
        torch.tensor([0.011707376997860985, 0.0, 0.0], dtype=torch.float64),
        rtol=0.0,
        atol=1e-8,
    )
    torch.testing.assert_close(
        hessian,
        torch.diag(
            torch.tensor(
                [-0.0501808965397357, 0.004977258186891439, 0.004977258186891439],
                dtype=torch.float64,
            )
        ),
        rtol=0.0,
        atol=1e-8,
    )


def test_crossing_is_continuous_on_both_sides_of_each_sigma_tangency():
    radius = torch.tensor(2.0, dtype=torch.float64)
    mixed = torch.tensor(0.12, dtype=torch.float64)
    displacement = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
    lower = 1.5
    upper = 2.5

    def evaluate(sigma):
        return _single_receiver_surface_integral(
            displacement,
            radius,
            torch.tensor(sigma, dtype=torch.float64),
            mixed,
        )

    below_lower = evaluate(math.nextafter(lower, -math.inf))
    above_lower = evaluate(math.nextafter(lower, math.inf))
    below_upper = evaluate(math.nextafter(upper, -math.inf))
    above_upper = evaluate(math.nextafter(upper, math.inf))
    assert below_lower[1] == "all_outer"
    assert above_lower[1] == "crossing"
    assert below_upper[1] == "crossing"
    assert above_upper[1] == "all_inner"
    torch.testing.assert_close(below_lower[0], above_lower[0], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(below_upper[0], above_upper[0], rtol=0.0, atol=1e-12)


def _water_inputs(requires_grad=True):
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.926627, 0.0]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    rmin = torch.tensor([1.82, 0.3019, 0.3019], dtype=torch.float64)
    epsilon = torch.tensor([0.1520, 0.0157, 0.0157], dtype=torch.float64)
    return positions, rmin, epsilon


def test_public_scalar_reports_narrow_identity_and_live_receiver_sum():
    positions, rmin, epsilon = _water_inputs()
    result = single_sas_dispersion_from_rmin_epsilon(positions, rmin, epsilon)
    assert result.kernel_identity == KERNEL_IDENTITY
    assert result.certificate.covering_index == 0
    assert result.branch_labels == ("all_inner", "crossing", "crossing")
    assert result.runtime_finite_difference is False
    assert result.legacy_quadrature_used is False
    torch.testing.assert_close(
        result.energy_kcal_mol,
        result.receiver_energies_kcal_mol.sum(),
        rtol=0.0,
        atol=0.0,
    )
    assert result.receiver_energies_kcal_mol.requires_grad
    gradient = torch.autograd.grad(
        result.energy_kcal_mol, positions, create_graph=True
    )[0]
    hvp = torch.autograd.grad((gradient * torch.ones_like(gradient)).sum(), positions)[
        0
    ]
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(hvp).all()
    assert gradient.sum(dim=0).abs().max().item() < 1e-12


def test_public_scalar_has_one_certificate_one_selection_and_n_receivers(monkeypatch):
    from maple.function.calculator.extra_correction.implicit import (
        torch_single_sas_dispersion_v3 as module,
    )

    positions, rmin, epsilon = _water_inputs(requires_grad=False)
    calls = {"certificate": 0, "selection": 0, "receiver": 0}
    original_certificate = module.certify_single_covering_dispersion_sas
    original_selection = module._select_covering_sas_sphere
    original_receiver = module._single_receiver_surface_integral

    def certificate(*args):
        calls["certificate"] += 1
        return original_certificate(*args)

    def selection(*args):
        calls["selection"] += 1
        return original_selection(*args)

    def receiver(*args):
        calls["receiver"] += 1
        return original_receiver(*args)

    monkeypatch.setattr(module, "certify_single_covering_dispersion_sas", certificate)
    monkeypatch.setattr(module, "_select_covering_sas_sphere", selection)
    monkeypatch.setattr(module, "_single_receiver_surface_integral", receiver)
    module.single_sas_dispersion_from_rmin_epsilon(positions, rmin, epsilon)
    assert calls == {"certificate": 1, "selection": 1, "receiver": 3}


def test_public_scalar_is_rigid_and_permutation_invariant():
    positions, rmin, epsilon = _water_inputs(requires_grad=False)
    rotation = torch.tensor(
        [[0.36, -0.8, 0.48], [0.8, 0.0, -0.6], [0.48, 0.6, 0.64]],
        dtype=torch.float64,
    )
    permutation = torch.tensor([2, 0, 1])
    reference = single_sas_dispersion_from_rmin_epsilon(positions, rmin, epsilon)
    transformed = single_sas_dispersion_from_rmin_epsilon(
        (positions @ rotation.T + 4.0)[permutation],
        rmin[permutation],
        epsilon[permutation],
    )
    torch.testing.assert_close(
        reference.energy_kcal_mol, transformed.energy_kcal_mol, rtol=1e-13, atol=1e-13
    )
    torch.testing.assert_close(
        reference.receiver_energies_kcal_mol[permutation],
        transformed.receiver_energies_kcal_mol,
        rtol=1e-13,
        atol=1e-13,
    )


def test_public_scalar_fails_closed_without_cover_and_never_calls_legacy(monkeypatch):
    from maple.function.calculator.extra_correction.implicit import (
        torch_continuum_dispersion,
    )

    monkeypatch.setattr(
        torch_continuum_dispersion,
        "dispersion_from_rmin_epsilon",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy called")
        ),
    )
    with pytest.raises(ValueError, match="single-cover"):
        single_sas_dispersion_from_rmin_epsilon(
            torch.tensor([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=torch.float64),
            torch.tensor([1.0, 1.0], dtype=torch.float64),
            torch.tensor([0.1, 0.1], dtype=torch.float64),
        )


def test_zero_epsilon_receiver_is_preserved_without_nonfinite_derivatives():
    positions, rmin, epsilon = _water_inputs()
    epsilon = epsilon.clone()
    epsilon[2] = 0.0
    result = single_sas_dispersion_from_rmin_epsilon(positions, rmin, epsilon)
    assert result.receiver_energies_kcal_mol[2].item() == 0.0
    gradient = torch.autograd.grad(result.energy_kcal_mol, positions)[0]
    assert torch.isfinite(gradient).all()
    assert math.isfinite(result.energy_kcal_mol.item())
    assert WATER_DENSITY_PER_ANGSTROM3 > 0.0
