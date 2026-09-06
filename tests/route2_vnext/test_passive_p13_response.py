from __future__ import annotations

import pytest

from maple.solvation.models.passive_p13_response import (
    p13_induced_source_analytic_torch,
    p13_induced_source_from_scalar_torch,
    p13_response_mep_torch,
    radial_source_molecular_dipole_torch,
    radial_source_total_charge_torch,
)


def _factor_inputs(torch):
    generator = torch.Generator().manual_seed(20260827)
    dtype = torch.float64
    atom_count = 3
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, -0.3, 0.1], [-0.4, 0.9, 0.2]],
        dtype=dtype,
    )
    raw_alpha = torch.randn((atom_count, 3, 3), dtype=dtype, generator=generator)
    return {
        "positions_angstrom": positions,
        "edge_index": torch.tensor([[0, 0], [1, 2]], dtype=torch.int64),
        "atomic_polarizability": raw_alpha @ raw_alpha.transpose(-1, -2),
        "radial_gain": torch.randn((atom_count,), dtype=dtype, generator=generator),
        "vector_coefficients": torch.randn(
            (atom_count, 2, 2, 2), dtype=dtype, generator=generator
        ),
        "edge_coefficients": torch.randn((2, 2), dtype=dtype, generator=generator),
        "edge_cutoff": torch.tensor([0.8, 0.6], dtype=dtype),
        "scalar_radial_reshape": torch.tensor(0.1, dtype=dtype),
        "vector_radial_reshape": torch.tensor(-0.2, dtype=dtype),
        "l2_gain": torch.randn((atom_count,), dtype=dtype, generator=generator),
    }


def test_p13_induced_source_is_same_scalar_passive_and_charge_neutral() -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(91)
    inputs = _factor_inputs(torch)
    radial = torch.randn((4, 3, 8), dtype=torch.float64, generator=generator)
    l2 = torch.randn((4, 3, 5), dtype=torch.float64, generator=generator)
    energies, radial_source, l2_source = p13_induced_source_from_scalar_torch(
        radial_fields=radial,
        l2_fields=l2,
        l2_gain=inputs.pop("l2_gain"),
        create_graph=True,
        **inputs,
    )

    torch.testing.assert_close(
        radial_source_total_charge_torch(radial_source),
        torch.zeros(4, dtype=torch.float64),
        rtol=0.0,
        atol=2.0e-13,
    )
    direction_radial = torch.randn(radial.shape, dtype=radial.dtype, generator=generator)
    direction_l2 = torch.randn(l2.shape, dtype=l2.dtype, generator=generator)
    directional = torch.sum(radial_source * direction_radial) + torch.sum(
        l2_source * direction_l2
    )
    gradient = torch.autograd.grad(
        torch.sum(energies),
        (radial, l2),
        retain_graph=True,
    )
    torch.testing.assert_close(gradient[0], radial_source, rtol=0.0, atol=0.0)
    torch.testing.assert_close(gradient[1], l2_source, rtol=0.0, atol=0.0)
    assert bool(torch.isfinite(directional))


def test_p13_analytic_backend_matches_scalar_autograd_and_parameter_vjp() -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(192)
    base = _factor_inputs(torch)
    differentiable = {}
    for name, value in base.items():
        if name == "edge_index":
            differentiable[name] = value
        else:
            differentiable[name] = value.detach().clone().requires_grad_(True)
    radial = torch.randn(
        (3, 3, 8), dtype=torch.float64, generator=generator, requires_grad=True
    )
    l2 = torch.randn(
        (3, 3, 5), dtype=torch.float64, generator=generator, requires_grad=True
    )
    autograd_energy, autograd_radial, autograd_l2 = (
        p13_induced_source_from_scalar_torch(
            radial_fields=radial,
            l2_fields=l2,
            l2_gain=differentiable["l2_gain"],
            create_graph=True,
            **{key: value for key, value in differentiable.items() if key != "l2_gain"},
        )
    )
    analytic_energy, analytic_radial, analytic_l2 = (
        p13_induced_source_analytic_torch(
            radial_fields=radial,
            l2_fields=l2,
            **differentiable,
        )
    )
    torch.testing.assert_close(
        analytic_energy, autograd_energy, rtol=1.0e-13, atol=2.0e-14
    )
    torch.testing.assert_close(
        analytic_radial, autograd_radial, rtol=2.0e-13, atol=2.0e-13
    )
    torch.testing.assert_close(
        analytic_l2, autograd_l2, rtol=2.0e-13, atol=2.0e-13
    )
    torch.testing.assert_close(
        2.0 * analytic_energy,
        torch.sum(radial * analytic_radial, dim=(1, 2))
        + torch.sum(l2 * analytic_l2, dim=(1, 2)),
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    radial_weight = torch.randn(
        analytic_radial.shape, dtype=torch.float64, generator=generator
    )
    l2_weight = torch.randn(
        analytic_l2.shape, dtype=torch.float64, generator=generator
    )
    scalar_autograd = torch.sum(autograd_radial * radial_weight) + torch.sum(
        autograd_l2 * l2_weight
    )
    scalar_analytic = torch.sum(analytic_radial * radial_weight) + torch.sum(
        analytic_l2 * l2_weight
    )
    gradient_inputs = tuple(
        differentiable[name]
        for name in (
            "positions_angstrom",
            "atomic_polarizability",
            "radial_gain",
            "vector_coefficients",
            "edge_coefficients",
            "edge_cutoff",
            "scalar_radial_reshape",
            "vector_radial_reshape",
            "l2_gain",
        )
    )
    expected = torch.autograd.grad(
        scalar_autograd, gradient_inputs, retain_graph=True, create_graph=True
    )
    actual = torch.autograd.grad(
        scalar_analytic, gradient_inputs, retain_graph=True, create_graph=True
    )
    for analytic_value, autograd_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            analytic_value,
            autograd_value,
            rtol=1.0e-10,
            atol=1.0e-10,
        )


def test_p13_analytic_backend_passes_reduced_gradcheck_and_gradgradcheck() -> None:
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    generator = torch.Generator().manual_seed(36)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.3, -0.1]],
        dtype=dtype,
        requires_grad=True,
    )
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    raw_alpha = torch.randn((2, 3, 3), dtype=dtype, generator=generator)
    alpha = (raw_alpha @ raw_alpha.transpose(-1, -2)).requires_grad_(True)
    radial_gain = torch.randn(2, dtype=dtype, generator=generator, requires_grad=True)
    vector = torch.randn(
        (2, 2, 2, 2), dtype=dtype, generator=generator, requires_grad=True
    )
    edge = torch.randn((1, 2), dtype=dtype, generator=generator, requires_grad=True)
    cutoff = torch.tensor([0.7], dtype=dtype, requires_grad=True)
    scalar_mix = torch.tensor(0.08, dtype=dtype, requires_grad=True)
    vector_mix = torch.tensor(-0.04, dtype=dtype, requires_grad=True)
    l2_gain = torch.randn(2, dtype=dtype, generator=generator, requires_grad=True)
    radial = torch.randn(
        (1, 2, 8), dtype=dtype, generator=generator, requires_grad=True
    )
    l2 = torch.randn(
        (1, 2, 5), dtype=dtype, generator=generator, requires_grad=True
    )

    def function(*values):
        (
            radial_values,
            l2_values,
            position_values,
            alpha_values,
            radial_gain_values,
            vector_values,
            edge_values,
            cutoff_values,
            scalar_values,
            vector_mix_values,
            l2_gain_values,
        ) = values
        energy, source_radial, source_l2 = p13_induced_source_analytic_torch(
            radial_fields=radial_values,
            l2_fields=l2_values,
            positions_angstrom=position_values,
            edge_index=edge_index,
            atomic_polarizability=alpha_values,
            radial_gain=radial_gain_values,
            vector_coefficients=vector_values,
            edge_coefficients=edge_values,
            edge_cutoff=cutoff_values,
            scalar_radial_reshape=scalar_values,
            vector_radial_reshape=vector_mix_values,
            l2_gain=l2_gain_values,
        )
        return torch.cat((energy, source_radial.reshape(-1), source_l2.reshape(-1)))

    inputs = (
        radial,
        l2,
        positions,
        alpha,
        radial_gain,
        vector,
        edge,
        cutoff,
        scalar_mix,
        vector_mix,
        l2_gain,
    )
    assert torch.autograd.gradcheck(
        function,
        inputs,
        eps=1.0e-6,
        atol=2.0e-6,
        rtol=2.0e-5,
        fast_mode=True,
    )
    assert torch.autograd.gradgradcheck(
        function,
        inputs,
        eps=1.0e-6,
        atol=3.0e-6,
        rtol=3.0e-5,
        fast_mode=True,
    )


def test_p13_observable_maps_use_public_radial_moment_order() -> None:
    torch = pytest.importorskip("torch")
    source = torch.zeros((2, 2, 8), dtype=torch.float64)
    source[0, 0, 0] = 0.3
    source[0, 0, 1] = -0.1
    source[0, 0, 4] = 0.7
    source[0, 0, 2] = -0.2
    source[0, 0, 3] = 0.4
    source[0, 1, 0] = -0.2
    source[1] = 2.0 * source[0]
    positions = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64
    )
    dipole = radial_source_molecular_dipole_torch(
        radial_source=source,
        positions_angstrom=positions,
    )
    expected = torch.tensor([[0.9, -0.6, 0.4], [1.8, -1.2, 0.8]], dtype=torch.float64)
    torch.testing.assert_close(dipole, expected, rtol=0.0, atol=1.0e-15)
    torch.testing.assert_close(
        radial_source_total_charge_torch(source),
        torch.zeros(2, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-15,
    )

    radial_operator = torch.arange(32, dtype=torch.float64).reshape(2, 16) / 17.0
    l2_operator = torch.arange(20, dtype=torch.float64).reshape(2, 10) / 13.0
    l2_source = torch.ones((2, 2, 5), dtype=torch.float64)
    actual = p13_response_mep_torch(
        radial_source=source,
        l2_source=l2_source,
        radial_operator=radial_operator,
        l2_operator=l2_operator,
    )
    expected_mep = source.reshape(2, -1) @ radial_operator.T + l2_source.reshape(
        2, -1
    ) @ l2_operator.T
    torch.testing.assert_close(actual, expected_mep, rtol=0.0, atol=1.0e-15)
