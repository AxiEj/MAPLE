from __future__ import annotations

import pytest

from maple.solvation.models.split_order_passive_scalar import (
    split_order_passive_energy_torch,
)


def test_split_order_scalar_has_radial_response_and_field_rigid_l2_source() -> None:
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    generator = torch.Generator().manual_seed(20260827)
    atom_count = 3
    radial_dimension = 8 * atom_count
    l2_dimension = 5 * atom_count
    permanent_radial = torch.randn(
        (atom_count, 8), dtype=dtype, generator=generator
    )
    permanent_l2 = torch.randn((atom_count, 5), dtype=dtype, generator=generator)
    response_matrix = torch.randn(
        (11, radial_dimension), dtype=dtype, generator=generator
    )
    radial = torch.randn(
        radial_dimension, dtype=dtype, generator=generator, requires_grad=True
    )
    l2 = torch.randn(
        l2_dimension, dtype=dtype, generator=generator, requires_grad=True
    )

    def energy(radial_values, l2_values):
        factor = response_matrix @ radial_values
        return split_order_passive_energy_torch(
            vacuum_energy=torch.tensor(-4.2, dtype=dtype),
            permanent_radial_source=permanent_radial,
            permanent_l2_source=permanent_l2,
            radial_field=radial_values.reshape(atom_count, 8),
            l2_field=l2_values.reshape(atom_count, 5),
            radial_response_factor=factor,
        )

    value = energy(radial, l2)
    radial_source, l2_source = torch.autograd.grad(
        value, (radial, l2), create_graph=True
    )
    hessian = torch.autograd.functional.hessian(energy, (radial, l2))

    torch.testing.assert_close(
        radial_source,
        permanent_radial.reshape(-1)
        - response_matrix.T @ response_matrix @ radial,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    torch.testing.assert_close(
        l2_source,
        permanent_l2.reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    radial_hessian, radial_l2 = hessian[0]
    l2_radial, l2_hessian = hessian[1]
    torch.testing.assert_close(
        radial_hessian,
        -response_matrix.T @ response_matrix,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    torch.testing.assert_close(
        radial_l2,
        torch.zeros((radial_dimension, l2_dimension), dtype=dtype),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        l2_radial,
        torch.zeros((l2_dimension, radial_dimension), dtype=dtype),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        l2_hessian,
        torch.zeros((l2_dimension, l2_dimension), dtype=dtype),
        rtol=0.0,
        atol=0.0,
    )
    assert float(torch.linalg.eigvalsh(radial_hessian).max()) <= 2.0e-12
