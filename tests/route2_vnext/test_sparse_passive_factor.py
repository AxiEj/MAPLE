from __future__ import annotations

import numpy as np
import pytest

from maple.solvation.models.sparse_passive_factor import (
    sparse_passive_p13_factor_torch,
    sparse_passive_radial_factor_torch,
)


def _inputs(torch):
    dtype = torch.float64
    generator = torch.Generator().manual_seed(20260827)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, -0.2, 0.3], [-0.4, 1.0, 0.2]],
        dtype=dtype,
    )
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.int64)
    raw_alpha = torch.randn((3, 3, 3), dtype=dtype, generator=generator)
    alpha = raw_alpha @ raw_alpha.transpose(-1, -2)
    return {
        "radial_field": torch.randn((3, 8), dtype=dtype, generator=generator),
        "positions_angstrom": positions,
        "edge_index": edge_index,
        "atomic_polarizability": alpha,
        "radial_gain": torch.randn((3,), dtype=dtype, generator=generator),
        "vector_coefficients": torch.randn(
            (3, 2, 2, 2), dtype=dtype, generator=generator
        ),
        "edge_coefficients": torch.randn((2, 2), dtype=dtype, generator=generator),
        "edge_cutoff": torch.tensor([0.8, 0.6], dtype=dtype),
        "scalar_radial_reshape": torch.tensor(0.13, dtype=dtype),
        "vector_radial_reshape": torch.tensor(-0.08, dtype=dtype),
    }


def test_sparse_factor_is_constant_potential_gauge_invariant() -> None:
    torch = pytest.importorskip("torch")
    inputs = _inputs(torch)
    reference = sparse_passive_radial_factor_torch(**inputs)
    shifted = inputs["radial_field"].clone()
    shifted[:, 0] += 0.37
    shifted[:, 1] += 0.37

    actual = sparse_passive_radial_factor_torch(
        **{**inputs, "radial_field": shifted}
    )
    torch.testing.assert_close(actual, reference, rtol=0.0, atol=3.0e-15)


def test_sparse_factor_norm_is_rotation_and_edge_orientation_invariant() -> None:
    torch = pytest.importorskip("torch")
    inputs = _inputs(torch)
    angle = 0.71
    rotation = torch.tensor(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    field = inputs["radial_field"].clone()
    electric_1 = field[:, (4, 2, 3)] @ rotation.T
    electric_2 = field[:, (7, 5, 6)] @ rotation.T
    rotated_field = field.clone()
    rotated_field[:, (4, 2, 3)] = electric_1
    rotated_field[:, (7, 5, 6)] = electric_2
    rotated_inputs = {
        **inputs,
        "radial_field": rotated_field,
        "positions_angstrom": inputs["positions_angstrom"] @ rotation.T,
        "atomic_polarizability": (
            rotation[None] @ inputs["atomic_polarizability"] @ rotation.T[None]
        ),
    }
    reference = sparse_passive_radial_factor_torch(**inputs)
    rotated = sparse_passive_radial_factor_torch(**rotated_inputs)
    reversed_inputs = {
        **inputs,
        "edge_index": inputs["edge_index"].flip(0),
    }
    reversed_factor = sparse_passive_radial_factor_torch(**reversed_inputs)

    torch.testing.assert_close(
        torch.dot(rotated, rotated),
        torch.dot(reference, reference),
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    torch.testing.assert_close(
        torch.dot(reversed_factor, reversed_factor),
        torch.dot(reference, reference),
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_sparse_factor_generates_a_symmetric_negative_semidefinite_hessian() -> None:
    torch = pytest.importorskip("torch")
    inputs = _inputs(torch)

    def energy(flat_field):
        factor = sparse_passive_radial_factor_torch(
            **{**inputs, "radial_field": flat_field.reshape(3, 8)}
        )
        return -0.5 * torch.dot(factor, factor)

    field = inputs["radial_field"].reshape(-1).requires_grad_(True)
    hessian = torch.autograd.functional.hessian(energy, field)
    eigenvalues = torch.linalg.eigvalsh(0.5 * (hessian + hessian.T))
    gauge = torch.zeros_like(field).reshape(3, 8)
    gauge[:, 0] = 1.0
    gauge[:, 1] = 1.0

    torch.testing.assert_close(hessian, hessian.T, rtol=0.0, atol=2.0e-13)
    assert float(eigenvalues.max()) <= 2.0e-12
    torch.testing.assert_close(
        hessian @ gauge.reshape(-1),
        torch.zeros_like(field),
        rtol=0.0,
        atol=3.0e-13,
    )


def test_p13_block_factor_adds_local_l2_response_without_cross_terms() -> None:
    torch = pytest.importorskip("torch")
    inputs = _inputs(torch)
    l2_gain = torch.tensor([0.7, -0.4, 0.9], dtype=torch.float64)
    radial = inputs["radial_field"].reshape(-1)
    l2 = torch.randn((15,), dtype=torch.float64, generator=torch.Generator().manual_seed(8))

    def energy(radial_values, l2_values):
        factor = sparse_passive_p13_factor_torch(
            **{**inputs, "radial_field": radial_values.reshape(3, 8)},
            l2_field=l2_values.reshape(3, 5),
            l2_gain=l2_gain,
        )
        return -0.5 * torch.dot(factor, factor)

    hessian = torch.autograd.functional.hessian(energy, (radial, l2))
    radial_hessian, radial_l2 = hessian[0]
    l2_radial, l2_hessian = hessian[1]
    expected_l2 = -torch.diag(torch.repeat_interleave(l2_gain.square(), 5))

    assert float(torch.linalg.eigvalsh(radial_hessian).max()) <= 2.0e-12
    torch.testing.assert_close(l2_hessian, expected_l2, rtol=0.0, atol=2.0e-15)
    torch.testing.assert_close(
        radial_l2,
        torch.zeros_like(radial_l2),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        l2_radial,
        torch.zeros_like(l2_radial),
        rtol=0.0,
        atol=0.0,
    )
