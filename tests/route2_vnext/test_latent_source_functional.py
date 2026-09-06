from __future__ import annotations

import numpy as np
import pytest

from maple.solvation.models.latent_source_functional import (
    QuadraticLatentSourceFunctional,
    quadratic_latent_anchored_energy_torch,
)


def _functional() -> QuadraticLatentSourceFunctional:
    generator = np.random.default_rng(20260824)
    raw = generator.normal(size=(7, 7))
    hessian = raw.T @ raw + 0.7 * np.eye(7)
    linear = generator.normal(size=7)
    charge = np.asarray([1.0, 0.3, -0.2, 0.0, 0.0, 0.1, -0.4])
    mapping = generator.normal(size=(5, 7))
    return QuadraticLatentSourceFunctional(
        hessian=hessian,
        linear_term=linear,
        charge_covector=charge,
        total_charge=0.0,
        field_to_source=mapping,
    )


def test_latent_scalar_is_zero_anchored_and_charge_conserving() -> None:
    functional = _functional()
    zero = functional.solve(np.zeros(functional.field_dimension))
    field = np.linspace(-0.2, 0.25, functional.field_dimension)
    state = functional.solve(field)

    assert zero.anchored_energy == pytest.approx(0.0, abs=2.0e-14)
    assert zero.charge_constraint_residual < 1.0e-12
    assert state.charge_constraint_residual < 1.0e-12
    assert state.kkt_max_absolute_residual < 1.0e-11
    induced_latent = state.latent_source - zero.latent_source
    charge = functional.latent_charge_covector
    assert float(np.vdot(charge, induced_latent)) == pytest.approx(
        0.0,
        abs=2.0e-12,
    )
    assert functional.capabilities.enabled_tiers == ()


def test_energy_gradient_and_source_jvp_match_finite_differences() -> None:
    functional = _functional()
    generator = np.random.default_rng(42)
    field = generator.normal(size=functional.field_dimension) * 0.1
    direction = generator.normal(size=functional.field_dimension)
    step = 2.0e-6

    plus = functional.solve(field + step * direction)
    minus = functional.solve(field - step * direction)
    energy_fd = (plus.anchored_energy - minus.anchored_energy) / (2.0 * step)
    gradient = functional.energy_field_gradient(field)
    assert energy_fd == pytest.approx(
        float(np.vdot(gradient, direction)),
        rel=2.0e-9,
        abs=2.0e-9,
    )

    source_fd = (plus.physical_source - minus.physical_source) / (2.0 * step)
    np.testing.assert_allclose(
        functional.source_jvp(direction),
        source_fd,
        rtol=2.0e-9,
        atol=2.0e-9,
    )


def test_susceptibility_is_reciprocal_passive_and_energy_is_concave() -> None:
    functional = _functional()
    susceptibility = functional.susceptibility()
    eigenvalues = np.linalg.eigvalsh(susceptibility)
    generator = np.random.default_rng(91)
    left = generator.normal(size=functional.field_dimension)
    right = generator.normal(size=functional.field_dimension)

    np.testing.assert_allclose(
        susceptibility,
        susceptibility.T,
        rtol=0.0,
        atol=2.0e-13,
    )
    assert eigenvalues[0] >= -2.0e-12
    assert float(np.vdot(left, susceptibility @ right)) == pytest.approx(
        float(np.vdot(right, susceptibility @ left)),
        rel=2.0e-13,
        abs=2.0e-13,
    )
    hvp = functional.energy_field_hvp(left)
    assert float(np.vdot(left, hvp)) <= 2.0e-12


def test_constant_potential_direction_has_exact_fixed_charge_work() -> None:
    hessian = np.diag([1.2, 1.8, 2.1, 2.7])
    charge = np.asarray([1.0, 0.0, 0.0, 0.0])
    functional = QuadraticLatentSourceFunctional(
        hessian=hessian,
        linear_term=np.asarray([0.4, -0.2, 0.1, 0.3]),
        charge_covector=charge,
        total_charge=1.5,
        field_to_source=np.eye(4),
    )
    field = np.asarray([0.2, -0.1, 0.05, 0.3])
    shift = 0.37
    shifted = field + shift * charge

    energy_difference = (
        functional.solve(shifted).anchored_energy
        - functional.solve(field).anchored_energy
    )
    assert energy_difference == pytest.approx(-shift * 1.5, abs=2.0e-13)
    np.testing.assert_allclose(
        functional.induced_source(shifted),
        functional.induced_source(field),
        rtol=0.0,
        atol=2.0e-13,
    )


def test_nonpositive_latent_hessian_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        QuadraticLatentSourceFunctional(
            hessian=np.diag([1.0, 0.0]),
            linear_term=np.zeros(2),
            charge_covector=np.ones(2),
            total_charge=0.0,
            field_to_source=np.eye(2),
        )


def test_torch_scalar_autograd_generates_source_and_passive_hessian() -> None:
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    raw = torch.tensor(
        [[1.2, 0.1, -0.2], [0.3, 1.1, 0.2], [-0.1, 0.4, 0.9]],
        dtype=dtype,
    )
    hessian = raw.T @ raw + 0.8 * torch.eye(3, dtype=dtype)
    linear = torch.tensor([0.2, -0.1, 0.35], dtype=dtype)
    charge = torch.tensor([1.0, 0.2, -0.3], dtype=dtype)
    mapping = torch.tensor(
        [[1.0, 0.2, -0.1], [0.1, -0.3, 0.8]],
        dtype=dtype,
    )
    field = torch.tensor([0.12, -0.08], dtype=dtype, requires_grad=True)

    def energy(values):
        return quadratic_latent_anchored_energy_torch(
            hessian=hessian,
            linear_term=linear,
            charge_covector=charge,
            total_charge=0.0,
            field_to_source=mapping,
            field=values,
        )

    value = energy(field)
    gradient = torch.autograd.grad(value, field, create_graph=True)[0]
    hessian_field = torch.autograd.functional.hessian(energy, field)

    assert value.ndim == 0
    assert torch.all(torch.isfinite(gradient))
    torch.testing.assert_close(hessian_field, hessian_field.T, rtol=0.0, atol=1.0e-12)
    assert float(torch.linalg.eigvalsh(hessian_field).max()) <= 2.0e-12
