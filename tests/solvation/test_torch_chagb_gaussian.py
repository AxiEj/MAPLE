"""Contracts for the separately versioned Gaussian-sign CHA polar algebra."""

import math
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.torch_chagb import (
    cha_polar_from_inverse_born,
)
from maple.function.calculator.extra_correction.implicit.torch_chagb_gaussian import (
    GAUSSIAN_CHA_MODEL_IDENTITY,
    GaussianChaSizeDomainError,
    gaussian_cha_polar_from_inverse_born,
)


def _inputs(*, charges=None, requires_grad=False, scale=1.0):
    positions = (
        torch.tensor(
            [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.927297, 0.0]],
            dtype=torch.float64,
            requires_grad=requires_grad,
        )
        * scale
    )
    return (
        positions,
        torch.tensor(charges or [-0.8, 0.4, 0.4], dtype=torch.float64),
        torch.tensor([1.88, 1.04, 1.04], dtype=torch.float64),
        torch.tensor([0.5, 0.7, 0.7], dtype=torch.float64),
    )


def test_sigma_is_mandatory_positive_finite_scalar():
    inputs = _inputs()
    for sigma in (0.0, -0.1, float("nan"), float("inf"), 5e-324):
        with pytest.raises((TypeError, ValueError), match="sigma_e"):
            gaussian_cha_polar_from_inverse_born(*inputs, sigma_e=sigma)
    for invalid in (True, "0.01"):
        with pytest.raises((TypeError, ValueError), match="sigma_e"):
            gaussian_cha_polar_from_inverse_born(*inputs, sigma_e=cast(Any, invalid))


def test_erf_sign_is_zero_odd_bounded_and_reports_direct_tails():
    inputs = _inputs(charges=[0.0, 0.0, 0.0])
    zero = gaussian_cha_polar_from_inverse_born(*inputs, sigma_e=0.01)
    assert torch.equal(zero.smoothed_signs, torch.zeros(3, dtype=torch.float64))
    assert zero.charge_sigma_e == 0.01
    assert torch.equal(
        zero.weighted_signs_over_sigma, torch.zeros(3, dtype=torch.float64)
    )
    assert torch.equal(zero.gaussian_erfc_tails, torch.ones(3, dtype=torch.float64))

    positive = gaussian_cha_polar_from_inverse_born(
        *_inputs(charges=[0.02, -0.01, -0.01]), sigma_e=0.01
    )
    negative = gaussian_cha_polar_from_inverse_born(
        *_inputs(charges=[-0.02, 0.01, 0.01]), sigma_e=0.01
    )
    assert torch.allclose(positive.smoothed_signs, -negative.smoothed_signs)
    assert bool((positive.smoothed_signs.abs() <= 1.0).all())
    expected_tail = torch.erfc(
        positive.weighted_signs_over_sigma.abs() / math.sqrt(2.0)
    )
    assert torch.equal(positive.gaussian_erfc_tails, expected_tail)
    expected_scaled_tail = torch.special.erfcx(
        positive.weighted_signs_over_sigma.abs() / math.sqrt(2.0)
    )
    assert torch.equal(positive.gaussian_erfcx_tails, expected_scaled_tail)


def test_small_sigma_recovers_legacy_away_from_weighted_charge_zero():
    inputs = _inputs()
    exact = cha_polar_from_inverse_born(*inputs)
    smooth = gaussian_cha_polar_from_inverse_born(*inputs, sigma_e=1.0e-8)
    for field in ("polar_kcal_mol", "self_kcal_mol", "pair_kcal_mol"):
        assert torch.allclose(
            getattr(smooth, field), getattr(exact, field), atol=1e-11, rtol=0
        )
    assert GAUSSIAN_CHA_MODEL_IDENTITY == "chagb-r6-pbsa-gaussian-sign-v1"
    assert smooth.scope == "gaussian-sign-polar-algebra-with-supplied-inverse-born"


def test_live_inverse_born_chain_matches_explicit_chain_rule_contraction():
    positions, charges, radii, inverse = _inputs(requires_grad=True)
    inverse = inverse + 0.02 * positions.square().sum(dim=1)
    result = gaussian_cha_polar_from_inverse_born(
        positions, charges, radii, inverse, sigma_e=0.01
    )
    live_gradient, inverse_partial = torch.autograd.grad(
        result.polar_kcal_mol, (positions, inverse), retain_graph=True
    )
    detached_inverse_result = gaussian_cha_polar_from_inverse_born(
        positions, charges, radii, inverse.detach(), sigma_e=0.01
    )
    direct_gradient = torch.autograd.grad(
        detached_inverse_result.polar_kcal_mol, positions
    )[0]
    born_chain = live_gradient - direct_gradient
    expected_born_chain = inverse_partial[:, None] * 0.04 * positions

    assert torch.isfinite(live_gradient).all()
    assert not torch.equal(born_chain, torch.zeros_like(born_chain))
    assert torch.allclose(born_chain, expected_born_chain, atol=1e-12, rtol=1e-12)


def test_strict_gaussian_size_domain_rejects_before_legacy_ten_angstrom_switch():
    inputs = _inputs(scale=20.0)
    with pytest.raises(GaussianChaSizeDomainError, match="strictly below 9.5"):
        gaussian_cha_polar_from_inverse_born(*inputs, sigma_e=0.01)


@pytest.mark.parametrize("radius", [9.5, math.nextafter(9.5, math.inf)])
def test_mathematical_size_at_or_above_cap_fails_closed_despite_roundoff(radius):
    with pytest.raises(GaussianChaSizeDomainError, match="strictly below 9.5"):
        gaussian_cha_polar_from_inverse_born(
            torch.zeros((1, 3), dtype=torch.float64),
            torch.tensor([0.1], dtype=torch.float64),
            torch.tensor([radius], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            sigma_e=0.01,
        )
