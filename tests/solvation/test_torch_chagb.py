"""Tests of the CHA polar algebra only, not a complete solvent-force model."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _kernel():
    return importlib.import_module(
        "maple.function.calculator.extra_correction.implicit.torch_chagb"
    )


def _tensor(values, *, requires_grad=False):
    return torch.tensor(values, dtype=torch.float64, requires_grad=requires_grad)


def _case():
    return (
        _tensor([[0.0, 0.0, 0.0], [1.2, 0.3, 0.0], [0.2, 1.1, 0.1]]),
        _tensor([0.35, -0.55, 0.2]),
        _tensor([1.7, 1.5, 1.2]),
        _tensor([0.40, 0.50, 0.48]),
    )


@pytest.mark.parametrize("charge", [-1.0, 0.0, 1.0])
def test_single_site_matches_the_closed_form(charge):
    module = _kernel()
    radius = 1.7
    raw_born = 2.1
    result = module.cha_polar_from_inverse_born(
        _tensor([[1.2, -0.3, 0.5]]),
        _tensor([charge]),
        _tensor([radius]),
        _tensor([1.0 / raw_born]),
    )
    beta = 0.571412 / 78.5
    dielectric = (1.0 - 1.0 / 78.5) / (1.0 + beta)
    mu = 1.0 + np.sign(charge) * 0.586 / (raw_born + (1.4 - 0.52))
    expected = (
        -0.5
        * (18.2223 * charge) ** 2
        * dielectric
        * (1.0 / (raw_born * mu) + beta / radius)
    )
    assert float(result.electrostatic_size_angstrom) == pytest.approx(radius, abs=1e-13)
    assert float(result.polar_kcal_mol) == pytest.approx(expected, abs=1e-11)
    assert float(result.pair_kcal_mol) == 0.0
    assert float(result.cha_factors[0]) == pytest.approx(mu, abs=1e-13)


def test_energy_parts_close_without_nonpolar_or_gas_terms():
    result = _kernel().cha_polar_from_inverse_born(*_case())
    torch.testing.assert_close(
        result.polar_kcal_mol,
        result.self_kcal_mol + result.pair_kcal_mol,
        rtol=0.0,
        atol=1e-12,
    )
    assert result.complete_coordinate_graph is False
    assert result.scope == "polar-algebra-with-supplied-inverse-born"
    assert not hasattr(result, "forces_hartree_per_angstrom")
    assert not hasattr(_kernel(), "TorchChaGB")


def test_source_default_real_inverse_born_shift_and_threshold_are_preserved():
    shift = _kernel().cha_inverse_born_shift
    size = _tensor([9.999, 10.0, 10.001])
    actual = shift(size)
    slope = float(np.float32(0.0015))
    intercept = float(np.float32(0.01))
    expected = _tensor([0.0, slope * 10.0 + intercept, slope * 10.001 + intercept])
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-16)
    assert float(actual[1]) > 0.024


def test_zero_charges_give_zero_polar_energy_and_neutral_mu():
    positions, charges, radii, inverse_born = _case()
    result = _kernel().cha_polar_from_inverse_born(
        positions, torch.zeros_like(charges), radii, inverse_born
    )
    assert float(result.polar_kcal_mol) == 0.0
    torch.testing.assert_close(result.cha_factors, torch.ones_like(charges))


def test_effective_charge_sign_switch_is_not_smoothed():
    module = _kernel()
    # This neutral algebra example supplies Born radii independently of the
    # positions. It is deliberately NOT an R6/SES molecular-force benchmark.
    threshold = np.sqrt(4.0 * np.log(2.0) / 1.47)
    qi = _tensor([0.4, -0.8, 0.4])
    radii = _tensor([1.0, 1.0, 1.0])
    inv = _tensor([0.5, 0.5, 0.5])
    outputs = []
    for delta in (-1e-8, 1e-8):
        outputs.append(
            module.cha_polar_from_inverse_born(
                _tensor([[0.0, 0.0, 0.0], [threshold + delta, 0, 0], [10, 0, 0]]),
                qi,
                radii,
                inv,
            )
        )
    assert float(outputs[0].effective_charges_e[0]) < 0
    assert float(outputs[1].effective_charges_e[0]) > 0
    assert float(outputs[0].cha_factors[0]) == pytest.approx(
        1 - 0.586 / (2.0 + 1.4 - 0.52)
    )
    assert float(outputs[1].cha_factors[0]) == pytest.approx(
        1 + 0.586 / (2.0 + 1.4 - 0.52)
    )
    assert abs(float(outputs[1].polar_kcal_mol - outputs[0].polar_kcal_mol)) > 0.01


def test_active_sign_switch_refuses_a_derivative_claim():
    threshold = np.sqrt(4.0 * np.log(2.0) / 1.47)
    positions = _tensor(
        [[0.0, 0.0, 0.0], [threshold, 0.0, 0.0], [10.0, 0.0, 0.0]],
        requires_grad=True,
    )
    with pytest.raises(ValueError, match="effective-charge sign switch"):
        _kernel().cha_polar_from_inverse_born(
            positions,
            _tensor([0.4, -0.8, 0.4]),
            _tensor([1.0, 1.0, 1.0]),
            _tensor([0.5, 0.5, 0.5]),
        )


def test_electrostatic_size_switch_refuses_a_derivative_claim():
    with pytest.raises(ValueError, match="size-shift switch"):
        _kernel().cha_polar_from_inverse_born(
            _tensor([[0.0, 0.0, 0.0]], requires_grad=True),
            _tensor([1.0]),
            _tensor([10.0]),
            _tensor([0.5]),
        )


def test_algebra_partials_pass_first_and_second_order_gradcheck():
    module = _kernel()
    positions, charges, radii, inverse_born = _case()
    positions.requires_grad_(True)
    inverse_born.requires_grad_(True)

    def energy(x, inv):
        return module.cha_polar_from_inverse_born(x, charges, radii, inv).polar_kcal_mol

    assert torch.autograd.gradcheck(
        energy, (positions, inverse_born), eps=1e-6, atol=1e-5, rtol=1e-4
    )
    assert torch.autograd.gradgradcheck(
        energy, (positions, inverse_born), eps=1e-6, atol=1e-5, rtol=1e-4
    )


def test_geometry_dependent_inverse_born_tensor_is_not_detached():
    module = _kernel()
    positions, charges, radii, base = _case()
    positions.requires_grad_(True)

    def composed(x):
        # A known test graph, not a substitute for actual R6 geometry.
        inv = base + 0.01 * ((x - x.mean(dim=0)) ** 2).sum(dim=1)
        return module.cha_polar_from_inverse_born(x, charges, radii, inv).polar_kcal_mol

    assert torch.autograd.gradcheck(
        composed, (positions,), eps=1e-6, atol=1e-5, rtol=1e-4
    )
    inv = base + 0.01 * ((positions - positions.mean(dim=0)) ** 2).sum(dim=1)
    linked = torch.autograd.grad(composed(positions), positions)[0]
    frozen = torch.autograd.grad(
        module.cha_polar_from_inverse_born(
            positions, charges, radii, inv.detach()
        ).polar_kcal_mol,
        positions,
    )[0]
    assert torch.linalg.vector_norm(linked - frozen) > 1e-5


def test_translation_rotation_and_permutation_preserve_scalar():
    function = _kernel().cha_polar_from_inverse_born
    x, q, radii, inv = _case()
    baseline = function(x, q, radii, inv).polar_kcal_mol
    rotated = x @ _tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    permutation = torch.tensor([2, 0, 1])
    for moved, qi, ri, ui in (
        (x + _tensor([4.0, -2.0, 0.75]), q, radii, inv),
        (rotated, q, radii, inv),
        (x[permutation], q[permutation], radii[permutation], inv[permutation]),
    ):
        torch.testing.assert_close(
            function(moved, qi, ri, ui).polar_kcal_mol, baseline, rtol=1e-12, atol=1e-11
        )


@pytest.mark.parametrize(
    "bad", ["empty", "float32", "nan", "zero-radius", "bad-born", "shape", "collision"]
)
def test_invalid_inputs_fail_closed(bad):
    x, q, radii, inv = _case()
    if bad == "empty":
        x, q, radii, inv = x[:0], q[:0], radii[:0], inv[:0]
    elif bad == "float32":
        x = x.float()
    elif bad == "nan":
        x[0, 0] = float("nan")
    elif bad == "zero-radius":
        radii[0] = 0.0
    elif bad == "bad-born":
        inv[0] = -0.5
    elif bad == "shape":
        inv = inv[:, None]
    elif bad == "collision":
        x[1] = x[0]
    with pytest.raises((ValueError, TypeError)):
        _kernel().cha_polar_from_inverse_born(x, q, radii, inv)
