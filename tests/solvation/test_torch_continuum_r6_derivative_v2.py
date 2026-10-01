"""Assembly and worst-terminal derivative proof for R6 derivative v2."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.torch_chagb_gaussian import (
    gaussian_cha_polar_from_inverse_born,
)
from maple.function.calculator.extra_correction.implicit.torch_continuum_chagb_domain import (
    CertifiedLocalPatchScope,
    certify_three_site_domain,
)
from maple.function.calculator.extra_correction.implicit.torch_continuum_r6_derivative_v2 import (
    _r6_inverse_born_v2,
)

POSITIONS = np.asarray(
    [
        [0.010538813934279549, 0.39164232041464536, 1.8309399258250978e-16],
        [0.768508800461066, -0.19237452253773285, -4.846003055979318e-18],
        [-0.77772761439535, -0.1507877978769215, -8.236316583063313e-17],
    ],
    dtype=np.float64,
)
RADII = torch.tensor([1.88, 1.04, 1.04], dtype=torch.float64)
CHARGES = torch.tensor(
    [-0.7846666666666666, 0.3923333333333333, 0.3923333333333333],
    dtype=torch.float64,
)


def _domain(positions):
    domain = certify_three_site_domain(positions, RADII)
    assert isinstance(domain, CertifiedLocalPatchScope)
    return domain


def _polar_energy(positions, order=64):
    inverse = _r6_inverse_born_v2(positions, RADII, _domain(positions), order)
    return gaussian_cha_polar_from_inverse_born(
        positions, CHARGES, RADII, inverse, sigma_e=0.01
    ).polar_kcal_mol


def _five_point_force(flat_index, h):
    energies = []
    for multiplier in (-2, -1, 1, 2):
        displaced = POSITIONS.copy().reshape(-1)
        displaced[flat_index] += multiplier * h
        tensor = torch.tensor(displaced.reshape(3, 3), dtype=torch.float64)
        energies.append(float(_polar_energy(tensor).detach()))
    derivative = (energies[0] - 8 * energies[1] + 8 * energies[2] - energies[3]) / (
        12 * h
    )
    return -derivative


def test_worst_v1_terminal_order64_same_scalar_ad_fd_meets_frozen_force_gates():
    positions = torch.tensor(POSITIONS, dtype=torch.float64, requires_grad=True)
    energy = _polar_energy(positions, order=64)
    analytic = -torch.autograd.grad(energy, positions)[0].detach().numpy()
    by_step = []
    for h in (2e-5, 1e-5, 5e-6):
        by_step.append(
            np.asarray([_five_point_force(index, h) for index in range(9)]).reshape(
                3, 3
            )
        )
    finest = by_step[-1]
    error = analytic - finest
    assert np.max(np.abs(error)) <= 2e-5
    assert np.sqrt(np.mean(error**2)) <= 5e-6
    assert np.max(np.abs(by_step[-1] - by_step[-2])) <= 1e-5


def test_inverse_born_assembly_is_live_finite_and_order_bounded():
    positions = torch.tensor(POSITIONS, dtype=torch.float64, requires_grad=True)
    inverse = _r6_inverse_born_v2(positions, RADII, _domain(positions), 64)
    assert inverse.shape == (3,)
    assert torch.isfinite(inverse).all()
    assert bool((inverse > 0.0).all())
    gradient = torch.autograd.grad(inverse.sum(), positions)[0]
    assert torch.isfinite(gradient).all()
    assert not torch.equal(gradient, torch.zeros_like(gradient))


def test_assembly_rejects_untyped_domain_or_exposed_triple_scope():
    positions = torch.tensor(POSITIONS, dtype=torch.float64)
    with pytest.raises(TypeError):
        _r6_inverse_born_v2(positions, RADII, object(), 64)
