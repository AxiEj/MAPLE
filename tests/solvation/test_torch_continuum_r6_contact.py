"""Contact-patch R6 flux only; this is not a closed multiatom SES."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.torch_continuum_r6_patches import (
    contact_patch_inverse_cube,
)


def _two_body_contact_reference(
    first: float, second: float, distance: float
) -> np.ndarray:
    a, b, d, probe = first, second, distance, 0.88
    expanded_a, expanded_b = a + probe, b + probe
    if d >= expanded_a + expanded_b:
        limits = [(0, 1.0), (1, 1.0)]
    elif d <= abs(expanded_a - expanded_b):
        outer = 0 if expanded_a > expanded_b else 1
        limits = [(outer, 1.0)]
    else:
        along = (expanded_a**2 - expanded_b**2 + d**2) / (2.0 * d)
        limits = [(0, along / expanded_a), (1, (d - along) / expanded_b)]
    values = np.zeros(2)
    for source, upper in limits:
        radius = (a, b)[source]
        values[source] += (upper + 1.0) / (2.0 * radius**3)
        cross, error = quad(
            lambda cosine: (radius - d * cosine)
            / (radius**2 + d**2 - 2.0 * radius * d * cosine) ** 3,
            -1.0,
            upper,
            epsabs=1e-13,
            epsrel=1e-12,
        )
        assert error < 1e-11
        values[1 - source] += 0.5 * radius**2 * cross
    return values


def test_one_atom_contact_flux_is_exact_isolated_sphere():
    radius = 2.12
    positions = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64)
    result = contact_patch_inverse_cube(
        positions, torch.tensor([radius], dtype=torch.float64)
    )
    assert result.complete_ses is False
    assert abs(result.inverse_cube_per_angstrom3.item() - radius**-3) < 1e-10


@pytest.mark.parametrize(
    "radii,distance",
    [
        ((2.12, 1.88), 1.5),
        ((1.3241121166457712, 1.4505066279865835), 4.350501094750514),
        ((1.4, 1.2), 5.0),
        ((2.0, 1.0), 0.5),
    ],
)
def test_two_atom_contact_flux_matches_axial_reference(radii, distance):
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=torch.float64
    )
    result = contact_patch_inverse_cube(
        positions, torch.tensor(radii, dtype=torch.float64), mu_order=32, phi_order=32
    )
    expected = _two_body_contact_reference(*radii, distance)
    assert (
        np.max(np.abs(result.inverse_cube_per_angstrom3.detach().numpy() - expected))
        < 2e-9
    )


@pytest.mark.parametrize(
    "radii,distance",
    [
        ((2.12, 1.88), 1.5),
        ((1.3241121166457712, 1.4505066279865835), 4.350501094750514),
    ],
)
def test_contact_chart_refinement_reaches_independent_oracle(radii, distance):
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=torch.float64
    )
    intrinsic = torch.tensor(radii, dtype=torch.float64)
    reference = _two_body_contact_reference(*radii, distance)
    coarse = contact_patch_inverse_cube(positions, intrinsic, mu_order=16, phi_order=16)
    fine = contact_patch_inverse_cube(positions, intrinsic, mu_order=48, phi_order=48)
    coarse_error = np.max(np.abs(coarse.inverse_cube_per_angstrom3.numpy() - reference))
    fine_error = np.max(np.abs(fine.inverse_cube_per_angstrom3.numpy() - reference))
    assert fine_error < coarse_error
    assert fine_error < 1e-12


def test_two_atom_contact_flux_force_matches_own_scalar_difference():
    intrinsic = torch.tensor([2.12, 1.88], dtype=torch.float64)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.2, -0.1]],
        dtype=torch.float64,
        requires_grad=True,
    )

    def scalar(x):
        return contact_patch_inverse_cube(x, intrinsic).inverse_cube_per_angstrom3.sum()

    gradient = torch.autograd.grad(scalar(positions), positions)[0]
    step = 1e-4
    plus = positions.detach().clone()
    minus = positions.detach().clone()
    plus[1, 1] += step
    minus[1, 1] -= step
    numerical = (scalar(plus).item() - scalar(minus).item()) / (2.0 * step)
    assert abs(gradient[1, 1].item() - numerical) < 2e-5
    assert torch.max(torch.abs(gradient.sum(dim=0))).item() < 1e-8


def test_three_atom_contact_flux_is_rotation_covariant():
    height = 3.0 * math.sqrt(3.0) / 2.0
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.5, height, 0.0]],
        dtype=torch.float64,
    )
    intrinsic = torch.full((3,), 2.12, dtype=torch.float64)
    rotation = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    base = contact_patch_inverse_cube(positions, intrinsic, mu_order=48, phi_order=48)
    turned = contact_patch_inverse_cube(
        positions @ rotation.T + 2.0, intrinsic, mu_order=48, phi_order=48
    )
    assert (
        torch.max(
            torch.abs(
                base.inverse_cube_per_angstrom3 - turned.inverse_cube_per_angstrom3
            )
        )
        < 2e-6
    )
