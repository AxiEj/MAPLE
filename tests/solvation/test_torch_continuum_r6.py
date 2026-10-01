"""Independent exterior-volume canaries for the unregistered R6 SES slice."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.torch_continuum_r6 import (
    inverse_born_from_ses,
)
from maple.function.calculator.extra_correction.implicit.torch_continuum_r6_patches import (
    contact_patch_inverse_cube,
    pair_torus_inverse_cube,
)
from maple.function.calculator.extra_correction.implicit.torch_chagb import (
    cha_polar_from_inverse_born,
)


def _exterior_volume_reference(
    first: float, second: float, distance: float
) -> np.ndarray:
    """Independent cylindrical exterior R6 integral, not a surface quadrature."""
    a, b, d, probe = first, second, distance, 0.88
    expanded_a, expanded_b = a + probe, b + probe

    if d >= expanded_a + expanded_b:
        segments = [(-a, a, "first"), (a, d - b, "empty"), (d - b, d + b, "second")]
        lower, upper = -a, d + b
    elif d <= abs(expanded_a - expanded_b):
        outer = "first" if expanded_a > expanded_b else "second"
        center, radius = (0.0, a) if outer == "first" else (d, b)
        lower, upper = center - radius, center + radius
        segments = [(lower, upper, outer)]
    else:
        along = (expanded_a**2 - expanded_b**2 + d**2) / (2.0 * d)
        circle_radius = math.sqrt(expanded_a**2 - along**2)
        contact_a = a * along / expanded_a
        contact_b = d + b * (along - d) / expanded_b
        lower, upper = -a, d + b
        if circle_radius > probe:
            segments = [
                (-a, contact_a, "first"),
                (contact_a, contact_b, "torus"),
                (contact_b, d + b, "second"),
            ]
        else:
            opening = math.acos(circle_radius / probe)
            cut_left = along - probe * math.sin(opening)
            cut_right = along + probe * math.sin(opening)
            segments = [
                (-a, contact_a, "first"),
                (contact_a, cut_left, "torus"),
                (cut_left, cut_right, "empty"),
                (cut_right, contact_b, "torus"),
                (contact_b, d + b, "second"),
            ]

    def cross_section_squared(z: float, kind: str) -> float:
        if kind == "first":
            return max(0.0, a * a - z * z)
        if kind == "second":
            return max(0.0, b * b - (z - d) ** 2)
        if kind == "torus":
            radial = circle_radius - math.sqrt(max(0.0, probe**2 - (z - along) ** 2))
            return radial**2
        return 0.0

    values = []
    for target in (0.0, d):
        integral = -1.0 / (3.0 * (lower - target) ** 3)
        integral += 1.0 / (3.0 * (upper - target) ** 3)
        for start, stop, kind in segments:
            if stop <= start:
                raise AssertionError("reference segments are not positively ordered")
            contribution, error = quad(
                lambda z: ((z - target) ** 2 + cross_section_squared(z, kind)) ** -2,
                start,
                stop,
                epsabs=1e-12,
                epsrel=1e-12,
                limit=1000,
            )
            assert error < 1e-9
            integral += contribution
        values.append((3.0 * integral / 8.0) ** (1.0 / 3.0))
    return np.asarray(values)


@pytest.mark.parametrize(
    "radii,distance,regime",
    [
        ((2.12, 1.88), 1.5, "regular-torus"),
        ((1.3241121166457712, 1.4505066279865835), 4.350501094750514, "spindle-torus"),
        ((1.0899102780582082, 2.1834859558811206), 4.879939493705011, "spindle-torus"),
        ((1.4, 1.2), 5.0, "disjoint"),
        ((2.0, 1.0), 0.5, "contained"),
    ],
)
def test_two_site_r6_matches_independent_exterior_volume(radii, distance, regime):
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=torch.float64
    )
    result = inverse_born_from_ses(positions, torch.tensor(radii, dtype=torch.float64))
    expected = _exterior_volume_reference(*radii, distance)
    assert result.regime == regime
    assert (
        np.max(np.abs(result.inverse_born_per_angstrom.detach().numpy() - expected))
        < 2e-10
    )


def test_one_site_r6_is_exact_born_sphere():
    positions = torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float64)
    radius = torch.tensor([2.12], dtype=torch.float64)
    result = inverse_born_from_ses(positions, radius)
    assert result.regime == "isolated"
    assert abs(result.inverse_born_per_angstrom.item() - 1.0 / radius.item()) < 1e-14


@pytest.mark.parametrize(
    "radii,distance",
    [
        ((2.12, 1.88), 1.5),
        ((1.3241121166457712, 1.4505066279865835), 4.350501094750514),
    ],
)
def test_two_site_r6_coordinate_derivative_matches_independent_volume(radii, distance):
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [distance, 0.2, -0.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    actual_distance = (
        torch.linalg.vector_norm(positions[1] - positions[0]).detach().item()
    )
    result = inverse_born_from_ses(positions, torch.tensor(radii, dtype=torch.float64))
    gradient = torch.autograd.grad(result.inverse_born_per_angstrom.sum(), positions)[0]
    step = 1e-4
    plus = _exterior_volume_reference(*radii, actual_distance + step).sum()
    minus = _exterior_volume_reference(*radii, actual_distance - step).sum()
    radial_derivative = (plus - minus) / (2.0 * step)
    direction = (positions[1] - positions[0]).detach().numpy() / actual_distance
    assert (
        np.max(np.abs(gradient[1].detach().numpy() - radial_derivative * direction))
        < 2e-7
    )
    assert np.max(np.abs(gradient.sum(dim=0).detach().numpy())) < 1e-12


def test_r6_requires_complete_multiatom_ses_before_use():
    with pytest.raises(NotImplementedError, match="multiatom SES"):
        inverse_born_from_ses(
            torch.zeros((3, 3), dtype=torch.float64),
            torch.ones(3, dtype=torch.float64),
        )


def test_seeded_regular_and_spindle_geometries_match_exterior_volume():
    rng = np.random.default_rng(20260923)
    regimes = set()
    maximum_error = 0.0
    for _ in range(100):
        first = float(rng.uniform(1.0, 2.5))
        second = float(rng.uniform(1.0, 2.5))
        distance = float(
            rng.uniform(abs(first - second) + 0.15, first + second + 1.76 - 0.15)
        )
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=torch.float64
        )
        result = inverse_born_from_ses(
            positions, torch.tensor([first, second], dtype=torch.float64)
        )
        reference = _exterior_volume_reference(first, second, distance)
        regimes.add(result.regime)
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(result.inverse_born_per_angstrom.numpy() - reference))),
        )
    assert regimes == {"regular-torus", "spindle-torus"}
    assert maximum_error < 2e-10


def test_spindle_pinch_is_an_explicit_derivative_event():
    radius = 1.4
    probe = 0.88
    pinch_distance = 2.0 * math.sqrt((radius + probe) ** 2 - probe**2)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [pinch_distance, 0.0, 0.0]], dtype=torch.float64
    )
    with pytest.raises(ValueError, match="pinching spindle"):
        inverse_born_from_ses(
            positions, torch.tensor([radius, radius], dtype=torch.float64)
        )


@pytest.mark.parametrize(
    "radii,distance",
    [
        ((2.12, 1.88), 1.5),
        ((1.3241121166457712, 1.4505066279865835), 4.350501094750514),
    ],
)
def test_two_site_r6_hvp_is_finite_within_one_regime(radii, distance):
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [distance, 0.2, -0.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    radii_tensor = torch.tensor(radii, dtype=torch.float64)
    direction = torch.tensor([[0.2, -0.1, 0.3], [-0.3, 0.4, -0.2]], dtype=torch.float64)
    result = inverse_born_from_ses(positions, radii_tensor)
    gradient = torch.autograd.grad(
        result.inverse_born_per_angstrom.sum(), positions, create_graph=True
    )[0]
    hvp = torch.autograd.grad((gradient * direction).sum(), positions)[0]
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(hvp).all()


@pytest.mark.parametrize(
    "radii,distance",
    [
        ((2.12, 1.88), 1.5),
        ((1.3241121166457712, 1.4505066279865835), 4.350501094750514),
    ],
)
def test_two_site_cha_polar_force_includes_born_geometry_response(radii, distance):
    intrinsic = torch.tensor(radii, dtype=torch.float64)
    charges = torch.tensor([0.4, -0.4], dtype=torch.float64)

    def scalar(positions):
        inverse = inverse_born_from_ses(positions, intrinsic).inverse_born_per_angstrom
        return cha_polar_from_inverse_born(
            positions, charges, intrinsic, inverse
        ).polar_kcal_mol

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [distance, 0.2, -0.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    with torch.autograd.detect_anomaly():
        energy = scalar(positions)
        gradient = torch.autograd.grad(energy, positions)[0]
    step = 1e-4
    plus = positions.detach().clone()
    minus = positions.detach().clone()
    plus[1, 1] += step
    minus[1, 1] -= step
    numerical = (scalar(plus).item() - scalar(minus).item()) / (2.0 * step)
    assert abs(gradient[1, 1].item() - numerical) < 2e-5
    assert torch.max(torch.abs(gradient.sum(dim=0))).item() < 1e-10


@pytest.mark.parametrize(
    "radii,distance",
    [
        ((2.12, 1.88), 1.5),
        ((1.3241121166457712, 1.4505066279865835), 4.350501094750514),
    ],
)
def test_pair_torus_flux_closes_independent_two_atom_exterior_volume(radii, distance):
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=torch.float64
    )
    intrinsic = torch.tensor(radii, dtype=torch.float64)
    torus = pair_torus_inverse_cube(
        positions, intrinsic, (0, 1), theta_order=32, meridian_order=32
    )
    contact = contact_patch_inverse_cube(
        positions, intrinsic, mu_order=48, phi_order=48
    )
    reference = _exterior_volume_reference(*radii, distance) ** 3
    combined = torus.inverse_cube_per_angstrom3 + contact.inverse_cube_per_angstrom3
    assert torus.complete_ses is False
    assert np.max(np.abs(combined.detach().numpy() - reference)) < 2e-10


def test_three_atom_pair_torus_flux_is_rotation_covariant_and_differentiable():
    height = 3.0 * math.sqrt(3.0) / 2.0
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.5, height, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    intrinsic = torch.full((3,), 2.12, dtype=torch.float64)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    baseline = pair_torus_inverse_cube(
        positions, intrinsic, (0, 1), theta_order=48, meridian_order=48
    )
    turned = pair_torus_inverse_cube(
        positions.detach() @ rotation.T + 2.0,
        intrinsic,
        (0, 1),
        theta_order=48,
        meridian_order=48,
    )
    assert (
        torch.max(
            torch.abs(
                baseline.inverse_cube_per_angstrom3.detach()
                - turned.inverse_cube_per_angstrom3
            )
        )
        < 2e-8
    )
    gradient = torch.autograd.grad(
        baseline.inverse_cube_per_angstrom3.sum(), positions
    )[0]
    assert torch.isfinite(gradient).all()
