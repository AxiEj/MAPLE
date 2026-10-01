"""Concave three-contact probe patches; no multi-probe overlap claim."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.torch_continuum_r6_patches import (
    contact_patch_inverse_cube,
    local_patch_gauss_flux,
    pair_torus_inverse_cube,
    triple_probe_patch_inverse_cube,
)
from maple.function.calculator.extra_correction.implicit.torch_continuum_ses_geometry import (
    triple_probe_centers,
)
from maple.function.calculator.extra_correction.implicit.torch_chagb import (
    cha_polar_from_inverse_born,
)


def _equilateral():
    height = 3.0 * math.sqrt(3.0) / 2.0
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.5, height, 0.0]],
        dtype=torch.float64,
    )
    intrinsic = torch.full((3,), 2.12, dtype=torch.float64)
    return positions, intrinsic


def test_probe_patch_areas_match_independent_spherical_solid_angle():
    positions, intrinsic = _equilateral()
    geometry = triple_probe_centers(positions, intrinsic, (0, 1, 2))
    result = triple_probe_patch_inverse_cube(positions, intrinsic, order=48)
    assert result.complete_ses is False
    assert result.patch_areas_angstrom2.shape == (2,)
    assert result.inverse_cube_per_angstrom3.shape == (3,)
    for index, center in enumerate(geometry.centers):
        directions = (positions - center) / (intrinsic + 0.88)[:, None]
        determinant = torch.dot(
            directions[0], torch.linalg.cross(directions[1], directions[2])
        )
        denominator = 1.0 + sum(
            torch.dot(directions[first], directions[second])
            for first, second in ((0, 1), (1, 2), (2, 0))
        )
        exact_area = 2.0 * torch.atan2(determinant.abs(), denominator) * 0.88**2
        assert (
            abs(result.patch_areas_angstrom2[index].item() - exact_area.item()) < 1e-10
        )


def test_probe_patch_flux_is_rigidly_invariant_and_coordinate_connected():
    positions, intrinsic = _equilateral()
    positions.requires_grad_(True)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    baseline = triple_probe_patch_inverse_cube(positions, intrinsic, order=48)
    turned = triple_probe_patch_inverse_cube(
        positions.detach() @ rotation.T + 4.0, intrinsic, order=48
    )
    assert (
        torch.max(
            torch.abs(
                baseline.inverse_cube_per_angstrom3.detach()
                - turned.inverse_cube_per_angstrom3
            )
        )
        < 1e-10
    )
    scalar = baseline.inverse_cube_per_angstrom3.sum()
    gradient = torch.autograd.grad(scalar, positions)[0]
    assert torch.isfinite(gradient).all()
    assert torch.max(torch.abs(gradient.sum(dim=0))).item() < 1e-10
    step = 1e-4
    plus = positions.detach().clone()
    minus = positions.detach().clone()
    plus[1, 0] += step
    minus[1, 0] -= step
    numerical = (
        triple_probe_patch_inverse_cube(plus, intrinsic, order=48)
        .inverse_cube_per_angstrom3.sum()
        .item()
        - triple_probe_patch_inverse_cube(minus, intrinsic, order=48)
        .inverse_cube_per_angstrom3.sum()
        .item()
    ) / (2.0 * step)
    assert abs(gradient[1, 0].item() - numerical) < 2e-6


def test_probe_patch_rejects_untrimmed_four_atom_overlap():
    positions, intrinsic = _equilateral()
    extra = torch.tensor([[1.5, 0.8, 5.0]], dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="probe-patch clipping"):
        triple_probe_patch_inverse_cube(
            torch.cat((positions, extra)),
            torch.cat((intrinsic, torch.tensor([1.5], dtype=torch.float64))),
        )


def test_three_site_overlapping_probe_patches_fail_closed():
    positions = torch.tensor(
        [
            [0.17920422329642754, -1.5817956760039478, -0.04972809818756493],
            [1.1820154305790893, -2.2851647303846394, -0.7779467937683708],
            [-3.232078657661435, -2.192214157634459, -3.130949564245945],
        ],
        dtype=torch.float64,
    )
    intrinsic = torch.tensor(
        [1.5182226355507369, 1.9931665844866822, 1.3961391083594525],
        dtype=torch.float64,
    )
    geometry = triple_probe_centers(positions, intrinsic, (0, 1, 2))
    assert len(geometry.centers) == 2
    assert torch.linalg.vector_norm(geometry.centers[0] - geometry.centers[1]) < 1.76
    with pytest.raises(ValueError, match="Overlapping probe spheres"):
        triple_probe_patch_inverse_cube(positions, intrinsic)


def test_three_atom_patch_surface_satisfies_gauss_closure():
    positions, intrinsic = _equilateral()
    closure = local_patch_gauss_flux(positions, intrinsic, order=48)
    assert closure.shape == (3,)
    assert torch.max(torch.abs(closure - 1.0)).item() < 2e-8


def test_contact_latitude_arc_birth_is_not_a_molecular_topology_failure():
    positions = torch.tensor(
        [
            [1.2493754057392656, -0.03985466154323591, 0.12145113529742908],
            [-1.2789295031575014, 0.7731330770947582, -0.9168054941284457],
            [-0.24293545444449316, -1.8840433467085722, -2.067374692353932],
        ],
        dtype=torch.float64,
    )
    intrinsic = torch.tensor(
        [1.4688652001514841, 2.3090226046075975, 2.192138406897148],
        dtype=torch.float64,
    )
    closure = local_patch_gauss_flux(positions, intrinsic, order=24)
    assert torch.max(torch.abs(closure - 1.0)).item() < 2e-7


def test_simple_three_site_cha_polar_graph_closes_all_cartesian_directions():
    """A local smooth-branch canary, not a general SES force qualification."""
    positions, intrinsic = _equilateral()
    charges = torch.tensor([0.4, -0.2, -0.2], dtype=torch.float64)

    def polar_energy(coords, order):
        flux = contact_patch_inverse_cube(
            coords, intrinsic, mu_order=order, phi_order=order
        ).inverse_cube_per_angstrom3
        for pair in ((0, 1), (0, 2), (1, 2)):
            flux = (
                flux
                + pair_torus_inverse_cube(
                    coords, intrinsic, pair, theta_order=order, meridian_order=order
                ).inverse_cube_per_angstrom3
            )
        flux = (
            flux
            + triple_probe_patch_inverse_cube(
                coords, intrinsic, order=order
            ).inverse_cube_per_angstrom3
        )
        assert torch.all(flux > 0.0)
        return cha_polar_from_inverse_born(
            coords, charges, intrinsic, flux.pow(1.0 / 3.0)
        ).polar_kcal_mol

    positions.requires_grad_(True)
    energy = polar_energy(positions, 32)
    gradient = torch.autograd.grad(energy, positions)[0]
    refined = polar_energy(positions.detach(), 48)
    assert abs(energy.item() - refined.item()) < 1e-8
    step = 1e-4
    numerical = torch.empty_like(positions)
    for atom in range(3):
        for axis in range(3):
            plus = positions.detach().clone()
            minus = positions.detach().clone()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            numerical[atom, axis] = (
                polar_energy(plus, 32) - polar_energy(minus, 32)
            ) / (2.0 * step)
    assert torch.max(torch.abs(gradient - numerical)).item() < 2e-5
    assert torch.max(torch.abs(gradient.sum(dim=0))).item() < 1e-10
