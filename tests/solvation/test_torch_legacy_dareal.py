from __future__ import annotations

import math

import pytest
import torch

from maple.solvation.surfaces.legacy_dareal import (
    LegacyDAREALConfig,
    LegacyDAREALTopologyError,
    legacy_dareal_areas_torch,
)


def _tensor(values, *, requires_grad=False):
    return torch.tensor(values, dtype=torch.float64, requires_grad=requires_grad)


def test_isolated_and_disjoint_spheres_have_full_area():
    single = legacy_dareal_areas_torch(_tensor([[0.0, 0.0, 0.0]]), _tensor([2.0]))
    assert single.areas_angstrom2.item() == pytest.approx(16.0 * math.pi, abs=1e-12)
    disjoint = legacy_dareal_areas_torch(
        _tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]), _tensor([1.0, 1.0])
    )
    assert disjoint.areas_angstrom2.tolist() == pytest.approx(
        [4.0 * math.pi] * 2, abs=1e-12
    )


def test_two_equal_overlapping_spheres_match_spherical_cap_formula():
    distance = 1.0
    result = legacy_dareal_areas_torch(
        _tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], requires_grad=True),
        _tensor([1.0, 1.0]),
    )
    expected = 2.0 * math.pi * (1.0 + distance / 2.0)
    assert result.areas_angstrom2.tolist() == pytest.approx(
        [expected, expected], abs=1e-12
    )


def test_contained_sphere_is_buried_without_changing_radii():
    result = legacy_dareal_areas_torch(
        _tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]), _tensor([2.0, 0.5])
    )
    assert result.areas_angstrom2.tolist() == pytest.approx(
        [16.0 * math.pi, 0.0], abs=1e-12
    )
    assert result.diagnostics.atom_branches == ("free", "contained")


def test_multicircle_polygon_is_twice_differentiable():
    positions = _tensor(
        [[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047], [0.0, -0.763239, -0.477047]],
        requires_grad=True,
    )
    radii = _tensor([1.92, 1.60, 1.60])
    result = legacy_dareal_areas_torch(positions, radii)
    assert "polygon" in result.diagnostics.atom_branches
    hessian = torch.autograd.functional.hessian(
        lambda value: legacy_dareal_areas_torch(value, radii).areas_angstrom2.sum(),
        positions,
    )
    assert torch.isfinite(hessian).all()


def test_topology_hash_is_stable_in_patch_and_changes_with_selected_arcs():
    radii = _tensor([1.92, 1.60, 1.60])
    reference = _tensor(
        [[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047], [0.0, -0.763239, -0.477047]]
    )
    nearby = reference.clone()
    nearby[1, 0] += 1.0e-5
    first = legacy_dareal_areas_torch(reference, radii).diagnostics.topology_sha256
    second = legacy_dareal_areas_torch(nearby, radii).diagnostics.topology_sha256
    disjoint = legacy_dareal_areas_torch(
        _tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]), radii
    ).diagnostics.topology_sha256
    assert first == second
    assert first != disjoint


def test_tangent_topology_fails_closed():
    with pytest.raises(LegacyDAREALTopologyError, match="tangent/containment"):
        legacy_dareal_areas_torch(
            _tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), _tensor([1.0, 1.0])
        )


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        (
            [
                [0.0, 0.0, 0.0],
                [1.1948939100958142, -1.257556058450997, 0.1628114045348581],
                [0.34995012805085546, -1.3773127035455652, -0.3629411868136929],
            ],
            "polygon-area modulo branch",
        ),
        (
            [
                [0.0, 0.0, 0.0],
                [0.30853595501340303, -0.20948742689356026, -0.7850173556984639],
                [-1.1116981738495975, -0.15458866989313425, -0.2563854228399214],
                [-0.7640428756729688, -1.156267253755817, 0.7207904992178946],
            ],
            "boundary-angle sort order",
        ),
    ],
)
def test_adversarial_polygon_branch_margins_fail_closed(positions, message):
    with pytest.raises(LegacyDAREALTopologyError, match=message):
        legacy_dareal_areas_torch(
            _tensor(positions),
            torch.ones(len(positions), dtype=torch.float64),
            config=LegacyDAREALConfig(relative_topology_tolerance=0.05),
        )
