"""Strict domain certificate for the exact single-cover dispersion kernel."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.torch_continuum_dispersion import (
    DISPERSION_PROBE_ANGSTROM,
)
from maple.function.calculator.extra_correction.implicit.torch_continuum_dispersion_domain import (
    SingleCoverSASCertificate,
    SingleCoverSASRejection,
    certify_single_covering_dispersion_sas,
)


def _water():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.926627, 0.0]],
        dtype=torch.float64,
    )
    rmin = torch.tensor([1.82, 0.3019, 0.3019], dtype=torch.float64)
    return positions, rmin


def test_water_has_unique_oxygen_cover_with_live_dispersion_radii():
    positions, rmin = _water()
    result = certify_single_covering_dispersion_sas(positions, rmin)
    assert isinstance(result, SingleCoverSASCertificate)
    assert result.status == "certified"
    assert result.reason == "UNIQUE_STRICT_COVER"
    assert result.covering_index == 0
    assert result.dtype == "torch.float64"
    assert result.device == "cpu"
    torch.testing.assert_close(
        result.sas_radii_angstrom,
        rmin + DISPERSION_PROBE_ANGSTROM,
        rtol=0.0,
        atol=0.0,
    )
    assert result.containment_margins_angstrom.shape == (2,)
    assert result.minimum_margin_angstrom.item() > 0.54


def test_certificate_is_permutation_and_rigid_motion_invariant():
    positions, rmin = _water()
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    permutation = torch.tensor([2, 0, 1])
    moved = positions @ rotation.T + torch.tensor([2.0, -3.0, 0.5])
    result = certify_single_covering_dispersion_sas(
        moved[permutation], rmin[permutation]
    )
    assert isinstance(result, SingleCoverSASCertificate)
    assert result.covering_index == 1


@pytest.mark.parametrize(
    ("positions", "rmin", "reason"),
    [
        (
            torch.zeros((2, 2), dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            "INVALID_POSITIONS",
        ),
        (
            torch.zeros((2, 3), dtype=torch.float32),
            torch.ones(2, dtype=torch.float64),
            "INVALID_POSITIONS",
        ),
        (
            torch.tensor([[0.0, 0.0, 0.0], [math.nan, 0.0, 0.0]], dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            "NONFINITE_INPUT",
        ),
        (
            torch.zeros((2, 3), dtype=torch.float64),
            torch.ones(3, dtype=torch.float64),
            "INVALID_RMIN",
        ),
        (
            torch.zeros((2, 3), dtype=torch.float64),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            "NONPOSITIVE_RMIN",
        ),
    ],
)
def test_certificate_returns_typed_rejections(positions, rmin, reason):
    result = certify_single_covering_dispersion_sas(positions, rmin)
    assert isinstance(result, SingleCoverSASRejection)
    assert result.status == "rejected"
    assert result.reason == reason


def test_disjoint_and_duplicate_spheres_are_rejected():
    disjoint = certify_single_covering_dispersion_sas(
        torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([1.0, 1.0], dtype=torch.float64),
    )
    duplicate = certify_single_covering_dispersion_sas(
        torch.zeros((2, 3), dtype=torch.float64),
        torch.tensor([1.0, 1.0], dtype=torch.float64),
    )
    assert isinstance(disjoint, SingleCoverSASRejection)
    assert disjoint.reason == "NO_UNIQUE_COVER"
    assert isinstance(duplicate, SingleCoverSASRejection)
    assert duplicate.reason == "CONTAINMENT_GUARD"


def test_scale_aware_guard_rejects_tangency_and_admits_clear_strict_cover():
    outer_rmin = 2.0 - DISPERSION_PROBE_ANGSTROM
    inner_rmin = 1.0 - DISPERSION_PROBE_ANGSTROM
    outer_radius = outer_rmin + DISPERSION_PROBE_ANGSTROM
    inner_radius = inner_rmin + DISPERSION_PROBE_ANGSTROM
    tangent_distance = outer_radius - inner_radius
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [tangent_distance, 0.0, 0.0]], dtype=torch.float64
    )
    rmin = torch.tensor([outer_rmin, inner_rmin], dtype=torch.float64)
    tangent = certify_single_covering_dispersion_sas(positions, rmin)
    assert isinstance(tangent, SingleCoverSASRejection)
    assert tangent.reason == "CONTAINMENT_GUARD"

    scale = max(1.0, outer_radius, tangent_distance + inner_radius)
    guard = 64.0 * torch.finfo(torch.float64).eps * scale
    inside_ulp = torch.nextafter(
        torch.tensor(tangent_distance, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
    )
    outside_ulp = torch.nextafter(
        torch.tensor(tangent_distance, dtype=torch.float64),
        torch.tensor(math.inf, dtype=torch.float64),
    )
    for guarded_distance in (inside_ulp, outside_ulp):
        positions[1, 0] = guarded_distance
        guarded = certify_single_covering_dispersion_sas(positions, rmin)
        assert isinstance(guarded, SingleCoverSASRejection)
        assert guarded.reason == "CONTAINMENT_GUARD"

    positions[1, 0] = tangent_distance - 2.0 * guard
    inside = certify_single_covering_dispersion_sas(positions, rmin)
    assert isinstance(inside, SingleCoverSASCertificate)
    positions[1, 0] = tangent_distance - 0.5 * guard
    guarded = certify_single_covering_dispersion_sas(positions, rmin)
    assert isinstance(guarded, SingleCoverSASRejection)
    assert guarded.reason == "CONTAINMENT_GUARD"


def test_exact_nextafter_fixture_resolves_the_frozen_64_eps_boundary():
    epsilon = torch.finfo(torch.float64).eps
    threshold_radius = 2.0 / (1.0 - 64.0 * epsilon)
    below = math.nextafter(threshold_radius, -math.inf)
    above = math.nextafter(threshold_radius, math.inf)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)

    def certify(cover_radius):
        rmin = torch.tensor(
            [
                cover_radius - DISPERSION_PROBE_ANGSTROM,
                1.0 - DISPERSION_PROBE_ANGSTROM,
            ],
            dtype=torch.float64,
        )
        return certify_single_covering_dispersion_sas(positions, rmin)

    for radius in (below, threshold_radius):
        result = certify(radius)
        assert isinstance(result, SingleCoverSASRejection)
        assert result.reason == "CONTAINMENT_GUARD"
    result = certify(above)
    assert isinstance(result, SingleCoverSASCertificate)
    assert result.reason == "UNIQUE_STRICT_COVER"
