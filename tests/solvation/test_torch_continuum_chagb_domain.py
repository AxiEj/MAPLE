"""Point-domain contracts for the unregistered three-site CHA foundation."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.torch_continuum_chagb_domain import (
    CertifiedLocalPatchScope,
    DomainCertificationFailure,
    DomainFailureReason,
    certify_three_site_domain,
)
from maple.function.calculator.extra_correction.implicit.torch_continuum_ses_geometry import (
    PROBE_ANGSTROM,
    triple_probe_centers,
)


def _tensor(rows):
    return torch.tensor(rows, dtype=torch.float64)


def _rotate_translate(positions):
    angle = 0.731
    rotation = _tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return positions @ rotation.T + _tensor([[1.7, -2.2, 0.4]])


def test_certifies_nonvacuous_single_full_probe_circle_and_is_rigid_invariant():
    positions = _tensor([[0.0, 0.0, 0.0], [4.2, 0.0, 0.0], [0.0, 10.0, 0.0]])
    radii = _tensor([1.5, 1.5, 1.5])

    first = certify_three_site_domain(positions, radii)
    moved = certify_three_site_domain(_rotate_translate(positions), radii)

    assert isinstance(first, CertifiedLocalPatchScope)
    assert first.active_pairs == ((0, 1),)
    assert first.fully_occluded_pairs == ()
    assert first.scope == "three-site-r6-geometric-point-only"
    assert isinstance(moved, CertifiedLocalPatchScope)
    assert moved.active_pairs == first.active_pairs
    assert dict(moved.raw_margins) == pytest.approx(dict(first.raw_margins), abs=2e-14)


def test_certificate_tracks_atom_permutation_without_changing_margins():
    positions = _tensor([[0.0, 0.0, 0.0], [4.2, 0.0, 0.0], [0.0, 10.0, 0.0]])
    radii = _tensor([1.5, 1.5, 1.5])
    permutation = torch.tensor([2, 0, 1])

    original = certify_three_site_domain(positions, radii)
    permuted = certify_three_site_domain(positions[permutation], radii[permutation])

    assert isinstance(original, CertifiedLocalPatchScope)
    assert isinstance(permuted, CertifiedLocalPatchScope)
    assert sorted(dict(original.raw_margins).values()) == pytest.approx(
        sorted(dict(permuted.raw_margins).values()), abs=2e-14
    )


@pytest.mark.parametrize(
    ("positions", "radii", "reason"),
    [
        (
            [[0.0, 0.0, 0.0], [4.2, 0.0, 0.0]],
            [1.5, 1.5],
            DomainFailureReason.UNSUPPORTED_SITE_COUNT,
        ),
        (
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
            [1.5, 1.5, 1.5],
            DomainFailureReason.COINCIDENT_CENTERS,
        ),
        (
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 3.45, 0.0]],
            [1.5, 1.5, 1.5],
            DomainFailureReason.EXPOSED_TRIPLE_PROBE,
        ),
        (
            [[0.0, 0.0, 0.0], [4.4, 0.0, 0.0], [2.2, 2.0, 0.0]],
            [1.5, 1.5, 1.5],
            # With exactly three balls, a partial pair circle terminates at
            # exposed triple contacts, so predicate 2 rejects it first.
            DomainFailureReason.EXPOSED_TRIPLE_PROBE,
        ),
        (
            [[0.0, 0.0, 0.0], [4.45, 0.0, 0.0], [0.0, 10.0, 0.0]],
            [1.5, 1.5, 1.5],
            DomainFailureReason.PINCHED_TORUS,
        ),
        (
            [[0.0, 0.0, 0.0], [4.2, 0.0, 0.0], [2.1, 3.7, 0.0]],
            [1.5, 1.5, 1.5],
            DomainFailureReason.PROBE_CIRCLE_SEPARATION,
        ),
        (
            [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [0.0, 9.0, 0.0]],
            [1.5, 1.5, 1.5],
            DomainFailureReason.NO_ACTIVE_PROBE_CIRCLE,
        ),
    ],
)
def test_each_unsupported_point_returns_a_typed_failure(positions, radii, reason):
    result = certify_three_site_domain(_tensor(positions), _tensor(radii))

    assert isinstance(result, DomainCertificationFailure)
    assert result.reason is reason
    assert result.raw_margins


def test_margin_boundary_is_rejected_not_rounded_into_domain():
    radius = 1.5
    probe = 0.88
    distance = 2.0 * math.sqrt((radius + probe) ** 2 - probe**2)
    result = certify_three_site_domain(
        _tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0], [0.0, 10.0, 0.0]]),
        _tensor([radius, radius, radius]),
    )
    assert isinstance(result, DomainCertificationFailure)
    assert result.reason is DomainFailureReason.PINCHED_TORUS


def test_synthetic_overlapping_probe_geometry_fails_closed():
    result = certify_three_site_domain(
        _tensor([[0.0, 0.0, 0.0], [4.2, 0.0, 0.0], [0.0, 4.2, 0.0]]),
        _tensor([1.5, 1.5, 1.5]),
    )
    assert isinstance(result, DomainCertificationFailure)
    assert result.reason in {
        DomainFailureReason.EXPOSED_TRIPLE_PROBE,
        DomainFailureReason.PROBE_CIRCLE_SEPARATION,
    }


def test_source_bound_methanol_has_overlapping_exposed_probe_centers_and_is_unsupported():
    source = (
        Path(__file__).parent / "data/amber_gb_reference/audit/methanol/normalized.mol2"
    )
    payload = source.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "184e1f418e625b11eff55c6dd9eaafcf01d8efd1738fed3c01fcc90964a4c852"
    )
    lines = payload.decode().splitlines()
    start = lines.index("@<TRIPOS>ATOM") + 1
    stop = lines.index("@<TRIPOS>BOND")
    atoms = {
        fields[1]: [float(value) for value in fields[2:5]]
        for fields in (line.split() for line in lines[start:stop] if line.strip())
    }
    all_positions = _tensor(
        [
            atoms[f"{element}{index}"]
            for element, index in (
                ("C", 1),
                ("O", 1),
                ("H", 1),
                ("H", 2),
                ("H", 3),
                ("H", 4),
            )
        ]
    )
    all_radii = _tensor([2.12, 1.88, 1.04, 1.04, 1.04, 1.04])
    # Evaluate the C1/O1/OH-H4 triple in the complete six-atom environment so
    # local exposure is checked against all three remaining methanol atoms.
    geometry = triple_probe_centers(all_positions, all_radii, (0, 1, 5))
    separation = torch.linalg.vector_norm(geometry.centers[1] - geometry.centers[0])
    assert geometry.locally_exposed.tolist() == [True, True]
    assert float(separation) == pytest.approx(1.1381980149, abs=5.0e-10)
    assert float(separation) < 2.0 * PROBE_ANGSTROM

    result = certify_three_site_domain(all_positions, all_radii)
    assert isinstance(result, DomainCertificationFailure)
    assert result.reason is DomainFailureReason.UNSUPPORTED_SITE_COUNT
