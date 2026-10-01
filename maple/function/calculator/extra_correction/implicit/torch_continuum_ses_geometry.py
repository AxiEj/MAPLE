"""Local rolling-probe intersection primitives, not a complete molecular SES.

Pair circles and three-sphere probe centers retain Torch coordinate graphs.
Local occlusion by other expanded atomic balls is checked, but connectivity to
the unbounded solvent region, torus/probe-patch trimming, and surface ownership
are deliberately NOT inferred from these primitives. Do not integrate them as
a whole-molecule R6 boundary without those missing geometric contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from .torch_chagb import EFFECTIVE_PROBE_ANGSTROM as PROBE_ANGSTROM

if TYPE_CHECKING:
    from torch import Tensor


_DEGENERACY_MARGIN_ANGSTROM = 1.0e-9
# Reject genuinely pinched R6 meridians; this is not a quadrature tolerance.
R6_GEOMETRY_EVENT_MARGIN_ANGSTROM = 1.0e-8


@dataclass(frozen=True)
class PairProbeCircle:
    center: Tensor
    radius: Tensor
    axis: Tensor
    complete_ses_patch: bool = False


@dataclass(frozen=True)
class PairProbeArcs:
    """Locally unoccluded center-circle arcs; no exterior-connectivity claim."""

    circle: PairProbeCircle
    basis_u: Tensor
    basis_v: Tensor
    intervals: Tensor
    pair_indices: tuple[int, int]
    exterior_accessibility_verified: bool = False


@dataclass(frozen=True)
class TripleProbeCenters:
    centers: Tensor
    locally_exposed: Tensor
    atom_indices: tuple[int, int, int]
    exterior_accessibility_verified: bool = False


def _validate_inputs(positions, intrinsic_radii, minimum_count: int) -> None:
    import torch

    if not isinstance(positions, torch.Tensor) or not isinstance(
        intrinsic_radii, torch.Tensor
    ):
        raise TypeError("SES coordinates and intrinsic radii must be Torch tensors.")
    if (
        positions.dtype != torch.float64
        or intrinsic_radii.dtype != torch.float64
        or positions.ndim != 2
        or positions.shape[1] != 3
        or len(positions) < minimum_count
        or intrinsic_radii.shape != (len(positions),)
        or intrinsic_radii.device != positions.device
    ):
        raise TypeError("SES inputs require same-device float64 [N,3] and [N] tensors.")
    if not bool(
        torch.isfinite(positions).all() and torch.isfinite(intrinsic_radii).all()
    ):
        raise ValueError("SES coordinates and radii must be finite.")
    if not bool((intrinsic_radii > 0.0).all()):
        raise ValueError("SES intrinsic radii must be positive.")


def pair_probe_contact_circle(
    positions_angstrom: Tensor, intrinsic_radii_angstrom: Tensor
) -> PairProbeCircle:
    """Intersection circle of two probe-expanded atomic spheres, if regular."""
    import torch

    _validate_inputs(positions_angstrom, intrinsic_radii_angstrom, 2)
    if len(positions_angstrom) != 2:
        raise ValueError("Pair probe circle accepts exactly two atoms.")
    displacement = positions_angstrom[1] - positions_angstrom[0]
    distance = torch.linalg.vector_norm(displacement)
    if bool(distance <= _DEGENERACY_MARGIN_ANGSTROM):
        raise ValueError("Coincident pair centers have no unique probe circle.")
    radii = intrinsic_radii_angstrom + PROBE_ANGSTROM
    if bool(
        distance >= radii.sum() - _DEGENERACY_MARGIN_ANGSTROM
        or distance <= torch.abs(radii[0] - radii[1]) + _DEGENERACY_MARGIN_ANGSTROM
    ):
        raise ValueError("No regular pair probe-circle intersection exists.")
    axis = displacement / distance
    along = (radii[0].square() - radii[1].square() + distance.square()) / (
        2.0 * distance
    )
    radius = torch.sqrt(radii[0].square() - along.square())
    center = positions_angstrom[0] + along * axis
    return PairProbeCircle(center=center, radius=radius, axis=axis)


def _add_covered_arc(covered, phase, width, zero, two_pi) -> None:
    start, stop = phase - width, phase + width
    if bool(start < 0.0):
        covered.extend(((zero, stop), (start + two_pi, two_pi)))
    elif bool(stop > two_pi):
        covered.extend(((start, two_pi), (zero, stop - two_pi)))
    else:
        covered.append((start, stop))


def _exposed_after_covered_arcs(covered, template):
    """Exact complement of circular covered arcs, retaining endpoint graphs."""
    import torch

    zero = template.new_tensor(0.0)
    two_pi = template.new_tensor(2.0 * math.pi)
    covered.sort(key=lambda arc: float(arc[0].detach()))
    merged: list[tuple[Tensor, Tensor]] = []
    for start, stop in covered:
        if not merged or bool(start > merged[-1][1]):
            merged.append((start, stop))
        else:
            previous_start, previous_stop = merged[-1]
            merged[-1] = (previous_start, torch.maximum(previous_stop, stop))
    exposed: list[tuple[Tensor, Tensor]] = []
    cursor = zero
    for start, stop in merged:
        if bool(start > cursor):
            exposed.append((cursor, start))
        cursor = torch.maximum(cursor, stop)
    if bool(cursor < two_pi):
        exposed.append((cursor, two_pi))
    return (
        torch.stack([torch.stack(arc) for arc in exposed])
        if exposed
        else template.new_empty((0, 2))
    )


def pair_probe_exposed_arcs(
    positions_angstrom: Tensor,
    intrinsic_radii_angstrom: Tensor,
    pair_indices: tuple[int, int],
) -> PairProbeArcs:
    """Clip one pair's probe-center circle against all other expanded balls.

    The resulting arcs are only locally unoccluded. They do not prove
    connectivity to exterior solvent or resolve probe-probe self-overlaps.
    """
    import torch

    _validate_inputs(positions_angstrom, intrinsic_radii_angstrom, 2)
    if (
        not isinstance(pair_indices, tuple)
        or len(pair_indices) != 2
        or any(
            type(index) is not int or not 0 <= index < len(positions_angstrom)
            for index in pair_indices
        )
        or pair_indices[0] == pair_indices[1]
    ):
        raise ValueError("Pair probe indices must name two distinct atoms.")
    circle = pair_probe_contact_circle(
        positions_angstrom[list(pair_indices)],
        intrinsic_radii_angstrom[list(pair_indices)],
    )
    axis = circle.axis
    least_parallel = int(torch.argmin(axis.detach().abs()).item())
    coordinate_axis = torch.eye(3, dtype=axis.dtype, device=axis.device)[least_parallel]
    basis_u = torch.linalg.cross(axis, coordinate_axis)
    basis_u = basis_u / torch.linalg.vector_norm(basis_u)
    basis_v = torch.linalg.cross(axis, basis_u)
    two_pi = positions_angstrom.new_tensor(2.0 * math.pi)
    zero = positions_angstrom.new_tensor(0.0)
    expanded = intrinsic_radii_angstrom + PROBE_ANGSTROM
    covered: list[tuple[Tensor, Tensor]] = []
    for other in range(len(positions_angstrom)):
        if other in pair_indices:
            continue
        displacement = positions_angstrom[other] - circle.center
        transverse = displacement - torch.dot(displacement, axis) * axis
        transverse_length = torch.linalg.vector_norm(transverse)
        amplitude = 2.0 * circle.radius * transverse_length
        constant = torch.dot(displacement, displacement) + circle.radius.square()
        other_squared = expanded[other].square()
        if bool(amplitude <= _DEGENERACY_MARGIN_ANGSTROM**2):
            margin = constant - other_squared
            if bool(torch.abs(margin) <= _DEGENERACY_MARGIN_ANGSTROM**2):
                raise ValueError("Whole-circle probe contact is a topology event.")
            if bool(margin < 0.0):
                return PairProbeArcs(
                    circle,
                    basis_u,
                    basis_v,
                    positions_angstrom.new_empty((0, 2)),
                    pair_indices,
                )
            continue
        threshold = (constant - other_squared) / amplitude
        if bool(threshold >= 1.0 + _DEGENERACY_MARGIN_ANGSTROM):
            continue
        if bool(threshold <= -1.0 - _DEGENERACY_MARGIN_ANGSTROM):
            return PairProbeArcs(
                circle,
                basis_u,
                basis_v,
                positions_angstrom.new_empty((0, 2)),
                pair_indices,
            )
        if bool(torch.abs(torch.abs(threshold) - 1.0) <= _DEGENERACY_MARGIN_ANGSTROM):
            raise ValueError(
                "Tangential third-sphere probe contact is a topology event."
            )
        phase = torch.remainder(
            torch.atan2(torch.dot(transverse, basis_v), torch.dot(transverse, basis_u)),
            two_pi,
        )
        width = torch.acos(threshold)
        _add_covered_arc(covered, phase, width, zero, two_pi)

    intervals = _exposed_after_covered_arcs(covered, positions_angstrom)
    return PairProbeArcs(circle, basis_u, basis_v, intervals, pair_indices)


def contact_exposed_azimuth(
    positions_angstrom: Tensor,
    intrinsic_radii_angstrom: Tensor,
    *,
    owner: int,
    polar_cosine: Tensor,
) -> Tensor:
    """Exposed azimuth arcs on one atomic contact sphere at fixed polar cosine.

    This is one exact circle slice of a local contact patch. It is not the
    integrated contact surface or a proof of global solvent accessibility.
    """
    import torch

    _validate_inputs(positions_angstrom, intrinsic_radii_angstrom, 1)
    if type(owner) is not int or not 0 <= owner < len(positions_angstrom):
        raise ValueError("Contact owner must be a valid atom index.")
    if (
        not isinstance(polar_cosine, torch.Tensor)
        or polar_cosine.dtype != torch.float64
        or polar_cosine.device != positions_angstrom.device
        or polar_cosine.shape != ()
        or not bool(torch.isfinite(polar_cosine))
        or not bool(torch.abs(polar_cosine) < 1.0)
    ):
        raise ValueError(
            "Contact polar cosine must be a finite interior float64 scalar."
        )
    expanded = intrinsic_radii_angstrom + PROBE_ANGSTROM
    horizontal = torch.sqrt(1.0 - polar_cosine.square())
    zero = positions_angstrom.new_tensor(0.0)
    two_pi = positions_angstrom.new_tensor(2.0 * math.pi)
    covered: list[tuple[Tensor, Tensor]] = []
    for other in range(len(positions_angstrom)):
        if other == owner:
            continue
        displacement = positions_angstrom[other] - positions_angstrom[owner]
        distance = torch.linalg.vector_norm(displacement)
        if bool(distance <= _DEGENERACY_MARGIN_ANGSTROM):
            raise ValueError(
                "Coincident contact atoms have ambiguous surface ownership."
            )
        direction = displacement / distance
        threshold_cosine = (
            distance.square() + expanded[owner].square() - expanded[other].square()
        ) / (2.0 * expanded[owner] * distance)
        transverse_squared = direction[:2].square().sum()
        if bool(transverse_squared <= _DEGENERACY_MARGIN_ANGSTROM**2):
            margin = polar_cosine * direction[2] - threshold_cosine
            if bool(margin > 0.0):
                return positions_angstrom.new_empty((0, 2))
            continue
        amplitude = horizontal * torch.sqrt(transverse_squared)
        threshold = (threshold_cosine - polar_cosine * direction[2]) / amplitude
        # An arc appearing at one latitude is an expected integration-chart
        # endpoint, not a change of molecular SES topology. The caller splits
        # the latitude domain at these endpoints; do not reject near them.
        if bool(threshold >= 1.0):
            continue
        if bool(threshold <= -1.0):
            return positions_angstrom.new_empty((0, 2))
        phase = torch.remainder(torch.atan2(direction[1], direction[0]), two_pi)
        _add_covered_arc(covered, phase, torch.acos(threshold), zero, two_pi)
    return _exposed_after_covered_arcs(covered, positions_angstrom)


def triple_probe_centers(
    positions_angstrom: Tensor,
    intrinsic_radii_angstrom: Tensor,
    atom_indices: tuple[int, int, int],
) -> TripleProbeCenters:
    """Two local three-contact probe centers, excluding strict burial.

    An empty result means the three expanded spheres do not meet. It does not
    certify solvent accessibility of a candidate in a buried internal cavity.
    """
    import torch

    _validate_inputs(positions_angstrom, intrinsic_radii_angstrom, 3)
    if (
        not isinstance(atom_indices, tuple)
        or len(atom_indices) != 3
        or any(
            type(index) is not int or not 0 <= index < len(positions_angstrom)
            for index in atom_indices
        )
        or len(set(atom_indices)) != 3
    ):
        raise ValueError("Triple probe indices must be three distinct atom indices.")
    first, second, third = atom_indices
    positions = positions_angstrom
    expanded = intrinsic_radii_angstrom + PROBE_ANGSTROM
    ab = positions[second] - positions[first]
    ac = positions[third] - positions[first]
    ab_distance = torch.linalg.vector_norm(ab)
    if bool(ab_distance <= _DEGENERACY_MARGIN_ANGSTROM):
        raise ValueError("Degenerate triple contains coincident centers.")
    axis = ab / ab_distance
    projection = torch.dot(ac, axis)
    perpendicular = ac - projection * axis
    perpendicular_length = torch.linalg.vector_norm(perpendicular)
    if bool(perpendicular_length <= _DEGENERACY_MARGIN_ANGSTROM):
        raise ValueError("Degenerate or collinear triple has no unique probe centers.")
    second_axis = perpendicular / perpendicular_length
    third_axis = torch.linalg.cross(axis, second_axis)
    x = (
        expanded[first].square() - expanded[second].square() + ab_distance.square()
    ) / (2.0 * ab_distance)
    y = (
        expanded[first].square()
        - expanded[third].square()
        + torch.dot(ac, ac)
        - 2.0 * projection * x
    ) / (2.0 * perpendicular_length)
    height_squared = expanded[first].square() - x.square() - y.square()
    if bool(height_squared < -(_DEGENERACY_MARGIN_ANGSTROM**2)):
        return TripleProbeCenters(
            centers=positions.new_empty((0, 3)),
            locally_exposed=torch.empty(
                (0,), dtype=torch.bool, device=positions.device
            ),
            atom_indices=atom_indices,
        )
    if bool(height_squared <= _DEGENERACY_MARGIN_ANGSTROM**2):
        raise ValueError("Tangent triple probe center is a topology event.")
    base = positions[first] + x * axis + y * second_axis
    height = torch.sqrt(height_squared)
    candidates = torch.stack((base - height * third_axis, base + height * third_axis))
    other_indices = [
        index for index in range(len(positions)) if index not in atom_indices
    ]
    if other_indices:
        separation = torch.linalg.vector_norm(
            candidates[:, None, :] - positions[None, other_indices, :], dim=-1
        )
        margins = separation - expanded[None, other_indices]
        if bool((torch.abs(margins) <= _DEGENERACY_MARGIN_ANGSTROM).any()):
            raise ValueError("A four-way probe contact is a topology event.")
        local_exposure = (margins > 0.0).all(dim=1)
    else:
        local_exposure = torch.ones(2, dtype=torch.bool, device=positions.device)
    return TripleProbeCenters(
        centers=candidates,
        locally_exposed=local_exposure,
        atom_indices=atom_indices,
    )
