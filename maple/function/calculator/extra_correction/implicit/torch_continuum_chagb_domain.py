"""Conservative point-domain certificate for a three-site rolling-probe SES.

This module proves only R6 rolling-probe geometry facts at one geometry.  It
does not inspect CHA charges or electrostatic-size branches and deliberately
makes no claim about a finite-difference stencil, line segment, or optimizer
step.

Why an internal solvent void is impossible under the supported hypotheses
--------------------------------------------------------------------------
The centers of at most three *closed, finite, positive-radius* balls lie in an
affine plane ``P``.  From any point outside every ball, choose the normal ray
whose signed distance moves away from ``P`` (either normal if the point is in
``P``).  Because every center lies in ``P``, squared distance to every center
is nondecreasing along that ray and tends to infinity.  The ray therefore
never enters a ball and reaches the unbounded exterior.  At sufficiently
large radius the two sides connect around the bounded union, so every exterior
point belongs to that same component.  Thus no isolated 3-D complement void
exists.  The argument does not apply to periodic images, non-ball primitives,
zero/infinite radii, or four or more non-coplanar centers; all are outside this
certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING, ClassVar, TypeAlias

from .torch_continuum_ses_geometry import (
    PROBE_ANGSTROM,
    pair_probe_exposed_arcs,
    triple_probe_centers,
)

if TYPE_CHECKING:
    from torch import Tensor


Pair = tuple[int, int]
RawMargins = tuple[tuple[str, float], ...]


class DomainFailureReason(str, Enum):
    INVALID_INPUT = "invalid-input"
    UNSUPPORTED_SITE_COUNT = "unsupported-site-count"
    COINCIDENT_CENTERS = "coincident-centers"
    DEGENERATE_PAIR_INTERSECTION = "degenerate-pair-intersection"
    EXPOSED_TRIPLE_PROBE = "exposed-triple-probe"
    PARTIAL_PROBE_CIRCLE = "partial-probe-circle"
    PINCHED_TORUS = "pinched-torus"
    PROBE_CIRCLE_SEPARATION = "probe-circle-separation"
    NO_ACTIVE_PROBE_CIRCLE = "no-active-probe-circle"
    GEOMETRY_PRIMITIVE_REJECTED = "geometry-primitive-rejected"


@dataclass(frozen=True)
class CertifiedLocalPatchScope:
    """Non-vacuous R6 geometry certificate for one three-site point."""

    active_pairs: tuple[Pair, ...]
    fully_occluded_pairs: tuple[Pair, ...]
    raw_margins: RawMargins
    scope: ClassVar[str] = "three-site-r6-geometric-point-only"
    complement_void_proof: ClassVar[str] = "coplanar-centers-normal-ray"
    segment_certified: ClassVar[bool] = False


@dataclass(frozen=True)
class DomainCertificationFailure:
    """Typed fail-closed result retaining diagnostics computed before rejection."""

    reason: DomainFailureReason
    message: str
    raw_margins: RawMargins
    active_pairs: tuple[Pair, ...] = ()


ThreeSiteDomainResult: TypeAlias = CertifiedLocalPatchScope | DomainCertificationFailure


def _failure(reason, message, margins, active=()):
    return DomainCertificationFailure(reason, message, tuple(margins), tuple(active))


def certify_three_site_domain(
    positions_angstrom: Tensor,
    intrinsic_radii_angstrom: Tensor,
    *,
    margin_angstrom: float = 1.0e-8,
) -> ThreeSiteDomainResult:
    """Certify one conservative, complete-patch three-site R6 geometry domain.

    A pair circle is ``active`` only when it is regular and wholly exposed.
    Regular but wholly occluded circles are recorded and omitted from the SES.
    Partial arcs, pinched tori, overlapping probe tubes, and vacuous geometries
    fail closed.  Detached scalar comparisons select the local geometric
    branch, while production quadrature retains the coordinate graph.  CHA
    charge-sign and size-switch diagnostics belong to the scalar assembly.
    """
    import torch

    margins: list[tuple[str, float]] = []
    if (
        not isinstance(margin_angstrom, (int, float))
        or isinstance(margin_angstrom, bool)
        or not math.isfinite(margin_angstrom)
        or margin_angstrom <= 0.0
    ):
        return _failure(
            DomainFailureReason.INVALID_INPUT,
            "margin_angstrom must be a finite positive real number.",
            (
                (
                    "margin_angstrom",
                    (
                        float(margin_angstrom)
                        if isinstance(margin_angstrom, (int, float))
                        else math.nan
                    ),
                ),
            ),
        )
    if not isinstance(positions_angstrom, torch.Tensor) or not isinstance(
        intrinsic_radii_angstrom, torch.Tensor
    ):
        return _failure(
            DomainFailureReason.INVALID_INPUT,
            "Coordinates and radii must be Torch tensors.",
            (("tensor_inputs", -1.0),),
        )
    site_count = int(positions_angstrom.shape[0]) if positions_angstrom.ndim else 0
    margins.append(("site_count_minus_three", float(site_count - 3)))
    if site_count != 3:
        return _failure(
            DomainFailureReason.UNSUPPORTED_SITE_COUNT,
            "The point certificate is restricted to exactly three sites.",
            margins,
        )
    if (
        positions_angstrom.dtype != torch.float64
        or intrinsic_radii_angstrom.dtype != torch.float64
        or positions_angstrom.shape != (3, 3)
        or intrinsic_radii_angstrom.shape != (3,)
        or positions_angstrom.device != intrinsic_radii_angstrom.device
        or not bool(torch.isfinite(positions_angstrom).all())
        or not bool(torch.isfinite(intrinsic_radii_angstrom).all())
        or not bool((intrinsic_radii_angstrom > 0.0).all())
    ):
        return _failure(
            DomainFailureReason.INVALID_INPUT,
            "Three-site inputs require same-device finite float64 [3,3] and [3].",
            margins,
        )

    expanded = intrinsic_radii_angstrom + PROBE_ANGSTROM
    regular_pairs: list[Pair] = []
    circles = {}
    for first, second in ((0, 1), (0, 2), (1, 2)):
        distance = torch.linalg.vector_norm(
            positions_angstrom[second] - positions_angstrom[first]
        )
        distance_value = float(distance.detach())
        margins.append((f"pair_{first}{second}_center_distance", distance_value))
        if distance_value <= margin_angstrom:
            return _failure(
                DomainFailureReason.COINCIDENT_CENTERS,
                "Distinct sites must have distinct centers.",
                margins,
            )
        outer = float((expanded[first] + expanded[second] - distance).detach())
        inner = float(
            (distance - torch.abs(expanded[first] - expanded[second])).detach()
        )
        margins.extend(
            (
                (f"pair_{first}{second}_outer_intersection", outer),
                (f"pair_{first}{second}_inner_intersection", inner),
            )
        )
        if abs(outer) <= margin_angstrom or abs(inner) <= margin_angstrom:
            return _failure(
                DomainFailureReason.DEGENERATE_PAIR_INTERSECTION,
                "A probe-expanded sphere tangency is not a regular branch.",
                margins,
            )
        if outer > margin_angstrom and inner > margin_angstrom:
            regular_pairs.append((first, second))

    if len(regular_pairs) == 3:
        try:
            triple = triple_probe_centers(
                positions_angstrom, intrinsic_radii_angstrom, (0, 1, 2)
            )
        except ValueError as exc:
            return _failure(
                DomainFailureReason.GEOMETRY_PRIMITIVE_REJECTED,
                str(exc),
                margins,
            )
        exposed_count = int(triple.locally_exposed.sum().item())
        margins.append(("exposed_triple_probe_count", float(exposed_count)))
        if exposed_count:
            return _failure(
                DomainFailureReason.EXPOSED_TRIPLE_PROBE,
                "Exposed triple-contact probe patches are outside this domain.",
                margins,
            )
    else:
        margins.append(("exposed_triple_probe_count", 0.0))

    active: list[Pair] = []
    occluded: list[Pair] = []
    two_pi = 2.0 * math.pi
    for pair in regular_pairs:
        try:
            arcs = pair_probe_exposed_arcs(
                positions_angstrom, intrinsic_radii_angstrom, pair
            )
        except ValueError as exc:
            return _failure(
                DomainFailureReason.GEOMETRY_PRIMITIVE_REJECTED,
                str(exc),
                margins,
                active,
            )
        intervals = arcs.intervals
        exposed_angle = float((intervals[:, 1] - intervals[:, 0]).sum().detach())
        margins.append((f"pair_{pair[0]}{pair[1]}_exposed_angle", exposed_angle))
        if len(intervals) == 0:
            occluded.append(pair)
            continue
        if not (
            len(intervals) == 1
            and abs(float(intervals[0, 0].detach())) <= 1.0e-12
            and abs(float(intervals[0, 1].detach()) - two_pi) <= 1.0e-12
        ):
            return _failure(
                DomainFailureReason.PARTIAL_PROBE_CIRCLE,
                "A partially exposed probe circle needs explicit arc ownership.",
                margins,
                active,
            )
        torus_margin = (
            float((arcs.circle.radius - PROBE_ANGSTROM).detach()) - margin_angstrom
        )
        margins.append((f"pair_{pair[0]}{pair[1]}_torus_radius", torus_margin))
        if torus_margin <= 0.0:
            return _failure(
                DomainFailureReason.PINCHED_TORUS,
                "An active probe circle is not wider than the probe radius.",
                margins,
                active,
            )
        active.append(pair)
        circles[pair] = arcs.circle

    if not active:
        return _failure(
            DomainFailureReason.NO_ACTIVE_PROBE_CIRCLE,
            "A certificate with no exposed rolling-probe circle is disallowed.",
            margins,
        )

    for index, first_pair in enumerate(active):
        for second_pair in active[index + 1 :]:
            first_circle = circles[first_pair]
            second_circle = circles[second_pair]
            displacement = second_circle.center - first_circle.center
            distance = torch.linalg.vector_norm(displacement)
            if float(distance.detach()) <= margin_angstrom:
                separation = -math.inf
            else:
                direction = displacement / distance
                first_span = first_circle.radius * torch.sqrt(
                    torch.clamp_min(
                        1.0 - torch.dot(first_circle.axis, direction).square(), 0.0
                    )
                )
                second_span = second_circle.radius * torch.sqrt(
                    torch.clamp_min(
                        1.0 - torch.dot(second_circle.axis, direction).square(), 0.0
                    )
                )
                separation = (
                    float(
                        (
                            distance - first_span - second_span - 2.0 * PROBE_ANGSTROM
                        ).detach()
                    )
                    - margin_angstrom
                )
            label = (
                f"circle_{first_pair[0]}{first_pair[1]}_"
                f"{second_pair[0]}{second_pair[1]}_separation"
            )
            margins.append((label, separation))
            if separation <= 0.0:
                return _failure(
                    DomainFailureReason.PROBE_CIRCLE_SEPARATION,
                    "Active probe tubes are not conservatively separated.",
                    margins,
                    active,
                )

    return CertifiedLocalPatchScope(tuple(active), tuple(occluded), tuple(margins))
