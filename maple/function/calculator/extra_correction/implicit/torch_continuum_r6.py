"""Unregistered continuous R6 SES reference for one or two atomic spheres.

The two-site geometry is axially symmetric, reducing each exposed atomic cap
and rolling-probe torus to a one-dimensional Torch quadrature. Spindle tori
are clipped at the symmetry axis rather than integrated through their
self-intersection. This is not a multiatom SES constructor or a solvent
provider; three-or-more-site inputs fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .torch_continuum_ses_geometry import (
    PROBE_ANGSTROM,
    R6_GEOMETRY_EVENT_MARGIN_ANGSTROM as _GEOMETRY_EVENT_MARGIN_ANGSTROM,
)
from .torch_sphere_union_geometry import legendre_rule

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class R6LimitedResult:
    """Graph-connected inverse Born radii for a strictly limited SES domain."""

    inverse_born_per_angstrom: Tensor
    regime: str
    quadrature_order: int
    complete_multiatom_ses: bool = False


def _validate(positions, radii, order: int) -> None:
    import torch

    if not isinstance(positions, torch.Tensor) or not isinstance(radii, torch.Tensor):
        raise TypeError("R6 positions and radii must be Torch tensors.")
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError("R6 positions must have nonempty shape (N, 3).")
    if len(positions) > 2:
        raise NotImplementedError(
            "Complete multiatom SES contact/torus/probe-patch geometry is not "
            "implemented; no R6 force or provider may use this limited kernel."
        )
    if (
        positions.dtype != torch.float64
        or radii.dtype != torch.float64
        or radii.shape != (len(positions),)
        or radii.device != positions.device
    ):
        raise TypeError("R6 inputs must be same-device float64 tensors [N,3] and [N].")
    if not bool(torch.isfinite(positions).all() and torch.isfinite(radii).all()):
        raise ValueError("R6 geometry must be finite.")
    if not bool((radii > 0.0).all()):
        raise ValueError("R6 atomic radii must be positive.")
    if isinstance(order, bool) or not isinstance(order, int) or order < 8:
        raise ValueError("R6 Gauss order must be an integer at least 8.")


def inverse_born_from_ses(
    positions_angstrom: Tensor,
    intrinsic_radii_angstrom: Tensor,
    *,
    order: int = 64,
) -> R6LimitedResult:
    """Evaluate the published R6 surface flux on a complete 1–2 site SES.

    The rolling probe is frozen at 0.88 A for this CHA profile. Returned
    inverse radii precede the existing CHA electrostatic-size shift. No grid
    smoothing, OBC radius, SAS union, fitted correction, or Amber executable
    enters this mathematical reference slice.
    """
    import torch

    positions = positions_angstrom
    radii = intrinsic_radii_angstrom
    _validate(positions, radii, order)
    if len(radii) == 1:
        return R6LimitedResult(radii.reciprocal(), "isolated", order)

    distance = torch.linalg.vector_norm(positions[1] - positions[0])
    if bool(distance <= _GEOMETRY_EVENT_MARGIN_ANGSTROM):
        raise ValueError("Coincident atom centers are outside the two-site R6 domain.")
    expanded = radii + PROBE_ANGSTROM
    first, second = radii.unbind()
    expanded_first, expanded_second = expanded.unbind()
    outer_contact = expanded_first + expanded_second
    inner_contact = torch.abs(expanded_first - expanded_second)
    if bool(
        torch.abs(distance - outer_contact) <= _GEOMETRY_EVENT_MARGIN_ANGSTROM
    ) or bool(torch.abs(distance - inner_contact) <= _GEOMETRY_EVENT_MARGIN_ANGSTROM):
        raise ValueError(
            "Rolling-probe sphere tangency is outside the smooth R6 branch."
        )

    nodes, weights = legendre_rule(order, str(positions.device))
    inverse_cube = torch.zeros(2, dtype=torch.float64, device=positions.device)

    def contact_flux(source: int, upper: Tensor) -> Tensor:
        radius = radii[source]
        target_distance = torch.stack(
            (torch.zeros_like(distance), distance)
            if source == 0
            else (distance, torch.zeros_like(distance))
        )
        midpoint = 0.5 * (upper - 1.0)
        half_width = 0.5 * (upper + 1.0)
        cosine = midpoint + half_width * nodes
        separation_squared = (
            radius.square()
            + target_distance[None, :].square()
            - 2.0 * radius * target_distance[None, :] * cosine[:, None]
        )
        if bool((separation_squared <= 0.0).any()):
            raise ValueError("R6 contact integrand reached an atomic center.")
        kernel = (
            radius - target_distance[None, :] * cosine[:, None]
        ) / separation_squared.pow(3)
        return (
            0.5 * radius.square() * half_width * (weights[:, None] * kernel).sum(dim=0)
        )

    if bool(distance > outer_contact):
        inverse_cube = contact_flux(0, torch.ones_like(distance))
        inverse_cube = inverse_cube + contact_flux(1, torch.ones_like(distance))
        regime = "disjoint"
    elif bool(distance < inner_contact):
        outer = 0 if bool(expanded_first > expanded_second) else 1
        inverse_cube = contact_flux(outer, torch.ones_like(distance))
        regime = "contained"
    else:
        along = (
            expanded_first.square() - expanded_second.square() + distance.square()
        ) / (2.0 * distance)
        circle_radius = torch.sqrt(expanded_first.square() - along.square())
        inverse_cube = contact_flux(0, along / expanded_first)
        inverse_cube = inverse_cube + contact_flux(
            1, (distance - along) / expanded_second
        )

        first_angle = torch.atan2(-along, circle_radius)
        second_angle = torch.atan2(distance - along, circle_radius)
        if bool(
            torch.abs(circle_radius - PROBE_ANGSTROM) <= _GEOMETRY_EVENT_MARGIN_ANGSTROM
        ):
            raise ValueError(
                "A pinching spindle torus is outside the smooth R6 branch."
            )
        if bool(circle_radius > PROBE_ANGSTROM):
            intervals = ((first_angle, second_angle),)
            regime = "regular-torus"
        else:
            cut = torch.acos(circle_radius / PROBE_ANGSTROM)
            intervals = ((first_angle, -cut), (cut, second_angle))
            if any(bool(stop <= start) for start, stop in intervals):
                raise ValueError(
                    "Spindle torus has no ordered exposed meridional arcs."
                )
            regime = "spindle-torus"

        target_axis = torch.stack((torch.zeros_like(distance), distance))
        for start, stop in intervals:
            half_width = 0.5 * (stop - start)
            angle = 0.5 * (start + stop) + half_width * nodes
            axial = along + PROBE_ANGSTROM * torch.sin(angle)
            radial = circle_radius - PROBE_ANGSTROM * torch.cos(angle)
            if bool((radial <= 0.0).any()):
                raise ValueError("Self-intersecting torus area was not fully clipped.")
            delta_axis = axial[:, None] - target_axis[None, :]
            separation_squared = delta_axis.square() + radial[:, None].square()
            normal_dot = (
                -delta_axis * torch.sin(angle)[:, None]
                + radial[:, None] * torch.cos(angle)[:, None]
            )
            kernel = radial[:, None] * normal_dot / separation_squared.pow(3)
            inverse_cube = inverse_cube + (
                0.5
                * PROBE_ANGSTROM
                * half_width
                * (weights[:, None] * kernel).sum(dim=0)
            )

    if not bool(torch.isfinite(inverse_cube).all() and (inverse_cube > 0.0).all()):
        raise RuntimeError(
            "R6 surface flux did not yield positive finite inverse radii."
        )
    return R6LimitedResult(inverse_cube.pow(1.0 / 3.0), regime, order)
