"""Unregistered Torch R6 flux contributions on locally exposed SES patches.

These contact, torus, and probe-patch integrals are not a complete molecular
surface. Global exterior connectivity, probe-probe clipping, and surface
ownership are not established; no provider may use their sum as Born radii.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from .torch_continuum_ses_geometry import (
    PROBE_ANGSTROM,
    R6_GEOMETRY_EVENT_MARGIN_ANGSTROM as _GEOMETRY_EVENT_MARGIN_ANGSTROM,
    _validate_inputs,
    contact_exposed_azimuth,
    pair_probe_exposed_arcs,
    triple_probe_centers,
)
from .torch_sphere_union_geometry import legendre_rule

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class ContactFluxResult:
    """Atomic-contact contribution to inverse-Born cubed, not a closed SES."""

    inverse_cube_per_angstrom3: Tensor
    polar_order: int
    azimuth_order: int
    complete_ses: bool = False


@dataclass(frozen=True)
class PairTorusFluxResult:
    """One pair's locally exposed reentrant flux, not a closed SES."""

    inverse_cube_per_angstrom3: Tensor
    pair_indices: tuple[int, int]
    regime: str
    exposed_arc_count: int
    complete_ses: bool = False


@dataclass(frozen=True)
class ProbePatchFluxResult:
    """Three-contact concave probe flux, before any multi-probe clipping."""

    inverse_cube_per_angstrom3: Tensor
    patch_areas_angstrom2: Tensor
    complete_ses: bool = False


def _contact_latitude_breakpoints(positions, intrinsic_radii, owner):
    """Spherical-cap and local triple-contact z coordinates on one atom."""
    import torch

    expanded = intrinsic_radii + PROBE_ANGSTROM
    points = [positions.new_tensor(-1.0), positions.new_tensor(1.0)]
    neighbors = []
    for other in range(len(positions)):
        if other == owner:
            continue
        delta = positions[other] - positions[owner]
        distance = torch.linalg.vector_norm(delta)
        if bool(distance <= _GEOMETRY_EVENT_MARGIN_ANGSTROM):
            raise ValueError("Coincident atoms have ambiguous R6 contact ownership.")
        if bool(
            distance >= expanded[owner] + expanded[other]
            or distance <= torch.abs(expanded[owner] - expanded[other])
        ):
            continue
        neighbors.append(other)
        direction = delta / distance
        along = (
            expanded[owner].square() - expanded[other].square() + distance.square()
        ) / (2.0 * distance)
        radius = torch.sqrt(expanded[owner].square() - along.square())
        z_span = radius * torch.sqrt(torch.clamp_min(1.0 - direction[2].square(), 0.0))
        center_z = along * direction[2]
        points.extend(
            (
                (center_z - z_span) / expanded[owner],
                (center_z + z_span) / expanded[owner],
            )
        )
    for index, second in enumerate(neighbors):
        for third in neighbors[index + 1 :]:
            try:
                triple = triple_probe_centers(
                    positions, intrinsic_radii, (owner, second, third)
                )
            except ValueError as exc:
                if "collinear" in str(exc):
                    continue
                raise
            for center, exposed in zip(triple.centers, triple.locally_exposed):
                if bool(exposed):
                    points.append((center[2] - positions[owner, 2]) / expanded[owner])
    ordered = sorted(points, key=lambda value: float(value.detach()))
    unique = [ordered[0]]
    for point in ordered[1:]:
        if float((point - unique[-1]).detach()) > 1e-12:
            unique.append(point)
    return unique


def _contact_patch_flux(
    positions_angstrom: Tensor,
    intrinsic_radii_angstrom: Tensor,
    *,
    mu_order: int,
    phi_order: int,
    squared_distance_power: float,
) -> Tensor:
    """Integrate all locally exposed atomic-contact pieces of a radial flux.

    This returns only one component of inverse Born cubed. Pair reentrant
    tori, concave probe patches, and exterior-connectivity classification
    remain missing, so these values must not enter a CHA provider directly.
    """
    import torch

    positions = positions_angstrom
    intrinsic = intrinsic_radii_angstrom
    _validate_inputs(positions, intrinsic, 1)
    if len(positions) > 12:
        raise RuntimeError(
            "Unregistered contact-patch quadrature is capped at 12 atoms."
        )
    for name, order in (("mu_order", mu_order), ("phi_order", phi_order)):
        if isinstance(order, bool) or not isinstance(order, int) or order < 8:
            raise ValueError(f"R6 contact {name} must be an integer at least 8.")
    mu_nodes, mu_weights = legendre_rule(mu_order, str(positions.device))
    phi_nodes, phi_weights = legendre_rule(phi_order, str(positions.device))
    flux = torch.zeros(len(positions), dtype=torch.float64, device=positions.device)
    for owner, radius in enumerate(intrinsic):
        breakpoints = _contact_latitude_breakpoints(positions, intrinsic, owner)
        for lower, upper in zip(breakpoints[:-1], breakpoints[1:]):
            half_mu = 0.5 * (upper - lower)
            # Exposed azimuth intervals are born with square-root width at a
            # contact-circle latitude extremum. The sine map has zero slope
            # at both segment ends, resolving that integrable endpoint cusp
            # without changing the underlying surface functional.
            chart_angle = 0.5 * math.pi * mu_nodes
            mu_values = 0.5 * (upper + lower) + half_mu * torch.sin(chart_angle)
            mu_jacobians = half_mu * 0.5 * math.pi * torch.cos(chart_angle)
            for mu, mu_weight, mu_jacobian in zip(mu_values, mu_weights, mu_jacobians):
                intervals = contact_exposed_azimuth(
                    positions, intrinsic, owner=owner, polar_cosine=mu
                )
                horizontal = torch.sqrt(1.0 - mu.square())
                for start, stop in intervals:
                    half_phi = 0.5 * (stop - start)
                    phi = 0.5 * (stop + start) + half_phi * phi_nodes
                    normal = torch.stack(
                        (
                            horizontal * torch.cos(phi),
                            horizontal * torch.sin(phi),
                            mu.expand_as(phi),
                        ),
                        dim=1,
                    )
                    surface = positions[owner] + radius * normal
                    displacement = surface[:, None, :] - positions[None, :, :]
                    separation_squared = displacement.square().sum(dim=-1)
                    if bool((separation_squared <= 0.0).any()):
                        raise ValueError("Contact R6 node meets an atomic center.")
                    kernel = (displacement * normal[:, None, :]).sum(
                        dim=-1
                    ) / separation_squared.pow(squared_distance_power)
                    flux = flux + (
                        radius.square()
                        * mu_jacobian
                        * mu_weight
                        * half_phi
                        / (4.0 * math.pi)
                        * (phi_weights[:, None] * kernel).sum(dim=0)
                    )
    if not bool(torch.isfinite(flux).all()):
        raise RuntimeError("Contact R6 flux quadrature produced a nonfinite value.")
    return flux


def contact_patch_inverse_cube(
    positions_angstrom: Tensor,
    intrinsic_radii_angstrom: Tensor,
    *,
    mu_order: int = 32,
    phi_order: int = 32,
) -> ContactFluxResult:
    """Return only the locally exposed contact contribution to R6 inverse cubed."""
    flux = _contact_patch_flux(
        positions_angstrom,
        intrinsic_radii_angstrom,
        mu_order=mu_order,
        phi_order=phi_order,
        squared_distance_power=3.0,
    )
    return ContactFluxResult(flux, mu_order, phi_order)


def _pair_torus_flux(
    positions_angstrom: Tensor,
    intrinsic_radii_angstrom: Tensor,
    pair_indices: tuple[int, int],
    *,
    theta_order: int,
    meridian_order: int,
    squared_distance_power: float,
) -> tuple[Tensor, str, int]:
    """Integrate one locally exposed regular or clipped-spindle torus.

    Other expanded atomic balls clip the probe-center circle. This does not
    include concave probe patches or global exterior-connectivity/ownership
    checks, so the flux cannot be promoted to a whole-molecule Born radius.
    """
    import torch

    positions = positions_angstrom
    intrinsic = intrinsic_radii_angstrom
    _validate_inputs(positions, intrinsic, 2)
    if len(positions) > 12:
        raise RuntimeError("Unregistered pair-torus quadrature is capped at 12 atoms.")
    for name, order in (
        ("theta_order", theta_order),
        ("meridian_order", meridian_order),
    ):
        if isinstance(order, bool) or not isinstance(order, int) or order < 8:
            raise ValueError(f"R6 torus {name} must be an integer at least 8.")
    arcs = pair_probe_exposed_arcs(positions, intrinsic, pair_indices)
    circle = arcs.circle
    first, second = pair_indices
    axis = circle.axis
    distance = torch.linalg.vector_norm(positions[second] - positions[first])
    along = torch.dot(circle.center - positions[first], axis)
    radius = circle.radius
    first_angle = torch.atan2(-along, radius)
    second_angle = torch.atan2(distance - along, radius)
    if bool(torch.abs(radius - PROBE_ANGSTROM) <= _GEOMETRY_EVENT_MARGIN_ANGSTROM):
        raise ValueError("A pinching spindle torus is outside the smooth R6 branch.")
    if bool(radius > PROBE_ANGSTROM):
        meridians = ((first_angle, second_angle),)
        regime = "regular-torus"
    else:
        opening = torch.acos(radius / PROBE_ANGSTROM)
        meridians = ((first_angle, -opening), (opening, second_angle))
        if any(bool(stop <= start) for start, stop in meridians):
            raise ValueError("Spindle torus has no ordered exposed meridional arcs.")
        regime = "spindle-torus"
    flux = torch.zeros(len(positions), dtype=torch.float64, device=positions.device)
    theta_nodes, theta_weights = legendre_rule(theta_order, str(positions.device))
    meridian_nodes, meridian_weights = legendre_rule(
        meridian_order, str(positions.device)
    )
    for theta_start, theta_stop in arcs.intervals:
        theta_half = 0.5 * (theta_stop - theta_start)
        theta = 0.5 * (theta_start + theta_stop) + theta_half * theta_nodes
        radial_direction = (
            torch.cos(theta)[:, None] * arcs.basis_u
            + torch.sin(theta)[:, None] * arcs.basis_v
        )
        for meridian_start, meridian_stop in meridians:
            meridian_half = 0.5 * (meridian_stop - meridian_start)
            meridian = (
                0.5 * (meridian_start + meridian_stop) + meridian_half * meridian_nodes
            )
            axial = along + PROBE_ANGSTROM * torch.sin(meridian)
            radial = radius - PROBE_ANGSTROM * torch.cos(meridian)
            if bool((radial <= 0.0).any()):
                raise ValueError("Self-intersecting torus area was not fully clipped.")
            surface = (
                positions[first]
                + axial[None, :, None] * axis
                + radial[None, :, None] * radial_direction[:, None, :]
            )
            normal = (
                -torch.sin(meridian)[None, :, None] * axis
                + torch.cos(meridian)[None, :, None] * radial_direction[:, None, :]
            )
            displacement = surface[:, :, None, :] - positions[None, None, :, :]
            separation_squared = displacement.square().sum(dim=-1)
            if bool((separation_squared <= 0.0).any()):
                raise ValueError("R6 torus node meets an atomic center.")
            kernel = (displacement * normal[:, :, None, :]).sum(
                dim=-1
            ) / separation_squared.pow(squared_distance_power)
            weights = (
                theta_half
                * meridian_half
                * PROBE_ANGSTROM
                / (4.0 * math.pi)
                * theta_weights[:, None]
                * meridian_weights[None, :]
                * radial[None, :]
            )
            flux = flux + (weights[:, :, None] * kernel).sum(dim=(0, 1))
    if not bool(torch.isfinite(flux).all()):
        raise RuntimeError("R6 pair-torus quadrature produced a nonfinite flux.")
    return flux, regime, len(arcs.intervals)


def pair_torus_inverse_cube(
    positions_angstrom: Tensor,
    intrinsic_radii_angstrom: Tensor,
    pair_indices: tuple[int, int],
    *,
    theta_order: int = 32,
    meridian_order: int = 32,
) -> PairTorusFluxResult:
    """Return only one locally exposed pair-torus R6 inverse-cubed flux."""
    flux, regime, arc_count = _pair_torus_flux(
        positions_angstrom,
        intrinsic_radii_angstrom,
        pair_indices,
        theta_order=theta_order,
        meridian_order=meridian_order,
        squared_distance_power=3.0,
    )
    return PairTorusFluxResult(flux, pair_indices, regime, arc_count)


def _triple_probe_patch_flux(
    positions_angstrom: Tensor,
    intrinsic_radii_angstrom: Tensor,
    *,
    order: int,
    squared_distance_power: float,
) -> tuple[Tensor, Tensor]:
    """Integrate unclipped three-contact spherical triangles for exactly 3 atoms.

    A normalized barycentric chart maps a square onto each great-circle
    spherical triangle. Its analytic solid-angle Jacobian carries the moving
    curved area into Torch AD. Four-or-more atoms require probe-probe patch
    intersection clipping and fail closed here.
    """
    import torch

    positions = positions_angstrom
    intrinsic = intrinsic_radii_angstrom
    _validate_inputs(positions, intrinsic, 3)
    if len(positions) != 3:
        raise NotImplementedError(
            "Multiatom probe-patch clipping is not implemented for four or more atoms."
        )
    if isinstance(order, bool) or not isinstance(order, int) or order < 8:
        raise ValueError("R6 probe-patch order must be an integer at least 8.")
    geometry = triple_probe_centers(positions, intrinsic, (0, 1, 2))
    if len(geometry.centers) == 2 and bool(
        torch.linalg.vector_norm(geometry.centers[1] - geometry.centers[0])
        <= 2.0 * PROBE_ANGSTROM + _GEOMETRY_EVENT_MARGIN_ANGSTROM
    ):
        raise ValueError("Overlapping probe spheres need explicit patch trimming.")
    nodes, weights = legendre_rule(order, str(positions.device))
    first_chart = 0.5 * (nodes + 1.0)
    chart_weights = 0.5 * weights
    flux = torch.zeros(3, dtype=torch.float64, device=positions.device)
    areas = []
    expanded = intrinsic + PROBE_ANGSTROM
    for center, locally_exposed in zip(geometry.centers, geometry.locally_exposed):
        if not bool(locally_exposed):
            continue
        directions = (positions - center) / expanded[:, None]
        first, second, third = directions.unbind(dim=0)
        determinant = torch.dot(first, torch.linalg.cross(second, third))
        if bool(torch.abs(determinant) <= 1e-10):
            raise ValueError("Degenerate concave probe triangle is a topology event.")
        denominator = (
            1.0
            + torch.dot(first, second)
            + torch.dot(second, third)
            + torch.dot(third, first)
        )
        if bool(denominator <= 0.0):
            raise ValueError("Probe triangle exceeds the validated minor hemisphere.")
        edge = (1.0 - first_chart)[:, None] * (second - first) + first_chart[
            :, None
        ] * (third - first)
        raw = first + first_chart[:, None, None] * edge[None, :, :]
        length = torch.linalg.vector_norm(raw, dim=-1)
        if bool((length <= 1e-10).any()):
            raise ValueError("Probe triangle chart became singular.")
        normal_from_probe = raw / length[:, :, None]
        solid_angle_weight = (
            first_chart[:, None]
            * torch.abs(determinant)
            / length.pow(3)
            * chart_weights[:, None]
            * chart_weights[None, :]
        )
        area_weight = PROBE_ANGSTROM**2 * solid_angle_weight
        surface = center + PROBE_ANGSTROM * normal_from_probe
        displacement = surface[:, :, None, :] - positions[None, None, :, :]
        separation_squared = displacement.square().sum(dim=-1)
        if bool((separation_squared <= 0.0).any()):
            raise ValueError("R6 probe-patch node meets an atomic center.")
        outward_normal = -normal_from_probe
        kernel = (displacement * outward_normal[:, :, None, :]).sum(
            dim=-1
        ) / separation_squared.pow(squared_distance_power)
        flux = flux + (area_weight[:, :, None] * kernel).sum(dim=(0, 1)) / (
            4.0 * math.pi
        )
        areas.append(area_weight.sum())
    patch_areas = torch.stack(areas) if areas else positions.new_empty((0,))
    if not bool(torch.isfinite(flux).all()):
        raise RuntimeError(
            "R6 concave probe-patch quadrature produced a nonfinite flux."
        )
    return flux, patch_areas


def triple_probe_patch_inverse_cube(
    positions_angstrom: Tensor,
    intrinsic_radii_angstrom: Tensor,
    *,
    order: int = 48,
) -> ProbePatchFluxResult:
    """Return only the unclipped three-contact R6 inverse-cubed flux."""
    flux, areas = _triple_probe_patch_flux(
        positions_angstrom,
        intrinsic_radii_angstrom,
        order=order,
        squared_distance_power=3.0,
    )
    return ProbePatchFluxResult(flux, areas)


def local_patch_gauss_flux(
    positions_angstrom: Tensor,
    intrinsic_radii_angstrom: Tensor,
    *,
    order: int = 48,
) -> Tensor:
    """Diagnose closed-surface ownership for a simple three-site SES.

    The radial-power-three Gauss flux must equal one for each enclosed atom.
    This necessary topological check does not certify a general SES, R6
    accuracy, exterior connectivity, or derivative continuity.
    """
    if len(positions_angstrom) != 3:
        raise NotImplementedError("Gauss patch closure is restricted to three sites.")
    contact = _contact_patch_flux(
        positions_angstrom,
        intrinsic_radii_angstrom,
        mu_order=order,
        phi_order=order,
        squared_distance_power=1.5,
    )
    pair_fluxes = [
        _pair_torus_flux(
            positions_angstrom,
            intrinsic_radii_angstrom,
            pair,
            theta_order=order,
            meridian_order=order,
            squared_distance_power=1.5,
        )[0]
        for pair in ((0, 1), (0, 2), (1, 2))
    ]
    probe, _ = _triple_probe_patch_flux(
        positions_angstrom,
        intrinsic_radii_angstrom,
        order=order,
        squared_distance_power=1.5,
    )
    return contact + sum(pair_fluxes) + probe
