"""Unregistered Torch sigma-split PBSA dispersion surface quadrature.

The boundary is the exposed surface of `rmin+sprob` spheres, not the SAV
cavity boundary or the polar R6 SES. This mathematical slice is not a complete
CHA/PBSA provider. The reported z-rule error excludes azimuthal error and is
not a certified force/error enclosure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from .torch_sphere_union_geometry import (
    exposed_circle_arcs,
    legendre_rule,
    pair_xy_geometry,
    section_circles,
    sphere_z_breakpoints,
    validate_sphere_events,
)

if TYPE_CHECKING:
    from torch import Tensor


DISPERSION_PROBE_ANGSTROM = 0.557
WATER_OXYGEN_RMIN_ANGSTROM = 1.7683
WATER_OXYGEN_EPSILON_KCAL_MOL = 0.1520
WATER_DENSITY_PER_ANGSTROM3 = 0.03333 * 1.129
_TWO_TO_NEGATIVE_ONE_SIXTH = 2.0 ** (-1.0 / 6.0)


@dataclass(frozen=True)
class DispersionResult:
    energy_kcal_mol: Tensor
    estimated_z_quadrature_error_kcal_mol: float
    accepted_intervals: int
    z_nodes_evaluated: int
    azimuthal_order: int


def _split_arcs_at_sigma(starts, stops, positions, source, circle_radius, z, sigma):
    """Analytic azimuthal roots of each site's `r=sigma` branch surface."""
    import torch

    offset = positions[source, :2] - positions[:, :2]
    zero_xy = offset.square().sum(dim=-1) == 0.0
    safe_offset = torch.where(
        zero_xy[:, None],
        torch.tensor([1.0, 0.0], dtype=positions.dtype, device=positions.device),
        offset,
    )
    xy_distance = torch.where(
        zero_xy, 0.0, torch.linalg.vector_norm(safe_offset, dim=-1)
    )
    amplitude = 2.0 * circle_radius[:, None] * xy_distance[None, :]
    constant = (
        xy_distance[None, :].square()
        + circle_radius[:, None].square()
        + (z[:, None] - positions[None, :, 2]).square()
    )
    safe_amplitude = torch.where(amplitude > 0.0, amplitude, 1.0)
    cosine = (sigma[None, :].square() - constant) / safe_amplitude
    crosses = (amplitude > 1e-12) & (cosine > -1.0) & (cosine < 1.0)
    safe_cosine = torch.where(crosses, cosine, 0.0)
    margin = 4.0 * torch.finfo(positions.dtype).eps
    opening = torch.acos(torch.clamp(safe_cosine, -1.0 + margin, 1.0 - margin))
    phase = torch.atan2(safe_offset[:, 1], safe_offset[:, 0])[None, :]
    first = torch.remainder(phase - opening, 2.0 * math.pi)
    second = torch.remainder(phase + opening, 2.0 * math.pi)
    roots = torch.cat((first, second), dim=1)
    valid = torch.cat((crosses, crosses), dim=1)
    roots = roots[:, None, :].expand(-1, starts.shape[1], -1)
    inside = (
        valid[:, None, :]
        & (roots > starts[:, :, None] + 1e-12)
        & (roots < stops[:, :, None] - 1e-12)
    )
    roots = torch.where(inside, roots, stops[:, :, None])
    roots = torch.sort(roots, dim=-1).values
    return (
        torch.cat((starts[:, :, None], roots), dim=-1),
        torch.cat((roots, stops[:, :, None]), dim=-1),
    )


def _validate(positions, rmin, epsilon, atol, rtol, phi_order, max_depth):
    import torch

    if (
        not isinstance(positions, torch.Tensor)
        or positions.dtype != torch.float64
        or positions.ndim != 2
        or positions.shape[1] != 3
        or positions.shape[0] == 0
    ):
        raise TypeError("Dispersion positions must be nonempty float64 Torch [N,3].")
    for name, value in (("rmin", rmin), ("epsilon", epsilon)):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float64
            or value.shape != (len(positions),)
            or value.device != positions.device
        ):
            raise TypeError(f"Dispersion {name} must be same-device float64 [N].")
    if not bool(
        torch.isfinite(positions).all()
        and torch.isfinite(rmin).all()
        and torch.isfinite(epsilon).all()
    ):
        raise ValueError("Dispersion inputs must be finite.")
    if not bool((rmin > 0.0).all()) or not bool((epsilon >= 0.0).all()):
        raise ValueError("Dispersion rmin must be positive and epsilon nonnegative.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value <= 0.0
        for value in (atol, rtol)
    ):
        raise ValueError("Dispersion tolerances must be finite and positive.")
    if isinstance(phi_order, bool) or not isinstance(phi_order, int) or phi_order < 8:
        raise ValueError("Azimuthal order must be an integer >=8.")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError("Dispersion max_depth must be a positive integer.")
    validate_sphere_events(
        positions, rmin + DISPERSION_PROBE_ANGSTROM, label="dispersion SAS"
    )


_ARC_CHUNK = 64
_PEAK_TEMPORARY_BYTES = 256 * 1024 * 1024


def _arc_chunk_contribution(
    positions,
    source_radius,
    sigma,
    raw_a,
    raw_b,
    inside_coefficient,
    z,
    circle_radius,
    starts,
    stops,
    phi_nodes,
    phi_weights,
    source,
):
    """Integrate a bounded batch of nonempty exposed subarcs."""
    import torch

    half = 0.5 * (stops - starts)
    phi = 0.5 * (starts + stops)[:, None] + half[:, None] * phi_nodes
    cosine = torch.cos(phi)
    sine = torch.sin(phi)
    surface = torch.stack(
        (
            positions[source, 0] + circle_radius[:, None] * cosine,
            positions[source, 1] + circle_radius[:, None] * sine,
            z[:, None].expand_as(phi),
        ),
        dim=-1,
    )
    normal = torch.stack(
        (
            circle_radius[:, None] * cosine / source_radius,
            circle_radius[:, None] * sine / source_radius,
            ((z - positions[source, 2]) / source_radius)[:, None].expand_as(phi),
        ),
        dim=-1,
    )
    displacement = surface[:, :, None, :] - positions[None, None, :, :]
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    if bool((distance <= 1e-12).any()):
        raise ValueError("An exposed dispersion surface node meets a solute site.")
    inverse_six = distance.pow(-6)
    outside = -raw_b / 3.0 * inverse_six + raw_a / 9.0 * inverse_six.square()
    inside = inside_coefficient / distance.pow(3)
    radial = torch.where(distance >= sigma, outside, inside)
    normal_dot = (displacement * normal[:, :, None, :]).sum(dim=-1)
    weights = (half[:, None] * phi_weights * source_radius)[:, :, None]
    return WATER_DENSITY_PER_ANGSTROM3 * (radial * normal_dot * weights).sum(dim=(1, 2))


def _section_dispersion(
    positions, sas_radii, sigma, mixed_epsilon, z, xy_distance, angle, phi_order
):
    import torch

    section_radii, active = section_circles(positions, sas_radii, z)
    phi_nodes, phi_weights = legendre_rule(phi_order, str(positions.device))
    raw_b = 4.0 * mixed_epsilon * sigma.pow(6)
    raw_a = raw_b * sigma.pow(6)
    inside_coefficient = -raw_b / (3.0 * sigma.pow(3)) + raw_a / (9.0 * sigma.pow(9))
    value = torch.zeros(len(z), dtype=z.dtype, device=z.device)
    work_units = 0
    for source in range(len(sas_radii)):
        starts, stops, circle_radius, visible = exposed_circle_arcs(
            source, section_radii, active, xy_distance, angle
        )
        starts, stops = _split_arcs_at_sigma(
            starts, stops, positions, source, circle_radius, z, sigma
        )
        selected = ((stops > starts) & visible[:, None, None]).nonzero()
        projected_work = len(selected) * phi_order * len(sas_radii)
        if work_units + projected_work > 10_000_000:
            raise RuntimeError("Dispersion z-section exceeds its work budget.")
        for offset in range(0, len(selected), _ARC_CHUNK):
            rows = selected[offset : offset + _ARC_CHUNK]
            z_index, arc_index, split_index = rows.unbind(dim=1)
            args = (
                positions,
                sas_radii[source],
                sigma,
                raw_a,
                raw_b,
                inside_coefficient,
                z[z_index],
                circle_radius[z_index],
                starts[z_index, arc_index, split_index],
                stops[z_index, arc_index, split_index],
                phi_nodes,
                phi_weights,
                source,
            )
            contribution = _arc_chunk_contribution(*args)
            value = value.index_add(0, z_index, contribution)
            work_units += len(rows) * phi_order * len(sas_radii)
    return value, work_units


def _sigma_z_breakpoints(positions, sas_radii, sigma, existing):
    """z loci where a SAS patch crosses a site's sigma-split sphere."""
    import torch

    points = list(existing)
    z_axis = torch.tensor(
        [0.0, 0.0, 1.0], dtype=positions.dtype, device=positions.device
    )
    for source in range(len(sas_radii)):
        for site in range(len(sigma)):
            displacement = positions[site] - positions[source]
            distance = torch.linalg.vector_norm(displacement)
            if not bool(
                (torch.abs(sas_radii[source] - sigma[site]) < distance)
                & (distance < sas_radii[source] + sigma[site])
            ):
                continue
            direction = displacement / distance
            along = (
                sas_radii[source].square() - sigma[site].square() + distance.square()
            ) / (2.0 * distance)
            circle_center = positions[source] + along * direction
            circle_radius = torch.sqrt(sas_radii[source].square() - along.square())
            upward = z_axis - direction[2] * direction
            length = torch.linalg.vector_norm(upward)
            if bool(length <= 1e-12):
                points.append(circle_center[2])
                continue
            z_direction = upward / length
            for sign in (-1.0, 1.0):
                extremum = circle_center + sign * circle_radius * z_direction
                if any(
                    bool(
                        torch.linalg.vector_norm(extremum - positions[other])
                        < sas_radii[other] - 1e-10
                    )
                    for other in range(len(sas_radii))
                    if other != source
                ):
                    continue
                points.append(extremum[2])
    ordered = sorted(points, key=lambda value: float(value.detach()))
    deduplicated = [ordered[0]]
    for value in ordered[1:]:
        if float((value - deduplicated[-1]).detach()) > 1e-12:
            deduplicated.append(value)
    return deduplicated


def dispersion_from_rmin_epsilon(
    positions_angstrom: Tensor,
    lj_rmin_angstrom: Tensor,
    lj_epsilon_kcal_mol: Tensor,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-8,
    phi_order: int = 48,
    max_depth: int = 24,
) -> DispersionResult:
    """Frozen `inp=2,decompopt=2,use_rmin=1` attractive term only.

    The sigma branch is retained exactly. Fixed Gauss azimuthal order requires
    an independent paired-order convergence check before any force claim.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    # Guard before geometry validation builds any pairwise [N,N,3] arrays.
    if (
        isinstance(positions_angstrom, torch.Tensor)
        and positions_angstrom.ndim == 2
        and isinstance(phi_order, int)
        and not isinstance(phi_order, bool)
    ):
        count = len(positions_angstrom)
        if (
            count > 6
            and torch.is_grad_enabled()
            and any(
                isinstance(value, torch.Tensor) and value.requires_grad
                for value in (positions_angstrom, lj_rmin_angstrom, lj_epsilon_kcal_mol)
            )
        ):
            raise RuntimeError(
                "Unregistered dispersion AD is limited to six sites until "
                "whole-molecule graph memory is qualified."
            )
        # Non-reentrant backward may replay the entire accepted section and
        # retain sorting/branch tensors across all source spheres. Account
        # for N sources, int64 permutations, and multiple live intermediates.
        geometry_bytes = 24 * (2 * count + 1) * (2 * count) * count * 8 * 8
        chunk_bytes = _ARC_CHUNK * phi_order * count * 3 * 8 * 8
        if count > 30 or geometry_bytes + chunk_bytes > _PEAK_TEMPORARY_BYTES:
            raise RuntimeError(
                "Dispersion quadrature exceeds its geometry-and-chunk memory budget."
            )
    _validate(
        positions_angstrom,
        lj_rmin_angstrom,
        lj_epsilon_kcal_mol,
        atol,
        rtol,
        phi_order,
        max_depth,
    )
    # The whole accepted z-section, including arc-root construction, is
    # checkpointed below so retained AD memory scales with inputs/outputs.
    centered = positions_angstrom - positions_angstrom.mean(dim=0)
    sas_radii = lj_rmin_angstrom + DISPERSION_PROBE_ANGSTROM
    sigma = (lj_rmin_angstrom + WATER_OXYGEN_RMIN_ANGSTROM) * _TWO_TO_NEGATIVE_ONE_SIXTH
    mixed_epsilon = torch.sqrt(lj_epsilon_kcal_mol * WATER_OXYGEN_EPSILON_KCAL_MOL)
    xy_distance, angle = pair_xy_geometry(centered)
    lower = torch.min(centered[:, 2] - sas_radii)
    upper = torch.max(centered[:, 2] + sas_radii)
    full_width = float((upper - lower).detach())
    breakpoints = _sigma_z_breakpoints(
        centered,
        sas_radii,
        sigma,
        sphere_z_breakpoints(centered, sas_radii, lower, upper),
    )
    stack = [
        (breakpoints[index], breakpoints[index + 1], 0)
        for index in range(len(breakpoints) - 1)
    ]
    accepted = []
    estimated_error = 0.0
    z_nodes = 0
    work_units = 0

    def section_values(
        positions, radii, sigma_values, epsilon_values, z, distances, angles
    ):
        return _section_dispersion(
            positions,
            radii,
            sigma_values,
            epsilon_values,
            z,
            distances,
            angles,
            phi_order,
        )[0]

    def integrate(first, last, order):
        nonlocal z_nodes, work_units
        nodes, weights = legendre_rule(order, str(centered.device))
        half = 0.5 * (last - first)
        z = 0.5 * (first + last) + half * nodes
        args = (centered, sas_radii, sigma, mixed_epsilon, z, xy_distance, angle)
        if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in args):
            values = checkpoint(
                section_values,
                *args,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            units = 0  # charged from the matched no-grad estimate below
        else:
            values, units = _section_dispersion(*args, phi_order)
        z_nodes += order
        work_units += units
        return half * torch.dot(weights, values), units

    while stack:
        first, last, depth = stack.pop()
        # Error estimation and rejected intervals do not belong to the
        # differentiable scalar. Recompute only accepted fine intervals with
        # bounded activation checkpointing for their eventual backward pass.
        with torch.no_grad():
            coarse, _ = integrate(first, last, 12)
            fine_estimate, fine_units = integrate(first, last, 24)
        error = float((fine_estimate - coarse).abs())
        budget = atol * float((last - first).detach()) / full_width
        budget += rtol * abs(float(fine_estimate))
        # A narrow interval can receive a budget below float64 summation
        # roundoff. The global estimated-error check below still applies.
        budget = max(
            budget,
            32.0 * torch.finfo(torch.float64).eps * max(1.0, abs(float(fine_estimate))),
        )
        if error <= budget:
            if torch.is_grad_enabled():
                fine, _ = integrate(first, last, 24)
                work_units += 2 * fine_units  # forward and backward replay
            else:
                fine = fine_estimate
            accepted.append(fine)
            estimated_error += error
        elif depth >= max_depth:
            raise RuntimeError(
                f"Dispersion z quadrature did not converge at depth {depth}: "
                f"estimated {error} > budget {budget}."
            )
        else:
            middle = 0.5 * (first + last)
            stack.extend(((middle, last, depth + 1), (first, middle, depth + 1)))
        if (
            len(accepted) + len(stack) > 10_000
            or z_nodes > 1_000_000
            or work_units > 100_000_000
        ):
            raise RuntimeError("Dispersion quadrature exceeded its work budget.")
    energy = torch.stack(accepted).sum()
    if not bool(torch.isfinite(energy)):
        raise RuntimeError("Dispersion quadrature produced a nonfinite energy.")
    if estimated_error > atol + rtol * abs(float(energy.detach())):
        raise RuntimeError("Dispersion summed z-rule error exceeds its global budget.")
    return DispersionResult(energy, estimated_error, len(accepted), z_nodes, phi_order)
