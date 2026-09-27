"""Unregistered Torch quadrature for the sharp PBSA solvent-accessible volume.

This is only the cavity-volume slice of a proposed continuum CHA/PBSA profile.
It does not calculate R6 Born radii, dispersion, a complete solvent energy, or
production forces. On each z section the union-of-disks area is evaluated from
its exposed circular arcs; adaptive Gauss integration acts only along z.
The numerical quadrature error is an estimate, not a rigorous bound.
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


CAVITY_PROBE_ANGSTROM = 1.3
CAVITY_COEFFICIENT_KCAL_MOL_ANGSTROM3 = 0.0378
CAVITY_OFFSET_KCAL_MOL = -0.5692

# This archived reference implementation deliberately has a finite work
# envelope. Every attempted interval evaluates 16 + 32 section nodes.
_MAX_ADAPTIVE_DEPTH = 20
_MAX_INTERVAL_ATTEMPTS = 100_000
_MAX_AREA_EVALUATIONS = 48 * _MAX_INTERVAL_ATTEMPTS
# Preflight the dense NxN geometry and both extrema's worst-case third-site
# containment scans in ``sphere_z_breakpoints`` before materializing work.
_MAX_DENSE_PAIR_ENTRIES = 250_000
_MAX_BREAKPOINT_CONTAINMENT_CHECKS = 2_000_000


@dataclass(frozen=True)
class SavVolumeResult:
    volume_angstrom3: Tensor
    estimated_quadrature_error_angstrom3: float
    accepted_intervals: int
    area_evaluations: int


@dataclass(frozen=True)
class CavityResult:
    energy_kcal_mol: Tensor
    volume_angstrom3: Tensor
    estimated_energy_error_kcal_mol: float
    accepted_intervals: int
    area_evaluations: int


def _arc_area(center_x, center_y, radius, start, stop):
    return 0.5 * (
        radius * center_x * (stop.sin() - start.sin())
        + radius * center_y * (start.cos() - stop.cos())
        + radius.square() * (stop - start)
    )


def _section_area(positions, radii, z, xy_distance, angle):
    """Union-of-disks area at a vector of z nodes, with a live Torch graph."""
    import torch

    count = len(radii)
    section_radii, active = section_circles(positions, radii, z)
    area = torch.zeros(len(z), dtype=z.dtype, device=z.device)
    for source in range(count):
        starts, stops, radius, visible = exposed_circle_arcs(
            source, section_radii, active, xy_distance, angle
        )
        contribution = _arc_area(
            positions[source, 0],
            positions[source, 1],
            radius[:, None],
            starts,
            stops,
        ).sum(dim=1)
        area = area + torch.where(visible, contribution, 0.0)
    return area


def _validate(positions, radii, atol, rtol, max_depth):
    import torch

    if (
        not isinstance(positions, torch.Tensor)
        or positions.dtype != torch.float64
        or positions.ndim != 2
        or positions.shape[1] != 3
        or positions.shape[0] == 0
    ):
        raise TypeError("SAV positions must be a nonempty float64 Torch [N,3] tensor.")
    if (
        not isinstance(radii, torch.Tensor)
        or radii.dtype != torch.float64
        or radii.shape != (len(positions),)
        or radii.device != positions.device
    ):
        raise TypeError("SAV radii must be a same-device float64 Torch [N] tensor.")
    if not bool(torch.isfinite(positions).all() and torch.isfinite(radii).all()):
        raise ValueError("SAV geometry must be finite.")
    if not bool((radii > 0.0).all()):
        raise ValueError("SAV radii must be positive.")
    minimum_tolerance = torch.finfo(torch.float64).eps
    if any(
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < minimum_tolerance
        for value in (atol, rtol)
    ):
        raise ValueError(
            "SAV tolerances must be finite real numbers no smaller than float64 epsilon."
        )
    if (
        isinstance(max_depth, bool)
        or not isinstance(max_depth, int)
        or not 1 <= max_depth <= _MAX_ADAPTIVE_DEPTH
    ):
        raise ValueError(
            f"SAV max_depth must be an integer in [1, {_MAX_ADAPTIVE_DEPTH}]."
        )


def _admit_sav_geometry_work(site_count: int) -> None:
    """Bound pair storage and worst-case cubic breakpoint traversal."""
    dense_pair_entries = site_count * site_count
    distinct_pairs = site_count * (site_count - 1) // 2
    containment_checks = 2 * distinct_pairs * max(0, site_count - 2)
    if (
        dense_pair_entries > _MAX_DENSE_PAIR_ENTRIES
        or containment_checks > _MAX_BREAKPOINT_CONTAINMENT_CHECKS
    ):
        raise RuntimeError(
            "SAV geometry work budget exceeded: projected "
            f"{dense_pair_entries} dense pair entries and "
            f"{containment_checks} breakpoint containment checks; limits are "
            f"{_MAX_DENSE_PAIR_ENTRIES} and "
            f"{_MAX_BREAKPOINT_CONTAINMENT_CHECKS}, respectively."
        )


def _strict_midpoint(first, last):
    """Bisect an interval or fail when Float64 has no interior midpoint."""
    middle = 0.5 * (first + last)
    if bool((middle == first) | (middle == last)):
        raise RuntimeError(
            "SAV quadrature cannot refine an interval because its float64 midpoint "
            "collapsed to an endpoint; the requested precision is unattainable."
        )
    return middle


def sphere_union_volume(
    positions_angstrom: Tensor,
    radii_angstrom: Tensor,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
    max_depth: int = 14,
) -> SavVolumeResult:
    """Sharp union volume with a differentiable adaptive Torch quadrature.

    Tolerances apply to a paired Gauss-rule estimate, not a certified error
    enclosure. This research slice remains unregistered until convergence and
    complete-component force gates pass.
    """
    import torch

    _validate(positions_angstrom, radii_angstrom, atol, rtol, max_depth)
    _admit_sav_geometry_work(len(positions_angstrom))
    validate_sphere_events(positions_angstrom, radii_angstrom, label="SAV")
    centered = positions_angstrom - positions_angstrom.mean(dim=0)
    xy_distance, angle = pair_xy_geometry(centered)
    lower = torch.min(centered[:, 2] - radii_angstrom)
    upper = torch.max(centered[:, 2] + radii_angstrom)
    total_width = float((upper - lower).detach())
    accepted = []
    points = sphere_z_breakpoints(centered, radii_angstrom, lower, upper)
    stack = [(points[index], points[index + 1], 0) for index in range(len(points) - 1)]
    if len(stack) > _MAX_INTERVAL_ATTEMPTS:
        raise RuntimeError("SAV quadrature exceeded its interval work budget.")
    estimated_error = 0.0
    evaluations = 0
    attempted_intervals = 0
    rejected_intervals = 0

    def integrate(first, last, order):
        nonlocal evaluations
        nodes, weights = legendre_rule(order, str(centered.device))
        half = 0.5 * (last - first)
        z = 0.5 * (first + last) + half * nodes
        area = _section_area(centered, radii_angstrom, z, xy_distance, angle)
        evaluations += order
        return half * torch.dot(weights, area)

    while stack:
        if (
            attempted_intervals >= _MAX_INTERVAL_ATTEMPTS
            or evaluations > _MAX_AREA_EVALUATIONS - 48
        ):
            raise RuntimeError(
                "SAV quadrature exceeded its work budget "
                f"({len(accepted)} accepted, {rejected_intervals} rejected, "
                f"{evaluations} area evaluations)."
            )
        first, last, depth = stack.pop()
        attempted_intervals += 1
        coarse = integrate(first, last, 16)
        fine = integrate(first, last, 32)
        error = float((fine - coarse).detach().abs())
        local_budget = atol * float((last - first).detach()) / total_width
        local_budget += rtol * abs(float(fine.detach()))
        if error <= local_budget:
            accepted.append(fine)
            estimated_error += error
        else:
            rejected_intervals += 1
            if depth >= max_depth:
                raise RuntimeError(
                    f"SAV quadrature did not converge at depth {depth}: "
                    f"estimated {error} > budget {local_budget}."
                )
            middle = _strict_midpoint(first, last)
            stack.append((middle, last, depth + 1))
            stack.append((first, middle, depth + 1))
    volume = torch.stack(accepted).sum()
    if not bool(torch.isfinite(volume)) or not bool(volume > 0.0):
        raise RuntimeError("SAV quadrature produced a nonfinite or nonpositive volume.")
    global_budget = atol + rtol * abs(float(volume.detach()))
    if estimated_error > global_budget:
        raise RuntimeError(
            "SAV summed quadrature error exceeds its global tolerance: "
            f"estimated {estimated_error} > budget {global_budget}."
        )
    return SavVolumeResult(volume, estimated_error, len(accepted), evaluations)


def cavity_from_rmin(
    positions_angstrom: Tensor,
    lj_rmin_angstrom: Tensor,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
    max_depth: int = 14,
) -> CavityResult:
    """Frozen PBSA `inp=2,use_rmin=1,use_sav=1` cavity formula only."""
    import torch

    if not isinstance(lj_rmin_angstrom, torch.Tensor) or not bool(
        torch.isfinite(lj_rmin_angstrom).all() and (lj_rmin_angstrom > 0.0).all()
    ):
        raise ValueError("Cavity input rmin must be a finite positive tensor.")
    sav = sphere_union_volume(
        positions_angstrom,
        lj_rmin_angstrom + CAVITY_PROBE_ANGSTROM,
        atol=atol,
        rtol=rtol,
        max_depth=max_depth,
    )
    energy = (
        CAVITY_COEFFICIENT_KCAL_MOL_ANGSTROM3 * sav.volume_angstrom3
        + CAVITY_OFFSET_KCAL_MOL
    )
    return CavityResult(
        energy,
        sav.volume_angstrom3,
        CAVITY_COEFFICIENT_KCAL_MOL_ANGSTROM3
        * sav.estimated_quadrature_error_angstrom3,
        sav.accepted_intervals,
        sav.area_evaluations,
    )
