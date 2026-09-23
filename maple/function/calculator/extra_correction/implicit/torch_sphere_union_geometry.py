"""Internal sharp sphere-union circle exposures and z-event quadrature rules.

This geometry utility carries no CHA, cavity, or dispersion energy formula.
Branch decisions are discrete only at genuine sphere/topology events; selected
arc endpoints and quadrature bounds retain their Torch coordinate graph.
"""

from __future__ import annotations

from functools import lru_cache
import math

TWO_PI = 2.0 * math.pi


@lru_cache(maxsize=16)
def legendre_rule(order: int, device: str):
    """Float64 Gauss-Legendre nodes/weights from a Torch Jacobi eigensolve."""
    import torch

    if isinstance(order, bool) or not isinstance(order, int) or order < 2:
        raise ValueError("Legendre order must be an integer >=2.")
    indices = torch.arange(1, order, dtype=torch.float64, device=device)
    off_diagonal = indices / torch.sqrt(4.0 * indices.square() - 1.0)
    jacobi = torch.diag(off_diagonal, 1) + torch.diag(off_diagonal, -1)
    nodes, eigenvectors = torch.linalg.eigh(jacobi)
    return nodes, 2.0 * eigenvectors[0].square()


def pair_xy_geometry(positions):
    """Precompute differentiable pair distances and polar angles in the xy plane."""
    import torch

    delta = positions[None, :, :2] - positions[:, None, :2]
    zero_xy = delta.square().sum(dim=-1) == 0.0
    safe_direction = torch.where(
        zero_xy[:, :, None],
        torch.tensor([1.0, 0.0], dtype=positions.dtype, device=positions.device),
        delta,
    )
    distance = torch.where(
        zero_xy, 0.0, torch.linalg.vector_norm(safe_direction, dim=-1)
    )
    angle = torch.remainder(
        torch.atan2(safe_direction[:, :, 1], safe_direction[:, :, 0]), TWO_PI
    )
    return distance, angle


def section_circles(positions, radii, z):
    """Cross-sectional disk radii and active masks for all z nodes."""
    import torch

    squared = radii[None, :].square() - (z[:, None] - positions[None, :, 2]).square()
    active = squared > 0.0
    # sqrt(0) on an inactive circle poisons backward even if visibility is
    # later zero: its local derivative is infinite and 0 * infinity is NaN.
    safe_squared = torch.where(active, squared, 1.0)
    return torch.where(active, torch.sqrt(safe_squared), 0.0), active


def validate_sphere_events(positions, radii, *, label: str) -> None:
    """Reject coincident and tangent spheres before interpreting AD forces."""
    import torch

    displacement = positions[:, None, :] - positions[None, :, :]
    distances = torch.linalg.vector_norm(displacement, dim=-1)
    first, second = torch.triu_indices(
        len(radii), len(radii), offset=1, device=radii.device
    )
    pair_distance = distances[first, second]
    radius_difference = torch.abs(radii[first] - radii[second])
    if bool(((pair_distance == 0.0) & (radius_difference == 0.0)).any()):
        raise ValueError(f"duplicate {label} spheres have ambiguous owner derivatives.")
    tolerance = 1.0e-9
    if bool(
        ((pair_distance - (radii[first] + radii[second])).abs() <= tolerance).any()
    ) or bool(((pair_distance - radius_difference).abs() <= tolerance).any()):
        raise ValueError(f"{label} sphere tangency is outside the smooth branch.")


def exposed_circle_arcs(source, section_radii, active, xy_distance, angle):
    """All nonzero exposed angular intervals of one source circle.

    Returned starts/stops have shape [z_nodes, 2*N+1]; zero-length entries
    are retained for static tensor shapes. `visible` rejects a whole buried
    source circle. This is disk-union geometry, not an area approximation.
    """
    import torch

    count = section_radii.shape[1]
    r = section_radii[:, source : source + 1]
    distance = xy_distance[source][None, :]
    other_active = active.clone()
    other_active[:, source] = False
    covered_whole = other_active & (distance + r <= section_radii)
    if bool(covered_whole.any()):
        equal_circle = (distance == 0.0) & (r == section_radii)
        indices = torch.arange(count, device=r.device)[None, :]
        covered_whole = covered_whole & (~equal_circle | (indices < source))
    visible = active[:, source] & ~covered_whole.any(dim=1)
    proper = (
        active[:, source : source + 1]
        & other_active
        & (distance < r + section_radii)
        & (distance > torch.abs(r - section_radii))
    )
    safe_distance = torch.where(distance > 0.0, distance, 1.0)
    safe_r = torch.where(r > 0.0, r, 1.0)
    cosine = (distance.square() + r.square() - section_radii.square()) / (
        2.0 * safe_distance * safe_r
    )
    safe_cosine = torch.where(proper, cosine, 0.0)
    margin = 4.0 * torch.finfo(r.dtype).eps
    half_angle = torch.acos(torch.clamp(safe_cosine, -1.0 + margin, 1.0 - margin))
    start = angle[source][None, :] - half_angle
    stop = angle[source][None, :] + half_angle
    low_wrap = proper & (start < 0.0)
    high_wrap = proper & (stop > TWO_PI)
    two_pi = torch.as_tensor(TWO_PI, dtype=r.dtype, device=r.device)
    first_start = torch.where(low_wrap, 0.0, start)
    first_stop = torch.where(low_wrap | high_wrap, stop.clamp(max=TWO_PI), stop)
    second_start = torch.where(low_wrap, start + TWO_PI, 0.0)
    second_stop = torch.where(low_wrap, TWO_PI, stop - TWO_PI)
    second_valid = low_wrap | high_wrap
    first_start = torch.where(proper, first_start, two_pi)
    first_stop = torch.where(proper, first_stop, two_pi)
    second_start = torch.where(second_valid, second_start, two_pi)
    second_stop = torch.where(second_valid, second_stop, two_pi)
    starts = torch.cat((first_start, second_start), dim=1)
    stops = torch.cat((first_stop, second_stop), dim=1)
    starts, permutation = torch.sort(starts, dim=1)
    stops = torch.gather(stops, 1, permutation)
    rightmost = torch.cummax(stops, dim=1).values
    previous = torch.cat((torch.zeros_like(rightmost[:, :1]), rightmost[:, :-1]), dim=1)
    gap_end = torch.maximum(starts, previous)
    tail_start = rightmost[:, -1:]
    tail_end = torch.maximum(tail_start, two_pi)
    return (
        torch.cat((previous, tail_start), dim=1),
        torch.cat((gap_end, tail_end), dim=1),
        r[:, 0],
        visible,
    )


def sphere_z_breakpoints(positions, radii, lower, upper):
    """Sphere extrema and exposed pair-circle z extrema, with graph edges."""
    import torch

    count = len(radii)
    points = [lower, upper]
    points.extend(positions[:, 2] - radii)
    points.extend(positions[:, 2] + radii)
    z_axis = torch.tensor(
        [0.0, 0.0, 1.0], dtype=positions.dtype, device=positions.device
    )
    for first in range(count):
        for second in range(first + 1, count):
            displacement = positions[second] - positions[first]
            distance = torch.linalg.vector_norm(displacement)
            if not bool(
                (torch.abs(radii[first] - radii[second]) < distance)
                & (distance < radii[first] + radii[second])
            ):
                continue
            direction = displacement / distance
            along = (
                radii[first].square() - radii[second].square() + distance.square()
            ) / (2.0 * distance)
            circle_center = positions[first] + along * direction
            circle_radius = torch.sqrt(radii[first].square() - along.square())
            upward = z_axis - direction[2] * direction
            upward_norm = torch.linalg.vector_norm(upward)
            if bool(upward_norm <= 1e-12):
                points.append(circle_center[2])
                continue
            z_direction = upward / upward_norm
            for sign in (-1.0, 1.0):
                extremum = circle_center + sign * circle_radius * z_direction
                buried = False
                for other in range(count):
                    if other in (first, second):
                        continue
                    if bool(
                        torch.linalg.vector_norm(extremum - positions[other])
                        < radii[other] - 1e-10
                    ):
                        buried = True
                        break
                if not buried:
                    points.append(extremum[2])
    ordered = sorted(points, key=lambda item: float(item.detach()))
    result = [ordered[0]]
    for value in ordered[1:]:
        if float((value - result[-1]).detach()) > 1e-12:
            result.append(value)
    return result
