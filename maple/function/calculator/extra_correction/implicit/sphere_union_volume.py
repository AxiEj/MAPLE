"""Sharp union-of-spheres volume and center derivatives.

The implementation slices the sphere union perpendicular to ``z``.  Each
cross-section is a union of disks whose exposed circular arcs are integrated
exactly.  Adaptive quadrature is used only in the remaining one-dimensional
integral.  No surface smoothing or fitted parameters are introduced.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad_vec

_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class SphereUnionVolumeResult:
    """Volume calculation result in the units supplied by the caller."""

    volume: float
    gradient: np.ndarray
    quadrature_error: float
    diagnostics: dict[str, float | int]


def _validate_geometry(positions, radii) -> tuple[np.ndarray, np.ndarray]:
    try:
        raw_centers = np.asarray(positions, dtype=object)
    except ValueError:
        raw_centers = None
    try:
        raw_sizes = np.asarray(radii, dtype=object)
    except ValueError:
        raw_sizes = None
    if raw_centers is not None and any(
        isinstance(value, (bool, np.bool_)) for value in raw_centers.flat
    ):
        raise TypeError("positions must contain numbers, not booleans.")
    if raw_sizes is not None and any(
        isinstance(value, (bool, np.bool_)) for value in raw_sizes.flat
    ):
        raise TypeError("radii must contain numbers, not booleans.")
    centers = np.asarray(positions, dtype=float)
    sizes = np.asarray(radii, dtype=float)
    if centers.ndim != 2 or centers.shape[1:] != (3,):
        raise ValueError("positions must have shape (n, 3).")
    if sizes.ndim != 1 or sizes.shape[0] != centers.shape[0]:
        raise ValueError("radii must have shape (n,) matching positions.")
    if centers.shape[0] == 0:
        raise ValueError("at least one sphere is required.")
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sizes)):
        raise ValueError("positions and radii must be finite.")
    if np.any(sizes <= 0.0):
        raise ValueError("radii must be positive.")

    scale = max(1.0, float(np.max(sizes)))
    duplicate_tolerance = 32.0 * np.finfo(float).eps * scale
    for first, second in itertools.combinations(range(len(sizes)), 2):
        if (
            abs(sizes[first] - sizes[second]) <= duplicate_tolerance
            and np.linalg.norm(centers[first] - centers[second]) <= duplicate_tolerance
        ):
            raise ValueError(
                "duplicate sphere geometry has no unique per-sphere center gradient."
            )
    return centers, sizes


def _positive_finite_float(name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite positive number, not boolean.")
    try:
        converted = float(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a finite positive number.") from exc
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite positive number.") from exc
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return converted


def _geometry_diagnostics(
    centers: np.ndarray, radii: np.ndarray, tolerance: float
) -> dict[str, int]:
    counts = {
        "disjoint_pairs": 0,
        "contained_pairs": 0,
        "external_tangent_pairs": 0,
        "internal_tangent_pairs": 0,
        "intersecting_pairs": 0,
    }
    for first, second in itertools.combinations(range(len(radii)), 2):
        distance = float(np.linalg.norm(centers[second] - centers[first]))
        outer = radii[first] + radii[second]
        inner = abs(radii[first] - radii[second])
        if abs(distance - outer) <= tolerance:
            counts["external_tangent_pairs"] += 1
        elif abs(distance - inner) <= tolerance:
            counts["internal_tangent_pairs"] += 1
        elif distance > outer:
            counts["disjoint_pairs"] += 1
        elif distance < inner:
            counts["contained_pairs"] += 1
        else:
            counts["intersecting_pairs"] += 1
    return counts


def _strictly_buried(
    point: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    excluded: tuple[int, ...],
    tolerance: float,
) -> bool:
    for index in range(len(radii)):
        if index in excluded:
            continue
        if np.linalg.norm(point - centers[index]) < radii[index] - tolerance:
            return True
    return False


def _pair_breakpoints(
    centers: np.ndarray,
    radii: np.ndarray,
    tolerance: float,
    *,
    exposed_only: bool = False,
) -> list[float]:
    points: list[float] = []
    for first, second in itertools.combinations(range(len(radii)), 2):
        displacement = centers[second] - centers[first]
        distance = float(np.linalg.norm(displacement))
        if distance <= tolerance:
            continue
        if distance > radii[first] + radii[second] + tolerance:
            continue
        if distance < abs(radii[first] - radii[second]) - tolerance:
            continue
        direction = displacement / distance
        along = (radii[first] ** 2 - radii[second] ** 2 + distance**2) / (
            2.0 * distance
        )
        circle_center = centers[first] + along * direction
        circle_radius_squared = max(0.0, radii[first] ** 2 - along**2)
        circle_radius = math.sqrt(circle_radius_squared)
        z_direction = np.array([0.0, 0.0, 1.0]) - direction[2] * direction
        z_direction_norm = float(np.linalg.norm(z_direction))
        if z_direction_norm <= tolerance:
            # The intersection circle is horizontal.  Its single z level must
            # remain because burial can vary around the circle.
            points.append(float(circle_center[2]))
            continue
        z_direction /= z_direction_norm
        for sign in (-1.0, 1.0):
            extremum = circle_center + sign * circle_radius * z_direction
            if exposed_only and _strictly_buried(
                extremum,
                centers,
                radii,
                (first, second),
                tolerance,
            ):
                continue
            points.append(float(extremum[2]))
    return points


def _triple_breakpoints(
    centers: np.ndarray,
    radii: np.ndarray,
    tolerance: float,
    *,
    exposed_only: bool = False,
) -> list[float]:
    points: list[float] = []
    for first, second, third in itertools.combinations(range(len(radii)), 3):
        base = centers[first]
        matrix = 2.0 * np.stack((centers[second] - base, centers[third] - base))
        if np.linalg.matrix_rank(matrix, tol=tolerance) < 2:
            continue
        rhs = np.array(
            [
                radii[first] ** 2
                - radii[index] ** 2
                + np.dot(centers[index], centers[index])
                - np.dot(base, base)
                for index in (second, third)
            ]
        )
        particular = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
        direction = np.cross(matrix[0], matrix[1])
        direction /= np.linalg.norm(direction)
        offset = particular - base
        linear = float(np.dot(direction, offset))
        constant = float(np.dot(offset, offset) - radii[first] ** 2)
        discriminant = linear**2 - constant
        if discriminant < -tolerance:
            continue
        root = math.sqrt(max(0.0, discriminant))
        for parameter in (-linear - root, -linear + root):
            intersection = particular + parameter * direction
            if exposed_only and _strictly_buried(
                intersection,
                centers,
                radii,
                (first, second, third),
                tolerance,
            ):
                continue
            points.append(float(intersection[2]))
    return points


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for start, stop in intervals[1:]:
        previous_start, previous_stop = merged[-1]
        if start <= previous_stop:
            merged[-1] = (previous_start, max(previous_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def _exposed_intervals(
    index: int,
    xy: np.ndarray,
    section_radii: np.ndarray,
    active: np.ndarray,
    tolerance: float,
) -> list[tuple[float, float]]:
    radius = section_radii[index]
    covered: list[tuple[float, float]] = []
    for other in active:
        if other == index:
            continue
        displacement = xy[other] - xy[index]
        distance = float(np.linalg.norm(displacement))
        other_radius = section_radii[other]
        if distance <= tolerance:
            if radius < other_radius - tolerance or (
                abs(radius - other_radius) <= tolerance and index > other
            ):
                return []
            continue
        if distance + radius <= other_radius + tolerance:
            return []
        if distance >= radius + other_radius - tolerance:
            continue
        if distance + other_radius <= radius + tolerance:
            continue
        cosine = (distance**2 + radius**2 - other_radius**2) / (2.0 * distance * radius)
        half_width = math.acos(float(np.clip(cosine, -1.0, 1.0)))
        middle = math.atan2(displacement[1], displacement[0]) % _TWO_PI
        start = middle - half_width
        stop = middle + half_width
        if start < 0.0:
            covered.extend(((start + _TWO_PI, _TWO_PI), (0.0, stop)))
        elif stop > _TWO_PI:
            covered.extend(((start, _TWO_PI), (0.0, stop - _TWO_PI)))
        else:
            covered.append((start, stop))

    merged = _merge_intervals(covered)
    exposed: list[tuple[float, float]] = []
    cursor = 0.0
    for start, stop in merged:
        if start > cursor:
            exposed.append((cursor, start))
        cursor = max(cursor, stop)
    if cursor < _TWO_PI:
        exposed.append((cursor, _TWO_PI))
    return exposed


def volume_and_gradient(
    positions,
    radii,
    *,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
    limit: int = 2000,
    prune_buried_breakpoints: bool = True,
) -> SphereUnionVolumeResult:
    """Return sharp sphere-union volume and derivatives w.r.t. sphere centers.

    ``gradient[i]`` is ``dV/dpositions[i]``.  Derivatives are evaluated by the
    boundary shape derivative, not by finite differences.  The functional is
    nonsmooth at duplicate sphere geometry, which is rejected explicitly.
    Breakpoints strictly inside another sphere can be omitted without changing
    the sharp union boundary; disabling that optimization is useful for audits.
    """

    centers, sizes = _validate_geometry(positions, radii)
    relative_tolerance = _positive_finite_float("rtol", rtol)
    absolute_tolerance = _positive_finite_float("atol", atol)
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)):
        raise TypeError("limit must be a positive integer, not boolean or fractional.")
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")
    subdivision_limit = int(limit)
    if not isinstance(prune_buried_breakpoints, (bool, np.bool_)):
        raise TypeError("prune_buried_breakpoints must be boolean.")

    # Centering limits cancellation in Green's-theorem area contributions.
    centered = centers - np.mean(centers, axis=0)
    scale = max(1.0, float(np.max(sizes)), float(np.ptp(centered, axis=0).max()))
    geometry_tolerance = 64.0 * np.finfo(float).eps * scale
    lower = float(np.min(centered[:, 2] - sizes))
    upper = float(np.max(centered[:, 2] + sizes))
    breakpoints = list(centered[:, 2] - sizes) + list(centered[:, 2] + sizes)
    breakpoints.extend(
        _pair_breakpoints(
            centered,
            sizes,
            geometry_tolerance,
            exposed_only=bool(prune_buried_breakpoints),
        )
    )
    breakpoints.extend(
        _triple_breakpoints(
            centered,
            sizes,
            geometry_tolerance,
            exposed_only=bool(prune_buried_breakpoints),
        )
    )
    points = sorted(
        {
            float(np.clip(point, lower, upper))
            for point in breakpoints
            if lower < point < upper
        }
    )

    sphere_count = len(sizes)
    evaluations = 0

    def integrand(z: float) -> np.ndarray:
        nonlocal evaluations
        evaluations += 1
        dz = z - centered[:, 2]
        squared = sizes**2 - dz**2
        active = np.flatnonzero(squared > 0.0)
        section_radii = np.zeros(sphere_count)
        section_radii[active] = np.sqrt(squared[active])
        values = np.zeros(1 + 4 * sphere_count)
        for index in active:
            radius = section_radii[index]
            arcs = _exposed_intervals(
                int(index),
                centered[:, :2],
                section_radii,
                active,
                geometry_tolerance,
            )
            for start, stop in arcs:
                sine_difference = math.sin(stop) - math.sin(start)
                cosine_difference = math.cos(start) - math.cos(stop)
                angle = stop - start
                values[0] += 0.5 * (
                    radius * centered[index, 0] * sine_difference
                    + radius * centered[index, 1] * cosine_difference
                    + radius**2 * angle
                )
                offset = 1 + 3 * index
                values[offset] += radius * sine_difference
                values[offset + 1] += radius * cosine_difference
                values[offset + 2] += dz[index] * angle
                values[1 + 3 * sphere_count + index] += sizes[index] * angle
        return values

    integral, error, quadrature_info = quad_vec(
        integrand,
        lower,
        upper,
        epsrel=relative_tolerance,
        epsabs=absolute_tolerance,
        limit=subdivision_limit,
        points=points,
        full_output=True,
    )
    integral = np.asarray(integral)
    error = float(error)
    if (
        not quadrature_info.success
        or not np.all(np.isfinite(integral))
        or not math.isfinite(error)
    ):
        raise RuntimeError(
            "sphere-union quadrature failed "
            f"(status {quadrature_info.status}): {quadrature_info.message}; "
            f"estimated error={error!r}."
        )
    volume = float(integral[0])
    gradient = np.asarray(integral[1 : 1 + 3 * sphere_count]).reshape((-1, 3))
    areas = np.asarray(integral[1 + 3 * sphere_count :])
    divergence_volume = float(
        (np.sum(centered * gradient) + np.dot(sizes, areas)) / 3.0
    )
    diagnostics: dict[str, float | int] = {
        **_geometry_diagnostics(centered, sizes, geometry_tolerance),
        "breakpoint_count": 2 * sphere_count + len(points),
        "evaluations": evaluations,
        "quadrature_status": int(quadrature_info.status),
        "quadrature_interval_count": len(quadrature_info.intervals),
        "buried_breakpoint_pruning": bool(prune_buried_breakpoints),
        "divergence_volume": divergence_volume,
        "divergence_volume_error": abs(divergence_volume - volume),
        "net_gradient_norm": float(np.linalg.norm(np.sum(gradient, axis=0))),
    }
    return SphereUnionVolumeResult(
        volume=volume,
        gradient=gradient,
        quadrature_error=error,
        diagnostics=diagnostics,
    )
