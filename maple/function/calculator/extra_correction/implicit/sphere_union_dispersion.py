"""Sharp sphere-union PBSA dispersion energy and coordinate gradient.

This module evaluates the sigma-decomposed Lennard-Jones boundary functional
on the exposed surface of a union of solvent-accessible spheres.  Surface arcs
are exact; fixed-order Gauss--Legendre quadrature is used in azimuth and
adaptive quadrature only along the slicing axis.  The returned derivative is
the coordinate gradient (force is its negative).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad_vec

from .sphere_union_volume import (
    _exposed_intervals,
    _pair_breakpoints,
    _positive_finite_float,
    _triple_breakpoints,
    _validate_geometry,
)

_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class SphereUnionDispersionResult:
    """Dispersion result in the caller's consistent length/energy units.

    ``quadrature_error`` is SciPy's adaptive z-quadrature estimate for the
    fixed ``phi_order`` calculation.  It does not estimate azimuthal error;
    callers must compare paired azimuthal orders to establish that convergence.
    """

    energy: float
    gradient: np.ndarray
    quadrature_error: float
    diagnostics: dict[str, float | int]


def _site_array(name: str, values, count: int, *, positive: bool) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if any(isinstance(value, (bool, np.bool_)) for value in raw.flat):
        raise TypeError(f"{name} must contain numeric values, not booleans.")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc
    if array.ndim != 1 or array.shape[0] != count:
        raise ValueError(f"{name} must have shape ({count},).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive.")
    if not positive and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    return array


def _positive_integer(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _radial_crossings(
    start: float,
    stop: float,
    circle_center: np.ndarray,
    circle_radius: float,
    z: float,
    site_center: np.ndarray,
    sigma: float,
    tolerance: float,
) -> list[float]:
    """Angles inside an arc where distance to a site equals ``sigma``."""

    xy_offset = circle_center[:2] - site_center[:2]
    amplitude = 2.0 * circle_radius * float(np.linalg.norm(xy_offset))
    if amplitude <= tolerance:
        return []
    constant = (
        float(np.dot(xy_offset, xy_offset))
        + circle_radius**2
        + (z - site_center[2]) ** 2
        - sigma**2
    )
    target = -constant / amplitude
    if target < -1.0 - tolerance or target > 1.0 + tolerance:
        return []
    phase = math.atan2(xy_offset[1], xy_offset[0])
    offset = math.acos(float(np.clip(target, -1.0, 1.0)))
    crossings: list[float] = []
    for root in (phase - offset, phase + offset):
        normalized = root % _TWO_PI
        if start + tolerance < normalized < stop - tolerance:
            crossings.append(normalized)
    return crossings


def _sigma_breakpoints(
    centers: np.ndarray,
    sas_radii: np.ndarray,
    sigmas: np.ndarray,
    tolerance: float,
) -> list[float]:
    """Relevant z extrema of SAS/sigma-sphere intersections.

    A sigma sphere changes only the radial potential branch; it never occludes
    the SAS.  An extremum can nevertheless be omitted when it lies strictly
    inside a third SAS ball, because that neighborhood is not exposed union
    boundary.
    """

    points: list[float] = []
    for source in range(len(sas_radii)):
        for site in range(len(sigmas)):
            displacement = centers[site] - centers[source]
            distance = float(np.linalg.norm(displacement))
            if distance <= tolerance:
                continue
            if distance > sas_radii[source] + sigmas[site] + tolerance:
                continue
            if distance < abs(sas_radii[source] - sigmas[site]) - tolerance:
                continue
            direction = displacement / distance
            along = (sas_radii[source] ** 2 - sigmas[site] ** 2 + distance**2) / (
                2.0 * distance
            )
            circle_center = centers[source] + along * direction
            circle_radius = math.sqrt(max(0.0, sas_radii[source] ** 2 - along**2))
            z_direction = np.array([0.0, 0.0, 1.0]) - direction[2] * direction
            z_direction_norm = float(np.linalg.norm(z_direction))
            if z_direction_norm <= tolerance:
                points.append(float(circle_center[2]))
                continue
            z_direction /= z_direction_norm
            for sign in (-1.0, 1.0):
                extremum = circle_center + sign * circle_radius * z_direction
                buried = any(
                    other != source
                    and np.linalg.norm(extremum - centers[other])
                    < sas_radii[other] - tolerance
                    for other in range(len(sas_radii))
                )
                if not buried:
                    points.append(float(extremum[2]))
    return points


def dispersion_energy_and_gradient(
    positions,
    sas_radii,
    sigma,
    epsilon,
    density,
    *,
    phi_order: int = 48,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
    limit: int = 2000,
) -> SphereUnionDispersionResult:
    """Evaluate the same-functional PBSA dispersion energy and gradient.

    ``sas_radii`` define the sharp solvent-accessible union surface.  ``sigma``
    and ``epsilon`` are the per-site Lennard-Jones parameters after any force-
    field preparation; that preparation is deliberately outside this routine.
    SAS radii are independent inputs and must not be replaced by radii from a
    distinct volume-cavity functional.  Every site/exposed-surface pair is
    included: no finite-cutoff correction is implemented, and no equivalence
    is claimed for geometries approaching Amber's effective 999.9-Angstrom
    ``cutnb=0`` distance.
    ``gradient[i]`` is ``dE/dpositions[i]`` and is not a force.
    """

    centers, radii = _validate_geometry(positions, sas_radii)
    count = len(radii)
    sigmas = _site_array("sigma", sigma, count, positive=True)
    epsilons = _site_array("epsilon", epsilon, count, positive=False)
    solvent_density = _positive_finite_float("density", density)
    relative_tolerance = _positive_finite_float("rtol", rtol)
    absolute_tolerance = _positive_finite_float("atol", atol)
    azimuth_order = _positive_integer("phi_order", phi_order)
    subdivision_limit = _positive_integer("limit", limit)

    # Centering improves translation covariance and reduces cancellation.
    centered = centers - np.mean(centers, axis=0)
    scale = max(
        1.0,
        float(np.max(radii)),
        float(np.max(sigmas)),
        float(np.ptp(centered, axis=0).max()),
    )
    geometry_tolerance = 64.0 * np.finfo(float).eps * scale
    lower = float(np.min(centered[:, 2] - radii))
    upper = float(np.max(centered[:, 2] + radii))
    breakpoints = list(centered[:, 2] - radii) + list(centered[:, 2] + radii)
    breakpoints.extend(
        _pair_breakpoints(
            centered,
            radii,
            geometry_tolerance,
            exposed_only=True,
        )
    )
    breakpoints.extend(
        _triple_breakpoints(
            centered,
            radii,
            geometry_tolerance,
            exposed_only=True,
        )
    )
    # The radial potential changes branch on each site's sigma sphere.  Sigma
    # spheres are branch loci only and are never included as SAS occluders.
    breakpoints.extend(_sigma_breakpoints(centered, radii, sigmas, geometry_tolerance))
    points = sorted(
        {
            float(np.clip(point, lower, upper))
            for point in breakpoints
            if lower < point < upper
        }
    )

    nodes, weights = np.polynomial.legendre.leggauss(azimuth_order)
    raw_b = 4.0 * epsilons * sigmas**6
    raw_a = raw_b * sigmas**6
    evaluations = 0
    azimuth_evaluations = 0

    def integrand(z: float) -> np.ndarray:
        nonlocal evaluations, azimuth_evaluations
        evaluations += 1
        dz = z - centered[:, 2]
        squared = radii**2 - dz**2
        active = np.flatnonzero(squared > 0.0)
        section_radii = np.zeros(count)
        section_radii[active] = np.sqrt(squared[active])
        values = np.zeros(1 + 3 * count)

        for source_value in active:
            source = int(source_value)
            circle_radius = float(section_radii[source])
            arcs = _exposed_intervals(
                source,
                centered[:, :2],
                section_radii,
                active,
                geometry_tolerance,
            )
            for start, stop in arcs:
                splits = [start, stop]
                for site in range(count):
                    splits.extend(
                        _radial_crossings(
                            start,
                            stop,
                            centered[source],
                            circle_radius,
                            z,
                            centered[site],
                            sigmas[site],
                            geometry_tolerance,
                        )
                    )
                splits = sorted(set(splits))
                for arc_start, arc_stop in itertools.pairwise(splits):
                    half_width = 0.5 * (arc_stop - arc_start)
                    middle = 0.5 * (arc_stop + arc_start)
                    angles = middle + half_width * nodes
                    arc_weights = half_width * weights
                    azimuth_evaluations += azimuth_order

                    cosines = np.cos(angles)
                    sines = np.sin(angles)
                    xyz = np.column_stack(
                        (
                            centered[source, 0] + circle_radius * cosines,
                            centered[source, 1] + circle_radius * sines,
                            np.full(azimuth_order, z),
                        )
                    )
                    normal = np.column_stack(
                        (
                            circle_radius * cosines / radii[source],
                            circle_radius * sines / radii[source],
                            np.full(azimuth_order, dz[source] / radii[source]),
                        )
                    )
                    surface_weights = radii[source] * arc_weights
                    displacement = xyz[:, None, :] - centered[None, :, :]
                    distances = np.linalg.norm(displacement, axis=2)
                    if np.any(distances <= geometry_tolerance):
                        raise RuntimeError(
                            "an integration node coincides with a dispersion site."
                        )

                    outside = distances >= sigmas[None, :]
                    inverse_six = np.zeros_like(distances)
                    inverse_six[outside] = distances[outside] ** -6
                    potentials = (
                        raw_a[None, :] * inverse_six**2 - raw_b[None, :] * inverse_six
                    )

                    radial = np.empty_like(distances)
                    radial[outside] = (
                        solvent_density
                        * (
                            -raw_b[None, :] * inverse_six / 3.0
                            + raw_a[None, :] * inverse_six**2 / 9.0
                        )[outside]
                    )
                    inside = ~outside
                    inside_coefficient = solvent_density * (
                        -raw_b / (3.0 * sigmas**3) + raw_a / (9.0 * sigmas**9)
                    )
                    radial[inside] = (inside_coefficient[None, :] / distances**3)[
                        inside
                    ]

                    normal_dot = np.einsum("qik,qk->qi", displacement, normal)
                    values[0] += float(
                        np.sum(surface_weights[:, None] * radial * normal_dot)
                    )
                    weighted_normals = surface_weights[:, None] * normal
                    values[1:] += (
                        solvent_density
                        * np.einsum("qi,qk->ik", potentials, weighted_normals)
                    ).reshape(-1)
                    values[1 + 3 * source : 1 + 3 * (source + 1)] -= (
                        solvent_density
                        * np.einsum(
                            "q,qk->k",
                            np.sum(potentials, axis=1),
                            weighted_normals,
                        )
                    )
        return values

    integral, error, info = quad_vec(
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
        not info.success
        or not np.all(np.isfinite(integral))
        or not math.isfinite(error)
    ):
        raise RuntimeError(
            "sphere-union dispersion quadrature failed "
            f"(status {info.status}): {info.message}; estimated error={error!r}."
        )
    gradient = np.asarray(integral[1:]).reshape((-1, 3))
    diagnostics: dict[str, float | int] = {
        "breakpoint_count": 2 * count + len(points),
        "evaluations": evaluations,
        "azimuth_evaluations": azimuth_evaluations,
        "azimuth_error_estimated": False,
        "phi_order": azimuth_order,
        "quadrature_status": int(info.status),
        "quadrature_interval_count": len(info.intervals),
        "net_gradient_norm": float(np.linalg.norm(np.sum(gradient, axis=0))),
    }
    return SphereUnionDispersionResult(
        energy=float(integral[0]),
        gradient=gradient,
        quadrature_error=error,
        diagnostics=diagnostics,
    )
