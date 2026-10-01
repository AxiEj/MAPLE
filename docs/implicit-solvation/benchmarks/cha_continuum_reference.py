"""Independent NumPy/SciPy reference for the continuum CHA/PBSA foundation.

This module is validation code, not a runtime solvent provider.  It deliberately
does not import the Torch SES, chart, R6, or CHA scalar implementations.  The R6
surface is constructed from raw centers and radii using atomic contact spheres
and complete rolling-probe tori.  The admitted geometry is intentionally narrow:
covered cap boundaries on any atom must be disjoint or nested and every pair
probe circle must be wholly exposed or wholly occluded by every third expanded
sphere.  Other arrangements fail closed rather than being approximated.

``quad_vec`` supplies an adaptive meridional error estimate.  Fixed azimuthal
Gauss--Legendre orders 64, 96, and 128 are all retained so their disagreement is
visible.  These estimates are diagnostics, not rigorous error bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import math
import struct
from typing import Iterable

import numpy as np
from scipy.integrate import quad, quad_vec

from maple.function.calculator.extra_correction.implicit.sphere_union_dispersion import (
    dispersion_energy_and_gradient,
)
from maple.function.calculator.extra_correction.implicit.sphere_union_volume import (
    volume_and_gradient,
)

PROBE_ANGSTROM = 0.88
CAVITY_PROBE_ANGSTROM = 1.3
CAVITY_COEFFICIENT_KCAL_MOL_ANGSTROM3 = 0.0378
CAVITY_OFFSET_KCAL_MOL = -0.5692
DISPERSION_PROBE_ANGSTROM = 0.557
WATER_OXYGEN_RMIN_ANGSTROM = 1.7683
WATER_OXYGEN_EPSILON_KCAL_MOL = 0.1520
WATER_DENSITY_PER_ANGSTROM3 = 0.03333 * 1.129

AMBER_CHARGE_SCALE = 18.2223
ALPB_ALPHA = 0.571412
CHA_TAU = 1.47
CHA_ROH_ANGSTROM = 0.586
SIZE_SWITCH_ANGSTROM = 10.0
_SHIFT_SLOPE = struct.unpack("f", struct.pack("f", 0.0015))[0]
_SHIFT_INTERCEPT = struct.unpack("f", struct.pack("f", 0.01))[0]
_TWO_TO_NEGATIVE_ONE_SIXTH = 2.0 ** (-1.0 / 6.0)
_GEOMETRY_TOLERANCE = 2.0e-12


class UnsupportedR6Geometry(ValueError):
    """Raised when the independent reference's proved geometry domain is left."""


@dataclass(frozen=True)
class R6RefinementLevel:
    azimuth_order: int
    inverse_cube_per_angstrom3: np.ndarray
    inverse_born_per_angstrom: np.ndarray
    gauss_closure_vector_angstrom2: np.ndarray
    inverse_cube_quad_error_estimate_per_angstrom3: float
    meridian_evaluations: int


@dataclass(frozen=True)
class R6ReferenceResult:
    inverse_cube_per_angstrom3: np.ndarray
    inverse_born_per_angstrom: np.ndarray
    gauss_closure_vector_angstrom2: np.ndarray
    levels: tuple[R6RefinementLevel, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class ChaContinuumReferenceResult:
    polar_kcal_mol: float
    cavity_kcal_mol: float
    dispersion_kcal_mol: float
    total_kcal_mol: float
    cavity_volume_angstrom3: float
    inverse_cube_per_angstrom3: np.ndarray
    inverse_born_per_angstrom: np.ndarray
    shifted_inverse_born_per_angstrom: np.ndarray
    born_radii_angstrom: np.ndarray
    effective_charges_e: np.ndarray
    cha_factors: np.ndarray
    r6: R6ReferenceResult
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _Cap:
    axis: np.ndarray
    cosine: float
    neighbor: int


@dataclass(frozen=True)
class _Torus:
    first: int
    second: int
    axis: np.ndarray
    circle_center: np.ndarray
    circle_radius: float
    first_angle: float
    second_angle: float


@dataclass(frozen=True)
class _ChaPolarState:
    polar_kcal_mol: float
    shifted_inverse_born_per_angstrom: np.ndarray
    born_radii_angstrom: np.ndarray
    effective_charges_e: np.ndarray
    electrostatic_size_angstrom: float
    cha_factors: np.ndarray


def _geometry(positions, radii) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(positions)
    sizes = np.asarray(radii)
    if centers.dtype == np.bool_ or sizes.dtype == np.bool_:
        raise TypeError("R6 positions and radii must be real, not boolean.")
    try:
        centers = centers.astype(float, copy=False)
        sizes = sizes.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("R6 positions and radii must be real arrays.") from exc
    if centers.ndim != 2 or centers.shape[1:] != (3,) or len(centers) == 0:
        raise ValueError("R6 positions must have nonempty shape (N, 3).")
    if sizes.shape != (len(centers),):
        raise ValueError("R6 radii must have shape (N,).")
    if len(centers) > 3:
        raise UnsupportedR6Geometry(
            "the independent exterior proof covers at most three sites."
        )
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sizes)):
        raise ValueError("R6 geometry must be finite.")
    if np.any(sizes <= 0.0):
        raise ValueError("R6 radii must be positive.")
    if len(centers) > 1:
        differences = centers[:, None, :] - centers[None, :, :]
        distances = np.linalg.norm(differences, axis=-1)
        distances[np.diag_indices(len(centers))] = math.inf
        if np.any(distances <= _GEOMETRY_TOLERANCE):
            raise ValueError("R6 atom centers must be distinct.")
    return centers, sizes


def _positive_float(name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive real number.")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number.")
    return number


def _orders(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(islice(iter(values), 4))
    if len(result) > 3:
        raise ValueError("azimuth_orders may contain at most three refinement levels.")
    if not result:
        raise ValueError("azimuth_orders must contain integers at least 8.")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 8
        for value in result
    ):
        raise ValueError("azimuth_orders must contain integers at least 8.")
    if tuple(sorted(set(result))) != result:
        raise ValueError("azimuth_orders must be unique and strictly increasing.")
    if max(result) > 128:
        raise ValueError("public oracle azimuthal order is limited to at most 128.")
    return result


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial = np.zeros(3)
    trial[int(np.argmin(np.abs(axis)))] = 1.0
    first = np.cross(axis, trial)
    first /= np.linalg.norm(first)
    return first, np.cross(axis, first)


def _cap_relation(first: _Cap, second: _Cap) -> str:
    first_radius = math.acos(float(np.clip(first.cosine, -1.0, 1.0)))
    second_radius = math.acos(float(np.clip(second.cosine, -1.0, 1.0)))
    separation = math.acos(float(np.clip(np.dot(first.axis, second.axis), -1.0, 1.0)))
    tolerance = 5.0e-11
    if separation + first_radius <= second_radius + tolerance:
        return "first-in-second"
    if separation + second_radius <= first_radius + tolerance:
        return "second-in-first"
    if separation >= first_radius + second_radius - tolerance:
        return "disjoint"
    return "intersecting"


def _maximal_caps(caps: list[_Cap], source: int) -> list[_Cap]:
    keep = [True] * len(caps)
    for first in range(len(caps)):
        for second in range(first + 1, len(caps)):
            relation = _cap_relation(caps[first], caps[second])
            if relation == "first-in-second":
                keep[first] = False
            elif relation == "second-in-first":
                keep[second] = False
            elif relation != "disjoint":
                raise UnsupportedR6Geometry(
                    f"covered cap boundaries intersect on site {source}; "
                    "the independent zero-triple oracle rejects this arrangement."
                )
    return [cap for cap, retained in zip(caps, keep, strict=True) if retained]


def _construct_surface(
    centers: np.ndarray, radii: np.ndarray, probe: float
) -> tuple[list[list[_Cap]], list[bool], list[_Torus], dict[str, int]]:
    expanded = radii + probe
    count = len(radii)
    caps: list[list[_Cap]] = [[] for _ in range(count)]
    fully_covered = [False] * count
    candidate_tori: list[_Torus] = []
    regular_pairs = 0
    contained_pairs = 0
    disjoint_pairs = 0

    for first in range(count):
        for second in range(first + 1, count):
            delta = centers[second] - centers[first]
            distance = float(np.linalg.norm(delta))
            axis = delta / distance
            outer = expanded[first] + expanded[second]
            inner = abs(expanded[first] - expanded[second])
            if (
                abs(distance - outer) <= _GEOMETRY_TOLERANCE
                or abs(distance - inner) <= _GEOMETRY_TOLERANCE
            ):
                raise UnsupportedR6Geometry("expanded-sphere tangency is unsupported.")
            if distance > outer:
                disjoint_pairs += 1
                continue
            if distance < inner:
                contained_pairs += 1
                if distance + expanded[first] < expanded[second]:
                    fully_covered[first] = True
                elif distance + expanded[second] < expanded[first]:
                    fully_covered[second] = True
                continue

            regular_pairs += 1
            along = (expanded[first] ** 2 - expanded[second] ** 2 + distance**2) / (
                2.0 * distance
            )
            circle_radius = math.sqrt(max(0.0, expanded[first] ** 2 - along**2))
            if circle_radius <= probe + _GEOMETRY_TOLERANCE:
                raise UnsupportedR6Geometry(
                    "pair probe-circle radius is not greater than the probe radius."
                )
            caps[first].append(_Cap(axis, along / expanded[first], second))
            caps[second].append(
                _Cap(-axis, (distance - along) / expanded[second], first)
            )
            candidate_tori.append(
                _Torus(
                    first=first,
                    second=second,
                    axis=axis,
                    circle_center=centers[first] + along * axis,
                    circle_radius=circle_radius,
                    first_angle=math.atan2(-along, circle_radius),
                    second_angle=math.atan2(distance - along, circle_radius),
                )
            )

    retained_caps = [
        [] if fully_covered[index] else _maximal_caps(site_caps, index)
        for index, site_caps in enumerate(caps)
    ]
    tori: list[_Torus] = []
    occluded_tori = 0
    for torus in candidate_tori:
        exposed = True
        for third in range(count):
            if third in (torus.first, torus.second):
                continue
            offset = centers[third] - torus.circle_center
            axial = float(np.dot(offset, torus.axis))
            perpendicular = math.sqrt(
                max(0.0, float(np.dot(offset, offset)) - axial**2)
            )
            minimum = math.hypot(axial, perpendicular - torus.circle_radius)
            maximum = math.hypot(axial, perpendicular + torus.circle_radius)
            if minimum >= expanded[third] + _GEOMETRY_TOLERANCE:
                continue
            if maximum <= expanded[third] - _GEOMETRY_TOLERANCE:
                exposed = False
                occluded_tori += 1
                break
            raise UnsupportedR6Geometry(
                f"pair probe circle ({torus.first}, {torus.second}) is only partly "
                f"occluded by expanded site {third}."
            )
        if exposed:
            tori.append(torus)

    minimum_tube_separation_margin = math.inf
    for first in range(len(tori)):
        for second in range(first + 1, len(tori)):
            delta = tori[second].circle_center - tori[first].circle_center
            distance = float(np.linalg.norm(delta))
            if distance <= _GEOMETRY_TOLERANCE:
                raise UnsupportedR6Geometry(
                    "active pair circles have coincident centers; tube separation fails."
                )
            direction = delta / distance
            first_extent = tori[first].circle_radius * math.sqrt(
                max(0.0, 1.0 - float(np.dot(tori[first].axis, direction)) ** 2)
            )
            second_extent = tori[second].circle_radius * math.sqrt(
                max(0.0, 1.0 - float(np.dot(tori[second].axis, direction)) ** 2)
            )
            margin = distance - first_extent - second_extent - 2.0 * probe
            minimum_tube_separation_margin = min(minimum_tube_separation_margin, margin)
            if margin <= _GEOMETRY_TOLERANCE:
                raise UnsupportedR6Geometry(
                    "active rolling-probe tube separation bound failed: "
                    f"margin={margin:.16g} angstrom."
                )

    diagnostics = {
        "regular_pair_count": regular_pairs,
        "disjoint_pair_count": disjoint_pairs,
        "contained_pair_count": contained_pairs,
        "active_torus_count": len(tori),
        "fully_occluded_torus_count": occluded_tori,
        "retained_covered_cap_count": sum(map(len, retained_caps)),
        "fully_covered_contact_sphere_count": sum(fully_covered),
        "minimum_active_tube_separation_margin_angstrom": (
            None
            if math.isinf(minimum_tube_separation_margin)
            else minimum_tube_separation_margin
        ),
    }
    return retained_caps, fully_covered, tori, diagnostics


def _integrate_sphere_patch(
    center: np.ndarray,
    radius: float,
    targets: np.ndarray,
    axis: np.ndarray,
    lower_cosine: float,
    phi_nodes: np.ndarray,
    phi_weights: np.ndarray,
    *,
    epsabs: float,
    epsrel: float,
) -> tuple[np.ndarray, float, int]:
    first, second = _basis(axis)
    phi = math.pi * (phi_nodes + 1.0)
    phi_weight = math.pi * phi_weights
    cosine_phi = np.cos(phi)
    sine_phi = np.sin(phi)
    evaluations = 0

    def meridian(cosine_theta: float) -> np.ndarray:
        nonlocal evaluations
        evaluations += 1
        sine_theta = math.sqrt(max(0.0, 1.0 - cosine_theta**2))
        normals = cosine_theta * axis[None, :] + sine_theta * (
            cosine_phi[:, None] * first[None, :] + sine_phi[:, None] * second[None, :]
        )
        surface = center[None, :] + radius * normals
        displacement = surface[:, None, :] - targets[None, :, :]
        squared = np.einsum("ptk,ptk->pt", displacement, displacement)
        if np.any(squared <= 1.0e-28):
            raise UnsupportedR6Geometry("an R6 surface passes through an atom center.")
        normal_dot = np.einsum("pk,ptk->pt", normals, displacement)
        flux = (
            radius**2
            * np.sum(phi_weight[:, None] * normal_dot / squared**3, axis=0)
            / (4.0 * math.pi)
        )
        return flux

    value, error = quad_vec(
        meridian,
        lower_cosine,
        1.0,
        epsabs=epsabs,
        epsrel=epsrel,
        quadrature="gk21",
    )
    return np.asarray(value), float(error), evaluations


def _integrate_torus(
    torus: _Torus,
    targets: np.ndarray,
    probe: float,
    phi_nodes: np.ndarray,
    phi_weights: np.ndarray,
    *,
    epsabs: float,
    epsrel: float,
) -> tuple[np.ndarray, float, int]:
    first, second = _basis(torus.axis)
    phi = math.pi * (phi_nodes + 1.0)
    phi_weight = math.pi * phi_weights
    radial_direction = (
        np.cos(phi)[:, None] * first[None, :] + np.sin(phi)[:, None] * second[None, :]
    )
    evaluations = 0

    def meridian(angle: float) -> np.ndarray:
        nonlocal evaluations
        evaluations += 1
        radial = torus.circle_radius - probe * math.cos(angle)
        if radial <= 0.0:
            raise UnsupportedR6Geometry("self-intersecting torus entered quadrature.")
        surface = (
            torus.circle_center[None, :]
            + probe * math.sin(angle) * torus.axis[None, :]
            + radial * radial_direction
        )
        normals = (
            -math.sin(angle) * torus.axis[None, :] + math.cos(angle) * radial_direction
        )
        displacement = surface[:, None, :] - targets[None, :, :]
        squared = np.einsum("ptk,ptk->pt", displacement, displacement)
        if np.any(squared <= 1.0e-28):
            raise UnsupportedR6Geometry("an R6 torus passes through an atom center.")
        normal_dot = np.einsum("pk,ptk->pt", normals, displacement)
        area_factor = probe * radial
        flux = (
            area_factor
            * np.sum(phi_weight[:, None] * normal_dot / squared**3, axis=0)
            / (4.0 * math.pi)
        )
        return flux

    value, error = quad_vec(
        meridian,
        torus.first_angle,
        torus.second_angle,
        epsabs=epsabs,
        epsrel=epsrel,
        quadrature="gk21",
    )
    return np.asarray(value), float(error), evaluations


def r6_inverse_born_reference(
    positions_angstrom,
    intrinsic_radii_angstrom,
    *,
    probe_angstrom: float = PROBE_ANGSTROM,
    azimuth_orders: Iterable[int] = (64, 96, 128),
    epsabs: float = 1.0e-12,
    epsrel: float = 1.0e-12,
) -> R6ReferenceResult:
    """Evaluate inverse Born radii from an independently constructed R6 SES."""

    centers, radii = _geometry(positions_angstrom, intrinsic_radii_angstrom)
    probe = _positive_float("probe_angstrom", probe_angstrom)
    absolute_tolerance = _positive_float("epsabs", epsabs)
    relative_tolerance = _positive_float("epsrel", epsrel)
    orders = _orders(azimuth_orders)

    # This exact limit is both a canary and avoids assigning quadrature noise to
    # a surface whose R6 flux is known analytically.
    if len(centers) == 1:
        inverse_cube = radii**-3
        analytic_levels = tuple(
            R6RefinementLevel(
                order,
                inverse_cube.copy(),
                radii**-1,
                np.zeros(3),
                0.0,
                0,
            )
            for order in orders
        )
        return R6ReferenceResult(
            inverse_cube.copy(),
            radii**-1,
            np.zeros(3),
            analytic_levels,
            {
                "geometry_scope": "single-sphere-analytic",
                "error_estimates_are_rigorous_bounds": False,
                "active_torus_count": 0,
            },
        )

    caps, fully_covered, tori, geometry_diagnostics = _construct_surface(
        centers, radii, probe
    )
    levels: list[R6RefinementLevel] = []
    z_axis = np.array([0.0, 0.0, 1.0])
    for order in orders:
        phi_nodes, phi_weights = np.polynomial.legendre.leggauss(order)
        accumulated = np.zeros(len(centers) + 3)
        estimated_error = 0.0
        evaluations = 0
        for source in range(len(centers)):
            if fully_covered[source]:
                continue
            # Evaluate each target in a chart whose pole points toward that
            # target.  This avoids subtracting two poorly resolved, very large
            # numbers when an atom center lies just inside a contact sphere.
            # Covered caps still use their own local axes, independently of
            # the production global-z charts.
            for target in range(len(centers)):
                displacement = centers[target] - centers[source]
                distance = float(np.linalg.norm(displacement))
                target_cap = next(
                    (cap for cap in caps[source] if cap.neighbor == target), None
                )
                if target_cap is None:
                    base_axis = z_axis if distance == 0.0 else displacement / distance
                    base_lower = -1.0
                else:
                    # Integrate the complement of the target-facing cap
                    # directly.  This removes the near-singular full-minus-cap
                    # cancellation for an atom center just inside a contact
                    # sphere while preserving exactly the same surface.
                    base_axis = -target_cap.axis
                    base_lower = -target_cap.cosine
                value, error, count = _integrate_sphere_patch(
                    centers[source],
                    radii[source],
                    centers[target : target + 1],
                    base_axis,
                    base_lower,
                    phi_nodes,
                    phi_weights,
                    epsabs=absolute_tolerance,
                    epsrel=relative_tolerance,
                )
                accumulated[target] += value[0]
                estimated_error += error
                evaluations += count
                for cap in caps[source]:
                    if cap is target_cap:
                        continue
                    value, error, count = _integrate_sphere_patch(
                        centers[source],
                        radii[source],
                        centers[target : target + 1],
                        cap.axis,
                        cap.cosine,
                        phi_nodes,
                        phi_weights,
                        epsabs=absolute_tolerance,
                        epsrel=relative_tolerance,
                    )
                    accumulated[target] -= value[0]
                    estimated_error += error
                    evaluations += count
            for cap in caps[source]:
                accumulated[len(centers) :] -= (
                    math.pi * radii[source] ** 2 * (1.0 - cap.cosine**2) * cap.axis
                )
        for torus in tori:
            value, error, count = _integrate_torus(
                torus,
                centers,
                probe,
                phi_nodes,
                phi_weights,
                epsabs=absolute_tolerance,
                epsrel=relative_tolerance,
            )
            accumulated[: len(centers)] += value
            estimated_error += error
            evaluations += count
            start_value = (
                torus.circle_radius * math.cos(torus.first_angle)
                - 0.5 * probe * math.cos(torus.first_angle) ** 2
            )
            stop_value = (
                torus.circle_radius * math.cos(torus.second_angle)
                - 0.5 * probe * math.cos(torus.second_angle) ** 2
            )
            accumulated[len(centers) :] += (
                2.0 * math.pi * probe * (stop_value - start_value) * torus.axis
            )

        inverse_cube = accumulated[: len(centers)]
        if not np.all(np.isfinite(inverse_cube)) or np.any(inverse_cube <= 0.0):
            raise RuntimeError("R6 surface flux did not yield positive finite values.")
        levels.append(
            R6RefinementLevel(
                order,
                inverse_cube.copy(),
                np.cbrt(inverse_cube),
                accumulated[len(centers) :].copy(),
                estimated_error,
                evaluations,
            )
        )

    finest = levels[-1]
    refinement_differences = [
        float(
            np.max(
                np.abs(
                    levels[index + 1].inverse_cube_per_angstrom3
                    - levels[index].inverse_cube_per_angstrom3
                )
            )
        )
        for index in range(len(levels) - 1)
    ]
    diagnostics: dict[str, object] = {
        **geometry_diagnostics,
        "geometry_scope": "disjoint-or-nested-caps-complete-pair-tori",
        "error_estimates_are_rigorous_bounds": False,
        "azimuth_refinement_max_abs_inverse_cube": refinement_differences,
        "gauss_closure_norm_angstrom2": float(
            np.linalg.norm(finest.gauss_closure_vector_angstrom2)
        ),
    }
    return R6ReferenceResult(
        finest.inverse_cube_per_angstrom3.copy(),
        finest.inverse_born_per_angstrom.copy(),
        finest.gauss_closure_vector_angstrom2.copy(),
        tuple(levels),
        diagnostics,
    )


def _site_values(name: str, values, count: int, *, nonnegative=False) -> np.ndarray:
    result = np.asarray(values)
    if result.dtype == np.bool_:
        raise TypeError(f"{name} must be real, not boolean.")
    try:
        result = result.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real array.") from exc
    if result.shape != (count,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite array with shape ({count},).")
    if nonnegative:
        if np.any(result < 0.0):
            raise ValueError(f"{name} must be nonnegative.")
    elif np.any(result <= 0.0):
        raise ValueError(f"{name} must be positive.")
    return result


def _strict_containing_sphere(centers: np.ndarray, radii: np.ndarray) -> int | None:
    scale = max(1.0, float(np.max(radii)), float(np.ptp(centers, axis=0).max()))
    tolerance = 64.0 * np.finfo(float).eps * scale
    for candidate in range(len(radii)):
        if all(
            site == candidate
            or np.linalg.norm(centers[site] - centers[candidate]) + radii[site]
            < radii[candidate] - tolerance
            for site in range(len(radii))
        ):
            return candidate
    return None


def _contained_volume_scalar_reference(
    radius: float, *, epsabs: float, epsrel: float
) -> tuple[float, float]:
    value, error = quad(
        lambda distance: 4.0 * math.pi * distance**2,
        0.0,
        radius,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    return float(value), float(error)


def _contained_dispersion_scalar_reference(
    centers: np.ndarray,
    boundary_center: int,
    boundary_radius: float,
    sigma: np.ndarray,
    mixed_epsilon: np.ndarray,
    density: float,
    *,
    epsabs: float,
    epsrel: float,
) -> tuple[float, float]:
    total = 0.0
    estimated_error = 0.0

    def analytic_tail(site: int, lower: float) -> float:
        return (
            16.0
            * math.pi
            * density
            * mixed_epsilon[site]
            * (
                sigma[site] ** 12 / (9.0 * lower**9)
                - sigma[site] ** 6 / (3.0 * lower**3)
            )
        )

    for site in range(len(centers)):
        offset = float(np.linalg.norm(centers[site] - centers[boundary_center]))
        if offset == 0.0:
            value = analytic_tail(site, max(float(sigma[site]), boundary_radius))
            total += value
            estimated_error += 8.0 * np.finfo(float).eps * abs(value)
            continue

        lower = max(float(sigma[site]), boundary_radius - offset)
        upper = boundary_radius + offset
        if lower < upper:

            def radial_integrand(distance: float) -> float:
                potential = (
                    4.0
                    * mixed_epsilon[site]
                    * ((sigma[site] / distance) ** 12 - (sigma[site] / distance) ** 6)
                )
                solid_angle = (
                    2.0
                    * math.pi
                    * (
                        1.0
                        + (distance**2 + offset**2 - boundary_radius**2)
                        / (2.0 * distance * offset)
                    )
                )
                return density * potential * distance**2 * solid_angle

            middle, error = quad(
                radial_integrand,
                lower,
                upper,
                epsabs=epsabs,
                epsrel=epsrel,
            )
            total += float(middle)
            estimated_error += float(error)
        tail_lower = max(float(sigma[site]), upper)
        tail = analytic_tail(site, tail_lower)
        total += tail
        estimated_error += 8.0 * np.finfo(float).eps * abs(tail)
    return total, estimated_error


def _cha_polar(
    centers: np.ndarray,
    charges: np.ndarray,
    effective_radii: np.ndarray,
    inverse_born: np.ndarray,
) -> _ChaPolarState:
    weights = effective_radii**3
    mass = float(np.sum(weights))
    center = np.sum(weights[:, None] * centers, axis=0) / mass
    centered = centers - center
    second_moment = centered.T @ (weights[:, None] * centered)
    sphere_moment = (2.0 / 5.0) * float(np.sum(weights * effective_radii**2))
    inertia = (np.trace(second_moment) + sphere_moment) * np.eye(3) - second_moment
    determinant = float(np.linalg.det(inertia))
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise RuntimeError("CHA electrostatic inertia determinant is not positive.")
    electrostatic_size = math.sqrt(2.5 / mass) * determinant ** (1.0 / 6.0)
    shift = (
        0.0
        if electrostatic_size < SIZE_SWITCH_ANGSTROM
        else _SHIFT_SLOPE * electrostatic_size + _SHIFT_INTERCEPT
    )
    shifted_inverse = inverse_born + shift
    born = 1.0 / shifted_inverse
    differences = centers[:, None, :] - centers[None, :, :]
    distances_squared = np.einsum("ijk,ijk->ij", differences, differences)
    native_charges = charges * AMBER_CHARGE_SCALE
    born_products = born[:, None] * born[None, :]
    charge_weights = np.exp(-CHA_TAU * distances_squared / born_products)
    effective_native = charge_weights @ native_charges
    effective_charges = effective_native / AMBER_CHARGE_SCALE
    cha_factors = 1.0 + np.sign(effective_native) * CHA_ROH_ANGSTROM / (
        born + PROBE_ANGSTROM
    )
    if np.any(cha_factors <= 0.0):
        raise RuntimeError("CHA factors are not positive.")
    beta = ALPB_ALPHA / 78.5
    dielectric = (1.0 - 1.0 / 78.5) / (1.0 + beta)
    size_term = beta / electrostatic_size
    self_energy = (
        -0.5
        * dielectric
        * np.sum(native_charges**2 * (shifted_inverse / cha_factors + size_term))
    )
    first, second = np.triu_indices(len(centers), 1)
    pair_r2 = distances_squared[first, second]
    pair_born = born_products[first, second]
    radicand = (
        pair_r2
        + pair_born
        * np.exp(-0.25 * pair_r2 / pair_born)
        * cha_factors[first]
        * cha_factors[second]
    )
    pair_energy = -dielectric * np.sum(
        native_charges[first] * native_charges[second] * (radicand**-0.5 + size_term)
    )
    return _ChaPolarState(
        polar_kcal_mol=float(self_energy + pair_energy),
        shifted_inverse_born_per_angstrom=shifted_inverse,
        born_radii_angstrom=born,
        effective_charges_e=effective_charges,
        electrostatic_size_angstrom=electrostatic_size,
        cha_factors=cha_factors,
    )


def cha_continuum_reference(
    positions_angstrom,
    charges_e,
    intrinsic_radii_angstrom,
    lj_rmin_angstrom,
    lj_epsilon_kcal_mol,
    *,
    r6_probe_angstrom: float = PROBE_ANGSTROM,
    azimuth_orders: Iterable[int] = (64, 96, 128),
    epsabs: float = 1.0e-12,
    epsrel: float = 1.0e-12,
) -> ChaContinuumReferenceResult:
    """Assemble the frozen continuum CHA polar, PBSA-SAV and dispersion scalar.

    ``intrinsic_radii_angstrom`` is the CHA-remapped ``radi`` vector used by
    both the R6 SES and the electrostatic-size algebra.  It is not the original
    Bondi vector retained separately in topology provenance.
    """

    centers, radii = _geometry(positions_angstrom, intrinsic_radii_angstrom)
    count = len(centers)
    # Charges are signed; only their finiteness/shape is constrained.
    if np.asarray(charges_e).dtype == np.bool_:
        raise TypeError("charges_e must be real, not boolean.")
    charges = np.asarray(charges_e, dtype=float)
    if charges.shape != (count,) or not np.all(np.isfinite(charges)):
        raise ValueError(f"charges_e must be finite with shape ({count},).")
    rmin = _site_values("lj_rmin_angstrom", lj_rmin_angstrom, count)
    epsilon = _site_values(
        "lj_epsilon_kcal_mol", lj_epsilon_kcal_mol, count, nonnegative=True
    )

    r6 = r6_inverse_born_reference(
        centers,
        radii,
        probe_angstrom=r6_probe_angstrom,
        azimuth_orders=azimuth_orders,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    polar_state = _cha_polar(centers, charges, radii, r6.inverse_born_per_angstrom)

    cavity = volume_and_gradient(
        centers,
        rmin + CAVITY_PROBE_ANGSTROM,
        rtol=epsrel,
        atol=epsabs,
    )
    cavity_energy = (
        CAVITY_COEFFICIENT_KCAL_MOL_ANGSTROM3 * cavity.volume + CAVITY_OFFSET_KCAL_MOL
    )
    cavity_radii = rmin + CAVITY_PROBE_ANGSTROM
    cavity_container = _strict_containing_sphere(centers, cavity_radii)
    cavity_scalar_volume = None
    cavity_scalar_error = None
    cavity_scalar_delta = None
    if cavity_container is not None:
        cavity_scalar_volume, cavity_scalar_error = _contained_volume_scalar_reference(
            float(cavity_radii[cavity_container]),
            epsabs=epsabs,
            epsrel=epsrel,
        )
        cavity_scalar_delta = float(cavity.volume - cavity_scalar_volume)
    sigma = (rmin + WATER_OXYGEN_RMIN_ANGSTROM) * _TWO_TO_NEGATIVE_ONE_SIXTH
    mixed_epsilon = np.sqrt(epsilon * WATER_OXYGEN_EPSILON_KCAL_MOL)
    dispersion_radii = rmin + DISPERSION_PROBE_ANGSTROM
    dispersion = dispersion_energy_and_gradient(
        centers,
        dispersion_radii,
        sigma,
        mixed_epsilon,
        WATER_DENSITY_PER_ANGSTROM3,
        phi_order=max(_orders(azimuth_orders)),
        rtol=epsrel,
        atol=epsabs,
    )
    dispersion_container = _strict_containing_sphere(centers, dispersion_radii)
    dispersion_scalar_energy = None
    dispersion_scalar_error = None
    dispersion_scalar_delta = None
    if dispersion_container is not None:
        dispersion_scalar_energy, dispersion_scalar_error = (
            _contained_dispersion_scalar_reference(
                centers,
                dispersion_container,
                float(dispersion_radii[dispersion_container]),
                sigma,
                mixed_epsilon,
                WATER_DENSITY_PER_ANGSTROM3,
                epsabs=epsabs,
                epsrel=epsrel,
            )
        )
        dispersion_scalar_delta = float(dispersion.energy - dispersion_scalar_energy)
    total = polar_state.polar_kcal_mol + cavity_energy + dispersion.energy
    return ChaContinuumReferenceResult(
        polar_kcal_mol=polar_state.polar_kcal_mol,
        cavity_kcal_mol=float(cavity_energy),
        dispersion_kcal_mol=float(dispersion.energy),
        total_kcal_mol=float(total),
        cavity_volume_angstrom3=float(cavity.volume),
        inverse_cube_per_angstrom3=r6.inverse_cube_per_angstrom3.copy(),
        inverse_born_per_angstrom=r6.inverse_born_per_angstrom.copy(),
        shifted_inverse_born_per_angstrom=(
            polar_state.shifted_inverse_born_per_angstrom.copy()
        ),
        born_radii_angstrom=polar_state.born_radii_angstrom.copy(),
        effective_charges_e=polar_state.effective_charges_e.copy(),
        cha_factors=polar_state.cha_factors.copy(),
        r6=r6,
        diagnostics={
            "electrostatic_size_angstrom": polar_state.electrostatic_size_angstrom,
            "cavity_mixed_vector_quad_error": cavity.quadrature_error,
            "dispersion_mixed_vector_quad_error": dispersion.quadrature_error,
            "mixed_vector_quad_errors_are_scalar_uncertainties": False,
            "cavity_scalar_reference_available": cavity_container is not None,
            "cavity_scalar_reference_volume_angstrom3": cavity_scalar_volume,
            "cavity_scalar_error_estimate_angstrom3": cavity_scalar_error,
            "cavity_generic_minus_scalar_reference_angstrom3": cavity_scalar_delta,
            "dispersion_scalar_reference_available": (dispersion_container is not None),
            "dispersion_scalar_reference_energy_kcal_mol": (dispersion_scalar_energy),
            "dispersion_scalar_error_estimate_kcal_mol": dispersion_scalar_error,
            "dispersion_generic_minus_scalar_reference_kcal_mol": (
                dispersion_scalar_delta
            ),
            "error_estimates_are_rigorous_bounds": False,
            "production_geometry_helpers_imported": False,
        },
    )
