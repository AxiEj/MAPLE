"""SO(3)-covariant Coulomb single-layer matrices on spherical harmonics.

The unknown represented here is charge per unit solid angle.  In the
orthonormal real-harmonic convention, the self-sphere spectrum is therefore

``k_e * 4*pi / (a * (2*l + 1))``.

Cross-sphere blocks are valid for nested, intersecting, tangent, and separated
spheres.  The source-sphere harmonic is integrated analytically first, giving
its exact piecewise solid/irregular-harmonic potential.  Only the remaining
invariant polar coordinate is integrated numerically, split at the sphere
intersection.  The finite azimuthal contraction is exact for the retained
harmonic bandwidth.

No laboratory-fixed cavity grid or molecular body frame enters the
mathematical definition.  A local pair-axis section is used only to conjugate
an exact SO(2)-commuting canonical block; its arbitrary transverse gauge
cancels.  Coordinate derivatives are not exposed by this reference module.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math

import numpy as np
from ase.units import Bohr

from maple.solvation.api.units import HARTREE_TO_EV

from .harmonic_coefficients import (
    _bounded_lmax,
    _positive_int,
    _real_harmonic_design,
    real_wigner_matrix,
)

HARMONIC_SINGLE_LAYER_CONTRACT_ID = (
    "maple.route2.continuum.coulomb-single-layer-harmonic-shells.v1"
)
HARMONIC_SINGLE_LAYER_PROVIDER_ID = (
    "maple.route2.continuum.harmonic-single-layer.impl.v1"
)
HARMONIC_SPHERE_PAIR_TOPOLOGY_CONTRACT_ID = (
    "maple.route2.continuum.harmonic-sphere-pair-topology.v1"
)
HARMONIC_SPHERE_PAIR_TOPOLOGY_PROVIDER_ID = (
    "maple.route2.continuum.harmonic-sphere-pair-topology.impl.v1"
)
SPHERE_TANGENCY_EVENT_TOLERANCE_ANGSTROM = 1.0e-12
COULOMB_EV_ANGSTROM_PER_E2 = HARTREE_TO_EV * Bohr


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=float)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=32)
def _legendre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    count = _positive_int(order, name="radial_quadrature_order")
    if count > 4096:
        raise ValueError("radial_quadrature_order exceeds the bounded contract.")
    points, weights = np.polynomial.legendre.leggauss(count)
    return _readonly(points), _readonly(weights)


def _interval_rule(
    lower: float, upper: float, *, order: int
) -> tuple[np.ndarray, np.ndarray]:
    if upper <= lower:
        return np.empty(0), np.empty(0)
    points, weights = _legendre_rule(order)
    scale = 0.5 * (upper - lower)
    shift = 0.5 * (upper + lower)
    return scale * points + shift, scale * weights


def _harmonic_labels(lmax: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (ell, order) for ell in range(lmax + 1) for order in range(-ell, ell + 1)
    )


def _canonical_cross_inverse_angstrom(
    *,
    target_radius: float,
    source_radius: float,
    distance: float,
    lmax: int,
    radial_quadrature_order: int,
) -> np.ndarray:
    """Return the pair-axis block before the Coulomb unit conversion."""

    dimension = (lmax + 1) ** 2
    intersection_cosine = (source_radius**2 - distance**2 - target_radius**2) / (
        2.0 * distance * target_radius
    )
    if -1.0 < intersection_cosine < 1.0:
        intervals = ((-1.0, intersection_cosine), (intersection_cosine, 1.0))
    else:
        intervals = ((-1.0, 1.0),)

    azimuthal_count = max(1, 2 * lmax + 1)
    azimuth = 2.0 * np.pi * np.arange(azimuthal_count) / azimuthal_count
    azimuthal_weight = 2.0 * np.pi / azimuthal_count
    cosine_phi = np.cos(azimuth)
    sine_phi = np.sin(azimuth)
    labels = _harmonic_labels(lmax)
    result = np.zeros((dimension, dimension), dtype=float)

    for lower, upper in intervals:
        cosine, polar_weights = _interval_rule(
            lower, upper, order=radial_quadrature_order
        )
        sine = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
        directions = np.stack(
            (
                np.repeat(sine, azimuthal_count) * np.tile(cosine_phi, len(cosine)),
                np.repeat(sine, azimuthal_count) * np.tile(sine_phi, len(cosine)),
                np.repeat(cosine, azimuthal_count),
            ),
            axis=1,
        )
        target_points = target_radius * directions
        target_points[:, 2] += distance
        radial_distance = np.linalg.norm(target_points, axis=1)
        if np.any(radial_distance <= 0.0):
            raise RuntimeError("pair-axis quadrature reached an undefined direction.")
        source_directions = target_points / radial_distance[:, None]
        target_design = _real_harmonic_design(directions, lmax=lmax)
        source_design = _real_harmonic_design(source_directions, lmax=lmax)

        potential_scale = np.empty((len(radial_distance), dimension), dtype=float)
        for column, (ell, _) in enumerate(labels):
            inside = radial_distance < source_radius
            radial = np.where(
                inside,
                radial_distance**ell / source_radius ** (ell + 1),
                source_radius**ell / radial_distance ** (ell + 1),
            )
            potential_scale[:, column] = 4.0 * np.pi * radial / (2 * ell + 1)
        point_weights = np.repeat(polar_weights * azimuthal_weight, azimuthal_count)
        result += target_design.T @ (
            point_weights[:, None] * source_design * potential_scale
        )

    # The exact canonical block commutes with every z-axis rotation.  This
    # finite group contraction is exact on the retained representation because
    # matrix elements contain azimuthal frequencies no larger than 2*lmax.
    projected = np.zeros_like(result)
    stabilizer_order = max(1, 2 * lmax + 1)
    for index in range(stabilizer_order):
        angle = 2.0 * np.pi * index / stabilizer_order
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        representation = real_wigner_matrix(rotation, lmax=lmax)
        projected += representation.T @ result @ representation / stabilizer_order
    stabilizer_defect = float(np.linalg.norm(projected - result))
    stabilizer_scale = max(float(np.linalg.norm(result)), 1.0)
    if stabilizer_defect > 5.0e-12 * stabilizer_scale:
        raise RuntimeError(
            "finite azimuthal contraction violated the exact SO(2) "
            "pair-axis selection rule."
        )
    return projected


def canonical_harmonic_cross_block(
    *,
    target_radius_angstrom: object,
    source_radius_angstrom: object,
    distance_angstrom: object,
    lmax: int,
    radial_quadrature_order: int = 128,
) -> np.ndarray:
    """Return one source-to-target block with the pair direction along ``+z``.

    The result has units eV/e^2 for coefficients representing charge per unit
    solid angle.  It is not restricted to non-overlapping spheres.
    """

    target_radius = _positive_float(
        target_radius_angstrom, name="target_radius_angstrom"
    )
    source_radius = _positive_float(
        source_radius_angstrom, name="source_radius_angstrom"
    )
    distance = _positive_float(distance_angstrom, name="distance_angstrom")
    maximum = _bounded_lmax(lmax)
    order = _positive_int(radial_quadrature_order, name="radial_quadrature_order")
    if order > 4096:
        raise ValueError("radial_quadrature_order exceeds the bounded contract.")
    if distance <= 1.0e-12:
        raise ValueError("distance_angstrom must separate distinct sphere centres.")
    return _readonly(
        COULOMB_EV_ANGSTROM_PER_E2
        * _canonical_cross_inverse_angstrom(
            target_radius=target_radius,
            source_radius=source_radius,
            distance=distance,
            lmax=maximum,
            radial_quadrature_order=order,
        )
    )


def _rotation_from_positive_z(direction: np.ndarray) -> np.ndarray:
    """Return one proper-rotation section mapping ``+z`` to ``direction``.

    The section itself is not a molecular frame.  Any other valid section
    differs by an SO(2) stabilizer action that cancels against the canonical
    block.
    """

    unit = np.asarray(direction, dtype=float)
    unit = unit / np.linalg.norm(unit)
    cosine = float(unit[2])
    if cosine >= 1.0 - 1.0e-14:
        return np.eye(3)
    if cosine <= -1.0 + 1.0e-14:
        return np.diag([1.0, -1.0, -1.0])
    axis = np.asarray([-unit[1], unit[0], 0.0])
    sine = float(np.linalg.norm(axis))
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + cross + cross @ cross * ((1.0 - cosine) / sine**2)


def _geometry(
    positions_angstrom: object, radii_angstrom: object
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions_angstrom, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[0] < 1
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError(
            "positions_angstrom must be finite with shape (sphere_count, 3)."
        )
    radii = np.asarray(radii_angstrom, dtype=float)
    if (
        radii.shape != (len(positions),)
        or not np.all(np.isfinite(radii))
        or np.any(radii <= 0.0)
    ):
        raise ValueError(
            "radii_angstrom must be finite and positive with shape " "(sphere_count,)."
        )
    return np.ascontiguousarray(positions), np.ascontiguousarray(radii)


@dataclass(frozen=True, slots=True)
class HarmonicSpherePairTopology:
    """Rigid-motion-invariant sphere-pair stratum identity.

    ``relations`` contains each unordered sphere pair once.  The margin is
    the minimum distance, in coordinate space, to either the internal or
    external tangency surface.  It is ``None`` for a one-sphere system.
    """

    topology_sha256: str
    relations: tuple[tuple[int, int, str], ...]
    minimum_tangency_margin_angstrom: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_id": HARMONIC_SPHERE_PAIR_TOPOLOGY_CONTRACT_ID,
            "topology_sha256": self.topology_sha256,
            "relations": [list(item) for item in self.relations],
            "minimum_tangency_margin_angstrom": (self.minimum_tangency_margin_angstrom),
        }


def harmonic_sphere_pair_topology(
    positions_angstrom: object,
    radii_angstrom: object,
) -> HarmonicSpherePairTopology:
    """Classify every unordered sphere pair and fail on tangency events."""

    positions, radii = _geometry(positions_angstrom, radii_angstrom)
    relations: list[tuple[int, int, str]] = []
    margins: list[float] = []
    for first in range(len(positions)):
        for second in range(first + 1, len(positions)):
            distance = float(np.linalg.norm(positions[second] - positions[first]))
            if not np.isfinite(distance) or distance <= 1.0e-12:
                raise ValueError(
                    "sphere centres must be distinct for harmonic sphere-pair "
                    "topology."
                )
            internal_tangency = abs(float(radii[first] - radii[second]))
            external_tangency = float(radii[first] + radii[second])
            internal_margin = abs(distance - internal_tangency)
            external_margin = abs(distance - external_tangency)
            margin = min(internal_margin, external_margin)
            if margin <= SPHERE_TANGENCY_EVENT_TOLERANCE_ANGSTROM:
                raise ValueError("sphere pair lies on a tangency event surface.")
            if distance < internal_tangency:
                relation = "nested"
            elif distance < external_tangency:
                relation = "intersecting"
            else:
                relation = "separated"
            relations.append((first, second, relation))
            margins.append(margin)
    payload = {
        "contract_id": HARMONIC_SPHERE_PAIR_TOPOLOGY_CONTRACT_ID,
        "provider_id": HARMONIC_SPHERE_PAIR_TOPOLOGY_PROVIDER_ID,
        "radii_angstrom": tuple(float(value) for value in radii),
        "relations": relations,
    }
    return HarmonicSpherePairTopology(
        topology_sha256=_sha(payload),
        relations=tuple(relations),
        minimum_tangency_margin_angstrom=min(margins) if margins else None,
    )


def harmonic_single_layer_operator(
    *,
    positions_angstrom: object,
    radii_angstrom: object,
    lmax: int,
    radial_quadrature_order: int = 128,
) -> np.ndarray:
    """Assemble the dense Coulomb single-layer operator in eV/e^2.

    Cross blocks are computed once and inserted with their exact transpose;
    eigenvalues are checked but never clipped or regularized.
    """

    positions, radii = _geometry(positions_angstrom, radii_angstrom)
    maximum = _bounded_lmax(lmax)
    order = _positive_int(radial_quadrature_order, name="radial_quadrature_order")
    if order > 4096:
        raise ValueError("radial_quadrature_order exceeds the bounded contract.")
    block_dimension = (maximum + 1) ** 2
    operator = np.zeros(
        (len(positions) * block_dimension, len(positions) * block_dimension),
        dtype=float,
    )

    for atom, radius in enumerate(radii):
        offset = atom * block_dimension
        for ell in range(maximum + 1):
            section = slice(offset + ell * ell, offset + (ell + 1) * (ell + 1))
            eigenvalue = (
                COULOMB_EV_ANGSTROM_PER_E2
                * 4.0
                * np.pi
                / (float(radius) * (2 * ell + 1))
            )
            operator[section, section] = eigenvalue * np.eye(2 * ell + 1)

    for target in range(len(positions)):
        target_section = slice(target * block_dimension, (target + 1) * block_dimension)
        for source in range(target + 1, len(positions)):
            source_section = slice(
                source * block_dimension, (source + 1) * block_dimension
            )
            displacement = positions[target] - positions[source]
            distance = float(np.linalg.norm(displacement))
            if distance <= 1.0e-12:
                raise ValueError(
                    "sphere centres must be distinct for harmonic single-layer "
                    "assembly."
                )
            canonical = canonical_harmonic_cross_block(
                target_radius_angstrom=float(radii[target]),
                source_radius_angstrom=float(radii[source]),
                distance_angstrom=distance,
                lmax=maximum,
                radial_quadrature_order=order,
            )
            pair_rotation = _rotation_from_positive_z(displacement / distance)
            representation = real_wigner_matrix(pair_rotation, lmax=maximum)
            block = representation @ canonical @ representation.T
            operator[target_section, source_section] = block
            operator[source_section, target_section] = block.T

    eigenvalues = np.linalg.eigvalsh(operator)
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0.0:
        raise ValueError(
            "harmonic Coulomb single-layer operator is not strictly positive "
            "definite; coincident/redundant surfaces or insufficient invariant "
            "quadrature are not admissible."
        )
    return _readonly(operator)


__all__ = [
    "COULOMB_EV_ANGSTROM_PER_E2",
    "HARMONIC_SPHERE_PAIR_TOPOLOGY_CONTRACT_ID",
    "HARMONIC_SPHERE_PAIR_TOPOLOGY_PROVIDER_ID",
    "HARMONIC_SINGLE_LAYER_CONTRACT_ID",
    "HARMONIC_SINGLE_LAYER_PROVIDER_ID",
    "SPHERE_TANGENCY_EVENT_TOLERANCE_ANGSTROM",
    "HarmonicSpherePairTopology",
    "canonical_harmonic_cross_block",
    "harmonic_sphere_pair_topology",
    "harmonic_single_layer_operator",
]
