"""Differentiable coefficient-space primitives for harmonic Galerkin PCM.

All Torch imports are lazy. Geometry-dependent integrations are one-dimensional
in rotational invariants; finite sphere rules contract only declared finite
harmonic products. This module owns E(R), K(R), and V(R) assembly but no public
continuum scalar or capability declaration.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any

import numpy as np
from ase.units import Bohr, Hartree


from .harmonics import (
    _exact_bandlimited_sphere_rule,
    _real_harmonic_design,
    real_wigner_matrix,
)
HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE = 64
from .point_l1 import POINT_SOURCE_SHELL_EVENT_TOLERANCE_ANGSTROM
from .point_l1 import COULOMB_EV_ANGSTROM_PER_E2, harmonic_sphere_pair_topology


def _torch():
    return __import__("torch")


def _constant(reference: Any, values: object):
    return reference.new_tensor(np.asarray(values, dtype=float))


@lru_cache(maxsize=64)
def _legendre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    points, weights = np.polynomial.legendre.leggauss(order)
    points.setflags(write=False)
    weights.setflags(write=False)
    return points, weights


@lru_cache(maxsize=64)
def _legendre_design(order: int, lmax: int) -> tuple[np.ndarray, ...]:
    points, weights = _legendre_rule(order)
    design = np.polynomial.legendre.legvander(points, lmax)
    design.setflags(write=False)
    return points, weights, design


@lru_cache(maxsize=64)
def _finite_band_rule(required_degree: int) -> tuple[np.ndarray, ...]:
    if (
        required_degree < 0
        or required_degree > HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE
    ):
        raise ValueError("finite harmonic contraction exceeds its degree bound.")
    polar_order = required_degree // 2 + 1
    azimuthal_order = required_degree + 1
    cosine, polar_weights = np.polynomial.legendre.leggauss(polar_order)
    azimuth = 2.0 * np.pi * np.arange(azimuthal_order) / azimuthal_order
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
    directions = np.stack(
        (
            np.repeat(sine, azimuthal_order) * np.tile(np.cos(azimuth), polar_order),
            np.repeat(sine, azimuthal_order) * np.tile(np.sin(azimuth), polar_order),
            np.repeat(cosine, azimuthal_order),
        ),
        axis=1,
    )
    weights = np.repeat(
        polar_weights * (2.0 * np.pi / azimuthal_order), azimuthal_order
    )
    directions.setflags(write=False)
    weights.setflags(write=False)
    return directions, weights


@lru_cache(maxsize=256)
def _finite_band_design(
    required_degree: int, lmax: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache one exact rule and harmonic design for repeated contractions."""

    directions, weights = _finite_band_rule(required_degree)
    design = _real_harmonic_design(directions, lmax=lmax)
    design.setflags(write=False)
    return directions, weights, design


def _torch_real_harmonic_design(directions: Any, *, lmax: int):
    """Evaluate MAPLE's real harmonic basis with Cartesian recurrences."""

    torch = _torch()
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape (point_count, 3).")
    x, y, z = directions.unbind(-1)
    values: dict[tuple[int, int, str], Any] = {}
    complex_xy = torch.complex(x, y)
    for order in range(lmax + 1):
        if order == 0:
            cosine = torch.ones_like(z)
            sine = torch.zeros_like(z)
        else:
            sectoral = complex_xy**order
            double_factorial = math.prod(range(1, 2 * order, 2))
            phase = (-1) ** order * double_factorial
            cosine = phase * sectoral.real
            sine = phase * sectoral.imag
        cosine_sequence = {order: cosine}
        sine_sequence = {order: sine}
        if order < lmax:
            cosine_sequence[order + 1] = (2 * order + 1) * z * cosine
            sine_sequence[order + 1] = (2 * order + 1) * z * sine
        for ell in range(order + 2, lmax + 1):
            cosine_sequence[ell] = (
                (2 * ell - 1) * z * cosine_sequence[ell - 1]
                - (ell + order - 1) * cosine_sequence[ell - 2]
            ) / (ell - order)
            sine_sequence[ell] = (
                (2 * ell - 1) * z * sine_sequence[ell - 1]
                - (ell + order - 1) * sine_sequence[ell - 2]
            ) / (ell - order)
        for ell in range(order, lmax + 1):
            values[(ell, order, "c")] = cosine_sequence[ell]
            values[(ell, order, "s")] = sine_sequence[ell]

    columns = []
    for ell in range(lmax + 1):
        for signed_order in range(-ell, ell + 1):
            order = abs(signed_order)
            normalization = math.sqrt(
                (2 * ell + 1)
                * math.exp(math.lgamma(ell - order + 1) - math.lgamma(ell + order + 1))
                / (4.0 * np.pi)
            )
            if signed_order < 0:
                column = (
                    math.sqrt(2.0)
                    * ((-1) ** order)
                    * normalization
                    * values[(ell, order, "s")]
                )
            elif signed_order == 0:
                column = normalization * values[(ell, 0, "c")]
            else:
                column = (
                    math.sqrt(2.0)
                    * ((-1) ** order)
                    * normalization
                    * values[(ell, order, "c")]
                )
            columns.append(column)
    return torch.stack(columns, dim=-1)


def _torch_flat_step(values: Any):
    torch = _torch()
    interior = torch.abs(values) < 1.0
    denominator = torch.where(interior, 1.0 - values * values, torch.ones_like(values))
    transition = torch.sigmoid(2.0 * values / denominator)
    return torch.where(
        values <= -1.0,
        torch.zeros_like(values),
        torch.where(values >= 1.0, torch.ones_like(values), transition),
    )


def _constant_coefficients(reference: Any, *, lmax: int, value: float):
    result = reference.new_zeros(((lmax + 1) ** 2,))
    result[0] = value * math.sqrt(4.0 * np.pi)
    return result


def _pair_exposure_coefficients(
    displacement: Any,
    *,
    radius_i: float,
    radius_j: float,
    transition_width: float,
    lmax: int,
    radial_order: int,
):
    distance = _torch().linalg.vector_norm(displacement)
    distance_value = float(distance.detach().cpu())
    if not np.isfinite(distance_value) or distance_value <= 1.0e-12:
        raise ValueError("sphere centres must remain distinct.")
    z_minimum = (distance - radius_i) ** 2 - radius_j**2
    z_maximum = (distance + radius_i) ** 2 - radius_j**2
    if float(z_minimum.detach().cpu()) >= transition_width:
        return "exposed", _constant_coefficients(displacement, lmax=lmax, value=1.0)
    if float(z_maximum.detach().cpu()) <= -transition_width:
        return "buried", _constant_coefficients(displacement, lmax=lmax, value=0.0)

    nodes, weights, legendre = _legendre_design(radial_order, lmax)
    cosine = _constant(displacement, nodes)
    weight_tensor = _constant(displacement, weights)
    legendre_tensor = _constant(displacement, legendre)
    signed_overlap = (
        radius_i**2 + distance**2 - 2.0 * radius_i * distance * cosine - radius_j**2
    )
    exposure = _torch_flat_step(signed_overlap / transition_width)
    zonal = 2.0 * np.pi * ((weight_tensor * exposure) @ legendre_tensor)
    direction = displacement / distance
    harmonics = _torch_real_harmonic_design(direction[None, :], lmax=lmax)[0]
    return "transition", _torch().cat(
        [
            zonal[ell] * harmonics[ell * ell : (ell + 1) * (ell + 1)]
            for ell in range(lmax + 1)
        ]
    )


def _project_product(factors: tuple[Any, ...], *, lmax: int, reference: Any):
    if not factors:
        return _constant_coefficients(reference, lmax=lmax, value=1.0)
    required_degree = (len(factors) + 1) * lmax
    _directions, weights, raw_design = _finite_band_design(required_degree, lmax)
    design = _constant(reference, raw_design)
    values = reference.new_ones((len(_directions),))
    for factor in factors:
        values = values * (design @ factor)
    return design.T @ (_constant(reference, weights) * values)


def _project_product_with_leave_one_out(
    factors: tuple[Any, ...], *, lmax: int, reference: Any
):
    """Project one finite product and every leave-one-out product exactly."""

    if not factors:
        return _constant_coefficients(reference, lmax=lmax, value=1.0), ()
    required_degree = (len(factors) + 1) * lmax
    directions, weights, raw_design = _finite_band_design(required_degree, lmax)
    design = _constant(reference, raw_design)
    factor_values = _torch().stack(tuple(design @ factor for factor in factors))
    prefix = [reference.new_ones((len(directions),))]
    for values in factor_values:
        prefix.append(prefix[-1] * values)
    suffix = [None] * (len(factors) + 1)
    suffix[len(factors)] = reference.new_ones((len(directions),))
    for index in range(len(factors) - 1, -1, -1):
        suffix[index] = factor_values[index] * suffix[index + 1]
    leave_values = _torch().stack(
        tuple(prefix[index] * suffix[index + 1] for index in range(len(factors)))
    )
    weights_tensor = _constant(reference, weights)
    full = design.T @ (weights_tensor * prefix[-1])
    leave = (leave_values * weights_tensor[None, :]) @ design
    return full, tuple(leave[index] for index in range(len(factors)))


def _weighted_basis_block(coefficients: Any, *, exposure_lmax: int, basis_lmax: int):
    product_lmax = exposure_lmax + basis_lmax
    required_degree = exposure_lmax + basis_lmax + product_lmax
    _directions, weights, raw_exposure_design = _finite_band_design(
        required_degree, exposure_lmax
    )
    _directions, _weights, raw_basis_design = _finite_band_design(
        required_degree, basis_lmax
    )
    _directions, _weights, raw_product_design = _finite_band_design(
        required_degree, product_lmax
    )
    exposure_design = _constant(coefficients, raw_exposure_design)
    basis_design = _constant(coefficients, raw_basis_design)
    product_design = _constant(coefficients, raw_product_design)
    exposure = exposure_design @ coefficients
    return product_design.T @ (
        (_constant(coefficients, weights) * exposure)[:, None] * basis_design
    )


def _assemble_weighted_basis(
    positions: Any,
    *,
    radii: tuple[float, ...],
    transition_width: float,
    surface_lmax: int,
    exposure_lmax: int,
    radial_order: int,
):
    torch = _torch()
    blocks = []
    for atom_i in range(len(radii)):
        active_factors = []
        buried = False
        for atom_j in range(len(radii)):
            if atom_i == atom_j:
                continue
            state, factor = _pair_exposure_coefficients(
                positions[atom_j] - positions[atom_i],
                radius_i=radii[atom_i],
                radius_j=radii[atom_j],
                transition_width=transition_width,
                lmax=exposure_lmax,
                radial_order=radial_order,
            )
            if state == "buried":
                buried = True
                break
            if state == "transition":
                active_factors.append(factor)
        coefficients = (
            _constant_coefficients(positions, lmax=exposure_lmax, value=0.0)
            if buried
            else _project_product(
                tuple(active_factors), lmax=exposure_lmax, reference=positions
            )
        )
        blocks.append(
            _weighted_basis_block(
                coefficients,
                exposure_lmax=exposure_lmax,
                basis_lmax=surface_lmax,
            )
        )
    return torch.block_diag(*blocks)


def _rotation_from_positive_z(direction: Any):
    torch = _torch()
    unit = direction / torch.linalg.vector_norm(direction)

    def minimal_rotation(base: Any, target: Any):
        axis = torch.linalg.cross(base, target)
        cosine = torch.sum(base * target)
        zero = target.new_zeros(())
        cross = torch.stack(
            (
                torch.stack((zero, -axis[2], axis[1])),
                torch.stack((axis[2], zero, -axis[0])),
                torch.stack((-axis[1], axis[0], zero)),
            )
        )
        identity = torch.eye(3, dtype=target.dtype, device=target.device)
        # (1-cos(theta))/sin(theta)^2 = 1/(1+cos(theta)).  This
        # form retains the correct transverse derivative at theta=0.
        return identity + cross + cross @ cross / (1.0 + cosine)

    positive_z = unit.new_tensor([0.0, 0.0, 1.0])
    if float(unit[2].detach().cpu()) > -0.5:
        return minimal_rotation(positive_z, unit)
    negative_z = -positive_z
    half_turn = unit.new_tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return minimal_rotation(negative_z, unit) @ half_turn


def _torch_wigner_matrix(rotation: Any, *, lmax: int):
    torch = _torch()
    directions, weights = _exact_bandlimited_sphere_rule(lmax)
    base = _constant(rotation, _real_harmonic_design(directions, lmax=lmax))
    rotated_directions = _constant(rotation, directions) @ rotation
    rotated = _torch_real_harmonic_design(rotated_directions, lmax=lmax)
    weight_tensor = _constant(rotation, weights)
    blocks = []
    for ell in range(lmax + 1):
        section = slice(ell * ell, (ell + 1) * (ell + 1))
        blocks.append(
            base[:, section].T @ (weight_tensor[:, None] * rotated[:, section])
        )
    return torch.block_diag(*blocks)


def _canonical_layer_cross_block(
    reference: Any,
    *,
    target_radius: float,
    source_radius: float,
    distance: Any,
    lmax: int,
    radial_order: int,
    layer: str,
):
    """Return one pair-axis single- or double-layer harmonic block.

    The single-layer source coefficients denote charge per unit solid angle.
    The double-layer coefficients denote a boundary potential.  For distinct
    spheres the latter is the source-radius normal derivative of the former;
    its self-sphere principal value is assembled separately below.
    """

    if layer not in {"single", "double"}:
        raise ValueError("layer must be 'single' or 'double'.")
    torch = _torch()
    intersection = (source_radius**2 - distance**2 - target_radius**2) / (
        2.0 * distance * target_radius
    )
    intersection_value = float(intersection.detach().cpu())
    intervals: tuple[tuple[object, object], ...]
    if -1.0 < intersection_value < 1.0:
        intervals = ((-1.0, intersection), (intersection, 1.0))
    else:
        intervals = ((-1.0, 1.0),)

    nodes, weights = _legendre_rule(radial_order)
    azimuthal_count = max(1, 2 * lmax + 1)
    azimuth = 2.0 * np.pi * np.arange(azimuthal_count) / azimuthal_count
    cosine_phi = _constant(reference, np.cos(azimuth))
    sine_phi = _constant(reference, np.sin(azimuth))
    dimension = (lmax + 1) ** 2
    result = reference.new_zeros((dimension, dimension))
    labels = tuple(
        (ell, order) for ell in range(lmax + 1) for order in range(-ell, ell + 1)
    )

    for raw_lower, raw_upper in intervals:
        lower = (
            reference.new_tensor(raw_lower)
            if isinstance(raw_lower, float)
            else raw_lower
        )
        upper = (
            reference.new_tensor(raw_upper)
            if isinstance(raw_upper, float)
            else raw_upper
        )
        scale = 0.5 * (upper - lower)
        shift = 0.5 * (upper + lower)
        cosine = scale * _constant(reference, nodes) + shift
        polar_weights = scale * _constant(reference, weights)
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=0.0))
        directions = torch.stack(
            (
                torch.repeat_interleave(sine, azimuthal_count)
                * cosine_phi.repeat(len(cosine)),
                torch.repeat_interleave(sine, azimuthal_count)
                * sine_phi.repeat(len(cosine)),
                torch.repeat_interleave(cosine, azimuthal_count),
            ),
            dim=1,
        )
        target_points = target_radius * directions
        target_points = target_points + reference.new_tensor([0.0, 0.0, 1.0]) * distance
        radial_distance = torch.linalg.vector_norm(target_points, dim=1)
        source_directions = target_points / radial_distance[:, None]
        target_design = _torch_real_harmonic_design(directions, lmax=lmax)
        source_design = _torch_real_harmonic_design(source_directions, lmax=lmax)
        inside = radial_distance < source_radius
        scales = []
        for ell, _ in labels:
            if layer == "single":
                radial = torch.where(
                    inside,
                    radial_distance**ell / source_radius ** (ell + 1),
                    source_radius**ell / radial_distance ** (ell + 1),
                )
            else:
                # a_source^2 d/da_source of the Laplace single layer.
                # This is dimensionless and uses the outward source normal.
                radial = torch.where(
                    inside,
                    -(ell + 1) * (radial_distance / source_radius) ** ell,
                    ell * (source_radius / radial_distance) ** (ell + 1),
                )
            scales.append(4.0 * np.pi * radial / (2 * ell + 1))
        potential_scale = torch.stack(scales, dim=1)
        point_weights = torch.repeat_interleave(
            polar_weights * (2.0 * np.pi / azimuthal_count), azimuthal_count
        )
        result = result + target_design.T @ (
            point_weights[:, None] * source_design * potential_scale
        )

    stabilizer_order = max(1, 2 * lmax + 1)
    projected = reference.new_zeros(result.shape)
    for index in range(stabilizer_order):
        angle = 2.0 * np.pi * index / stabilizer_order
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        representation = _constant(reference, real_wigner_matrix(rotation, lmax=lmax))
        projected = (
            projected + representation.T @ result @ representation / stabilizer_order
        )
    return projected


def _canonical_cross_block(
    reference: Any,
    *,
    target_radius: float,
    source_radius: float,
    distance: Any,
    lmax: int,
    radial_order: int,
):
    return COULOMB_EV_ANGSTROM_PER_E2 * _canonical_layer_cross_block(
        reference,
        target_radius=target_radius,
        source_radius=source_radius,
        distance=distance,
        lmax=lmax,
        radial_order=radial_order,
        layer="single",
    )


def _canonical_double_layer_cross_block(
    reference: Any,
    *,
    target_radius: float,
    source_radius: float,
    distance: Any,
    lmax: int,
    radial_order: int,
):
    return _canonical_layer_cross_block(
        reference,
        target_radius=target_radius,
        source_radius=source_radius,
        distance=distance,
        lmax=lmax,
        radial_order=radial_order,
        layer="double",
    )


def _assemble_single_layer(
    positions: Any,
    *,
    radii: tuple[float, ...],
    lmax: int,
    radial_order: int,
):
    torch = _torch()
    dimension = (lmax + 1) ** 2
    row_blocks: list[list[Any]] = [[None for _ in radii] for _ in radii]
    for atom, radius in enumerate(radii):
        diagonal_blocks = []
        for ell in range(lmax + 1):
            eigenvalue = (
                COULOMB_EV_ANGSTROM_PER_E2 * 4.0 * np.pi / (radius * (2 * ell + 1))
            )
            diagonal_blocks.append(
                positions.new_tensor(eigenvalue)
                * torch.eye(2 * ell + 1, dtype=positions.dtype, device=positions.device)
            )
        row_blocks[atom][atom] = torch.block_diag(*diagonal_blocks)
    for target in range(len(radii)):
        for source in range(target + 1, len(radii)):
            displacement = positions[target] - positions[source]
            distance = torch.linalg.vector_norm(displacement)
            if float(distance.detach().cpu()) <= 1.0e-12:
                raise ValueError("sphere centres must remain distinct.")
            canonical = _canonical_cross_block(
                positions,
                target_radius=radii[target],
                source_radius=radii[source],
                distance=distance,
                lmax=lmax,
                radial_order=radial_order,
            )
            rotation = _rotation_from_positive_z(displacement / distance)
            representation = _torch_wigner_matrix(rotation, lmax=lmax)
            block = representation @ canonical @ representation.T
            row_blocks[target][source] = block
            row_blocks[source][target] = block.T
    rows = [torch.cat(tuple(blocks), dim=1) for blocks in row_blocks]
    operator = torch.cat(rows, dim=0)
    if operator.shape != (len(radii) * dimension, len(radii) * dimension):
        raise RuntimeError("harmonic single-layer assembly has an invalid shape.")
    return operator


def _assemble_double_layer(
    positions: Any,
    *,
    radii: tuple[float, ...],
    lmax: int,
    radial_order: int,
):
    """Assemble the dimensionless Laplace double-layer principal value.

    The source normal points out of each sphere.  The diagonal spectrum is the
    arithmetic trace of the interior/exterior limits,
    ``-2*pi/(2*l+1)``.  Ordered cross blocks are assembled independently
    because the double-layer operator is not generally symmetric.
    """

    torch = _torch()
    harmonic_sphere_pair_topology(
        np.asarray(positions.detach().cpu(), dtype=float),
        radii,
    )
    dimension = (lmax + 1) ** 2
    row_blocks: list[list[Any]] = [[None for _ in radii] for _ in radii]
    for atom in range(len(radii)):
        diagonal_blocks = [
            positions.new_tensor(-2.0 * np.pi / (2 * ell + 1))
            * torch.eye(2 * ell + 1, dtype=positions.dtype, device=positions.device)
            for ell in range(lmax + 1)
        ]
        row_blocks[atom][atom] = torch.block_diag(*diagonal_blocks)
    for target, target_radius in enumerate(radii):
        for source, source_radius in enumerate(radii):
            if target == source:
                continue
            displacement = positions[target] - positions[source]
            distance = torch.linalg.vector_norm(displacement)
            if float(distance.detach().cpu()) <= 1.0e-12:
                raise ValueError("sphere centres must remain distinct.")
            canonical = _canonical_double_layer_cross_block(
                positions,
                target_radius=target_radius,
                source_radius=source_radius,
                distance=distance,
                lmax=lmax,
                radial_order=radial_order,
            )
            rotation = _rotation_from_positive_z(displacement / distance)
            representation = _torch_wigner_matrix(rotation, lmax=lmax)
            row_blocks[target][source] = representation @ canonical @ representation.T
    rows = [torch.cat(tuple(blocks), dim=1) for blocks in row_blocks]
    operator = torch.cat(rows, dim=0)
    if operator.shape != (len(radii) * dimension, len(radii) * dimension):
        raise RuntimeError("harmonic double-layer assembly has an invalid shape.")
    return operator


def _point_monopole_coefficients(
    displacement: Any,
    *,
    target_radius: float,
    lmax: int,
):
    """Differentiable exact point-charge potential on one target sphere."""

    torch = _torch()
    distance = torch.linalg.vector_norm(displacement)
    distance_value = float(distance.detach().cpu())
    if distance_value <= 1.0e-12:
        result = displacement.new_zeros(((lmax + 1) ** 2,))
        result[0] = COULOMB_EV_ANGSTROM_PER_E2 * math.sqrt(4.0 * np.pi) / target_radius
        return result
    if (
        abs(distance_value - target_radius)
        <= POINT_SOURCE_SHELL_EVENT_TOLERANCE_ANGSTROM
    ):
        raise ValueError("point source lies on a target sphere event surface.")

    direction = displacement / distance
    harmonics = _torch_real_harmonic_design(direction[None, :], lmax=lmax)[0]
    blocks = []
    for ell in range(lmax + 1):
        radial = (
            distance**ell / target_radius ** (ell + 1)
            if distance_value < target_radius
            else target_radius**ell / distance ** (ell + 1)
        )
        section = slice(ell * ell, (ell + 1) * (ell + 1))
        blocks.append(
            COULOMB_EV_ANGSTROM_PER_E2
            * 4.0
            * np.pi
            * radial
            / (2 * ell + 1)
            * harmonics[section]
        )
    return torch.cat(blocks)


def _assemble_point_l0_source(
    positions: Any,
    *,
    radii: tuple[float, ...],
    lmax: int,
):
    """Map atomic ``[q,0,0,0]`` sources to raw boundary coefficients."""

    torch = _torch()
    rows = []
    dimension = (lmax + 1) ** 2
    for target, target_radius in enumerate(radii):
        row_blocks = []
        for source in range(len(radii)):
            displacement = positions[source] - positions[target]
            distance = float(torch.linalg.vector_norm(displacement).detach().cpu())
            if target != source and distance <= 1.0e-12:
                raise ValueError("distinct point-source centres must not coincide.")
            block = positions.new_zeros((dimension, 4))
            block[:, 0] = _point_monopole_coefficients(
                displacement,
                target_radius=target_radius,
                lmax=lmax,
            )
            row_blocks.append(block)
        rows.append(torch.cat(tuple(row_blocks), dim=1))
    return torch.cat(tuple(rows), dim=0)


def _point_kernel(radius_angstrom: Any):
    """Return the point-monopole kernel in atomic units."""

    return Bohr / radius_angstrom


def _quadrature_point_monopole_coefficients(
    displacement: Any,
    *,
    target_radius: float,
    lmax: int,
    radial_order: int,
):
    """Project one point potential without an inside/outside topology branch."""

    torch = _torch()
    distance = torch.linalg.vector_norm(displacement)
    distance_value = float(distance.detach().cpu())
    if not np.isfinite(distance_value) or distance_value <= 1.0e-12:
        raise ValueError("distinct point-source centres must remain distinct.")
    if abs(distance_value - target_radius) <= 1.0e-12:
        raise ValueError("a distinct point source lies on a target sphere.")
    nodes, weights, legendre = _legendre_design(radial_order, lmax)
    cosine = _constant(displacement, nodes)
    radius_squared = (
        target_radius**2 + distance**2 - 2.0 * target_radius * distance * cosine
    )
    radius = torch.sqrt(torch.clamp(radius_squared, min=0.0))
    zonal = (
        2.0
        * np.pi
        * (
            (_constant(displacement, weights) * _point_kernel(radius))
            @ _constant(displacement, legendre)
        )
    )
    direction = displacement / distance
    harmonics = _torch_real_harmonic_design(direction[None, :], lmax=lmax)[0]
    return torch.cat(
        [
            zonal[ell] * harmonics[ell * ell : (ell + 1) * (ell + 1)]
            for ell in range(lmax + 1)
        ]
    )


def _self_quadrature_point_source_block(
    reference: Any, *, target_radius: float, lmax: int
):
    torch = _torch()
    radius = reference.new_tensor(target_radius, requires_grad=True)
    kernel = _point_kernel(radius)
    (radial_derivative,) = torch.autograd.grad(kernel, (radius,), create_graph=True)
    block = reference.new_zeros(((lmax + 1) ** 2, 4))
    block[0, 0] = math.sqrt(4.0 * np.pi) * kernel
    if lmax >= 1:
        amplitude = -radial_derivative * math.sqrt(4.0 * np.pi / 3.0)
        block[1:4, 1:4] = amplitude * torch.eye(
            3, dtype=reference.dtype, device=reference.device
        )
    return block


def _distinct_quadrature_point_source_block(
    displacement: Any,
    *,
    target_radius: float,
    lmax: int,
    radial_order: int,
):
    torch = _torch()

    def monopole(candidate):
        return _quadrature_point_monopole_coefficients(
            candidate,
            target_radius=target_radius,
            lmax=lmax,
            radial_order=radial_order,
        )

    coefficient = monopole(displacement)
    jacobian = torch.func.jacfwd(monopole)(displacement)
    # Authoritative raw real-l=1 order is (m0,m1,m-1)=(y,z,x).
    return torch.stack(
        (coefficient, jacobian[:, 1], jacobian[:, 2], jacobian[:, 0]), dim=1
    )


def _assemble_quadrature_point_source(
    positions: Any,
    *,
    radii: tuple[float, ...],
    lmax: int,
    radial_order: int,
):
    """Assemble the smooth point ``[q,y,z,x]`` source operator.

    Unlike the legacy exact inside/outside coefficient formula, distinct
    source centres use one invariant radial projection on both sides of a
    target sphere. The physical source-on-surface singularity remains an exact
    fail-closed event; no finite clearance or active-set threshold is used.
    """

    torch = _torch()
    rows = []
    for target, target_radius in enumerate(radii):
        row_blocks = []
        for source in range(len(radii)):
            displacement = positions[source] - positions[target]
            distance = float(torch.linalg.vector_norm(displacement).detach().cpu())
            block = (
                _self_quadrature_point_source_block(
                    positions,
                    target_radius=target_radius,
                    lmax=lmax,
                )
                if distance <= 1.0e-12
                else _distinct_quadrature_point_source_block(
                    displacement,
                    target_radius=target_radius,
                    lmax=lmax,
                    radial_order=radial_order,
                )
            )
            row_blocks.append(Hartree * block)
        rows.append(torch.cat(tuple(row_blocks), dim=1))
    return torch.cat(tuple(rows), dim=0)


def _assemble_quadrature_point_monopole_source(
    positions: Any,
    *,
    radii: tuple[float, ...],
    lmax: int,
    radial_order: int,
):
    """Assemble the point-l0 source map without unused dipole Jacobians.

    The returned operator retains the registered four-component atomic source
    shape, but columns 1:4 are identically zero.  This exact specialization is
    reserved for model profiles, such as frozen-charge AIMNet2, whose scalar
    contract rejects nonzero l=1 source components.
    """

    rows = []
    dimension = (lmax + 1) ** 2
    for target, target_radius in enumerate(radii):
        row_blocks = []
        for source in range(len(radii)):
            displacement = positions[source] - positions[target]
            distance = float(_torch().linalg.vector_norm(displacement).detach().cpu())
            if target != source and distance <= 1.0e-12:
                raise ValueError("distinct point-source centres must remain distinct.")
            block = positions.new_zeros((dimension, 4))
            if distance <= 1.0e-12:
                block[0, 0] = math.sqrt(4.0 * np.pi) * _point_kernel(target_radius)
            else:
                block[:, 0] = _quadrature_point_monopole_coefficients(
                    displacement,
                    target_radius=target_radius,
                    lmax=lmax,
                    radial_order=radial_order,
                )
            row_blocks.append(Hartree * block)
        rows.append(_torch().cat(tuple(row_blocks), dim=1))
    return _torch().cat(tuple(rows), dim=0)



__all__ = [name for name in globals() if name.startswith("_")]
