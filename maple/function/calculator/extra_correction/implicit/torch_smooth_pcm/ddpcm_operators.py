"""Finite-dielectric coefficient-space primitives for smooth harmonic ddPCM.

The conductor Schwarz operator lives in
:mod:`harmonic_schwarz_primitives`.  Finite-dielectric PCM adds the published
two-stage boundary equation

``A_epsilon G = A_infinity F`` and ``L X = G``

with ``A_alpha = 2*pi*g_alpha*I - D``.  The ddPCM double layer follows ddX's
regularized characteristic convention: it uses the *exterior* multipole
continuation and a centered smooth target characteristic.  The pair
characteristic is applied before projection, so no singular bare continuation
is ever formed on a buried target patch.  Remaining pair factors are combined
in finite coefficient algebra.  Every angular block is assembled in a
complete harmonic representation; pair directions enter only through exact
SO(3) representation matrices and invariant one-dimensional quadrature.

This module contains no public capability or Route-2 state logic.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, NamedTuple

import numpy as np

from .harmonics import _real_harmonic_design, real_wigner_matrix
from .schwarz import (
    _assemble_point_ddcosmo,
    _multiplication_matrix,
)
from .exposure import (
    _constant,
    _constant_coefficients,
    _legendre_rule,
    _pair_exposure_coefficients,
    _project_product_with_leave_one_out,
    _rotation_from_positive_z,
    _torch_flat_step,
    _torch_real_harmonic_design,
    _torch_wigner_matrix,
    _weighted_basis_block,
)


class HarmonicDDPCMAssembly(NamedTuple):
    schwarz_operator: Any
    source_operator: Any
    receiver: Any
    partition: Any
    centered_exposure_coefficients: Any
    localized_double_layer: Any
    conductor_operator: Any
    dielectric_operator: Any


@lru_cache(maxsize=32)
def _canonical_double_layer_static_design(
    target_lmax: int,
    source_lmax: int,
    radial_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    """Cache geometry-independent pair-axis quadrature and target harmonics."""

    nodes, weights = _legendre_rule(radial_order)
    azimuthal_count = max(1, target_lmax + source_lmax + 1)
    azimuth = 2.0 * np.pi * np.arange(azimuthal_count) / azimuthal_count
    sine = np.sqrt(np.maximum(0.0, 1.0 - nodes * nodes))
    directions = np.stack(
        (
            np.repeat(sine, azimuthal_count) * np.tile(np.cos(azimuth), len(nodes)),
            np.repeat(sine, azimuthal_count) * np.tile(np.sin(azimuth), len(nodes)),
            np.repeat(nodes, azimuthal_count),
        ),
        axis=1,
    )
    target_design = _real_harmonic_design(directions, lmax=target_lmax)
    point_weights = np.repeat(
        weights * (2.0 * np.pi / azimuthal_count), azimuthal_count
    )
    labels = tuple(
        (ell, order) for ell in range(source_lmax + 1) for order in range(-ell, ell + 1)
    )
    for values in (directions, target_design, point_weights):
        values.setflags(write=False)
    return directions, target_design, point_weights, labels


@lru_cache(maxsize=32)
def _canonical_double_layer_stabilizer(
    target_lmax: int,
    source_lmax: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Cache the geometry-independent axial group-average representations."""

    order = max(1, target_lmax + source_lmax + 1)
    representations = []
    for index in range(order):
        angle = 2.0 * np.pi * index / order
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        target = real_wigner_matrix(rotation, lmax=target_lmax)
        source = real_wigner_matrix(rotation, lmax=source_lmax)
        target.setflags(write=False)
        source.setflags(write=False)
        representations.append((target, source))
    return tuple(representations)


def _canonical_double_layer_cross_block(
    reference: Any,
    *,
    target_radius: float,
    source_radius: float,
    distance: Any,
    target_lmax: int,
    source_lmax: int,
    transition_width: float,
    radial_order: int,
):
    """Return a pair-axis, pair-localized Laplace double-layer block.

    Coefficients on the source sphere represent a boundary potential.  For a
    distinct sphere the double layer is ``a_source**2 d/da_source`` of the
    exterior single-layer continuation.  This matches ddX's ``dx_dense``:
    the exterior multipole expression is used throughout the centered switch
    layer, including its small inside half, rather than changing to the
    physical interior double-layer branch.  The centered pair characteristic
    is multiplied pointwise before projection, which both regularizes the
    continuation and removes the topology branch at sphere tangencies.  The
    source normal points outwards.  The self principal-value block is
    assembled separately.
    """

    torch = __import__("torch")
    directions_raw, target_design_raw, point_weights_raw, labels = (
        _canonical_double_layer_static_design(target_lmax, source_lmax, radial_order)
    )
    target_dimension = (target_lmax + 1) ** 2
    source_dimension = (source_lmax + 1) ** 2
    result = reference.new_zeros((target_dimension, source_dimension))
    directions = _constant(reference, directions_raw)
    target_points = target_radius * directions
    target_points = target_points + reference.new_tensor([0.0, 0.0, 1.0]) * distance
    radial_squared = torch.sum(target_points * target_points, dim=1)
    radial_distance = torch.sqrt(radial_squared)
    signed_overlap = radial_squared - source_radius**2
    pair_exposure = _torch_flat_step(signed_overlap / transition_width)
    active = pair_exposure > 0.0
    safe_distance = torch.where(
        active, radial_distance, torch.ones_like(radial_distance)
    )
    safe_directions = target_points / safe_distance[:, None]
    safe_directions = torch.where(
        active[:, None],
        safe_directions,
        reference.new_tensor([0.0, 0.0, 1.0]),
    )
    target_design = _constant(reference, target_design_raw)
    source_design = _torch_real_harmonic_design(safe_directions, lmax=source_lmax)
    scales = []
    for ell, _ in labels:
        if ell == 0:
            radial = torch.zeros_like(safe_distance)
        else:
            radial = ell * (source_radius / safe_distance) ** (ell + 1)
        scales.append(pair_exposure * (4.0 * np.pi / (2 * ell + 1)) * radial)
    potential_scale = torch.stack(scales, dim=1)
    point_weights = _constant(reference, point_weights_raw)
    result = target_design.T @ (
        point_weights[:, None] * source_design * potential_scale
    )

    # Project out numerical dependence on the arbitrary transverse pair frame.
    stabilizer = _canonical_double_layer_stabilizer(target_lmax, source_lmax)
    stabilizer_order = len(stabilizer)
    projected = reference.new_zeros(result.shape)
    for target_raw, source_raw in stabilizer:
        target_representation = _constant(reference, target_raw)
        source_representation = _constant(reference, source_raw)
        projected = (
            projected
            + target_representation.T
            @ result
            @ source_representation
            / stabilizer_order
        )
    return projected


def _centered_exposure_factors(
    positions: Any,
    *,
    radii: tuple[float, ...],
    transition_width: float,
    partition_lmax: int,
    radial_order: int,
):
    """Return centered pair factors and their full exposed products."""

    rows = []
    full = []
    remaining = []
    for target, target_radius in enumerate(radii):
        factors = {}
        active = []
        buried = False
        for source, source_radius in enumerate(radii):
            if target == source:
                continue
            state, coefficients = _pair_exposure_coefficients(
                positions[source] - positions[target],
                radius_i=target_radius,
                radius_j=source_radius,
                transition_width=transition_width,
                lmax=partition_lmax,
                radial_order=radial_order,
            )
            factors[source] = (state, coefficients)
            if state == "buried":
                buried = True
            elif state == "transition":
                active.append((source, coefficients))
        rows.append(factors)
        zero = _constant_coefficients(positions, lmax=partition_lmax, value=0.0)
        if buried:
            full.append(zero)
            remaining.append({source: zero for source in factors})
            continue
        exposed, leave = _project_product_with_leave_one_out(
            tuple(coefficients for _source, coefficients in active),
            lmax=partition_lmax,
            reference=positions,
        )
        full.append(exposed)
        leave_by_source = {
            source: leave[index] for index, (source, _coefficients) in enumerate(active)
        }
        remaining.append(
            {source: leave_by_source.get(source, exposed) for source in factors}
        )
    return tuple(rows), tuple(full), tuple(remaining)


def _assemble_localized_double_layer(
    positions: Any,
    *,
    radii: tuple[float, ...],
    transition_width: float,
    lmax: int,
    partition_lmax: int,
    radial_order: int,
):
    """Assemble ddX-like centered-characteristic double-layer blocks.

    Cross blocks are localized pairwise before harmonic projection.  This is
    essential: the exterior multipole continuation is singular on deeply
    buried target points, while its product with the flat centered
    characteristic is smooth and bounded.  Forming a bare full-sphere block
    first and multiplying it afterwards is therefore mathematically invalid.
    """

    torch = __import__("torch")
    dimension = (lmax + 1) ** 2
    physical_lmax = lmax + partition_lmax
    factor_rows, full_exposure, remaining_exposure = _centered_exposure_factors(
        positions,
        radii=radii,
        transition_width=transition_width,
        partition_lmax=partition_lmax,
        radial_order=radial_order,
    )
    row_blocks: list[list[Any]] = [[None for _ in radii] for _ in radii]
    for atom in range(len(radii)):
        diagonal_blocks = [
            positions.new_tensor(-2.0 * np.pi / (2 * ell + 1))
            * torch.eye(2 * ell + 1, dtype=positions.dtype, device=positions.device)
            for ell in range(lmax + 1)
        ]
        self_principal_value = torch.block_diag(*diagonal_blocks)
        exposed_multiplication = _multiplication_matrix(
            full_exposure[atom],
            partition_lmax=partition_lmax,
            lmax=lmax,
        )
        row_blocks[atom][atom] = exposed_multiplication @ self_principal_value

    for target, target_radius in enumerate(radii):
        for source, source_radius in enumerate(radii):
            if target == source:
                continue
            pair_state, _pair_coefficients = factor_rows[target][source]
            if pair_state == "buried":
                row_blocks[target][source] = positions.new_zeros((dimension, dimension))
                continue
            displacement = positions[target] - positions[source]
            distance = torch.linalg.vector_norm(displacement)
            distance_value = float(distance.detach().cpu())
            if not np.isfinite(distance_value) or distance_value <= 1.0e-12:
                raise ValueError("sphere centres must remain distinct.")
            canonical = _canonical_double_layer_cross_block(
                positions,
                target_radius=target_radius,
                source_radius=source_radius,
                distance=distance,
                target_lmax=physical_lmax,
                source_lmax=lmax,
                transition_width=transition_width,
                radial_order=radial_order,
            )
            rotation = _rotation_from_positive_z(displacement / distance)
            target_representation = _torch_wigner_matrix(rotation, lmax=physical_lmax)
            source_representation = _torch_wigner_matrix(rotation, lmax=lmax)
            pair_localized = target_representation @ canonical @ source_representation.T
            remaining_weighted_basis = _weighted_basis_block(
                remaining_exposure[target][source],
                exposure_lmax=partition_lmax,
                basis_lmax=lmax,
            )
            row_blocks[target][source] = remaining_weighted_basis.T @ pair_localized

    operator = torch.cat(
        tuple(torch.cat(tuple(blocks), dim=1) for blocks in row_blocks), dim=0
    )
    expected = len(radii) * dimension
    if operator.shape != (expected, expected):
        raise RuntimeError("harmonic double-layer assembly has an invalid shape.")
    return operator, torch.stack(full_exposure)


def _assemble_point_ddpcm(
    positions: Any,
    *,
    radii: tuple[float, ...],
    transition_width: float,
    lmax: int,
    partition_lmax: int,
    partition_radial_order: int,
    source_radial_order: int,
    double_layer_radial_order: int,
    dielectric: float,
    point_monopoles_only: bool = False,
) -> HarmonicDDPCMAssembly:
    """Assemble the fixed-coefficient smooth ddPCM two-stage system.

    If ``F=-B c``, the primal equations are

    ``A_epsilon G=A_infinity F`` and ``L X=G``.

    The Schwarz/source stage retains the validated one-sided local-potential
    partition.  The finite-dielectric double-layer stage uses a distinct
    centered ddPCM characteristic, as recommended by ddX.  Cross blocks apply
    that characteristic before projection and never form a singular bare
    exterior continuation.
    """

    torch = __import__("torch")
    schwarz, source_operator, receiver, partition = _assemble_point_ddcosmo(
        positions,
        radii=radii,
        transition_width=transition_width,
        lmax=lmax,
        partition_lmax=partition_lmax,
        partition_radial_order=partition_radial_order,
        source_radial_order=source_radial_order,
        point_monopoles_only=point_monopoles_only,
    )
    localized_double, centered_exposure = _assemble_localized_double_layer(
        positions,
        radii=radii,
        transition_width=transition_width,
        lmax=lmax,
        partition_lmax=partition_lmax,
        radial_order=double_layer_radial_order,
    )
    identity = torch.eye(
        schwarz.shape[0], dtype=positions.dtype, device=positions.device
    )
    conductor = 2.0 * np.pi * identity - localized_double
    jump = (dielectric + 1.0) / (dielectric - 1.0)
    finite_dielectric = 2.0 * np.pi * jump * identity - localized_double
    return HarmonicDDPCMAssembly(
        schwarz_operator=schwarz,
        source_operator=source_operator,
        receiver=receiver,
        partition=partition,
        centered_exposure_coefficients=centered_exposure,
        localized_double_layer=localized_double,
        conductor_operator=conductor,
        dielectric_operator=finite_dielectric,
    )


__all__ = [
    "HarmonicDDPCMAssembly",
    "_assemble_localized_double_layer",
    "_assemble_point_ddpcm",
    "_canonical_double_layer_cross_block",
    "_centered_exposure_factors",
]
