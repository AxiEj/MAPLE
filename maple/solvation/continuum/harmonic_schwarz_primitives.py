"""Differentiable coefficient-space Schwarz primitives for smooth ddCOSMO.

The weighted-shell Galerkin candidate multiplies every local charge trial
function by an exposure field.  Complete burial therefore annihilates a local
trial block and changes the range of the variational map.  This module uses the
domain-decomposition variables of ddCOSMO instead: every atom retains a fixed
block of *local reaction-potential* coefficients and overlap equations relate
the local harmonic representations.

All angular contractions operate on complete finite harmonic spaces.  The
only integrations of non-polynomial switching functions are invariant
one-dimensional Legendre contractions.  No laboratory-fixed surface grid is
part of the mathematical definition.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .harmonic_torch_primitives import (
    _assemble_quadrature_point_monopole_source,
    _assemble_quadrature_point_source,
    _constant,
    _constant_coefficients,
    _finite_band_design,
    _legendre_design,
    _project_product,
    _torch_flat_step,
    _torch_real_harmonic_design,
    _weighted_basis_block,
)


def _regular_solid_harmonic_design(vectors: Any, *, lmax: int):
    """Evaluate ``r**l Y_lm(r_hat)`` in MAPLE's real-harmonic convention.

    A Cartesian solid-harmonic recurrence avoids dividing by ``|r|``.  The
    evaluation therefore remains well defined when a target-sphere point
    coincides with a neighbouring ball centre.
    """

    torch = __import__("torch")
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors must have shape (point_count, 3).")
    x, y, z = vectors.unbind(-1)
    radius_squared = x * x + y * y + z * z
    complex_xy = torch.complex(x, y)
    values: dict[tuple[int, int, str], Any] = {}
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
                - (ell + order - 1) * radius_squared * cosine_sequence[ell - 2]
            ) / (ell - order)
            sine_sequence[ell] = (
                (2 * ell - 1) * z * sine_sequence[ell - 1]
                - (ell + order - 1) * radius_squared * sine_sequence[ell - 2]
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


def _inside_pair_factors(
    positions: Any,
    *,
    radii: tuple[float, ...],
    transition_width: float,
    partition_lmax: int,
    radial_order: int,
):
    """Return nonzero smooth inside factors for every target sphere."""

    rows: list[tuple[tuple[int, Any], ...]] = []
    for atom_i, radius_i in enumerate(radii):
        factors = []
        for atom_j, radius_j in enumerate(radii):
            if atom_i == atom_j:
                continue
            state, inside = _pair_inside_coefficients(
                positions[atom_j] - positions[atom_i],
                radius_i=radius_i,
                radius_j=radius_j,
                transition_width=transition_width,
                lmax=partition_lmax,
                radial_order=radial_order,
            )
            if state != "exposed":
                factors.append((atom_j, inside))
        rows.append(tuple(factors))
    return tuple(rows)


def _pair_inside_coefficients(
    displacement: Any,
    *,
    radius_i: float,
    radius_j: float,
    transition_width: float,
    lmax: int,
    radial_order: int,
):
    """Project a smooth inside factor whose exact support lies in ball ``j``.

    With ``z=|R_i+a_i u-R_j|^2-a_j^2`` the factor is one for
    ``z<=-transition_width``, changes smoothly only on ``(-width,0)``, and is
    exactly zero for ``z>=0``.  Thus the local harmonic continuation from ball
    ``j`` is never weighted outside that ball in the underlying continuous
    Schwarz equation.
    """

    torch = __import__("torch")
    distance = torch.linalg.vector_norm(displacement)
    distance_value = float(distance.detach().cpu())
    if not np.isfinite(distance_value) or distance_value <= 1.0e-12:
        raise ValueError("sphere centres must remain distinct.")
    minimum = (distance - radius_i) ** 2 - radius_j**2
    maximum = (distance + radius_i) ** 2 - radius_j**2
    if float(minimum.detach().cpu()) >= 0.0:
        return "exposed", _constant_coefficients(displacement, lmax=lmax, value=0.0)
    if float(maximum.detach().cpu()) <= -transition_width:
        return "buried", _constant_coefficients(displacement, lmax=lmax, value=1.0)

    nodes, weights, legendre = _legendre_design(radial_order, lmax)
    cosine = _constant(displacement, nodes)
    signed_overlap = (
        radius_i**2 + distance**2 - 2.0 * radius_i * distance * cosine - radius_j**2
    )
    exposure = _torch_flat_step(2.0 * signed_overlap / transition_width + 1.0)
    inside = 1.0 - exposure
    zonal = (
        2.0
        * np.pi
        * (
            (_constant(displacement, weights) * inside)
            @ _constant(displacement, legendre)
        )
    )
    direction = displacement / distance
    harmonics = _torch_real_harmonic_design(direction[None, :], lmax=lmax)[0]
    return "transition", torch.cat(
        [
            zonal[ell] * harmonics[ell * ell : (ell + 1) * (ell + 1)]
            for ell in range(lmax + 1)
        ]
    )


def _assemble_smooth_partition(
    positions: Any,
    *,
    radii: tuple[float, ...],
    transition_width: float,
    partition_lmax: int,
    radial_order: int,
):
    """Assemble exposed and permutation-symmetric overlap coefficients.

    For inside factors ``f_j`` the finite coefficient algebra uses

    ``e = prod_j (1-f_j)`` and

    ``omega_j = f_j int_0^1 prod_(k!=j) (1-t f_k) dt``.

    The one-dimensional Gauss rule is exact because the integrand is a finite
    polynomial in ``t``.  Consequently ``e + sum_j omega_j = 1`` holds in the
    retained coefficient space up to floating-point contraction error.  The
    individual weights approximate the underlying non-band-limited smooth
    functions; pointwise positivity after harmonic truncation is a numerical
    validation item, not claimed as a structural theorem.
    """

    one = _constant_coefficients(positions, lmax=partition_lmax, value=1.0)
    factor_rows = _inside_pair_factors(
        positions,
        radii=radii,
        transition_width=transition_width,
        partition_lmax=partition_lmax,
        radial_order=radial_order,
    )
    result = []
    for factors in factor_rows:
        overlap: dict[int, Any] = {}
        count = len(factors)
        if not count:
            exposed = one
        else:
            # Every product in this row has ``count`` factors and is projected
            # back through degree ``partition_lmax``.  Reuse one exact finite-
            # band design rather than rebuilding the same point values once
            # for every neighbour and every scalar Gauss node.
            required_degree = (count + 1) * partition_lmax
            _directions, sphere_weights, raw_design = _finite_band_design(
                required_degree, partition_lmax
            )
            design = _constant(positions, raw_design)
            sphere_weights_tensor = _constant(positions, sphere_weights)
            factor_values = __import__("torch").stack(
                tuple(design @ factor for _, factor in factors)
            )
            one_values = design @ one
            exposed_values = __import__("torch").prod(
                one_values[None, :] - factor_values, dim=0
            )
            exposed = design.T @ (sphere_weights_tensor * exposed_values)

            # omega_j has degree count-1 in t.  An n-point Gauss rule is exact
            # through degree 2*n-1.
            quadrature_order = max(1, (count + 1) // 2)
            nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
            nodes = 0.5 * (nodes + 1.0)
            weights = 0.5 * weights
            overlap_values = positions.new_zeros(factor_values.shape)
            for node, weight in zip(nodes, weights, strict=True):
                complements = one_values[None, :] - float(node) * factor_values
                prefix = [positions.new_ones(one_values.shape)]
                for values in complements:
                    prefix.append(prefix[-1] * values)
                suffix = [None] * (count + 1)
                suffix[count] = positions.new_ones(one_values.shape)
                for index in range(count - 1, -1, -1):
                    suffix[index] = complements[index] * suffix[index + 1]
                excluding = __import__("torch").stack(
                    tuple(prefix[index] * suffix[index + 1] for index in range(count))
                )
                overlap_values = overlap_values + (
                    float(weight) * factor_values * excluding
                )
            overlap_coefficients = (
                overlap_values * sphere_weights_tensor[None, :]
            ) @ design
            overlap = {
                atom_j: overlap_coefficients[index]
                for index, (atom_j, _factor) in enumerate(factors)
            }

        closure = exposed
        for coefficients in overlap.values():
            closure = closure + coefficients
        closure_error = float(
            __import__("torch")
            .max(__import__("torch").abs(closure - one))
            .detach()
            .cpu()
        )
        if not np.isfinite(closure_error) or closure_error > 2.0e-10:
            raise RuntimeError("smooth Schwarz partition lost coefficient closure.")
        result.append((exposed, overlap))
    return tuple(result)


def _multiplication_matrix(coefficients: Any, *, partition_lmax: int, lmax: int):
    """Return ``P_L M_coeff P_L`` using exact finite-band contraction."""

    full = _weighted_basis_block(
        coefficients,
        exposure_lmax=partition_lmax,
        basis_lmax=lmax,
    )
    dimension = (lmax + 1) ** 2
    return full[:dimension]


def _local_potential_translation(
    positions: Any,
    *,
    radii: tuple[float, ...],
    target: int,
    source: int,
    lmax: int,
):
    """Translate ball-``source`` regular-potential coefficients to sphere target."""

    directions, weights, raw_target_design = _finite_band_design(2 * lmax, lmax)
    target_design = _constant(positions, raw_target_design)
    relative = (
        positions[target]
        - positions[source]
        + radii[target] * _constant(positions, directions)
    ) / radii[source]
    source_values = _regular_solid_harmonic_design(relative, lmax=lmax)
    return target_design.T @ (_constant(positions, weights)[:, None] * source_values)


def _point_receiver(positions: Any, *, radii: tuple[float, ...], lmax: int):
    """Map local reaction-potential coefficients to ``[q,y,z,x]`` pairing."""

    torch = __import__("torch")
    dimension = (lmax + 1) ** 2
    rows = []
    value_scale = 1.0 / math.sqrt(4.0 * np.pi)
    gradient_scale = math.sqrt(3.0 / (4.0 * np.pi))
    for atom_i, radius in enumerate(radii):
        block = positions.new_zeros((4, dimension))
        block[0, 0] = value_scale
        if lmax >= 1:
            block[1:4, 1:4] = (gradient_scale / radius) * torch.eye(
                3, dtype=positions.dtype, device=positions.device
            )
        row = [positions.new_zeros((4, dimension)) for _ in radii]
        row[atom_i] = block
        rows.append(torch.cat(tuple(row), dim=1))
    return torch.cat(tuple(rows), dim=0)


def _assemble_point_ddcosmo(
    positions: Any,
    *,
    radii: tuple[float, ...],
    transition_width: float,
    lmax: int,
    partition_lmax: int,
    partition_radial_order: int,
    source_radial_order: int,
    point_monopoles_only: bool = False,
):
    """Return ``(L, B, C, partition)`` for a point-l<=1 ddCOSMO scalar.

    The forward equation is ``L X = -B c`` and the physical discrete energy
    before dielectric scaling is ``0.5 * c.T @ C @ X``.
    """

    torch = __import__("torch")
    atom_count = len(radii)
    dimension = (lmax + 1) ** 2
    physical_lmax = lmax + partition_lmax
    physical_dimension = (physical_lmax + 1) ** 2
    partition = _assemble_smooth_partition(
        positions,
        radii=radii,
        transition_width=transition_width,
        partition_lmax=partition_lmax,
        radial_order=partition_radial_order,
    )

    rows = []
    identity = torch.eye(dimension, dtype=positions.dtype, device=positions.device)
    for atom_i, (_exposed, overlap) in enumerate(partition):
        blocks = []
        for atom_j in range(atom_count):
            if atom_i == atom_j:
                block = identity
            elif atom_j not in overlap:
                block = positions.new_zeros((dimension, dimension))
            else:
                multiplication = _multiplication_matrix(
                    overlap[atom_j],
                    partition_lmax=partition_lmax,
                    lmax=lmax,
                )
                translation = _local_potential_translation(
                    positions,
                    radii=radii,
                    target=atom_i,
                    source=atom_j,
                    lmax=lmax,
                )
                block = -multiplication @ translation
            blocks.append(block)
        rows.append(torch.cat(tuple(blocks), dim=1))
    schwarz = torch.cat(tuple(rows), dim=0)

    source_assembler = (
        _assemble_quadrature_point_monopole_source
        if point_monopoles_only
        else _assemble_quadrature_point_source
    )
    raw_source = source_assembler(
        positions,
        radii=radii,
        lmax=physical_lmax,
        radial_order=source_radial_order,
    )
    source_operator = positions.new_zeros((atom_count * dimension, atom_count * 4))
    for atom_i, (exposed, _overlap) in enumerate(partition):
        weighted_test = _weighted_basis_block(
            exposed,
            exposure_lmax=partition_lmax,
            basis_lmax=lmax,
        )
        raw_block = raw_source[
            atom_i * physical_dimension : (atom_i + 1) * physical_dimension
        ]
        source_operator[atom_i * dimension : (atom_i + 1) * dimension] = (
            weighted_test.T @ raw_block
        )

    receiver = _point_receiver(positions, radii=radii, lmax=lmax)
    return schwarz, source_operator, receiver, partition


__all__ = [
    "_assemble_point_ddcosmo",
    "_assemble_smooth_partition",
    "_local_potential_translation",
    "_regular_solid_harmonic_design",
]
