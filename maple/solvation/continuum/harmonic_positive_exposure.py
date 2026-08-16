"""Positive Bernstein parent for smooth harmonic cavity moments.

The authoritative exposure on atom ``i`` is a finite positive polynomial

``e_i(u) = product_j p_ij(u)``

where every pair factor is a degree-four Bernstein polynomial whose control
values are samples of the compact smooth overlap switch.  The parent is never
reconstructed from a truncated harmonic series and used as a pointwise mask.
Instead, this module contracts the parent directly into

* its exact low harmonic moments; and
* ``W_i = P_L M_e P_L`` and ``I-W_i``.

The fixed tensor-product sphere rule is only an exact backend for the declared
finite polynomial degree.  It never samples a hard or infinite-band cavity
indicator.  Positive quadrature weights and the Bernstein convex-hull property
give structural area bounds and positive-semidefinite moment matrices without
clipping or eigenvalue repair.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .harmonic_coefficients import _bounded_lmax
from .harmonic_torch_primitives import (
    _constant,
    _finite_band_rule,
    _torch,
    _torch_flat_step,
    _torch_real_harmonic_design,
)

POSITIVE_BERNSTEIN_EXPOSURE_CONTRACT_ID = (
    "maple.route2.cavity.positive-bernstein-parent-moments.v1"
)
POSITIVE_BERNSTEIN_EXPOSURE_PROVIDER_ID = (
    "maple.route2.cavity.positive-bernstein-parent-moments.impl.v1"
)
POSITIVE_BERNSTEIN_PAIR_DEGREE = 4
POSITIVE_BERNSTEIN_MAXIMUM_TRANSITION_FACTORS = 30
POSITIVE_BERNSTEIN_MAXIMUM_INTEGRAND_DEGREE = 128
_SQRT_FOUR_PI = math.sqrt(4.0 * math.pi)
_PAIR_NODES = np.linspace(-1.0, 1.0, POSITIVE_BERNSTEIN_PAIR_DEGREE + 1)


@dataclass(frozen=True, slots=True)
class PositiveBernsteinExposureTensors:
    """Same-parent moments and Galerkin blocks for one geometry."""

    moments: Any
    multiplication_blocks: Any
    complement_blocks: Any
    transition_factor_counts: tuple[int, ...]
    maximum_integrand_degree: int


def _bernstein_factor_values(cosine: Any, control_values: Any):
    """Evaluate a degree-four Bernstein polynomial by de Casteljau."""

    values = control_values.expand(cosine.shape + control_values.shape).clone()
    x = 0.5 * (cosine + 1.0)
    one_minus_x = 1.0 - x
    for remaining in range(POSITIVE_BERNSTEIN_PAIR_DEGREE, 0, -1):
        values = (
            one_minus_x[..., None] * values[..., :remaining]
            + x[..., None] * values[..., 1 : remaining + 1]
        )
    return values[..., 0]


def _pair_factor_values(
    directions: Any,
    displacement: Any,
    *,
    radius_i: float,
    radius_j: float,
    transition_width: float,
):
    """Return one positive pair factor on an exact finite-band rule."""

    torch = _torch()
    distance = torch.linalg.vector_norm(displacement)
    distance_value = float(distance.detach().cpu())
    if not np.isfinite(distance_value) or distance_value <= 1.0e-12:
        raise ValueError(
            "positive Bernstein exposure requires distinct sphere centres."
        )
    nodes = _constant(displacement, _PAIR_NODES)
    signed_samples = (
        radius_i**2
        + distance * distance
        - 2.0 * radius_i * distance * nodes
        - radius_j**2
    )
    controls = _torch_flat_step(signed_samples / transition_width)
    unit_displacement = displacement / distance
    cosine = directions @ unit_displacement
    return _bernstein_factor_values(cosine, controls)


def _pair_state(
    displacement: Any,
    *,
    radius_i: float,
    radius_j: float,
    transition_width: float,
) -> str:
    distance = _torch().linalg.vector_norm(displacement)
    distance_value = float(distance.detach().cpu())
    if not np.isfinite(distance_value) or distance_value <= 1.0e-12:
        raise ValueError(
            "positive Bernstein exposure requires distinct sphere centres."
        )
    minimum = (distance - radius_i) ** 2 - radius_j**2
    maximum = (distance + radius_i) ** 2 - radius_j**2
    if float(minimum.detach().cpu()) >= transition_width:
        return "exposed"
    if float(maximum.detach().cpu()) <= -transition_width:
        return "buried"
    return "transition"


def _forward_error_tolerance(reference: Any, *, dimension: int):
    torch = _torch()
    return (
        64.0
        * max(1, dimension)
        * torch.finfo(reference.dtype).eps
        * torch.maximum(reference.new_tensor(1.0), torch.max(torch.abs(reference)))
    )


def assemble_positive_bernstein_exposure(
    positions: Any,
    *,
    radii: tuple[float, ...],
    transition_width: float,
    surface_lmax: int,
) -> PositiveBernsteinExposureTensors:
    """Assemble exact parent moments and ``P_L M_e P_L`` blocks.

    ``surface_lmax`` defines the continuum trial/test space.  Parent moments
    through ``2 * surface_lmax`` are retained because no higher moments can
    enter the multiplication matrix.  Every exact contraction is bounded by
    ``POSITIVE_BERNSTEIN_MAXIMUM_INTEGRAND_DEGREE`` and fails closed otherwise.
    """

    torch = _torch()
    maximum = _bounded_lmax(surface_lmax)
    if (
        not torch.is_tensor(positions)
        or positions.ndim != 2
        or positions.shape != (len(radii), 3)
        or not torch.is_floating_point(positions)
        or not bool(torch.isfinite(positions).all())
    ):
        raise ValueError("positions must be a finite floating tensor with shape (N,3).")
    if (
        not radii
        or any(not np.isfinite(value) or value <= 0.0 for value in radii)
        or not np.isfinite(transition_width)
        or transition_width <= 0.0
    ):
        raise ValueError("radii and transition_width must be finite and positive.")

    surface_dimension = (maximum + 1) ** 2
    moment_lmax = 2 * maximum
    moment_dimension = (moment_lmax + 1) ** 2
    moments_by_atom = []
    blocks = []
    complements = []
    transition_counts: list[int] = []
    maximum_degree = 0

    for atom_i, radius_i in enumerate(radii):
        transition_indices: list[int] = []
        buried = False
        for atom_j, radius_j in enumerate(radii):
            if atom_i == atom_j:
                continue
            state = _pair_state(
                positions[atom_j] - positions[atom_i],
                radius_i=radius_i,
                radius_j=radius_j,
                transition_width=transition_width,
            )
            if state == "buried":
                buried = True
                break
            if state == "transition":
                transition_indices.append(atom_j)

        transition_count = len(transition_indices)
        transition_counts.append(transition_count)
        if transition_count > POSITIVE_BERNSTEIN_MAXIMUM_TRANSITION_FACTORS:
            raise ValueError(
                "positive Bernstein exposure exceeds the frozen transition-factor cap."
            )
        required_degree = (
            POSITIVE_BERNSTEIN_PAIR_DEGREE * transition_count + 2 * maximum
        )
        maximum_degree = max(maximum_degree, required_degree)
        if required_degree > POSITIVE_BERNSTEIN_MAXIMUM_INTEGRAND_DEGREE:
            raise ValueError(
                "positive Bernstein exposure exceeds the frozen algebraic degree."
            )

        if buried:
            moments = positions.new_zeros((moment_dimension,))
            block = positions.new_zeros((surface_dimension, surface_dimension))
            complement = torch.eye(
                surface_dimension,
                dtype=positions.dtype,
                device=positions.device,
            )
        elif transition_count == 0:
            moments = positions.new_zeros((moment_dimension,))
            moments[0] = _SQRT_FOUR_PI
            block = torch.eye(
                surface_dimension,
                dtype=positions.dtype,
                device=positions.device,
            )
            complement = positions.new_zeros((surface_dimension, surface_dimension))
        else:
            raw_directions, raw_weights = _finite_band_rule(required_degree)
            directions = _constant(positions, raw_directions)
            weights = _constant(positions, raw_weights)
            parent = positions.new_ones((len(raw_directions),))
            for atom_j in transition_indices:  # stable input atom order
                factor = _pair_factor_values(
                    directions,
                    positions[atom_j] - positions[atom_i],
                    radius_i=radius_i,
                    radius_j=radii[atom_j],
                    transition_width=transition_width,
                )
                factor_tolerance = _forward_error_tolerance(
                    factor, dimension=POSITIVE_BERNSTEIN_PAIR_DEGREE + 1
                )
                if bool(
                    torch.logical_or(
                        factor < -factor_tolerance,
                        factor > 1.0 + factor_tolerance,
                    ).any()
                ):
                    raise RuntimeError(
                        "positive Bernstein pair factor violated its convex range."
                    )
                parent = parent * factor

            range_tolerance = _forward_error_tolerance(
                parent, dimension=len(raw_directions)
            )
            if bool(
                torch.logical_or(
                    parent < -range_tolerance,
                    parent > 1.0 + range_tolerance,
                ).any()
            ):
                raise RuntimeError(
                    "positive Bernstein parent violated its structural range."
                )
            moment_design = _torch_real_harmonic_design(directions, lmax=moment_lmax)
            surface_design = moment_design[:, :surface_dimension]
            moments = moment_design.T @ (weights * parent)
            block = surface_design.T @ ((weights * parent)[:, None] * surface_design)
            complement = surface_design.T @ (
                (weights * (1.0 - parent))[:, None] * surface_design
            )
            block = 0.5 * (block + block.T)
            complement = 0.5 * (complement + complement.T)

            identity = torch.eye(
                surface_dimension,
                dtype=positions.dtype,
                device=positions.device,
            )
            matrix_tolerance = _forward_error_tolerance(
                identity, dimension=surface_dimension * len(raw_directions)
            )
            if bool(
                torch.max(torch.abs(block + complement - identity)) > matrix_tolerance
            ):
                raise RuntimeError(
                    "positive Bernstein moment and complement blocks lost closure."
                )
            if bool(torch.linalg.eigvalsh(block)[0] < -matrix_tolerance) or bool(
                torch.linalg.eigvalsh(complement)[0] < -matrix_tolerance
            ):
                raise RuntimeError(
                    "positive Bernstein moment blocks lost positive semidefiniteness."
                )

        normalized_area = _SQRT_FOUR_PI * moments[0]
        area_tolerance = _forward_error_tolerance(
            normalized_area.reshape(1), dimension=max(1, required_degree)
        )
        if bool(
            torch.logical_or(
                normalized_area < -area_tolerance,
                normalized_area > 4.0 * math.pi + area_tolerance,
            )
        ):
            raise RuntimeError(
                "positive Bernstein parent area violated its structural bounds."
            )
        moments_by_atom.append(moments)
        blocks.append(block)
        complements.append(complement)

    return PositiveBernsteinExposureTensors(
        moments=torch.stack(moments_by_atom, dim=0),
        multiplication_blocks=torch.stack(blocks, dim=0),
        complement_blocks=torch.stack(complements, dim=0),
        transition_factor_counts=tuple(transition_counts),
        maximum_integrand_degree=maximum_degree,
    )


__all__ = [
    "POSITIVE_BERNSTEIN_EXPOSURE_CONTRACT_ID",
    "POSITIVE_BERNSTEIN_EXPOSURE_PROVIDER_ID",
    "POSITIVE_BERNSTEIN_MAXIMUM_INTEGRAND_DEGREE",
    "POSITIVE_BERNSTEIN_MAXIMUM_TRANSITION_FACTORS",
    "POSITIVE_BERNSTEIN_PAIR_DEGREE",
    "PositiveBernsteinExposureTensors",
    "assemble_positive_bernstein_exposure",
]
