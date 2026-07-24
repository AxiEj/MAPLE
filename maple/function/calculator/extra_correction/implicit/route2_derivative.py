"""Fixed-cavity energy-gradient ingredients for the Route-2 adjoint.

This module differentiates the mutual-polarization energy with respect to the
neutral density coordinates at fixed nuclear geometry and fixed PCM cavity.
It deliberately does not expose a nuclear force.
"""

from __future__ import annotations

import numpy as np

from .gto_density import external_field_to_density_order
from .route2_response import (
    ReactionFieldLinearMap,
    project_neutral_density_tangent,
)


def _validated_block(
    values: np.ndarray,
    *,
    expected_shape: tuple[int, int] | None,
    name: str,
) -> np.ndarray:
    block = np.asarray(values, dtype=float)
    valid_shape = (
        block.ndim == 2
        and block.shape[1] == 4
        and block.shape[0] > 0
        and (expected_shape is None or block.shape == expected_shape)
    )
    if not valid_shape or not np.all(np.isfinite(block)):
        shape = "(n_atoms, 4)" if expected_shape is None else str(expected_shape)
        raise ValueError(
            f"{name} must be finite with shape {shape}; received {block.shape}."
        )
    return block


def fixed_cavity_energy_density_gradient(
    reaction_field: ReactionFieldLinearMap,
    *,
    reaction_field_values: np.ndarray,
    intrinsic_energy_field_gradient: np.ndarray,
) -> np.ndarray:
    """Return the physical fixed-cavity energy gradient in neutral density space.

    At fixed geometry/cavity let ``f=P c`` and

    ``E(c) = E_MACE,intrinsic(f) + 0.5 c.T Q f``.

    For the reciprocal ``MATRIXSYMM=TRUE`` response used by Route 2,
    differentiating the half-coupling gives the full ``Q f`` term, so

    ``dE/dc = P.T g_f + Q f``

    with ``g_f=dE_MACE,intrinsic/df``.  The returned right-hand side is its
    orthogonal projection into the zero-total-monopole tangent space.  CDS and
    all coordinate/cavity derivatives are outside this fixed-cavity quantity.
    """

    if getattr(reaction_field, "reciprocal_energy_pairing", False) is not True:
        raise ValueError(
            "The fixed-cavity energy gradient requires a reciprocal reaction "
            "field with MATRIXSYMM=TRUE."
        )
    field = _validated_block(
        reaction_field_values,
        expected_shape=None,
        name="reaction_field_values",
    )
    field_gradient = _validated_block(
        intrinsic_energy_field_gradient,
        expected_shape=field.shape,
        name="intrinsic_energy_field_gradient",
    )
    mace_chain = _validated_block(
        reaction_field.adjoint(field_gradient),
        expected_shape=field.shape,
        name="reaction-field energy VJP",
    )
    polarization_gradient = external_field_to_density_order(field)
    return project_neutral_density_tangent(
        mace_chain + polarization_gradient
    )


__all__ = ["fixed_cavity_energy_density_gradient"]
