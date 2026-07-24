"""Energy-gradient ingredients for the Route-2 fixed-point adjoint.

This module owns the neutral density-space right-hand side and the composed
fixed-surface/operator coordinate-gradient slice.  It deliberately does not
expose a nuclear force because cavity/operator motion and SMD CDS derivatives
remain separate, incomplete terms.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .gto_density import (
    density_to_external_field_order,
    external_field_to_density_order,
)
from .route2_response import (
    DensityResponseLinearization,
    ReactionFieldLinearMap,
    project_neutral_density_tangent,
)


class FixedSurfaceReactionField(ReactionFieldLinearMap, Protocol):
    """Fixed-surface reaction map with a nuclear-coordinate pairing VJP."""

    atom_count: int
    reciprocal_energy_pairing: bool

    def position_vjp(
        self,
        density: np.ndarray,
        field_cotangent: np.ndarray,
    ) -> np.ndarray:
        """Differentiate ``<field_cotangent, P_R density>`` at fixed surface."""


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


def _validated_coordinate_block(
    values: np.ndarray,
    *,
    atom_count: int,
    name: str,
) -> np.ndarray:
    block = np.asarray(values, dtype=float)
    expected_shape = (atom_count, 3)
    if block.shape != expected_shape or not np.all(np.isfinite(block)):
        raise ValueError(
            f"{name} must be finite with shape {expected_shape}; "
            f"received {block.shape}."
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


def fixed_surface_solvation_coordinate_gradient(
    reaction_field: FixedSurfaceReactionField,
    density_response: DensityResponseLinearization,
    *,
    density_coefficients: np.ndarray,
    intrinsic_energy_field_gradient: np.ndarray,
    adjoint_solution: np.ndarray,
    adjoint_density_position_vjp: np.ndarray,
    solvent_fixed_field_forces_ev_per_angstrom: np.ndarray,
    gas_forces_ev_per_angstrom: np.ndarray,
    neutral_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Compose the fixed-surface coupled solvation-energy coordinate gradient.

    With ``f=P_R c``, residual
    ``r=Pi0[c-M(R,f)]``, and adjoint
    ``J_c r.T lambda = Pi0[dE/dc]``, this returns

    ``d(E_intrinsic(R,f)-E_gas(R)+0.5*c.T*Q*f)/dR``

    while density coefficients, tessera centres, and the PCM response operator
    are treated according to the implicit-function/adjoint decomposition.  The
    three field cotangents entering the single ``P_R`` position VJP are:

    - ``dE_intrinsic/df``;
    - ``0.5*Q.T*c`` from the PCM half-coupling;
    - ``J_M.T*lambda`` from the implicit density response.

    ``adjoint_density_position_vjp`` must be
    ``(dM/dR|f).T*lambda`` with the supplied atom-indexed field samples held
    fixed.  The returned value is an energy gradient in eV/Angstrom, not a
    force.  Cavity/operator motion and SMD CDS derivatives are absent.
    """

    if neutral_tolerance <= 0.0:
        raise ValueError("Neutral tangent tolerance must be positive.")
    if getattr(reaction_field, "reciprocal_energy_pairing", False) is not True:
        raise ValueError(
            "The fixed-surface coordinate gradient requires a reciprocal "
            "reaction field with MATRIXSYMM=TRUE."
        )

    density = _validated_block(
        density_coefficients,
        expected_shape=None,
        name="density_coefficients",
    )
    atom_count = density.shape[0]
    if reaction_field.atom_count != atom_count:
        raise ValueError(
            "Reaction-field atom count does not match density coefficients "
            f"({reaction_field.atom_count} != {atom_count})."
        )
    field_gradient = _validated_block(
        intrinsic_energy_field_gradient,
        expected_shape=density.shape,
        name="intrinsic_energy_field_gradient",
    )
    adjoint = _validated_block(
        adjoint_solution,
        expected_shape=density.shape,
        name="adjoint_solution",
    )
    charge_sum = abs(float(np.sum(adjoint[:, 0])))
    charge_scale = max(1.0, float(np.linalg.norm(adjoint[:, 0], ord=1)))
    if charge_sum > neutral_tolerance * charge_scale:
        raise ValueError(
            "adjoint_solution must lie in the neutral density tangent space "
            f"(monopole sum={float(np.sum(adjoint[:, 0])):.6e})."
        )

    response_field_cotangent = _validated_block(
        density_response.vjp(adjoint),
        expected_shape=density.shape,
        name="MACE density-response field VJP",
    )
    half_coupling_field_cotangent = (
        0.5 * density_to_external_field_order(density)
    )
    combined_field_cotangent = (
        field_gradient
        + half_coupling_field_cotangent
        + response_field_cotangent
    )
    reaction_position_gradient = _validated_coordinate_block(
        reaction_field.position_vjp(
            density,
            combined_field_cotangent,
        ),
        atom_count=atom_count,
        name="fixed-surface reaction-field position VJP",
    )
    density_position_gradient = _validated_coordinate_block(
        adjoint_density_position_vjp,
        atom_count=atom_count,
        name="adjoint_density_position_vjp",
    )
    solvent_forces = _validated_coordinate_block(
        solvent_fixed_field_forces_ev_per_angstrom,
        atom_count=atom_count,
        name="solvent_fixed_field_forces_ev_per_angstrom",
    )
    gas_forces = _validated_coordinate_block(
        gas_forces_ev_per_angstrom,
        atom_count=atom_count,
        name="gas_forces_ev_per_angstrom",
    )

    # Forces are negative coordinate gradients.  Therefore the direct
    # derivative of E_intrinsic(solvent)-E_gas is F_gas-F_solvent.
    result = (
        gas_forces
        - solvent_forces
        + reaction_position_gradient
        + density_position_gradient
    )
    if not np.all(np.isfinite(result)):
        raise RuntimeError(
            "Fixed-surface solvation coordinate gradient is non-finite."
        )
    return result


__all__ = [
    "FixedSurfaceReactionField",
    "fixed_cavity_energy_density_gradient",
    "fixed_surface_solvation_coordinate_gradient",
]
