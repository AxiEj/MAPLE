"""Provider-neutral continuum-operator coordinate derivatives for Route 2.

For the energy-conjugate surface response

``q_energy = Q_sym(R) v``

this module contracts the geometry derivative of the continuum operator without
materializing ``dQ_sym/dR``.  Both surface potentials are held fixed in the
surface-node representation:

``d <u, Q_sym(R) v> / dR``.

Solute-MEP projection, reaction-field back-projection, and CDS geometry terms
remain separate.  A backend that cannot differentiate the exact operator used
for its energy must fail closed rather than report a partial force.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
    ExternalMEPCavityResponse,
    SurfaceChargeState,
)


EXTERNAL_MEP_OPERATOR_DERIVATIVE_CONTRACT_VERSION = 1


class ExternalMEPCavityOperatorDerivative(
    ExternalMEPCavityResponse,
    Protocol,
):
    """Continuum response with an exact energy-operator coordinate VJP."""

    operator_derivative_contract_version: int

    def operator_position_vjp(
        self,
        left_surface_potential_hartree_per_e: np.ndarray,
        right_surface_potential_hartree_per_e: np.ndarray,
    ) -> np.ndarray:
        """Return ``d<u,Q_sym(R)v>/dR`` in Hartree/Angstrom.

        The left and right surface potentials are held fixed.  The backend owns
        every cavity/surface contribution required to differentiate the same
        discrete energy-conjugate operator used by :meth:`solve`.
        """


def _validated_surface_potential(
    values: np.ndarray,
    *,
    surface_size: int,
    name: str,
) -> np.ndarray:
    potential = np.asarray(values, dtype=float)
    if potential.shape != (surface_size,) or not np.all(
        np.isfinite(potential)
    ):
        raise ValueError(
            f"{name} must be finite with shape ({surface_size},); "
            f"received {potential.shape}."
        )
    return potential


def continuum_operator_position_vjp(
    response: ExternalMEPCavityResponse,
    left_surface_potential_hartree_per_e: np.ndarray,
    right_surface_potential_hartree_per_e: np.ndarray,
) -> np.ndarray:
    """Contract the exact energy-conjugate continuum-operator derivative.

    For a nonsymmetric direct IEFPCM response ``Q=K^-1 R``, the operator in this
    contract is ``Q_sym=(Q+Q.T)/2``.  Consequently the returned bilinear VJP is
    symmetric in its left and right surface potentials.  The function does not
    approximate missing GePol/PCMSolver derivatives.
    """

    if (
        getattr(response, "contract_version", None)
        != EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
    ):
        raise ValueError(
            "Unsupported external-MEP continuum-response contract version."
        )
    if not getattr(response, "energy_response_is_reciprocal", False):
        raise ValueError(
            "Continuum operator derivatives require a reciprocal "
            "energy-conjugate response."
        )

    derivative_version = getattr(
        response,
        "operator_derivative_contract_version",
        None,
    )
    if derivative_version is None:
        raise NotImplementedError(
            "The continuum backend does not provide an operator derivative."
        )
    if derivative_version != (
        EXTERNAL_MEP_OPERATOR_DERIVATIVE_CONTRACT_VERSION
    ):
        raise ValueError(
            "Unsupported external-MEP operator-derivative contract version."
        )

    points = np.asarray(response.surface_points_bohr, dtype=float)
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(
            "Continuum surface points must be finite with shape (n_surface, 3)."
        )
    left = _validated_surface_potential(
        left_surface_potential_hartree_per_e,
        surface_size=points.shape[0],
        name="left_surface_potential_hartree_per_e",
    )
    right = _validated_surface_potential(
        right_surface_potential_hartree_per_e,
        surface_size=points.shape[0],
        name="right_surface_potential_hartree_per_e",
    )

    atomic_numbers = np.asarray(response.atomic_numbers, dtype=float)
    if (
        atomic_numbers.ndim != 1
        or atomic_numbers.size == 0
        or not np.all(np.isfinite(atomic_numbers))
    ):
        raise ValueError(
            "Continuum atomic numbers must be a finite one-dimensional array."
        )
    implementation = getattr(response, "operator_position_vjp", None)
    if not callable(implementation):
        raise NotImplementedError(
            "The continuum backend does not provide an operator derivative."
        )
    result = np.asarray(implementation(left, right), dtype=float)
    expected_shape = (atomic_numbers.size, 3)
    if result.shape != expected_shape or not np.all(np.isfinite(result)):
        raise ValueError(
            "Continuum operator position VJP must be finite with shape "
            f"{expected_shape}; received {result.shape}."
        )
    return result.copy()


def polarization_operator_position_gradient(
    response: ExternalMEPCavityResponse,
    state: SurfaceChargeState,
) -> np.ndarray:
    """Return the operator-only gradient of one polarization energy.

    With both occurrences of the surface MEP held fixed,

    ``dE_pol/dR|operator = 0.5 * d<v,Q_sym(R)v>/dR``.

    Surface-MEP motion and all non-electrostatic terms are intentionally absent.
    """

    potential = np.asarray(
        state.surface_potential_hartree_per_e,
        dtype=float,
    )
    return 0.5 * continuum_operator_position_vjp(
        response,
        potential,
        potential,
    )


__all__ = [
    "EXTERNAL_MEP_OPERATOR_DERIVATIVE_CONTRACT_VERSION",
    "ExternalMEPCavityOperatorDerivative",
    "continuum_operator_position_vjp",
    "polarization_operator_position_gradient",
]
