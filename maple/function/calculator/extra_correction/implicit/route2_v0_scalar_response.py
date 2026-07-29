"""Zero-training field-space scalar response for the Route-2 V0 falsifier.

This module does not change the legacy fixed-point route or expose a public
solvation model.  It evaluates one fixed-geometry local-field construction from
the frozen MACE-POLAR checkpoint:

``W0(f) = E(f) - E(0) + <c_M(f), f> - grad(E)(0) . f``

and defines the candidate physical source as the density-dual gradient of that
single scalar.  The zero-field linear anchor preserves the frozen model density
at ``f=0`` without changing the scalar Hessian.  No density projection, fitting,
response tempering, or experimental calibration is applied.

Only the existing ``(n_atoms, 4)`` local potential/gradient interface is
admitted.  Native exact-GTO features intentionally remain excluded because the
repository has not established a density-to-feature conjugacy map.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

from .electrostatic_pairing import (
    MACE_POLAR_L1_PAIRING,
    ElectrostaticPairing,
)


class DensityResponseVJP(Protocol):
    """Transpose action of the frozen model field-to-density response."""

    def vjp(self, density_cotangent: np.ndarray) -> np.ndarray:
        """Map a raw density cotangent into external Cartesian field order."""
        ...


def _validated_block(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    valid_shape = (
        array.ndim == 2
        and array.shape[0] > 0
        and array.shape[1] == 4
        and (expected_shape is None or array.shape == expected_shape)
    )
    if not valid_shape or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must be finite with shape (n_atoms, 4); "
            f"received {array.shape}."
        )
    return array


def _validated_scalar(value: float, *, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


@dataclass(frozen=True)
class Route2V0ScalarResponseState:
    """Auditable state of one anchored scalar-response evaluation."""

    field_conditioned_model_energy_ev: float
    zero_field_model_energy_ev: float
    reaction_field_values_ev: np.ndarray
    model_density_coefficients: np.ndarray
    field_conditioned_energy_field_gradient: np.ndarray
    zero_field_energy_field_gradient: np.ndarray
    density_response_cotangent: np.ndarray
    scalar_field_gradient: np.ndarray
    response_density_coefficients: np.ndarray
    zero_field_anchor_work_ev: float
    model_density_field_coupling_ev: float
    response_density_field_coupling_ev: float
    anchored_field_scalar_change_ev: float
    electrostatic_energy_ev: float
    response_total_charge_e: float
    field_interface: str = "local-potential-gradient-v1"
    construction: str = "anchored-frozen-checkpoint-scalar-response-v0"

    def __post_init__(self) -> None:
        array_names = (
            "reaction_field_values_ev",
            "model_density_coefficients",
            "field_conditioned_energy_field_gradient",
            "zero_field_energy_field_gradient",
            "density_response_cotangent",
            "scalar_field_gradient",
            "response_density_coefficients",
        )
        shape: tuple[int, int] | None = None
        for name in array_names:
            values = _validated_block(
                getattr(self, name),
                name=name,
                expected_shape=shape,
            )
            shape = values.shape
            immutable = np.array(values, copy=True)
            immutable.setflags(write=False)
            object.__setattr__(self, name, immutable)
        scalar_names = (
            "field_conditioned_model_energy_ev",
            "zero_field_model_energy_ev",
            "zero_field_anchor_work_ev",
            "model_density_field_coupling_ev",
            "response_density_field_coupling_ev",
            "anchored_field_scalar_change_ev",
            "electrostatic_energy_ev",
            "response_total_charge_e",
        )
        for name in scalar_names:
            object.__setattr__(
                self,
                name,
                _validated_scalar(getattr(self, name), name=name),
            )
        if self.field_interface != "local-potential-gradient-v1":
            raise ValueError("Route-2 V0 admits only the local-field interface.")
        if self.construction != "anchored-frozen-checkpoint-scalar-response-v0":
            raise ValueError("Unsupported Route-2 V0 scalar construction.")


def evaluate_anchored_scalar_response_v0(
    *,
    field_conditioned_model_energy_ev: float,
    zero_field_model_energy_ev: float,
    reaction_field_values_ev: np.ndarray,
    model_density_coefficients: np.ndarray,
    field_conditioned_energy_field_gradient: np.ndarray,
    zero_field_energy_field_gradient: np.ndarray,
    density_response: DensityResponseVJP,
    pairing: ElectrostaticPairing = MACE_POLAR_L1_PAIRING,
) -> Route2V0ScalarResponseState:
    """Evaluate the frozen-checkpoint V0 source and its unique energy ledger.

    The response density is the pairing-aware gradient of the anchored scalar
    ``W0``.  If ``Q`` maps external field order to raw density-dual order, then

    ``c_V0 = Q^-T [g(f) + Q^T c_M(f) + J_M(f)^T Q f - g(0)]``.

    The corresponding field-space electrostatic candidate is

    ``Delta G_V0 = W0(f) - 1/2 <c_V0(f), f>``.

    When the frozen model already satisfies the variational conjugacy identity,
    ``c_V0`` equals the model density and this expression reduces exactly to the
    legacy model-energy-change plus continuum half-coupling ledger.
    """

    field = _validated_block(
        reaction_field_values_ev,
        name="reaction_field_values_ev",
    )
    shape = field.shape
    model_density = _validated_block(
        model_density_coefficients,
        name="model_density_coefficients",
        expected_shape=shape,
    )
    energy_gradient = _validated_block(
        field_conditioned_energy_field_gradient,
        name="field_conditioned_energy_field_gradient",
        expected_shape=shape,
    )
    zero_gradient = _validated_block(
        zero_field_energy_field_gradient,
        name="zero_field_energy_field_gradient",
        expected_shape=shape,
    )
    field_energy = _validated_scalar(
        field_conditioned_model_energy_ev,
        name="field_conditioned_model_energy_ev",
    )
    zero_energy = _validated_scalar(
        zero_field_model_energy_ev,
        name="zero_field_model_energy_ev",
    )

    density_dual_field = pairing.field_to_density_order(field)
    density_response_cotangent = _validated_block(
        density_response.vjp(density_dual_field),
        name="density_response.vjp(field)",
        expected_shape=shape,
    )
    model_density_external = pairing.density_to_field_order(model_density)
    scalar_gradient = (
        energy_gradient
        + model_density_external
        + density_response_cotangent
        - zero_gradient
    )
    response_density = pairing.field_to_density_order(scalar_gradient)

    zero_anchor_work = float(np.vdot(zero_gradient, field))
    model_coupling = pairing.pair(model_density, field)
    response_coupling = pairing.pair(response_density, field)
    anchored_scalar_change = (
        field_energy - zero_energy + model_coupling - zero_anchor_work
    )
    electrostatic_energy = anchored_scalar_change - 0.5 * response_coupling

    return Route2V0ScalarResponseState(
        field_conditioned_model_energy_ev=field_energy,
        zero_field_model_energy_ev=zero_energy,
        reaction_field_values_ev=field,
        model_density_coefficients=model_density,
        field_conditioned_energy_field_gradient=energy_gradient,
        zero_field_energy_field_gradient=zero_gradient,
        density_response_cotangent=density_response_cotangent,
        scalar_field_gradient=scalar_gradient,
        response_density_coefficients=response_density,
        zero_field_anchor_work_ev=zero_anchor_work,
        model_density_field_coupling_ev=model_coupling,
        response_density_field_coupling_ev=response_coupling,
        anchored_field_scalar_change_ev=anchored_scalar_change,
        electrostatic_energy_ev=electrostatic_energy,
        response_total_charge_e=float(np.sum(response_density[:, 0])),
    )


__all__ = [
    "DensityResponseVJP",
    "Route2V0ScalarResponseState",
    "evaluate_anchored_scalar_response_v0",
]
