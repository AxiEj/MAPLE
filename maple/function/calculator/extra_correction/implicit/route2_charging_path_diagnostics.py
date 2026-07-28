"""Data-only diagnostics for Route-2 thermodynamic charging paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_SIMPSON_SAMPLE_COUNT_ERROR = (
    "Nested charging-path Simpson quadrature requires a 4k + 1 sample count "
    "of at least five."
)


def _symmetric_relative_defect(
    absolute_defect: float,
    left_magnitude: float,
    right_magnitude: float,
) -> float:
    scale = left_magnitude + right_magnitude
    if scale == 0.0:
        return 0.0 if absolute_defect == 0.0 else float("inf")
    return absolute_defect / scale


def _uniform_composite_simpson(values: np.ndarray, spacing: float) -> float:
    return float(
        spacing
        / 3.0
        * (
            values[0]
            + values[-1]
            + 4.0 * np.sum(values[1:-1:2])
            + 2.0 * np.sum(values[2:-1:2])
        )
    )


@dataclass(frozen=True)
class ChargingPathEnergyDiagnostic:
    """Endpoint-ledger defect against one preregistered charging integral."""

    sample_count: int
    charging_integral_ev: float
    coarse_charging_integral_ev: float
    quadrature_refinement_difference_ev: float
    quadrature_error_estimate_ev: float
    endpoint_model_scalar_change_ev: float
    endpoint_unscaled_polarization_ev: float
    endpoint_operational_energy_ev: float
    variational_model_scalar_change_prediction_ev: float
    model_scalar_change_defect_ev: float
    endpoint_vs_charging_defect_ev: float
    relative_endpoint_vs_charging_defect: float
    quadrature: str = "nested-uniform-composite-simpson-richardson-v1"
    model_scalar_semantics: str = (
        "The same field-conditioned model scalar used in the endpoint "
        "ledger, excluding the explicit continuum polarization energy; it "
        "is not presumed intrinsic or variational."
    )
    identity: str = "Delta G_charge = integral_0^1 U(c_lambda) d lambda"
    endpoint_ledger: str = "Delta E_model + U(c_1)"
    interpretation: str = (
        "A zero defect establishes internal charging-path consistency for "
        "the sampled operational ledger only; it does not establish QM "
        "accuracy or a variational learned density model. A nonzero defect is "
        "inconclusive unless it is stable to further grid refinement and "
        "materially exceeds the reported quadrature error estimate."
    )

    def __post_init__(self) -> None:
        if self.sample_count < 5 or self.sample_count % 4 != 1:
            raise ValueError(_SIMPSON_SAMPLE_COUNT_ERROR)
        signed_metrics = (
            self.charging_integral_ev,
            self.coarse_charging_integral_ev,
            self.quadrature_refinement_difference_ev,
            self.endpoint_model_scalar_change_ev,
            self.endpoint_unscaled_polarization_ev,
            self.endpoint_operational_energy_ev,
            self.variational_model_scalar_change_prediction_ev,
            self.model_scalar_change_defect_ev,
            self.endpoint_vs_charging_defect_ev,
        )
        if not all(np.isfinite(value) for value in signed_metrics):
            raise ValueError("Charging-path energy metrics must be finite.")
        if (
            not np.isfinite(self.quadrature_error_estimate_ev)
            or self.quadrature_error_estimate_ev < 0.0
        ):
            raise ValueError(
                "The charging-path quadrature error estimate must be finite "
                "and nonnegative."
            )
        if (
            not np.isfinite(self.relative_endpoint_vs_charging_defect)
            or self.relative_endpoint_vs_charging_defect < 0.0
        ):
            raise ValueError(
                "The relative charging-path defect must be finite and nonnegative."
            )
        if self.quadrature != "nested-uniform-composite-simpson-richardson-v1":
            raise ValueError("Unsupported charging-path quadrature.")


def charging_path_energy_diagnostic(
    *,
    coupling_lambdas: np.ndarray,
    model_scalar_energies_ev: np.ndarray,
    unscaled_polarization_energies_ev: np.ndarray,
) -> ChargingPathEnergyDiagnostic:
    """Compare the endpoint Route-2 ledger with thermodynamic integration.

    At each coupling value ``lambda``, the caller must solve the same
    algorithmic response equation with the reaction field scaled by
    ``lambda`` and record the unscaled polarization energy

    ``U(c_lambda) = 1/2 <c_lambda, P c_lambda>``.

    If a joint variational functional
    ``F_lambda = E_0[c] + lambda U[c]`` exists, the envelope theorem gives

    ``Delta F = integral_0^1 U(c_lambda) d lambda``.

    ``model_scalar_energies_ev`` must contain the same field-conditioned model
    scalar used by the endpoint Route-2 ledger, before the explicit continuum
    polarization energy is added. The name does not assert that this scalar is
    intrinsic or variational; the diagnostic tests that candidate
    interpretation.

    The endpoint candidate ``Delta E_model + U(c_1)`` must then equal the
    charging integral. This function reports the signed discrepancy without
    changing or selecting a production energy ledger. The relative defect is
    symmetrically normalized as ``abs(A-C)/(abs(A)+abs(C))``. Fine and
    every-other-point Simpson estimates provide the Richardson error estimate
    ``abs(S_fine-S_coarse)/15``. A nonzero endpoint defect alone is
    inconclusive unless it is stable to further refinement and materially
    larger than that quadrature estimate.
    """

    lambdas = np.asarray(coupling_lambdas, dtype=float)
    model_energies = np.asarray(model_scalar_energies_ev, dtype=float)
    polarization_energies = np.asarray(
        unscaled_polarization_energies_ev,
        dtype=float,
    )
    if (
        lambdas.ndim != 1
        or model_energies.ndim != 1
        or polarization_energies.ndim != 1
        or lambdas.shape != model_energies.shape
        or lambdas.shape != polarization_energies.shape
    ):
        raise ValueError(
            "Charging-path samples must be one-dimensional with one shared shape."
        )
    sample_count = int(lambdas.size)
    if sample_count < 5 or sample_count % 4 != 1:
        raise ValueError(_SIMPSON_SAMPLE_COUNT_ERROR)
    if not (
        np.all(np.isfinite(lambdas))
        and np.all(np.isfinite(model_energies))
        and np.all(np.isfinite(polarization_energies))
    ):
        raise ValueError("Charging-path samples must be finite.")
    if lambdas[0] != 0.0 or lambdas[-1] != 1.0:
        raise ValueError("Charging-path coupling lambda endpoints must be 0 and 1.")
    steps = np.diff(lambdas)
    if np.any(steps <= 0.0):
        raise ValueError("Charging-path coupling lambdas must be strictly increasing.")
    if not np.allclose(steps, steps[0], rtol=1.0e-12, atol=1.0e-15):
        raise ValueError(
            "Charging-path Simpson quadrature requires a uniform lambda grid."
        )

    charging_integral = _uniform_composite_simpson(
        polarization_energies,
        float(steps[0]),
    )
    coarse_charging_integral = _uniform_composite_simpson(
        polarization_energies[::2],
        float(2.0 * steps[0]),
    )
    refinement_difference = charging_integral - coarse_charging_integral
    quadrature_error_estimate = abs(refinement_difference) / 15.0
    model_change = float(model_energies[-1] - model_energies[0])
    endpoint_polarization = float(polarization_energies[-1])
    endpoint_operational = model_change + endpoint_polarization
    predicted_model_change = charging_integral - endpoint_polarization
    model_change_defect = model_change - predicted_model_change
    endpoint_defect = endpoint_operational - charging_integral
    relative_defect = _symmetric_relative_defect(
        abs(endpoint_defect),
        abs(endpoint_operational),
        abs(charging_integral),
    )
    return ChargingPathEnergyDiagnostic(
        sample_count=sample_count,
        charging_integral_ev=charging_integral,
        coarse_charging_integral_ev=coarse_charging_integral,
        quadrature_refinement_difference_ev=refinement_difference,
        quadrature_error_estimate_ev=quadrature_error_estimate,
        endpoint_model_scalar_change_ev=model_change,
        endpoint_unscaled_polarization_ev=endpoint_polarization,
        endpoint_operational_energy_ev=endpoint_operational,
        variational_model_scalar_change_prediction_ev=predicted_model_change,
        model_scalar_change_defect_ev=model_change_defect,
        endpoint_vs_charging_defect_ev=endpoint_defect,
        relative_endpoint_vs_charging_defect=relative_defect,
    )


__all__ = [
    "ChargingPathEnergyDiagnostic",
    "charging_path_energy_diagnostic",
]
