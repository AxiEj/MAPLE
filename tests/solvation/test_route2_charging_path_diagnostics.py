from __future__ import annotations

import math

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_charging_path_diagnostics import (
    charging_path_energy_diagnostic,
)


def test_charging_path_identity_closes_for_a_variational_synthetic_path():
    lambdas = np.linspace(0.0, 1.0, 5)
    unscaled_polarization = -2.0 - 3.0 * lambdas**2
    model_scalar = np.linspace(10.0, 12.0, lambdas.size)

    diagnostic = charging_path_energy_diagnostic(
        coupling_lambdas=lambdas,
        model_scalar_energies_ev=model_scalar,
        unscaled_polarization_energies_ev=unscaled_polarization,
    )

    assert diagnostic.sample_count == 5
    assert diagnostic.charging_integral_ev == pytest.approx(-3.0)
    assert diagnostic.coarse_charging_integral_ev == pytest.approx(-3.0)
    assert diagnostic.quadrature_refinement_difference_ev == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert diagnostic.quadrature_error_estimate_ev == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert diagnostic.endpoint_model_scalar_change_ev == pytest.approx(2.0)
    assert diagnostic.variational_model_scalar_change_prediction_ev == pytest.approx(
        2.0
    )
    assert diagnostic.endpoint_operational_energy_ev == pytest.approx(-3.0)
    assert diagnostic.model_scalar_change_defect_ev == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert diagnostic.endpoint_vs_charging_defect_ev == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert diagnostic.relative_endpoint_vs_charging_defect == pytest.approx(
        0.0,
        abs=1.0e-15,
    )


def test_charging_path_identity_exposes_a_nonvariational_endpoint_ledger():
    lambdas = np.linspace(0.0, 1.0, 5)
    unscaled_polarization = -2.0 - 3.0 * lambdas**2
    model_scalar = np.linspace(10.0, 12.4, lambdas.size)

    diagnostic = charging_path_energy_diagnostic(
        coupling_lambdas=lambdas,
        model_scalar_energies_ev=model_scalar,
        unscaled_polarization_energies_ev=unscaled_polarization,
    )

    assert diagnostic.model_scalar_change_defect_ev == pytest.approx(0.4)
    assert diagnostic.endpoint_vs_charging_defect_ev == pytest.approx(0.4)
    assert diagnostic.relative_endpoint_vs_charging_defect == pytest.approx(0.4 / 5.6)


def test_nonpolynomial_variational_path_reports_quadrature_uncertainty():
    lambdas = np.linspace(0.0, 1.0, 9)
    unscaled_polarization = np.exp(lambdas)
    model_scalar = 10.0 - lambdas

    diagnostic = charging_path_energy_diagnostic(
        coupling_lambdas=lambdas,
        model_scalar_energies_ev=model_scalar,
        unscaled_polarization_energies_ev=unscaled_polarization,
    )

    exact_integral = math.e - 1.0
    true_quadrature_error = abs(diagnostic.charging_integral_ev - exact_integral)
    assert true_quadrature_error > 0.0
    assert diagnostic.quadrature_error_estimate_ev == pytest.approx(
        true_quadrature_error,
        rel=0.01,
    )
    assert abs(diagnostic.endpoint_vs_charging_defect_ev) == pytest.approx(
        true_quadrature_error
    )
    assert "inconclusive" in diagnostic.interpretation


@pytest.mark.parametrize(
    ("lambdas", "model_scalar", "polarization", "message"),
    (
        (
            [0.0, 0.5, 1.0, 1.5],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            "4k \\+ 1",
        ),
        (
            [0.0, 0.2, 0.5, 0.75, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            "uniform",
        ),
        (
            [0.1, 0.3, 0.5, 0.75, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            "0 and 1",
        ),
        (
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, np.nan, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            "finite",
        ),
        (
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            "shared shape",
        ),
        (
            np.linspace(0.0, 1.0, 7),
            np.zeros(7),
            np.zeros(7),
            "4k \\+ 1",
        ),
    ),
)
def test_charging_path_identity_fails_closed_on_invalid_samples(
    lambdas,
    model_scalar,
    polarization,
    message,
):
    with pytest.raises(ValueError, match=message):
        charging_path_energy_diagnostic(
            coupling_lambdas=lambdas,
            model_scalar_energies_ev=model_scalar,
            unscaled_polarization_energies_ev=polarization,
        )
