from __future__ import annotations

import numpy as np

from maple.function.calculator.extra_correction.implicit.route2_v0_qeq_gas_response import (
    build_route2_v0_rappe_goddard_gas_monopole_response,
)


def test_qeq_gas_response_is_neutral_passive_and_translation_invariant():
    symbols = ("C", "O", "H")
    positions = np.asarray(
        [[-0.8, 0.3, 0.0], [0.7, -0.4, 0.2], [0.1, 0.8, -0.6]],
        dtype=float,
    )
    response = build_route2_v0_rappe_goddard_gas_monopole_response(
        symbols,
        positions,
    )
    translated = build_route2_v0_rappe_goddard_gas_monopole_response(
        symbols,
        positions + np.asarray([3.7, -2.1, 0.4]),
    )

    assert response.electronic_minimum_neutral_curvature_hartree_per_e2 > (
        response.stability_threshold_hartree_per_e2
    )
    assert np.min(np.linalg.eigvalsh(response.polarizability_bohr3)) >= -1.0e-12
    np.testing.assert_allclose(
        response.polarizability_bohr3,
        translated.polarizability_bohr3,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_qeq_gas_response_matches_analytic_dipole_and_energy_derivatives():
    response = build_route2_v0_rappe_goddard_gas_monopole_response(
        ("C", "O", "H"),
        np.asarray(
            [[-0.8, 0.3, 0.0], [0.7, -0.4, 0.2], [0.1, 0.8, -0.6]],
            dtype=float,
        ),
    )
    field = np.asarray([1.7e-3, -2.2e-3, 0.9e-3])
    state = response.solve_uniform_field(field)

    assert state.charge_constraint_residual_e < 1.0e-12
    assert state.stationarity_residual_inf_hartree_per_e < 1.0e-12
    np.testing.assert_allclose(
        state.induced_dipole_e_bohr,
        response.polarizability_bohr3 @ field,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert state.induction_energy_hartree < 0.0
    np.testing.assert_allclose(
        state.induction_energy_hartree,
        -0.5 * field @ response.polarizability_bohr3 @ field,
        rtol=0.0,
        atol=1.0e-12,
    )

    step = 1.0e-4
    for direction in range(3):
        delta = np.zeros(3)
        delta[direction] = step
        plus = response.solve_uniform_field(delta)
        minus = response.solve_uniform_field(-delta)
        dipole_derivative = (
            plus.induced_dipole_e_bohr - minus.induced_dipole_e_bohr
        ) / (2.0 * step)
        energy_second_derivative = -(
            plus.induction_energy_hartree
            + minus.induction_energy_hartree
        ) / step**2
        np.testing.assert_allclose(
            dipole_derivative,
            response.polarizability_bohr3[:, direction],
            rtol=0.0,
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            energy_second_derivative,
            response.polarizability_bohr3[direction, direction],
            rtol=0.0,
            atol=1.0e-10,
        )
