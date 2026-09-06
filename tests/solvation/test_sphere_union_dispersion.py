from __future__ import annotations

import math

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit import (
    sphere_union_dispersion,
)

dispersion_energy_and_gradient = sphere_union_dispersion.dispersion_energy_and_gradient


def _single_sphere_energy(radius, sigma, epsilon, density):
    raw_b = 4.0 * epsilon * sigma**6
    raw_a = raw_b * sigma**6
    lower = max(radius, sigma)
    return (
        4.0 * math.pi * density * (raw_a / (9.0 * lower**9) - raw_b / (3.0 * lower**3))
    )


@pytest.mark.parametrize(("radius", "sigma"), [(1.8, 1.2), (1.1, 1.6)])
def test_single_sphere_matches_analytic_radial_integral(radius, sigma):
    result = dispersion_energy_and_gradient(
        [[3.2, -1.7, 0.4]], [radius], [sigma], [0.21], 0.0334
    )

    assert result.energy == pytest.approx(
        _single_sphere_energy(radius, sigma, 0.21, 0.0334), rel=2e-9, abs=2e-11
    )
    assert result.gradient == pytest.approx(np.zeros((1, 3)), abs=2e-11)
    assert result.diagnostics["quadrature_status"] == 0


def test_zero_epsilon_has_no_lj_contribution():
    result = dispersion_energy_and_gradient(
        [[0, 0, 0], [1.1, 0.2, -0.1]],
        [1.4, 1.2],
        [1.0, 1.1],
        [0.0, 0.0],
        0.02,
    )
    assert result.energy == 0.0
    assert result.gradient == pytest.approx(np.zeros((2, 3)), abs=0.0)


def _asymmetric_case():
    return (
        np.array([[-0.65, 0.10, -0.20], [0.72, -0.31, 0.37], [0.04, 0.91, 0.18]]),
        np.array([1.35, 1.16, 1.02]),
        np.array([0.92, 1.07, 0.81]),
        np.array([0.19, 0.13, 0.22]),
        0.028,
    )


@pytest.mark.parametrize(
    ("positions", "radii", "sigma", "epsilon", "density"),
    [
        (
            np.array([[-0.65, 0.10, -0.20], [0.72, -0.31, 0.37]]),
            np.array([1.35, 1.16]),
            np.array([0.92, 1.07]),
            np.array([0.19, 0.13]),
            0.028,
        ),
        _asymmetric_case(),
    ],
)
def test_all_coordinate_gradients_match_energy_finite_difference(
    positions, radii, sigma, epsilon, density
):
    result = dispersion_energy_and_gradient(
        positions,
        radii,
        sigma,
        epsilon,
        density,
        phi_order=32,
        rtol=2e-8,
        atol=2e-10,
    )
    step = 2e-5
    finite_difference = np.empty_like(positions)
    for atom in range(len(positions)):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            finite_difference[atom, axis] = (
                dispersion_energy_and_gradient(
                    plus,
                    radii,
                    sigma,
                    epsilon,
                    density,
                    phi_order=32,
                    rtol=2e-8,
                    atol=2e-10,
                ).energy
                - dispersion_energy_and_gradient(
                    minus,
                    radii,
                    sigma,
                    epsilon,
                    density,
                    phi_order=32,
                    rtol=2e-8,
                    atol=2e-10,
                ).energy
            ) / (2.0 * step)

    assert result.gradient == pytest.approx(finite_difference, rel=3e-4, abs=3e-6)
    assert result.gradient.sum(axis=0) == pytest.approx(np.zeros(3), abs=2e-9)


def test_rigid_motion_permutation_and_torque_covariance():
    positions, radii, sigma, epsilon, density = _asymmetric_case()
    axis = np.array([1.0, -2.0, 0.7])
    axis /= np.linalg.norm(axis)
    angle = 0.61
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = (
        np.eye(3) * math.cos(angle)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * cross
    )
    reference = dispersion_energy_and_gradient(
        positions, radii, sigma, epsilon, density
    )
    moved = dispersion_energy_and_gradient(
        positions @ rotation.T + [10.0, -7.0, 2.5], radii, sigma, epsilon, density
    )
    order = np.array([2, 0, 1])
    permuted = dispersion_energy_and_gradient(
        positions[order], radii[order], sigma[order], epsilon[order], density
    )

    assert moved.energy == pytest.approx(reference.energy, rel=2e-8, abs=2e-10)
    assert moved.gradient == pytest.approx(reference.gradient @ rotation.T, abs=2e-7)
    assert permuted.energy == pytest.approx(reference.energy, rel=2e-9, abs=2e-11)
    assert permuted.gradient == pytest.approx(reference.gradient[order], abs=2e-8)
    centered = positions - positions.mean(axis=0)
    assert np.sum(np.cross(centered, reference.gradient), axis=0) == pytest.approx(
        np.zeros(3), abs=2e-8
    )


def test_azimuth_order_and_tolerance_refinement_are_stable():
    positions, radii, sigma, epsilon, density = _asymmetric_case()
    coarse = dispersion_energy_and_gradient(
        positions, radii, sigma, epsilon, density, phi_order=24, rtol=2e-6, atol=2e-8
    )
    fine = dispersion_energy_and_gradient(
        positions, radii, sigma, epsilon, density, phi_order=96, rtol=2e-10, atol=2e-12
    )

    assert coarse.energy == pytest.approx(fine.energy, rel=2e-5, abs=2e-7)
    assert coarse.gradient == pytest.approx(fine.gradient, rel=3e-5, abs=2e-7)
    assert fine.quadrature_error >= 0.0
    assert fine.diagnostics["evaluations"] > 0
    assert fine.diagnostics["azimuth_evaluations"] > 0
    assert fine.diagnostics["azimuth_error_estimated"] is False


@pytest.mark.parametrize(
    ("keyword", "value", "exception", "message"),
    [
        ("sas_radii", [0.0], ValueError, "positive"),
        ("sigma", [0.0], ValueError, "positive"),
        ("epsilon", [-0.1], ValueError, "nonnegative"),
        ("density", 0.0, ValueError, "positive"),
        ("density", True, TypeError, "boolean"),
        ("phi_order", 2.5, TypeError, "integer"),
        ("limit", True, TypeError, "integer"),
    ],
)
def test_invalid_inputs_fail_closed(keyword, value, exception, message):
    arguments = {
        "positions": [[0, 0, 0]],
        "sas_radii": [1.2],
        "sigma": [0.9],
        "epsilon": [0.2],
        "density": 0.03,
    }
    arguments[keyword] = value
    with pytest.raises(exception, match=message):
        dispersion_energy_and_gradient(**arguments)


def test_unconverged_quadrature_fails_closed():
    positions, radii, sigma, epsilon, density = _asymmetric_case()
    with pytest.raises(RuntimeError, match=r"quadrature failed \(status 1\)"):
        dispersion_energy_and_gradient(
            positions,
            radii,
            sigma,
            epsilon,
            density,
            phi_order=24,
            rtol=1e-13,
            atol=1e-15,
            limit=1,
        )
