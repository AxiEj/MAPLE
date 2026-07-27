"""Independent reference-contract tests for Route-2 aqueous SMD.

These tests intentionally encode primary/reference-implementation constants
rather than deriving expected values from MAPLE's own tables.  They are red on
the audited Route-2 head and must turn green only by correcting the scientific
implementation, not by weakening the expected values.

References
----------
Marenich, Cramer, and Truhlar, J. Phys. Chem. B 2009, 113, 6378-6396.
NWChem ``src/solvation/mnsol.F``.
PySCF ``pyscf/solvent/smd.py`` and ``pyscf/solvent/smd_experiment.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    aqueous_atomic_surface_tension_position_vjp,
    aqueous_atomic_surface_tensions,
    smd_water_coulomb_radii,
)


# Atomic-number-indexed SMD Coulomb radii.  Si=2.47 A is intentionally not in
# MAPLE's current chemical domain; omitting Si must not shift P/S/Cl labels.
_REFERENCE_SUPPORTED_COULOMB_RADII_ANGSTROM = {
    "H": 1.20,
    "C": 1.85,
    "N": 1.89,
    "O": 1.52,
    "F": 1.73,
    "P": 2.12,
    "S": 2.49,
    "Cl": 2.38,
    "Br": 2.60,
    "I": 2.74,
}

_NC3_REFERENCE_DISTANCE_ANGSTROM = 1.225
_NC3_SWITCH_WIDTH_ANGSTROM = 0.065
_NC3_TENSION_COEFFICIENT = 84.10


def _reference_switch_with_derivative(
    distance: float,
    reference: float,
    width: float,
) -> tuple[float, float]:
    """Published SMD switching value and d(value)/d(distance)."""

    if distance >= reference + width:
        return 0.0, 0.0
    denominator = distance - reference - width
    value = math.exp(width / denominator)
    derivative = -width * value / (denominator * denominator)
    return value, derivative


def test_supported_smd_coulomb_radii_follow_atomic_number_reference_table():
    symbols = tuple(_REFERENCE_SUPPORTED_COULOMB_RADII_ANGSTROM)
    expected = np.asarray(
        [_REFERENCE_SUPPORTED_COULOMB_RADII_ANGSTROM[symbol] for symbol in symbols]
    )

    observed = smd_water_coulomb_radii(symbols)

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "distance_angstrom",
    [1.10, 1.20, 1.224, 1.24, 1.28],
)
def test_nitrogen_surface_tension_contains_independent_short_range_nc3_switch(
    distance_angstrom: float,
):
    # A two-atom N-C system makes the ordinary N-C coordination environment
    # exactly zero.  The published +84.10 term therefore remains as a clean,
    # independent N-C3 discriminator with no oxygen or carbonyl atom present.
    symbols = ("N", "C")
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [distance_angstrom, 0.0, 0.0]],
        dtype=float,
    )
    switch, _ = _reference_switch_with_derivative(
        distance_angstrom,
        _NC3_REFERENCE_DISTANCE_ANGSTROM,
        _NC3_SWITCH_WIDTH_ANGSTROM,
    )
    expected_nitrogen_tension = _NC3_TENSION_COEFFICIENT * switch

    observed = aqueous_atomic_surface_tensions(symbols, positions)

    assert observed[0] == pytest.approx(
        expected_nitrogen_tension,
        rel=2.0e-14,
        abs=2.0e-14,
    )
    assert observed[1] == pytest.approx(129.74, rel=0.0, abs=0.0)


def test_nitrogen_nc3_switch_is_zero_outside_its_published_cutoff():
    symbols = ("N", "C")
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [1.30, 0.0, 0.0]],
        dtype=float,
    )

    observed = aqueous_atomic_surface_tensions(symbols, positions)

    assert observed[0] == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_nitrogen_nc3_position_vjp_matches_independent_analytic_reference():
    distance = 1.20
    symbols = ("N", "C")
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
        dtype=float,
    )
    objective_cotangent = np.asarray([1.0, 0.0])
    _, switch_derivative = _reference_switch_with_derivative(
        distance,
        _NC3_REFERENCE_DISTANCE_ANGSTROM,
        _NC3_SWITCH_WIDTH_ANGSTROM,
    )
    direction_n_minus_c = np.asarray([-1.0, 0.0, 0.0])
    nitrogen_gradient = (
        _NC3_TENSION_COEFFICIENT
        * switch_derivative
        * direction_n_minus_c
    )
    expected = np.vstack((nitrogen_gradient, -nitrogen_gradient))

    observed = aqueous_atomic_surface_tension_position_vjp(
        symbols,
        positions,
        objective_cotangent,
    )

    np.testing.assert_allclose(observed, expected, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(observed.sum(axis=0), np.zeros(3), atol=1.0e-14)
