from __future__ import annotations

import math

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.route2_v0_periodic_coulomb import (
    Route2V0PeriodicCoulombOperator,
    V0_PERIODIC_COULOMB_CONSTRUCTION,
    amber_rism_qv_to_sqrt_bohr,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)


def _grid(
    shape: tuple[int, int, int] = (10, 8, 6),
    cell_lengths_bohr: tuple[float, float, float] = (4.0, 5.0, 6.0),
) -> RegularCartesianGrid:
    return RegularCartesianGrid(
        origin_bohr=np.array([-1.0, -1.5, -2.0]),
        spacing_bohr=np.asarray(cell_lengths_bohr) / np.asarray(shape),
        shape=shape,
    )


def _neutral_field(
    grid: RegularCartesianGrid,
    rng: np.random.Generator,
) -> np.ndarray:
    result = rng.normal(size=grid.shape)
    result -= np.mean(result)
    return result


def _neutral_site_density(
    grid: RegularCartesianGrid,
    scales: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    result = rng.normal(scale=0.02, size=(scales.size, *grid.shape))
    weighted = np.einsum("a,axyz->xyz", scales, result, optimize=True)
    result[0] -= np.mean(weighted) / scales[0]
    return result


def test_periodic_poisson_matches_one_exact_reciprocal_mode_and_zero_gauge():
    grid = _grid()
    operator = Route2V0PeriodicCoulombOperator(grid)
    x = grid.spacing_bohr[0] * np.arange(grid.shape[0])
    wavevector = 2.0 * math.pi / operator.cell_lengths_bohr[0]
    charge_density = (
        0.02
        * np.cos(wavevector * x)[:, None, None]
        * np.ones((1, grid.shape[1], grid.shape[2]))
    )

    potential = operator.potential_from_charge_density(charge_density)
    expected = 4.0 * math.pi * charge_density / wavevector**2

    np.testing.assert_allclose(potential, expected, rtol=0.0, atol=3.0e-15)
    assert abs(float(np.mean(potential))) < 3.0e-16
    assert operator.fourier_green_bohr2[0, 0, 0] == 0.0
    assert operator.construction == V0_PERIODIC_COULOMB_CONSTRUCTION


def test_periodic_poisson_preserves_the_source_defined_gaussian_smear_factor():
    grid = _grid()
    smear_bohr = 0.8
    operator = Route2V0PeriodicCoulombOperator(grid, smear_bohr=smear_bohr)
    x = grid.spacing_bohr[0] * np.arange(grid.shape[0])
    wavevector = 2.0 * math.pi / operator.cell_lengths_bohr[0]
    charge_density = (
        0.02
        * np.cos(wavevector * x)[:, None, None]
        * np.ones((1, grid.shape[1], grid.shape[2]))
    )
    damping = math.exp(-0.25 * smear_bohr**2 * wavevector**2)

    potential = operator.potential_from_charge_density(charge_density)
    expected = 4.0 * math.pi * damping * charge_density / wavevector**2

    np.testing.assert_allclose(potential, expected, rtol=0.0, atol=3.0e-15)
    assert operator.smear_bohr == pytest.approx(smear_bohr)


def test_periodic_coulomb_pairing_is_reciprocal_and_translation_covariant():
    rng = np.random.default_rng(20260729)
    operator = Route2V0PeriodicCoulombOperator(_grid())
    left = _neutral_field(operator.grid, rng)
    right = _neutral_field(operator.grid, rng)
    potential_left = operator.potential_from_charge_density(left)
    potential_right = operator.potential_from_charge_density(right)
    volume = operator.grid.volume_element_bohr3

    lhs = volume * np.sum(left * potential_right)
    rhs = volume * np.sum(right * potential_left)
    assert lhs == pytest.approx(rhs, rel=2.0e-13, abs=2.0e-13)

    shift = (3, -2, 1)
    shifted = np.roll(right, shift=shift, axis=(0, 1, 2))
    shifted_potential = operator.potential_from_charge_density(shifted)
    np.testing.assert_allclose(
        shifted_potential,
        np.roll(potential_right, shift=shift, axis=(0, 1, 2)),
        rtol=0.0,
        atol=3.0e-15,
    )
    assert operator.dimensionless_energy(shifted) == pytest.approx(
        operator.dimensionless_energy(right),
        rel=2.0e-13,
        abs=2.0e-13,
    )


def test_site_kernel_is_the_derivative_of_the_same_periodic_coulomb_scalar():
    rng = np.random.default_rng(29)
    operator = Route2V0PeriodicCoulombOperator(_grid())
    scales = np.array([1.5, -0.75])
    density = _neutral_site_density(operator.grid, scales, rng)
    direction = _neutral_site_density(operator.grid, scales, rng)
    epsilon = 1.0e-5

    finite_difference = (
        operator.dimensionless_energy_from_site_density_difference(
            density + epsilon * direction,
            scales,
        )
        - operator.dimensionless_energy_from_site_density_difference(
            density - epsilon * direction,
            scales,
        )
    ) / (2.0 * epsilon)
    kernel = operator.site_kernel_from_density_difference(density, scales)
    analytic = operator.grid.volume_element_bohr3 * np.sum(kernel * direction)

    assert finite_difference == pytest.approx(analytic, rel=2.0e-9, abs=2.0e-11)


def test_periodic_coulomb_grid_refinement_preserves_an_exact_mode_energy():
    energies = []
    for shape in ((8, 8, 8), (16, 16, 16)):
        operator = Route2V0PeriodicCoulombOperator(
            _grid(shape, cell_lengths_bohr=(4.0, 4.0, 4.0))
        )
        x = operator.grid.spacing_bohr[0] * np.arange(shape[0])
        wavevector = 2.0 * math.pi / 4.0
        amplitude = 0.015
        charge_density = (
            amplitude
            * np.cos(wavevector * x)[:, None, None]
            * np.ones((1, shape[1], shape[2]))
        )
        energy = operator.dimensionless_energy(charge_density)
        expected = (
            math.pi
            * amplitude**2
            * np.prod(operator.cell_lengths_bohr)
            / (wavevector**2)
        )
        assert energy == pytest.approx(expected, rel=2.0e-13, abs=2.0e-13)
        energies.append(energy)

    assert energies[0] == pytest.approx(energies[1], rel=2.0e-13, abs=2.0e-13)


def test_periodic_coulomb_rejects_non_neutral_density_instead_of_background():
    operator = Route2V0PeriodicCoulombOperator(_grid())
    non_neutral = np.zeros(operator.grid.shape)
    non_neutral[0, 0, 0] = 1.0

    with pytest.raises(ValueError, match="neutral"):
        operator.potential_from_charge_density(non_neutral)
    with pytest.raises(ValueError, match="neutral"):
        operator.dimensionless_energy(non_neutral)
    with pytest.raises(ValueError, match="tolerance"):
        operator.potential_from_charge_density(
            np.zeros(operator.grid.shape),
            neutrality_relative_tolerance=0.0,
        )
    with pytest.raises(ValueError, match="SMEAR"):
        Route2V0PeriodicCoulombOperator(_grid(), smear_bohr=-0.1)


def test_rism_qv_length_conversion_and_site_shape_validation_are_explicit():
    qv = np.array([-20.071094037563302, 10.035547018781651])
    np.testing.assert_allclose(
        amber_rism_qv_to_sqrt_bohr(qv),
        qv / math.sqrt(Bohr),
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="nonempty"):
        amber_rism_qv_to_sqrt_bohr(np.array([]))
    operator = Route2V0PeriodicCoulombOperator(_grid())
    with pytest.raises(ValueError, match="shape"):
        operator.site_kernel_from_density_difference(
            np.zeros(operator.grid.shape),
            np.array([1.0]),
        )
