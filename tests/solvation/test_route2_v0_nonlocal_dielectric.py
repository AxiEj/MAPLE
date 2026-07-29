from __future__ import annotations

import math

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_nonlocal_dielectric import (
    Route2V0NonlocalDielectricOperator,
    V0_NONLOCAL_DIELECTRIC_CONSTRUCTION,
    V0_NONLOCAL_DIELECTRIC_SCOPE,
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


def _reciprocal_even_dielectric(grid: RegularCartesianGrid) -> np.ndarray:
    reciprocal_axes = tuple(
        2.0 * math.pi * np.fft.fftfreq(grid.shape[axis], d=grid.spacing_bohr[axis])
        for axis in range(3)
    )
    wavevector_squared = sum(
        axis_values**2 for axis_values in np.meshgrid(*reciprocal_axes, indexing="ij")
    )
    return 1.0 + 3.0 * np.exp(-0.18 * wavevector_squared)


def test_nonlocal_dielectric_matches_one_exact_reciprocal_mode_and_zero_gauge():
    grid = _grid()
    dielectric = _reciprocal_even_dielectric(grid)
    operator = Route2V0NonlocalDielectricOperator(grid, dielectric)
    x = grid.spacing_bohr[0] * np.arange(grid.shape[0])
    wavevector = 2.0 * math.pi / operator.cell_lengths_bohr[0]
    charge_density = (
        0.02
        * np.cos(wavevector * x)[:, None, None]
        * np.ones((1, grid.shape[1], grid.shape[2]))
    )
    epsilon_at_mode = dielectric[1, 0, 0]
    multiplier = 4.0 * math.pi * (1.0 / epsilon_at_mode - 1.0) / wavevector**2

    potential = operator.reaction_potential_hartree_per_e(charge_density)

    np.testing.assert_allclose(
        potential,
        multiplier * charge_density,
        rtol=0.0,
        atol=3.0e-15,
    )
    assert abs(float(np.mean(potential))) < 3.0e-16
    assert operator.fourier_reaction_green_bohr2[0, 0, 0] == 0.0
    assert operator.construction == V0_NONLOCAL_DIELECTRIC_CONSTRUCTION
    assert operator.response_scope == V0_NONLOCAL_DIELECTRIC_SCOPE
    dielectric[1, 0, 0] = 1.0
    assert operator.fourier_dielectric_spectrum[1, 0, 0] == pytest.approx(
        epsilon_at_mode
    )
    with pytest.raises(ValueError, match="read-only"):
        operator.fourier_dielectric_spectrum[1, 0, 0] = 1.0


def test_nonlocal_dielectric_scalar_derivative_is_its_reaction_potential():
    rng = np.random.default_rng(20260729)
    operator = Route2V0NonlocalDielectricOperator(
        _grid(),
        _reciprocal_even_dielectric(_grid()),
    )
    density = _neutral_field(operator.grid, rng)
    direction = _neutral_field(operator.grid, rng)
    epsilon = 1.0e-5

    finite_difference = (
        operator.polarization_energy_hartree(density + epsilon * direction)
        - operator.polarization_energy_hartree(density - epsilon * direction)
    ) / (2.0 * epsilon)
    analytic = operator.grid.volume_element_bohr3 * np.sum(
        operator.reaction_potential_hartree_per_e(density) * direction
    )

    assert finite_difference == pytest.approx(analytic, rel=2.0e-9, abs=2.0e-11)


def test_nonlocal_dielectric_pairing_is_reciprocal_passive_and_translation_covariant():
    rng = np.random.default_rng(41)
    grid = _grid()
    operator = Route2V0NonlocalDielectricOperator(
        grid,
        _reciprocal_even_dielectric(grid),
    )
    left = _neutral_field(grid, rng)
    right = _neutral_field(grid, rng)

    assert operator.reaction_pairing_hartree(left, right) == pytest.approx(
        operator.reaction_pairing_hartree(right, left),
        rel=2.0e-13,
        abs=2.0e-13,
    )
    assert operator.polarization_energy_hartree(left) <= 1.0e-14
    assert operator.polarization_energy_hartree(right) <= 1.0e-14

    shift = (3, -2, 1)
    shifted = np.roll(right, shift=shift, axis=(0, 1, 2))
    np.testing.assert_allclose(
        operator.reaction_potential_hartree_per_e(shifted),
        np.roll(
            operator.reaction_potential_hartree_per_e(right),
            shift=shift,
            axis=(0, 1, 2),
        ),
        rtol=0.0,
        atol=3.0e-15,
    )
    assert operator.polarization_energy_hartree(shifted) == pytest.approx(
        operator.polarization_energy_hartree(right),
        rel=2.0e-13,
        abs=2.0e-13,
    )


def test_nonlocal_dielectric_rejects_invalid_spectrum_or_non_neutral_source():
    grid = _grid()
    dielectric = _reciprocal_even_dielectric(grid)

    with pytest.raises(ValueError, match="shape"):
        Route2V0NonlocalDielectricOperator(grid, np.array(78.4))
    with pytest.raises(ValueError, match="epsilon"):
        Route2V0NonlocalDielectricOperator(grid, np.ones(grid.shape) - 1.0e-6)
    with pytest.raises(ValueError, match="real-valued"):
        Route2V0NonlocalDielectricOperator(grid, dielectric.astype(complex))
    with pytest.raises(ValueError, match="reciprocal-even"):
        non_even = dielectric.copy()
        non_even[1, 0, 0] += 0.01
        Route2V0NonlocalDielectricOperator(grid, non_even)
    with pytest.raises(ValueError, match="construction"):
        Route2V0NonlocalDielectricOperator(
            grid,
            dielectric,
            construction="not-route2-v0",
        )
    with pytest.raises(ValueError, match="response scope"):
        Route2V0NonlocalDielectricOperator(
            grid,
            dielectric,
            response_scope="total-solvation",
        )
    with pytest.raises(ValueError, match="reciprocity tolerance"):
        Route2V0NonlocalDielectricOperator(
            grid,
            dielectric,
            reciprocity_relative_tolerance=0.0,
        )

    operator = Route2V0NonlocalDielectricOperator(grid, dielectric)
    non_neutral = np.zeros(grid.shape)
    non_neutral[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="neutral"):
        operator.reaction_potential_hartree_per_e(non_neutral)
    with pytest.raises(ValueError, match="neutral"):
        operator.polarization_energy_hartree(non_neutral)
    with pytest.raises(ValueError, match="neutral"):
        operator.reaction_pairing_hartree(np.zeros(grid.shape), non_neutral)
    with pytest.raises(ValueError, match="tolerance"):
        operator.reaction_potential_hartree_per_e(
            np.zeros(grid.shape),
            neutrality_relative_tolerance=0.0,
        )
