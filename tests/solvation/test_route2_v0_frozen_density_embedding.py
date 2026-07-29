from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_frozen_density_embedding import (
    Route2V0FrozenDensityPauliOverlap,
    THOMAS_FERMI_KINETIC_COEFFICIENT,
    V0_FROZEN_DENSITY_PAULI_CONSTRUCTION,
    V0_FROZEN_DENSITY_PAULI_SCOPE,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)


def _grid() -> RegularCartesianGrid:
    return RegularCartesianGrid(
        origin_bohr=np.array([-1.0, -1.5, -2.0]),
        spacing_bohr=np.array([0.4, 0.5, 0.6]),
        shape=(7, 6, 5),
    )


def _state(
    solute: np.ndarray,
    solvent: np.ndarray,
) -> Route2V0FrozenDensityPauliOverlap:
    return Route2V0FrozenDensityPauliOverlap(_grid(), solute, solvent)


def test_pauli_overlap_is_nonnegative_exactly_zero_without_overlap_and_exchange_symmetric():
    grid = _grid()
    solute = np.zeros(grid.shape)
    solvent = np.zeros(grid.shape)
    solute[1, 2, 3] = 0.5
    solvent[5, 4, 1] = 0.7

    disjoint = _state(solute, solvent)
    assert disjoint.nonadditive_kinetic_energy_hartree() == 0.0

    solvent[1, 2, 3] = 0.7
    overlap = _state(solute, solvent)
    swapped = _state(solvent, solute)

    assert overlap.nonadditive_kinetic_energy_hartree() > 0.0
    assert overlap.nonadditive_kinetic_energy_hartree() == pytest.approx(
        swapped.nonadditive_kinetic_energy_hartree(), rel=0.0, abs=1.0e-15
    )
    np.testing.assert_allclose(
        overlap.solute_potential_hartree_per_e(),
        swapped.solvent_potential_hartree_per_e(),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert THOMAS_FERMI_KINETIC_COEFFICIENT == pytest.approx(
        0.3 * (3.0 * np.pi**2) ** (2.0 / 3.0)
    )


def test_pauli_overlap_potentials_are_exact_scalar_derivatives():
    rng = np.random.default_rng(20260729)
    grid = _grid()
    solute = 0.2 + rng.random(grid.shape)
    solvent = 0.2 + rng.random(grid.shape)
    solute_direction = rng.normal(size=grid.shape)
    solvent_direction = rng.normal(size=grid.shape)
    state = _state(solute, solvent)
    epsilon = 1.0e-5

    solute_fd = (
        _state(
            solute + epsilon * solute_direction, solvent
        ).nonadditive_kinetic_energy_hartree()
        - _state(
            solute - epsilon * solute_direction, solvent
        ).nonadditive_kinetic_energy_hartree()
    ) / (2.0 * epsilon)
    solute_analytic = grid.volume_element_bohr3 * np.sum(
        state.solute_potential_hartree_per_e() * solute_direction
    )
    solvent_fd = (
        _state(
            solute, solvent + epsilon * solvent_direction
        ).nonadditive_kinetic_energy_hartree()
        - _state(
            solute, solvent - epsilon * solvent_direction
        ).nonadditive_kinetic_energy_hartree()
    ) / (2.0 * epsilon)
    solvent_analytic = grid.volume_element_bohr3 * np.sum(
        state.solvent_potential_hartree_per_e() * solvent_direction
    )

    assert solute_fd == pytest.approx(solute_analytic, rel=2.0e-9, abs=2.0e-10)
    assert solvent_fd == pytest.approx(solvent_analytic, rel=2.0e-9, abs=2.0e-10)


def test_pauli_overlap_is_translation_covariant_without_modifying_inputs():
    rng = np.random.default_rng(41)
    grid = _grid()
    solute = rng.random(grid.shape)
    solvent = rng.random(grid.shape)
    state = _state(solute, solvent)
    shift = (3, -2, 1)
    shifted = _state(
        np.roll(solute, shift=shift, axis=(0, 1, 2)),
        np.roll(solvent, shift=shift, axis=(0, 1, 2)),
    )

    assert shifted.nonadditive_kinetic_energy_hartree() == pytest.approx(
        state.nonadditive_kinetic_energy_hartree(), rel=2.0e-13, abs=2.0e-13
    )
    np.testing.assert_allclose(
        shifted.solute_potential_hartree_per_e(),
        np.roll(
            state.solute_potential_hartree_per_e(),
            shift=shift,
            axis=(0, 1, 2),
        ),
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        shifted.solvent_potential_hartree_per_e(),
        np.roll(
            state.solvent_potential_hartree_per_e(),
            shift=shift,
            axis=(0, 1, 2),
        ),
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(state.solute_electron_density_e_per_bohr3, solute)
    np.testing.assert_allclose(state.solvent_electron_density_e_per_bohr3, solvent)
    with pytest.raises(ValueError, match="read-only"):
        state.solute_electron_density_e_per_bohr3[0, 0, 0] = 0.0


def test_pauli_overlap_rejects_invalid_densities_or_construction_tags():
    grid = _grid()
    density = np.ones(grid.shape)

    with pytest.raises(ValueError, match="shape"):
        _state(np.ones((2, 2, 2)), density)
    negative = density.copy()
    negative[0, 0, 0] = -1.0e-15
    with pytest.raises(ValueError, match="nonnegative"):
        _state(negative, density)
    nonfinite = density.copy()
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _state(density, nonfinite)
    with pytest.raises(ValueError, match="construction"):
        Route2V0FrozenDensityPauliOverlap(
            grid,
            density,
            density,
            construction="not-route2-v0",
        )
    with pytest.raises(ValueError, match="scope"):
        Route2V0FrozenDensityPauliOverlap(
            grid,
            density,
            density,
            interaction_scope="total-solvation",
        )
