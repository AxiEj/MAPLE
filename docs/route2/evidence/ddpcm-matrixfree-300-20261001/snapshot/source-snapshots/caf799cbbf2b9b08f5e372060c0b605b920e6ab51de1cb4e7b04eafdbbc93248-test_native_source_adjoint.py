from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


pytest.importorskip("pyddx")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from native_source_adjoint import solve_native_reference

from maple.function.calculator.extra_correction.implicit.pyddx_pcm_response import (
    PyDDXReactionFieldLinearMap,
)
from maple.solvation.api.units import HARTREE_TO_EV


POSITIONS = np.array([[0.0, 0.0, 0.0], [2.4, 0.2, -0.1]])
RADII = np.array([1.7, 1.5])
SOURCE = np.array([[0.4, 0.1, -0.1, 0.2], [-0.4, -0.05, 0.12, -0.18]])


def test_tiled_source_vjp_matches_existing_basis_projection_at_full_settings() -> None:
    native = solve_native_reference(
        POSITIONS,
        RADII,
        SOURCE,
        lmax=15,
        n_lebedev=1202,
        source_tile=1,
    )
    existing = PyDDXReactionFieldLinearMap(
        POSITIONS,
        RADII,
        continuum_model="pcm",
        dielectric=78.39,
        lmax=15,
        n_lebedev=1202,
        solver_tolerance=1.0e-12,
        eta=0.1,
    )
    expected_energy = existing.polarization_energy_hartree(SOURCE) * HARTREE_TO_EV
    expected_source_gradient = (
        existing._raw_reaction_gradient_hartree(SOURCE) * HARTREE_TO_EV
    )
    expected_position_gradient = (
        existing.polarization_position_gradient_ev_per_angstrom(SOURCE)
    )
    assert native.energy_ev == pytest.approx(expected_energy, abs=2.0e-13)
    np.testing.assert_allclose(
        native.source_gradient_ev,
        expected_source_gradient,
        rtol=0.0,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        native.position_gradient_ev_per_angstrom,
        expected_position_gradient,
        rtol=0.0,
        # Separate native iterative solves can stop at different points within
        # the identical 1e-12 residual contract; the project force gate is
        # 1e-8 eV/A, so keep this comparison below half that gate.
        atol=5.0e-9,
    )
    assert 0.5 * np.vdot(SOURCE, native.source_gradient_ev) == pytest.approx(
        native.energy_ev, abs=3.0e-12
    )
    assert native.forward_iterations > 0
    assert native.adjoint_iterations > 0


def test_native_fixed_source_position_gradient_matches_central_difference() -> None:
    native = solve_native_reference(POSITIONS, RADII, SOURCE, lmax=2, n_lebedev=50)
    step = 1.0e-5
    numerical = np.empty_like(POSITIONS)
    for atom in range(len(POSITIONS)):
        for axis in range(3):
            plus = POSITIONS.copy()
            minus = POSITIONS.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            e_plus = solve_native_reference(
                plus, RADII, SOURCE, lmax=2, n_lebedev=50
            ).energy_ev
            e_minus = solve_native_reference(
                minus, RADII, SOURCE, lmax=2, n_lebedev=50
            ).energy_ev
            numerical[atom, axis] = (e_plus - e_minus) / (2.0 * step)
    np.testing.assert_allclose(
        native.position_gradient_ev_per_angstrom,
        numerical,
        rtol=0.0,
        atol=3.0e-8,
    )


@pytest.mark.parametrize("source_tile", [True, 1.5, 0])
def test_source_tile_rejects_non_positive_integers(source_tile) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        solve_native_reference(
            POSITIONS,
            RADII,
            SOURCE,
            lmax=2,
            n_lebedev=50,
            source_tile=source_tile,
        )


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), 1.0e-11])
def test_native_tolerance_cannot_relax_or_be_nonfinite(tolerance: float) -> None:
    with pytest.raises(ValueError, match="at most 1e-12"):
        solve_native_reference(
            POSITIONS,
            RADII,
            SOURCE,
            lmax=2,
            n_lebedev=50,
            tolerance=tolerance,
        )


def test_native_inputs_reject_nonfinite_or_nonpositive_geometry() -> None:
    bad_positions = POSITIONS.copy()
    bad_positions[0, 0] = np.nan
    with pytest.raises(ValueError, match="positions_angstrom"):
        solve_native_reference(bad_positions, RADII, SOURCE, lmax=2, n_lebedev=50)
    bad_radii = RADII.copy()
    bad_radii[0] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        solve_native_reference(POSITIONS, bad_radii, SOURCE, lmax=2, n_lebedev=50)
