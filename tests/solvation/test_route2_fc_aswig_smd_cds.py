from __future__ import annotations

import math

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.route2_fc_aswig_smd_cds import (
    FixedTopologyAqueousSMDCDS,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    HARTREE_TO_KCAL_MOL,
    aqueous_atomic_surface_tensions,
    smd_sasa_radii,
)


_SIX_POINT_SPHERE = np.asarray(
    [
        [1.0, 0.0, 0.0, 1.0 / 6.0],
        [-1.0, 0.0, 0.0, 1.0 / 6.0],
        [0.0, 1.0, 0.0, 1.0 / 6.0],
        [0.0, -1.0, 0.0, 1.0 / 6.0],
        [0.0, 0.0, 1.0, 1.0 / 6.0],
        [0.0, 0.0, -1.0, 1.0 / 6.0],
    ],
    dtype=float,
)
_SWITCHING_CONSTANT = 4.84566077868


def _cds(symbols: tuple[str, ...], positions: np.ndarray) -> FixedTopologyAqueousSMDCDS:
    return FixedTopologyAqueousSMDCDS(
        symbols,
        positions,
        _unit_sphere=_SIX_POINT_SPHERE,
        _switching_constant=_SWITCHING_CONSTANT,
    )


def test_isolated_atom_has_exact_full_area_zero_gradient_and_explicit_provenance():
    cds = _cds(("H",), np.zeros((1, 3)))
    result = cds.result()
    radius = float(smd_sasa_radii(("H",))[0])
    expected_area = 4.0 * math.pi * radius**2
    tension = float(aqueous_atomic_surface_tensions(("H",), np.zeros((1, 3)))[0])
    expected_energy = tension * expected_area / (1000.0 * HARTREE_TO_KCAL_MOL)

    assert result.surface_size == 6
    assert result.atom_areas_angstrom2[0] == pytest.approx(expected_area)
    assert result.energy_hartree == pytest.approx(expected_energy)
    np.testing.assert_allclose(
        result.position_gradient_hartree_per_angstrom,
        np.zeros((1, 3)),
        rtol=0.0,
        atol=1.0e-16,
    )
    provenance = cds.runtime_provenance
    assert provenance["strict_original_smd_equivalence"] is False
    assert provenance["same_scalar_coordinate_derivative"] is True


def test_fixed_topology_cds_keeps_all_candidates_at_a_burial_crossing():
    cds = _cds(
        ("H", "H"),
        np.asarray([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
    )
    result = cds.result()
    assert result.surface_size == 12
    amplitudes = cds.surface.exposure_amplitudes
    assert np.count_nonzero(amplitudes == 0.0) >= 2
    assert result.surface_size == 2 * result.grid_points_per_atom


def test_cds_gradient_matches_the_same_scalar_finite_difference_through_switch():
    symbols = ("H", "C")
    positions = np.asarray([[0.0, 0.0, 0.0], [2.5, 0.2, -0.1]])
    analytic = _cds(symbols, positions).result().position_gradient_hartree_per_angstrom

    def energy(displaced: np.ndarray) -> float:
        return _cds(symbols, displaced).result().energy_hartree

    step = 1.0e-5
    finite_difference = np.empty_like(analytic)
    for atom_index in range(len(symbols)):
        for cartesian_index in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom_index, cartesian_index] += step
            minus[atom_index, cartesian_index] -= step
            finite_difference[atom_index, cartesian_index] = (
                energy(plus) - energy(minus)
            ) / (2.0 * step)

    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=0.0,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        np.sum(analytic, axis=0),
        np.zeros(3),
        rtol=0.0,
        atol=2.0e-15,
    )


def test_cds_central_difference_stabilizes_under_switch_refinement():
    symbols = ("H", "H")

    def energy(distance: float) -> float:
        return _cds(
            symbols,
            np.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
        ).result().energy_hartree

    center = 2.5
    steps = np.asarray([0.10, 0.05, 0.025, 0.0125, 0.00625])
    derivatives = np.asarray(
        [(energy(center + step) - energy(center - step)) / (2.0 * step) for step in steps]
    )
    deltas = np.abs(np.diff(derivatives))
    assert np.all(deltas[1:] < deltas[:-1])
    assert np.all(deltas[:-1] / deltas[1:] > 3.0)


def test_surface_area_conversion_is_from_bohr_squared_not_an_empirical_scale():
    cds = _cds(("H", "H"), np.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    result = cds.result()
    parent = cds.surface.parent_atom_indices
    raw_bohr2 = np.bincount(
        parent,
        weights=cds.surface.effective_areas_bohr2,
        minlength=2,
    )
    np.testing.assert_allclose(result.atom_areas_angstrom2, raw_bohr2 * Bohr**2)
