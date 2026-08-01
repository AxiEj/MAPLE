from __future__ import annotations

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.gto_density import (
    point_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.route2_nonuniform_response import (
    localized_point_charge_atom_jets_hartree,
    molecular_dipole_response_from_density_coefficients,
    select_farthest_exterior_point_charge_modes,
    weighted_response_matrix_discrepancy,
)


def test_farthest_point_modes_are_deterministic_and_well_separated():
    points = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -2.0],
        ]
    )
    atoms = np.zeros((1, 3))

    modes = select_farthest_exterior_point_charge_modes(
        points,
        atoms,
        mode_count=4,
    )

    assert modes.source_surface_indices.tolist() == [0, 1, 2, 3]
    assert np.array_equal(modes.source_points_bohr, points[[0, 1, 2, 3]])
    assert not modes.source_points_bohr.flags.writeable


def test_localized_point_charge_jet_is_the_derivative_of_its_potential():
    positions = np.asarray([[0.3, -0.2, 0.4]])
    source = np.asarray([[3.1, -1.2, 2.4]])
    jet = localized_point_charge_atom_jets_hartree(positions, source)[0, 0]
    step_angstrom = 1.0e-5
    for axis in range(3):
        plus = positions.copy()
        minus = positions.copy()
        plus[0, axis] += step_angstrom
        minus[0, axis] -= step_angstrom
        value_plus = localized_point_charge_atom_jets_hartree(plus, source)[0, 0, 0]
        value_minus = localized_point_charge_atom_jets_hartree(minus, source)[0, 0, 0]
        derivative_per_angstrom = (value_plus - value_minus) / (2.0 * step_angstrom)
        assert derivative_per_angstrom == pytest.approx(
            jet[axis + 1] / Bohr,
            rel=2.0e-10,
            abs=2.0e-10,
        )


def test_density_response_dipole_matches_far_field_multipole_moment():
    positions = np.asarray([[0.0, 0.0, 0.0], [1.2, -0.4, 0.7]])
    response = np.asarray(
        [
            [-0.2, 0.3, -0.1, 0.2],
            [0.2, -0.2, 0.4, 0.1],
        ]
    )
    expected = np.asarray([0.3, 0.1, 0.3]) + 0.2 * positions[1]

    observed = molecular_dipole_response_from_density_coefficients(
        positions,
        response,
    )

    assert np.allclose(observed, expected, rtol=0.0, atol=1.0e-15)
    far = np.asarray([[1.0e6, -2.0e6, 3.0e6]])
    potential = point_multipole_potential(far, positions, response)[0]
    far_bohr = far[0]
    dipole_bohr = expected / Bohr
    asymptotic = float(np.dot(far_bohr, dipole_bohr) / np.linalg.norm(far_bohr) ** 3)
    assert potential == pytest.approx(asymptotic, rel=2.0e-6)


def test_weighted_response_matrix_metric_keeps_every_mode_visible():
    reference = np.asarray([[1.0, 2.0], [2.0, -1.0], [0.5, 0.25]])
    candidate = reference.copy()
    candidate[:, 1] += np.asarray([0.2, -0.1, 0.05])
    result = weighted_response_matrix_discrepancy(
        candidate,
        reference,
        np.asarray([1.0, 2.0, 3.0]),
    )

    assert result["weighted_relative_by_mode"][0] == 0.0
    assert result["weighted_relative_by_mode"][1] > 0.0
    assert result["maximum_weighted_relative_mode"] == pytest.approx(
        result["weighted_relative_by_mode"][1]
    )
    assert result["weighted_relative_frobenius"] > 0.0
