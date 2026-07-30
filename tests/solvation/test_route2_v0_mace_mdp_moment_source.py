from __future__ import annotations

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.route2_v0_mace_mdp_moment_source import (
    atomic_partition_to_l1_gto_coefficients,
    induced_dipole_source_map,
    induced_source_charge_and_dipole,
)


def _weights() -> np.ndarray:
    return np.asarray(
        [
            [[0.40, 0.02, -0.01], [0.01, 0.25, 0.03], [0.00, -0.02, 0.31]],
            [[0.60, -0.02, 0.01], [-0.01, 0.75, -0.03], [0.00, 0.02, 0.69]],
        ],
        dtype=float,
    )


def test_atomic_moment_partition_maps_to_exact_neutral_dipole_source():
    weights = _weights()
    positions = np.asarray([[-0.4, 0.1, 0.2], [0.7, -0.3, 0.5]])
    molecular_dipole_ebohr = np.asarray([0.37, -0.21, 0.48])

    coefficients = atomic_partition_to_l1_gto_coefficients(
        weights, molecular_dipole_ebohr
    )
    moment = induced_source_charge_and_dipole(
        positions, coefficients, sigma_angstrom=1.5
    )

    assert coefficients.shape == (2, 1, 4)
    np.testing.assert_allclose(coefficients[:, :, 0], 0.0, atol=0.0)
    np.testing.assert_allclose(moment[0], 0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        moment[1:] / Bohr,
        molecular_dipole_ebohr,
        rtol=0.0,
        atol=2.0e-15,
    )


def test_source_map_matches_direct_partition_for_every_linear_combination():
    weights = _weights()
    molecular_dipole_ebohr = np.asarray([-0.53, 0.14, 0.27])

    direct = atomic_partition_to_l1_gto_coefficients(weights, molecular_dipole_ebohr)
    source_map = induced_dipole_source_map(weights)
    via_map = np.einsum("arij,j->ari", source_map, molecular_dipole_ebohr)

    np.testing.assert_allclose(via_map, direct, rtol=0.0, atol=2.0e-15)


@pytest.mark.parametrize(
    "weights, dipole",
    [
        (np.ones((2, 3)), np.ones(3)),
        (np.ones((2, 3, 3)), np.ones(2)),
        (np.asarray([[[np.nan] * 3] * 3]), np.ones(3)),
    ],
)
def test_source_map_rejects_invalid_atomic_partition_inputs(weights, dipole):
    with pytest.raises(ValueError):
        atomic_partition_to_l1_gto_coefficients(weights, dipole)


def test_source_moment_helper_rejects_nonfinite_geometry():
    coefficients = atomic_partition_to_l1_gto_coefficients(_weights(), np.ones(3))
    with pytest.raises(ValueError):
        induced_source_charge_and_dipole(
            np.asarray([[0.0, 0.0, np.nan], [1.0, 0.0, 0.0]]),
            coefficients,
            sigma_angstrom=1.5,
        )
