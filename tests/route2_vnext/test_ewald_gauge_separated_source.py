from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.gto_density import (
    gaussian_multipole_potential,
    point_multipole_potential,
)
from maple.solvation.release.ewald_gauge_separated_source import (
    EwaldGaugeSeparatedPermanentSource,
    build_ewald_gauge_separated_permanent_source,
)


def _raw(charges: np.ndarray, dipoles_xyz: np.ndarray) -> np.ndarray:
    return np.concatenate((charges[:, None], dipoles_xyz[:, (1, 2, 0)]), axis=1)


def test_direct_sum_closes_exact_mdp_charge_and_dipole() -> None:
    positions = np.array([[0.2, -0.5, 0.3], [1.1, 0.4, -0.7]])
    mdp = _raw(np.array([0.35, -0.35]), np.array([[0.2, 0.1, -0.3], [0.4, -0.2, 0.1]]))
    polar = _raw(
        np.array([0.15, -0.15]), np.array([[-0.1, 0.3, 0.2], [0.1, 0.2, -0.2]])
    )
    candidate = build_ewald_gauge_separated_permanent_source(
        mdp_source4_raw_l1=mdp,
        polar_zero_source4_raw_l1=polar,
    )

    np.testing.assert_allclose(
        candidate.far_field_source4_raw_l1, mdp, rtol=0.0, atol=3.0e-17
    )
    closure = candidate.molecular_closure(positions)
    expected_dipole = np.sum(
        np.array([0.35, -0.35])[:, None] * positions
        + np.array([[0.2, 0.1, -0.3], [0.4, -0.2, 0.1]]),
        axis=0,
    )
    assert closure["far_field_total_charge_e"] == pytest.approx(0.0, abs=2e-16)
    np.testing.assert_allclose(
        closure["far_field_molecular_dipole_eangstrom"],
        expected_dipole,
        rtol=0.0,
        atol=2e-16,
    )


def test_far_field_converges_to_unscreened_mdp_multipoles() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.9, -0.2, 0.1]])
    mdp = _raw(np.array([0.4, -0.4]), np.array([[0.2, 0.0, 0.1], [0.0, -0.1, 0.2]]))
    polar = _raw(np.array([0.1, -0.1]), np.array([[0.0, 0.2, 0.0], [0.1, 0.0, -0.1]]))
    candidate = build_ewald_gauge_separated_permanent_source(
        mdp_source4_raw_l1=mdp,
        polar_zero_source4_raw_l1=polar,
    )
    points_bohr = np.array([[400.0, 170.0, -90.0], [-330.0, 260.0, 210.0]])
    actual = point_multipole_potential(
        points_bohr, positions, candidate.point_source4_raw_l1
    ) + gaussian_multipole_potential(
        points_bohr,
        positions,
        candidate.gaussian_correction4_raw_l1,
        sigma_angstrom=candidate.sigma_angstrom,
    )
    expected = point_multipole_potential(points_bohr, positions, mdp)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)


def test_candidate_is_rotation_translation_covariant() -> None:
    positions = np.array([[0.1, 0.2, -0.3], [1.2, -0.4, 0.5]])
    charges = np.array([0.25, -0.25])
    mdp_dipoles = np.array([[0.2, 0.1, -0.2], [0.3, -0.1, 0.4]])
    polar_dipoles = np.array([[0.0, -0.2, 0.1], [0.1, 0.2, -0.1]])
    mdp = _raw(charges, mdp_dipoles)
    polar = _raw(charges, polar_dipoles)
    angle = 0.71
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    shift = np.array([2.3, -1.7, 0.8])
    transformed_positions = positions @ rotation.T + shift
    candidate = build_ewald_gauge_separated_permanent_source(
        mdp_source4_raw_l1=mdp,
        polar_zero_source4_raw_l1=polar,
    )
    transformed = build_ewald_gauge_separated_permanent_source(
        mdp_source4_raw_l1=_raw(charges, mdp_dipoles @ rotation.T),
        polar_zero_source4_raw_l1=_raw(charges, polar_dipoles @ rotation.T),
    )
    original_mu = candidate.molecular_closure(positions)[
        "far_field_molecular_dipole_eangstrom"
    ]
    transformed_mu = transformed.molecular_closure(transformed_positions)[
        "far_field_molecular_dipole_eangstrom"
    ]
    np.testing.assert_allclose(transformed_mu, original_mu @ rotation.T, atol=3e-16)


def test_charge_mismatch_and_nonofficial_width_fail_closed() -> None:
    mdp = np.zeros((2, 4))
    polar = np.zeros((2, 4))
    mdp[0, 0] = 0.1
    with pytest.raises(RuntimeError, match="same total charge"):
        build_ewald_gauge_separated_permanent_source(
            mdp_source4_raw_l1=mdp,
            polar_zero_source4_raw_l1=polar,
        )
    with pytest.raises(ValueError, match="official MACE-POLAR width"):
        EwaldGaugeSeparatedPermanentSource(
            point_source4_raw_l1=polar,
            gaussian_correction4_raw_l1=polar,
            sigma_angstrom=1.6,
        )
