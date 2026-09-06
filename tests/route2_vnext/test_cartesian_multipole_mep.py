from __future__ import annotations

import numpy as np
import pytest
from ase.units import Bohr

from maple.solvation.release.cartesian_multipole_mep import (
    cartesian_atomic_multipole_potential,
    weighted_mep_metrics,
)


def test_cartesian_multipole_terms_match_directional_derivatives() -> None:
    point = np.asarray([[1.2, -0.7, 2.1]], dtype=float)
    radius = float(np.linalg.norm(point[0]))
    charge = 0.3
    dipole_bohr = np.asarray([0.2, -0.1, 0.05])
    quadrupole_bohr2 = np.asarray(
        [[0.4, 0.03, -0.02], [0.03, -0.2, 0.01], [-0.02, 0.01, 0.1]]
    )
    octupole_bohr3 = np.zeros((3, 3, 3))
    octupole_bohr3[0, 0, 0] = 0.6
    octupole_bohr3[0, 1, 2] = 0.07
    octupole_bohr3[0, 2, 1] = 0.07
    octupole_bohr3[1, 0, 2] = 0.07
    octupole_bohr3[1, 2, 0] = 0.07
    octupole_bohr3[2, 0, 1] = 0.07
    octupole_bohr3[2, 1, 0] = 0.07

    value = cartesian_atomic_multipole_potential(
        points_bohr=point,
        centers_angstrom=np.zeros((1, 3)),
        charges_e=np.asarray([charge]),
        dipoles_eangstrom=dipole_bohr[None, :] * Bohr,
        quadrupoles_eangstrom2=quadrupole_bohr2[None, :] * Bohr**2,
        octupoles_eangstrom3=octupole_bohr3[None, :] * Bohr**3,
    )[0]
    d = point[0]
    expected = charge / radius + np.dot(dipole_bohr, d) / radius**3
    expected += (
        0.5
        * (3.0 * d @ quadrupole_bohr2 @ d - radius**2 * np.trace(quadrupole_bohr2))
        / radius**5
    )
    cubic = np.einsum("i,j,k,ijk", d, d, d, octupole_bohr3)
    traces = (
        np.einsum("iik->k", octupole_bohr3)
        + np.einsum("iji->j", octupole_bohr3)
        + np.einsum("ijj->i", octupole_bohr3)
    )
    expected += (15.0 * cubic / radius**7 - 3.0 * d @ traces / radius**5) / 6.0
    assert value == pytest.approx(expected, rel=0.0, abs=2.0e-15)


def test_cartesian_multipole_potential_is_rigid_translation_invariant() -> None:
    rng = np.random.default_rng(20260817)
    centers = rng.normal(size=(4, 3))
    points_bohr = rng.normal(size=(9, 3)) + np.asarray([5.0, -3.0, 4.0])
    shift_angstrom = np.asarray([0.7, -1.1, 0.2])
    arguments = {
        "charges_e": rng.normal(size=4),
        "dipoles_eangstrom": rng.normal(size=(4, 3)),
        "quadrupoles_eangstrom2": rng.normal(size=(4, 3, 3)),
        "octupoles_eangstrom3": rng.normal(size=(4, 3, 3, 3)),
    }
    reference = cartesian_atomic_multipole_potential(
        points_bohr=points_bohr,
        centers_angstrom=centers,
        **arguments,
    )
    translated = cartesian_atomic_multipole_potential(
        points_bohr=points_bohr + shift_angstrom / Bohr,
        centers_angstrom=centers + shift_angstrom,
        **arguments,
    )
    np.testing.assert_allclose(translated, reference, rtol=0.0, atol=3.0e-13)


def test_weighted_mep_metrics_use_normalized_positive_area() -> None:
    metrics = weighted_mep_metrics(
        np.asarray([1.0, 3.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([1.0, 3.0]),
    )
    assert metrics.root_mean_square_error_hartree_per_e == pytest.approx(np.sqrt(3.25))
    assert metrics.mean_absolute_error_hartree_per_e == pytest.approx(1.75)
    assert metrics.maximum_absolute_error_hartree_per_e == 2.0
    assert metrics.reference_root_mean_square_hartree_per_e == pytest.approx(
        np.sqrt(0.75)
    )
    with pytest.raises(ValueError, match="positive area"):
        weighted_mep_metrics([0.0], [0.0], [0.0])
