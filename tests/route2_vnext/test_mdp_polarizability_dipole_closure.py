from __future__ import annotations

import numpy as np
import pytest

from maple.solvation.release.mdp_polarizability_dipole_closure import (
    close_dipole_with_mdp_polarizability,
    molecular_dipole_eangstrom,
)


def _source() -> np.ndarray:
    return np.asarray([[0.2, 0.01, -0.02, 0.03], [-0.2, -0.04, 0.02, 0.01]])


def _positions() -> np.ndarray:
    return np.asarray([[0.0, 0.0, 0.0], [1.2, -0.3, 0.4]])


def _atomic_alpha() -> np.ndarray:
    return np.asarray(
        [
            [[0.8, 0.1, 0.0], [0.1, 0.5, 0.0], [0.0, 0.0, 0.4]],
            [[0.4, -0.1, 0.0], [-0.1, 0.7, 0.0], [0.0, 0.0, 0.6]],
        ]
    )


def _rotate_raw_l1(source: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = source.copy()
    cartesian = source[:, (3, 1, 2)] @ rotation.T
    result[:, 1] = cartesian[:, 1]
    result[:, 2] = cartesian[:, 2]
    result[:, 3] = cartesian[:, 0]
    return result


def test_closure_is_exact_and_matches_uniform_linear_response() -> None:
    source = _source()
    positions = _positions()
    atomic = _atomic_alpha()
    total = np.sum(atomic, axis=0)
    target = molecular_dipole_eangstrom(source, positions) + np.asarray(
        [0.05, -0.03, 0.02]
    )
    result = close_dipole_with_mdp_polarizability(
        source4_raw_l1=source,
        positions_angstrom=positions,
        target_molecular_dipole_eangstrom=target,
        atomic_polarizabilities_eangstrom2_per_volt=atomic,
        molecular_polarizability_eangstrom2_per_volt=total,
    )
    expected_field = np.linalg.solve(
        total, target - molecular_dipole_eangstrom(source, positions)
    )
    assert np.allclose(
        result.equivalent_uniform_field_volt_per_angstrom, expected_field
    )
    assert np.allclose(result.dipole_correction_eangstrom, atomic @ expected_field)
    assert np.allclose(result.source_molecular_dipole_eangstrom, target, atol=1.0e-14)
    assert np.array_equal(result.source4_raw_l1[:, 0], source[:, 0])
    assert result.closure_residual_eangstrom < 1.0e-14


def test_translation_and_rotation_covariance() -> None:
    source = _source()
    positions = _positions()
    atomic = _atomic_alpha()
    total = np.sum(atomic, axis=0)
    target = molecular_dipole_eangstrom(source, positions) + np.asarray(
        [0.05, -0.03, 0.02]
    )
    reference = close_dipole_with_mdp_polarizability(
        source4_raw_l1=source,
        positions_angstrom=positions,
        target_molecular_dipole_eangstrom=target,
        atomic_polarizabilities_eangstrom2_per_volt=atomic,
        molecular_polarizability_eangstrom2_per_volt=total,
    )
    angle = 0.73
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = close_dipole_with_mdp_polarizability(
        source4_raw_l1=_rotate_raw_l1(source, rotation),
        positions_angstrom=positions @ rotation.T + np.asarray([2.1, -0.4, 0.8]),
        target_molecular_dipole_eangstrom=target @ rotation.T,
        atomic_polarizabilities_eangstrom2_per_volt=np.einsum(
            "ij,ajk,lk->ail", rotation, atomic, rotation
        ),
        molecular_polarizability_eangstrom2_per_volt=rotation @ total @ rotation.T,
    )
    assert np.allclose(
        rotated.source4_raw_l1,
        _rotate_raw_l1(reference.source4_raw_l1, rotation),
        atol=2.0e-14,
    )
    assert np.allclose(
        rotated.equivalent_uniform_field_volt_per_angstrom,
        reference.equivalent_uniform_field_volt_per_angstrom @ rotation.T,
        atol=2.0e-14,
    )


def test_indefinite_atomic_contributions_are_allowed_but_total_must_be_spd() -> None:
    source = _source()
    positions = _positions()
    atomic = _atomic_alpha()
    atomic[0, 0, 0] = -0.1
    total = np.sum(atomic, axis=0)
    target = molecular_dipole_eangstrom(source, positions) + np.asarray(
        [0.01, 0.0, 0.0]
    )
    result = close_dipole_with_mdp_polarizability(
        source4_raw_l1=source,
        positions_angstrom=positions,
        target_molecular_dipole_eangstrom=target,
        atomic_polarizabilities_eangstrom2_per_volt=atomic,
        molecular_polarizability_eangstrom2_per_volt=total,
    )
    assert result.closure_residual_eangstrom < 1.0e-14

    bad_atomic = atomic.copy()
    bad_atomic[0, 0, 0] += -1.0 - total[0, 0]
    bad_total = np.sum(bad_atomic, axis=0)
    with pytest.raises(ValueError, match="positive definite"):
        close_dipole_with_mdp_polarizability(
            source4_raw_l1=source,
            positions_angstrom=positions,
            target_molecular_dipole_eangstrom=target,
            atomic_polarizabilities_eangstrom2_per_volt=bad_atomic,
            molecular_polarizability_eangstrom2_per_volt=bad_total,
        )


def test_reconstruction_condition_and_field_gates_fail_closed() -> None:
    source = _source()
    positions = _positions()
    atomic = _atomic_alpha()
    total = np.sum(atomic, axis=0)
    target = molecular_dipole_eangstrom(source, positions) + np.asarray([2.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="field exceeds"):
        close_dipole_with_mdp_polarizability(
            source4_raw_l1=source,
            positions_angstrom=positions,
            target_molecular_dipole_eangstrom=target,
            atomic_polarizabilities_eangstrom2_per_volt=atomic,
            molecular_polarizability_eangstrom2_per_volt=total,
        )

    drifted = atomic.copy()
    drifted[0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="do not reconstruct"):
        close_dipole_with_mdp_polarizability(
            source4_raw_l1=source,
            positions_angstrom=positions,
            target_molecular_dipole_eangstrom=molecular_dipole_eangstrom(
                source, positions
            ),
            atomic_polarizabilities_eangstrom2_per_volt=drifted,
            molecular_polarizability_eangstrom2_per_volt=total,
        )
