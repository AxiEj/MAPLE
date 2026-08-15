from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.gto_density import (
    cartesian_multipoles,
)
from maple.solvation.models.mace_mdp import (
    POLARIZABILITY_BOHR3_PER_EANGSTROM2_PER_VOLT,
    build_mace_mdp_moment_state,
)


def _state():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, -0.1, 0.3]])
    charges = np.array([-0.2, 0.2])
    dipoles = np.array([[0.1, 0.2, -0.1], [-0.05, 0.03, 0.02]])
    atomic_alpha = np.array(
        [
            [[0.20, 0.01, 0.00], [0.01, 0.25, 0.02], [0.00, 0.02, 0.30]],
            [[0.30, -0.01, 0.02], [-0.01, 0.35, 0.00], [0.02, 0.00, 0.40]],
        ]
    )
    total_alpha = np.sum(atomic_alpha, axis=0)
    total_dipole = np.sum(charges[:, None] * positions + dipoles, axis=0)
    return build_mace_mdp_moment_state(
        configuration_sha256="1" * 64,
        model_input_sha256_value="2" * 64,
        atomic_numbers=np.array([6, 8]),
        positions_angstrom=positions,
        charges_e=charges,
        atomic_dipoles_eangstrom=dipoles,
        atomic_polarizabilities_eangstrom2_per_volt=atomic_alpha,
        public_dipole_eangstrom=total_dipole,
        public_polarizability_eangstrom2_per_volt=total_alpha,
    )


def test_mace_mdp_state_preserves_moments_pairing_and_polarizability():
    state = _state()
    charges, dipoles = cartesian_multipoles(state.source4_raw_l1)
    assert np.array_equal(charges, state.charges_e)
    assert np.array_equal(dipoles, state.atomic_dipoles_eangstrom)
    assert state.total_charge_e == pytest.approx(0.0, abs=1.0e-15)
    assert np.allclose(np.sum(state.atomic_dipole_weights, axis=0), np.eye(3))
    assert np.allclose(
        state.polarizability_bohr3,
        state.public_polarizability_eangstrom2_per_volt
        * POLARIZABILITY_BOHR3_PER_EANGSTROM2_PER_VOLT,
    )
    assert len(state.state_sha256) == 64


def test_mace_mdp_state_owns_immutable_constructor_arrays():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, -0.1, 0.3]])
    state = _state()
    positions[:] = 999.0
    assert not np.any(state.positions_angstrom == 999.0)
    with pytest.raises(ValueError):
        state.charges_e.setflags(write=True)


def test_mace_mdp_state_rejects_nonreciprocal_or_nonpositive_response():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    charges = np.array([-0.1, 0.1])
    dipoles = np.zeros((2, 3))
    bad = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="reciprocal"):
        build_mace_mdp_moment_state(
            configuration_sha256="1" * 64,
            model_input_sha256_value="2" * 64,
            atomic_numbers=[6, 8],
            positions_angstrom=positions,
            charges_e=charges,
            atomic_dipoles_eangstrom=dipoles,
            atomic_polarizabilities_eangstrom2_per_volt=np.stack(
                (0.5 * bad, 0.5 * bad)
            ),
            public_dipole_eangstrom=np.array([0.1, 0.0, 0.0]),
            public_polarizability_eangstrom2_per_volt=bad,
        )


def test_mace_mdp_state_rejects_atomic_moment_mismatch():
    state = _state()
    with pytest.raises(ValueError, match="public dipole"):
        build_mace_mdp_moment_state(
            configuration_sha256=state.configuration_sha256,
            model_input_sha256_value=state.model_input_sha256,
            atomic_numbers=state.atomic_numbers,
            positions_angstrom=state.positions_angstrom,
            charges_e=state.charges_e,
            atomic_dipoles_eangstrom=state.atomic_dipoles_eangstrom,
            atomic_polarizabilities_eangstrom2_per_volt=(
                state.atomic_polarizabilities_eangstrom2_per_volt
            ),
            public_dipole_eangstrom=state.public_dipole_eangstrom + 1.0,
            public_polarizability_eangstrom2_per_volt=(
                state.public_polarizability_eangstrom2_per_volt
            ),
        )
