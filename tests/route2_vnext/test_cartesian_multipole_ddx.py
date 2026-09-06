from __future__ import annotations

import numpy as np
import pytest
from ase.units import Bohr

from maple.solvation.release.cartesian_multipole_ddx import (
    cartesian_atomic_multipoles_to_pyddx_l3,
    cartesian_to_pyddx_l3_transform,
)
from maple.solvation.release.cartesian_multipole_mep import (
    cartesian_atomic_multipole_potential,
)

pyddx = pytest.importorskip("pyddx")


def test_basis_change_is_full_rank_and_replays_its_collocation() -> None:
    transform = cartesian_to_pyddx_l3_transform()
    assert transform.collocation_rank == 16
    assert transform.collocation_condition_number < 500.0
    assert transform.maximum_collocation_residual < 2.0e-14


def test_random_multiatom_qpqo_potential_matches_pyddx() -> None:
    rng = np.random.default_rng(20260817)
    atom_count = 4
    centers_angstrom = rng.normal(size=(atom_count, 3))
    model = pyddx.Model(
        "pcm",
        centers_angstrom.T / Bohr,
        np.full(atom_count, 5.0),
        78.39,
        lmax=8,
        n_lebedev=434,
        enable_fmm=False,
        n_proc=1,
    )
    charges = rng.normal(size=atom_count)
    dipoles = rng.normal(size=(atom_count, 3))
    quadrupoles = rng.normal(size=(atom_count, 3, 3))
    octupoles = rng.normal(size=(atom_count, 3, 3, 3))
    multipoles = cartesian_atomic_multipoles_to_pyddx_l3(
        charges_e=charges,
        dipoles_eangstrom=dipoles,
        quadrupoles_eangstrom2=quadrupoles,
        octupoles_eangstrom3=octupoles,
    )
    observed = np.asarray(
        model.multipole_electrostatics(multipoles, derivative_order=0)["phi"]
    )
    expected = cartesian_atomic_multipole_potential(
        points_bohr=np.asarray(model.cavity).T,
        centers_angstrom=centers_angstrom,
        charges_e=charges,
        dipoles_eangstrom=dipoles,
        quadrupoles_eangstrom2=quadrupoles,
        octupoles_eangstrom3=octupoles,
    )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2.0e-12)


def test_trace_and_antisymmetric_cartesian_parts_are_exterior_null_modes() -> None:
    quadrupole = np.eye(3)[None, :, :] * 0.7
    octupole = np.zeros((1, 3, 3, 3))
    octupole[0, 0, 1, 2] = 0.4
    octupole[0, 1, 0, 2] = -0.4
    multipoles = cartesian_atomic_multipoles_to_pyddx_l3(
        charges_e=np.zeros(1),
        dipoles_eangstrom=np.zeros((1, 3)),
        quadrupoles_eangstrom2=quadrupole,
        octupoles_eangstrom3=octupole,
    )
    assert np.max(np.abs(multipoles)) < 3.0e-15
