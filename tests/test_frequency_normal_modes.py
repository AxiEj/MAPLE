from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.vibrations.data import VibrationsData

from maple.function.dispatcher.frequency.normal_modes import (
    EV_PER_ANGSTROM2_AMU_TO_WAVENUMBER_CM1,
    analyze_cartesian_hessian,
    mass_weighted_basis_to_cartesian,
    rigid_body_subspaces,
)

WATER_MASSES_AMU = np.array([15.999, 1.008, 1.008])
WATER_POSITIONS_A = np.array(
    [
        [0.0, 0.0, 0.1173],
        [0.0, 0.7572, -0.4692],
        [0.0, -0.7572, -0.4692],
    ]
)


def test_nonlinear_rigid_and_vibrational_subspaces_are_mass_orthogonal():
    subspaces = rigid_body_subspaces(WATER_MASSES_AMU, WATER_POSITIONS_A)

    assert subspaces.translation_rank == 3
    assert subspaces.rotation_rank == 3
    assert subspaces.vibrational_rank == 3
    assert subspaces.is_linear is False

    combined = np.concatenate(
        (
            subspaces.translation_basis_mass_weighted,
            subspaces.rotation_basis_mass_weighted,
            subspaces.vibrational_basis_mass_weighted,
        ),
        axis=1,
    )
    assert combined.T @ combined == pytest.approx(np.eye(9), abs=2.0e-14)


def test_linear_molecule_has_five_rigid_and_one_vibrational_mode():
    masses = np.array([12.0, 16.0])
    positions = np.array([[0.0, 0.0, -0.6], [0.0, 0.0, 0.45]])

    subspaces = rigid_body_subspaces(masses, positions)

    assert subspaces.translation_rank == 3
    assert subspaces.rotation_rank == 2
    assert subspaces.vibrational_rank == 1
    assert subspaces.is_linear is True


def test_cartesian_hessian_analysis_recovers_known_vibrational_spectrum():
    subspaces = rigid_body_subspaces(WATER_MASSES_AMU, WATER_POSITIONS_A)
    vibrational_basis = subspaces.vibrational_basis_mass_weighted
    expected_eigenvalues = np.array([1.0, 4.0, 9.0])
    hessian_mass_weighted = (
        vibrational_basis @ np.diag(expected_eigenvalues) @ vibrational_basis.T
    )
    square_root_mass = np.sqrt(np.repeat(WATER_MASSES_AMU, 3))
    hessian_cartesian = (
        square_root_mass[:, None] * hessian_mass_weighted * square_root_mass[None, :]
    )

    analysis = analyze_cartesian_hessian(
        hessian_cartesian,
        WATER_MASSES_AMU,
        WATER_POSITIONS_A,
    )

    assert analysis.eigenvalues_eV_per_A2_amu == pytest.approx(
        expected_eigenvalues,
        abs=2.0e-14,
    )
    assert analysis.frequencies_cm1 == pytest.approx(
        EV_PER_ANGSTROM2_AMU_TO_WAVENUMBER_CM1 * np.array([1.0, 2.0, 3.0]),
        abs=2.0e-11,
    )
    assert analysis.modes_mass_weighted.T @ analysis.modes_mass_weighted == (
        pytest.approx(np.eye(3), abs=2.0e-14)
    )
    mass_metric = np.diag(np.repeat(WATER_MASSES_AMU, 3))
    assert (
        analysis.modes_cartesian_per_sqrt_amu.T
        @ mass_metric
        @ analysis.modes_cartesian_per_sqrt_amu
    ) == pytest.approx(np.eye(3), abs=2.0e-14)


def test_vibrational_frequencies_match_ase_for_same_ev_hessian():
    subspaces = rigid_body_subspaces(WATER_MASSES_AMU, WATER_POSITIONS_A)
    vibrational_basis = subspaces.vibrational_basis_mass_weighted
    expected_eigenvalues = np.array([1.0, 4.0, 9.0])
    hessian_mass_weighted = (
        vibrational_basis @ np.diag(expected_eigenvalues) @ vibrational_basis.T
    )
    square_root_mass = np.sqrt(np.repeat(WATER_MASSES_AMU, 3))
    hessian_cartesian = (
        square_root_mass[:, None] * hessian_mass_weighted * square_root_mass[None, :]
    )
    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=WATER_POSITIONS_A,
        masses=WATER_MASSES_AMU,
    )

    analysis = analyze_cartesian_hessian(
        hessian_cartesian,
        WATER_MASSES_AMU,
        WATER_POSITIONS_A,
    )
    ase_frequencies = VibrationsData.from_2d(
        atoms,
        hessian_cartesian,
    ).get_frequencies()

    assert analysis.frequencies_cm1 == pytest.approx(
        np.sort(ase_frequencies.real)[-3:],
        rel=5.0e-9,
        abs=1.0e-8,
    )


def test_mass_weighted_basis_can_be_returned_as_unit_cartesian_directions():
    subspaces = rigid_body_subspaces(WATER_MASSES_AMU, WATER_POSITIONS_A)
    cartesian = mass_weighted_basis_to_cartesian(
        subspaces.rotation_basis_mass_weighted,
        WATER_MASSES_AMU,
        normalize_columns=True,
    )

    assert np.linalg.norm(cartesian, axis=0) == pytest.approx(np.ones(3), abs=2.0e-14)


def test_asymmetric_cartesian_hessian_is_rejected_instead_of_hidden():
    hessian = np.zeros((9, 9))
    hessian[0, 1] = 1.0e-5

    with pytest.raises(ValueError, match="not symmetric"):
        analyze_cartesian_hessian(
            hessian,
            WATER_MASSES_AMU,
            WATER_POSITIONS_A,
            symmetry_tolerance_eV_per_A2=1.0e-8,
        )
