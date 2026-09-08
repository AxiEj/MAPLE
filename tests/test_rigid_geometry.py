import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from maple.function.dispatcher.frequency.frequency import R_GAS, MWFrequency
from maple.function.dispatcher.md.utils import is_linear_molecule
from maple.function.utility.rigid_body import (
    mass_weighted_rigid_basis,
    rotational_dof,
)


def _oco(angle_degrees: float) -> Atoms:
    angle = np.deg2rad(angle_degrees)
    return Atoms(
        "OCO",
        positions=[[-1.16, 0.0, 0.0], [0.0, 0.0, 0.0], [1.16 * -np.cos(angle), 1.16 * np.sin(angle), 0.0]],
    )


def _frequency_job(atoms: Atoms) -> MWFrequency:
    job = MWFrequency("/dev/null", atoms, ilowfreq=0)
    job.verbosity = 0
    return job


def test_near_linear_geometry_has_three_rotational_dof() -> None:
    atoms = _oco(170.0)
    job = _frequency_job(atoms)

    assert not is_linear_molecule(atoms)
    assert not job._is_linear_molecule()
    assert mass_weighted_rigid_basis(atoms).shape[1] == 6
    assert rotational_dof(atoms) == 3


def test_rotational_thermal_partition_matches_rigid_geometry_rank() -> None:
    linear = _frequency_job(_oco(180.0)).compute_thermo(np.empty(0))
    bent = _frequency_job(_oco(170.0)).compute_thermo(np.empty(0))

    assert linear.h_rot_kjmol == pytest.approx(R_GAS * 298.15e-3)
    assert bent.h_rot_kjmol == pytest.approx(1.5 * R_GAS * 298.15e-3)


def test_atom_and_strictly_linear_rigid_dimensions() -> None:
    atom = _frequency_job(Atoms("He"))
    linear = _frequency_job(_oco(180.0))

    assert mass_weighted_rigid_basis(atom.atoms).shape[1] == 3
    assert mass_weighted_rigid_basis(linear.atoms).shape[1] == 5
    assert rotational_dof(atom.atoms) == 0
    assert rotational_dof(linear.atoms) == 2
    assert atom.compute_thermo(np.empty(0)).h_rot_kjmol == 0.0


def test_periodic_rigid_basis_contains_translations_only() -> None:
    atoms = _oco(170.0)
    atoms.set_cell([10.0, 10.0, 10.0])
    atoms.set_pbc(True)

    assert mass_weighted_rigid_basis(atoms).shape[1] == 3
    assert rotational_dof(atoms) == 0
    assert not is_linear_molecule(atoms)


def test_fixatoms_does_not_change_geometry_classification() -> None:
    atoms = _oco(170.0)
    atoms.set_constraint(FixAtoms(indices=[0]))

    assert mass_weighted_rigid_basis(atoms).shape[1] == 6
    assert rotational_dof(atoms) == 3
