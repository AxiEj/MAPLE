import numpy as np
import pytest
from ase.build import bulk
from ase.io import read

from maple.function.dispatcher.optimization.algorithm import _common
from maple.function.utility.xyz_io import write_xyz as utility_write_xyz


@pytest.fixture
def periodic_cu():
    pytest.importorskip("ase.calculators.emt")
    from ase.calculators.emt import EMT

    atoms = bulk("Cu", "fcc", a=3.6)
    atoms.calc = EMT()
    return atoms


def test_common_writer_round_trips_periodic_cell_and_pbc(tmp_path, periodic_cu):
    path = tmp_path / "cu_opt_traj.xyz"

    _common.write_xyz(str(path), [periodic_cu], energies=[periodic_cu.get_potential_energy()])

    round_trip = read(str(path))
    assert all(round_trip.pbc)
    np.testing.assert_allclose(round_trip.get_cell().array, periodic_cu.get_cell().array)


@pytest.mark.parametrize("writer", [_common.write_xyz, utility_write_xyz])
def test_shared_writer_round_trips_energies(tmp_path, periodic_cu, writer):
    path = tmp_path / f"energy_{writer.__module__}.xyz"
    energy = periodic_cu.get_potential_energy()

    writer(str(path), [periodic_cu], energies=[energy])

    round_trip = read(str(path))
    np.testing.assert_allclose(round_trip.get_potential_energy(), energy)
