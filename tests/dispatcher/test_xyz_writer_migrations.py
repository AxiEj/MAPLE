import importlib
from types import SimpleNamespace

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.build import bulk
from ase.io import read

from maple.function.dispatcher.scan.scan import Scan
from maple.function.utility.xyz_io import write_xyz as utility_write_xyz


scan_lbfgs = importlib.import_module("maple.function.dispatcher.scan.algorithm.LBFGS")
prfo_module = importlib.import_module("maple.function.dispatcher.ts.algorithm.PRFO")


def _periodic_cu():
    return bulk("Cu", "fcc", a=3.6)


def test_scan_writer_uses_shared_extxyz_round_trip(tmp_path):
    atoms = _periodic_cu()
    path = tmp_path / "scan.xyz"

    assert scan_lbfgs.write_xyz is utility_write_xyz

    scan_lbfgs.write_xyz(str(path), [atoms])

    round_trip = read(str(path))
    assert all(round_trip.pbc)
    np.testing.assert_allclose(round_trip.get_cell().array, atoms.get_cell().array)


def test_scan_final_stream_round_trips_periodic_metadata(tmp_path):
    atoms = _periodic_cu()
    atoms.calc = SinglePointCalculator(atoms, energy=-0.75, free_energy=-0.75)
    path = tmp_path / "scan_final.xyz"
    path.write_text("")
    recorder = SimpleNamespace(
        xyz_file=str(path),
        _current_index=1,
        _total_combinations=1,
        mode="relaxed",
    )
    coords = []
    energies = []

    Scan._record_result(recorder, atoms, [2.5], coords, energies)

    round_trip = read(str(path))
    assert all(round_trip.pbc)
    np.testing.assert_allclose(round_trip.get_cell().array, atoms.get_cell().array)
    assert int(round_trip.info["image"]) == 1
    assert round_trip.info["scan_coord"] == "[2.5000]"
    np.testing.assert_allclose(round_trip.get_potential_energy(), -0.75)
    assert coords == [[2.5]]
    assert energies == [-0.75]


def test_prfo_wrapper_keeps_signature_and_round_trips_metadata(tmp_path):
    atoms = _periodic_cu()
    path = tmp_path / "prfo.xyz"

    prfo_module.write_xyz(str(path), atoms, energy=-1.25, iteration=9)

    round_trip = read(str(path))
    assert all(round_trip.pbc)
    np.testing.assert_allclose(round_trip.get_cell().array, atoms.get_cell().array)
    assert int(round_trip.info["image"]) == 9
    np.testing.assert_allclose(round_trip.get_potential_energy(), -1.25)


neb_module = importlib.import_module("maple.function.dispatcher.ts.algorithm.neb")
dimer_module = importlib.import_module("maple.function.dispatcher.ts.algorithm.dimer")


def test_prfo_append_trajectory_overwrites_then_appends_with_periodic_cell(tmp_path):
    path = tmp_path / "prfo_traj.xyz"
    prfo_module.append_xyz_trajectory(str(path), _periodic_cu(), energy=-1.0, iteration=0)
    prfo_module.append_xyz_trajectory(str(path), _periodic_cu(), energy=-2.0, iteration=1)
    prfo_module.append_xyz_trajectory(str(path), _periodic_cu(), energy=-3.0, iteration=2)

    frames = read(str(path), ":")
    assert [int(at.info["image"]) for at in frames] == [0, 1, 2]
    for frame in frames:
        assert all(frame.pbc)
        np.testing.assert_allclose(frame.get_cell().array, _periodic_cu().get_cell().array)
    np.testing.assert_allclose(
        [frame.get_potential_energy() for frame in frames], [-1.0, -2.0, -3.0]
    )


def test_neb_write_all_images_appends_iterations_with_periodic_cell(tmp_path):
    path = tmp_path / "neb_traj.xyz"
    images = [_periodic_cu(), _periodic_cu(), _periodic_cu()]
    energies_iter0 = [-1.0, -1.1, -1.2]
    energies_iter1 = [-1.5, -1.6, -1.7]

    neb_module.write_all_images_xyz(str(path), images, energies=energies_iter0, iteration=0)
    neb_module.write_all_images_xyz(str(path), images, energies=energies_iter1, iteration=1)

    frames = read(str(path), ":")
    assert len(frames) == 6
    assert [int(frame.info["image"]) for frame in frames] == [0, 1, 2, 0, 1, 2]
    assert [int(frame.info["iteration"]) for frame in frames] == [0, 0, 0, 1, 1, 1]
    for frame in frames:
        assert all(frame.pbc)
        np.testing.assert_allclose(frame.get_cell().array, _periodic_cu().get_cell().array)


def test_dimer_write_all_images_appends_iterations_with_periodic_cell(tmp_path):
    path = tmp_path / "dimer_traj.xyz"
    dimer_module.write_all_images_xyz(str(path), _periodic_cu(), energy=-1.0, iteration=0)
    dimer_module.write_all_images_xyz(str(path), _periodic_cu(), energy=-2.0, iteration=1)

    frames = read(str(path), ":")
    assert len(frames) == 2
    assert [int(frame.info["iteration"]) for frame in frames] == [0, 1]
    for frame in frames:
        assert all(frame.pbc)
        np.testing.assert_allclose(frame.get_cell().array, _periodic_cu().get_cell().array)
    np.testing.assert_allclose(
        [frame.get_potential_energy() for frame in frames], [-1.0, -2.0]
    )
