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
