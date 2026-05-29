import pytest
import struct
import numpy as np
from ase import Atoms

from maple.function.dispatcher.md.dcd_writer import DCDWriter
from maple.function.read.filereader.dcd_reader import DCDReader


def _atoms() -> Atoms:
    return Atoms(
        "He",
        positions=[[0.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )


def test_dcd_round_trips_sub_picosecond_timestep(tmp_path):
    path = tmp_path / "traj.dcd"

    with DCDWriter(path, natoms=1, timestep=0.1, is_periodic=True) as writer:
        writer.write_frame(_atoms())

    with DCDReader(path) as reader:
        assert reader.nframes == 1
        assert reader.timestep == pytest.approx(0.1)


def test_dcd_append_preserves_float_timestep_and_updates_frame_count(tmp_path):
    path = tmp_path / "traj.dcd"

    with DCDWriter(path, natoms=1, timestep=0.1, is_periodic=True) as writer:
        writer.write_frame(_atoms())

    writer = DCDWriter.open_for_append(path)
    writer.write_frame(_atoms())
    writer.close()

    with DCDReader(path) as reader:
        assert reader.nframes == 2
        assert reader.timestep == pytest.approx(0.1)


def test_dcd_triclinic_pbc_roundtrip_cell_positions_and_append_count(tmp_path):
    path = tmp_path / "triclinic.dcd"
    cell = np.array(
        [
            [8.0, 0.0, 0.0],
            [1.5, 7.5, 0.0],
            [0.7, 1.1, 9.0],
        ],
        dtype=float,
    )
    atoms = Atoms(
        "He2",
        positions=[[0.1, 0.2, 0.3], [2.4, 3.1, 4.2]],
        cell=cell,
        pbc=True,
    )
    atoms2 = atoms.copy()
    atoms2.positions += np.array([[0.01, 0.02, 0.03], [-0.02, 0.01, -0.01]])

    with DCDWriter(path, natoms=2, timestep=0.25, is_periodic=True) as writer:
        writer.write_frame(atoms)

    writer = DCDWriter.open_for_append(path)
    writer.write_frame(atoms2)
    writer.close()

    with DCDReader(path) as reader:
        assert reader.nframes == 2
        assert reader.timestep == pytest.approx(0.25)
        first = reader.read_frame(0)
        second = reader.read_frame(1)

    np.testing.assert_allclose(first.cell.cellpar(), atoms.cell.cellpar(), rtol=0, atol=1e-10)
    np.testing.assert_allclose(first.positions, atoms.positions, rtol=0, atol=1e-6)
    np.testing.assert_allclose(second.cell.cellpar(), atoms2.cell.cellpar(), rtol=0, atol=1e-10)
    np.testing.assert_allclose(second.positions, atoms2.positions, rtol=0, atol=1e-6)


def test_dcd_reader_accepts_legacy_integer_delta_field(tmp_path):
    path = tmp_path / "legacy.dcd"

    with DCDWriter(path, natoms=1, timestep=0.1, is_periodic=True) as writer:
        writer.write_frame(_atoms())

    with open(path, "r+b") as handle:
        handle.seek(4 + 9 * 4)
        handle.write(struct.pack("<i", 1))

    with DCDReader(path) as reader:
        assert reader.timestep == pytest.approx(1000.0)

    writer = DCDWriter.open_for_append(path)
    assert writer.timestep == pytest.approx(1000.0)
    writer.close()
