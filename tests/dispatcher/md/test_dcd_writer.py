import pytest
import struct
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
