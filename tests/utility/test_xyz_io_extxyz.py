import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from maple.function.utility.xyz_io import write_xyz


def _atoms(symbol="Cu", x=0.0):
    return Atoms(symbol, positions=[[x, 0.0, 0.0]], cell=[3.6, 3.6, 3.6], pbc=True)


def test_empty_atoms_list_creates_or_truncates_file(tmp_path):
    path = tmp_path / "empty.xyz"
    path.write_text("stale")

    write_xyz(str(path), [])

    assert path.read_text() == ""


def test_invalid_mode_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="mode must be 'w' or 'a'"):
        write_xyz(str(tmp_path / "bad.xyz"), [_atoms()], mode="x")


def test_energy_metadata_validation_is_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="energy_key"):
        write_xyz(str(tmp_path / "bad-key.xyz"), [_atoms()], [1.0], energy_key="")
    with pytest.raises(ValueError, match="match atoms_list length"):
        write_xyz(str(tmp_path / "bad-length.xyz"), [_atoms()], [])


def test_legacy_positional_mode_and_start_index_remain_compatible(tmp_path):
    path = tmp_path / "legacy-positional.xyz"

    write_xyz(str(path), [_atoms()], [1.25], "w", 4)

    reread = read(str(path))
    assert int(reread.info["image"]) == 4
    assert reread.get_potential_energy() == pytest.approx(1.25)


def test_append_adds_frames_and_overwrite_replaces_existing_file(tmp_path):
    path = tmp_path / "traj.xyz"
    frames = [_atoms("Cu", 0.0), _atoms("Cu", 1.0)]

    write_xyz(str(path), frames, start_index=0)
    write_xyz(str(path), [_atoms("Cu", 2.0)], mode="a", start_index=2)

    reread = read(str(path), ":")
    assert [int(at.info["image"]) for at in reread] == [0, 1, 2]
    np.testing.assert_allclose(
        [at.positions[0, 0] for at in reread],
        [0.0, 1.0, 2.0],
    )

    write_xyz(str(path), [_atoms("Cu", 3.0)], start_index=7)

    reread = read(str(path), ":")
    assert len(reread) == 1
    assert int(reread[0].info["image"]) == 7
    np.testing.assert_allclose(reread[0].positions[0, 0], 3.0)


def test_start_index_none_preserves_existing_image(tmp_path):
    path = tmp_path / "preserve.xyz"
    atoms = _atoms()
    atoms.info["image"] = 42

    write_xyz(str(path), [atoms], start_index=None)

    reread = read(str(path))
    assert int(reread.info["image"]) == 42


def test_start_index_renumbers_frames(tmp_path):
    path = tmp_path / "renumber.xyz"
    frames = [_atoms("Cu", 0.0), _atoms("Cu", 1.0)]
    frames[0].info["image"] = 99

    write_xyz(str(path), frames, start_index=5)

    reread = read(str(path), ":")
    assert [int(at.info["image"]) for at in reread] == [5, 6]
