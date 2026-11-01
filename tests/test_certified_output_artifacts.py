from pathlib import Path
import pytest
from maple.function.dispatcher.artifacts import atomic_output_path, retire_certified_output

def test_retire_certified_output_archives_owned_regular_file_without_copying(tmp_path):
    current = tmp_path / "job_stringts_ts.xyz"
    current.write_bytes(b"validated-old-bytes\n")

    first_archive = retire_certified_output(current)

    assert first_archive == str(tmp_path / "job_stringts_ts.previous-1.xyz")
    assert Path(first_archive).read_bytes() == b"validated-old-bytes\n"
    assert not current.exists()

    current.write_bytes(b"newer-validated-bytes\n")
    second_archive = retire_certified_output(current)
    assert second_archive == str(tmp_path / "job_stringts_ts.previous-2.xyz")
    assert Path(second_archive).read_bytes() == b"newer-validated-bytes\n"

def test_retire_certified_output_rejects_symlink_and_non_file(tmp_path):
    target = tmp_path / "target.xyz"
    target.write_bytes(b"target")
    symlink = tmp_path / "owned_ts.xyz"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="symlink or non-file"):
        retire_certified_output(symlink)

    directory = tmp_path / "directory_ts.xyz"
    directory.mkdir()
    with pytest.raises(ValueError, match="symlink or non-file"):
        retire_certified_output(directory)


def test_atomic_output_path_publishes_only_after_success(tmp_path):
    current = tmp_path / "certified.xyz"
    with pytest.raises(RuntimeError, match="writer failed"):
        with atomic_output_path(current) as temporary:
            Path(temporary).write_bytes(b"partial")
            assert not current.exists()
            raise RuntimeError("writer failed")
    assert not current.exists()

    with atomic_output_path(current) as temporary:
        Path(temporary).write_bytes(b"complete")
        assert not current.exists()
    assert current.read_bytes() == b"complete"

