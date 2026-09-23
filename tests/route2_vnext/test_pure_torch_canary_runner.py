"""Failure-preserving evidence writer for the Torch/legacy canary."""

import json
from pathlib import Path

import pytest

from tools.route2_release import run_pure_mace_polar_torch_canary as runner


def test_progress_is_explicitly_provisional(tmp_path):
    runner._write_progress(tmp_path, {"case": "water-water"}, "legacy_force_complete")
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["stage"] == "legacy_force_complete"
    assert progress["case"] == "water-water"
    assert progress["pass"] is False
    assert progress["execution_complete"] is False


def test_atomic_write_preserves_previous_evidence_on_interrupted_stage(
    tmp_path, monkeypatch
):
    destination = tmp_path / "progress.json"
    destination.write_text('{"stage":"previous"}\n')
    original = Path.write_text

    def interrupted_write(path, content, *args, **kwargs):
        if path.name == "progress.json.tmp":
            original(path, "{", *args, **kwargs)
            raise OSError("interrupted after partial write")
        return original(path, content, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", interrupted_write)
    with pytest.raises(OSError, match="interrupted"):
        runner._write(destination, {"stage": "new"})
    assert json.loads(destination.read_text()) == {"stage": "previous"}
