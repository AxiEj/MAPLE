from __future__ import annotations

from pathlib import Path
import os
import sys

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from research_identity import (
    public_unreachability,
    research_python_hashes,
    verify_tracked_freeze,
    write_json_new,
)
from research_receipt import _validate_tests, run_guarded


def test_frozen_manifest_and_public_unreachability_remain_intact() -> None:
    frozen = verify_tracked_freeze()
    assert frozen["manifest_entry_count"] == 312
    assert frozen["tracked_diff_sha256"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    negative = public_unreachability()
    assert negative["passed"] is True
    assert negative["registry_research_entries"] == []


def test_research_hashes_and_receipt_boundaries(tmp_path: Path) -> None:
    hashes = research_python_hashes()
    assert {
        "research_identity.py",
        "research_receipt.py",
        "test_research_identity.py",
    }.issubset(hashes)
    assert [path.name for path in _validate_tests(["test_research_identity.py"])] == [
        "test_research_identity.py"
    ]
    with pytest.raises(ValueError, match="not allowlisted"):
        _validate_tests(["../../tests/route2_vnext/test_mace_polar_native_fp64.py"])
    receipt = tmp_path / "receipt.json"
    write_json_new(receipt, {"scientific_admitted": False, "gate_complete": False})
    with pytest.raises(FileExistsError):
        write_json_new(receipt, {})


def test_guarded_subprocess_retains_logs_and_status(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    result = run_guarded(
        [sys.executable, "-c", "import sys; print('ok'); print('note', file=sys.stderr)"],
        stdout_path=stdout,
        stderr_path=stderr,
        environment=os.environ.copy(),
        memory_limit_bytes=512 * 1024**2,
        wall_limit_seconds=10.0,
    )
    assert result["exit_status"] == 0
    assert result["termination_reason"] is None
    assert stdout.read_text().strip() == "ok"
    assert stderr.read_text().strip() == "note"
    with pytest.raises(FileExistsError, match="overwrite"):
        run_guarded(
            [sys.executable, "-c", "pass"],
            stdout_path=stdout,
            stderr_path=stderr,
            environment=os.environ.copy(),
            memory_limit_bytes=512 * 1024**2,
            wall_limit_seconds=10.0,
        )
