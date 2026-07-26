from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Mapping


def assert_source_files_match_execution_commit(
    root: Path,
    artifact: Mapping[str, object],
) -> None:
    """Verify immutable source hashes against the recorded execution commit.

    Comparing an old result artifact with the current working tree makes the
    artifact freeze future implementation work.  The immutable evidence is the
    tuple ``(execution_git_head, source_files_sha256)``, so validation must read
    the exact Git objects used to produce the result.
    """

    execution_git_head = artifact.get("execution_git_head")
    if not isinstance(execution_git_head, str) or len(execution_git_head) != 40:
        raise AssertionError(
            "Immutable benchmark artifacts must record a full "
            "execution_git_head."
        )
    source_hashes = artifact.get("source_files_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise AssertionError(
            "Immutable benchmark artifacts must record source_files_sha256."
        )

    for relative, expected in source_hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise AssertionError(
                "source_files_sha256 must map relative paths to hex digests."
            )
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "show",
                f"{execution_git_head}:{relative}",
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            diagnostic = result.stderr.decode("utf-8", errors="replace").strip()
            raise AssertionError(
                "Could not read source-bound benchmark input "
                f"{relative!r} at {execution_git_head}: {diagnostic}"
            )
        observed = hashlib.sha256(result.stdout).hexdigest()
        if observed != expected:
            raise AssertionError(
                "Source-bound benchmark hash mismatch for "
                f"{relative!r} at {execution_git_head}: expected {expected}, "
                f"observed {observed}."
            )
