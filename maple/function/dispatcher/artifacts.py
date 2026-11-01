"""Shared ownership-safe helpers for dispatcher output artifacts."""

from __future__ import annotations

import os
import stat
from os import PathLike


def retire_certified_output(exact_owned_path: str | PathLike[str]) -> str | None:
    """Atomically archive one exact job-owned certified output, if present."""
    current = os.path.abspath(os.fspath(exact_owned_path))
    if not os.path.lexists(current):
        return None
    mode = os.lstat(current).st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError("Refusing to retire a symlink or non-file certified output.")

    stem, extension = os.path.splitext(current)
    archive_index = 1
    while True:
        archive = f"{stem}.previous-{archive_index}{extension}"
        if not os.path.lexists(archive):
            os.replace(current, archive)
            return archive
        archive_index += 1
