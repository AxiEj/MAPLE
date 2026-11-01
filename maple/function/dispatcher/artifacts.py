"""Shared ownership-safe helpers for dispatcher output artifacts."""

from __future__ import annotations

import os
import stat
import tempfile
from contextlib import contextmanager
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


@contextmanager
def atomic_output_path(exact_owned_path: str | PathLike[str]):
    """Yield a sibling temporary path and publish it only after successful writing."""
    current = os.path.abspath(os.fspath(exact_owned_path))
    directory = os.path.dirname(current)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=directory,
        prefix=os.path.basename(current) + ".",
        suffix=".tmp",
    )
    os.close(descriptor)
    try:
        yield temporary
        os.replace(temporary, current)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
