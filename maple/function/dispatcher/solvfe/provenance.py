from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Iterable

from .protocol import raw_sha256


_LOWER_HEX = frozenset("0123456789abcdef")
EMPTY_GIT_STATUS_SHA256 = hashlib.sha256(b"").hexdigest()


def require_coherent_git_status(
    git_dirty: object,
    git_status_sha256: object,
) -> tuple[bool, str]:
    """Validate the boolean/digest representation of one Git worktree state."""

    if type(git_dirty) is not bool:
        raise ValueError("Git dirty state must be boolean.")
    if (
        not isinstance(git_status_sha256, str)
        or len(git_status_sha256) != 64
        or any(character not in _LOWER_HEX for character in git_status_sha256)
    ):
        raise ValueError(
            "Git status SHA256 must be a lowercase 64-character "
            "hexadecimal digest."
        )
    status_is_empty = git_status_sha256 == EMPTY_GIT_STATUS_SHA256
    if not git_dirty and not status_is_empty:
        raise ValueError(
            "Git status SHA256 must be the empty-status digest when "
            "Git dirty is false."
        )
    if git_dirty and status_is_empty:
        raise ValueError(
            "Git status SHA256 cannot be the empty-status digest when "
            "Git dirty is true."
        )
    return git_dirty, git_status_sha256


def collect_implementation_provenance(
    project_root: str | Path,
    relative_paths: Iterable[str | Path],
    *,
    schema: str = "maple-route-a-bulk-water-implementation-v1",
) -> dict:
    """Hash one executing source surface and bind it to its Git state."""

    root = Path(project_root).expanduser().resolve()
    normalized_paths: set[str] = set()
    for value in relative_paths:
        relative_path = Path(value)
        if relative_path.is_absolute():
            raise ValueError("Implementation paths must be project-relative.")
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "Implementation paths must remain inside the project root."
            ) from exc
        if not candidate.is_file():
            raise ValueError(
                f"Implementation file does not exist: {relative_path}"
            )
        normalized_paths.add(relative_path.as_posix())
    if not normalized_paths:
        raise ValueError("At least one implementation path is required.")

    def git_output(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ("git", *arguments),
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            command = " ".join(("git", *arguments))
            raise RuntimeError(
                "GIT_PROVENANCE_UNAVAILABLE: failed to collect required "
                f"implementation identity with '{command}'."
            ) from exc
        return completed.stdout.strip()

    git_head = git_output("rev-parse", "HEAD")
    if len(git_head) not in {40, 64} or any(
        character not in _LOWER_HEX
        for character in git_head
    ):
        raise RuntimeError(
            "GIT_PROVENANCE_UNAVAILABLE: Git HEAD is not a full lowercase "
            "object ID."
        )
    git_status = git_output(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    git_dirty, git_status_sha256 = require_coherent_git_status(
        bool(git_status),
        hashlib.sha256(git_status.encode("utf-8")).hexdigest(),
    )
    return {
        "schema": schema,
        "project_root": root.as_posix(),
        "git_head": git_head,
        "git_dirty": git_dirty,
        "git_status_sha256": git_status_sha256,
        "implementation_file_sha256": {
            relative_path: raw_sha256(root / relative_path)
            for relative_path in sorted(normalized_paths)
        },
    }


__all__ = [
    "EMPTY_GIT_STATUS_SHA256",
    "collect_implementation_provenance",
    "require_coherent_git_status",
]
