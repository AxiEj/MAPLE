"""Fail-closed, source-bound evidence primitives.

Release evidence is meaningful only when the executable files are immutable
Git objects and the computation does not change the checkout.  These helpers
bind that identity without importing optional scientific runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import ModuleType
from typing import Iterable, Mapping


def _run(root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        arguments,
        cwd=root,
        check=False,
        capture_output=True,
        text=not binary,
    )
    if result.returncode != 0:
        diagnostic = result.stderr if result.stderr else result.stdout
        if isinstance(diagnostic, bytes):
            diagnostic = diagnostic.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Command {arguments!r} failed with exit {result.returncode}: "
            f"{str(diagnostic).strip()}"
        )
    return result.stdout


def sha256_file(path: str | Path) -> str:
    """Hash one regular file without loading it fully into memory."""

    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Expected a regular file, received {resolved}.")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Hash one JSON-native value using the evidence canonicalization."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class RepositorySnapshot:
    """Exact clean Git identity before and after a release computation."""

    root: Path
    head: str
    tree: str
    status_porcelain: str

    @classmethod
    def capture(
        cls, root: str | Path, *, require_clean: bool = True
    ) -> "RepositorySnapshot":
        resolved = Path(root).expanduser().resolve(strict=True)
        if not (resolved / ".git").exists():
            top = str(_run(resolved, "git", "rev-parse", "--show-toplevel")).strip()
            resolved = Path(top).resolve(strict=True)
        head = str(_run(resolved, "git", "rev-parse", "HEAD")).strip()
        tree = str(_run(resolved, "git", "rev-parse", "HEAD^{tree}")).strip()
        status = str(
            _run(
                resolved,
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            )
        ).strip()
        if len(head) != 40 or len(tree) != 40:
            raise RuntimeError("Release evidence requires full Git object identities.")
        if require_clean and status:
            raise RuntimeError(
                "Release evidence requires a clean working tree; status was:\n" + status
            )
        return cls(resolved, head, tree, status)

    @property
    def clean(self) -> bool:
        return not self.status_porcelain

    def assert_unchanged(self) -> None:
        """Fail when HEAD, tree, or working-tree cleanliness changed mid-run."""

        current = type(self).capture(self.root, require_clean=False)
        if current.head != self.head or current.tree != self.tree:
            raise RuntimeError("Repository HEAD/tree changed during evidence capture.")
        if current.status_porcelain != self.status_porcelain:
            raise RuntimeError("Repository working-tree state changed during capture.")

    def as_dict(self) -> dict[str, object]:
        return {
            "head": self.head,
            "tree": self.tree,
            "clean": self.clean,
            "status_porcelain": self.status_porcelain,
        }


def _source_path(module: ModuleType) -> Path | None:
    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str) or not raw:
        return None
    try:
        path = Path(raw).resolve(strict=True)
    except OSError:
        return None
    if path.suffix in (".pyc", ".pyo"):
        try:
            path = Path(importlib.util.source_from_cache(str(path))).resolve(
                strict=True
            )
        except (ValueError, OSError):
            return None
    return path if path.suffix == ".py" and path.is_file() else None


def collect_loaded_repository_sources(
    root: str | Path,
    *,
    modules: Mapping[str, ModuleType] | None = None,
    required_paths: Iterable[str | Path] = (),
) -> tuple[str, ...]:
    """Return tracked Python sources actually loaded from one checkout.

    Dynamic collection avoids an incomplete hand-written dependency list while
    keeping external packages bound by their own version/source provenance.
    Only repository-relative tracked files are returned.
    """

    resolved = Path(root).expanduser().resolve(strict=True)
    candidates: set[str] = set()
    for module in (sys.modules if modules is None else modules).values():
        if not isinstance(module, ModuleType):
            continue
        path = _source_path(module)
        if path is None:
            continue
        try:
            relative = path.relative_to(resolved).as_posix()
        except ValueError:
            continue
        candidates.add(relative)
    for raw in required_paths:
        path = Path(raw)
        path = (resolved / path).resolve() if not path.is_absolute() else path.resolve()
        try:
            candidates.add(path.relative_to(resolved).as_posix())
        except ValueError as exc:
            raise RuntimeError(
                f"Loaded/required repository Python sources are not tracked: {path}"
            ) from exc
    tracked = set(str(_run(resolved, "git", "ls-files", "--", "*.py")).splitlines())
    missing = sorted(candidates - tracked)
    if missing:
        raise RuntimeError(
            "Loaded/required repository Python sources are not tracked: "
            + ", ".join(missing)
        )
    if not candidates:
        raise RuntimeError("No loaded repository source files were discovered.")
    return tuple(sorted(candidates))


def committed_source_hashes(
    snapshot: RepositorySnapshot,
    relative_paths: Iterable[str],
) -> dict[str, str]:
    """Hash Git blobs and prove the current files are the same bytes."""

    result: dict[str, str] = {}
    for relative in sorted(set(relative_paths)):
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Source path must be repository-relative: {relative!r}.")
        blob = _run(
            snapshot.root,
            "git",
            "show",
            f"{snapshot.head}:{relative}",
            binary=True,
        )
        if not isinstance(blob, bytes):  # pragma: no cover - internal contract
            raise RuntimeError("Binary Git object capture returned text unexpectedly.")
        digest = hashlib.sha256(blob).hexdigest()
        current = snapshot.root / relative
        if not current.is_file() or sha256_file(current) != digest:
            raise RuntimeError(
                f"Current source {relative!r} does not match Git {snapshot.head}."
            )
        result[relative] = digest
    if not result:
        raise ValueError("At least one source file must be bound.")
    return result


def checkpoint_record(path: str | Path) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve(strict=True)
    stat = resolved.stat()
    return {
        "role": "mace-polar-1-m-official-checkpoint",
        "resolved_path": str(resolved),
        "bytes": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for name in names:
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def _numpy_runtime() -> dict[str, object] | None:
    try:
        import numpy as np
    except ImportError:
        return None
    stream = io.StringIO()
    try:
        import contextlib

        with contextlib.redirect_stdout(stream):
            np.show_config()
    except Exception as error:  # diagnostic only; never hides the exception text
        return {"version": np.__version__, "show_config_error": repr(error)}
    return {"version": np.__version__, "show_config": stream.getvalue()}


def _torch_runtime() -> dict[str, object] | None:
    try:
        import torch
    except ImportError:
        return None
    return {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "default_dtype": str(torch.get_default_dtype()),
        "threads": torch.get_num_threads(),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_debug_mode": torch.get_deterministic_debug_mode(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def _cpu_model() -> str | None:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return platform.processor() or None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def runtime_record() -> dict[str, object]:
    """Capture the runtime fields required by the Route-2 evidence contract."""

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "packages": _package_versions(
            (
                "maple",
                "ase",
                "numpy",
                "scipy",
                "torch",
                "mace-torch",
                "graph-longrange",
                "pyscf",
                "pyddx",
            )
        ),
        "numpy": _numpy_runtime(),
        "torch": _torch_runtime(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_LAUNCH_BLOCKING",
                "PYTHONHASHSEED",
            )
        },
    }


def write_external_json_artifact(
    snapshot: RepositorySnapshot,
    output: str | Path,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Atomically write JSON outside the checkout and return its file record."""

    destination = Path(output).expanduser().resolve()
    try:
        destination.relative_to(snapshot.root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "Release diagnostic output must be outside the source checkout so "
            "evidence capture cannot dirty the tested tree."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


__all__ = [
    "RepositorySnapshot",
    "canonical_json_sha256",
    "checkpoint_record",
    "collect_loaded_repository_sources",
    "committed_source_hashes",
    "runtime_record",
    "sha256_file",
    "write_external_json_artifact",
]
