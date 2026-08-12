#!/usr/bin/env python3
"""Capture source/runtime identity without importing optional MAPLE modules."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone


PACKAGE_NAMES = (
    "maple",
    "ase",
    "numpy",
    "scipy",
    "pytest",
    "torch",
    "mace-torch",
    "pyscf",
    "pyddx",
)


def _run(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(raw_path: str) -> dict[str, object]:
    path = Path(raw_path).expanduser().resolve(strict=True)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


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
    }


def _git_record() -> dict[str, object]:
    status = _run("git", "status", "--porcelain=v1")
    return {
        "head": _run("git", "rev-parse", "HEAD"),
        "tree": _run("git", "rev-parse", "HEAD^{tree}"),
        "status_porcelain": status,
        "clean": status == "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--command", action="append", default=[])
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--native-library", action="append", default=[])
    parser.add_argument("--source-file", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "schema_version": "route2-runtime-manifest-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_record(),
        "commands": list(args.command),
        "runtime": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": _package_versions(),
            "torch": _torch_runtime(),
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "CUDA_VISIBLE_DEVICES",
                    "PYTHONHASHSEED",
                )
            },
        },
        "checkpoints": [_file_record(path) for path in args.checkpoint],
        "native_libraries": [
            _file_record(path) for path in args.native_library
        ],
        "source_files": [_file_record(path) for path in args.source_file],
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
