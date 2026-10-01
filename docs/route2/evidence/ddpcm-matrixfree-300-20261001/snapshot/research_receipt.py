#!/usr/bin/env python3
"""Serial pytest receipts with sampled process-tree monitoring and a watchdog.

The sampler is not an OS-enforced memory cap. Large probes additionally use
the separately verified systemd/cgroup memory.max and memory.swap.max limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from research_identity import (
    research_python_hashes,
    verify_tracked_freeze,
    write_json_new,
)

ALLOWED_TESTS = frozenset(
    {
        "test_native_source_adjoint.py",
        "test_streamed_ddpcm.py",
        "test_research_identity.py",
        "test_compiled_value_operator.py",
        "test_cgroup_probe.py",
        "test_fused_value_operator.py",
    }
)
DEFAULT_MEMORY_LIMIT_BYTES = 8 * 1024**3
DEFAULT_WALL_LIMIT_SECONDS = 300.0


def _validate_tests(names: list[str]) -> list[Path]:
    if not names:
        raise ValueError("At least one --test-file is required.")
    paths = []
    for name in names:
        if name not in ALLOWED_TESTS:
            raise ValueError(f"Research test is not allowlisted: {name}")
        path = (HERE / name).resolve()
        if path.parent != HERE or not path.is_file():
            raise FileNotFoundError(f"Allowlisted research test is unavailable: {name}")
        paths.append(path)
    return paths


def _memory_tree(process: psutil.Process) -> tuple[int, int]:
    rss = 0
    uss = 0
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except psutil.NoSuchProcess:
        pass
    for item in processes:
        try:
            info = item.memory_full_info()
            rss += int(info.rss)
            uss += int(getattr(info, "uss", 0))
        except psutil.NoSuchProcess:
            continue
    return rss, uss


def run_guarded(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    environment: dict[str, str],
    memory_limit_bytes: int,
    wall_limit_seconds: float,
) -> dict[str, Any]:
    if stdout_path.exists() or stderr_path.exists():
        raise FileExistsError("Refusing to overwrite prior stdout/stderr evidence.")
    begun = time.monotonic()
    maximum_rss = 0
    maximum_uss = 0
    termination_reason = None
    with (
        stdout_path.open("x", encoding="utf-8") as stdout,
        stderr_path.open("x", encoding="utf-8") as stderr,
    ):
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        observed = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                rss, uss = _memory_tree(observed)
            except (psutil.AccessDenied, OSError) as error:
                termination_reason = f"memory-observation-failed:{type(error).__name__}"
                rss = uss = 0
            maximum_rss = max(maximum_rss, rss)
            maximum_uss = max(maximum_uss, uss)
            elapsed = time.monotonic() - begun
            if rss > memory_limit_bytes or uss > memory_limit_bytes:
                termination_reason = "memory-limit"
            elif elapsed > wall_limit_seconds:
                termination_reason = "walltime-limit"
            if termination_reason is not None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(0.05)
        exit_status = child.wait()
    return {
        "command": command,
        "exit_status": exit_status,
        "wall_seconds": time.monotonic() - begun,
        "maximum_process_tree_rss_bytes": maximum_rss,
        "maximum_process_tree_uss_bytes": maximum_uss,
        "memory_limit_bytes": memory_limit_bytes,
        "wall_limit_seconds": wall_limit_seconds,
        "termination_reason": termination_reason,
        "memory_monitor_kind": "sampled-process-tree-not-an-OS-memory-cap",
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_sha256": hashlib.sha256(stdout_path.read_bytes()).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr_path.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--memory-limit-bytes", type=int, default=DEFAULT_MEMORY_LIMIT_BYTES
    )
    parser.add_argument(
        "--wall-limit-seconds", type=float, default=DEFAULT_WALL_LIMIT_SECONDS
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite prior receipt: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tests = _validate_tests(args.test_file)
    stdout_path = args.output.with_suffix(args.output.suffix + ".stdout")
    stderr_path = args.output.with_suffix(args.output.suffix + ".stderr")
    tracked_before = verify_tracked_freeze()
    research_before = research_python_hashes()
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(ROOT),
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
        }
    )
    command = [sys.executable, "-m", "pytest", "-q", *map(str, tests)]
    execution = run_guarded(
        command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        environment=environment,
        memory_limit_bytes=args.memory_limit_bytes,
        wall_limit_seconds=args.wall_limit_seconds,
    )
    tracked_after = verify_tracked_freeze()
    research_after = research_python_hashes()
    tracked_unchanged = tracked_before == tracked_after
    research_unchanged = research_before == research_after
    payload = {
        "schema": "matrixfree-research-pytest-receipt-v1",
        "scientific_admitted": False,
        "gate_complete": False,
        "ledger_is_model_accuracy_evidence": False,
        "tests": [path.name for path in tests],
        "environment_contract": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "WANDB_MODE",
            )
        },
        "tracked_freeze_before": tracked_before,
        "tracked_freeze_after": tracked_after,
        "research_python_sha256_before": research_before,
        "research_python_sha256_after": research_after,
        "tracked_unchanged": tracked_unchanged,
        "research_python_unchanged": research_unchanged,
        "execution": execution,
        "completed": (
            execution["exit_status"] == 0
            and execution["termination_reason"] is None
            and tracked_unchanged
            and research_unchanged
        ),
    }
    write_json_new(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if payload["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
