#!/usr/bin/env python3
"""Run every frozen AIMNet2 MNSol partition row as a resumable shard.

The scientific worker remains ``run_mnsol_aimnet2_multisolvent_pilot.py``.
This scheduler only supplies every index from a previously frozen complete
partition selection, bounds process concurrency, and resumes solely from
source-bound shard pairs.  All row-level artifacts stay below ``.omx``.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
from subprocess import Popen
import subprocess
import sys
import time
from typing import IO, Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / (
    "docs/implicit-solvation/benchmarks/" "run_mnsol_aimnet2_multisolvent_pilot.py"
)
EXPECTED_ARTIFACT = "route2-mnsol-aimnet2-multisolvent-pilot-v1"
EXPECTED_RUN_KIND = "partition-record-shard"
MAXIMUM_WORKERS = 8


class SchedulerError(RuntimeError):
    """The scheduler cannot safely classify or launch the requested run."""


@dataclass(frozen=True, slots=True)
class ShardState:
    index: int
    directory: Path
    private_path: Path
    summary_path: Path
    log_path: Path
    should_run: bool
    reason: str
    command: tuple[str, ...]


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise SchedulerError(f"Invalid JSON shard {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SchedulerError(f"JSON shard must be an object: {path}")
    return value


def _clean_head() -> str:
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise SchedulerError("MNSol partition execution requires a clean checkout.")
    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(head) != 40:
        raise SchedulerError("Could not resolve the execution Git commit.")
    return head


def _below_private_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    private_root = (REPO_ROOT / ".omx").resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError as exc:
        raise SchedulerError("MNSol shard output root must remain below .omx.") from exc
    return resolved


def _selection_contract(path: Path) -> dict[str, Any]:
    selection = _load_object(path)
    if selection.get("artifact") != "route2-mnsol-partition-selection-v1":
        raise SchedulerError("Scheduler requires a complete partition selection.")
    partition = str(selection.get("partition", ""))
    if partition not in {"development", "confirmation"}:
        raise SchedulerError("Partition selection identity is unsupported.")
    count = selection.get("record_count")
    records = selection.get("selected_records")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or not isinstance(records, list)
        or len(records) != count
    ):
        raise SchedulerError("Partition selection record count is invalid.")
    if [record.get("selection_index") for record in records] != list(range(count)):
        raise SchedulerError("Partition selection indices are not contiguous.")
    fingerprint = str(selection.get("selection_fingerprint", ""))
    if len(fingerprint) != 64:
        raise SchedulerError("Partition selection fingerprint is invalid.")
    return selection


def _matching_complete(
    private: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    index: int,
    head: str,
    selection: Mapping[str, Any],
    checkpoint_sha256: str,
) -> bool:
    records = private.get("records")
    return (
        private.get("artifact") == EXPECTED_ARTIFACT
        and private.get("schema_version") == 1
        and private.get("do_not_commit") is True
        and private.get("complete_panel") is False
        and private.get("run_kind") == EXPECTED_RUN_KIND
        and private.get("execution_git_head") == head
        and private.get("selection_fingerprint")
        == selection.get("selection_fingerprint")
        and private.get("checkpoint", {}).get("sha256") == checkpoint_sha256
        and isinstance(records, list)
        and len(records) == 1
        and isinstance(records[0], dict)
        and records[0].get("selection_index") == index
        and summary.get("artifact") == EXPECTED_ARTIFACT
        and summary.get("schema_version") == 1
        and summary.get("do_not_commit") is True
        and summary.get("complete_panel") is False
        and summary.get("run_kind") == EXPECTED_RUN_KIND
        and summary.get("execution_git_head") == head
        and summary.get("selection_fingerprint")
        == selection.get("selection_fingerprint")
        and summary.get("selection_indices") == [index]
        and summary.get("checkpoint", {}).get("sha256") == checkpoint_sha256
    )


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_command(
    args: argparse.Namespace, index: int, directory: Path
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(RUNNER),
        "--source",
        str(args.source),
        "--protocol",
        str(args.protocol),
        "--selection",
        str(args.selection),
        "--pilot-selection",
        str(args.pilot_selection),
        "--checkpoint",
        str(args.checkpoint),
        "--private-output",
        str(directory / "private.json"),
        "--public-output",
        str(directory / "summary.json"),
        "--record-index",
        str(index),
    )


def _inspect(
    args: argparse.Namespace,
    *,
    index: int,
    head: str,
    selection: Mapping[str, Any],
    checkpoint_sha256: str,
) -> ShardState:
    directory = args.output_root / "shards" / f"index-{index:03d}"
    private_path = directory / "private.json"
    summary_path = directory / "summary.json"
    log_path = args.output_root / "logs" / f"index-{index:03d}.log"
    try:
        private = _load_object(private_path)
        summary = _load_object(summary_path)
    except FileNotFoundError:
        private = summary = None
    if (
        private is not None
        and summary is not None
        and _matching_complete(
            private,
            summary,
            index=index,
            head=head,
            selection=selection,
            checkpoint_sha256=checkpoint_sha256,
        )
    ):
        return ShardState(
            index=index,
            directory=directory,
            private_path=private_path,
            summary_path=summary_path,
            log_path=log_path,
            should_run=False,
            reason="source-bound shard pair complete",
            command=(),
        )
    return ShardState(
        index=index,
        directory=directory,
        private_path=private_path,
        summary_path=summary_path,
        log_path=log_path,
        should_run=True,
        reason="missing or contract-mismatched shard pair",
        command=_build_command(args, index, directory),
    )


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    return environment


def _append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _run(states: Sequence[ShardState], *, maximum_workers: int) -> int:
    pending = deque(state for state in states if state.should_run)
    running: dict[int, tuple[Popen[str], IO[str], ShardState]] = {}
    failed: list[int] = []
    halt = False
    completed = sum(not state.should_run for state in states)
    total = len(states)

    while pending or running:
        while pending and not halt and len(running) < maximum_workers:
            state = pending.popleft()
            state.directory.mkdir(parents=True, exist_ok=True)
            _append_log(state.log_path, "# " + shlex.join(state.command))
            output = state.log_path.open("a", encoding="utf-8")
            process = Popen(
                list(state.command),
                cwd=REPO_ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=_child_environment(),
                text=True,
            )
            running[state.index] = (process, output, state)
            print(f"[START] {state.index:03d} pid={process.pid}", flush=True)
            time.sleep(0.03)

        if halt and not running:
            break

        finished: list[tuple[int, int]] = []
        for index, (process, output, state) in tuple(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            output.close()
            running.pop(index)
            finished.append((index, return_code))
            if return_code == 0:
                completed += 1
                _append_log(state.log_path, "# exit=0")
                print(f"[DONE] {index:03d} ({completed}/{total})", flush=True)
            else:
                failed.append(index)
                halt = True
                _append_log(state.log_path, f"# exit={return_code}")
                print(f"[FAIL] {index:03d} rc={return_code}", flush=True)
        if not finished:
            time.sleep(0.1)

    if failed:
        print("Failed indices: " + ", ".join(f"{index:03d}" for index in failed))
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one complete frozen AIMNet2 MNSol partition."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--pilot-selection", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--index-start", type=int, default=0)
    parser.add_argument("--index-stop", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name in ("source", "protocol", "selection", "pilot_selection", "checkpoint"):
        value = getattr(args, name).expanduser().resolve()
        if not value.is_file():
            raise FileNotFoundError(value)
        setattr(args, name, value)
    args.output_root = _below_private_root(args.output_root)
    selection = _selection_contract(args.selection)
    count = int(selection["record_count"])
    stop = count if args.index_stop is None else args.index_stop
    if not (0 <= args.index_start < stop <= count):
        raise ValueError(f"Requested index range must lie inside [0, {count}).")
    maximum_workers = max(1, min(MAXIMUM_WORKERS, args.max_workers))
    head = _clean_head()
    checkpoint_sha256 = _sha256(args.checkpoint)
    states = [
        _inspect(
            args,
            index=index,
            head=head,
            selection=selection,
            checkpoint_sha256=checkpoint_sha256,
        )
        for index in range(args.index_start, stop)
    ]
    run_count = sum(state.should_run for state in states)
    print(
        f"partition={selection['partition']} range={args.index_start}:{stop} "
        f"run={run_count} skip={len(states) - run_count} head={head[:8]}"
    )
    if args.status:
        for state in states:
            action = "run" if state.should_run else "skip"
            print(f"[{state.index:03d}] {action}: {state.reason}")
        return 0
    if args.dry_run:
        for state in states:
            if state.should_run:
                print(shlex.join(state.command))
        return 0
    return _run(states, maximum_workers=maximum_workers)


if __name__ == "__main__":
    raise SystemExit(main())
