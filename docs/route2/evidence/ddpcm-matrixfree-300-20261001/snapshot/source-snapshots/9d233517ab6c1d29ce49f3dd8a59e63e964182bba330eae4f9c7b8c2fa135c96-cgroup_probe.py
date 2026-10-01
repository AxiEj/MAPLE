#!/usr/bin/env python3
"""Run allowlisted research probes in an OS-enforced user systemd scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from research_identity import (
    research_python_hashes,
    verify_tracked_freeze,
    write_json_new,
)
from research_receipt import run_guarded

ALLOWED_SCRIPTS = frozenset(
    {"probe_streamed.py", "probe_native.py", "research_identity.py"}
)
HARD_MEMORY_BYTES = 8 * 1024**3
MAX_WALL_SECONDS = 1200.0
CRITICAL_MEMORY_EVENTS = ("max", "oom", "oom_kill", "oom_group_kill")


def _artifact_paths(output: Path) -> dict[str, Path]:
    return {
        "receipt": output,
        "systemd_stdout": output.with_suffix(output.suffix + ".systemd.stdout"),
        "systemd_stderr": output.with_suffix(output.suffix + ".systemd.stderr"),
        "worker": output.with_suffix(output.suffix + ".worker.json"),
        "child_stdout": output.with_suffix(output.suffix + ".child.stdout"),
        "child_stderr": output.with_suffix(output.suffix + ".child.stderr"),
    }


def _ensure_new_artifacts(paths: dict[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite prior evidence: {existing}")
    paths["receipt"].parent.mkdir(parents=True, exist_ok=True)


def _validate_script(
    script: str, script_args: list[str], *, wrapper_output: Path | None = None
) -> list[str]:
    if script not in ALLOWED_SCRIPTS:
        raise ValueError(f"research probe is not allowlisted: {script}")
    path = (HERE / script).resolve()
    if path.parent != HERE or not path.is_file():
        raise FileNotFoundError(f"allowlisted research probe is unavailable: {script}")
    arguments = list(script_args)
    if arguments and arguments[0] == "--":
        arguments.pop(0)
    if arguments.count("--output") != 1:
        raise ValueError("child probe arguments must include --output")
    output_index = arguments.index("--output")
    if output_index + 1 >= len(arguments):
        raise ValueError("child --output requires a path")
    child_output = Path(arguments[output_index + 1]).expanduser()
    child_output = (
        child_output.resolve()
        if child_output.is_absolute()
        else (ROOT / child_output).resolve()
    )
    if child_output.exists():
        raise FileExistsError(f"refusing to overwrite child evidence: {child_output}")
    if wrapper_output is not None:
        reserved = {
            path.resolve()
            for path in _artifact_paths(wrapper_output.resolve()).values()
        }
        if child_output in reserved:
            raise ValueError("child --output collides with wrapper evidence")
    return [sys.executable, str(path), *arguments]


def _read_scalar(path: Path) -> int | str:
    value = path.read_text(encoding="utf-8").strip()
    return int(value) if value.isdecimal() else value


def _read_events(path: Path) -> dict[str, int]:
    return {
        name: int(value)
        for name, value in (
            line.split(maxsplit=1)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    }


def _owned_cgroup() -> tuple[Path, str]:
    unified = None
    for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
        hierarchy, controllers, relative = line.split(":", 2)
        if hierarchy == "0" and controllers == "":
            unified = relative
            break
    if unified is None:
        raise RuntimeError("unified cgroup v2 membership is unavailable")
    path = Path("/sys/fs/cgroup") / unified.lstrip("/")
    if not path.is_dir():
        raise RuntimeError(f"owned cgroup path is unavailable: {path}")
    return path, unified


def _validate_hard_limits(cgroup: Path) -> dict[str, object]:
    memory_max = _read_scalar(cgroup / "memory.max")
    swap_max = _read_scalar(cgroup / "memory.swap.max")
    if memory_max != HARD_MEMORY_BYTES:
        raise RuntimeError(
            f"owned scope memory.max is {memory_max}, expected {HARD_MEMORY_BYTES}"
        )
    if swap_max != 0:
        raise RuntimeError(f"owned scope memory.swap.max is {swap_max}, expected 0")
    return {
        "memory_max_bytes": memory_max,
        "memory_swap_max_bytes": swap_max,
        "memory_events_before": _read_events(cgroup / "memory.events"),
        "memory_current_before_bytes": _read_scalar(cgroup / "memory.current"),
    }


def _critical_event_deltas(
    before: dict[str, int], after: dict[str, int]
) -> dict[str, int]:
    return {
        name: int(after.get(name, 0)) - int(before.get(name, 0))
        for name in CRITICAL_MEMORY_EVENTS
    }


def _worker(
    *,
    worker_output: Path,
    child_stdout: Path,
    child_stderr: Path,
    wall_limit_seconds: float,
    command: list[str],
) -> int:
    payload: dict[str, Any] = {
        "schema": "matrixfree-owned-cgroup-worker-v1",
        "scientific_admitted": False,
        "gate_complete": False,
        "completed": False,
    }
    exit_status = 1
    try:
        cgroup, membership = _owned_cgroup()
        hard_limits = _validate_hard_limits(cgroup)
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "WANDB_MODE": "disabled",
                "PYTHONPATH": str(ROOT),
            }
        )
        execution = run_guarded(
            command,
            stdout_path=child_stdout,
            stderr_path=child_stderr,
            environment=environment,
            # Deliberately disable sampled-memory termination here: cgroup
            # memory.max is the OS-enforced boundary.  run_guarded remains the
            # process-tree telemetry and wall-time supervisor.
            memory_limit_bytes=sys.maxsize,
            wall_limit_seconds=wall_limit_seconds,
        )
        events_after = _read_events(cgroup / "memory.events")
        critical_deltas = _critical_event_deltas(
            hard_limits["memory_events_before"], events_after
        )
        kernel_memory_clean = all(delta == 0 for delta in critical_deltas.values())
        kernel_memory_peak = _read_scalar(cgroup / "memory.peak")
        kernel_peak_within_limit = (
            isinstance(kernel_memory_peak, int)
            and kernel_memory_peak <= HARD_MEMORY_BYTES
        )
        payload.update(
            {
                "cgroup_membership": membership,
                "cgroup_path": str(cgroup),
                "hard_limits": hard_limits,
                "sampled_supervisor": {
                    **execution,
                    "is_hard_rss_enforcement": False,
                    "role": "sampled process-tree telemetry and wall supervisor",
                },
                "kernel_memory_peak_bytes": kernel_memory_peak,
                "kernel_memory_peak_within_limit": kernel_peak_within_limit,
                "kernel_memory_events_after": events_after,
                "kernel_critical_event_deltas": critical_deltas,
                "kernel_memory_events_clean": kernel_memory_clean,
                "kernel_memory_current_after_bytes": _read_scalar(
                    cgroup / "memory.current"
                ),
            }
        )
        payload["completed"] = (
            execution["exit_status"] == 0
            and execution["termination_reason"] is None
            and kernel_memory_clean
            and kernel_peak_within_limit
        )
        if not kernel_memory_clean:
            payload["failure"] = {
                "type": "KernelMemoryEventIncrease",
                "message": (
                    "owned cgroup recorded a critical memory event increase: "
                    f"{critical_deltas}"
                ),
            }
        elif not kernel_peak_within_limit:
            payload["failure"] = {
                "type": "KernelMemoryPeakExceeded",
                "message": (
                    f"owned cgroup memory.peak {kernel_memory_peak} exceeds "
                    f"memory.max {HARD_MEMORY_BYTES}"
                ),
            }
        exit_status = 0 if payload["completed"] else 1
    except Exception as error:
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    write_json_new(worker_output, payload)
    return exit_status


def _unit_name(output: Path) -> str:
    digest = hashlib.sha256(str(output.resolve()).encode("utf-8")).hexdigest()[:20]
    return f"maple-mf-cap-{digest}"


def _systemctl(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        ["systemctl", "--user", *command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=15.0,
        check=False,
    )
    return {
        "command": list(result.args),
        "exit_status": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _owned_unit_state(unit: str) -> dict[str, Any]:
    active = _systemctl(["is-active", unit])
    show = _systemctl(
        [
            "show",
            "--no-pager",
            "--property=LoadState,ActiveState,ControlGroup",
            unit,
        ]
    )
    state = active["stdout"].strip()
    properties = {}
    for line in show["stdout"].splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            properties[name] = value
    load_state = properties.get("LoadState")
    shown_active_state = properties.get("ActiveState")
    relative = properties.get("ControlGroup", "")
    cgroup = Path("/sys/fs/cgroup") / relative.lstrip("/") if relative else None
    processes: list[int] = []
    cgroup_query_success = cgroup is None
    if cgroup is not None:
        procs = cgroup / "cgroup.procs"
        if procs.is_file():
            try:
                processes = [
                    int(value)
                    for value in procs.read_text(encoding="utf-8").splitlines()
                    if value.strip()
                ]
                cgroup_query_success = True
            except (OSError, ValueError):
                cgroup_query_success = False
        else:
            # A successfully queried inactive/not-found unit may already have
            # had its empty transient cgroup removed.
            cgroup_query_success = True
    recognized_states = {
        "active",
        "reloading",
        "inactive",
        "failed",
        "activating",
        "deactivating",
        "maintenance",
        "refreshing",
        "unknown",
    }
    explicit_not_found = (
        show["exit_status"] == 0
        and load_state == "not-found"
        and shown_active_state == "inactive"
        and not relative
        and active["exit_status"] in {3, 4}
        and state in {"inactive", "unknown"}
    )
    loaded_query_valid = (
        show["exit_status"] == 0
        and load_state not in {None, "", "not-found"}
        and shown_active_state in recognized_states - {"unknown"}
        and active["exit_status"] in {0, 3}
        and state in recognized_states - {"unknown"}
        and state == shown_active_state
    )
    queries_valid = explicit_not_found or loaded_query_valid
    inactive = explicit_not_found or (
        loaded_query_valid and shown_active_state in {"inactive", "failed"}
    )
    return {
        "active_query": active,
        "show_query": show,
        "active_state": state,
        "shown_active_state": shown_active_state,
        "load_state": load_state,
        "explicit_not_found": explicit_not_found,
        "queries_valid": queries_valid,
        "cgroup_path": str(cgroup) if cgroup is not None else None,
        "cgroup_query_success": cgroup_query_success,
        "cgroup_processes": processes,
        "inactive": inactive,
        "empty": cgroup_query_success and not processes,
        "verified_inactive_and_empty": (
            queries_valid and inactive and cgroup_query_success and not processes
        ),
    }


def _cleanup_owned_unit(unit: str) -> dict[str, Any]:
    """Stop/kill exactly one deterministic unit and verify it is quiescent."""

    actions = [_systemctl(["stop", unit])]
    observations = []
    for _ in range(20):
        observation = _owned_unit_state(unit)
        observations.append(observation)
        if observation["verified_inactive_and_empty"]:
            break
        time.sleep(0.05)
    if not observations[-1]["verified_inactive_and_empty"]:
        actions.append(_systemctl(["kill", "--kill-whom=all", "--signal=KILL", unit]))
        actions.append(_systemctl(["stop", unit]))
        observations.append(_owned_unit_state(unit))
    return {
        "scope": "deterministic-owned-unit-only",
        "unit": unit,
        "actions": actions,
        "observations": observations,
        "verified_inactive_and_empty": observations[-1]["verified_inactive_and_empty"],
    }


def _validate_worker_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("owned scope worker receipt is not a JSON object")
    hard_limits = payload.get("hard_limits")
    supervisor = payload.get("sampled_supervisor")
    expected_deltas = {name: 0 for name in CRITICAL_MEMORY_EVENTS}
    valid = (
        payload.get("schema") == "matrixfree-owned-cgroup-worker-v1"
        and payload.get("scientific_admitted") is False
        and payload.get("gate_complete") is False
        and payload.get("completed") is True
        and isinstance(payload.get("cgroup_membership"), str)
        and ".scope" in payload["cgroup_membership"]
        and isinstance(hard_limits, dict)
        and hard_limits.get("memory_max_bytes") == HARD_MEMORY_BYTES
        and hard_limits.get("memory_swap_max_bytes") == 0
        and payload.get("kernel_memory_events_clean") is True
        and payload.get("kernel_critical_event_deltas") == expected_deltas
        and payload.get("kernel_memory_peak_within_limit") is True
        and isinstance(payload.get("kernel_memory_peak_bytes"), int)
        and payload["kernel_memory_peak_bytes"] <= HARD_MEMORY_BYTES
        and isinstance(supervisor, dict)
        and supervisor.get("exit_status") == 0
        and supervisor.get("termination_reason") is None
    )
    if not valid:
        raise RuntimeError("owned scope worker receipt is incomplete or inconsistent")
    return payload


def _launch(
    *,
    output: Path,
    wall_limit_seconds: float,
    command: list[str],
) -> dict[str, Any]:
    if not 0.0 < wall_limit_seconds <= MAX_WALL_SECONDS:
        raise ValueError("wall limit must be in (0, 1200] seconds")
    paths = _artifact_paths(output)
    unit = _unit_name(output)
    try:
        _ensure_new_artifacts(paths)
    except Exception as error:
        if not output.exists():
            write_json_new(
                output,
                {
                    "schema": "matrixfree-owned-cgroup-receipt-v1",
                    "scientific_admitted": False,
                    "gate_complete": False,
                    "completed": False,
                    "unit": f"{unit}.scope",
                    "failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                },
            )
        raise
    worker_wall = max(0.1, wall_limit_seconds - 2.0)
    worker_command = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        f"--unit={unit}",
        "-p",
        "MemoryAccounting=yes",
        "-p",
        "MemoryMax=8G",
        "-p",
        "MemorySwapMax=0",
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--worker-output",
        str(paths["worker"]),
        "--child-stdout",
        str(paths["child_stdout"]),
        "--child-stderr",
        str(paths["child_stderr"]),
        "--wall-limit-seconds",
        str(worker_wall),
        "--command-json",
        json.dumps(command),
    ]
    tracked_before = None
    research_before = None
    tracked_after = None
    research_after = None
    outer = None
    worker_payload = None
    post_worker_unit_state = None
    launch_attempted = False
    failure: dict[str, str] | None = None
    payload = {
        "schema": "matrixfree-owned-cgroup-receipt-v1",
        "scientific_admitted": False,
        "gate_complete": False,
        "completed": False,
        "unit": f"{unit}.scope",
        "unit_unique_per_output_sha256": hashlib.sha256(
            str(output.resolve()).encode("utf-8")
        ).hexdigest(),
        "hard_limit_contract": {
            "mechanism": "systemd-user-transient-scope-cgroup-v2",
            "memory_max_bytes": HARD_MEMORY_BYTES,
            "memory_swap_max_bytes": 0,
        },
    }
    try:
        tracked_before = verify_tracked_freeze()
        research_before = research_python_hashes()
        launch_attempted = True
        outer = run_guarded(
            worker_command,
            stdout_path=paths["systemd_stdout"],
            stderr_path=paths["systemd_stderr"],
            environment=os.environ.copy(),
            memory_limit_bytes=sys.maxsize,
            wall_limit_seconds=wall_limit_seconds,
        )
        if outer["exit_status"] != 0 or outer["termination_reason"] is not None:
            raise RuntimeError(f"systemd wrapper failed: {outer}")
        if not paths["worker"].is_file():
            raise RuntimeError("owned scope did not retain a worker receipt")
        worker_payload = _validate_worker_payload(
            json.loads(paths["worker"].read_text(encoding="utf-8"))
        )
        post_worker_unit_state = _owned_unit_state(f"{unit}.scope")
        if not post_worker_unit_state["verified_inactive_and_empty"]:
            raise RuntimeError(
                "owned scope retained active processes after the producer exited"
            )
        tracked_after = verify_tracked_freeze()
        research_after = research_python_hashes()
        if tracked_before != tracked_after:
            raise RuntimeError("tracked freeze drifted during owned-scope execution")
        if research_before != research_after:
            raise RuntimeError("research Python sources drifted during execution")
        payload["completed"] = True
    except Exception as error:
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        if tracked_after is None:
            try:
                tracked_after = verify_tracked_freeze()
            except Exception as error:
                tracked_after = {"failure": f"{type(error).__name__}: {error}"}
                if failure is None:
                    failure = {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
        if research_after is None:
            try:
                research_after = research_python_hashes()
            except Exception as error:
                research_after = {"failure": f"{type(error).__name__}: {error}"}
                if failure is None:
                    failure = {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
        if tracked_before != tracked_after or research_before != research_after:
            payload["completed"] = False
            if failure is None:
                failure = {
                    "type": "SourceIdentityDrift",
                    "message": "tracked or research source identity drifted",
                    "traceback": "",
                }
        cleanup = None
        if launch_attempted and not payload["completed"]:
            try:
                cleanup = _cleanup_owned_unit(f"{unit}.scope")
            except Exception as error:
                cleanup = {
                    "unit": f"{unit}.scope",
                    "verified_inactive_and_empty": False,
                    "failure": f"{type(error).__name__}: {error}",
                }
            if not cleanup.get("verified_inactive_and_empty", False):
                payload["completed"] = False
                if failure is None:
                    failure = {
                        "type": "OwnedUnitCleanupFailure",
                        "message": "owned unit was not verified inactive and empty",
                        "traceback": "",
                    }
        payload.update(
            {
                "systemd_execution": (
                    {
                        **outer,
                        "is_hard_rss_enforcement": False,
                        "role": "sampled wrapper telemetry and wall supervisor",
                    }
                    if outer is not None
                    else None
                ),
                "worker": worker_payload,
                "owned_unit_state_after_worker": post_worker_unit_state,
                "worker_receipt_sha256": (
                    hashlib.sha256(paths["worker"].read_bytes()).hexdigest()
                    if paths["worker"].is_file()
                    else None
                ),
                "owned_unit_cleanup_on_failure": cleanup,
                "tracked_freeze_before": tracked_before,
                "tracked_freeze_after": tracked_after,
                "research_python_sha256_before": research_before,
                "research_python_sha256_after": research_after,
                "tracked_unchanged": tracked_before == tracked_after,
                "research_python_unchanged": research_before == research_after,
            }
        )
        if failure is not None:
            payload["failure"] = failure
    write_json_new(output, payload)
    return payload


def run_cgroup_command_for_testing(
    command: list[str], output: Path, wall_limit_seconds: float = 30.0
) -> dict[str, Any]:
    """Exercise the real owned-scope boundary with a tiny local command."""

    return _launch(
        output=output,
        wall_limit_seconds=wall_limit_seconds,
        command=list(command),
    )


def _worker_main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-output", type=Path, required=True)
    parser.add_argument("--child-stdout", type=Path, required=True)
    parser.add_argument("--child-stderr", type=Path, required=True)
    parser.add_argument("--wall-limit-seconds", type=float, required=True)
    parser.add_argument("--command-json", required=True)
    args = parser.parse_args(arguments)
    command = json.loads(args.command_json)
    if not isinstance(command, list) or not all(
        isinstance(item, str) for item in command
    ):
        raise ValueError("worker command must be a JSON string list")
    return _worker(
        worker_output=args.worker_output,
        child_stdout=args.child_stdout,
        child_stderr=args.child_stderr,
        wall_limit_seconds=args.wall_limit_seconds,
        command=command,
    )


def main(arguments: list[str] | None = None) -> int:
    argv = sys.argv[1:] if arguments is None else list(arguments)
    if argv and argv[0] == "_worker":
        return _worker_main(argv[1:])
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-limit-seconds", type=float, default=MAX_WALL_SECONDS)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = _validate_script(
        args.script, args.script_args, wrapper_output=args.output
    )
    payload = _launch(
        output=args.output,
        wall_limit_seconds=args.wall_limit_seconds,
        command=command,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if payload["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
