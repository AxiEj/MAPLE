#!/usr/bin/env python3
"""Run bounded, source-bound pure MACE-POLAR non-MD CPU/CUDA shards.

Each device/case is a separate public ``maple INPUT OUTPUT`` subprocess.  The
runner consumes the v2 ``*_status.json`` sidecar rather than guessing success
from prose logs or a returned geometry.  A bounded cap therefore remains an
executed, finite, *non-converged* result.  Infrastructure, source, identity,
and device failures fail closed while already completed cases stay on disk.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/route2/preregistrations/pure-mace-polar-nonmd-v2.json"
STATUS_SCHEMA = "maple-pure-nonmd-status-v2"
FIXTURE_SOURCES = (
    ROOT / "docs/route2/evidence/pure-mace-polar-workflows-water-v1.json",
    ROOT / "docs/route2/evidence/pure-mace-polar-workflows-ammonia-ts-v1.json",
)


@dataclass(frozen=True)
class ProcessResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_seconds: float


ProcessRunner = Callable[..., ProcessResult]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def source_manifest() -> dict[str, str]:
    """Hash executable source plus the prospective protocol."""
    paths = sorted((ROOT / "maple").rglob("*.py"))
    paths += [PROTOCOL, Path(__file__).resolve(), *FIXTURE_SOURCES]
    return {str(path.relative_to(ROOT)): _sha(path) for path in paths}


def _git_record() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True
        ).strip()

    return {
        "head": git("rev-parse", "HEAD"),
        "status_porcelain": git("status", "--porcelain=v1", "--untracked-files=all"),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("maple", "torch", "mace-torch", "pyddx", "pyscf", "ase"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("schema") != "maple-pure-mace-polar-nonmd-canary-v2":
        raise ValueError("unexpected non-MD canary protocol schema")
    if list(protocol["cases"]) != protocol["case_order"]:
        raise ValueError("protocol cases and case_order differ")
    for source in protocol["fixture_sources"].values():
        path = ROOT / source["path"]
        if not path.is_file() or _sha(path) != source["sha256"]:
            raise ValueError(f"archived fixture source changed: {source['path']}")
    return protocol


def parse_csv(value: str, label: str) -> tuple[str, ...]:
    values = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError(f"{label} must be a nonempty comma-separated list")
    return values


def _normalize_selector(selector: str) -> str:
    return selector.strip().lower().replace(":", ".").replace("/", ".")


def select_cases(protocol: Mapping[str, Any], selectors: Sequence[str]) -> tuple[str, ...]:
    """Expand family selectors without silently widening a requested shard."""
    order = tuple(protocol["case_order"])
    chosen: list[str] = []
    for raw in selectors:
        selector = _normalize_selector(raw)
        if selector == "all":
            expansion = order
        elif selector in protocol["cases"]:
            expansion = (selector,)
        else:
            expansion = tuple(case for case in order if case.split(".", 1)[0] == selector)
            if not expansion:
                raise ValueError(f"unknown task selector: {raw}")
        for case in expansion:
            if case in chosen:
                raise ValueError(f"task case selected more than once: {case}")
            chosen.append(case)
    if not chosen:
        raise ValueError("at least one task case is required")
    return tuple(chosen)


def validate_devices(devices: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for device in devices:
        device = device.strip().lower()
        if device != "cpu" and not (
            device.startswith("cuda:") and device[5:].isdigit()
        ):
            raise ValueError("devices must be cpu or an explicit cuda:N")
        if device in normalized:
            raise ValueError(f"device selected more than once: {device}")
        normalized.append(device)
    return tuple(normalized)


def _identity(protocol: Mapping[str, Any], device: str) -> Mapping[str, str]:
    return protocol["identities"]["cpu" if device == "cpu" else "cuda"]


def _format_structure(fixture: Mapping[str, Any]) -> str:
    rows = [f"{fixture['charge']} {fixture['multiplicity']}"]
    for symbol, xyz in zip(fixture["symbols"], fixture["positions_angstrom"], strict=True):
        rows.append(f"{symbol} " + " ".join(f"{float(value):.15f}" for value in xyz))
    return "\n".join(rows)


def render_input(protocol: Mapping[str, Any], device: str, case_id: str) -> str:
    case = protocol["cases"][case_id]
    fixture = protocol["fixtures"][case["fixture"]]
    identity = _identity(protocol, device)
    settings = [
        f"#model={protocol['model']}",
        f"#device={device}",
        "#level=extratight",
        case["task_line"],
        "#solv(method=smd,provider=pyddx,"
        f"profile={identity['profile']},implicit={protocol['solvent']},"
        "response=frozen,experimental=true)",
    ]
    structures = fixture.get("images", [fixture])
    body = "\n\n&\n\n".join(_format_structure(image) for image in structures)
    post = case.get("post_lines", [])
    if post:
        body += "\n\n" + "\n".join(post)
    return "\n".join(settings) + "\n\n" + body + "\n"


def _run_subprocess(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str], timeout: float
) -> ProcessResult:
    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return ProcessResult(
            process.returncode, stdout, stderr, False, time.monotonic() - started
        )
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # The child exited between the timeout and signal.
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        # communicate() returns the complete text, including the prefix already
        # present as bytes in TimeoutExpired.output; do not concatenate it twice.
        return ProcessResult(
            process.returncode, stdout, stderr, True, time.monotonic() - started
        )


def _all_finite(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool) or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return False


def _finite_measurement(status: Mapping[str, Any]) -> bool:
    metrics = status.get("final_metrics")
    geometries = status.get("geometry")
    if not isinstance(metrics, Mapping) or not metrics:
        return False
    if not isinstance(geometries, list) or not geometries:
        return False
    for geometry in geometries:
        if not isinstance(geometry, Mapping):
            return False
        symbols = geometry.get("symbols")
        positions = geometry.get("positions_angstrom")
        if not isinstance(symbols, list) or not symbols:
            return False
        if not isinstance(positions, list) or len(positions) != len(symbols):
            return False
    return _all_finite(metrics) and _all_finite(geometries)


def _status_path(output: Path) -> Path:
    return output.with_name(output.stem + "_status.json")


def _load_status(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "missing_status"
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "invalid_status_json"
    required = {
        "workflow", "method", "executed", "converged", "termination_reason",
        "iterations", "final_metrics", "geometry", "trace", "actual_device",
        "profile", "termination_class",
    }
    if status.get("schema") != STATUS_SCHEMA or not required.issubset(status):
        return status, "invalid_status_contract"
    scalar = status.get("scalar_contract_id", status.get("scalar"))
    if not isinstance(scalar, str) or not scalar:
        return status, "invalid_status_contract"
    if (
        "scalar" in status
        and "scalar_contract_id" in status
        and status["scalar"] != status["scalar_contract_id"]
    ):
        return status, "invalid_status_contract"
    status["scalar_contract_id"] = scalar
    if type(status["executed"]) is not bool or type(status["converged"]) is not bool:
        return status, "invalid_status_contract"
    if status["termination_class"] not in {
        "converged", "bounded_nonconvergence", "validation_failure",
        "execution_failure",
    }:
        return status, "invalid_status_contract"
    return status, None


def execute_case(
    protocol: Mapping[str, Any],
    device: str,
    case_id: str,
    case_dir: Path,
    env: Mapping[str, str],
    process_runner: ProcessRunner = _run_subprocess,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=False)
    input_path = (case_dir / "job.inp").resolve()
    output_path = (case_dir / "job.out").resolve()
    input_path.write_text(render_input(protocol, device, case_id), encoding="utf-8")
    timeout = float(protocol["cases"][case_id]["timeout_seconds"])
    command = [sys.executable, "-m", "maple.main", str(input_path), str(output_path)]
    try:
        process = process_runner(command, cwd=case_dir, env=env, timeout=timeout)
    except Exception as exc:
        process = ProcessResult(
            None, "", f"{type(exc).__name__}: {exc}\n", False, 0.0
        )
        runner_error = f"{type(exc).__name__}: {exc}"
    else:
        runner_error = None
    (case_dir / "stdout.log").write_text(process.stdout, encoding="utf-8")
    (case_dir / "stderr.log").write_text(process.stderr, encoding="utf-8")
    base: dict[str, Any] = {
        "case_id": case_id,
        "requested_device": device,
        "input_sha256": _sha(input_path),
        "command": command,
        "timeout_seconds": timeout,
        "wall_seconds": process.elapsed_seconds,
        "returncode": process.returncode,
        "timed_out": process.timed_out,
        "stdout_sha256": _sha(case_dir / "stdout.log"),
        "stderr_sha256": _sha(case_dir / "stderr.log"),
        "status_path": str(_status_path(output_path).relative_to(case_dir)),
        "executed": False,
        "finite": False,
        "numerical": {
            "converged": None,
            "status": None,
            "iterations": None,
            "final_metrics": None,
        },
    }
    if process.timed_out:
        base["infrastructure_status"] = "timeout"
        return base
    if runner_error is not None:
        base["infrastructure_status"] = "runner_exception"
        base["runner_error"] = runner_error
        return base

    status, error = _load_status(_status_path(output_path))
    if error:
        base["infrastructure_status"] = error
        if status is not None:
            base["raw_status"] = status
        return base

    assert status is not None
    base["raw_status"] = status
    base["executed"] = status["executed"]
    base["finite"] = _finite_measurement(status)
    base["numerical"] = {
        "converged": status["converged"],
        "status": status["termination_reason"],
        "iterations": status["iterations"],
        "final_metrics": status["final_metrics"],
    }
    expected = _identity(protocol, device)
    identity_ok = (
        status["actual_device"] == device
        and status["profile"] == expected["profile"]
        and status["scalar_contract_id"] == expected["scalar_id"]
    )
    base["resolved_identity"] = {
        "actual_device": status["actual_device"],
        "profile": status["profile"],
        "scalar_contract_id": status["scalar_contract_id"],
        "matches_request": identity_ok,
    }
    workflow_ok = status["workflow"] == case_id.split(".", 1)[0]
    base["resolved_workflow"] = {
        "workflow": status["workflow"],
        "method": status["method"],
        "matches_request_family": workflow_ok,
    }
    if not workflow_ok:
        base["infrastructure_status"] = "workflow_mismatch"
    elif not identity_ok:
        base["infrastructure_status"] = "identity_or_device_mismatch"
    elif not status["executed"]:
        base["infrastructure_status"] = "algorithm_not_executed"
    elif not base["finite"]:
        base["infrastructure_status"] = "nonfinite_result"
    elif status["termination_class"] in {"validation_failure", "execution_failure"}:
        base["infrastructure_status"] = status["termination_class"]
    elif process.returncode == 0 and status["converged"] and status["termination_class"] == "converged":
        base["infrastructure_status"] = "completed"
    elif not status["converged"] and status["termination_class"] == "bounded_nonconvergence":
        base["infrastructure_status"] = "completed_nonconverged"
    else:
        base["infrastructure_status"] = "inconsistent_exit_status"
    return base


def run_selected_cases(
    protocol: Mapping[str, Any],
    devices: Sequence[str],
    cases: Sequence[str],
    output_dir: Path,
    env: Mapping[str, str],
    process_runner: ProcessRunner = _run_subprocess,
) -> dict[str, Any]:
    before = source_manifest()
    result: dict[str, Any] = {"cases": {}, "source_files_sha256_before": before}
    stop = False
    for device in devices:
        device_dir = output_dir / device.replace(":", "_")
        for case_id in cases:
            try:
                record = execute_case(
                    protocol,
                    device,
                    case_id,
                    device_dir / case_id.replace(".", "_"),
                    env,
                    process_runner,
                )
            except Exception as exc:
                record = {
                    "case_id": case_id,
                    "requested_device": device,
                    "executed": False,
                    "finite": False,
                    "infrastructure_status": "runner_exception",
                    "runner_error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "numerical": {
                        "converged": None,
                        "status": None,
                        "iterations": None,
                        "final_metrics": None,
                    },
                }
            result["cases"][f"{device}/{case_id}"] = record
            after = source_manifest()
            result["source_files_sha256_after"] = after
            result["source_unchanged"] = before == after
            _write_json(output_dir / "progress.json", result)
            if before != after:
                result["stop_reason"] = "source_changed"
                stop = True
                break
        if stop:
            break
    if "source_files_sha256_after" not in result:
        result["source_files_sha256_after"] = source_manifest()
    result.setdefault(
        "source_unchanged", before == result["source_files_sha256_after"]
    )
    return result


def create_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def _maximum_absolute_difference(left: Any, right: Any) -> float:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        differences = [
            _maximum_absolute_difference(a, b) for a, b in zip(left, right, strict=True)
        ]
        return max(differences, default=0.0)
    raise ValueError("comparison payload shapes differ")


def measurement_comparisons(
    protocol: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    devices: Sequence[str],
    cases: Sequence[str],
) -> dict[str, Any]:
    """Evaluate only preregistered same-geometry E/F/H replay and parity."""
    reference = protocol["cpu_prechange_reference"]
    tolerance = protocol["parity_tolerances"]
    specifications = (
        ("sp", "energy_eV", "energy_eV"),
        ("sp", "forces_eV_per_A", "forces_eV_per_A"),
        ("freq.mw", "hessian_eV_per_A2", "hessian_eV_per_A2"),
    )
    result: dict[str, Any] = {"cpu_replay": {}, "cpu_cuda_parity": {}}
    for case_id, metric, tolerance_key in specifications:
        if case_id not in cases:
            continue
        cpu = records.get(f"cpu/{case_id}")
        if "cpu" in devices:
            measured = None if cpu is None else cpu.get("numerical", {}).get(
                "final_metrics", {}
            ).get(metric)
            try:
                error = _maximum_absolute_difference(measured, reference[metric])
            except (TypeError, ValueError):
                error = None
            result["cpu_replay"][metric] = {
                "maximum_absolute_difference": error,
                "tolerance": tolerance[tolerance_key],
                "pass": error is not None and error <= tolerance[tolerance_key],
            }
        if "cpu" in devices and any(device.startswith("cuda:") for device in devices):
            for device in devices:
                if not device.startswith("cuda:"):
                    continue
                cuda = records.get(f"{device}/{case_id}")
                left = None if cpu is None else cpu.get("numerical", {}).get(
                    "final_metrics", {}
                ).get(metric)
                right = None if cuda is None else cuda.get("numerical", {}).get(
                    "final_metrics", {}
                ).get(metric)
                try:
                    error = _maximum_absolute_difference(left, right)
                except (TypeError, ValueError):
                    error = None
                result["cpu_cuda_parity"][f"{device}/{metric}"] = {
                    "maximum_absolute_difference": error,
                    "tolerance": tolerance[tolerance_key],
                    "pass": error is not None and error <= tolerance[tolerance_key],
                }
    checks = [
        check for group in result.values() for check in group.values()
    ]
    result["required_checks_pass"] = all(check["pass"] for check in checks)
    return result


def _checkpoint_path(argument: Path | None) -> Path:
    value = argument or (Path(os.environ["ROUTE2_MACE_CHECKPOINT"]) if os.environ.get("ROUTE2_MACE_CHECKPOINT") else None)
    if value is None:
        raise ValueError(
            "provide --checkpoint or set ROUTE2_MACE_CHECKPOINT to the official checkpoint"
        )
    path = value.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"checkpoint does not exist: {path}")
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_protocol()
    devices = validate_devices(parse_csv(args.devices, "devices"))
    cases = select_cases(protocol, parse_csv(args.tasks, "tasks"))
    checkpoint = _checkpoint_path(args.checkpoint)
    if _sha(checkpoint) != protocol["checkpoint_sha256"]:
        raise ValueError("checkpoint bytes do not match the preregistered identity")
    output_dir = args.output_dir.resolve()
    create_output_directory(output_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["ROUTE2_MACE_CHECKPOINT"] = str(checkpoint)
    env["PYTHONHASHSEED"] = str(protocol["seed"])
    env["MAPLE_NONMD_SEED"] = str(protocol["seed"])
    started = time.monotonic()
    git_before = _git_record()
    result: dict[str, Any] = {
        "schema": "maple-pure-mace-polar-nonmd-canary-result-v2",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": _sha(PROTOCOL),
        "requested_devices": list(devices),
        "requested_cases": list(cases),
        "git_before": git_before,
        "runtime": {
            "python": sys.version,
            "packages": _package_versions(),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha(checkpoint),
            "dtype": protocol["dtype"],
            "seed": protocol["seed"],
            "identities": protocol["identities"],
        },
        "claim_boundary": protocol["purpose"],
        "confirmation_partition_opened": False,
        "scientific_release_admitted": False,
    }
    selected = run_selected_cases(protocol, devices, cases, output_dir, env)
    result.update(selected)
    result["git_after"] = _git_record()
    result["git_head_unchanged"] = (
        result["git_before"]["head"] == result["git_after"]["head"]
    )
    result["elapsed_seconds"] = time.monotonic() - started
    result["runtime"]["checkpoint_sha256_after"] = (
        _sha(checkpoint) if checkpoint.is_file() else None
    )
    result["runtime"]["checkpoint_unchanged"] = (
        result["runtime"]["checkpoint_sha256"]
        == result["runtime"]["checkpoint_sha256_after"]
    )
    expected_count = len(devices) * len(cases)
    records = list(result["cases"].values())
    acceptable = {"completed", "completed_nonconverged"}
    result["measurement_comparisons"] = measurement_comparisons(
        protocol, result["cases"], devices, cases
    )
    result["execution_complete"] = bool(
        len(records) == expected_count
        and result["source_unchanged"]
        and result["git_head_unchanged"]
        and result["runtime"]["checkpoint_unchanged"]
        and result["measurement_comparisons"]["required_checks_pass"]
        and all(item["infrastructure_status"] in acceptable for item in records)
    )
    result["numerically_converged_cases"] = sorted(
        key for key, item in result["cases"].items()
        if item["numerical"]["converged"] is True
    )
    result["numerically_nonconverged_cases"] = sorted(
        key for key, item in result["cases"].items()
        if item["numerical"]["converged"] is False
    )
    result["unrun_or_infrastructure_failed_cases"] = sorted(
        set(f"{device}/{case}" for device in devices for case in cases)
        - set(result["cases"])
    ) + sorted(
        key for key, item in result["cases"].items()
        if item["infrastructure_status"] not in acceptable
    )
    _write_json(output_dir / "result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="cpu,cuda:0")
    parser.add_argument(
        "--tasks",
        default="all",
        help="comma-separated case IDs or families (sp,opt,scan,freq,ts,irc,all)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        result = run(args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    summary = {
        "requested_devices": result["requested_devices"],
        "requested_cases": result["requested_cases"],
        "execution_complete": result["execution_complete"],
        "numerically_converged": len(result["numerically_converged_cases"]),
        "numerically_nonconverged": len(result["numerically_nonconverged_cases"]),
        "failed_or_unrun": result["unrun_or_infrastructure_failed_cases"],
        "elapsed_seconds": result["elapsed_seconds"],
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if result["execution_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
