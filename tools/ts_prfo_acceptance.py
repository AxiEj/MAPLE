#!/usr/bin/env python3
"""Run and validate MAPLE TS-PRFO acceptance cases.

A case passes only when MAPLE reports Normal Termination and the final TS-mode
validation reports exactly one non-trivial imaginary frequency.  This script is
for precision-preserving performance work: wall time is reported, but speed is
not considered successful unless the scientific TS gate passes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CaseResult:
    name: str
    input: str
    output: str
    prep_input: str | None
    prep_output: str | None
    prep_exit_code: int | None
    prep_wall_sec: float | None
    prep_ts_xyz: str | None
    opt_input: str | None
    opt_output: str | None
    opt_exit_code: int | None
    opt_converged: bool | None
    opt_wall_sec: float | None
    exit_code: int
    external_wall_sec: float
    maple_wall_sec: float | None
    normal_termination: bool
    max_iterations_reached: bool
    imaginary_frequencies: int | None
    lowest_frequency_cm1: float | None
    final_energy: float | None
    passed: bool


def _replace_model(text: str, model: str | None) -> str:
    if not model:
        return text
    return re.sub(r"(?im)^#\s*model\s*=\s*[^\n]+", f"#model={model}", text, count=1)


def _rewrite_xyz_paths(text: str, base_dir: Path) -> str:
    """Make example ``XYZ`` references portable when copied into ``run_root``.

    MAPLE examples sometimes contain absolute paths from the original author's
    checkout.  For acceptance runs, keep existing valid paths as-is; otherwise
    replace them with ``base_dir / basename`` when that local file exists.
    """

    def repl(match: re.Match[str]) -> str:
        prefix, raw_path = match.group(1), match.group(2).strip()
        path = Path(raw_path)
        if path.exists():
            return match.group(0)
        candidate = base_dir / path.name
        if candidate.exists():
            return f"{prefix}{candidate}"
        return match.group(0)

    return re.sub(r"(?im)^(\s*XYZ\s+)([^\n]+)$", repl, text)


def _replace_or_add_task(text: str, task_line: str) -> str:
    """Replace the first task directive with ``task_line``."""
    pattern = re.compile(r"(?im)^#\s*(?:sp|opt|ts|scan|freq|irc|md)\b[^\n]*")
    if pattern.search(text):
        return pattern.sub(task_line, text, count=1)
    return task_line + "\n" + text


def _extract_settings(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip().startswith("#")]


def _read_xyz_atoms(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"empty xyz file: {path}")
    n_atoms = int(lines[0].strip())
    return lines[2 : 2 + n_atoms]


def _write_input_from_xyz(settings: list[str], xyz_lines: list[str], path: Path) -> None:
    path.write_text("\n".join(settings + [""] + xyz_lines) + "\n")


def _parse_output(path: Path) -> dict:
    text = path.read_text(errors="ignore") if path.exists() else ""
    energies = [float(m.group(1)) for m in re.finditer(r"Energy:\s+([-+0-9.]+)", text)]
    imag = None
    m = re.search(r"Imaginary frequencies \([^)]*\):\s+(\d+)", text)
    if m:
        imag = int(m.group(1))
    lowest = None
    m = re.search(r"Lowest frequency:\s+([-+0-9.]+)\s+cm", text)
    if m:
        lowest = float(m.group(1))
    maple_wall = None
    m = re.search(r"Total wall time:\s+([-+0-9.]+)\s+s", text)
    if m:
        maple_wall = float(m.group(1))
    return {
        "normal": "Normal Termination" in text,
        "max_iter": "Maximum Iterations Reached" in text,
        "imag": imag,
        "lowest": lowest,
        "maple_wall": maple_wall,
        "energy": energies[-1] if energies else None,
        "opt_converged": bool(re.search(r"\bconverged at iteration\b", text, re.I)),
    }


def _run_maple(input_path: Path, output_path: Path, env: dict[str, str], case_dir: Path, stem: str):
    cmd = [sys.executable, "-m", "maple.main", str(input_path), str(output_path)]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=Path.cwd(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    external = time.perf_counter() - t0
    (case_dir / f"{stem}_stdout.txt").write_text(proc.stdout)
    (case_dir / f"{stem}_stderr.txt").write_text(proc.stderr)
    return proc, external


def run_case(
    case: str,
    run_root: Path,
    device: str | None,
    *,
    nebts_first: bool = False,
    neb_max_iter: int | None = None,
    neb_params_extra: tuple[str, ...] = (),
    opt_first: bool = False,
    opt_method: str = "lbfgs",
    opt_max_iter: int | None = None,
    ts_max_iter: int | None = None,
) -> CaseResult:
    # Format: name=input_path[:model_override].
    # The model override may include MAPLE model options, e.g.
    # ``uma(size=uma-s-1p2)``.
    if "=" not in case:
        raise ValueError("case must be NAME=INPUT[:MODEL]")
    name, spec = case.split("=", 1)
    parts = spec.split(":", 1)
    input_path = Path(parts[0]).resolve()
    model = parts[1] if len(parts) == 2 else None

    case_dir = run_root / name
    case_dir.mkdir(parents=True, exist_ok=True)
    run_input = case_dir / input_path.name
    run_output = case_dir / (input_path.stem + ".out")
    source_text = _replace_model(input_path.read_text(), model)
    source_text = _rewrite_xyz_paths(source_text, input_path.parent)
    ts_params = "method=prfo" if ts_max_iter is None else f"method=prfo,max_iter={ts_max_iter}"
    run_input.write_text(_replace_or_add_task(source_text, f"#ts({ts_params})"))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = device

    prep_input = prep_output = prep_ts_xyz = None
    prep_exit_code = None
    prep_wall = None
    opt_input = opt_output = None
    opt_exit_code = None
    opt_converged = None
    opt_wall = None

    def early_failure(reason: str) -> CaseResult:
        run_output.write_text(f"Acceptance preparation failed: {reason}\n")
        return CaseResult(
            name=name,
            input=str(run_input),
            output=str(run_output),
            prep_input=str(prep_input) if prep_input is not None else None,
            prep_output=str(prep_output) if prep_output is not None else None,
            prep_exit_code=prep_exit_code,
            prep_wall_sec=prep_wall,
            prep_ts_xyz=str(prep_ts_xyz) if prep_ts_xyz is not None else None,
            opt_input=str(opt_input) if opt_input is not None else None,
            opt_output=str(opt_output) if opt_output is not None else None,
            opt_exit_code=opt_exit_code,
            opt_converged=opt_converged,
            opt_wall_sec=opt_wall,
            exit_code=1,
            external_wall_sec=0.0,
            maple_wall_sec=None,
            normal_termination=False,
            max_iterations_reached=False,
            imaginary_frequencies=None,
            lowest_frequency_cm1=None,
            final_energy=None,
            passed=False,
        )

    if nebts_first and opt_first:
        raise ValueError("--nebts-first and --opt-first are separate preparation modes")

    if nebts_first:
        neb_params = "method=neb,refine=nebts"
        if neb_max_iter is not None:
            neb_params += f",max_iter={neb_max_iter}"
        for item in neb_params_extra:
            neb_params += f",{item}"
        prep_input = case_dir / f"{input_path.stem}_nebts.inp"
        prep_output = case_dir / f"{input_path.stem}_nebts.out"
        prep_input.write_text(_replace_or_add_task(source_text, f"#ts({neb_params})"))
        prep_proc, prep_external = _run_maple(prep_input, prep_output, env, case_dir, "nebts")
        prep_exit_code = prep_proc.returncode
        prep_wall = prep_external

        prep_ts_xyz = prep_output.with_name(prep_output.stem + "_nebts_ts.xyz")
        if prep_exit_code == 0 and prep_ts_xyz.exists():
            settings = [
                f"#ts({ts_params})" if re.match(r"(?i)^#\s*ts\b", line.strip()) else line
                for line in _extract_settings(prep_input.read_text())
            ]
            _write_input_from_xyz(settings, _read_xyz_atoms(prep_ts_xyz), run_input)
        else:
            return early_failure(
                f"NEBTS preparation did not produce {prep_ts_xyz}"
            )

    if opt_first:
        opt_params = f"method={opt_method}"
        if opt_max_iter is not None:
            opt_params += f",max_iter={opt_max_iter}"
        opt_input = case_dir / f"{input_path.stem}_opt.inp"
        opt_output = case_dir / f"{input_path.stem}_opt.out"
        opt_input.write_text(_replace_or_add_task(source_text, f"#opt({opt_params})"))
        opt_proc, opt_external = _run_maple(opt_input, opt_output, env, case_dir, "opt")
        opt_exit_code = opt_proc.returncode
        opt_wall = opt_external
        opt_parsed = _parse_output(opt_output)
        opt_converged = opt_parsed["opt_converged"]

        opt_xyz = opt_output.with_name(opt_output.stem + "_opt.xyz")
        if opt_exit_code != 0 or not opt_converged or not opt_xyz.exists():
            return early_failure(
                f"OPT preparation did not converge or produce {opt_xyz}"
            )
        settings = [
            line for line in _extract_settings(run_input.read_text())
            if not re.match(r"(?i)^#\s*opt\b", line.strip())
        ]
        settings = [
            f"#ts({ts_params})" if re.match(r"(?i)^#\s*ts\b", line.strip()) else line
            for line in settings
        ]
        _write_input_from_xyz(settings, _read_xyz_atoms(opt_xyz), run_input)

    proc, external = _run_maple(run_input, run_output, env, case_dir, "ts")

    parsed = _parse_output(run_output)
    passed = (
        proc.returncode == 0
        and parsed["normal"]
        and parsed["imag"] == 1
        and not parsed["max_iter"]
        and (
            not nebts_first
            or (prep_exit_code == 0 and prep_ts_xyz is not None and prep_ts_xyz.exists())
        )
        and (
            not opt_first
            or (opt_exit_code == 0 and opt_converged)
        )
    )
    return CaseResult(
        name=name,
        input=str(run_input),
        output=str(run_output),
        prep_input=str(prep_input) if prep_input is not None else None,
        prep_output=str(prep_output) if prep_output is not None else None,
        prep_exit_code=prep_exit_code,
        prep_wall_sec=prep_wall,
        prep_ts_xyz=str(prep_ts_xyz) if prep_ts_xyz is not None else None,
        opt_input=str(opt_input) if opt_input is not None else None,
        opt_output=str(opt_output) if opt_output is not None else None,
        opt_exit_code=opt_exit_code,
        opt_converged=opt_converged,
        opt_wall_sec=opt_wall,
        exit_code=proc.returncode,
        external_wall_sec=external,
        maple_wall_sec=parsed["maple_wall"],
        normal_termination=parsed["normal"],
        max_iterations_reached=parsed["max_iter"],
        imaginary_frequencies=parsed["imag"],
        lowest_frequency_cm1=parsed["lowest"],
        final_energy=parsed["energy"],
        passed=passed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Acceptance case NAME=INPUT[:MODEL_OVERRIDE], e.g. case=inp.inp:uma(size=uma-s-1p2)",
    )
    parser.add_argument("--run-root", default=None, help="Output directory")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--nebts-first", action="store_true", help="Run same-model NEB(refine=nebts), then PRFO from <neb>_nebts_ts.xyz")
    parser.add_argument("--neb-max-iter", type=int, default=None)
    parser.add_argument(
        "--neb-param",
        action="append",
        default=[],
        help="Extra key=value parameter appended to the NEB preparation task",
    )
    parser.add_argument("--opt-first", action="store_true", help="Run same-model OPT first, then PRFO from <opt>_opt.xyz")
    parser.add_argument("--opt-method", default="lbfgs")
    parser.add_argument("--opt-max-iter", type=int, default=None)
    parser.add_argument("--ts-max-iter", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args()

    run_root = Path(args.run_root or f"/tmp/maple-ts-prfo-acceptance-{int(time.time())}")
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True)

    results = [
        run_case(
            case,
            run_root,
            args.cuda_visible_devices,
            nebts_first=args.nebts_first,
            neb_max_iter=args.neb_max_iter,
            neb_params_extra=tuple(args.neb_param),
            opt_first=args.opt_first,
            opt_method=args.opt_method,
            opt_max_iter=args.opt_max_iter,
            ts_max_iter=args.ts_max_iter,
        )
        for case in args.case
    ]
    payload = {"run_root": str(run_root), "results": [asdict(r) for r in results]}

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"RUN_ROOT={run_root}")
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            wall = r.maple_wall_sec if r.maple_wall_sec is not None else r.external_wall_sec
            print(
                f"{status} {r.name}: wall={wall:.3f}s exit={r.exit_code} "
                f"normal={r.normal_termination} imag={r.imaginary_frequencies} "
                f"lowest={r.lowest_frequency_cm1} output={r.output}"
            )
            if r.opt_input is not None:
                print(
                    f"  opt: exit={r.opt_exit_code} converged={r.opt_converged} "
                    f"wall={r.opt_wall_sec:.3f}s output={r.opt_output}"
                )
            if r.prep_input is not None:
                print(
                    f"  prep: exit={r.prep_exit_code} wall={r.prep_wall_sec:.3f}s "
                    f"output={r.prep_output} ts_xyz={r.prep_ts_xyz}"
                )

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
