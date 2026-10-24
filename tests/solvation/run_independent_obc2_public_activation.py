#!/usr/bin/env python3
"""Run the final independent-OBC2 release gate through installed ``maple`` only.

The historical wheelhouse is a read-only dependency source.  All generated
inputs, raw streams, exit codes and receipts are written under the new dated
benchmark directory.  This harness must not be used for candidate qualification:
its analytic tasks require the packaged pointer and qualification receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import math
import re
import shutil
import subprocess
import tempfile
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / ".omx/benchmarks/route1-independent-obc2-default-20261023"
HISTORICAL_WHEELHOUSE = (
    ROOT / ".omx/benchmarks/route1-independent-recheck-20260930T145212Z/"
    "isolation-source/.omx/recheck/wheelhouse"
)
DEFAULT_VENV = RUN_ROOT / "fresh-activation-venv"
DEFAULT_RUNTIME = RUN_ROOT / "public-cli-runtime"
DEFAULT_RECEIPT = RUN_ROOT / "public-cli-activation.json"
FIXTURE = ROOT / "tests/solvation/data/obc2_openmm_8_5_2_reference_v1.json"
ANI2X = ROOT / "maple/function/calculator/model/ani2x.pt"
ANI2X_SHA256 = "c8c41e89d5ccbe7b5c9f46ce4a262b913ad0c428a3a3742cc6e83ac10d4d43aa"
WATER_MOL2 = """@<TRIPOS>MOLECULE
WATER
 3 2 1 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
      1 O1          0.000000    0.000000    0.000000 O.3       1 WAT       -0.834000
      2 H1          0.957200    0.000000    0.000000 H         1 WAT        0.417000
      3 H2         -0.239987    0.927297    0.000000 H         1 WAT        0.417000
@<TRIPOS>BOND
     1    1    2 1
     2    1    3 1
@<TRIPOS>SUBSTRUCTURE
     1 WAT         1 TEMP              0 ****  ****    0 ROOT
"""
WATER_MOL2_SHA256 = "54a77f1a50e0f37782aba1244143b660a8dba554444f5a342d243eefa10f27a9"
TASK_ORDER = (
    "sp", "opt", "scan", "nve", "nvt", "analytic_freq", "prfo",
    "direct_hvp_dimer", "numerical_freq", "force_difference_dimer",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def under_run_root(path: Path, role: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(RUN_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{role} must remain under {RUN_ROOT}: {resolved}") from exc
    return resolved


def sealed_fixture() -> dict:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload = dict(fixture)
    recorded = payload.pop("content_sha256")
    observed = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    if observed != recorded or fixture.get("comparison_count") != 48:
        raise ValueError("Frozen portable oracle is incomplete or changed")
    return fixture


def render_input(task: str, mol2: Path, checkpoint: Path) -> str:
    analytic = task in {"analytic_freq", "prfo", "direct_hvp_dimer"}
    numerical = task == "numerical_freq"
    hessian = "analytic" if analytic else "numerical" if numerical else None
    model_options = f"model_path={checkpoint}"
    if hessian:
        model_options += f",hessian={hessian}"
    task_lines = {
        "sp": "#sp(verbose=1)",
        "opt": "#opt(method=lbfgs,max_iter=2,verbose=0,log_final_paths=false)",
        "scan": "#scan(method=lbfgs,mode=rigid)",
        "nve": "#md(ensemble=nve,steps=2,timestep=0.05,traj_every=1,log_every=1,remove_com_every=0,rst_every=100)",
        "nvt": "#md(ensemble=nvt,steps=2,timestep=0.05,temperature=298.15,traj_every=1,log_every=1,random_seed=20261023,remove_com_every=0,rst_every=100)",
        "analytic_freq": "#freq(method=mw,ilowfreq=2)",
        "prfo": "#ts(method=prfo,max_iter=1)",
        "direct_hvp_dimer": "#ts(method=dimer,use_hvp=true,max_iter=1,rot_max_iter=1,n_init=force)",
        "numerical_freq": "#freq(method=mw,ilowfreq=2)",
        "force_difference_dimer": "#ts(method=dimer,use_hvp=false,max_iter=1,rot_max_iter=1,n_init=force)",
    }
    lines = [
        f"#model=ani2x({model_options})", task_lines[task], "#device=cpu",
        "#charge(source=mol2,label=frozen-public-activation)",
        "#solv(implicit=water,method=gb,model=obc2,profile=obc2-mbondi2,nonpolar=ace,experimental=true)",
        "", "0 1", f"MOL2 {mol2}",
    ]
    if task == "scan":
        lines += ["", "S 1 2 0.01 2"]
    return "\n".join(lines) + "\n"


GUARD = r'''
import importlib.abc, importlib.metadata, socket, sys
class _BlockOpenMM(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "openmm" or fullname.startswith("openmm."):
            raise AssertionError("forbidden OpenMM import: " + fullname)
        return None
sys.meta_path.insert(0, _BlockOpenMM())
_version = importlib.metadata.version
_distribution = importlib.metadata.distribution
def _guarded_version(name):
    if str(name).lower() == "openmm": raise AssertionError("forbidden OpenMM metadata lookup")
    return _version(name)
def _guarded_distribution(name):
    if str(name).lower() == "openmm": raise AssertionError("forbidden OpenMM metadata lookup")
    return _distribution(name)
importlib.metadata.version = _guarded_version
importlib.metadata.distribution = _guarded_distribution
def _deny(*args, **kwargs): raise AssertionError("network access is forbidden")
socket.socket.connect = _deny
socket.create_connection = _deny
'''.strip() + "\n"


def clean_environment() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONNOUSERSITE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MAPLE_OFFLINE="1")
    return env


def run_raw(command: list[str], cwd: Path, env: dict[str, str], stem: Path, timeout: int) -> dict:
    timed_out = False
    try:
        completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True,
                                   check=False, timeout=timeout)
        returncode = completed.returncode
        stdout_text = completed.stdout
        stderr_text = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout_text = exc.stdout or ""
        stderr_text = exc.stderr or ""
        if isinstance(stdout_text, bytes):
            stdout_text = stdout_text.decode(errors="replace")
        if isinstance(stderr_text, bytes):
            stderr_text = stderr_text.decode(errors="replace")
        stderr_text += f"\nTIMEOUT after {timeout} seconds\n"
    stem.parent.mkdir(parents=True, exist_ok=True)
    stdout = stem.with_suffix(".stdout"); stderr = stem.with_suffix(".stderr")
    stdout.write_text(stdout_text, encoding="utf-8")
    stderr.write_text(stderr_text, encoding="utf-8")
    stem.with_suffix(".exitcode").write_text(f"{returncode}\n", encoding="ascii")
    return {"command": command, "cwd": str(cwd), "returncode": returncode,
            "timed_out": timed_out,
            "stdout": str(stdout), "stdout_sha256": sha256(stdout),
            "stderr": str(stderr), "stderr_sha256": sha256(stderr)}


def _xyz_frames(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    frames = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        count = int(lines[cursor])
        end = cursor + count + 2
        if end > len(lines):
            raise ValueError(f"truncated XYZ frame: {path}")
        coordinates = [
            [float(value) for value in line.split()[1:4]]
            for line in lines[cursor + 2 : end]
        ]
        if len(coordinates) != count or not all(
            math.isfinite(value) for row in coordinates for value in row
        ):
            raise ValueError(f"invalid XYZ coordinates: {path}")
        frames.append({"comment": lines[cursor + 1], "coordinates": coordinates})
        cursor = end
    return frames


def _md_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 6:
            raise ValueError(f"invalid MD thermodynamics row: {line}")
        step = int(fields[0])
        values = [float(value) for value in fields[1:]]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("non-finite MD thermodynamics")
        records.append({"step": step, "values": values})
    return records


def _mol2_with_positions(source: Path, destination: Path, positions: list[list[float]]) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index("@<TRIPOS>ATOM") + 1
        end = lines.index("@<TRIPOS>BOND", start)
    except ValueError as exc:
        raise ValueError("frozen MOL2 atom/bond sections are missing") from exc
    atom_lines = lines[start:end]
    if len(atom_lines) != len(positions):
        raise ValueError("optimized geometry atom count changed")
    rewritten = []
    for line, xyz in zip(atom_lines, positions):
        fields = line.split()
        if len(fields) < 6 or len(xyz) != 3:
            raise ValueError("invalid MOL2 atom record")
        rewritten.append(
            " ".join(
                fields[:2]
                + [f"{float(value):.10f}" for value in xyz]
                + fields[5:]
            )
        )
    lines[start:end] = rewritten
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_task(task: str, runtime_cwd: Path, *, stem: str | None = None) -> dict:
    stem = stem or task
    output = runtime_cwd / f"{stem}.out"
    if not output.is_file():
        raise ValueError(f"missing task output: {output}")
    text = output.read_text(encoding="utf-8")
    fatal_patterns = (
        "ERROR:",
        "Traceback (most recent call last)",
        "Frequency analysis failed",
        "Calculation failed",
    )
    fatal_markers = [pattern for pattern in fatal_patterns if pattern in text]
    incomplete_patterns = (
        "did NOT converge",
        "Maximum Iterations Reached",
        "Dimer did not converge to a TS candidate",
    )
    incomplete_markers = [pattern for pattern in incomplete_patterns if pattern in text]
    manifest_path = output.with_suffix(output.suffix + ".implicit") / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    solvation = manifest["solvation"]
    provider_passed = bool(
        solvation.get("runtime_provider") == "torch-obc2"
        and solvation.get("runtime_openmm_required") is False
        and solvation.get("provider") == "torch-obc2"
    )
    evidence: dict[str, object] = {}
    if task == "sp":
        match = re.search(r"^Energy:\s+([-+0-9.eE]+) Hartree$", text, re.MULTILINE)
        gradient_section = text.split("Gradients (Hartree/Angstrom):", 1)
        gradient_values = []
        if len(gradient_section) == 2:
            for line in gradient_section[1].split("\n\n", 1)[0].splitlines():
                fields = line.split()
                if len(fields) == 5 and fields[0].isdigit():
                    try:
                        gradient_values.extend(float(value) for value in fields[2:5])
                    except ValueError:
                        pass
        evidence = {
            "finite_energy": bool(match and math.isfinite(float(match.group(1)))),
            "force_component_count": len(gradient_values),
            "finite_forces": bool(
                gradient_values and all(math.isfinite(value) for value in gradient_values)
            ),
        }
    elif task == "opt":
        frames = _xyz_frames(runtime_cwd / "opt_opt.xyz")
        energies = [float(re.search(r"Energy =\s*([-+0-9.eE]+)", row["comment"]).group(1)) for row in frames]
        evidence = {"frame_count": len(frames), "finite_energies": all(map(math.isfinite, energies)),
                    "bounded_nonconvergence_recorded": "did NOT converge" in text}
    elif task == "scan":
        frames = _xyz_frames(runtime_cwd / "scan_scan_final.xyz")
        energies = [float(re.search(r"Energy =\s*([-+0-9.eE]+)", row["comment"]).group(1)) for row in frames]
        evidence = {"frame_count": len(frames), "finite_energies": all(map(math.isfinite, energies)),
                    "completion_marker": "Scan completed! Total points: 3" in text}
    elif task in {"nve", "nvt"}:
        records = _md_records(runtime_cwd / f"{task}_md_thermo.dat")
        trajectory = _xyz_frames(runtime_cwd / f"{task}_md_traj.xyz")
        evidence = {"thermo_steps": [row["step"] for row in records],
                    "trajectory_frame_count": len(trajectory),
                    "completion_marker": f"{task.upper()} simulation completed successfully." in text}
    elif task == "analytic_freq":
        evidence = {"completion_marker": "Frequency analysis completed" in text}
    elif task == "numerical_freq":
        evidence = {
            "completion_marker": "Frequency analysis completed" in text,
            "analytic_derivative_audit_absent": not (
                manifest_path.with_name("derivative-use.json").exists()
            ),
        }
    elif task == "prfo":
        evidence = {"started": "Starting Transition State Search (TS) with RS-PRFO" in text,
                    "minimum_not_misreported_as_ts": "Normal Termination" not in text}
    elif task == "direct_hvp_dimer":
        evidence = {"derivative_marker": "Derivative mode                         ....  hvp" in text}
    elif task == "force_difference_dimer":
        evidence = {"dimer_started": "DIMER INITIAL STATE" in text,
                    "force_difference_selected": (
                        "Derivative mode                         ....  finite_difference"
                        in text
                    )}
    expected_derivative = {
        "analytic_freq": "hessian",
        "prfo": "hessian",
        "direct_hvp_dimer": "directional-hvp",
    }.get(task)
    derivative_use = None
    if expected_derivative is not None:
        derivative_path = manifest_path.with_name("derivative-use.json")
        derivative_use = json.loads(derivative_path.read_text(encoding="utf-8"))
        if not (
            derivative_use.get("status") == "success"
            and derivative_use.get("derivative") == expected_derivative
            and derivative_use.get("runtime_provider") == "torch-obc2"
        ):
            raise ValueError(f"{task} derivative-use audit failed")
    if task == "sp":
        evidence_passed = bool(
            evidence["finite_energy"] is True
            and evidence["finite_forces"] is True
            and evidence["force_component_count"] > 0
        )
    elif task == "opt":
        evidence_passed = bool(
            evidence["frame_count"] >= 1 and evidence["finite_energies"] is True
        )
    elif task == "scan":
        evidence_passed = bool(
            evidence["frame_count"] >= 3
            and evidence["finite_energies"] is True
            and evidence["completion_marker"] is True
        )
    elif task in {"nve", "nvt"}:
        evidence_passed = bool(
            evidence["thermo_steps"] == [1, 2]
            and evidence["trajectory_frame_count"] == 2
            and evidence["completion_marker"] is True
        )
    else:
        evidence_passed = all(value is True for value in evidence.values())
    passed = not fatal_markers and provider_passed and evidence_passed
    return {
        "fatal_markers": fatal_markers,
        "incomplete_markers": incomplete_markers,
        "provider": {
            "runtime_provider": solvation.get("runtime_provider"),
            "runtime_openmm_required": solvation.get("runtime_openmm_required"),
            "provider": solvation.get("provider"),
            "passed": provider_passed,
        },
        "evidence": evidence,
        "derivative_use": derivative_use,
        "passed": passed,
    }


def describe(args: argparse.Namespace) -> dict:
    return {"stage": "installed activation, not candidate qualification", "network_allowed": False,
            "system_site_packages": False, "pythonpath_allowed": False,
            "openmm_allowed": False, "console_script_only": True, "task_order": list(TASK_ORDER),
            "post_opt_replay": "opt_final_sp",
            "wheelhouse_source": str(args.wheelhouse), "venv": str(args.venv),
            "runtime": str(args.runtime), "receipt": str(args.receipt),
            "checkpoint_sha256": ANI2X_SHA256}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--candidate-wheel", type=Path)
    parser.add_argument("--wheelhouse", type=Path, default=HISTORICAL_WHEELHOUSE)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    args.venv = under_run_root(args.venv, "venv")
    args.runtime = under_run_root(args.runtime, "runtime")
    args.receipt = under_run_root(args.receipt, "receipt")
    if args.describe:
        print(json.dumps(describe(args), indent=2, sort_keys=True)); return
    fixture = sealed_fixture()
    row = next(r for r in fixture["records"] if (r["case"], r["nonpolar"], r["geometry"]) == ("methanol", "ace", "base"))
    source_mol2 = ROOT / row["source_mol2_relative_path"]
    if sha256(source_mol2) != row["source_mol2_sha256"] or sha256(ANI2X) != ANI2X_SHA256:
        raise ValueError("Frozen public input or ANI2x checkpoint changed")
    if args.runtime.exists(): shutil.rmtree(args.runtime)
    assets = args.runtime / "assets"; inputs = args.runtime / "inputs"; raw = args.runtime / "raw"; cwd = args.runtime / "cwd"
    for path in (assets, inputs, raw, cwd): path.mkdir(parents=True)
    methanol = assets / "methanol.mol2"; water = assets / "water.mol2"
    checkpoint = assets / "ani2x.pt"
    shutil.copyfile(source_mol2, methanol)
    water.write_text(WATER_MOL2, encoding="utf-8")
    if sha256(water) != WATER_MOL2_SHA256:
        raise AssertionError("Frozen water input changed")
    shutil.copyfile(ANI2X, checkpoint)
    rendered = {}
    for task in TASK_ORDER:
        # Curvature tasks use the frozen water geometry; product E/F tasks use
        # the independently frozen non-water methanol oracle input.
        molecule = water if task in {"analytic_freq", "prfo", "direct_hvp_dimer", "numerical_freq", "force_difference_dimer"} else methanol
        path = inputs / f"{task}.inp"; path.write_text(render_input(task, molecule, checkpoint), encoding="utf-8")
        rendered[task] = {"path": str(path), "sha256": sha256(path),
                          "molecule": molecule.stem, "molecule_sha256": sha256(molecule)}
    if args.render_only:
        result = {**describe(args), "inputs": rendered, "passed": True}
        atomic_json(args.receipt, result); print(json.dumps(result, indent=2, sort_keys=True)); return
    if args.candidate_wheel is None or not zipfile.is_zipfile(args.candidate_wheel):
        raise ValueError("--candidate-wheel must name the final built wheel")
    if args.venv.exists(): shutil.rmtree(args.venv)
    venv.EnvBuilder(with_pip=True, clear=True, system_site_packages=False).create(args.venv)
    python = args.venv / "bin/python"; maple = args.venv / "bin/maple"; env = clean_environment()
    install = run_raw([str(python), "-I", "-m", "pip", "install", "--no-index", "--no-cache-dir",
                       f"--find-links={args.wheelhouse}", f"{args.candidate_wheel}[implicit-obc2-torch]"],
                      cwd, env, raw / "install", 900)
    if install["returncode"] != 0:
        failure = {
            **describe(args),
            "candidate_wheel": {"path": str(args.candidate_wheel), "sha256": sha256(args.candidate_wheel)},
            "inputs": rendered,
            "install": install,
            "tasks": {task: {"status": "NOT_RUN_INSTALL_FAILED"} for task in TASK_ORDER},
            "failed_stage": "install",
            "passed": False,
        }
        atomic_json(args.receipt, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        raise SystemExit(1)
    purelib = subprocess.check_output([str(python), "-I", "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"], text=True).strip()
    # Prove physical absence before enabling the stricter runtime trap.  Once
    # enabled, even probing OpenMM metadata is intentionally an error.
    probe_code = "import importlib.util,importlib.metadata,json,maple,sys;\ntry: d=importlib.metadata.distribution('openmm')\nexcept importlib.metadata.PackageNotFoundError: d=None\nprint(json.dumps({'maple_file':maple.__file__,'sys_path':sys.path,'openmm_spec':None if importlib.util.find_spec('openmm') is None else 'present','openmm_distribution':None if d is None else 'present','openmm_loaded':any(k=='openmm' or k.startswith('openmm.') for k in sys.modules)}))"
    probe = run_raw([str(python), "-I", "-c", probe_code], cwd, env, raw / "preflight", 60)
    if probe["returncode"] != 0:
        failure = {
            **describe(args), "inputs": rendered, "install": install,
            "preflight": probe,
            "tasks": {task: {"status": "NOT_RUN_PREFLIGHT_FAILED"} for task in TASK_ORDER},
            "failed_stage": "preflight", "passed": False,
        }
        atomic_json(args.receipt, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        raise SystemExit(1)
    probe_payload = json.loads(Path(probe["stdout"]).read_text(encoding="utf-8"))
    if any(probe_payload[key] is not None for key in ("openmm_spec", "openmm_distribution")) or probe_payload["openmm_loaded"]:
        raise RuntimeError("OpenMM is present in the fresh activation environment")
    maple_file = Path(probe_payload["maple_file"]).resolve()
    maple_file.relative_to(Path(purelib).resolve())
    if any(Path(entry).resolve() == ROOT.resolve() for entry in probe_payload["sys_path"] if entry):
        raise RuntimeError("source checkout leaked onto activation sys.path")
    Path(purelib, "maple_activation_guard.py").write_text(GUARD, encoding="utf-8")
    Path(purelib, "maple_activation_guard.pth").write_text("import maple_activation_guard\n", encoding="utf-8")
    tasks = {}
    for task in TASK_ORDER:
        tasks[task] = run_raw([str(maple), str(inputs / f"{task}.inp"), str(cwd / f"{task}.out")],
                              cwd, env, raw / task, 1800)
        output = cwd / f"{task}.out"
        tasks[task]["input_sha256"] = rendered[task]["sha256"]
        tasks[task]["output"] = str(output)
        tasks[task]["output_sha256"] = sha256(output) if output.is_file() else None
        try:
            tasks[task]["audit"] = audit_task(task, cwd)
        except Exception as exc:
            tasks[task]["audit"] = {
                "passed": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
    # Replay the final OPT geometry through a new public console-script SP.  The
    # MOL2 charges/topology remain frozen; only coordinates come from OPT.
    try:
        opt_frames = _xyz_frames(cwd / "opt_opt.xyz")
        optimized_mol2 = assets / "methanol-opt-final.mol2"
        _mol2_with_positions(methanol, optimized_mol2, opt_frames[-1]["coordinates"])
        replay_input = inputs / "opt_final_sp.inp"
        replay_input.write_text(render_input("sp", optimized_mol2, checkpoint), encoding="utf-8")
        rendered["opt_final_sp"] = {
            "path": str(replay_input),
            "sha256": sha256(replay_input),
            "molecule": "methanol-opt-final",
            "molecule_sha256": sha256(optimized_mol2),
        }
        replay = run_raw(
            [str(maple), str(replay_input), str(cwd / "opt_final_sp.out")],
            cwd, env, raw / "opt_final_sp", 1800,
        )
        replay_output = cwd / "opt_final_sp.out"
        replay["input_sha256"] = rendered["opt_final_sp"]["sha256"]
        replay["output"] = str(replay_output)
        replay["output_sha256"] = sha256(replay_output) if replay_output.is_file() else None
        replay["audit"] = audit_task("sp", cwd, stem="opt_final_sp")
    except Exception as exc:
        replay = {
            "returncode": None,
            "audit": {
            "passed": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        }
    tasks["opt_final_sp"] = replay
    guard_canaries = {
        "openmm_import": run_raw(
            [str(python), "-c", "import openmm"], cwd, env,
            raw / "guard-openmm-import", 60,
        ),
        "openmm_metadata": run_raw(
            [str(python), "-c", "import importlib.metadata; importlib.metadata.distribution('openmm')"],
            cwd, env, raw / "guard-openmm-metadata", 60,
        ),
        "network": run_raw(
            [str(python), "-c", "import socket; socket.create_connection(('127.0.0.1', 9))"],
            cwd, env, raw / "guard-network", 60,
        ),
    }
    expected_guard_messages = {
        "openmm_import": "forbidden OpenMM import",
        "openmm_metadata": "forbidden OpenMM metadata lookup",
        "network": "network access is forbidden",
    }
    guard_passed = True
    for name, record in guard_canaries.items():
        stderr_text = Path(record["stderr"]).read_text(encoding="utf-8")
        record["expected_failure_observed"] = bool(
            record["returncode"] != 0 and expected_guard_messages[name] in stderr_text
        )
        guard_passed = guard_passed and record["expected_failure_observed"]
    postflight_code = (
        "import json,pathlib,sys,sysconfig;"
        "p=pathlib.Path(sysconfig.get_paths()['purelib']);"
        "hits=sorted(x.name for x in p.iterdir() if x.name.lower()=='openmm' or "
        "x.name.lower().startswith('openmm-') or x.name.lower().startswith('openmm_') or "
        "x.name.lower().startswith('openmm.'));"
        "print(json.dumps({'physical_openmm_paths':hits,'openmm_loaded':any(k=='openmm' or "
        "k.startswith('openmm.') for k in sys.modules),'guard_loaded':'maple_activation_guard' in sys.modules}))"
    )
    postflight = run_raw(
        [str(python), "-I", "-c", postflight_code], cwd, env,
        raw / "postflight-physical-absence", 60,
    )
    postflight_payload = (
        json.loads(Path(postflight["stdout"]).read_text(encoding="utf-8"))
        if postflight["returncode"] == 0
        else {}
    )
    postflight["physical_absence_passed"] = bool(
        not postflight_payload.get("physical_openmm_paths")
        and postflight_payload.get("openmm_loaded") is False
        and postflight_payload.get("guard_loaded") is True
    )
    passed = bool(
        probe["returncode"] == 0
        and all(
            row["returncode"] == 0 and row["audit"]["passed"] is True
            for row in tasks.values()
        )
        and guard_passed
        and postflight["physical_absence_passed"]
    )
    receipt = {**describe(args), "candidate_wheel": {"path": str(args.candidate_wheel), "sha256": sha256(args.candidate_wheel)},
               "wheelhouse_files": {p.name: sha256(p) for p in sorted(args.wheelhouse.glob("*.whl"))},
               "inputs": rendered, "install": install, "preflight": probe,
               "tasks": tasks, "guard_canaries": guard_canaries,
               "postflight": postflight, "passed": passed}
    atomic_json(args.receipt, receipt); print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed: raise SystemExit(1)


if __name__ == "__main__":
    main()
