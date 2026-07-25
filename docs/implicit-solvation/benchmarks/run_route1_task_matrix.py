#!/usr/bin/env python3
"""Run Route 1 through MAPLE's actual SP, OPT, SCAN, and MD dispatchers."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core import (  # noqa: E402
    command_provenance,
    load_json,
    seal_artifact,
    sha256_file,
    write_json_atomic,
)

FORMULA = "E_solution(R) = E_MLIP,gas(R) + G_polar(R,q_fixed) + G_nonpolar(R)"
REQUIRED_SOLVATION_KEYS = {
    "gas_energy_hartree",
    "delta_g_solv_hartree",
    "combined_energy_hartree",
    "components_hartree",
    "provenance",
}
SCAN_COMMENT = re.compile(
    r"Scanning combination \d+/\d+: \[([^\]]+)\]\s+Energy = "
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
)


def _load_am1bcc_record(
    manifest_path: Path,
    compound_id: str,
    *,
    atom_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = load_json(manifest_path)
    matches = [
        record
        for record in manifest.get("records", [])
        if record.get("compound_id") == compound_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one AM1-BCC source record for {compound_id}, found {len(matches)}."
        )
    record = matches[0]
    charges = np.asarray(record.get("am1bcc_charges_e"), dtype=np.float64)
    if charges.shape != (atom_count,) or not np.isfinite(charges).all():
        raise ValueError(
            "AM1-BCC manifest does not contain one finite charge per atom."
        )
    return charges, record


def _task_line(
    task: str,
    *,
    opt_max_iter: int,
    md_steps: int,
    md_timestep_fs: float,
    md_temperature_kelvin: float,
    md_random_seed: int,
) -> str:
    if task == "sp":
        return "#sp(verbose=1)"
    if task == "opt":
        return (
            "#opt(method=lbfgs,"
            f"max_iter={opt_max_iter},verbose=0,log_final_paths=false)"
        )
    if task == "scan":
        return "#scan(method=lbfgs,mode=rigid)"
    if task == "md":
        return (
            "#md(ensemble=nvt,"
            f"steps={md_steps},"
            f"timestep={md_timestep_fs:.12g},"
            f"temperature={md_temperature_kelvin:.12g},"
            "traj_every=1,log_every=1,"
            f"random_seed={md_random_seed},"
            "remove_com_every=0,rst_every=100)"
        )
    raise ValueError(f"Unsupported task: {task!r}.")


def render_input(
    *,
    model: str,
    device: str,
    mol2_path: Path,
    task: str,
    openmm_platform: str,
    opt_max_iter: int,
    scan_atoms: tuple[int, int],
    scan_step_angstrom: float,
    scan_steps: int,
    md_steps: int = 4,
    md_timestep_fs: float = 0.05,
    md_temperature_kelvin: float = 298.15,
    md_random_seed: int = 20260725,
) -> str:
    lines = [
        f"#model={model}",
        _task_line(
            task,
            opt_max_iter=opt_max_iter,
            md_steps=md_steps,
            md_timestep_fs=md_timestep_fs,
            md_temperature_kelvin=md_temperature_kelvin,
            md_random_seed=md_random_seed,
        ),
        f"#device={device}",
        "#charge(source=mol2,label=am1bcc-frozen-manifest)",
        (
            "#solv(implicit=water,method=gb,model=obc2,nonpolar=ace,"
            f"platform={openmm_platform},experimental=true)"
        ),
        "",
        "0 1",
        f"MOL2 {mol2_path}",
    ]
    if task == "scan":
        lines.extend(
            [
                "",
                (
                    f"S {scan_atoms[0]} {scan_atoms[1]} "
                    f"{scan_step_angstrom:.12g} {scan_steps}"
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def read_xyz_frames(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    frames: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        try:
            atom_count = int(lines[cursor])
        except ValueError as exc:
            raise ValueError(f"Invalid XYZ atom count at line {cursor + 1}.") from exc
        end = cursor + atom_count + 2
        if end > len(lines):
            raise ValueError(f"Truncated XYZ frame at line {cursor + 1}.")
        comment = lines[cursor + 1]
        symbols: list[str] = []
        positions: list[list[float]] = []
        for line in lines[cursor + 2 : end]:
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Invalid XYZ coordinate line: {line!r}.")
            symbols.append(fields[0])
            positions.append([float(value) for value in fields[1:4]])
        frames.append(
            {
                "comment": comment,
                "symbols": symbols,
                "positions_angstrom": positions,
            }
        )
        cursor = end
    return frames


def parse_md_thermodynamics(path: Path) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(
                f"Invalid NVT thermodynamics row at line {line_number}: {line!r}."
            )
        values = [float(value) for value in fields]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"Non-finite NVT thermodynamics row at line {line_number}."
            )
        records.append(
            {
                "step": int(values[0]),
                "time_fs": values[1],
                "temperature_kelvin": values[2],
                "kinetic_energy_hartree": values[3],
                "potential_energy_hartree": values[4],
                "total_energy_hartree": values[5],
                "synchronized_temperature_kelvin": values[6],
                "synchronized_kinetic_energy_hartree": values[7],
                "synchronized_total_energy_hartree": values[8],
            }
        )
    return records


def parse_scan_frames(path: Path) -> list[dict[str, Any]]:
    parsed = []
    for frame in read_xyz_frames(path):
        match = SCAN_COMMENT.fullmatch(frame["comment"])
        if match is None:
            raise ValueError(f"Unrecognized MAPLE SCAN comment: {frame['comment']!r}.")
        coordinate_values = [
            float(value.strip()) for value in match.group(1).split(",")
        ]
        parsed.append(
            {
                **frame,
                "coordinate_values": coordinate_values,
                "energy_hartree": float(match.group(2)),
            }
        )
    return parsed


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _execute_task(
    *,
    model: str,
    task: str,
    device: str,
    mol2_path: Path,
    work_dir: Path,
    openmm_platform: str,
    opt_max_iter: int,
    scan_atoms: tuple[int, int],
    scan_step_angstrom: float,
    scan_steps: int,
    md_steps: int,
    md_timestep_fs: float,
    md_temperature_kelvin: float,
    md_random_seed: int,
    reference_positions: np.ndarray,
) -> dict[str, Any]:
    from maple.function.engine import engine
    from maple.function.timer import timer

    stem = f"{model}-{task}"
    input_path = work_dir / f"{stem}.inp"
    output_path = work_dir / f"{stem}.out"
    input_path.write_text(
        render_input(
            model=model,
            device=device,
            mol2_path=mol2_path,
            task=task,
            openmm_platform=openmm_platform,
            opt_max_iter=opt_max_iter,
            scan_atoms=scan_atoms,
            scan_step_angstrom=scan_step_angstrom,
            scan_steps=scan_steps,
            md_steps=md_steps,
            md_timestep_fs=md_timestep_fs,
            md_temperature_kelvin=md_temperature_kelvin,
            md_random_seed=md_random_seed,
        ),
        encoding="utf-8",
    )

    timer.reset()
    maple_engine = engine()
    maple_engine(str(input_path), str(output_path))
    results = dict(maple_engine.atoms.calc.results)
    solvation = dict(results.get("solvation", {}))
    missing_solvation = sorted(REQUIRED_SOLVATION_KEYS - set(solvation))
    if missing_solvation:
        raise ValueError(
            f"{model}/{task} omitted solvation fields: {', '.join(missing_solvation)}."
        )

    energy_hartree = float(results["energy"])
    gas_hartree = float(solvation["gas_energy_hartree"])
    solvent_hartree = float(solvation["delta_g_solv_hartree"])
    combined_hartree = float(solvation["combined_energy_hartree"])
    output_text = output_path.read_text(encoding="utf-8")
    audit_manifest = Path(str(output_path) + ".implicit") / "manifest.json"
    positions = np.asarray(maple_engine.atoms.get_positions(), dtype=np.float64)

    record: dict[str, Any] = {
        "task": task,
        "actual_maple_engine_dispatcher": True,
        "resolved_device": str(maple_engine.device),
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output": str(output_path),
        "audit_manifest": str(audit_manifest),
        "audit_manifest_sha256": (
            sha256_file(audit_manifest) if audit_manifest.is_file() else None
        ),
        "energy_hartree": energy_hartree,
        "gas_energy_hartree": gas_hartree,
        "solvent_energy_hartree": solvent_hartree,
        "combined_energy_hartree": combined_hartree,
        "energy_closure_abs_hartree": abs(
            combined_hartree - gas_hartree - solvent_hartree
        ),
        "solvation_components_hartree": _json_safe(solvation["components_hartree"]),
        "solvation_provenance": _json_safe(solvation["provenance"]),
        "final_position_rms_displacement_angstrom": float(
            np.sqrt(np.mean((positions - reference_positions) ** 2))
        ),
        "common_checks": {
            "finite_energy": all(
                math.isfinite(value)
                for value in (
                    energy_hartree,
                    gas_hartree,
                    solvent_hartree,
                    combined_hartree,
                )
            ),
            "energy_result_matches_composed_energy": (
                abs(energy_hartree - combined_hartree) <= 1.0e-10
            ),
            "energy_closure": (
                abs(combined_hartree - gas_hartree - solvent_hartree) <= 1.0e-10
            ),
            "structured_solvation_result": not missing_solvation,
            "audit_manifest": audit_manifest.is_file(),
            "no_runtime_error": "ERROR:" not in output_text,
        },
    }

    if task == "sp":
        forces = np.asarray(results.get("forces"), dtype=np.float64)
        required_output = (
            "Gas-phase MLIP energy:",
            "Solvation free-energy correction",
            "Combined E_MLIP(gas)+Delta G_solv:",
        )
        record["sp"] = {
            "force_component_count": int(forces.size),
            "force_max_abs_hartree_per_angstrom": float(np.max(np.abs(forces))),
            "output_contains_composition": all(
                marker in output_text for marker in required_output
            ),
        }
        record["task_checks"] = {
            "finite_combined_forces": (
                forces.shape == reference_positions.shape
                and bool(np.isfinite(forces).all())
            ),
            "output_contains_composition": record["sp"]["output_contains_composition"],
        }
    elif task == "opt":
        opt_path = output_path.with_name(f"{output_path.stem}_opt.xyz")
        trajectory_path = output_path.with_name(f"{output_path.stem}_opt_traj.xyz")
        trajectory_frames = read_xyz_frames(trajectory_path)
        record["opt"] = {
            "max_iter": opt_max_iter,
            "final_geometry": str(opt_path),
            "final_geometry_sha256": sha256_file(opt_path),
            "trajectory": str(trajectory_path),
            "trajectory_sha256": sha256_file(trajectory_path),
            "trajectory_frame_count": len(trajectory_frames),
        }
        record["task_checks"] = {
            "final_geometry_written": opt_path.is_file(),
            "trajectory_written": trajectory_path.is_file(),
            "expected_trajectory_frames": len(trajectory_frames) == opt_max_iter + 1,
            "geometry_changed": (
                record["final_position_rms_displacement_angstrom"] > 0.0
            ),
        }
    elif task == "scan":
        scan_path = output_path.with_name(f"{output_path.stem}_scan_final.xyz")
        scan_frames = parse_scan_frames(scan_path)
        left, right = (index - 1 for index in scan_atoms)
        observed_distances = [
            float(
                np.linalg.norm(
                    np.asarray(frame["positions_angstrom"][left])
                    - np.asarray(frame["positions_angstrom"][right])
                )
            )
            for frame in scan_frames
        ]
        requested_values = [
            float(frame["coordinate_values"][0]) for frame in scan_frames
        ]
        coordinate_errors = [
            abs(observed - requested)
            for observed, requested in zip(observed_distances, requested_values)
        ]
        step_errors = [
            abs((observed - observed_distances[0]) - point_index * scan_step_angstrom)
            for point_index, observed in enumerate(observed_distances)
        ]
        record["scan"] = {
            "mode": "rigid",
            "coordinate": "distance",
            "atoms_one_based": list(scan_atoms),
            "step_angstrom": scan_step_angstrom,
            "steps": scan_steps,
            "expected_point_count": scan_steps + 1,
            "point_count": len(scan_frames),
            "coordinate_values_angstrom": requested_values,
            "observed_distances_angstrom": observed_distances,
            "reported_coordinate_max_abs_error_angstrom": max(coordinate_errors),
            "step_sequence_max_abs_error_angstrom": max(step_errors),
            "energies_hartree": [
                float(frame["energy_hartree"]) for frame in scan_frames
            ],
            "trajectory": str(scan_path),
            "trajectory_sha256": sha256_file(scan_path),
        }
        record["task_checks"] = {
            "scan_trajectory_written": scan_path.is_file(),
            "expected_scan_point_count": len(scan_frames) == scan_steps + 1,
            "finite_scan_energies": all(
                math.isfinite(frame["energy_hartree"]) for frame in scan_frames
            ),
            "reported_coordinates_match_xyz": max(coordinate_errors) <= 5.0e-5,
            "requested_step_sequence_realized": max(step_errors) <= 1.0e-8,
            "dispatcher_completion_recorded": "Scan completed!" in output_text,
        }
    elif task == "md":
        thermo_path = output_path.with_name(f"{output_path.stem}_md_thermo.dat")
        trajectory_path = output_path.with_name(f"{output_path.stem}_md_traj.xyz")
        summary_path = output_path.with_name(f"{output_path.stem}_md_summary.txt")
        final_path = output_path.with_name(f"{output_path.stem}_final.xyz")
        checkpoint_path = output_path.with_name(f"{output_path.stem}_md.rst")
        thermodynamics = parse_md_thermodynamics(thermo_path)
        trajectory_frames = read_xyz_frames(trajectory_path)
        record["md"] = {
            "ensemble": "nvt",
            "step_count": len(thermodynamics),
            "requested_step_count": md_steps,
            "timestep_fs": md_timestep_fs,
            "temperature_kelvin": md_temperature_kelvin,
            "random_seed": md_random_seed,
            "finite_thermodynamics": bool(thermodynamics),
            "thermodynamics": str(thermo_path),
            "thermodynamics_sha256": sha256_file(thermo_path),
            "trajectory": str(trajectory_path),
            "trajectory_sha256": sha256_file(trajectory_path),
            "trajectory_frame_count": len(trajectory_frames),
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "final_geometry": str(final_path),
            "final_geometry_sha256": sha256_file(final_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "initial_temperature_kelvin": float(
                thermodynamics[0]["temperature_kelvin"]
            ),
            "final_temperature_kelvin": float(
                thermodynamics[-1]["temperature_kelvin"]
            ),
        }
        record["task_checks"] = {
            "expected_thermodynamics_rows": len(thermodynamics) == md_steps,
            "expected_trajectory_frames": len(trajectory_frames) == md_steps,
            "finite_thermodynamics": bool(thermodynamics),
            "summary_written": summary_path.is_file(),
            "final_geometry_written": final_path.is_file(),
            "checkpoint_written": checkpoint_path.is_file(),
            "dispatcher_completion_recorded": (
                "MD SIMULATION COMPLETED" in output_text
            ),
        }

    record["all_checks_pass"] = all(record["common_checks"].values()) and all(
        record["task_checks"].values()
    )
    del maple_engine
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return record


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from maple.function.read.filereader.mol2_reader import MOL2Reader

    mol2_path = Path(args.mol2).resolve()
    charge_manifest = Path(args.charge_manifest).resolve()
    for path in (mol2_path, charge_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    atoms = MOL2Reader(str(mol2_path), charge=0, mult=1, validate_charge=True)
    expected_charges, source_record = _load_am1bcc_record(
        charge_manifest,
        args.compound_id,
        atom_count=len(atoms),
    )
    charge_error = float(np.max(np.abs(atoms.get_initial_charges() - expected_charges)))
    if charge_error > 1.0e-10:
        raise ValueError(
            "Normalized MOL2 charges do not match the frozen AM1-BCC source record."
        )

    scan_atoms = tuple(args.scan_atoms)
    if any(index < 1 or index > len(atoms) for index in scan_atoms):
        raise ValueError("SCAN atom indices must address the supplied molecule.")
    reference_positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    models = []
    for model in args.model:
        tasks = {
            task: _execute_task(
                model=model,
                task=task,
                device=args.device,
                mol2_path=mol2_path,
                work_dir=work_dir,
                openmm_platform=args.openmm_platform,
                opt_max_iter=args.opt_max_iter,
                scan_atoms=scan_atoms,
                scan_step_angstrom=args.scan_step,
                scan_steps=args.scan_steps,
                md_steps=args.md_steps,
                md_timestep_fs=args.md_timestep,
                md_temperature_kelvin=args.md_temperature,
                md_random_seed=args.md_random_seed,
                reference_positions=reference_positions,
            )
            for task in ("sp", "opt", "scan", "md")
        }
        tasks["opt"]["opt"]["energy_change_from_sp_hartree"] = (
            tasks["opt"]["energy_hartree"] - tasks["sp"]["energy_hartree"]
        )
        tasks["opt"]["task_checks"]["combined_energy_not_raised"] = (
            tasks["opt"]["opt"]["energy_change_from_sp_hartree"] <= 1.0e-10
        )
        tasks["opt"]["all_checks_pass"] = all(
            tasks["opt"]["common_checks"].values()
        ) and all(tasks["opt"]["task_checks"].values())

        checkpoint = REPOSITORY_ROOT / "maple/function/calculator/model" / f"{model}.pt"
        models.append(
            {
                "model": model,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "tasks": tasks,
                "all_checks_pass": all(
                    record["all_checks_pass"] for record in tasks.values()
                ),
            }
        )

    solvent_spread = float(
        np.ptp([model["tasks"]["sp"]["solvent_energy_hartree"] for model in models])
    )
    device = torch.device(models[0]["tasks"]["sp"]["resolved_device"])
    result = {
        "schema_version": 1,
        "artifact_type": "route1-maple-engine-task-matrix-local-trace",
        "command_provenance": command_provenance(
            __file__,
            vars(args),
            repository_root=REPOSITORY_ROOT,
            environment_variables=(
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            ),
        ),
        "claim_scope": (
            "Actual MAPLE engine/dispatcher plumbing and short-task evidence for "
            "the named MLIP checkpoints on one molecule; not broad OPT/SCAN/MD "
            "stability, thermodynamic hydration accuracy, or confirmation evidence."
        ),
        "formula": FORMULA,
        "route": {
            "name": "Additive fixed-charge PB/GB implicit solvation",
            "role": "Baseline/Product Route",
            "retraining": False,
            "gas_phase_mm_energy": False,
            "hydration_label_residual": False,
        },
        "compound_id": args.compound_id,
        "molecule": args.molecule_name,
        "atom_count": len(atoms),
        "inputs": {
            "mol2": str(mol2_path),
            "mol2_sha256": sha256_file(mol2_path),
            "charge_manifest": str(charge_manifest),
            "charge_manifest_sha256": sha256_file(charge_manifest),
            "source_record": source_record,
        },
        "charge": {
            "method": "AM1-BCC",
            "lifecycle": "fixed",
            "transport": "normalized MOL2",
            "max_abs_difference_from_manifest_e": charge_error,
        },
        "solvent": {
            "polar": "OpenMM OBC-II",
            "nonpolar": "OpenMM ACE",
            "platform": args.openmm_platform,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "openmm": importlib.metadata.version("openmm"),
            "requested_device": args.device,
            "resolved_device": str(device),
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "model_count": len(models),
        "models": models,
        "cross_model": {
            "sp_solvent_energy_spread_hartree": solvent_spread,
            "same_fixed_geometry_charge_radius_nonpolar_provider": True,
            "passes": solvent_spread <= 1.0e-12,
        },
        "all_checks_pass": all(model["all_checks_pass"] for model in models)
        and solvent_spread <= 1.0e-12,
        "limitations": [
            "one 23-atom neutral development molecule",
            (
                f"{len(models)} named registered MLIP "
                f"checkpoint{'s' if len(models) != 1 else ''}"
            ),
            "two-step OPT and three-point rigid SCAN are task-plumbing smokes only",
            "four-step NVT is a force-path and dispatcher smoke, not equilibrated sampling",
            "fixed AM1-BCC charges are read once from a normalized MOL2",
            "no FreeSolv confirmation labels or residual model are used",
            "absolute paths make this artifact locally traceable, not checkout-portable",
        ],
    }
    return seal_artifact(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mol2", required=True)
    parser.add_argument("--charge-manifest", required=True)
    parser.add_argument("--compound-id", required=True)
    parser.add_argument("--molecule-name", required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--device", default="gpu0")
    parser.add_argument("--openmm-platform", default="Reference")
    parser.add_argument("--opt-max-iter", type=int, default=2)
    parser.add_argument("--scan-atoms", nargs=2, type=int, default=(8, 9))
    parser.add_argument("--scan-step", type=float, default=0.02)
    parser.add_argument("--scan-steps", type=int, default=2)
    parser.add_argument("--md-steps", type=int, default=4)
    parser.add_argument("--md-timestep", type=float, default=0.05)
    parser.add_argument("--md-temperature", type=float, default=298.15)
    parser.add_argument("--md-random-seed", type=int, default=20260725)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.model:
        args.model = ["maceoff23m", "aimnet2"]
    if (
        args.opt_max_iter < 1
        or args.scan_step <= 0
        or args.scan_steps < 1
        or args.md_steps < 1
        or args.md_timestep <= 0
        or args.md_temperature <= 0
    ):
        raise ValueError("OPT, SCAN, and MD controls must be positive.")
    result = run(args)
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_pass"]:
        raise SystemExit("Route 1 MAPLE task matrix failed.")


if __name__ == "__main__":
    main()
