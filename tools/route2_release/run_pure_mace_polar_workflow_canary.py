#!/usr/bin/env python3
"""Exercise exact pure-frozen MAPLE workflows without experimental labels.

Run from the repository with its pinned environment. Numerical criteria and
synthetic geometries live in the tracked prospective protocol. A failed or
unrequested task never becomes evidence for successful TS/FREQ operation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/route2/preregistrations/pure-mace-polar-workflows-v1.json"
TASKS = ("sp", "opt", "freq", "ts")


class WorkflowValidationError(RuntimeError):
    """A completed measurement did not satisfy the prospective canary."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict[str, str]:
    paths = sorted((ROOT / "maple").rglob("*.py"))
    paths += [PROTOCOL, Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): _sha(path) for path in paths}


def signed_ammonia_height(positions: np.ndarray) -> float:
    """Signed N distance to the ordered H-plane; invariant to rigid rotation."""
    points = np.asarray(positions, dtype=float)
    if points.shape != (4, 3) or not np.all(np.isfinite(points)):
        raise ValueError("ammonia requires four finite positions in N,H,H,H order")
    normal = np.cross(points[2] - points[1], points[3] - points[1])
    norm = float(np.linalg.norm(normal))
    if norm < 1.0e-12:
        raise ValueError("ammonia H-plane is degenerate")
    return float(np.dot(points[0] - points[1], normal / norm))


def _write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _atoms(protocol: dict, name: str) -> Atoms:
    fixture = protocol["fixtures"][name]
    return Atoms(
        fixture["symbols"],
        positions=fixture["positions_angstrom"],
        info={"charge": 0, "mult": 1},
    )


def _command(protocol: dict, task: str, output: Path, **options):
    from maple.function.read.command_control import CommandControl

    level = options.pop("level", "extratight")
    inner = ",".join(
        f"{key}={str(value).lower() if isinstance(value, bool) else value}"
        for key, value in options.items()
    )
    task_line = f"#{task}({inner})" if inner else f"#{task}"
    return CommandControl.from_settings(
        [
            "#model=macepolm",
            "#device=cpu",
            f"#level={level}",
            task_line,
            "#solv(method=smd,provider=pyddx,"
            f"profile={protocol['profile']},implicit={protocol['solvent']},"
            "response=frozen,experimental=true)",
        ],
        output_path=str(output),
    )


def _attach(protocol: dict, atoms: Atoms, output: Path):
    from maple.function.calculator.set_calculator import SetCalculator
    from maple.function.calculator.route2 import (
        is_pure_mace_polar_workflow_calculator,
    )

    command = _command(protocol, "sp", output)
    options = command.params["solv"]
    calc = SetCalculator(
        device="cpu",
        model=command.params["model"],
        output=str(output),
        atoms=atoms,
        implicit="smd",
        solvent=options["implicit"],
        solvation_options=options,
    ).set_calculator()
    if not is_pure_mace_polar_workflow_calculator(calc):
        raise WorkflowValidationError("factory did not select the pure total PES")
    atoms.calc = calc
    return calc


def _identity(protocol: dict, calc) -> dict:
    pes = calc.pes
    model = pes.model
    if (
        calc.profile_spec.name != protocol["profile"]
        or pes.scalar_contract_id != protocol["scalar_id"]
        or model.provenance.checkpoint_sha256 != protocol["checkpoint_sha256"]
    ):
        raise WorkflowValidationError(
            "resolved calculator identity differs from protocol"
        )
    return {
        "workflow_profile": calc.profile_spec.name,
        "scalar_contract_id": pes.scalar_contract_id,
        "model_provider_id": model.provider_id,
        "model_device": model.device,
        "model_dtype": model.dtype,
        "checkpoint_sha256": model.provenance.checkpoint_sha256,
        "model_configuration_sha256": model.configuration_sha256(),
        "model_provenance_sha256": model.provenance_sha256,
        "pes_configuration_sha256": pes.configuration_sha256(),
        "continuum_configuration_sha256": pes.continuum.configuration_sha256(),
        "solvent_term_configuration_sha256": pes.solvent_term.configuration_sha256(),
        "solvent": calc.solvent,
        "response": "frozen",
        "continuum": "ddPCM l15/n1202/tol1e-12/eta0.1/nproc1",
        "cds": "PySCF 2.13.1 SMD-CDS",
        "numerical_policy": pes._hessian_backend.policy_payload(),
    }


def _failure(stage: str, exc: Exception, atoms=None, calc=None) -> dict:
    result = {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    if atoms is not None:
        positions = atoms.get_positions()
        result["geometry_nonfinite"] = not bool(np.all(np.isfinite(positions)))
        if not result["geometry_nonfinite"]:
            result["positions_angstrom"] = positions.tolist()
    if calc is not None:
        result["last_ts_validation"] = getattr(calc, "last_ts_validation", None)
        result["last_hessian_evaluation_sha256"] = (
            None
            if calc.last_hessian_evaluation is None
            else calc.last_hessian_evaluation.evaluation_sha256
        )
    return result


def _dispatch(command, atoms: Atoms, output: Path) -> None:
    from maple.function.dispatcher.dispatcher import Dispatcher

    Dispatcher()(command, command.task, atoms, str(output))


def _optimization_metrics(atoms: Atoms) -> dict:
    metrics = {
        name: float(getattr(atoms, name))
        for name in (
            "max_f",
            "rms_f",
            "max_dp",
            "rms_dp",
            "f_max_th",
            "f_rms_th",
            "dp_max_th",
            "dp_rms_th",
        )
    }
    metrics["force_units"] = "Hartree/angstrom (existing dispatcher boundary)"
    metrics["optimizer_converged"] = all(
        metrics[value] <= metrics[limit]
        for value, limit in (
            ("max_f", "f_max_th"),
            ("rms_f", "f_rms_th"),
            ("max_dp", "dp_max_th"),
            ("rms_dp", "dp_rms_th"),
        )
    )
    return metrics


def _mode_record(atoms: Atoms, calc) -> tuple[dict, object]:
    from maple.solvation.derivatives.molecular_modes import (
        analyze_hessian_evaluation,
        hessian_numerical_diagnostics,
    )

    evaluated = calc.get_hessian_evaluation(atoms)
    modes = analyze_hessian_evaluation(atoms, evaluated)
    result = {
        "hessian_evaluation_sha256": evaluated.evaluation_sha256,
        "raw_hessian_eV_per_A2": evaluated.raw_hessian_eV_per_A2.tolist(),
        "error_estimates_eV_per_A2": evaluated.error_estimates_eV_per_A2.tolist(),
        "maximum_antisymmetry_eV_per_A2": evaluated.maximum_antisymmetry_eV_per_A2,
        "topology_guard_status": evaluated.topology_guard_status,
        "topology_observation_coverage": evaluated.topology_observation_coverage,
        "topology_changes": sum(evaluated.displaced_topology_changed),
        "maximum_energy_force_discrepancy_eV_per_A": max(
            evaluated.energy_force_discrepancies_eV_per_A, default=0.0
        ),
        "coarse_step_angstrom": evaluated.coarse_step_angstrom,
        "rigid_rank": modes.rigid_rank,
        "internal_dimension": modes.internal_dimension,
        "internal_eigenvalues_eV_per_A2_amu": modes.eigenvalues_eV_per_A2_amu.tolist(),
        "richardson_uncertainty_eV_per_A2_amu": modes.uncertainty_eV_per_A2_amu,
        "mode_statuses": list(modes.statuses),
        "resolved_negative_count": modes.resolved_negative_count,
        "resolved_positive_count": modes.resolved_positive_count,
        "uncertain_count": modes.uncertain_count,
        "is_resolved_minimum": modes.is_resolved_minimum,
        "is_resolved_index_one": modes.is_resolved_index_one,
        "numerical_diagnostics": hessian_numerical_diagnostics(evaluated),
    }
    return result, modes


def _relax(protocol: dict, atoms: Atoms, output: Path, specification: dict) -> dict:
    command = _command(
        protocol,
        "opt",
        output,
        **{
            key: specification[key]
            for key in ("method", "max_iter", "max_step", "level")
        },
    )
    initial_energy = float(atoms.get_potential_energy())
    _dispatch(command, atoms, output)
    result = _optimization_metrics(atoms)
    result.update(
        initial_energy_eV=initial_energy,
        final_energy_eV=float(atoms.get_potential_energy()),
        final_positions_angstrom=atoms.positions.tolist(),
        force_eV_per_A=atoms.get_forces().tolist(),
    )
    # The optimizer returns atoms on both normal termination and a step cap.
    result["optimizer_converged"] = bool(
        result["optimizer_converged"]
        and "LBFGS converged at iteration" in output.read_text()
    )
    return result


def _sp(protocol: dict, output_dir: Path) -> dict:
    atoms = _atoms(protocol, "water_sp")
    output = output_dir / "sp.out"
    calc = _attach(protocol, atoms, output)
    _dispatch(_command(protocol, "sp", output, verbose=1), atoms, output)
    energy, forces = float(atoms.get_potential_energy()), atoms.get_forces()
    baseline = protocol["sp_baseline"]
    energy_error = abs(energy - baseline["energy_eV"])
    force_error = float(np.max(np.abs(forces - baseline["forces_eV_per_A"])))
    return {
        "pass": bool(
            energy_error <= baseline["energy_replay_tolerance_eV"]
            and force_error <= baseline["force_replay_tolerance_eV_per_A"]
        ),
        "energy_eV": energy,
        "forces_eV_per_A": forces.tolist(),
        "baseline_energy_difference_eV": energy_error,
        "baseline_maximum_force_difference_eV_per_A": force_error,
        "workflow_kind": calc.workflow_kind,
        "identity": _identity(protocol, calc),
    }


def _water_workflows(protocol: dict, output_dir: Path, include_freq: bool) -> dict:
    atoms = _atoms(protocol, "water_opt")
    calc = None
    result = {}
    stage = "opt-factory"
    try:
        calc = _attach(protocol, atoms, output_dir / "water_factory.out")
        identity = _identity(protocol, calc)
        stage = "opt"
        optimized = _relax(
            protocol, atoms, output_dir / "opt.out", protocol["water_opt"]
        )
        result["opt"] = optimized
        optimized["identity"] = identity
        optimized["pass"] = bool(
            optimized["optimizer_converged"]
            and optimized["final_energy_eV"] < optimized["initial_energy_eV"]
        )
        _write(output_dir / "water-progress.json", result)
        if not optimized["pass"]:
            return result
        if include_freq:
            stage = "freq"
            output = output_dir / "freq.out"
            _dispatch(
                _command(
                    protocol, "freq", output, method="mw", treat_imag_as_real=False
                ),
                atoms,
                output,
            )
            record, _ = _mode_record(atoms, calc)
            record["identity"] = identity
            record["pass"] = bool(
                record["is_resolved_minimum"]
                and "Frequency analysis completed" in output.read_text()
            )
            result["freq"] = record
    except Exception as exc:
        task = "freq" if stage == "freq" else "opt"
        result[task] = {"pass": False, "failure": _failure(stage, exc, atoms, calc)}
    finally:
        _write(output_dir / "water-progress.json", result)
    return result


def _ts(protocol: dict, output_dir: Path) -> dict:
    atoms = _atoms(protocol, "ammonia_ts")
    calc = None
    current_atoms = atoms
    result = {"pass": False, "stage": "ts-factory"}
    try:
        calc = _attach(protocol, atoms, output_dir / "ts_factory.out")
        result["identity"] = _identity(protocol, calc)
        result["stage"] = "initial-index-check"
        initial, _ = _mode_record(atoms, calc)
        result["initial_modes"] = initial
        _write(output_dir / "ts-progress.json", result)
        if not initial["is_resolved_index_one"]:
            return result
        options = protocol["ts"]
        command = _command(
            protocol,
            "ts",
            output_dir / "ts.out",
            **{
                key: options[key]
                for key in (
                    "method",
                    "max_iter",
                    "recalc",
                    "hessian_update",
                    "trust_radius",
                    "level",
                )
            },
        )
        result["stage"] = "prfo"
        _dispatch(command, atoms, output_dir / "ts.out")
        result["optimization"] = _optimization_metrics(atoms)
        result["last_ts_validation"] = calc.last_ts_validation
        result["final_positions_angstrom"] = atoms.positions.tolist()
        result["ts_energy_eV"] = float(atoms.get_potential_energy())
        result["stage"] = "final-index-check"
        final, modes = _mode_record(atoms, calc)
        result["final_modes"] = final
        if not (
            result["optimization"]["optimizer_converged"]
            and result["last_ts_validation"]["workflow_success"]
            and final["is_resolved_index_one"]
        ):
            return result
        negative = modes.statuses.index("negative")
        direction = modes.modes_cartesian[negative].reshape(-1, 3)
        direction = direction / np.linalg.norm(direction)
        result["endpoints"] = []
        endpoint_spec = protocol["endpoint_checks"]
        for sign, name in ((1, "plus"), (-1, "minus")):
            result["stage"] = f"endpoint-{name}"
            endpoint = atoms.copy()
            endpoint.positions += sign * endpoint_spec["mode_step_angstrom"] * direction
            endpoint.calc = calc
            current_atoms = endpoint
            record = _relax(
                protocol, endpoint, output_dir / f"endpoint-{name}.out", endpoint_spec
            )
            record["signed_nitrogen_height_angstrom"] = signed_ammonia_height(
                endpoint.positions
            )
            record["energy_drop_from_ts_eV"] = (
                result["ts_energy_eV"] - record["final_energy_eV"]
            )
            result["endpoints"].append(record)
            _write(output_dir / "ts-progress.json", result)
        endpoints = result["endpoints"]
        result["pass"] = bool(
            all(
                item["optimizer_converged"]
                and item["energy_drop_from_ts_eV"]
                >= endpoint_spec["minimum_energy_drop_eV"]
                and abs(item["signed_nitrogen_height_angstrom"])
                >= endpoint_spec["minimum_abs_signed_height_angstrom"]
                for item in endpoints
            )
            and np.prod([item["signed_nitrogen_height_angstrom"] for item in endpoints])
            < 0
        )
        result["stage"] = "complete" if result["pass"] else "endpoint-validation"
    except Exception as exc:
        result["failure"] = _failure(result["stage"], exc, current_atoms, calc)
    finally:
        _write(output_dir / "ts-progress.json", result)
    return result


def run(args: argparse.Namespace) -> dict:
    import torch
    import maple

    if not Path(maple.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError(
            "wrong MAPLE checkout imported; set PYTHONPATH to this repository"
        )
    protocol = json.loads(PROTOCOL.read_text())
    if _sha(args.checkpoint) != protocol["checkpoint_sha256"]:
        raise ValueError(
            "checkpoint bytes do not match the prospective official identity"
        )
    # The existing official loader accepts this location; no alternative model.
    os.environ["ROUTE2_MACE_CHECKPOINT"] = str(args.checkpoint.resolve())
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    before = source_manifest()
    started = time.monotonic()
    result = {
        "schema": "maple-pure-frozen-workflow-canary-result-v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_tasks": list(args.tasks),
        "protocol_sha256": _sha(PROTOCOL),
        "source_files_sha256": before,
        "git_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
            ).strip()
        ),
        "runtime": {
            "python": sys.version,
            "packages": {
                name: metadata.version(name)
                for name in ("torch", "mace-torch", "pyddx", "pyscf", "ase")
            },
            "checkpoint_sha256": _sha(args.checkpoint),
            "device": "cpu",
            "threads": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
        },
        "claim_boundary": "Experimental workflow consistency on declared water/NH3 fixtures only; not new solvation accuracy, general reaction barriers, physical derivative certification, or release admission.",
        "confirmation_partition_opened": False,
        "scientific_release_admitted": False,
        "tasks": {},
    }
    try:
        if "sp" in args.tasks:
            result["tasks"]["sp"] = _sp(protocol, args.output_dir)
            _write(args.output_dir / "progress.json", result)
        if "opt" in args.tasks:
            result["tasks"].update(
                _water_workflows(protocol, args.output_dir, "freq" in args.tasks)
            )
            _write(args.output_dir / "progress.json", result)
        if "ts" in args.tasks:
            result["tasks"]["ts"] = _ts(protocol, args.output_dir)
    except Exception as exc:
        result["failure"] = _failure("task-dispatch", exc)
    result["source_unchanged"] = before == source_manifest()
    result["elapsed_seconds"] = time.monotonic() - started
    result["pass"] = bool(
        "failure" not in result
        and result["source_unchanged"]
        and set(result["tasks"]) == set(args.tasks)
        and all(item["pass"] for item in result["tasks"].values())
    )
    _write(args.output_dir / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="sp,opt,freq,ts")
    args = parser.parse_args()
    args.tasks = tuple(args.tasks.split(","))
    if (
        not args.tasks
        or any(task not in TASKS for task in args.tasks)
        or len(set(args.tasks)) != len(args.tasks)
    ):
        parser.error(
            "tasks must be a nonempty comma-separated subset of sp,opt,freq,ts"
        )
    if "freq" in args.tasks and "opt" not in args.tasks:
        parser.error("the water FREQ canary requires the OPT stage")
    result = run(args)
    print(
        json.dumps(
            {key: result[key] for key in ("requested_tasks", "pass", "elapsed_seconds")}
        ),
        flush=True,
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
