#!/usr/bin/env python3
"""Run the prebuilt explicit-inner/implicit-outer Route 1 engineering smoke."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import platform
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

FORMULA = (
    "E_cluster,outer(R)=E_MLIP,gas(cluster;R)"
    "+G_polar,outer(R,q_fixed)+G_nonpolar,outer(R)"
)
TASKS = ("sp", "opt", "scan")


def _task_line(task: str, opt_max_iter: int) -> str:
    if task == "sp":
        return "#sp(verbose=1)"
    if task == "opt":
        return (
            "#opt(method=lbfgs,"
            f"max_iter={opt_max_iter},verbose=0,log_final_paths=false)"
        )
    if task == "scan":
        return "#scan(method=lbfgs,mode=rigid)"
    raise ValueError(task)


def _render_input(
    *,
    model: str,
    device: str,
    mol2: Path,
    task: str,
    platform_name: str,
    opt_max_iter: int,
    scan_atoms: tuple[int, int],
    scan_step: float,
    scan_steps: int,
) -> str:
    lines = [
        f"#model={model}",
        _task_line(task, opt_max_iter),
        f"#device={device}",
        "#charge(source=mol2,label=prebuilt-fixed-cluster)",
        (
            "#solv(implicit=water,inner=prebuilt,method=gb,model=obc2,"
            f"nonpolar=ace,platform={platform_name},experimental=true)"
        ),
        "",
        "0 1",
        f"MOL2 {mol2}",
    ]
    if task == "scan":
        lines.extend(
            [
                "",
                f"S {scan_atoms[0]} {scan_atoms[1]} {scan_step:g} {scan_steps}",
            ]
        )
    return "\n".join(lines) + "\n"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _task_artifact(
    *,
    model: str,
    task: str,
    args: argparse.Namespace,
    mol2: Path,
    work_dir: Path,
) -> dict[str, Any]:
    from maple.function.engine import engine
    from maple.function.timer import timer

    stem = f"{model}-{task}"
    input_path = work_dir / f"{stem}.inp"
    output_path = work_dir / f"{stem}.out"
    input_path.write_text(
        _render_input(
            model=model,
            device=args.device,
            mol2=mol2,
            task=task,
            platform_name=args.openmm_platform,
            opt_max_iter=args.opt_max_iter,
            scan_atoms=tuple(args.scan_atoms),
            scan_step=args.scan_step,
            scan_steps=args.scan_steps,
        ),
        encoding="utf-8",
    )

    timer.reset()
    maple_engine = engine()
    maple_engine(str(input_path), str(output_path))
    results = dict(maple_engine.atoms.calc.results)
    solvation = dict(results["solvation"])
    provenance = dict(solvation["provenance"])
    gas = float(solvation["gas_energy_hartree"])
    correction = float(solvation["cluster_continuum_correction_hartree"])
    combined = float(solvation["combined_energy_hartree"])
    output_text = output_path.read_text(encoding="utf-8")
    audit_path = Path(f"{output_path}.implicit") / "manifest.json"
    audit = load_json(audit_path)

    checks = {
        "finite_energy_decomposition": all(
            math.isfinite(value) for value in (gas, correction, combined)
        ),
        "energy_closure": abs(combined - gas - correction) <= 1.0e-10,
        "result_matches_combined": abs(float(results["energy"]) - combined)
        <= 1.0e-10,
        "cluster_correction_key_present": (
            "cluster_continuum_correction_hartree" in solvation
        ),
        "delta_g_solv_key_absent": "delta_g_solv_hartree" not in solvation,
        "quantity_is_configurational_potential": (
            provenance.get("thermodynamic_quantity")
            == "fixed-shell cluster-continuum configurational potential"
        ),
        "absolute_free_energy_claim_is_false": (
            provenance.get("absolute_solvation_free_energy_claim") is False
        ),
        "audit_component_count_is_two": (
            audit["route"].get("cluster_component_count") == 2
        ),
        "audit_terms_not_computed_present": len(
            audit["route"].get("terms_not_computed", [])
        )
        == 4,
        "no_runtime_error": "ERROR:" not in output_text,
    }
    task_files: dict[str, Any] = {}
    if task == "sp":
        forces = np.asarray(results["forces"], dtype=np.float64)
        atom_index = args.force_check_atom - 1
        axis = args.force_check_axis
        step = args.force_check_step

        def displaced_energy(sign: float) -> float:
            displaced = maple_engine.atoms.copy()
            positions = displaced.get_positions()
            positions[atom_index, axis] += sign * step
            displaced.set_positions(positions)
            displaced.calc = maple_engine.atoms.calc
            return float(displaced.get_potential_energy())

        energy_plus = displaced_energy(1.0)
        energy_minus = displaced_energy(-1.0)
        finite_difference_force = -(energy_plus - energy_minus) / (2.0 * step)
        analytical_force = float(forces[atom_index, axis])
        force_error = abs(analytical_force - finite_difference_force)
        checks.update(
            {
                "finite_combined_forces": forces.shape == (9, 3)
                and bool(np.isfinite(forces).all()),
                "combined_force_matches_finite_difference": (
                    force_error <= args.force_check_tolerance
                ),
                "output_uses_cluster_quantity": (
                    "Combined fixed-shell cluster-continuum configurational potential:"
                    in output_text
                ),
                "output_rejects_absolute_free_energy_claim": (
                    "not an absolute solvation free energy" in output_text
                    and "Delta G_solv" not in output_text
                ),
            }
        )
        task_files["combined_force_check"] = {
            "atom_index_one_based": args.force_check_atom,
            "axis_zero_based": axis,
            "centered_step_angstrom": step,
            "analytical_hartree_per_angstrom": analytical_force,
            "finite_difference_hartree_per_angstrom": finite_difference_force,
            "absolute_error_hartree_per_angstrom": force_error,
            "absolute_tolerance_hartree_per_angstrom": (
                args.force_check_tolerance
            ),
        }
    elif task == "opt":
        for suffix in ("_opt.xyz", "_opt_traj.xyz"):
            path = output_path.with_name(output_path.stem + suffix)
            task_files[path.name] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
        checks["opt_outputs_written"] = len(task_files) == 2
    else:
        path = output_path.with_name(output_path.stem + "_scan_final.xyz")
        task_files[path.name] = {"path": str(path), "sha256": sha256_file(path)}
        checks["scan_output_written"] = path.is_file()
        checks["scan_completed"] = "Scan completed!" in output_text

    artifact = {
        "task": task,
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "audit_manifest": str(audit_path),
        "audit_manifest_sha256": sha256_file(audit_path),
        "gas_energy_hartree": gas,
        "cluster_continuum_correction_hartree": correction,
        "combined_energy_hartree": combined,
        "components_hartree": _json_safe(solvation["components_hartree"]),
        "provenance": _json_safe(provenance),
        "task_files": task_files,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }
    del maple_engine
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return artifact


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from maple.function.read.filereader.mol2_reader import MOL2Reader

    protocol_path = Path(args.protocol).resolve()
    mol2 = Path(args.mol2).resolve()
    for path in (protocol_path, mol2):
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = load_json(protocol_path)
    if protocol["runtime_contract"]["absolute_solvation_free_energy_claim"] is not False:
        raise ValueError("Protocol must reject an absolute solvation-free-energy claim.")
    if protocol["runtime_contract"]["formula"] != FORMULA:
        raise ValueError("Protocol/runtime cluster formula does not match the runner.")
    smoke = protocol["smoke"]
    expected_input = REPOSITORY_ROOT / smoke["input"]["path"]
    if mol2 != expected_input.resolve() or sha256_file(mol2) != smoke["input"]["sha256"]:
        raise ValueError("MOL2 path or hash does not match the frozen smoke input.")
    if args.model != smoke["models"]:
        raise ValueError("Model list/order does not match the frozen smoke protocol.")
    provider = smoke["provider"]
    if (
        provider != {
            "method": "gb",
            "model": "obc2",
            "nonpolar": "ace",
            "platform": args.openmm_platform,
        }
    ):
        raise ValueError("Outer provider controls do not match the frozen protocol.")
    task_controls = smoke["task_controls"]
    if (
        args.opt_max_iter != task_controls["opt_max_iter"]
        or args.scan_atoms != task_controls["scan_atoms_one_based"]
        or args.scan_step != task_controls["scan_step_angstrom"]
        or args.scan_steps != task_controls["scan_steps"]
    ):
        raise ValueError("OPT/SCAN controls do not match the frozen protocol.")
    force_check = smoke["combined_force_check"]
    axis_index = {"x": 0, "y": 1, "z": 2}[force_check["axis"]]
    if (
        args.force_check_atom != force_check["atom_index_one_based"]
        or args.force_check_axis != axis_index
        or args.force_check_step != force_check["centered_step_angstrom"]
        or args.force_check_tolerance
        != force_check["absolute_tolerance_hartree_per_angstrom"]
    ):
        raise ValueError("Force-check controls do not match the frozen protocol.")
    atoms = MOL2Reader(
        str(mol2),
        charge=0,
        mult=1,
        validate_charge=True,
        allow_disconnected=True,
    )
    if atoms.info["mol2"]["component_count"] != 2:
        raise ValueError("The frozen smoke input must contain exactly two components.")
    if not 1 <= args.force_check_atom <= len(atoms):
        raise ValueError("force-check-atom must address the supplied cluster.")
    if args.force_check_step <= 0.0 or args.force_check_tolerance <= 0.0:
        raise ValueError("Force-check step and tolerance must be positive.")

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for model in args.model:
        tasks = {
            task: _task_artifact(
                model=model,
                task=task,
                args=args,
                mol2=mol2,
                work_dir=work_dir,
            )
            for task in TASKS
        }
        checkpoint = (
            REPOSITORY_ROOT / "maple" / "function" / "calculator" / "model" / f"{model}.pt"
        )
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

    sp_corrections = [
        record["tasks"]["sp"]["cluster_continuum_correction_hartree"]
        for record in models
    ]
    result = {
        "schema_version": 1,
        "artifact_type": "route1-prebuilt-inner-outer-engineering-smoke",
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
        "claim_scope": protocol["smoke"]["claim"],
        "formula": FORMULA,
        "thermodynamic_quantity": protocol["runtime_contract"][
            "thermodynamic_quantity"
        ],
        "absolute_solvation_free_energy_claim": False,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "input": {
            "mol2": str(mol2),
            "mol2_sha256": sha256_file(mol2),
            "atom_count": len(atoms),
            "component_count": atoms.info["mol2"]["component_count"],
            "component_charge_sums_e": atoms.info["mol2"][
                "component_charge_sums_e"
            ],
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "openmm": importlib.metadata.version("openmm"),
            "requested_device": args.device,
            "cuda": torch.version.cuda,
        },
        "models": models,
        "cross_model": {
            "sp_cluster_correction_spread_hartree": float(np.ptp(sp_corrections)),
            "same_fixed_geometry_fixed_charge_outer_provider": True,
            "passes": float(np.ptp(sp_corrections)) <= 1.0e-12,
        },
        "all_checks_pass": all(model["all_checks_pass"] for model in models)
        and float(np.ptp(sp_corrections)) <= 1.0e-12,
        "limitations": [
            "one illustrative methanol-plus-water fixed-shell cluster",
            "user-supplied illustrative fixed charges are not an accuracy-certified cluster charge model",
            "two-iteration OPT and three-point rigid SCAN are task-plumbing smokes",
            "no cluster formation, occupancy, standard-state, solvent-cluster reference, or ensemble term is computed",
            "no FreeSolv label, fitted selector, residual model, retraining, or gas-phase MM energy is used",
        ],
    }
    return seal_artifact(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--mol2", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--device", default="gpu0")
    parser.add_argument("--openmm-platform", default="Reference")
    parser.add_argument("--opt-max-iter", type=int, default=2)
    parser.add_argument("--scan-atoms", nargs=2, type=int, default=(2, 7))
    parser.add_argument("--scan-step", type=float, default=0.02)
    parser.add_argument("--scan-steps", type=int, default=2)
    parser.add_argument("--force-check-atom", type=int, default=8)
    parser.add_argument("--force-check-axis", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--force-check-step", type=float, default=0.003)
    parser.add_argument("--force-check-tolerance", type=float, default=5.0e-5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact = run(args)
    write_json_atomic(args.output, artifact)
    print(json.dumps({"output": args.output, "all_checks_pass": artifact["all_checks_pass"]}))


if __name__ == "__main__":
    main()
