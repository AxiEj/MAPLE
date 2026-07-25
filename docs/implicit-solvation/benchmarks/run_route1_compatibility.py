#!/usr/bin/env python3
"""Exercise one fixed-charge Route 1 correction through multiple gas MLIPs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
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

FORMULA = "E_solution(R) = E_MLIP,gas(R) + G_polar(R,q_fixed) + G_nonpolar(R)"


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


def _calculate(calculator, atoms, properties: list[str]) -> dict[str, Any]:
    from ase.calculators.calculator import all_changes

    calculator.calculate(atoms, properties=properties, system_changes=all_changes)
    return dict(calculator.results)


def _combined_energy(calculator, atoms) -> float:
    return float(_calculate(calculator, atoms, ["energy"])["energy"])


def _run_model(
    *,
    model: str,
    device,
    atoms,
    output: Path,
    openmm_platform: str,
    fd_step_angstrom: float,
    opt_steps: int,
    scan_displacement_angstrom: float,
) -> dict[str, Any]:
    from ase.optimize import BFGS

    from maple.function.calculator.set_calculator import SetCalculator

    gas_output = output.with_name(f"{output.stem}-gas{output.suffix}")
    gas_calculator = SetCalculator(
        device,
        model,
        str(gas_output),
        atoms=atoms,
    ).set_calculator()
    combined_calculator = SetCalculator(
        device,
        model,
        str(output),
        atoms=atoms,
        implicit="gb",
        solvent="water",
        charge_options={"source": "mol2"},
        solvation_options={
            "method": "gb",
            "model": "obc2",
            "nonpolar": "ace",
            "platform": openmm_platform,
            "experimental": True,
        },
    ).set_calculator()
    correction = combined_calculator.solvent_correction

    gas = _calculate(gas_calculator, atoms, ["energy", "forces"])
    gas_energy = float(gas["energy"])
    gas_forces = np.asarray(gas["forces"], dtype=np.float64)

    combined = _calculate(combined_calculator, atoms, ["energy", "forces"])
    combined_energy = float(combined["energy"])
    combined_forces = np.asarray(combined["forces"], dtype=np.float64)
    solvent = correction.evaluate(
        atoms,
        need_forces=True,
        calculator=combined_calculator,
    )
    solvent_forces = np.asarray(
        solvent.forces_hartree_per_angstrom,
        dtype=np.float64,
    )

    finite_difference_forces = np.zeros_like(combined_forces)
    for atom_index in range(len(atoms)):
        for axis in range(3):
            plus = atoms.copy()
            minus = atoms.copy()
            plus.positions[atom_index, axis] += fd_step_angstrom
            minus.positions[atom_index, axis] -= fd_step_angstrom
            finite_difference_forces[atom_index, axis] = -(
                _combined_energy(combined_calculator, plus)
                - _combined_energy(combined_calculator, minus)
            ) / (2.0 * fd_step_angstrom)
    force_fd_errors = np.abs(combined_forces - finite_difference_forces)
    max_error_flat_index = int(np.argmax(force_fd_errors))
    max_error_atom, max_error_axis = np.unravel_index(
        max_error_flat_index,
        force_fd_errors.shape,
    )

    opt_atoms = atoms.copy()
    opt_atoms.calc = combined_calculator
    opt_initial = float(opt_atoms.get_potential_energy())
    optimizer = BFGS(opt_atoms, logfile=None, maxstep=0.01)
    optimizer.run(fmax=0.0, steps=opt_steps)
    opt_final = float(opt_atoms.get_potential_energy())

    displaced_geometry_energies = []
    displacement_atom_index = 0
    displacement_axis = 0
    for displacement in (-scan_displacement_angstrom, 0.0, scan_displacement_angstrom):
        displaced_atoms = atoms.copy()
        displaced_atoms.positions[
            displacement_atom_index,
            displacement_axis,
        ] += displacement
        displaced_geometry_energies.append(
            {
                "displacement_angstrom": displacement,
                "energy_hartree": _combined_energy(
                    combined_calculator,
                    displaced_atoms,
                ),
            }
        )

    checkpoint = REPOSITORY_ROOT / "maple/function/calculator/model" / f"{model}.pt"
    try:
        model_dtype = str(next(combined_calculator.model.parameters()).dtype)
    except (AttributeError, StopIteration):
        model_dtype = "unknown"
    return {
        "model": model,
        "model_dtype": model_dtype,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "sp": {
            "gas_energy_hartree": gas_energy,
            "solvent_energy_hartree": float(solvent.energy_hartree),
            "combined_energy_hartree": combined_energy,
            "energy_closure_abs_hartree": abs(
                combined_energy - gas_energy - solvent.energy_hartree
            ),
            "force_closure_max_abs_hartree_per_angstrom": float(
                np.max(np.abs(combined_forces - gas_forces - solvent_forces))
            ),
            "combined_potential_force_fd_all_coordinates": {
                "component_count": int(combined_forces.size),
                "step_angstrom": fd_step_angstrom,
                "max_error_atom_index": int(max_error_atom),
                "max_error_axis": int(max_error_axis),
                "max_absolute_error_hartree_per_angstrom": float(
                    np.max(force_fd_errors)
                ),
                "rms_error_hartree_per_angstrom": float(
                    np.sqrt(np.mean(force_fd_errors**2))
                ),
                "absolute_errors_hartree_per_angstrom": force_fd_errors.tolist(),
            },
        },
        "opt_smoke": {
            "steps_requested": opt_steps,
            "steps_run": optimizer.nsteps,
            "initial_energy_hartree": opt_initial,
            "final_energy_hartree": opt_final,
            "energy_change_hartree": opt_final - opt_initial,
        },
        "displaced_geometry_energy_smoke": {
            "atom_index": displacement_atom_index,
            "axis": displacement_axis,
            "points": displaced_geometry_energies,
        },
        "solvation_provenance": dict(solvent.provenance),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from maple.function.read.filereader.mol2_reader import MOL2Reader

    mol2_path = Path(args.mol2).resolve()
    manifest_path = Path(args.charge_manifest).resolve()
    for path in (mol2_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    atoms = MOL2Reader(str(mol2_path), charge=0, mult=1, validate_charge=True)
    expected_charges, source_record = _load_am1bcc_record(
        manifest_path,
        args.compound_id,
        atom_count=len(atoms),
    )
    charge_error = float(np.max(np.abs(atoms.get_initial_charges() - expected_charges)))
    if charge_error > 1.0e-10:
        raise ValueError(
            "Normalized MOL2 charges do not match the frozen AM1-BCC source record."
        )

    device = torch.device(args.device)
    records = [
        _run_model(
            model=model,
            device=device,
            atoms=atoms.copy(),
            output=work_dir / f"{model}.log",
            openmm_platform=args.openmm_platform,
            fd_step_angstrom=args.fd_step,
            opt_steps=args.opt_steps,
            scan_displacement_angstrom=args.scan_displacement,
        )
        for model in args.model
    ]
    solvent_energies = [record["sp"]["solvent_energy_hartree"] for record in records]
    acceptance = {
        "energy_closure_tolerance_hartree": 1.0e-10,
        # The gas model is evaluated separately for the gas and combined records.
        # This bound admits float32 repeat-evaluation noise while remaining far
        # below the force finite-difference gate.
        "force_closure_tolerance_hartree_per_angstrom": 1.0e-6,
        "force_fd_tolerance_hartree_per_angstrom": 1.0e-4,
        "opt_energy_tolerance_hartree": 1.0e-10,
    }
    for record in records:
        sp = record["sp"]
        record["passes"] = {
            "energy_closure": (
                sp["energy_closure_abs_hartree"]
                <= acceptance["energy_closure_tolerance_hartree"]
            ),
            "force_closure": (
                sp["force_closure_max_abs_hartree_per_angstrom"]
                <= acceptance["force_closure_tolerance_hartree_per_angstrom"]
            ),
            "combined_potential_force_fd_all_coordinates": (
                sp["combined_potential_force_fd_all_coordinates"][
                    "max_absolute_error_hartree_per_angstrom"
                ]
                <= acceptance["force_fd_tolerance_hartree_per_angstrom"]
            ),
            "opt_smoke": (
                record["opt_smoke"]["energy_change_hartree"]
                <= acceptance["opt_energy_tolerance_hartree"]
            ),
            "displaced_geometry_energy_smoke": all(
                np.isfinite(point["energy_hartree"])
                for point in record["displaced_geometry_energy_smoke"]["points"]
            ),
        }

    result = {
        "schema_version": 1,
        "artifact_type": "route1-multi-mlip-compatibility-local-trace",
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
            "Interface, additive-composition, derivative, and short task-smoke evidence "
            "for the named MLIP checkpoints on one molecule; not broad chemical accuracy "
            "or OPT/SCAN stability certification."
        ),
        "formula": FORMULA,
        "prohibited_terms": {
            "gas_phase_mm_energy": False,
            "retraining": False,
            "hydration_label_residual": False,
        },
        "compound_id": args.compound_id,
        "molecule": args.molecule_name,
        "atom_count": len(atoms),
        "inputs": {
            "mol2": str(mol2_path),
            "mol2_sha256": sha256_file(mol2_path),
            "charge_manifest": str(manifest_path),
            "charge_manifest_sha256": sha256_file(manifest_path),
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
            "device": str(device),
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "acceptance": acceptance,
        "models": records,
        "cross_model": {
            "solvent_energy_spread_hartree": float(np.ptp(solvent_energies)),
            "same_fixed_geometry_charge_radius_nonpolar_provider": True,
        },
        "all_checks_pass": all(all(record["passes"].values()) for record in records),
        "limitations": [
            "one 23-atom neutral development molecule",
            "two named registered MLIP checkpoints",
            "two-step BFGS and three manually displaced geometries are smoke tests only",
            "the displaced-geometry points do not execute MAPLE's SCAN task dispatcher",
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--openmm-platform", default="Reference")
    parser.add_argument("--fd-step", type=float, default=0.003)
    parser.add_argument("--opt-steps", type=int, default=2)
    parser.add_argument("--scan-displacement", type=float, default=0.03)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.model:
        args.model = ["maceoff23m", "aimnet2"]
    if args.fd_step <= 0 or args.opt_steps < 1 or args.scan_displacement <= 0:
        raise ValueError("Finite-difference, OPT, and SCAN controls must be positive.")
    result = run(args)
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_pass"]:
        raise SystemExit("Route 1 compatibility gate failed.")


if __name__ == "__main__":
    main()
