#!/usr/bin/env python3
"""Run a bounded AIMNet2 geometry--charge--ddPCM MNSol pilot.

The output contains row-level user-supplied MNSol data and therefore must stay
below ``.omx``.  The runner first relaxes the gas geometry with AIMNet2, then
relaxes the same molecule on the energy-consistent

``E_AIMNet2(R) + U_ddPCM(R,q(R)) + G_SMD-CDS(R)``

surface.  This isolates geometry-mediated charge response from the
fixed-geometry baseline without pretending AIMNet2 accepts a reaction field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, BENCHMARK_DIR):
    value = str(search_path)
    if value not in sys.path:
        sys.path.insert(0, value)

from ase import Atoms
from ase.optimize import LBFGS
from ase.units import Hartree
import numpy as np
import torch

from benchmark_core import sha256_file, write_json_atomic
from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
from mnsol_pilot import validate_frozen_mnsol_pilot_selection
from maple.function.calculator.aimnet._aimnet2_calculator import (
    AIMNet2Calculator,
)
from maple.function.calculator.extra_correction.implicit.aimnet2_geometry_coupling import (
    AIMNet2GeometryCoupledObjective,
    AIMNet2GeometryCoupledState,
    _make_aimnet2_geometry_coupled_research_ase_calculator,
    _require_research_optimization_convergence,
)
from maple.function.calculator.extra_correction.implicit.ddpcm_smd import (
    DDPCM_ETA,
    DDPCM_LMAX,
    DDPCM_N_LEBEDEV,
    DDPCM_SOLVER_TOLERANCE,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    HARTREE_TO_KCAL_MOL,
)


ARTIFACT = "route2-aimnet2-geometry-coupled-pilot-v1"
EV_TO_KCAL_MOL = HARTREE_TO_KCAL_MOL / Hartree


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        required=True,
        help="Zero-based indices in the frozen ten-record pilot.",
    )
    parser.add_argument("--gas-fmax", type=float, default=0.02)
    parser.add_argument("--gas-max-steps", type=int, default=100)
    parser.add_argument("--solution-fmax", type=float, default=0.05)
    parser.add_argument("--solution-max-steps", type=int, default=30)
    parser.add_argument("--max-step", type=float, default=0.04)
    parser.add_argument("--lmax", type=int, default=DDPCM_LMAX)
    parser.add_argument("--n-lebedev", type=int, default=DDPCM_N_LEBEDEV)
    return parser


def _require_private_output(path: Path) -> Path:
    resolved = path.resolve()
    private_root = (REPO_ROOT / ".omx").resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError as exc:
        raise ValueError(
            "Geometry-coupled row-level output must remain below .omx."
        ) from exc
    return resolved


def _max_force(forces: np.ndarray) -> float:
    values = np.asarray(forces, dtype=float)
    return float(np.max(np.linalg.norm(values, axis=1)))


def _aligned_geometry_metrics(
    reference_angstrom: np.ndarray,
    candidate_angstrom: np.ndarray,
) -> dict[str, float]:
    reference = np.asarray(reference_angstrom, dtype=float)
    candidate = np.asarray(candidate_angstrom, dtype=float)
    reference_centered = reference - np.mean(reference, axis=0)
    candidate_centered = candidate - np.mean(candidate, axis=0)
    covariance = candidate_centered.T @ reference_centered
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    aligned = candidate_centered @ rotation
    displacement = aligned - reference_centered
    norms = np.linalg.norm(displacement, axis=1)
    return {
        "aligned_rmsd_angstrom": float(
            np.sqrt(np.mean(np.sum(displacement**2, axis=1)))
        ),
        "aligned_maximum_atom_displacement_angstrom": float(np.max(norms)),
    }


def _state_components(state: AIMNet2GeometryCoupledState) -> dict[str, float]:
    return {
        "solute_energy_ev": state.solute_energy_ev,
        "polarization_energy_kcal_mol": (
            state.polarization_energy_kcal_mol
        ),
        "smd_cds_energy_kcal_mol": state.cds_energy_kcal_mol,
        "solution_energy_ev": state.solution_energy_ev,
        "maximum_total_gradient_ev_per_angstrom": _max_force(
            state.total_gradient_ev_per_angstrom
        ),
        "maximum_intrinsic_gradient_ev_per_angstrom": _max_force(
            state.intrinsic_gradient_ev_per_angstrom
        ),
        "maximum_continuum_fixed_source_gradient_ev_per_angstrom": (
            _max_force(
                state.continuum_fixed_source_gradient_ev_per_angstrom
            )
        ),
        "maximum_charge_response_gradient_ev_per_angstrom": _max_force(
            state.charge_response_gradient_ev_per_angstrom
        ),
        "maximum_cds_gradient_ev_per_angstrom": _max_force(
            state.cds_gradient_ev_per_angstrom
        ),
        "half_coupling_identity_error_ev": (
            state.half_coupling_identity_error_ev
        ),
    }


def _run_optimizer(
    atoms,
    *,
    logfile: Path,
    fmax: float,
    max_steps: int,
    max_step: float,
) -> tuple[LBFGS, float]:
    started = time.perf_counter()
    optimizer = LBFGS(
        atoms,
        logfile=str(logfile),
        maxstep=max_step,
    )
    optimizer.run(fmax=fmax, steps=max_steps)
    return optimizer, time.perf_counter() - started


def _git_state() -> dict[str, object]:
    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"head": head, "dirty": bool(status.strip())}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = _require_private_output(args.private_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if (
        args.gas_fmax <= 0.0
        or args.solution_fmax <= 0.0
        or args.gas_max_steps <= 0
        or args.solution_max_steps <= 0
        or args.max_step <= 0.0
    ):
        raise ValueError("Optimization thresholds and step limits must be positive.")

    protocol = load_mnsol_protocol(args.protocol)
    dataset = load_mnsol_v2012(args.source, protocol)
    selection_manifest = json.loads(
        args.selection.read_text(encoding="utf-8")
    )
    selection = validate_frozen_mnsol_pilot_selection(
        selection_manifest,
        dataset,
        protocol,
    )
    indices = tuple(dict.fromkeys(args.indices))
    if not indices or any(
        index < 0 or index >= len(selection) for index in indices
    ):
        raise ValueError("Pilot indices must lie within the frozen selection.")

    torch.set_num_threads(1)
    calculator = AIMNet2Calculator(
        torch.device("cpu"),
        model="aimnet2",
        model_path=str(checkpoint),
    )
    records: list[dict[str, object]] = []
    run_started = time.perf_counter()
    run_dir = output.parent / output.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    for index in indices:
        selected = selection[index]
        item = selected.eligible_record
        geometry = item.geometry
        record = item.record
        original = Atoms(
            numbers=geometry.atomic_numbers,
            positions=geometry.coordinates_angstrom,
            info={"charge": 0, "mult": 1},
        )
        record_dir = run_dir / f"selection-{index:02d}"
        record_dir.mkdir(parents=True, exist_ok=True)

        original_objective = AIMNet2GeometryCoupledObjective(
            original,
            calculator,
            solvent=selected.canonical_solvent,
            lmax=args.lmax,
            n_lebedev=args.n_lebedev,
        )
        original_started = time.perf_counter()
        original_state = original_objective.evaluate(original)
        original_seconds = time.perf_counter() - original_started
        original_fixed = (
            original_state.polarization_energy_kcal_mol
            + original_state.cds_energy_kcal_mol
        )

        gas = original.copy()
        gas.calc = calculator
        gas_optimizer, gas_seconds = _run_optimizer(
            gas,
            logfile=record_dir / "gas-lbfgs.log",
            fmax=args.gas_fmax,
            max_steps=args.gas_max_steps,
            max_step=args.max_step,
        )
        gas_forces = gas.get_forces()
        gas_maximum_force = _require_research_optimization_convergence(
            stage="gas",
            forces_ev_per_angstrom=gas_forces,
            target_fmax_ev_per_angstrom=args.gas_fmax,
            steps=gas_optimizer.nsteps,
            maximum_steps=args.gas_max_steps,
        )
        gas_charge_state = calculator.charge_state(gas)
        gas_reference_energy_ev = gas_charge_state.energy_ev

        solution_objective = AIMNet2GeometryCoupledObjective(
            gas,
            calculator,
            solvent=selected.canonical_solvent,
            lmax=args.lmax,
            n_lebedev=args.n_lebedev,
        )
        gas_geometry_state = solution_objective.evaluate(gas)
        gas_geometry_fixed = (
            gas_geometry_state.polarization_energy_kcal_mol
            + gas_geometry_state.cds_energy_kcal_mol
        )

        solution = gas.copy()
        solution.calc = _make_aimnet2_geometry_coupled_research_ase_calculator(
            solution_objective
        )
        history: list[dict[str, object]] = []

        def record_history() -> None:
            forces = solution.get_forces()
            state = solution.calc.last_state
            if state is None:
                raise RuntimeError(
                    "Geometry-coupled ASE calculator did not retain its state."
                )
            history.append(
                {
                    "step": len(history),
                    "solution_energy_ev": state.solution_energy_ev,
                    "maximum_force_ev_per_angstrom": _max_force(forces),
                    "charges_e": state.charges_e.tolist(),
                }
            )

        record_history()
        solution_started = time.perf_counter()
        solution_optimizer = LBFGS(
            solution,
            logfile=str(record_dir / "solution-lbfgs.log"),
            maxstep=args.max_step,
        )
        solution_optimizer.attach(record_history, interval=1)
        solution_optimizer.run(
            fmax=args.solution_fmax,
            steps=args.solution_max_steps,
        )
        solution_seconds = time.perf_counter() - solution_started
        solution_forces = solution.get_forces()
        solution_maximum_force = _require_research_optimization_convergence(
            stage="solution",
            forces_ev_per_angstrom=solution_forces,
            target_fmax_ev_per_angstrom=args.solution_fmax,
            steps=solution_optimizer.nsteps,
            maximum_steps=args.solution_max_steps,
        )
        final_state = solution.calc.last_state
        if final_state is None:
            raise RuntimeError("Missing final geometry-coupled state.")

        relaxed_delta_g = final_state.solvation_energy_kcal_mol(
            gas_reference_energy_ev=gas_reference_energy_ev
        )
        charge_step_residual = (
            float(
                np.max(
                    np.abs(
                        np.asarray(history[-1]["charges_e"])
                        - np.asarray(history[-2]["charges_e"])
                    )
                )
            )
            if len(history) >= 2
            else 0.0
        )
        methods = {
            "original_geometry_fixed_charge": {
                "total_solvation_kcal_mol": original_fixed,
                "signed_error_kcal_mol": (
                    original_fixed - record.delta_g_kcal_mol
                ),
                "absolute_error_kcal_mol": abs(
                    original_fixed - record.delta_g_kcal_mol
                ),
            },
            "aimnet2_gas_relaxed_fixed_charge": {
                "total_solvation_kcal_mol": gas_geometry_fixed,
                "signed_error_kcal_mol": (
                    gas_geometry_fixed - record.delta_g_kcal_mol
                ),
                "absolute_error_kcal_mol": abs(
                    gas_geometry_fixed - record.delta_g_kcal_mol
                ),
            },
            "aimnet2_geometry_coupled": {
                "solute_relaxation_kcal_mol": (
                    (final_state.solute_energy_ev - gas_reference_energy_ev)
                    * EV_TO_KCAL_MOL
                ),
                "polarization_energy_kcal_mol": (
                    final_state.polarization_energy_kcal_mol
                ),
                "smd_cds_energy_kcal_mol": (
                    final_state.cds_energy_kcal_mol
                ),
                "total_solvation_kcal_mol": relaxed_delta_g,
                "signed_error_kcal_mol": (
                    relaxed_delta_g - record.delta_g_kcal_mol
                ),
                "absolute_error_kcal_mol": abs(
                    relaxed_delta_g - record.delta_g_kcal_mol
                ),
            },
        }
        records.append(
            {
                "selection_index": index,
                "opaque_record_id": selected.opaque_record_id,
                "canonical_solvent": selected.canonical_solvent,
                "solute_name": record.solute_name,
                "formula": record.formula,
                "atom_count": len(original),
                "partition": item.partition,
                "experimental_delta_g_kcal_mol": record.delta_g_kcal_mol,
                "methods": methods,
                "gas_optimization": {
                    "steps": gas_optimizer.nsteps,
                    "converged": True,
                    "maximum_force_ev_per_angstrom": gas_maximum_force,
                    "wall_seconds": gas_seconds,
                    **_aligned_geometry_metrics(
                        original.positions,
                        gas.positions,
                    ),
                },
                "solution_optimization": {
                    "steps": solution_optimizer.nsteps,
                    "converged": True,
                    "maximum_force_ev_per_angstrom": solution_maximum_force,
                    "last_step_maximum_charge_change_e": (
                        charge_step_residual
                    ),
                    "wall_seconds": solution_seconds,
                    **_aligned_geometry_metrics(
                        gas.positions,
                        solution.positions,
                    ),
                },
                "original_state": _state_components(original_state),
                "gas_geometry_state": _state_components(
                    gas_geometry_state
                ),
                "solution_state": _state_components(final_state),
                "charges": {
                    "gas_e": gas_charge_state.charges_e.tolist(),
                    "solution_e": final_state.charges_e.tolist(),
                    "maximum_absolute_change_e": float(
                        np.max(
                            np.abs(
                                final_state.charges_e
                                - gas_charge_state.charges_e
                            )
                        )
                    ),
                },
                "coordinates_angstrom": {
                    "original": original.positions.tolist(),
                    "gas_relaxed": gas.positions.tolist(),
                    "solution_relaxed": solution.positions.tolist(),
                },
                "timing_seconds": {
                    "original_fixed_state": original_seconds,
                    "gas_optimization": gas_seconds,
                    "solution_optimization": solution_seconds,
                },
                "history": history,
            }
        )

    artifact = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "visibility": "private-user-supplied-mnsol-row-level",
        "do_not_commit": True,
        "git": _git_state(),
        "checkpoint": {
            "filename": checkpoint.name,
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "scientific_identity": {
            "gas_reference": "AIMNet2-relaxed gas minimum",
            "solution_scalar": (
                "E_AIMNet2(R) + U_ddPCM(R,q_AIMNet2(R)) + G_SMD-CDS(R)"
            ),
            "charge_response": "geometry-mediated R->q(R)",
            "fixed_geometry_electronic_mutual_polarization": False,
            "continuum": "pyddx ddPCM",
            "nonpolar": "PySCF 2.13.1 SMD-CDS",
            "strict_original_smd_equivalence": False,
        },
        "claim_boundary": (
            "Post-hoc bounded mechanism pilot. Geometry optimization updates "
            "AIMNet2 charges and ddPCM at every step with the complete "
            "first-derivative chain rule. It does not establish electronic "
            "mutual polarization, conformer/free-energy completeness, "
            "population accuracy, or a production MAPLE solution-phase PES."
        ),
        "selection_indices": list(indices),
        "optimization_parameters": {
            "gas_fmax_ev_per_angstrom": args.gas_fmax,
            "gas_max_steps": args.gas_max_steps,
            "solution_fmax_ev_per_angstrom": args.solution_fmax,
            "solution_max_steps": args.solution_max_steps,
            "max_step_angstrom": args.max_step,
        },
        "continuum_parameters": {
            "lmax": args.lmax,
            "n_lebedev": args.n_lebedev,
            "solver_tolerance": DDPCM_SOLVER_TOLERANCE,
            "eta": DDPCM_ETA,
        },
        "records": records,
        "total_wall_seconds": time.perf_counter() - run_started,
    }
    write_json_atomic(output, artifact)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
