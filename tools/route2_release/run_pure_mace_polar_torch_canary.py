#!/usr/bin/env python3
"""Source-bound old/new analytic-Hessian comparison on small synthetic fixtures.

This is an implementation audit, not a solvation-accuracy benchmark. Finite
differences occur here only, using the independent legacy analytic force and
checking the new backend's discrete topology at every displaced geometry.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIMITS = {"energy_eV": 1e-7, "force_eV_per_A": 1e-6, "hessian_eV_per_A2": 1e-4}
AUDIT_STEPS_ANGSTROM = (2e-5, 1e-5, 5e-6)
CASES = {
    "water-water": ("H2O", "water"),
    "methane-water": ("CH4", "water"),
    "methane-hexane": ("CH4", "hexane"),
}


def _source_manifest():
    paths = sorted((ROOT / "maple").rglob("*.py")) + [Path(__file__).resolve()]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_progress(output_dir, result, stage):
    _write(
        output_dir / "progress.json",
        {
            **result,
            "stage": stage,
            "pass": False,
            "execution_complete": False,
        },
    )


def _topology(pes, atoms):
    import torch

    from maple.solvation.experimental.cuda_execution import cuda_model_execution
    from contextlib import nullcontext

    context = nullcontext() if pes.device == "cpu" else cuda_model_execution()
    with context:
        positions = torch.tensor(
            atoms.positions, dtype=torch.float64, device=pes.device
        )
        zero = torch.zeros((len(atoms), 8), dtype=torch.float64, device=pes.device)
        ddpcm = pes.continuum.diagnostics(positions, zero)["topology"][
            "topology_sha256"
        ]
        cds = pes.solvent_term.evaluate_torch(
            positions
        ).diagnostics.dareal.topology_sha256
    return {
        "ddpcm": ddpcm,
        "cds": cds,
        "model": pes.model.topology_diagnostics(atoms)["pair_mask_order_sha256"],
    }


def _run_case(args, result):
    import torch
    from ase.build import molecule

    from maple.function.route2_smd_profiles import (
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE,
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2_PROFILE,
        route2_smd_profile_spec,
    )
    from maple.function.calculator.route2 import (
        PureMACEPolarDDXCalculator,
    )
    from maple.solvation.experimental.mace_polar_torch import (
        build_smd_mace_polar_torch_pes,
    )

    formula, solvent = CASES[args.case]
    atoms = molecule(formula)
    atoms.info.update(charge=0, mult=1)
    started = time.perf_counter()
    pes = build_smd_mace_polar_torch_pes(
        tuple(atoms.get_chemical_symbols()),
        solvent=solvent,
        device=args.device,
        checkpoint_path=args.checkpoint,
    )
    old_profile = (
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE
        if pes.device == "cpu"
        else PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2_PROFILE
    )
    # Use the actual v2 factory's derivative policy, not the generic builder's
    # stricter default. Failed historical runs remain separate evidence.
    os.environ["ROUTE2_MACE_CHECKPOINT"] = str(args.checkpoint.resolve())
    old_calculator = PureMACEPolarDDXCalculator(
        atoms=atoms,
        solvent=solvent,
        device=pes.device,
        profile_spec=route2_smd_profile_spec(old_profile),
    )
    reference = old_calculator.pes
    result.update(
        symbols=atoms.get_chemical_symbols(),
        positions_angstrom=atoms.positions.tolist(),
        solvent=solvent,
        device=pes.device,
        scalar_id=pes.scalar_contract_id,
        model_metadata=pes.model.metadata(),
        configuration_sha256=pes.configuration_sha256(),
        load_seconds=time.perf_counter() - started,
        resource_estimate=asdict(pes.continuum.estimate_resources(len(atoms))),
    )
    _write_progress(args.output_dir, result, "model_and_reference_loaded")
    print(f"loaded {args.case} on {pes.device}", flush=True)
    started = time.perf_counter()
    old = reference.evaluate_forces(atoms)
    result["legacy"] = {
        "energy_eV": old.central_state.total_energy_eV,
        "forces_eV_per_A": old.total_forces_eV_per_A.tolist(),
        "energy_force_seconds": time.perf_counter() - started,
        "profile_id": old_profile,
        "hessian_policy": reference._hessian_backend.policy_payload(),
    }
    _write_progress(args.output_dir, result, "legacy_force_complete")
    if args.legacy_hessian:
        started = time.perf_counter()
        try:
            numerical = reference.evaluate_hessian(atoms)
            result["legacy"]["hessian"] = {
                "status": "returned",
                "matrix_eV_per_A2": numerical.hessian_eV_per_A2.tolist(),
                "maximum_richardson_error_eV_per_A2": numerical.maximum_error_estimate_eV_per_A2,
                "seconds": time.perf_counter() - started,
            }
        except Exception as error:
            result["legacy"]["hessian"] = {
                "status": "failed",
                "type": type(error).__name__,
                "error": str(error),
                "seconds": time.perf_counter() - started,
            }
        _write_progress(args.output_dir, result, "legacy_hessian_attempted")
    if pes.device.startswith("cuda:"):
        torch.cuda.reset_peak_memory_stats(pes.device)
        torch.cuda.synchronize(pes.device)
    started = time.perf_counter()
    evaluated = pes.evaluate_hessian(atoms)
    if pes.device.startswith("cuda:"):
        torch.cuda.synchronize(pes.device)
    result["torch"] = {
        "energy_eV": evaluated.energy_eV,
        "forces_eV_per_A": evaluated.forces_eV_per_A.tolist(),
        "hessian_eV_per_A2": evaluated.hessian_eV_per_A2.tolist(),
        "hessian_seconds": time.perf_counter() - started,
        "raw_antisymmetry_eV_per_A2": evaluated.maximum_antisymmetry_eV_per_A2,
        "evaluation_sha256": evaluated.evaluation_sha256,
        "component_energies_eV": dict(evaluated.component_energies_eV),
        "uncertainty": None,
    }
    _write_progress(args.output_dir, result, "torch_hessian_complete")
    if pes.device.startswith("cuda:"):
        result["torch"]["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
            pes.device
        )
    errors = {
        "energy_eV": abs(evaluated.energy_eV - old.central_state.total_energy_eV),
        "force_eV_per_A": float(
            np.max(np.abs(evaluated.forces_eV_per_A - old.total_forces_eV_per_A))
        ),
    }
    result["parity_errors"] = errors
    result["topology"] = _topology(pes, atoms)
    direction = np.random.default_rng(20260922).normal(size=(len(atoms), 3))
    direction /= np.linalg.norm(direction)
    exact_hv = evaluated.hessian_eV_per_A2 @ direction.ravel()
    direct_hv = pes.hessian_vector_product(atoms, direction).ravel()
    errors["hvp_vs_dense_eV_per_A2"] = float(np.max(np.abs(direct_hv - exact_hv)))
    result["audit_direction"] = direction.tolist()
    audits = []
    for step in AUDIT_STEPS_ANGSTROM:
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions += step * direction
        minus.positions -= step * direction
        plus_topology, minus_topology = _topology(pes, plus), _topology(pes, minus)
        same = plus_topology == result["topology"] == minus_topology
        audit = {"step_angstrom": step, "same_topology": same}
        if same:
            independent_hv = -(
                reference.get_forces(plus) - reference.get_forces(minus)
            ).ravel() / (2 * step)
            audit.update(
                hvp_eV_per_A2=independent_hv.tolist(),
                maximum_error_eV_per_A2=float(
                    np.max(np.abs(independent_hv - exact_hv))
                ),
            )
        audits.append(audit)
        result["independent_legacy_gradient_fd_audit"] = audits
        _write_progress(
            args.output_dir, result, f"independent_audit_{len(audits)}_complete"
        )
    result["independent_legacy_gradient_fd_audit"] = audits
    audit_pass = all(
        entry["same_topology"]
        and entry["maximum_error_eV_per_A2"] <= LIMITS["hessian_eV_per_A2"]
        for entry in audits[-2:]
    )
    if audit_pass:
        stability = float(
            np.max(
                np.abs(
                    np.asarray(audits[-1]["hvp_eV_per_A2"])
                    - np.asarray(audits[-2]["hvp_eV_per_A2"])
                )
            )
        )
        result["fd_finest_step_agreement_eV_per_A2"] = stability
        audit_pass = stability <= LIMITS["hessian_eV_per_A2"]
    if result["legacy"].get("hessian", {}).get("status") == "returned":
        result["old_new_hessian_max_difference_eV_per_A2"] = float(
            np.max(
                np.abs(
                    evaluated.hessian_eV_per_A2
                    - np.asarray(result["legacy"]["hessian"]["matrix_eV_per_A2"])
                )
            )
        )
    result["numerical_pass"] = bool(
        errors["energy_eV"] <= LIMITS["energy_eV"]
        and errors["force_eV_per_A"] <= LIMITS["force_eV_per_A"]
        and errors["hvp_vs_dense_eV_per_A2"] <= LIMITS["hessian_eV_per_A2"]
        and evaluated.maximum_antisymmetry_eV_per_A2 <= LIMITS["hessian_eV_per_A2"]
        and audit_pass
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="water-water")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-hessian", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    before = _source_manifest()
    _write(args.output_dir / "source-manifest.json", before)
    result = {
        "schema": "maple-pure-torch-analytic-canary-v3",
        "case": args.case,
        "requested_device": args.device,
        "limits": LIMITS,
        "audit_steps_angstrom": AUDIT_STEPS_ANGSTROM,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "scientific_release_admitted": False,
        "claim_boundary": "Exact-scalar implementation parity on small synthetic fixtures; not new experimental accuracy, global C2, broad domain or large-system efficiency.",
    }
    started = time.perf_counter()
    try:
        _run_case(args, result)
    except Exception as error:
        result.update(
            numerical_pass=False,
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    result["source_unchanged"] = before == _source_manifest()
    result["elapsed_seconds"] = time.perf_counter() - started
    result["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["pass"] = result.get("numerical_pass", False) and result["source_unchanged"]
    _write(args.output_dir / "result.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("case", "pass", "source_unchanged", "elapsed_seconds")
            }
        ),
        flush=True,
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
