#!/usr/bin/env python3
"""Freeze the first matched-QM boundary-response audit before execution.

The preregistration opens no experimental solvation target.  It binds one
previously opened acetone gas-phase QM checkpoint, its already frozen
PCMSolver cavity, two finite-field steps, one independent basis control, both
ML checkpoints, the executable runtime, and every source file used by the
audit.  The later runner may report a mechanism defect but cannot select or
fit a source transformation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[2]
PARENT_PREREGISTRATION = Path(
    "docs/implicit-solvation/benchmarks/"
    "route2-gto-pcm-energy-projection-four-prereg-v1.json"
)
CASE_ID = "mobley_3867265"
CUTOFF_DIRECTORY = "cutoff-1e-10"
AU_ELECTRIC_FIELD_V_PER_ANGSTROM = 51.4220674763
REQUIRED_SOURCE_FILES = (
    "maple/function/calculator/extra_correction/implicit/gto_density.py",
    "maple/function/calculator/extra_correction/implicit/pcmsolver.py",
    "maple/solvation/coupling/adt_radial_shape.py",
    "maple/solvation/coupling/atomic_displacement_lift.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_adt.py",
    "maple/solvation/models/mace_polar_separated.py",
    "maple/solvation/reference/pyscf_pcmsolver.py",
    "maple/solvation/release/qm_boundary_response.py",
    "tools/route2_release/create_mace_mdp_polar_qm_boundary_response_preregistration.py",
    "tools/route2_release/run_mace_mdp_polar_qm_boundary_response.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--pcmsolver-library", type=Path, required=True)
    parser.add_argument("--mdp-checkpoint", type=Path, required=True)
    parser.add_argument("--polar-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--polar-device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object.")
    return payload


def _rebased(asset_root: Path, raw: object) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("asset path must be a nonempty string.")
    path = Path(raw)
    if path.is_file():
        return path.resolve()
    try:
        marker = path.parts.index(".omx")
    except ValueError as exc:
        raise ValueError(f"cannot rebase {raw!r} under the asset root.") from exc
    path = asset_root.joinpath(*path.parts[marker:]).resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _file_record(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _case_record(asset_root: Path) -> dict[str, object]:
    parent_path = asset_root / PARENT_PREREGISTRATION
    parent = _load_json(parent_path)
    records = parent.get("records")
    if not isinstance(records, list):
        raise ValueError("parent preregistration has no records.")
    matching = [
        item
        for item in records
        if isinstance(item, dict) and item.get("compound_id") == CASE_ID
    ]
    if len(matching) != 1:
        raise ValueError("parent preregistration has no unique acetone record.")
    inherited = matching[0]
    panel = (
        asset_root
        / ".omx/benchmarks/route2-gto-pcm-energy-projection-four-v1"
        / CASE_ID
        / CUTOFF_DIRECTORY
    )
    result_path = panel / "result.json"
    result = _load_json(result_path)
    inputs = result.get("inputs")
    if not isinstance(inputs, dict) or inputs.get("compound_id") != CASE_ID:
        raise ValueError("acetone projection result identity is invalid.")
    checkpoint = inputs.get("qm_checkpoint")
    pcm_input = inputs.get("parsed_pcm_input")
    if not isinstance(checkpoint, dict) or not isinstance(pcm_input, dict):
        raise ValueError("acetone projection result omits frozen input records.")
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("acetone checkpoint provenance is missing.")
    mol2 = asset_root / str(inherited["mol2_path"])
    paths = {
        "parent_preregistration": parent_path,
        "projection_result": result_path,
        "mol2": mol2,
        "qm_checkpoint": _rebased(asset_root, checkpoint.get("path")),
        "qm_ledger": _rebased(asset_root, provenance.get("ledger_path")),
        "pcm_input": _rebased(asset_root, pcm_input.get("path")),
        "frozen_surface": panel / "work/surface.npz",
        "frozen_qm_surface_mep": panel / "work/qm-surface-mep.npz",
    }
    return {
        "case_id": CASE_ID,
        "name": "acetone",
        "prior_target_status": (
            "vacuum-QM surface MEP and fixed-source PCM energy were opened in "
            "earlier representation audits; no experimental solvation target "
            "is read by this preregistration or runner"
        ),
        "files": {name: _file_record(path) for name, path in paths.items()},
    }


def _runtime(*, pcmsolver: Path, threads: int, polar_device: str) -> dict[str, object]:
    import numpy as np
    import pyscf
    import scipy
    import torch

    executable = Path(sys.executable).resolve(strict=True)
    return {
        "python_version": platform.python_version(),
        "python_executable": _file_record(executable),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pyscf_version": pyscf.__version__,
        "pyscf_init": _file_record(Path(pyscf.__file__).resolve(strict=True)),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available_at_lock": bool(torch.cuda.is_available()),
        "pcmsolver_library": _file_record(pcmsolver),
        "threads": int(threads),
        "polar_device": polar_device,
    }


def create(args: argparse.Namespace) -> dict[str, object]:
    if args.threads < 1:
        raise ValueError("threads must be positive.")
    asset_root = args.asset_root.expanduser().resolve(strict=True)
    pcmsolver = args.pcmsolver_library.expanduser().resolve(strict=True)
    mdp = args.mdp_checkpoint.expanduser().resolve(strict=True)
    polar = args.polar_checkpoint.expanduser().resolve(strict=True)
    missing_sources = [
        relative
        for relative in REQUIRED_SOURCE_FILES
        if not (SOURCE_ROOT / relative).is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(f"required source files are missing: {missing_sources}")
    payload: dict[str, object] = {
        "artifact": "route2-mace-mdp-polar-matched-qm-boundary-response-prereg-v1",
        "schema_version": 1,
        "status": "frozen-before-first-qm-induced-boundary-response-execution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(SOURCE_ROOT),
        "source_files_sha256": {
            relative: _sha256_file(SOURCE_ROOT / relative)
            for relative in REQUIRED_SOURCE_FILES
        },
        "inputs": {
            "case": _case_record(asset_root),
            "mace_mdp_checkpoint": _file_record(mdp),
            "mace_polar_checkpoint": _file_record(polar),
        },
        "runtime": _runtime(
            pcmsolver=pcmsolver,
            threads=args.threads,
            polar_device=args.polar_device,
        ),
        "reference_protocol": {
            "method": "omegaB97M-V",
            "primary_basis": "def2-tzvpd",
            "control_basis": "def2-tzvp",
            "reference": "RKS-density-fitting",
            "semilocal_grid_level": 3,
            "nonlocal_atom_grid": [50, 194],
            "nonlocal_prune": "pyscf.dft.gen_grid.sg1_prune",
            "scf_energy_tolerance_hartree": 1.0e-10,
            "scf_gradient_tolerance": 1.0e-7,
            "maximum_scf_cycles": 100,
            "field_gradient_steps_volt_per_angstrom": [1.0e-3, 5.0e-4],
            "control_basis_steps_volt_per_angstrom": [5.0e-4],
            "axes": ["x", "y", "z"],
            "native_coordinate": "g=grad(phi)=-physical-electric-field",
            "electronic_hcore_perturbation": "-g_dot_r",
            "atomic_unit_electric_field_volt_per_angstrom": (
                AU_ELECTRIC_FIELD_V_PER_ANGSTROM
            ),
            "qm_induced_surface_mep": "-Tr[(dD/dg) r_surface^-1]",
            "qm_induced_dipole": "-Tr[(dD/dg) r] in e*angstrom",
        },
        "candidate_protocol": {
            "permanent_source": "MACE-POLAR zero-field point q/p",
            "uniform_response": (
                "role-separated canonical ADT from negative MACE-MDP molecular "
                "polarizability times native uniform gradient"
            ),
            "comparison_surface": "identical frozen PCMSolver cavity",
            "comparison_metrics": [
                "area-weighted boundary-MEP relative L2",
                "passive-continuum response norm sqrt(-v^T R v)",
                "response-metric correlation",
                "molecular induced-dipole derivative",
            ],
        },
        "decision_contract": {
            "reference_uncertainty": (
                "per axis, max of primary finite-field step difference and "
                "primary/control-basis fine-step difference in the same response norm"
            ),
            "resolved_hybrid_defect": (
                "candidate error greater than five times the frozen reference "
                "uncertainty envelope"
            ),
            "not_a_fit": True,
            "no_projector_selected": True,
            "no_source_rescaling": True,
            "no_solvation_target_read": True,
            "no_capability_admission": True,
        },
        "claim_boundary": {
            "can_decide": [
                "whether current permanent boundary potential matches vacuum QM on one case",
                "whether the current ADT uniform response matches finite-field QM on three axes",
                "whether a discrepancy exceeds preregistered finite-field/basis uncertainty",
            ],
            "cannot_decide": [
                "505-molecule solvation accuracy",
                "nonuniform residual-response accuracy",
                "a new source projector or fitted repair",
                "force, Hessian, virial, optimization, frequency, or MD admission",
            ],
        },
    }
    payload["preregistration_sha256"] = _canonical_sha256(payload)
    return payload


def main() -> None:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = create(args)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    print(json.dumps({"output": str(output), "sha256": _sha256_file(output)}))


if __name__ == "__main__":
    main()
