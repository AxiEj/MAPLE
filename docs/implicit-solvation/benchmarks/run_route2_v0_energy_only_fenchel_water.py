#!/usr/bin/env python3
"""Run the preregistered MACE-energy-only Route-2 V0-FE falsifier.

The calculation evaluates the exact autograd Hessian of the frozen MACE-POLAR
intrinsic local-field energy.  It deliberately does not use the model's
separate density head: the question is whether the scalar energy itself could
legitimately define a zero-training electronic functional by Legendre--Fenchel
conjugacy on the neutral coefficient space.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ase
import numpy as np
import torch
from ase.build import molecule

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maple.function.calculator.extra_correction.implicit.gto_density import (
    density_to_external_field_order,
)
from maple.function.calculator.extra_correction.implicit.route2_response import (
    NeutralDensityCoordinates,
)
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.read.command_control import CommandControl
from maple.function.route2_smd_profiles import DDPCM_SMD_PROFILE

ARTIFACT_ID = "route2-v0-energy-only-fenchel-water-v1"
PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-energy-only-fenchel-water-prereg-v1.json"
)
INPUT_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-mace-local-field-thermodynamic-canary-v1.json"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/run_route2_v0_energy_only_fenchel_water.py"
)
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    PREREG_RELATIVE_PATH,
    "maple/function/calculator/mace/_macepol_calculator.py",
    "maple/function/calculator/extra_correction/implicit/electrostatic_pairing.py",
    "maple/function/calculator/extra_correction/implicit/gto_density.py",
    "maple/function/calculator/extra_correction/implicit/route2_response.py",
)
PUBLIC_SETTINGS = (
    "#model=macepol-m",
    "#sp(verbose=1)",
    (
        "#solv(implicit=water,method=smd,provider=pyddx,"
        f"profile={DDPCM_SMD_PROFILE},response=scf,"
        "standard_state=1m,experimental=true)"
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _geometry_sha256(atoms) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(atoms.numbers, dtype=np.int64).tobytes())
    digest.update(np.asarray(atoms.get_positions(), dtype=np.float64).tobytes())
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def _require_clean_source() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "The V0-FE falsifier requires a clean tracked source checkout; "
            f"git reported:\n{status}"
        )
    for relative in SOURCE_RELATIVE_PATHS:
        _git("ls-files", "--error-unmatch", relative)
    return _git("rev-parse", "HEAD")


def _source_hashes() -> dict[str, str]:
    return {
        relative: _sha256(REPO_ROOT / relative) for relative in SOURCE_RELATIVE_PATHS
    }


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _load_protocol_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    prereg_path = REPO_ROOT / PREREG_RELATIVE_PATH
    input_path = REPO_ROOT / INPUT_RELATIVE_PATH
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    source = json.loads(input_path.read_text(encoding="utf-8"))
    frozen = prereg["frozen_input"]
    if prereg["status"] != "preregistered-before-execution":
        raise RuntimeError("The V0-FE protocol is not preregistered.")
    if frozen["input_artifact"] != INPUT_RELATIVE_PATH:
        raise RuntimeError("The V0-FE frozen-input path changed.")
    if frozen["input_artifact_sha256"] != _sha256(input_path):
        raise RuntimeError("The V0-FE frozen-input artifact hash changed.")
    if source["artifact"] != "route2-mace-local-field-thermodynamic-canary-v1":
        raise RuntimeError("The V0-FE input artifact identity is invalid.")
    if source["checkpoint"]["sha256"] != frozen["checkpoint_sha256"]:
        raise RuntimeError("The V0-FE checkpoint does not match the preregistration.")
    return prereg, source


def _neutral_external_basis(atom_count: int) -> np.ndarray:
    coordinates = NeutralDensityCoordinates(atom_count)
    basis = np.empty((4 * atom_count, coordinates.dimension), dtype=float)
    for column in range(coordinates.dimension):
        reduced = np.zeros(coordinates.dimension, dtype=float)
        reduced[column] = 1.0
        basis[:, column] = density_to_external_field_order(
            coordinates.expand(reduced)
        ).reshape(-1)
    return basis


def main() -> int:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    git_head = _require_clean_source()
    prereg, source = _load_protocol_inputs()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    total_started = time.perf_counter()
    atoms = molecule("H2O")
    atoms.info.update(charge=0, mult=1, mol2={"atom_types": ["o", "h", "h"]})
    if _geometry_sha256(atoms) != source["molecule"]["geometry_sha256"]:
        raise RuntimeError("The frozen V0-FE water geometry changed.")

    parameters = CommandControl.from_settings(list(PUBLIC_SETTINGS)).as_dict()
    _sync(args.device)
    started = time.perf_counter()
    calculator = SetCalculator(
        args.device,
        parameters["model"],
        str(work_dir / "maple.out"),
        atoms=atoms,
        d4=bool(parameters.get("d4", False)),
        implicit=parameters["solv"]["method"],
        solvent=parameters["solv"]["implicit"],
        model_options=parameters.get("model_options"),
        solvation_options=parameters["solv"],
        charge_options=parameters.get("charge") or {},
    ).set_calculator()
    _sync(args.device)
    model_load_seconds = time.perf_counter() - started
    checkpoint = dict(getattr(calculator, "mace_polar_checkpoint_provenance", {}) or {})
    if checkpoint.get("sha256") != prereg["frozen_input"]["checkpoint_sha256"]:
        raise RuntimeError("The live MACE-POLAR checkpoint changed.")

    basis = _neutral_external_basis(len(atoms))
    field_dimension = 4 * len(atoms)

    def energy(flat_field: torch.Tensor) -> torch.Tensor:
        field = flat_field.reshape(len(atoms), 4)
        result = calculator.polar_output_torch(
            atoms,
            node_potential_ev=field[:, 0],
            node_gradient_ev_per_angstrom=field[:, 1:],
        )
        result_energy = result.get("energy")
        if result_energy is None or not torch.is_tensor(result_energy):
            raise RuntimeError("MACE-POLAR did not return a differentiable energy.")
        return result_energy.sum()

    zero = torch.zeros(
        field_dimension,
        dtype=calculator.dtype,
        device=calculator.device,
        requires_grad=True,
    )
    _sync(args.device)
    started = time.perf_counter()
    zero_energy = energy(zero)
    raw_gradient = torch.autograd.functional.jacobian(energy, zero, vectorize=True)
    raw_hessian = torch.autograd.functional.hessian(energy, zero, vectorize=True)
    _sync(args.device)
    energy_hessian_seconds = time.perf_counter() - started

    gradient = np.asarray(raw_gradient.detach().cpu(), dtype=float)
    hessian = np.asarray(raw_hessian.detach().cpu(), dtype=float)
    if hessian.shape != (field_dimension, field_dimension) or not np.all(
        np.isfinite(hessian)
    ):
        raise RuntimeError("The MACE intrinsic-energy Hessian is invalid.")
    neutral_hessian = basis.T @ hessian @ basis
    symmetric = 0.5 * (neutral_hessian + neutral_hessian.T)
    antisymmetric = 0.5 * (neutral_hessian - neutral_hessian.T)
    symmetric_norm = float(np.linalg.norm(symmetric))
    antisymmetric_norm = float(np.linalg.norm(antisymmetric))
    antisymmetric_ratio = antisymmetric_norm / max(symmetric_norm, 1.0e-30)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    maximum_eigenvalue = float(np.max(eigenvalues))
    gates = prereg["gates"]
    gate_results = {
        "neutral_hessian_reciprocity": antisymmetric_ratio
        <= float(gates["neutral_hessian_antisymmetric_frobenius_ratio_max"]),
        "neutral_hessian_concavity": maximum_eigenvalue
        <= float(gates["neutral_hessian_maximum_eigenvalue_ev_max"]),
    }
    failed_gates = [name for name, passed in gate_results.items() if not passed]
    status = "pass" if not failed_gates else "fail"

    artifact = {
        "artifact": ARTIFACT_ID,
        "schema_version": 1,
        "status": status,
        "failed_gates": failed_gates,
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "execution_git_head": git_head,
        "source_files_sha256": _source_hashes(),
        "preregistration": {
            "path": PREREG_RELATIVE_PATH,
            "sha256": _sha256(REPO_ROOT / PREREG_RELATIVE_PATH),
            "protocol_id": prereg["protocol_id"],
        },
        "frozen_input": {
            "path": INPUT_RELATIVE_PATH,
            "sha256": _sha256(REPO_ROOT / INPUT_RELATIVE_PATH),
            "artifact": source["artifact"],
        },
        "scientific_identity": {
            "solute_response_model": "official MACE-POLAR-1-M",
            "construction": prereg["construction"]["energy_scalar"],
            "field_interface": prereg["construction"]["field_interface"],
            "geometry_policy": "fixed archived ASE G2 water",
            "continuum_fixed_point_solved": False,
            "mace_density_output_used": False,
            "training_or_fine_tuning": False,
            "experimental_solvation_labels_read": False,
        },
        "claim_boundary": prereg["claim_boundary"],
        "stop_condition": prereg["stop_condition"],
        "gates": {"thresholds": gates, "passed": gate_results},
        "molecule": source["molecule"],
        "checkpoint": checkpoint,
        "energy_scalar": {
            "zero_field_intrinsic_energy_ev": float(zero_energy.detach().cpu()),
            "neutral_anchor_gradient_inf": float(
                np.linalg.norm(basis.T @ gradient, ord=np.inf)
            ),
            "raw_hessian_antisymmetric_inf": float(
                np.linalg.norm(hessian - hessian.T, ord=np.inf)
            ),
            "neutral_hessian_dimension": int(neutral_hessian.shape[0]),
            "neutral_hessian_symmetric_frobenius_norm_ev": symmetric_norm,
            "neutral_hessian_antisymmetric_frobenius_norm_ev": antisymmetric_norm,
            "neutral_hessian_antisymmetric_to_symmetric_frobenius_ratio": antisymmetric_ratio,
            "neutral_hessian_minimum_eigenvalue_ev": float(np.min(eigenvalues)),
            "neutral_hessian_maximum_eigenvalue_ev": maximum_eigenvalue,
            "neutral_hessian_eigenvalues_ev": eigenvalues.tolist(),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "ase": ase.__version__,
            "mace_torch": importlib.metadata.version("mace-torch"),
            "graph_longrange": importlib.metadata.version("graph-longrange"),
            "device": args.device,
            "device_name": torch.cuda.get_device_name()
            if args.device == "cuda"
            else platform.processor(),
            "ld_preload": os.environ.get("LD_PRELOAD", ""),
            "timing_seconds": {
                "model_load": model_load_seconds,
                "energy_gradient_and_exact_hessian": energy_hessian_seconds,
                "total": time.perf_counter() - total_started,
            },
        },
    }
    _write_exclusive_json(output, artifact)
    print(output)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
