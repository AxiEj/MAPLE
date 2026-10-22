"""Bounded CPU/CUDA v4 engineering replay; no experimental-accuracy admission."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from ase import Atoms

from maple.solvation.experimental.mace_polar_torch import build_smd_mace_polar_torch_pes

ROOT = Path(__file__).resolve().parents[2]


def source_manifest():
    paths = sorted((ROOT / "maple").rglob("*.py")) + [Path(__file__).resolve()]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def plain(value):
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve existing evidence: {args.output}")
    # Explicit CLI process bootstrap, never a library-global dtype mutation.
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    atoms = Atoms(
        "OH2",
        positions=[[0, 0, 0], [0.9572, 0, 0], [-0.239, 0.9266, 0]],
        info={"charge": 0, "mult": 1},
    )
    direction = np.arange(9).reshape(3, 3) / 9
    sources = source_manifest()
    record = {
        "artifact_id": "route2-pure-macepolar-torch-fp32-v4-engineering-tests",
        "schema_version": 1,
        "claim_boundary": "One water geometry, two solvents and two devices; engineering replay only, not complete-population scientific qualification.",
        "scientific_release_admitted": False,
        "physical_force_accuracy_admitted": False,
        "experimental_solvation_accuracy_admitted": False,
        "dataset_labels_used": False,
        "source_manifest": sources,
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda},
        "geometry": {
            "symbols": atoms.get_chemical_symbols(),
            "positions_angstrom": atoms.positions.tolist(),
        },
        "hvp_direction": direction.tolist(),
        "hvp_dense_max_budget_eV_per_A2": 2e-4,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for solvent in ("water", "hexane"):
        for device in ("cpu", "cuda:0"):
            start = time.perf_counter()
            case = {"solvent": solvent, "device": device, "pass": False}
            try:
                pes = build_smd_mace_polar_torch_pes(
                    atoms.get_chemical_symbols(), solvent=solvent, device=device
                )
                energy = pes.get_potential_energy(atoms)
                solvation = pes.get_solvation_energy(atoms)
                force = pes.get_forces(atoms)
                h = pes.evaluate_hessian(atoms)
                hv = pes.hessian_vector_product(atoms, direction)
                defect = float(
                    np.max(np.abs(hv.ravel() - h.hessian_eV_per_A2 @ direction.ravel()))
                )
                if defect >= record["hvp_dense_max_budget_eV_per_A2"]:
                    raise RuntimeError(
                        f"HVP/dense Hessian defect exceeds budget: {defect}"
                    )
                if (
                    h.dtype != "float32"
                    or pes.continuum.dtype is not torch.float32
                    or pes.solvent_term.dtype is not torch.float32
                ):
                    raise RuntimeError("Component computation precision is not FP32.")
                case.update(
                    {
                        "pass": True,
                        "energy_eV": energy,
                        "solvation_energy_eV": solvation,
                        "forces_eV_per_A": force.tolist(),
                        "hessian_eV_per_A2": h.hessian_eV_per_A2.tolist(),
                        "hvp_eV_per_A2": hv.tolist(),
                        "hvp_dense_defect_eV_per_A2": defect,
                        "computation_dtype": h.dtype,
                        "storage_dtype": h.storage_dtype,
                        "evaluation_sha256": h.evaluation_sha256,
                        "configuration_sha256": pes.configuration_sha256(),
                        "profile": pes.profile_id,
                        "scalar": pes.scalar_contract_id,
                        "continuum_policy": dict(pes.continuum.numerical_policy),
                        "diagnostics": plain(h.diagnostics),
                    }
                )
            except Exception as error:
                case["error"] = f"{type(error).__name__}: {error}"
            case["elapsed_seconds_descriptive_only"] = time.perf_counter() - start
            record["cases"].append(case)
            record["source_unchanged"] = sources == source_manifest()
            args.output.write_text(
                json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            print(solvent, device, case["pass"], case.get("error", ""), flush=True)
    if not record["source_unchanged"] or not all(c["pass"] for c in record["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
