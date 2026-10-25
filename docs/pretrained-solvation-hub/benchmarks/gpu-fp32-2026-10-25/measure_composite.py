"""Measure the actual scalar gas+AnySolv calculate path, not correction latency.

This diagnostic constructs the registered component classes directly to avoid a
circular composite-card admission. The correction's component card is enforced;
normal production inputs are replayed through SetCalculator after publication.
"""

from __future__ import annotations
import argparse, hashlib, json, time, gc
from pathlib import Path
from statistics import median
import numpy as np
import torch
from ase import Atoms
from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator
from maple.function.calculator.extra_correction.implicit.anysolv import (
    AnySolvBackend,
    ANYSOLV_MOE_CHECKPOINT_SHA256,
    ANYSOLV_SOURCE_REVISION,
)
from maple.function.calculator.solvation_execution import SolvationExecution
from maple.function.benchmarking.pretrained_solvation_gpu import (
    _memory,
    verify_checkpoint,
    AIMNET2_SHA256,
    CALCULATOR_ROOT,
    source_files,
)
from maple.function.benchmarking.solvation_gpu import (
    load_frozen_panel,
    FROZEN_PANEL_SHA256,
    canonical_sha256,
)

ROOT = Path(__file__).resolve().parents[4]
FILES = [
    "maple/function/calculator/aimnet/_aimnet2_calculator.py",
    "maple/function/calculator/set_calculator.py",
    "maple/function/calculator/calculator_base.py",
    "maple/function/calculator/extra_correction/implicit/anysolv.py",
    "maple/function/calculator/solvation_execution.py",
]


def sources(source_root):
    files = source_files("anysolv", Path(source_root), True)
    files += [
        ROOT / "maple/function/calculator/calculator_base.py",
        ROOT / "maple/function/calculator/set_calculator.py",
    ]
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostic", action="store_true", required=True)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--source-root", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    panel = load_frozen_panel(args.panel, FROZEN_PANEL_SHA256)
    case = next(
        c for c in panel["scalar_cases"] if c["case_id"] == "anysolv:C2H5OH:ethanol"
    )
    identity = verify_checkpoint(CALCULATOR_ROOT / "model/aimnet2.pt", AIMNET2_SHA256)
    before = sources(args.source_root)
    report = {
        "schema_version": 1,
        "model_id": "aimnet2-anysolv",
        "route": "anysolv",
        "scope": "public_production_calculate_gas_plus_correction",
        "panel_sha256": FROZEN_PANEL_SHA256,
        "case_id": case["case_id"],
        "checkpoint_sha256s": sorted([AIMNET2_SHA256, ANYSOLV_MOE_CHECKPOINT_SHA256]),
        "checkpoints": {
            "gas": AIMNET2_SHA256,
            "correction": ANYSOLV_MOE_CHECKPOINT_SHA256,
        },
        "source_identity": {
            "source_revision": ANYSOLV_SOURCE_REVISION,
            "source_root": str(Path(args.source_root).resolve()),
        },
        "source_revision": ANYSOLV_SOURCE_REVISION,
        "measurement_path": "AIMNet2Calculator.calculate -> CalcABC._finalize_results -> AnySolvBackend.evaluate",
        "measurement_scope": "combined_gas_plus_correction",
        "source_sha256_before": before,
    }
    for mode in ["reference", "optimized"]:
        execution = SolvationExecution("cuda:0", "float32", mode)
        base_execution = SolvationExecution("cuda:0", "float32", "reference")
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        gas = AIMNet2Calculator(
            device=torch.device("cuda:0"), model="aimnet2", model_path=identity["path"]
        )
        gas._set_scalar_composite_execution(base_execution)
        gas.solvent_correction = AnySolvBackend(
            solvent="ethanol",
            source_root=args.source_root,
            device="cuda:0",
            dtype="float32",
            inference=mode,
            execution_task="sp",
        )
        gas.implemented_properties = ["energy", "free_energy"]
        atoms = Atoms(
            numbers=case["numbers"],
            positions=case["positions_angstrom"],
            info={"charge": 0, "mult": 1},
        )
        atoms.calc = gas
        initial = time.perf_counter()
        initial_energy = atoms.get_potential_energy()
        torch.cuda.synchronize()
        first = time.perf_counter() - initial
        parent = (
            torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.allow_tf32 = True
            hostile = (
                torch.get_float32_matmul_precision(),
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
            )
            with torch.autocast("cuda", dtype=torch.float16):
                atoms.set_positions(
                    np.asarray(case["positions_angstrom"])
                    + 0.00037
                    * np.sin(np.arange(atoms.positions.size)).reshape(
                        atoms.positions.shape
                    )
                )
                atoms.get_potential_energy()
                torch.cuda.synchronize()
                if not torch.is_autocast_enabled("cuda") or hostile != (
                    torch.get_float32_matmul_precision(),
                    torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32,
                ):
                    raise ValueError("Caller hostile precision state was not restored")
        finally:
            torch.set_float32_matmul_precision(parent[0])
            torch.backends.cuda.matmul.allow_tf32 = parent[1]
            torch.backends.cudnn.allow_tf32 = parent[2]
        for i in range(3):
            atoms.set_positions(
                np.asarray(case["positions_angstrom"])
                + 0.001
                * (i + 1)
                * np.sin(np.arange(atoms.positions.size)).reshape(atoms.positions.shape)
            )
            atoms.get_potential_energy()
            torch.cuda.synchronize()
        samples = []
        geometries = []
        for i in range(10):
            positions = np.asarray(case["positions_angstrom"]) + 0.001 * (
                i + 1
            ) * np.sin(np.arange(atoms.positions.size)).reshape(atoms.positions.shape)
            atoms.set_positions(positions)
            geometries.append(
                canonical_sha256(
                    {
                        "numbers": case["numbers"],
                        "positions_angstrom": positions.tolist(),
                    }
                )
            )
            torch.cuda.synchronize()
            t = time.perf_counter()
            atoms.get_potential_energy()
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - t)
        with execution.precision_context():
            runtime = execution.runtime_context()
            effective_precision = execution.check_precision()
        report[mode] = {
            "execution": execution.provenance,
            "inference_mode": mode,
            "effective_precision": effective_precision,
            "resident_models": ["gas", "correction"],
            "runtime": runtime,
            "timing": {
                "status": "success",
                "case_id": case["case_id"],
                "warmup_count": 3,
                "cuda_synchronized": True,
                "fresh_inference": {
                    "count": 10,
                    "samples_seconds": samples,
                    "median_seconds": median(samples),
                    "geometry_sha256s": geometries,
                },
            },
            "memory": {
                "device_total_bytes": _memory(execution)["total_bytes"],
                "peak_reserved_bytes": _memory(execution)["peak_reserved_bytes"],
                "free_bytes_at_peak": _memory(execution)["free_bytes"],
                "measured_headroom_bytes": min(
                    _memory(execution)["free_bytes"],
                    _memory(execution)["total_bytes"]
                    - _memory(execution)["peak_reserved_bytes"],
                ),
            },
            "initialization_seconds": initial - started,
            "first_call_seconds": first,
            "initial_energy_hartree": initial_energy,
            "hostile_precision_restored": True,
            "gas_tensor_dtype": str(next(gas.model.parameters()).dtype),
            "gas_device": str(next(gas.model.parameters()).device),
            "correction_provenance": gas.solvent_correction.provenance,
        }
        print(mode, median(samples), "seconds combined fresh", flush=True)
        del atoms, gas
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    report["source_sha256_after"] = sources(args.source_root)
    report["source_drift"] = before != report["source_sha256_after"]
    if report["source_drift"]:
        raise ValueError("Source changed during composite measurement")
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
