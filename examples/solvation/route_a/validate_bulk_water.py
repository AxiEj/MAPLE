#!/usr/bin/env python3
"""Validate the periodic bulk-water Hamiltonian before Route A production."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maple.function.calculator.mace._mace_upstream_calculator import (
    MACEOff24Provider,
)
from maple.function.dispatcher.solvfe.bulk_water import (
    MACE_MD_WATERBOX_COMMIT,
    MACE_MD_WATERBOX_SHA256,
    MACE_MD_WATERBOX_URL,
    BulkWaterNVTConfig,
    load_water_box,
    run_bulk_water_nvt,
)
from maple.function.dispatcher.solvfe.provenance import (
    collect_implementation_provenance,
)


IMPLEMENTATION_PATHS = {
    "maple/function/dispatcher/solvfe/bulk_water.py",
    "maple/function/dispatcher/solvfe/protocol.py",
    "maple/function/dispatcher/solvfe/provenance.py",
    "maple/function/calculator/mace/_mace_upstream_calculator.py",
    "examples/solvation/route_a/validate_bulk_water.py",
}


def _runtime_provenance() -> dict[str, Any]:
    versions = {}
    for distribution in ("ase", "mace-torch", "numpy", "scipy", "torch"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    hardware: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    try:
        import torch

        hardware.update(
            {
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "cuda_device": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
    except ImportError:
        hardware.update(
            {
                "cuda_available": False,
                "cuda_version": None,
                "cuda_device": None,
            }
        )
    return {
        "versions": versions,
        "hardware": hardware,
        "numpy_float_error_policy": np.geterr(),
    }


def _maple_source_provenance() -> dict[str, Any]:
    return collect_implementation_provenance(
        PROJECT_ROOT,
        IMPLEMENTATION_PATHS,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a hash-bound, three-stage NVT validation of MACE-OFF24(M) "
            "on the pinned 64-water mace-md box."
        )
    )
    parser.add_argument("--waterbox", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--default-dtype",
        choices=("float32", "float64"),
        default="float64",
    )
    parser.add_argument("--temperature-k", type=float, default=298.15)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--thermalization-steps", type=int, default=1_000)
    parser.add_argument(
        "--thermalization-friction-per-fs",
        type=float,
        default=0.01,
    )
    parser.add_argument("--equilibration-steps", type=int, default=9_000)
    parser.add_argument(
        "--equilibration-friction-per-fs",
        type=float,
        default=0.001,
    )
    parser.add_argument("--production-steps", type=int, default=20_000)
    parser.add_argument(
        "--production-friction-per-fs",
        type=float,
        default=0.001,
    )
    parser.add_argument("--sample-interval-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_260_726)
    parser.add_argument("--rdf-bin-width-angstrom", type=float, default=0.05)
    parser.add_argument("--rdf-max-angstrom", type=float, default=6.0)
    parser.add_argument("--rdf-block-count", type=int, default=5)
    parser.add_argument(
        "--minimum-diagnostic-duration-ps",
        type=float,
        default=10.0,
    )
    parser.add_argument("--minimum-diagnostic-frames", type=int, default=500)
    return parser


def _progress(row: Mapping[str, float | int | str]) -> None:
    print(
        json.dumps(
            {
                "stage": row["stage"],
                "step": row["step"],
                "time_fs": row["time_fs"],
                "temperature_k": row["temperature_k"],
                "potential_energy_ev": row["potential_energy_ev"],
                "force_max_ev_per_angstrom": (
                    row["force_max_ev_per_angstrom"]
                ),
                "pressure_bar": row["pressure_bar"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    args = _parser().parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"OUTPUT_EXISTS: refusing to overwrite {output}")

    atoms, source = load_water_box(
        args.waterbox,
        expected_sha256=MACE_MD_WATERBOX_SHA256,
        expected_waters=64,
        source_url=MACE_MD_WATERBOX_URL,
        source_commit=MACE_MD_WATERBOX_COMMIT,
    )
    config = BulkWaterNVTConfig(
        temperature_k=args.temperature_k,
        timestep_fs=args.timestep_fs,
        thermalization_steps=args.thermalization_steps,
        thermalization_friction_per_fs=(
            args.thermalization_friction_per_fs
        ),
        equilibration_steps=args.equilibration_steps,
        equilibration_friction_per_fs=args.equilibration_friction_per_fs,
        production_steps=args.production_steps,
        production_friction_per_fs=args.production_friction_per_fs,
        sample_interval_steps=args.sample_interval_steps,
        seed=args.seed,
        rdf_bin_width_angstrom=args.rdf_bin_width_angstrom,
        rdf_max_angstrom=args.rdf_max_angstrom,
        rdf_block_count=args.rdf_block_count,
        minimum_diagnostic_duration_ps=(
            args.minimum_diagnostic_duration_ps
        ),
        minimum_diagnostic_frames=args.minimum_diagnostic_frames,
    )
    calculator = MACEOff24Provider(
        device=args.device,
        model="maceoff24m",
        checkpoint=args.checkpoint.expanduser().resolve().as_posix(),
        sha256=args.checkpoint_sha256,
        license_ack=True,
        default_dtype=args.default_dtype,
    )
    calculator_provenance = {
        "schema": "maple-route-a-bulk-water-calculator-v1",
        "calculator": calculator.provenance,
        "runtime": _runtime_provenance(),
    }
    result = run_bulk_water_nvt(
        atoms,
        calculator=calculator,
        config=config,
        source_provenance=source,
        calculator_provenance=calculator_provenance,
        implementation_provenance=_maple_source_provenance(),
        progress_callback=_progress,
    )
    result.write(output)
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "result_hash": result.result_hash,
                "gates": result.summary["gates"],
                "diagnostics": result.summary["diagnostics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return (
        0
        if result.summary["gates"]["engineering_stability_passed"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
