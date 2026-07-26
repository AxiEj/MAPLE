#!/usr/bin/env python3
"""Run a hash-bound ASE-MTK NPT validation of a periodic water Hamiltonian."""

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
from maple.function.dispatcher.solvfe.bulk_water import load_water_box
from maple.function.dispatcher.solvfe.bulk_water_npt import (
    NPT_IMPLEMENTATION_PATHS,
    BulkWaterNPTConfig,
    run_bulk_water_npt,
)
from maple.function.dispatcher.solvfe.provenance import (
    collect_implementation_provenance,
)


def _runtime_provenance() -> dict[str, Any]:
    versions = {}
    for distribution in (
        "ase",
        "mace-torch",
        "numpy",
        "scipy",
        "torch",
    ):
        try:
            versions[distribution] = importlib.metadata.version(
                distribution
            )
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
        NPT_IMPLEMENTATION_PATHS,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run stress-aware ASE IsotropicMTKNPT validation. This is a "
            "correct NPT ensemble but not an exact reproduction of the "
            "MACE-OFF24 OpenMM Monte-Carlo-barostat protocol."
        )
    )
    parser.add_argument("--waterbox", type=Path, required=True)
    parser.add_argument("--waterbox-sha256", required=True)
    parser.add_argument("--expected-waters", type=int, required=True)
    parser.add_argument("--source-url")
    parser.add_argument("--source-commit")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature-k", type=float, default=298.15)
    parser.add_argument("--pressure-bar", type=float, default=1.01325)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument(
        "--precondition-timestep-fs",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--precondition-steps",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "--precondition-friction-per-fs",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--equilibration-steps",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--production-steps",
        type=int,
        default=400_000,
    )
    parser.add_argument(
        "--sample-interval-steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--thermostat-damping-fs",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--barostat-damping-fs",
        type=float,
        default=1_000.0,
    )
    parser.add_argument("--seed", type=int, default=20_260_727)
    parser.add_argument(
        "--rdf-bin-width-angstrom",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--rdf-max-angstrom",
        type=float,
        default=6.0,
    )
    parser.add_argument("--rdf-block-count", type=int, default=8)
    parser.add_argument(
        "--cutoff-margin-angstrom",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--minimum-diagnostic-duration-ps",
        type=float,
        default=400.0,
    )
    parser.add_argument(
        "--minimum-diagnostic-frames",
        type=int,
        default=4_000,
    )
    parser.add_argument(
        "--pressure-mean-tolerance-bar",
        type=float,
        default=500.0,
    )
    return parser


def _progress(row: Mapping[str, float | int | str]) -> None:
    print(
        json.dumps(
            {
                "stage": row["stage"],
                "step": row["step"],
                "time_fs": row["time_fs"],
                "temperature_k": row["temperature_k"],
                "pressure_bar": row["pressure_bar"],
                "density_g_per_ml": row["density_g_per_ml"],
                "volume_angstrom3": row["volume_angstrom3"],
                "force_max_ev_per_angstrom": (
                    row["force_max_ev_per_angstrom"]
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    args = _parser().parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(
            f"OUTPUT_EXISTS: refusing to overwrite {output}"
        )
    atoms, source = load_water_box(
        args.waterbox,
        expected_sha256=args.waterbox_sha256,
        expected_waters=args.expected_waters,
        source_url=args.source_url,
        source_commit=args.source_commit,
    )
    calculator = MACEOff24Provider(
        device=args.device,
        model="maceoff24m",
        checkpoint=args.checkpoint.expanduser().resolve().as_posix(),
        sha256=args.checkpoint_sha256,
        license_ack=True,
    )
    config = BulkWaterNPTConfig(
        temperature_k=args.temperature_k,
        pressure_bar=args.pressure_bar,
        timestep_fs=args.timestep_fs,
        precondition_timestep_fs=args.precondition_timestep_fs,
        precondition_steps=args.precondition_steps,
        precondition_friction_per_fs=(
            args.precondition_friction_per_fs
        ),
        equilibration_steps=args.equilibration_steps,
        production_steps=args.production_steps,
        sample_interval_steps=args.sample_interval_steps,
        thermostat_damping_fs=args.thermostat_damping_fs,
        barostat_damping_fs=args.barostat_damping_fs,
        seed=args.seed,
        rdf_bin_width_angstrom=args.rdf_bin_width_angstrom,
        rdf_max_angstrom=args.rdf_max_angstrom,
        rdf_block_count=args.rdf_block_count,
        interaction_cutoff_angstrom=float(
            calculator.provenance["interaction_cutoff_angstrom"]
        ),
        cutoff_margin_angstrom=args.cutoff_margin_angstrom,
        minimum_diagnostic_duration_ps=(
            args.minimum_diagnostic_duration_ps
        ),
        minimum_diagnostic_frames=args.minimum_diagnostic_frames,
        pressure_mean_tolerance_bar=(
            args.pressure_mean_tolerance_bar
        ),
    )
    calculator_provenance = {
        "schema": "maple-route-a-bulk-water-calculator-v1",
        "calculator": calculator.provenance,
        "runtime": _runtime_provenance(),
    }
    result = run_bulk_water_npt(
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
