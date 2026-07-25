#!/usr/bin/env python3
"""Benchmark warm Route 1 energy+force composition against a named MM baseline."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

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


def timing_summary(milliseconds: list[float]) -> dict[str, float | int]:
    values = np.asarray(milliseconds, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Timing samples must be a non-empty finite vector.")
    return {
        "n": int(len(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "mean_ms": float(np.mean(values)),
    }


def benchmark_calls(
    function: Callable[[int], None],
    *,
    samples: int,
    warmups: int,
    synchronize: Callable[[], None],
) -> dict[str, float | int]:
    if samples < 1 or warmups < 0:
        raise ValueError("samples must be positive and warmups must be non-negative.")
    for index in range(warmups):
        function(index)
    synchronize()
    milliseconds = []
    for index in range(samples):
        synchronize()
        started = time.perf_counter()
        function(index)
        synchronize()
        milliseconds.append((time.perf_counter() - started) * 1000.0)
    return timing_summary(milliseconds)


def benchmark_paired_calls(
    first: Callable[[int], None],
    second: Callable[[int], None],
    *,
    samples: int,
    warmups: int,
    synchronize: Callable[[], None],
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    """Interleave two independent call paths to reduce ordering and drift bias."""
    if samples < 1 or warmups < 0:
        raise ValueError("samples must be positive and warmups must be non-negative.")

    def timed_call(function: Callable[[int], None], index: int) -> float:
        synchronize()
        started = time.perf_counter()
        function(index)
        synchronize()
        return (time.perf_counter() - started) * 1000.0

    for index in range(warmups):
        if index % 2:
            second(index)
            first(index)
        else:
            first(index)
            second(index)
    synchronize()

    first_milliseconds = []
    second_milliseconds = []
    for index in range(samples):
        if index % 2:
            second_milliseconds.append(timed_call(second, index))
            first_milliseconds.append(timed_call(first, index))
        else:
            first_milliseconds.append(timed_call(first, index))
            second_milliseconds.append(timed_call(second, index))
    return timing_summary(first_milliseconds), timing_summary(second_milliseconds)


def load_charge_vector(
    manifest_path: str | Path,
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
            f"Expected one charge-manifest record for {compound_id}, found {len(matches)}."
        )
    record = matches[0]
    charges = np.asarray(record.get("am1bcc_charges_e"), dtype=np.float64)
    if charges.shape != (atom_count,) or not np.isfinite(charges).all():
        raise ValueError(
            "Charge manifest does not contain one finite AM1-BCC charge per atom."
        )
    return charges, record


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from ase.calculators.calculator import all_changes
    from openmm import Context, Platform, VerletIntegrator, unit
    from openmm.app import AmberPrmtopFile, NoCutoff, OBC2

    from maple.function.calculator.set_calculator import SetCalculator
    from maple.function.read.filereader.mol2_reader import MOL2Reader

    if args.torch_threads is not None:
        if args.torch_threads < 1:
            raise ValueError("torch_threads must be positive.")
        torch.set_num_threads(args.torch_threads)

    mol2_path = Path(args.mol2).resolve()
    manifest_path = Path(args.charge_manifest).resolve()
    prmtop_path = Path(args.prmtop).resolve()
    for path in (mol2_path, manifest_path, prmtop_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
    charges, charge_record = load_charge_vector(
        manifest_path,
        args.compound_id,
        atom_count=len(atoms),
    )
    expected_mol2_hash = charge_record.get("source_mol2_sha256")
    observed_mol2_hash = sha256_file(mol2_path)
    if expected_mol2_hash and expected_mol2_hash != observed_mol2_hash:
        raise ValueError("MOL2 hash does not match the charge-manifest source record.")

    device = torch.device(args.device)
    positions = np.asarray(atoms.get_positions(), dtype=np.float64).copy()

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def set_positions(index: int) -> np.ndarray:
        displaced = positions.copy()
        displaced[0, 0] += (1.0 if index % 2 else -1.0) * 1.0e-5
        atoms.set_positions(displaced)
        return displaced

    available_platforms = [
        Platform.getPlatform(index).getName()
        for index in range(Platform.getNumPlatforms())
    ]
    requested_platforms = list(dict.fromkeys(args.openmm_platform))
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "route1-warm-energy-force-local-trace",
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
            "Named-platform, single-molecule warm local trace. Paired combined-minus-gas "
            "and correction-only timings are local observations without uncertainty "
            "intervals. An MM wall-time ratio is only a named same-host, single-thread "
            "local observation under the recorded CPU/Reference resource policy; it is "
            "not a production-MM or general throughput comparison."
        ),
        "compound_id": args.compound_id,
        "molecule": args.molecule_name,
        "atom_count": len(atoms),
        "task": "warm in-process energy+forces",
        "route1": {
            "gas_model": args.model,
            "charge": "AM1-BCC from frozen manifest",
            "polar": "OpenMM OBC-II",
            "nonpolar": "OpenMM ACE",
        },
        "mm_baseline": {
            "potential": "GAFF2 + OpenMM OBC-II",
            "input": str(prmtop_path),
        },
        "inputs": {
            "mol2": str(mol2_path),
            "mol2_sha256": observed_mol2_hash,
            "charge_manifest": str(manifest_path),
            "charge_manifest_sha256": sha256_file(manifest_path),
            "prmtop": str(prmtop_path),
            "prmtop_sha256": sha256_file(prmtop_path),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "openmm": importlib.metadata.version("openmm"),
            "device": str(device),
            "torch_num_threads": torch.get_num_threads(),
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "available_openmm_platforms": available_platforms,
        },
        "warm_energy_force": {},
        "ratios": {},
        "solvent_timing_observations": {},
        "mm_baseline_observations": {},
        "skipped_platforms": [],
        "limitations": [
            "single development molecule",
            "charge generation and model/context startup excluded",
            "gas and combined timings use interleaved independent calculator instances",
            "no OPT or SCAN wall-time comparison",
            "platform-specific timings are not transferable speed guarantees",
            "absolute paths make this artifact locally traceable, not checkout-portable",
        ],
    }
    prmtop = AmberPrmtopFile(str(prmtop_path))

    for requested in requested_platforms:
        match = next(
            (name for name in available_platforms if name.lower() == requested.lower()),
            None,
        )
        if match is None:
            result["skipped_platforms"].append(requested)
            continue
        suffix = match.lower()
        atoms.set_initial_charges(charges)
        log_path = Path(args.log).resolve()
        platform_log = log_path.with_name(f"{log_path.stem}-{suffix}{log_path.suffix}")
        gas_log = log_path.with_name(f"{log_path.stem}-gas-{suffix}{log_path.suffix}")
        gas_calculator = SetCalculator(
            device,
            args.model,
            str(gas_log),
            atoms=atoms,
        ).set_calculator()
        combined_calculator = SetCalculator(
            device,
            args.model,
            str(platform_log),
            atoms=atoms,
            implicit="gb",
            solvent="water",
            charge_options={
                "source": "mol2",
                "label": "am1bcc-frozen-manifest",
            },
            solvation_options={
                "method": "gb",
                "model": "obc2",
                "nonpolar": "ace",
                "platform": match,
                "experimental": True,
            },
        ).set_calculator()
        solvent = combined_calculator.solvent_correction
        system = prmtop.createSystem(
            nonbondedMethod=NoCutoff,
            implicitSolvent=OBC2,
            constraints=None,
        )
        integrator = VerletIntegrator(0.001 * unit.picoseconds)
        context = Context(system, integrator, Platform.getPlatformByName(match))

        def correction_call(index: int) -> None:
            set_positions(index)
            solvent.evaluate(atoms, need_forces=True)

        def gas_call(index: int) -> None:
            set_positions(index)
            gas_calculator.calculate(
                atoms,
                properties=["energy", "forces"],
                system_changes=all_changes,
            )

        def combined_call(index: int) -> None:
            set_positions(index)
            combined_calculator.calculate(
                atoms,
                properties=["energy", "forces"],
                system_changes=all_changes,
            )

        def mm_call(index: int) -> None:
            displaced = set_positions(index)
            context.setPositions(displaced * 0.1 * unit.nanometer)
            context.getState(getEnergy=True, getForces=True)

        correction_key = f"openmm_obc2_ace_correction_{suffix}"
        gas_key = f"{args.model}_gas_paired_{suffix}_{device.type}"
        combined_key = f"{args.model}_plus_obc2_ace_{suffix}"
        mm_key = f"gaff2_plus_obc2_{suffix}"
        gas_timing, combined_timing = benchmark_paired_calls(
            gas_call,
            combined_call,
            samples=args.samples,
            warmups=args.warmups,
            synchronize=synchronize,
        )
        result["warm_energy_force"][gas_key] = gas_timing
        result["warm_energy_force"][combined_key] = combined_timing
        for key, function in (
            (correction_key, correction_call),
            (mm_key, mm_call),
        ):
            result["warm_energy_force"][key] = benchmark_calls(
                function,
                samples=args.samples,
                warmups=args.warmups,
                synchronize=synchronize,
            )
        combined_median = result["warm_energy_force"][combined_key]["median_ms"]
        gas_median = result["warm_energy_force"][gas_key]["median_ms"]
        mm_median = result["warm_energy_force"][mm_key]["median_ms"]
        correction_median = result["warm_energy_force"][correction_key]["median_ms"]
        paired_fraction = combined_median / gas_median - 1.0
        correction_fraction = correction_median / gas_median
        result["ratios"][
            f"{suffix}_paired_combined_minus_gas_fraction_observed"
        ] = paired_fraction
        result["ratios"][
            f"{suffix}_correction_time_fraction_vs_gas_mlip"
        ] = correction_fraction
        result["solvent_timing_observations"][suffix] = {
            "paired_combined_minus_gas_fraction_observed": paired_fraction,
            "correction_time_fraction_vs_gas_mlip": correction_fraction,
            "interpretation": (
                "positive local paired observation; no uncertainty interval"
                if paired_fraction >= 0.0
                else "end-to-end difference is unresolved at this noise level; "
                "do not interpret the negative observation as a speedup"
            ),
        }
        comparable = (
            device.type == "cpu"
            and suffix == "reference"
            and torch.get_num_threads() == 1
        )
        result["mm_baseline_observations"][suffix] = {
            "mlip_plus_gb_over_mm_plus_gb": combined_median / mm_median,
            "same_host_local_observation": comparable,
            "not_production_throughput_comparison": True,
            "resource_policy": (
                "same-host CPU; PyTorch one thread; OpenMM Reference single-thread"
                if comparable
                else "heterogeneous or uncontrolled execution resources"
            ),
            "interpretation": (
                "named same-host single-thread local observation; not a "
                "production-MM or general throughput comparison"
                if comparable
                else "raw mixed-backend wall-time observation; not a local MM "
                "comparison or speed claim"
            ),
        }
    return seal_artifact(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mol2", required=True)
    parser.add_argument("--charge-manifest", required=True)
    parser.add_argument("--prmtop", required=True)
    parser.add_argument("--compound-id", required=True)
    parser.add_argument("--molecule-name", required=True)
    parser.add_argument("--model", default="maceoff23m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--openmm-platform", action="append", default=[])
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=6)
    parser.add_argument("--log", default="/tmp/maple-route1-performance.log")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.openmm_platform:
        args.openmm_platform = ["Reference", "CPU", "CUDA"]
    result = run_benchmark(args)
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
