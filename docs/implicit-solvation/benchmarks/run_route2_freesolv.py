#!/usr/bin/env python3
"""Prepare, run, and summarize the pinned MAPLE Route-2 FreeSolv benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core import (  # noqa: E402
    ensure_confirmation_lock,
    load_json,
    load_protocol,
    sha256_bytes,
    sha256_file,
    summarize_errors,
    write_json_atomic,
)
import freesolv_dataset as shared_runner  # noqa: E402
from maple.function.calculator.extra_correction.implicit.correction import (  # noqa: E402
    ImplicitSolvationCorrection,
)
from maple.function.calculator.extra_correction.implicit.smd import (  # noqa: E402
    SMDImplicitSolvation,
)
from maple.function.calculator.set_calculator import SetCalculator  # noqa: E402
from maple.function.read.command_control import CommandControl  # noqa: E402
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402


KCAL_PER_HARTREE = 627.5094740631
PUBLIC_ROUTE2_SETTINGS = (
    "#model=macepol-m",
    "#sp",
    "#solv(implicit=water,method=smd,response=scf,"
    "standard_state=1m,experimental=true)",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def prepare(args: argparse.Namespace) -> None:
    """Reuse the hash-pinned FreeSolv preparation and partition machinery."""

    shared_runner.prepare(args)
    work_dir = Path(args.work_dir).resolve()
    manifest_path = work_dir / "prepared.json"
    manifest = load_json(manifest_path)
    retained: list[dict[str, Any]] = []
    exclusions = list(manifest["exclusions"])
    for candidate in manifest["candidates"]:
        mol2_path = work_dir / candidate["mol2_relative_path"]
        atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
        try:
            SMDImplicitSolvation._validate_domain(atoms)
        except ValueError as exc:
            detail = str(exc)
            reason = (
                "outside-mass-domain"
                if "16 to 500 Da" in detail
                else "nonneutral_dataset_structure"
            )
            exclusions.append(
                {
                    "compound_id": candidate["compound_id"],
                    "reason_code": reason,
                    "detail": detail,
                }
            )
        else:
            retained.append(candidate)
    manifest["candidates"] = retained
    manifest["exclusions"] = exclusions
    manifest["candidate_count"] = len(retained)
    manifest["exclusion_count"] = len(exclusions)
    manifest["partition_counts"] = {
        name: sum(candidate["partition"] == name for candidate in retained)
        for name in ("development", "confirmation")
    }
    write_json_atomic(manifest_path, manifest)
    print(
        f"Route-2 domain retained {len(retained)} candidates; "
        f"excluded {len(exclusions)}."
    )


def _load_prepared(work_dir: Path, fingerprint: str) -> dict[str, Any]:
    return shared_runner._load_prepared(work_dir, fingerprint)


def _record_path(work_dir: Path, partition: str, compound_id: str) -> Path:
    return work_dir / "records" / partition / f"{compound_id}.json"


def _sync_device(device: str) -> None:
    if str(device).lower().startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def _public_route2_contract() -> dict[str, Any]:
    """Parse the exact public Route-2 input used by every benchmark record."""

    params = CommandControl.from_settings(list(PUBLIC_ROUTE2_SETTINGS)).as_dict()
    if params.get("task") != "sp":
        raise RuntimeError("The Route-2 benchmark public contract must remain an SP task.")
    return params


def _validate_protocol_contract(
    protocol: dict[str, Any], params: dict[str, Any]
) -> None:
    methods = protocol["methods"]
    solv = params["solv"]
    observed = {
        "model": params["model"],
        "response": solv["response"],
        "standard_state": solv["standard_state"],
        "profile": solv["profile"],
    }
    expected = {
        "model": methods["mace_model_selector"],
        "response": methods["response"],
        "standard_state": "1m",
        "profile": methods["pcmsolver_profile"],
    }
    if observed != expected:
        raise ValueError(
            "The public Route-2 parser contract and benchmark protocol disagree: "
            f"observed={observed}, expected={expected}."
        )


def _model_checkpoint_record() -> dict[str, Any]:
    from mace.calculators.foundations_models import download_mace_polar_checkpoint

    path = Path(download_mace_polar_checkpoint("polar-1-m")).resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _runtime_environment_record(
    protocol: dict[str, Any],
    device: str,
    *,
    mace_dtype: str,
) -> dict[str, Any]:
    mace_version = importlib.metadata.version("mace-torch")
    required = str(protocol["providers"]["mace_polar"]["required_version"])
    if mace_version != required:
        raise ValueError(
            f"Route-2 protocol requires mace-torch {required}; observed {mace_version}."
        )
    library = os.environ.get("PCMSOLVER_LIBRARY")
    library_path = Path(library).expanduser().resolve() if library else None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device,
        "mace_torch_version": mace_version,
        "mace_dtype": mace_dtype,
        "mace_checkpoint": _model_checkpoint_record(),
        "pcmsolver_library": (
            {
                "path": str(library_path),
                "sha256": sha256_file(library_path),
            }
            if library_path is not None and library_path.is_file()
            else {"path": library, "sha256": None}
        ),
        "pcmsolver_python_path": os.environ.get("PCMSOLVER_PYTHON_PATH"),
    }


def _environment_record(
    protocol: dict[str, Any], device: str, calculator
) -> dict[str, Any]:
    return _runtime_environment_record(
        protocol,
        device,
        mace_dtype=str(calculator.dtype),
    )


def _base_record(
    protocol: dict[str, Any],
    fingerprint: str,
    candidate: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": protocol["result_schema_version"],
        "attempt_id": candidate["compound_id"],
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "partition": candidate["partition"],
        "compound_id": candidate["compound_id"],
        "name": candidate["name"],
        "smiles": candidate["smiles"],
        "structure_group_sha256": candidate["structure_group_sha256"],
        "mol2_sha256": candidate["mol2_sha256"],
        "dataset_record_sha256": candidate["dataset_record_sha256"],
        "experimental_kcal_mol": candidate["experimental_kcal_mol"],
        "experimental_reference": candidate["experimental_reference"],
        "method": "official-mace-polar-1-m/smd-iefpcm-water/scf",
        "standard_state": "1M(gas)->1M(solution)",
        "environment": environment,
        "bins": {
            "functional_groups": candidate["functional_groups"],
            "element_class": candidate["element_class"],
            "size": candidate["size_bin"],
            "heteroatom_count": candidate["heteroatom_bin"],
            "flexibility": candidate["flexibility_bin"],
        },
        "recorded_at_utc": utc_now(),
    }


def _select_shard(
    candidates: list[dict[str, Any]],
    *,
    shard_count: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    if shard_count <= 0:
        raise ValueError("--shard-count must be positive.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < shard-count.")
    return candidates[shard_index::shard_count]


def _selected_candidates(
    manifest: dict[str, Any],
    *,
    partition: str,
    shard_count: int,
    shard_index: int,
    max_compounds: int | None,
) -> list[dict[str, Any]]:
    candidates = [
        candidate
        for candidate in manifest["candidates"]
        if candidate["partition"] == partition
    ]
    candidates = _select_shard(
        candidates,
        shard_count=shard_count,
        shard_index=shard_index,
    )
    if max_compounds is not None:
        if max_compounds <= 0:
            raise ValueError("--max-compounds must be positive.")
        candidates = candidates[:max_compounds]
    return candidates


def run(args: argparse.Namespace) -> None:
    protocol, fingerprint = load_protocol(args.protocol)
    if protocol.get("benchmark_kind") != "route2-macepolar-smd":
        raise ValueError("run_route2_freesolv.py requires a route2-macepolar-smd protocol.")
    params = _public_route2_contract()
    _validate_protocol_contract(protocol, params)
    work_dir = Path(args.work_dir).resolve()
    manifest = _load_prepared(work_dir, fingerprint)
    if args.partition == "confirmation":
        ensure_confirmation_lock(work_dir, fingerprint)

    shard_count = int(getattr(args, "shard_count", 1))
    shard_index = int(getattr(args, "shard_index", 0))
    candidates = _selected_candidates(
        manifest,
        partition=args.partition,
        shard_count=shard_count,
        shard_index=shard_index,
        max_compounds=args.max_compounds,
    )
    if shard_count > 1:
        print(
            f"Shard {shard_index + 1}/{shard_count} owns "
            f"{len(candidates)} deterministic candidates."
        )
    if args.max_compounds is not None:
        print("WARNING: max-compounds is an incomplete smoke; summarize will reject it.")

    calculator = None
    environment = None
    completed = 0
    skipped = 0

    for candidate in candidates:
        record_path = _record_path(work_dir, args.partition, candidate["compound_id"])
        if record_path.exists():
            skipped += 1
            continue
        mol2_path = work_dir / candidate["mol2_relative_path"]
        if sha256_file(mol2_path) != candidate["mol2_sha256"]:
            raise ValueError(f"MOL2 changed after preparation: {candidate['compound_id']}.")
        atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
        candidate_output = (
            work_dir / "provider-audit" / candidate["compound_id"] / "maple.out"
        )
        candidate_output.parent.mkdir(parents=True, exist_ok=True)
        audit_dir = Path(str(candidate_output) + ".implicit")

        if calculator is None:
            calculator = SetCalculator(
                args.device,
                params["model"],
                str(candidate_output),
                atoms=atoms,
                d4=bool(params.get("d4", False)),
                implicit=params["solv"]["method"],
                solvent=params["solv"]["implicit"],
                model_options=params.get("model_options"),
                solvation_options=params["solv"],
                charge_options=params.get("charge") or {},
            ).set_calculator()
            if str(calculator.dtype) != "torch.float64":
                raise RuntimeError("Route-2 benchmark requires the float64 MACE-POLAR path.")
            environment = _environment_record(protocol, args.device, calculator)

        assert environment is not None
        record = _base_record(protocol, fingerprint, candidate, environment)

        try:
            if calculator.solvent_correction is None or (
                Path(calculator.solvent_correction.audit_dir) != audit_dir
            ):
                calculator.solvent_correction = ImplicitSolvationCorrection(
                    atoms,
                    params.get("charge") or {},
                    params["solv"],
                    output=str(candidate_output),
                )
            correction = calculator.solvent_correction

            # Isolate the gas-only timing with the same already-loaded official
            # calculator, then restore the public solvent composition.
            calculator.solvent_correction = None
            calculator.reset()
            atoms.calc = calculator
            try:
                _sync_device(args.device)
                started = time.perf_counter()
                gas_reference_hartree = float(atoms.get_potential_energy())
                _sync_device(args.device)
                gas_wall_s = time.perf_counter() - started
            finally:
                calculator.solvent_correction = correction

            # This is the public integrated path: the calculator evaluates the
            # gas model, invokes ImplicitSolvationCorrection, and finalizes the
            # combined ASE result in one call.
            calculator.reset()
            _sync_device(args.device)
            started = time.perf_counter()
            combined_energy_hartree = float(atoms.get_potential_energy())
            _sync_device(args.device)
            route2_wall_s = time.perf_counter() - started

            solvation = calculator.results.get("solvation")
            if not isinstance(solvation, dict):
                raise RuntimeError(
                    "The integrated Route-2 calculator did not publish results['solvation']."
                )
            gas_energy_hartree = float(solvation["gas_energy_hartree"])
            predicted_hartree = float(solvation["delta_g_solv_hartree"])
            finalized_combined = float(solvation["combined_energy_hartree"])
            if abs(gas_reference_hartree - gas_energy_hartree) > 1.0e-8:
                raise RuntimeError(
                    "Gas-only timing energy disagrees with the integrated finalizer "
                    f"by {gas_reference_hartree - gas_energy_hartree:.3e} Hartree."
                )
            if abs(combined_energy_hartree - finalized_combined) > 1.0e-12:
                raise RuntimeError(
                    "ASE combined energy disagrees with the Route-2 finalizer result."
                )
            if abs(
                finalized_combined - (gas_energy_hartree + predicted_hartree)
            ) > 1.0e-12:
                raise RuntimeError(
                    "Route-2 energy composition failed: combined != gas + delta_G_solv."
                )
            if solvation.get("ase_free_energy_is_thermochemical_gibbs") is not False:
                raise RuntimeError(
                    "Route-2 must explicitly mark ASE free_energy as non-thermochemical."
                )

            predicted = predicted_hartree * KCAL_PER_HARTREE
            error = predicted - float(candidate["experimental_kcal_mol"])
            components = {
                key: float(value) * KCAL_PER_HARTREE
                for key, value in solvation["components_hartree"].items()
            }
            record.update(
                status="success",
                gas_energy_hartree=gas_energy_hartree,
                combined_energy_hartree=finalized_combined,
                predicted_kcal_mol=predicted,
                signed_error_kcal_mol=error,
                absolute_error_kcal_mol=abs(error),
                components_kcal_mol=components,
                provider_provenance=solvation["provenance"],
                public_input=list(PUBLIC_ROUTE2_SETTINGS),
                ase_free_energy_is_thermochemical_gibbs=False,
                timing_seconds={
                    "gas_mace": gas_wall_s,
                    "route2_integrated": route2_wall_s,
                    "route2_total": route2_wall_s,
                    "total_over_gas": route2_wall_s / gas_wall_s,
                    "model_load_excluded": True,
                },
                audit_directory=str(audit_dir),
            )
        except Exception as exc:
            record.update(
                status="failure",
                failure={
                    "phase": "route2",
                    "exception_class": type(exc).__name__,
                    "reason": str(exc),
                    "audit_directory": str(audit_dir),
                },
            )
        write_json_atomic(record_path, record)
        completed += 1

    print(f"Wrote {completed} Route-2 records; resumed/skipped {skipped}.")


def _fatal_error_excerpt(log_text: str) -> str | None:
    marker = "PCMSolver fatal error."
    offset = log_text.rfind(marker)
    if offset < 0:
        return None
    lines = log_text[offset:].splitlines()
    return "\n".join(lines[:8])


def _write_supervised_provider_failure(
    *,
    protocol: dict[str, Any],
    fingerprint: str,
    work_dir: Path,
    partition: str,
    candidate: dict[str, Any],
    device: str,
    returncode: int,
    log_path: Path,
    fatal_excerpt: str,
) -> None:
    record_path = _record_path(work_dir, partition, candidate["compound_id"])
    if record_path.exists():
        raise FileExistsError(f"Refusing to replace existing record: {record_path}.")
    audit_dir = (
        work_dir
        / "provider-audit"
        / candidate["compound_id"]
        / "maple.out.implicit"
    )
    if not audit_dir.is_dir():
        raise RuntimeError(
            "The failed worker did not create a PCMSolver audit directory; "
            "refusing to classify the process exit as a provider failure."
        )
    environment = _runtime_environment_record(
        protocol,
        device,
        mace_dtype=f"torch.{protocol['methods']['mace_default_dtype']}",
    )
    record = _base_record(protocol, fingerprint, candidate, environment)
    record.update(
        status="failure",
        failure={
            "phase": "pcmsolver-process",
            "exception_class": "PCMSolverFatalProcessExit",
            "reason": (
                f"PCMSolver terminated the isolated worker with return code "
                f"{returncode}.\n{fatal_excerpt}"
            ),
            "audit_directory": str(audit_dir),
            "supervisor_log": str(log_path),
        },
    )
    write_json_atomic(record_path, record)


def run_supervised(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    work_dir = Path(args.work_dir).resolve()
    protocol, fingerprint = load_protocol(protocol_path)
    if protocol.get("benchmark_kind") != "route2-macepolar-smd":
        raise ValueError(
            "run_route2_freesolv.py requires a route2-macepolar-smd protocol."
        )
    _validate_protocol_contract(protocol, _public_route2_contract())
    manifest = _load_prepared(work_dir, fingerprint)
    if args.partition == "confirmation":
        ensure_confirmation_lock(work_dir, fingerprint)

    shard_count = int(getattr(args, "shard_count", 1))
    shard_index = int(getattr(args, "shard_index", 0))
    candidates = _selected_candidates(
        manifest,
        partition=args.partition,
        shard_count=shard_count,
        shard_index=shard_index,
        max_compounds=args.max_compounds,
    )
    logs_dir = work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    attempt = 0

    while True:
        missing = [
            candidate
            for candidate in candidates
            if not _record_path(
                work_dir, args.partition, candidate["compound_id"]
            ).exists()
        ]
        if not missing:
            print(
                f"Supervised shard {shard_index + 1}/{shard_count} is complete."
            )
            return

        attempt += 1
        log_path = logs_dir / (
            f"supervised-{args.partition}-shard-{shard_index:03d}"
            f"-attempt-{attempt:03d}.log"
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run",
            "--protocol",
            str(protocol_path),
            "--work-dir",
            str(work_dir),
            "--partition",
            args.partition,
            "--device",
            args.device,
            "--shard-count",
            str(shard_count),
            "--shard-index",
            str(shard_index),
        ]
        if args.max_compounds is not None:
            command.extend(["--max-compounds", str(args.max_compounds)])

        with log_path.open("wb") as log_handle:
            result = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode == 0:
            continue

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        fatal_excerpt = _fatal_error_excerpt(log_text)
        if fatal_excerpt is None:
            raise RuntimeError(
                f"Route-2 worker exited with return code {result.returncode} "
                f"without a PCMSolver fatal marker; see {log_path}."
            )
        candidate = next(
            candidate
            for candidate in candidates
            if not _record_path(
                work_dir, args.partition, candidate["compound_id"]
            ).exists()
        )
        _write_supervised_provider_failure(
            protocol=protocol,
            fingerprint=fingerprint,
            work_dir=work_dir,
            partition=args.partition,
            candidate=candidate,
            device=args.device,
            returncode=result.returncode,
            log_path=log_path,
            fatal_excerpt=fatal_excerpt,
        )
        print(
            f"Recorded fatal PCMSolver provider failure for "
            f"{candidate['compound_id']}; restarting the shard."
        )


def _seed_for(base_seed: int, key: str) -> int:
    return (base_seed + int(sha256_bytes(key.encode("utf-8"))[:8], 16)) % (2**32)


def _metrics(
    records: list[dict[str, Any]],
    *,
    expected_count: int,
    statistics: dict[str, Any],
    seed_key: str,
) -> dict[str, Any]:
    errors = [
        float(record["signed_error_kcal_mol"])
        for record in records
        if record["status"] == "success"
    ]
    return summarize_errors(
        errors,
        expected_count=expected_count,
        resamples=int(statistics["bootstrap_resamples"]),
        confidence=float(statistics["bootstrap_confidence"]),
        seed=_seed_for(int(statistics["bootstrap_seed"]), seed_key),
    )


def summarize(args: argparse.Namespace) -> None:
    protocol, fingerprint = load_protocol(args.protocol)
    work_dir = Path(args.work_dir).resolve()
    manifest = _load_prepared(work_dir, fingerprint)
    confirmation_lock = None
    if args.partition == "confirmation":
        confirmation_lock = ensure_confirmation_lock(work_dir, fingerprint)

    candidates = {
        candidate["compound_id"]: candidate
        for candidate in manifest["candidates"]
        if candidate["partition"] == args.partition
    }
    record_dir = work_dir / "records" / args.partition
    expected = sorted(candidates)
    present = sorted(path.stem for path in record_dir.glob("*.json")) if record_dir.exists() else []
    if present != expected:
        missing = sorted(set(expected) - set(present))
        extra = sorted(set(present) - set(expected))
        raise ValueError(
            f"Attempt reconciliation failed: missing={len(missing)}, extra={len(extra)}."
        )

    records: list[dict[str, Any]] = []
    record_hashes: dict[str, str] = {}
    for compound_id in expected:
        path = record_dir / f"{compound_id}.json"
        record = load_json(path)
        if (
            record.get("attempt_id") != compound_id
            or record.get("protocol_fingerprint") != fingerprint
            or record.get("partition") != args.partition
            or record.get("status") not in {"success", "failure"}
        ):
            raise ValueError(f"Record identity/protocol/status mismatch: {path}.")
        records.append(record)
        record_hashes[compound_id] = sha256_file(path)

    statistics = protocol["statistics"]
    overall = _metrics(
        records,
        expected_count=len(expected),
        statistics=statistics,
        seed_key=f"{args.partition}/route2/overall",
    )
    successful = [record for record in records if record["status"] == "success"]
    ratios = np.asarray(
        [record["timing_seconds"]["total_over_gas"] for record in successful],
        dtype=float,
    )
    timings = {
        "n": int(ratios.size),
        "median_total_over_gas": float(np.median(ratios)) if ratios.size else None,
        "p90_total_over_gas": float(np.quantile(ratios, 0.9)) if ratios.size else None,
        "median_gas_mace_seconds": (
            float(np.median([r["timing_seconds"]["gas_mace"] for r in successful]))
            if successful
            else None
        ),
        "median_route2_total_seconds": (
            float(np.median([r["timing_seconds"]["route2_total"] for r in successful]))
            if successful
            else None
        ),
    }

    dimensions = {
        "functional_group": "functional_groups",
        "element_class": "element_class",
        "size": "size_bin",
        "heteroatom_count": "heteroatom_bin",
        "flexibility": "flexibility_bin",
    }
    strata: dict[str, Any] = {}
    for dimension, candidate_key in dimensions.items():
        labels = sorted(
            {
                label
                for candidate in candidates.values()
                for label in (
                    candidate[candidate_key]
                    if isinstance(candidate[candidate_key], list)
                    else [candidate[candidate_key]]
                )
            }
        )
        strata[dimension] = {}
        for label in labels:
            member_ids = {
                compound_id
                for compound_id, candidate in candidates.items()
                if label
                in (
                    candidate[candidate_key]
                    if isinstance(candidate[candidate_key], list)
                    else [candidate[candidate_key]]
                )
            }
            member_records = [
                record for record in records if record["compound_id"] in member_ids
            ]
            strata[dimension][label] = _metrics(
                member_records,
                expected_count=len(member_ids),
                statistics=statistics,
                seed_key=f"{args.partition}/route2/{dimension}/{label}",
            )

    accuracy_pass = (
        overall["failure_rate"] == 0.0
        and overall["mae"] is not None
        and overall["mae"] <= 1.5
    )
    runtime_pass = (
        timings["median_total_over_gas"] is not None
        and timings["median_total_over_gas"] <= 2.0
    )
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "partition": args.partition,
        "candidate_count": len(expected),
        "overall": overall,
        "timing": timings,
        "strata": strata,
        "predeclared_gates": {
            "accuracy": {
                "rule": protocol["confirmation"]["predeclared_accuracy_gate"],
                "passed_on_this_partition": accuracy_pass,
            },
            "runtime": {
                "rule": protocol["confirmation"]["predeclared_runtime_gate"],
                "passed_on_this_partition": runtime_pass,
            },
            "scientifically_certified": (
                args.partition == "confirmation" and accuracy_pass and runtime_pass
            ),
        },
        "confirmation_lock": confirmation_lock,
        "failures": [
            {
                "compound_id": record["compound_id"],
                **record["failure"],
            }
            for record in records
            if record["status"] == "failure"
        ],
        "record_sha256": record_hashes,
    }
    write_json_atomic(args.output, summary)
    print(
        f"Summarized {len(expected)} Route-2 records: "
        f"MAE={overall['mae']}, failure_rate={overall['failure_rate']}, "
        f"median total/gas={timings['median_total_over_gas']}."
    )


def freeze_confirmation(args: argparse.Namespace) -> None:
    shared_runner.freeze_confirmation(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--protocol", required=True)
    prepare_parser.add_argument("--work-dir", required=True)
    prepare_parser.add_argument("--source-dir")
    prepare_parser.set_defaults(func=prepare)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--protocol", required=True)
    run_parser.add_argument("--work-dir", required=True)
    run_parser.add_argument("--partition", choices=("development", "confirmation"), required=True)
    run_parser.add_argument("--device", default="cpu")
    run_parser.add_argument("--max-compounds", type=int)
    run_parser.add_argument("--shard-count", type=int, default=1)
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.set_defaults(func=run)

    supervised_parser = subparsers.add_parser("run-supervised")
    supervised_parser.add_argument("--protocol", required=True)
    supervised_parser.add_argument("--work-dir", required=True)
    supervised_parser.add_argument(
        "--partition", choices=("development", "confirmation"), required=True
    )
    supervised_parser.add_argument("--device", default="cpu")
    supervised_parser.add_argument("--max-compounds", type=int)
    supervised_parser.add_argument("--shard-count", type=int, default=1)
    supervised_parser.add_argument("--shard-index", type=int, default=0)
    supervised_parser.set_defaults(func=run_supervised)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--protocol", required=True)
    summary_parser.add_argument("--work-dir", required=True)
    summary_parser.add_argument("--partition", choices=("development", "confirmation"), required=True)
    summary_parser.add_argument("--output", required=True)
    summary_parser.set_defaults(func=summarize)

    freeze_parser = subparsers.add_parser("freeze-confirmation")
    freeze_parser.add_argument("--protocol", required=True)
    freeze_parser.add_argument("--work-dir", required=True)
    freeze_parser.add_argument("--proposed-default", required=True)
    freeze_parser.add_argument("--pass-rule", required=True)
    freeze_parser.set_defaults(func=freeze_confirmation)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
