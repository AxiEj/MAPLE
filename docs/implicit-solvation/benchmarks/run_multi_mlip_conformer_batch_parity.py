#!/usr/bin/env python3
"""Benchmark serial and model-native batched Route 1 conformer evaluation."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

from ase import Atoms
from ase.calculators.calculator import all_changes
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core import (  # noqa: E402
    canonical_json_bytes,
    command_provenance,
    load_json,
    seal_artifact,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
import run_multi_mlip_discrete_conformers as source_runner  # noqa: E402

from maple.function.calculator.set_calculator import SetCalculator  # noqa: E402
from maple.function.free_energy import (  # noqa: E402
    analyze_discrete_conformer_ensemble,
    evaluate_gas_conformer_energies,
)

KCAL_PER_HARTREE = source_runner.KCAL_PER_HARTREE
GPU_PROCESS_QUERY = (
    "nvidia-smi",
    "--query-compute-apps=pid,process_name",
    "--format=csv,noheader,nounits",
)


def _relative_to_repository(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not np.isfinite(value) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return float(value)


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _parse_gpu_compute_processes(stdout: str) -> list[dict[str, Any]]:
    processes = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",", maxsplit=1)]
        if len(fields) != 2:
            raise RuntimeError(f"Unexpected nvidia-smi process row: {line!r}.")
        try:
            pid = int(fields[0])
        except ValueError as exc:
            raise RuntimeError(
                f"Unexpected nvidia-smi process PID: {fields[0]!r}."
            ) from exc
        processes.append({"pid": pid, "process_name": fields[1]})
    return processes


def _exclusive_gpu_snapshot(stage: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            GPU_PROCESS_QUERY,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "GPU endpoint screening requires a successful nvidia-smi "
            "compute-process query."
        ) from exc
    observed = _parse_gpu_compute_processes(completed.stdout)
    competing = [process for process in observed if process["pid"] != os.getpid()]
    if competing:
        raise RuntimeError(
            f"GPU endpoint screening {stage} found competing compute processes: "
            f"{competing}."
        )
    return {
        "stage": stage,
        "query": list(GPU_PROCESS_QUERY),
        "runner_pid": os.getpid(),
        "observed_compute_processes": observed,
        "competing_compute_processes": competing,
        "passed": True,
    }


def _timing_preflight(max_load_per_logical_cpu: float) -> dict[str, Any]:
    gpu_preflight = _exclusive_gpu_snapshot("preflight")
    logical_cpu_count = os.cpu_count()
    if logical_cpu_count is None or logical_cpu_count < 1:
        raise RuntimeError("Cannot determine the logical CPU count for timing.")
    load_1m, load_5m, load_15m = os.getloadavg()
    load_per_logical_cpu = load_1m / logical_cpu_count
    if load_per_logical_cpu > max_load_per_logical_cpu:
        raise RuntimeError(
            "Exclusive timing preflight found excessive host load: "
            f"{load_per_logical_cpu:.6f} per logical CPU exceeds "
            f"{max_load_per_logical_cpu:.6f}."
        )
    return {
        "required": True,
        "evidence_scope": "preflight_and_postflight_snapshots_only",
        "continuous_host_isolation_monitored": False,
        "continuous_gpu_isolation_monitored": False,
        "whole_run_gpu_exclusivity_proven": False,
        "gpu_preflight": gpu_preflight,
        "host_preflight": {
            "logical_cpu_count": logical_cpu_count,
            "load_average_1m": load_1m,
            "load_average_5m": load_5m,
            "load_average_15m": load_15m,
            "load_per_logical_cpu": load_per_logical_cpu,
            "maximum_load_per_logical_cpu": max_load_per_logical_cpu,
            "passed": True,
        },
        "endpoint_snapshots_passed": None,
    }


def _seal_and_write_after_gpu_postflight(
    output: str | Path,
    artifact: dict[str, Any],
) -> None:
    timing_isolation = artifact["timing_isolation"]
    timing_isolation["gpu_postflight"] = _exclusive_gpu_snapshot("postflight")
    timing_isolation["endpoint_snapshots_passed"] = True
    seal_artifact(artifact)
    write_json_atomic(output, artifact)


def load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol_path = Path(path).resolve()
    protocol = load_json(protocol_path)
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise ValueError("Unsupported batch-parity protocol schema.")
    if protocol.get("source_partition") != "development":
        raise ValueError("Batch-parity evidence must remain development-only.")

    route = protocol["route"]
    if (
        route.get("gas_phase_mm_energy") is not False
        or route.get("hydration_label_residual") is not False
        or route.get("mlip_retraining") is not False
    ):
        raise ValueError("Batch-parity protocol violates the Route 1 contract.")

    source = protocol["source"]
    source_protocol_path = protocol_path.parent / source["protocol"]
    source_manifest_path = protocol_path.parent / source["manifest"]
    if sha256_file(source_protocol_path) != source["protocol_file_sha256"]:
        raise ValueError("Frozen source protocol hash changed.")
    if sha256_file(source_manifest_path) != source["manifest_file_sha256"]:
        raise ValueError("Frozen source manifest file hash changed.")
    source_protocol, source_fingerprint = source_runner.load_protocol(
        source_protocol_path
    )
    source_manifest = source_runner._load_manifest(
        source_manifest_path,
        protocol=source_protocol,
        fingerprint=source_fingerprint,
    )
    if source_manifest["content_sha256"] != source["manifest_content_sha256"]:
        raise ValueError("Frozen source manifest content hash changed.")
    if [model["name"] for model in source_protocol["models"]] != source[
        "expected_models"
    ]:
        raise ValueError("Frozen source model list changed.")
    if len(source_manifest["cases"]) != source["expected_case_count"]:
        raise ValueError("Frozen source case count changed.")
    if (
        sum(case["state_count"] for case in source_manifest["cases"])
        != source["expected_state_count"]
    ):
        raise ValueError("Frozen source state count changed.")

    execution = protocol["execution"]
    _positive_integer("batch_size", execution["batch_size"])
    repeat_count = _positive_integer("repeat_count", execution["repeat_count"])
    _positive_integer("warmup_state_count", execution["warmup_state_count"])
    _positive_float(
        "maximum_preflight_load_per_logical_cpu",
        execution["maximum_preflight_load_per_logical_cpu"],
    )
    if execution["properties"] != ["energy"]:
        raise ValueError("This parity protocol supports energy-only evaluation.")
    if execution.get("require_gpu_endpoint_snapshots") is not True:
        raise ValueError("Batch timing requires GPU endpoint snapshots.")
    order = execution["balanced_order"]
    if len(order) != repeat_count or any(
        sorted(repeat) != ["batch", "serial"] for repeat in order
    ):
        raise ValueError("Each repeat must evaluate serial and batch exactly once.")
    serial_first_count = sum(repeat == ["serial", "batch"] for repeat in order)
    batch_first_count = sum(repeat == ["batch", "serial"] for repeat in order)
    if repeat_count % 2 or serial_first_count != batch_first_count:
        raise ValueError(
            "Balanced order must use serial-first and batch-first equally often."
        )

    gates = protocol["gates"]
    for key in (
        "maximum_absolute_energy_difference_hartree",
        "maximum_relative_energy_difference_kcal_mol",
        "maximum_delta_g_difference_kcal_mol",
        "minimum_paired_repeat_speedup",
    ):
        _positive_float(key, gates[key])
    claim = protocol["claim_boundary"]
    if (
        claim.get("batch_vs_serial_same_mlip_only") is not True
        or claim.get("faster_than_bare_mm_claim") is not False
        or claim.get("solvent_accuracy_promotion") is not False
        or claim.get("hydration_free_energy_claim") is not False
        or claim.get("fixed_charge_solvent_functional_changed") is not False
        or claim.get("public_solvfe_eligible") is not False
    ):
        raise ValueError("Batch-parity claim boundary is not fail-closed.")

    return protocol, sha256_bytes(canonical_json_bytes(protocol))


def _load_source(
    protocol_path: Path,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    source = protocol["source"]
    source_protocol_path = protocol_path.parent / source["protocol"]
    source_manifest_path = protocol_path.parent / source["manifest"]
    source_protocol, source_fingerprint = source_runner.load_protocol(
        source_protocol_path
    )
    source_manifest = source_runner._load_manifest(
        source_manifest_path,
        protocol=source_protocol,
        fingerprint=source_fingerprint,
    )
    return (
        source_protocol,
        source_manifest,
        source_protocol_path,
        source_manifest_path,
    )


def _load_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    loaded = []
    for case in manifest["cases"]:
        positions = source_runner._load_case_positions(case)
        template = Atoms(
            numbers=case["atomic_numbers"],
            positions=positions[0],
        )
        template.info.update({"charge": 0, "mult": 1})
        loaded.append(
            {
                "metadata": case,
                "positions": positions,
                "template": template,
            }
        )
    return loaded


def _synchronize(device) -> None:
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)


def _evaluate_serial(calculator, cases, device) -> tuple[dict[str, list[float]], float]:
    output: dict[str, list[float]] = {}
    _synchronize(device)
    start = time.perf_counter()
    for case in cases:
        metadata = case["metadata"]
        atoms = case["template"].copy()
        energies = []
        for positions in case["positions"]:
            atoms.set_positions(positions)
            calculator.calculate(
                atoms,
                properties=["energy"],
                system_changes=all_changes,
            )
            energies.append(float(calculator.results["energy"]))
        output[metadata["compound_id"]] = energies
    _synchronize(device)
    return output, time.perf_counter() - start


def _evaluate_batch(
    calculator,
    cases,
    device,
    *,
    batch_size: int,
) -> tuple[dict[str, list[float]], float]:
    output: dict[str, list[float]] = {}
    _synchronize(device)
    start = time.perf_counter()
    for case in cases:
        metadata = case["metadata"]
        values = evaluate_gas_conformer_energies(
            calculator,
            case["template"],
            case["positions"],
            batch_size=batch_size,
        )
        output[metadata["compound_id"]] = values.tolist()
    _synchronize(device)
    return output, time.perf_counter() - start


def _select_warmup_structures(cases: list[dict[str, Any]], count: int) -> list[Atoms]:
    """Select exactly ``count`` states in deterministic manifest order."""
    warmup_atoms = []
    for case in cases:
        for positions in case["positions"]:
            atoms = case["template"].copy()
            atoms.set_positions(
                np.asarray(positions, dtype=np.float64).tolist(),
                apply_constraint=False,
            )
            warmup_atoms.append(atoms)
            if len(warmup_atoms) == count:
                return warmup_atoms
    raise RuntimeError(
        f"Requested {count} warmup states but only found {len(warmup_atoms)}."
    )


def _relative_energies(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return (array - float(np.min(array))) * KCAL_PER_HARTREE


def _delta_g(
    gas_relative_kcal_mol: np.ndarray,
    case: dict[str, Any],
    estimator: dict[str, Any],
) -> float:
    analysis = analyze_discrete_conformer_ensemble(
        gas_energy_kcal_mol=gas_relative_kcal_mol.tolist(),
        solvent_correction_kcal_mol=case["solvent_correction_kcal_mol"],
        temperature_kelvin=estimator["temperature_kelvin"],
        minimum_effective_conformer_count=estimator[
            "minimum_effective_conformer_count"
        ],
        maximum_dominant_weight=estimator["maximum_dominant_weight"],
        minimum_distribution_overlap=estimator["minimum_distribution_overlap"],
    )
    return float(analysis["delta_g_discrete_kcal_mol"])


def _source_hashes(calculator) -> dict[str, dict[str, str]]:
    calculator_source = inspect.getsourcefile(type(calculator))
    if calculator_source is None:
        raise RuntimeError(
            f"Cannot locate source for calculator type {type(calculator).__qualname__}."
        )
    paths = {
        "calculator": Path(calculator_source).resolve(),
        "set_calculator": (
            REPOSITORY_ROOT / "maple/function/calculator/set_calculator.py"
        ),
        "calculator_base": (
            REPOSITORY_ROOT / "maple/function/calculator/calculator_base.py"
        ),
        "batch_types": (REPOSITORY_ROOT / "maple/function/calculator/_batch_types.py"),
        "batch_utils": (REPOSITORY_ROOT / "maple/function/calculator/_batch_utils.py"),
        "conformer_evaluation": (
            REPOSITORY_ROOT / "maple/function/free_energy/conformer_evaluation.py"
        ),
        "discrete_conformers": (
            REPOSITORY_ROOT / "maple/function/free_energy/discrete_conformers.py"
        ),
        "source_runner": (
            REPOSITORY_ROOT
            / "docs/implicit-solvation/benchmarks/run_multi_mlip_discrete_conformers.py"
        ),
        "benchmark_core": (
            REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/benchmark_core.py"
        ),
    }
    if type(calculator).__name__ == "MACECalculator":
        paths["mace_batch_graph"] = (
            REPOSITORY_ROOT / "maple/function/calculator/mace/_batch_graph.py"
        )
        paths["mace_common"] = (
            REPOSITORY_ROOT / "maple/function/calculator/mace/_common.py"
        )
    return {
        name: {
            "path": _relative_to_repository(path),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }


def _model_environment(model, calculator, device, load_seconds) -> dict[str, Any]:
    import torch

    cutoff = getattr(calculator, "r_max", getattr(calculator, "cutoff", None))
    dtype = getattr(calculator, "dtype", None)
    return {
        "name": model["name"],
        "calculator_class": (
            f"{type(calculator).__module__}.{type(calculator).__qualname__}"
        ),
        "native_batch_capability": bool(calculator.supports_batch_energy_forces),
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "ase_version": importlib.metadata.version("ase"),
        "numpy_version": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "calculator_dtype": str(dtype) if dtype is not None else None,
        "calculator_cutoff_angstrom": (float(cutoff) if cutoff is not None else None),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "checkpoint_sha256": sha256_file(source_runner._checkpoint_path(model)),
        "load_seconds": load_seconds,
        "source_hashes": _source_hashes(calculator),
    }


PARITY_GATE_NAMES = (
    "all_model_case_records_present",
    "all_models_use_native_batch",
    "absolute_energy_parity_within_limit",
    "relative_energy_parity_within_limit",
    "discrete_delta_g_parity_within_limit",
)


def _engineering_decisions(
    model_summaries: dict[str, dict[str, Any]],
    gates: dict[str, float],
    *,
    all_model_case_records_present: bool,
    all_models_use_native_batch: bool,
) -> dict[str, Any]:
    """Apply correctness gates before admitting interface or speed claims."""
    max_absolute = max(
        summary["maximum_absolute_energy_difference_hartree"]
        for summary in model_summaries.values()
    )
    max_relative = max(
        summary["maximum_relative_energy_difference_kcal_mol"]
        for summary in model_summaries.values()
    )
    max_delta_g = max(
        summary["maximum_delta_g_difference_kcal_mol"]
        for summary in model_summaries.values()
    )
    min_speedup = min(summary["speedup"] for summary in model_summaries.values())
    min_paired_repeat_speedup = min(
        min(summary["speedup_by_repeat"]) for summary in model_summaries.values()
    )
    engineering_gates = {
        "all_model_case_records_present": all_model_case_records_present,
        "all_models_use_native_batch": all_models_use_native_batch,
        "absolute_energy_parity_within_limit": (
            max_absolute <= gates["maximum_absolute_energy_difference_hartree"]
        ),
        "relative_energy_parity_within_limit": (
            max_relative <= gates["maximum_relative_energy_difference_kcal_mol"]
        ),
        "discrete_delta_g_parity_within_limit": (
            max_delta_g <= gates["maximum_delta_g_difference_kcal_mol"]
        ),
        "every_paired_repeat_meets_speedup_floor": (
            min_paired_repeat_speedup >= gates["minimum_paired_repeat_speedup"]
        ),
    }
    batch_interface_admission_allowed = all(
        engineering_gates[name] for name in PARITY_GATE_NAMES
    )
    per_model_material_speedup = {
        name: (
            summary["maximum_absolute_energy_difference_hartree"]
            <= gates["maximum_absolute_energy_difference_hartree"]
            and summary["maximum_relative_energy_difference_kcal_mol"]
            <= gates["maximum_relative_energy_difference_kcal_mol"]
            and summary["maximum_delta_g_difference_kcal_mol"]
            <= gates["maximum_delta_g_difference_kcal_mol"]
            and min(summary["speedup_by_repeat"])
            >= gates["minimum_paired_repeat_speedup"]
        )
        for name, summary in model_summaries.items()
    }
    return {
        "observed": {
            "maximum_absolute_energy_difference_hartree": max_absolute,
            "maximum_relative_energy_difference_kcal_mol": max_relative,
            "maximum_delta_g_difference_kcal_mol": max_delta_g,
            "minimum_per_model_speedup": min_speedup,
            "minimum_paired_repeat_speedup": min_paired_repeat_speedup,
        },
        "engineering_gates": engineering_gates,
        "batch_interface_admission_allowed": batch_interface_admission_allowed,
        "universal_material_speedup_claim_allowed": (
            batch_interface_admission_allowed
            and engineering_gates["every_paired_repeat_meets_speedup_floor"]
        ),
        "per_model_material_speedup": per_model_material_speedup,
    }


def run(args: argparse.Namespace) -> None:
    import torch

    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_protocol(protocol_path)
    (
        source_protocol,
        manifest,
        source_protocol_path,
        source_manifest_path,
    ) = _load_source(protocol_path, protocol)
    cases = _load_cases(manifest)
    device = torch.device(protocol["execution"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Protocol requires CUDA but CUDA is unavailable.")
    timing_isolation = _timing_preflight(
        protocol["execution"]["maximum_preflight_load_per_logical_cpu"]
    )

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    first_atoms = cases[0]["template"]
    repeats = int(protocol["execution"]["repeat_count"])
    batch_size = int(protocol["execution"]["batch_size"])
    warmup_count = int(protocol["execution"]["warmup_state_count"])
    estimator = source_protocol["estimator"]

    records = []
    environments = {}
    model_summaries = {}
    for model in source_protocol["models"]:
        name = model["name"]
        print(f"Loading {name} on {device}.", flush=True)
        start = time.perf_counter()
        calculator = SetCalculator(
            device,
            name,
            str(work_dir / f"{name}.log"),
            atoms=first_atoms,
        ).set_calculator()
        load_seconds = time.perf_counter() - start
        if not calculator.supports_batch_energy_forces:
            raise RuntimeError(f"{name} does not expose a native batch path.")
        environments[name] = _model_environment(
            model,
            calculator,
            device,
            load_seconds,
        )

        warmup_structures = _select_warmup_structures(cases, warmup_count)
        for warmup_atoms in warmup_structures:
            calculator.calculate(
                warmup_atoms,
                properties=["energy"],
                system_changes=all_changes,
            )
        calculator.calculate_many(
            warmup_structures,
            properties=("energy",),
        )
        _synchronize(device)

        evaluations: dict[str, list[dict[str, Any]]] = {
            "serial": [],
            "batch": [],
        }
        for repeat_index, order in enumerate(protocol["execution"]["balanced_order"]):
            for mode in order:
                if mode == "serial":
                    energies, seconds = _evaluate_serial(
                        calculator,
                        cases,
                        device,
                    )
                else:
                    energies, seconds = _evaluate_batch(
                        calculator,
                        cases,
                        device,
                        batch_size=batch_size,
                    )
                if not math.isfinite(seconds) or seconds <= 0.0:
                    raise RuntimeError(f"Invalid {name}/{mode} timing.")
                evaluations[mode].append(
                    {
                        "seconds": seconds,
                        "energies": energies,
                    }
                )
                print(
                    f"{name} repeat {repeat_index + 1}/{repeats} "
                    f"{mode}: {seconds:.3f} s",
                    flush=True,
                )

        model_records = []
        for case in manifest["cases"]:
            compound_id = case["compound_id"]
            serial_repeats = [
                evaluation["energies"][compound_id]
                for evaluation in evaluations["serial"]
            ]
            batch_repeats = [
                evaluation["energies"][compound_id]
                for evaluation in evaluations["batch"]
            ]
            maximum_absolute_energy_difference = 0.0
            maximum_relative_energy_difference = 0.0
            maximum_delta_g_difference = 0.0
            serial_delta_g = []
            batch_delta_g = []
            for serial_values, batch_values in zip(
                serial_repeats,
                batch_repeats,
                strict=True,
            ):
                serial_array = np.asarray(serial_values, dtype=np.float64)
                batch_array = np.asarray(batch_values, dtype=np.float64)
                if (
                    serial_array.shape != batch_array.shape
                    or serial_array.shape != (case["state_count"],)
                    or not np.all(np.isfinite(serial_array))
                    or not np.all(np.isfinite(batch_array))
                ):
                    raise RuntimeError(
                        f"Invalid energy arrays for {name}/{compound_id}."
                    )
                serial_relative = _relative_energies(serial_values)
                batch_relative = _relative_energies(batch_values)
                serial_value = _delta_g(serial_relative, case, estimator)
                batch_value = _delta_g(batch_relative, case, estimator)
                serial_delta_g.append(serial_value)
                batch_delta_g.append(batch_value)
                maximum_absolute_energy_difference = max(
                    maximum_absolute_energy_difference,
                    float(np.max(np.abs(serial_array - batch_array))),
                )
                maximum_relative_energy_difference = max(
                    maximum_relative_energy_difference,
                    float(np.max(np.abs(serial_relative - batch_relative))),
                )
                maximum_delta_g_difference = max(
                    maximum_delta_g_difference,
                    abs(serial_value - batch_value),
                )

            record = {
                "model": name,
                "compound_id": compound_id,
                "state_count": case["state_count"],
                "state_file_sha256": case["state_file_sha256"],
                "solvent_correction_sha256": case["solvent_correction_sha256"],
                "serial_energy_hartree_by_repeat": serial_repeats,
                "batch_energy_hartree_by_repeat": batch_repeats,
                "serial_delta_g_kcal_mol_by_repeat": serial_delta_g,
                "batch_delta_g_kcal_mol_by_repeat": batch_delta_g,
                "maximum_absolute_energy_difference_hartree": (
                    maximum_absolute_energy_difference
                ),
                "maximum_relative_energy_difference_kcal_mol": (
                    maximum_relative_energy_difference
                ),
                "maximum_delta_g_difference_kcal_mol": (maximum_delta_g_difference),
            }
            records.append(record)
            model_records.append(record)

        serial_seconds = [
            float(evaluation["seconds"]) for evaluation in evaluations["serial"]
        ]
        batch_seconds = [
            float(evaluation["seconds"]) for evaluation in evaluations["batch"]
        ]
        serial_median = float(np.median(serial_seconds))
        batch_median = float(np.median(batch_seconds))
        speedup = serial_median / batch_median
        state_count = sum(record["state_count"] for record in model_records)
        model_summaries[name] = {
            "case_count": len(model_records),
            "state_count": state_count,
            "batch_size": batch_size,
            "serial_seconds_by_repeat": serial_seconds,
            "batch_seconds_by_repeat": batch_seconds,
            "serial_median_seconds": serial_median,
            "batch_median_seconds": batch_median,
            "serial_states_per_second": state_count / serial_median,
            "batch_states_per_second": state_count / batch_median,
            "speedup_by_repeat": [
                serial / batch
                for serial, batch in zip(
                    serial_seconds,
                    batch_seconds,
                    strict=True,
                )
            ],
            "speedup": speedup,
            "maximum_absolute_energy_difference_hartree": max(
                record["maximum_absolute_energy_difference_hartree"]
                for record in model_records
            ),
            "maximum_relative_energy_difference_kcal_mol": max(
                record["maximum_relative_energy_difference_kcal_mol"]
                for record in model_records
            ),
            "maximum_delta_g_difference_kcal_mol": max(
                record["maximum_delta_g_difference_kcal_mol"]
                for record in model_records
            ),
        }

    gates = protocol["gates"]
    decisions = _engineering_decisions(
        model_summaries,
        gates,
        all_model_case_records_present=(
            len(records) == len(source_protocol["models"]) * len(manifest["cases"])
        ),
        all_models_use_native_batch=all(
            environment["native_batch_capability"]
            for environment in environments.values()
        ),
    )
    artifact = {
        "schema_version": 1,
        "artifact_type": "route1-multi-mlip-conformer-batch-parity",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "source_partition": "development",
        "source_protocol": _relative_to_repository(source_protocol_path),
        "source_protocol_file_sha256": sha256_file(source_protocol_path),
        "source_manifest": _relative_to_repository(source_manifest_path),
        "source_manifest_file_sha256": sha256_file(source_manifest_path),
        "source_manifest_content_sha256": manifest["content_sha256"],
        "route": protocol["route"],
        "selection": {
            "case_count": len(manifest["cases"]),
            "state_count": sum(case["state_count"] for case in manifest["cases"]),
            "labels_used": False,
        },
        "execution": protocol["execution"],
        "timing_isolation": timing_isolation,
        "declared_gates": gates,
        "observed": decisions["observed"],
        "model_environments": environments,
        "model_summaries": model_summaries,
        "records": records,
        "engineering_gates": decisions["engineering_gates"],
        "per_model_material_speedup": decisions["per_model_material_speedup"],
        "batch_interface_admission_allowed": (
            decisions["batch_interface_admission_allowed"]
        ),
        "universal_material_speedup_claim_allowed": (
            decisions["universal_material_speedup_claim_allowed"]
        ),
        "decision_boundary": {
            "interface_admission_requires": list(PARITY_GATE_NAMES),
            "universal_material_speedup_claim_requires": [
                "batch_interface_admission_allowed",
                "every_paired_repeat_meets_speedup_floor",
            ],
            "interpretation": (
                "Numerical equivalence admits the common batch interface. "
                "The failed all-repeat speed gate separately forbids a universal "
                "material-speedup claim; speed evidence remains backend-specific."
            ),
        },
        "threshold_basis": protocol["threshold_basis"],
        "claim_boundary": protocol["claim_boundary"],
        "command_provenance": command_provenance(
            __file__,
            vars(args),
            repository_root=REPOSITORY_ROOT,
            environment_variables=(
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "PYTORCH_CUDA_ALLOC_CONF",
            ),
        ),
    }
    _seal_and_write_after_gpu_postflight(args.output, artifact)
    print(
        f"Wrote {len(records)} model/case records to " f"{Path(args.output).resolve()}."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
