#!/usr/bin/env python3
"""Run a provenance-bound MLIP/device canary through Route 1 composition.

This runner answers a deliberately narrow question: can each requested local
model execute the same fixed-charge OBC-II/ACE E/F/numerical-H calculation on
the requested device?  It records heterogeneous placement explicitly; MLIP
CUDA plus OpenMM CPU is never relabelled as an all-GPU calculation.  Failures
remain in the requested denominator and no numerical difference is promoted
to a universal accuracy or precision threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_core import (
    command_provenance,
    sha256_file,
    write_json_atomic,
)
from stationary_point_checks import analyze_stationary_point

from maple.function.calculator.calculator_base import calculator_execution
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.device import resolve_torch_device
from maple.function.dispatcher.frequency.frequency import Frequency
from maple.function.dispatcher.optimization.optimization import (
    Optimization,
)
from maple.function.dispatcher.ts.algorithm.PRFO import PRFO
from maple.function.read.filereader.mol2_reader import MOL2Reader

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    # ANI curvature requires the validated in-memory float64 preparation.
    "ani2x": {
        "model_options": {"hessian": "numerical", "dtype": "float64"},
        "checkpoint": "ani2x.pt",
    },
    # Preserve each non-ANI checkpoint's native precision.
    "aimnet2": {
        "model_options": {"hessian": "numerical"},
        "checkpoint": "aimnet2.pt",
    },
    "maceoff23m": {
        "model_options": {"hessian": "numerical"},
        "checkpoint": "maceoff23m.pt",
    },
}

SOURCE_PATHS = (
    "maple/function/calculator/ani/_ani_calculator.py",
    "maple/function/calculator/aimnet/_aimnet2_calculator.py",
    "maple/function/calculator/mace/_mace_calculator.py",
    "maple/function/calculator/calculator_base.py",
    "maple/function/calculator/set_calculator.py",
    "maple/function/calculator/extra_correction/implicit/charges.py",
    "maple/function/calculator/extra_correction/implicit/correction.py",
    "maple/function/calculator/extra_correction/implicit/openmm_gb.py",
    "maple/function/calculator/extra_correction/implicit/openmm_execution.py",
    "maple/function/calculator/extra_correction/implicit/radii.py",
    "maple/function/calculator/extra_correction/implicit/sphere_union_dispersion.py",
    "maple/function/dispatcher/optimization/algorithm/LBFGS.py",
    "maple/function/dispatcher/optimization/optimization.py",
    "maple/function/dispatcher/frequency/frequency.py",
    "maple/function/dispatcher/ts/algorithm/PRFO.py",
    "docs/implicit-solvation/benchmarks/benchmark_core.py",
    "docs/implicit-solvation/benchmarks/stationary_point_checks.py",
    "docs/implicit-solvation/benchmarks/run_mlip_device_canary.py",
)

PROTOCOL: dict[str, Any] = {
    "schema_version": 1,
    "protocol_id": "route1-mlip-device-canary-20260913-v1",
    "claim_scope": (
        "Local interface and execution canary for only the explicitly requested "
        "model/device pairs; not validation of every registered MAPLE model, "
        "scientific accuracy, performance scaling, or an all-GPU claim."
    ),
    "models": tuple(MODEL_CONFIGS),
    "devices": ("cpu", "cuda:0"),
    "molecule": "water",
    "endpoint": {
        "implicit": "gb",
        "solvent": "water",
        "method": "gb",
        "provider": "openmm",
        "model": "obc2",
        "profile": "obc2-mbondi2",
        "nonpolar": "ace",
        "platform": "CPU",
        "experimental": True,
    },
    "charge": {
        "source": "mol2",
        "mode": "fixed",
        "geometry": "keep",
        "label": "am1bcc-frozen-route1-foundation-20260913",
    },
    "numerical_hessian_step_angstrom": 5e-4,
    "optimizer": {
        "method": "lbfgs",
        "curvature": 5.0,
        "max_step": 0.05,
        "max_iter": 50,
        "memory": 5,
        "verbose": 1,
    },
    "prfo": {
        "max_iter": 20,
        "recalc": 1,
        "trust_radius": 0.2,
        "trust_min": 0.001,
        "trust_max": 1.0,
        "hessian_update": "bofill",
        "project_rigid_modes": True,
        "f_max_th": 1e-4,
        "f_rms_th": 7.5e-5,
        "dp_max_th": 3e-4,
        "dp_rms_th": 2e-4,
    },
    "stationary_acceptance": {
        "force_max_hartree_per_angstrom": 2.5e-4,
        "force_rms_hartree_per_angstrom": 1.5e-4,
        "negative_cutoff_cm1": 30.0,
    },
}


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def _array_hash(values: Any) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _provider_identity(calculator: Any) -> dict[str, Any]:
    provider = calculator.solvent_correction.provider
    charges = np.asarray(provider.charges, dtype=np.float64)
    radii = getattr(provider, "radii", None)
    if radii is None:
        radii = provider.radius_result.radii_angstrom
    radii = np.asarray(radii, dtype=np.float64)
    return {
        "charges_e": charges,
        "radii_angstrom": radii,
        "charges_sha256": _array_hash(charges),
        "radii_sha256": _array_hash(radii),
    }


def _evaluate(atoms: Any) -> tuple[float, np.ndarray]:
    # Forces first: every supported MLIP force call also returns energy.
    forces = np.asarray(atoms.get_forces(), dtype=np.float64).copy()
    energy = float(atoms.get_potential_energy(force_consistent=True))
    if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
        raise ValueError("calculator returned incomplete or nonfinite forces")
    if not np.isfinite(energy):
        raise ValueError("calculator returned nonfinite energy")
    return energy, forces


def _checkpoint_record(model: str) -> dict[str, Any]:
    filename = MODEL_CONFIGS[model]["checkpoint"]
    path = ROOT / "maple/function/calculator/model" / filename
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _source_hashes() -> dict[str, str]:
    """Hash the complete MAPLE Python source plus benchmark execution surface."""
    required = [ROOT / relative for relative in SOURCE_PATHS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required canary source is missing: {missing}")
    paths = set((ROOT / "maple").rglob("*.py"))
    paths.update(required)
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted(paths)
    }


def execution_matches_request(
    execution: dict[str, Any], *, device: str, openmm_platform: str
) -> dict[str, Any]:
    """Fail closed when reported component placement contradicts the request."""
    requested_device = str(device)
    model = execution.get("model") or {}
    reported = model.get("reported_device")
    parameter = model.get("first_parameter_device")
    device_checks = {
        "reported_device_matches": reported == requested_device,
        "parameter_device_matches_when_known": (
            parameter is None or parameter == requested_device
        ),
    }
    solvent = execution.get("solvent") or {}
    solvent_check = solvent.get("platform") == openmm_platform
    return {
        "requested_model_device": requested_device,
        "reported_model_device": reported,
        "first_parameter_device": parameter,
        "requested_solvent_platform": openmm_platform,
        "reported_solvent_platform": solvent.get("platform"),
        **device_checks,
        "solvent_platform_matches": solvent_check,
        "passed": bool(all(device_checks.values()) and solvent_check),
    }


def _build_calculator(
    atoms: Any,
    *,
    model: str,
    device: str,
    openmm_platform: str,
    output: Path,
) -> Any:
    import torch

    endpoint = deepcopy(PROTOCOL["endpoint"])
    endpoint["platform"] = openmm_platform
    implicit = endpoint.pop("implicit")
    solvent = endpoint.pop("solvent")
    return SetCalculator(
        torch.device(device),
        model,
        str(output),
        atoms=atoms,
        implicit=implicit,
        solvent=solvent,
        model_options=deepcopy(MODEL_CONFIGS[model]["model_options"]),
        solvation_options=endpoint,
        charge_options=deepcopy(PROTOCOL["charge"]),
    ).set_calculator()


def run_efh_case(
    source: Path,
    *,
    model: str,
    device: str,
    requested_device: str | None = None,
    openmm_platform: str,
    attempt_dir: Path,
) -> dict[str, Any]:
    """Run one requested pair and retain failure plus partial provenance."""
    attempt_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = _checkpoint_record(model)
    record: dict[str, Any] = {
        "model": model,
        "device": device,
        "requested_device": requested_device or device,
        "openmm_platform": openmm_platform,
        "status": "failed",
        "input": {"path": str(source), "sha256": sha256_file(source)},
        "checkpoint": checkpoint,
        "model_options": deepcopy(MODEL_CONFIGS[model]["model_options"]),
    }
    try:
        if not checkpoint["exists"]:
            raise FileNotFoundError(
                f"Local checkpoint is required for offline canary: {checkpoint['path']}"
            )
        atoms = MOL2Reader(str(source), charge=0, mult=1, validate_charge=True)
        calculator = _build_calculator(
            atoms,
            model=model,
            device=device,
            openmm_platform=openmm_platform,
            output=attempt_dir / "calculator.out",
        )
        atoms.calc = calculator
        record["execution_before_preparation"] = calculator_execution(calculator)
        identity_before = _provider_identity(calculator)

        prepare = getattr(calculator, "prepare_numerical_derivatives", None)
        recommended_step = prepare() if callable(prepare) else None
        record["recommended_numerical_step_angstrom"] = recommended_step
        record["execution_after_preparation"] = calculator_execution(calculator)

        energy, forces = _evaluate(atoms)
        hessian = np.asarray(
            calculator.get_hessian(
                atoms,
                delta=PROTOCOL["numerical_hessian_step_angstrom"],
            ),
            dtype=np.float64,
        )
        if hessian.shape != (3 * len(atoms), 3 * len(atoms)):
            raise ValueError(f"calculator returned invalid Hessian shape {hessian.shape}")
        if not np.isfinite(hessian).all():
            raise ValueError("calculator returned nonfinite Hessian")

        np.save(attempt_dir / "forces.npy", forces)
        np.save(attempt_dir / "hessian.npy", hessian)
        identity_after = _provider_identity(calculator)
        execution_after = calculator_execution(calculator)
        placement = execution_matches_request(
            execution_after,
            device=device,
            openmm_platform=openmm_platform,
        )
        identity_preserved = bool(
            identity_before["charges_sha256"]
            == identity_after["charges_sha256"]
            and identity_before["radii_sha256"]
            == identity_after["radii_sha256"]
        )
        record.update(
            {
                "energy_hartree": energy,
                "forces": {
                    "path": str(attempt_dir / "forces.npy"),
                    "sha256": sha256_file(attempt_dir / "forces.npy"),
                    "maximum_absolute_hartree_per_angstrom": float(
                        np.max(np.abs(forces))
                    ),
                    "rms_hartree_per_angstrom": float(np.sqrt(np.mean(forces**2))),
                },
                "hessian": {
                    "path": str(attempt_dir / "hessian.npy"),
                    "sha256": sha256_file(attempt_dir / "hessian.npy"),
                    "shape": list(hessian.shape),
                    "maximum_absolute_hartree_per_angstrom2": float(
                        np.max(np.abs(hessian))
                    ),
                    "diagnostics": getattr(
                        calculator, "last_numerical_hessian_diagnostics", None
                    ),
                },
                "fixed_identity_before": identity_before,
                "fixed_identity_after": identity_after,
                "fixed_identity_preserved": identity_preserved,
                "execution_after_evaluation": execution_after,
                "execution_request_check": placement,
                "inference_precision": getattr(
                    calculator, "inference_precision_provenance", None
                ),
            }
        )
        if not identity_preserved:
            raise RuntimeError("fixed solvent charges or radii changed during E/F/H")
        if not placement["passed"]:
            raise RuntimeError("reported component execution contradicts the request")
        record["status"] = "completed"
    except Exception as exc:  # noqa: BLE001 -- native failures belong in the matrix
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    write_json_atomic(attempt_dir / "result.json", json_safe(record))
    return record


def compare_device_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report raw CPU/device deltas; intentionally apply no universal gate."""
    comparisons = []
    for model in sorted({str(record["model"]) for record in records}):
        by_device = {
            str(record["device"]): record
            for record in records
            if record["model"] == model
        }
        cpu = by_device.get("cpu")
        accelerators = [
            record
            for device, record in by_device.items()
            if device != "cpu"
        ]
        for accelerator in accelerators:
            item: dict[str, Any] = {
                "model": model,
                "reference_device": "cpu",
                "accelerator_device": accelerator["device"],
                "status": "unavailable",
                "acceptance_gate": None,
            }
            if cpu and cpu.get("status") == accelerator.get("status") == "completed":
                cpu_forces = np.load(cpu["forces"]["path"])
                accelerator_forces = np.load(accelerator["forces"]["path"])
                cpu_hessian = np.load(cpu["hessian"]["path"])
                accelerator_hessian = np.load(accelerator["hessian"]["path"])
                item.update(
                    {
                        "status": "compared",
                        "absolute_energy_difference_hartree": abs(
                            cpu["energy_hartree"] - accelerator["energy_hartree"]
                        ),
                        "maximum_force_difference_hartree_per_angstrom": float(
                            np.max(np.abs(cpu_forces - accelerator_forces))
                        ),
                        "rms_force_difference_hartree_per_angstrom": float(
                            np.sqrt(np.mean((cpu_forces - accelerator_forces) ** 2))
                        ),
                        "maximum_hessian_difference_hartree_per_angstrom2": float(
                            np.max(np.abs(cpu_hessian - accelerator_hessian))
                        ),
                    }
                )
            comparisons.append(item)
    return comparisons


def _set_stationary_stops(atoms: Any) -> None:
    stops = PROTOCOL["prfo"]
    atoms.f_max_th = stops["f_max_th"]
    atoms.f_rms_th = stops["f_rms_th"]
    atoms.dp_max_th = stops["dp_max_th"]
    atoms.dp_rms_th = stops["dp_rms_th"]


def _iteration_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    matches = re.findall(
        r"(?:Iteration|iteration)\s*:?[ \t]*(\d+)",
        path.read_text(errors="replace"),
    )
    return max(map(int, matches), default=None)


def _stationary_metrics(atoms: Any) -> dict[str, float | bool]:
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    step_max = float(getattr(atoms, "max_dp", math.inf))
    step_rms = float(getattr(atoms, "rms_dp", math.inf))
    return {
        "force_max_hartree_per_angstrom": float(np.max(np.abs(forces))),
        "force_rms_hartree_per_angstrom": float(np.sqrt(np.mean(forces**2))),
        "step_max_angstrom": step_max,
        "step_rms_angstrom": step_rms,
        "solver_stops_satisfied": bool(
            np.max(np.abs(forces)) <= PROTOCOL["prfo"]["f_max_th"]
            and np.sqrt(np.mean(forces**2)) <= PROTOCOL["prfo"]["f_rms_th"]
            and step_max <= PROTOCOL["prfo"]["dp_max_th"]
            and step_rms <= PROTOCOL["prfo"]["dp_rms_th"]
        ),
    }


def _planar_ammonia(atoms: Any) -> None:
    numbers = np.asarray(atoms.get_atomic_numbers())
    nitrogen = np.flatnonzero(numbers == 7)
    hydrogens = np.flatnonzero(numbers == 1)
    if nitrogen.size != 1 or hydrogens.size != 3 or len(atoms) != 4:
        raise ValueError("TS canary requires one NH3 molecule")
    positions = atoms.get_positions().copy()
    h = positions[hydrogens]
    normal = np.cross(h[1] - h[0], h[2] - h[0])
    norm = np.linalg.norm(normal)
    if norm <= 1e-12:
        raise ValueError("hydrogen plane is degenerate")
    normal /= norm
    n = nitrogen[0]
    positions[n] -= np.dot(positions[n] - h.mean(axis=0), normal) * normal
    atoms.set_positions(positions)


def run_workflows(
    *,
    inputs_dir: Path,
    model: str,
    device: str,
    requested_device: str | None = None,
    openmm_platform: str,
    attempt_dir: Path,
) -> dict[str, Any]:
    """Execute real OPT->FREQ and PRFO canaries with bounded controls."""
    attempt_dir.mkdir(parents=True, exist_ok=False)
    record: dict[str, Any] = {
        "model": model,
        "device": device,
        "requested_device": requested_device or device,
        "status": "failed",
        "actual_execution_completed": False,
        "checkpoint": _checkpoint_record(model),
        "controls": {
            "optimizer": deepcopy(PROTOCOL["optimizer"]),
            "prfo": deepcopy(PROTOCOL["prfo"]),
        },
    }
    try:
        if not record["checkpoint"]["exists"]:
            raise FileNotFoundError(
                "Local checkpoint is required for offline workflow canary: "
                f"{record['checkpoint']['path']}"
            )
        water_path = inputs_dir / "water/fixed.mol2"
        ammonia_path = inputs_dir / "ammonia/fixed.mol2"
        for path in (water_path, ammonia_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        record["inputs"] = {
            "water": {"path": str(water_path), "sha256": sha256_file(water_path)},
            "ammonia": {
                "path": str(ammonia_path),
                "sha256": sha256_file(ammonia_path),
            },
        }
        water = MOL2Reader(str(water_path), charge=0, mult=1, validate_charge=True)
        water.calc = _build_calculator(
            water,
            model=model,
            device=device,
            openmm_platform=openmm_platform,
            output=attempt_dir / "water-calculator.out",
        )
        prepare = getattr(water.calc, "prepare_numerical_derivatives", None)
        if callable(prepare):
            prepare()
        water_identity_before = _provider_identity(water.calc)
        _set_stationary_stops(water)
        opt_output = attempt_dir / "water-opt.out"
        optimized = Optimization(
            deepcopy(PROTOCOL["optimizer"]), str(opt_output), water
        ).run()
        if optimized.calc is None:
            raise RuntimeError("OPT returned atoms without the composed calculator")
        opt_metrics = _stationary_metrics(optimized)
        freq_output = attempt_dir / "water-freq.out"
        Frequency(
            str(freq_output),
            optimized,
            paras={
                "method": "mw",
                "diagonalization_device": device,
                "verbosity": 1,
            },
        ).run()
        record["opt_freq_task_executed"] = True
        water_hessian = np.asarray(
            optimized.calc.get_hessian(
                optimized, delta=PROTOCOL["numerical_hessian_step_angstrom"]
            ),
            dtype=np.float64,
        )
        water_analysis = analyze_stationary_point(
            optimized,
            water_hessian,
            optimized.get_forces(),
            negative_cutoff_cm1=PROTOCOL["stationary_acceptance"][
                "negative_cutoff_cm1"
            ],
        )

        ammonia = MOL2Reader(str(ammonia_path), charge=0, mult=1, validate_charge=True)
        ammonia.calc = _build_calculator(
            ammonia,
            model=model,
            device=device,
            openmm_platform=openmm_platform,
            output=attempt_dir / "ammonia-calculator.out",
        )
        prepare = getattr(ammonia.calc, "prepare_numerical_derivatives", None)
        if callable(prepare):
            prepare()
        ammonia_identity_before = _provider_identity(ammonia.calc)
        _planar_ammonia(ammonia)
        _set_stationary_stops(ammonia)
        prfo_output = attempt_dir / "ammonia-prfo.out"
        saddle = PRFO(
            str(prfo_output), ammonia, paras=deepcopy(PROTOCOL["prfo"])
        ).run()
        record["prfo_task_executed"] = True
        if saddle.calc is None:
            raise RuntimeError("PRFO returned atoms without the composed calculator")
        saddle_hessian = np.asarray(
            saddle.calc.get_hessian(
                saddle, delta=PROTOCOL["numerical_hessian_step_angstrom"]
            ),
            dtype=np.float64,
        )
        saddle_analysis = analyze_stationary_point(
            saddle,
            saddle_hessian,
            saddle.get_forces(),
            negative_cutoff_cm1=PROTOCOL["stationary_acceptance"][
                "negative_cutoff_cm1"
            ],
        )
        water_identity_after = _provider_identity(optimized.calc)
        ammonia_identity_after = _provider_identity(saddle.calc)
        identities_preserved = bool(
            water_identity_before["charges_sha256"]
            == water_identity_after["charges_sha256"]
            and water_identity_before["radii_sha256"]
            == water_identity_after["radii_sha256"]
            and ammonia_identity_before["charges_sha256"]
            == ammonia_identity_after["charges_sha256"]
            and ammonia_identity_before["radii_sha256"]
            == ammonia_identity_after["radii_sha256"]
        )
        executions = {
            "opt_freq": calculator_execution(optimized.calc),
            "prfo": calculator_execution(saddle.calc),
        }
        placement = {
            name: execution_matches_request(
                value, device=device, openmm_platform=openmm_platform
            )
            for name, value in executions.items()
        }
        record.update(
            {
                "opt_freq": {
                    "actual_opt_iterations": _iteration_count(opt_output),
                    "optimizer_metrics": opt_metrics,
                    "frequency_output": str(freq_output),
                    "frequency_output_sha256": sha256_file(freq_output),
                    "single_step_spectrum_diagnostic": water_analysis,
                    "frequency_diagonalization_device_requested": device,
                },
                "prfo": {
                    "actual_iterations": _iteration_count(prfo_output),
                    "optimizer_metrics": _stationary_metrics(saddle),
                    "single_step_spectrum_diagnostic": saddle_analysis,
                },
                "fixed_identity": {
                    "water_before": water_identity_before,
                    "water_after": water_identity_after,
                    "ammonia_before": ammonia_identity_before,
                    "ammonia_after": ammonia_identity_after,
                    "preserved": identities_preserved,
                },
                "execution": executions,
                "execution_request_checks": placement,
            }
        )
        if not identities_preserved:
            raise RuntimeError("fixed solvent charges or radii changed during workflows")
        if not all(item["passed"] for item in placement.values()):
            raise RuntimeError("reported workflow component execution contradicts request")
        record["actual_execution_completed"] = True
        record["status"] = "completed"
    except Exception as exc:  # noqa: BLE001 -- native failures belong in the matrix
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    record["output_hashes"] = {
        str(path.relative_to(attempt_dir)): sha256_file(path)
        for path in sorted(attempt_dir.rglob("*"))
        if path.is_file() and path.name != "result.json"
    }
    write_json_atomic(attempt_dir / "result.json", json_safe(record))
    return record


def effective_protocol(
    *,
    models: list[str],
    devices: list[dict[str, str]],
    openmm_platform: str,
    workflow_models: list[str],
    workflow_devices: list[dict[str, str]],
) -> dict[str, Any]:
    result = deepcopy(PROTOCOL)
    result["models"] = list(models)
    result["devices"] = list(devices)
    result["endpoint"]["platform"] = openmm_platform
    result["workflow_models"] = list(workflow_models)
    result["workflow_devices"] = list(workflow_devices)
    return result


def resolve_device_specs(devices: list[str]) -> list[dict[str, str]]:
    """Retain user spelling while executing only the shared canonical device."""
    return [
        {"requested": str(device), "canonical": str(resolve_torch_device(device))}
        for device in devices
    ]


def summarize_execution(
    records: list[dict[str, Any]],
    workflow_records: list[dict[str, Any]],
    *,
    requested_case_count: int,
    requested_workflow_case_count: int,
    source_unchanged: bool,
    artifacts_unchanged: bool = True,
    case_artifacts_consistent: bool = True,
) -> dict[str, Any]:
    completed = sum(record.get("status") == "completed" for record in records)
    workflows_completed = sum(
        record.get("actual_execution_completed") is True
        for record in workflow_records
    )
    return {
        "requested_case_count": requested_case_count,
        "completed_case_count": completed,
        "failed_case_count": requested_case_count - completed,
        "requested_workflow_case_count": requested_workflow_case_count,
        "completed_workflow_case_count": workflows_completed,
        "failed_workflow_case_count": (
            requested_workflow_case_count - workflows_completed
        ),
        "actual_execution_all_completed": bool(
            completed == requested_case_count
            and workflows_completed == requested_workflow_case_count
            and source_unchanged
            and artifacts_unchanged
            and case_artifacts_consistent
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.device:
        raise ValueError("At least one --device is required")
    if bool(args.workflow_model) != bool(args.workflow_device):
        raise ValueError(
            "--workflow-model and --workflow-device must be supplied together"
        )
    inputs_dir = Path(args.inputs_dir).resolve()
    source = inputs_dir / "water/fixed.mol2"
    ammonia_source = inputs_dir / "ammonia/fixed.mol2"
    for path in (source, ammonia_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    models = args.model or list(MODEL_CONFIGS)
    unknown = sorted(set(models) - set(MODEL_CONFIGS))
    if unknown:
        raise ValueError(f"Canary has no explicit configuration for: {unknown}")
    unknown_workflows = sorted(set(args.workflow_model) - set(MODEL_CONFIGS))
    if unknown_workflows:
        raise ValueError(
            f"Canary has no explicit workflow configuration for: {unknown_workflows}"
        )
    if not set(args.workflow_model).issubset(models):
        raise ValueError("Every workflow model must also be selected by --model")
    device_specs = resolve_device_specs(args.device)
    workflow_device_specs = resolve_device_specs(args.workflow_device)
    if len(models) != len(set(models)):
        raise ValueError("Duplicate --model requests would overwrite attempt directories")
    for label, specs in (
        ("--device", device_specs),
        ("--workflow-device", workflow_device_specs),
    ):
        canonical = [spec["canonical"] for spec in specs]
        if len(canonical) != len(set(canonical)):
            raise ValueError(
                f"Duplicate canonical {label} requests would overwrite attempt directories"
            )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    protocol = effective_protocol(
        models=models,
        devices=device_specs,
        openmm_platform=args.openmm_platform,
        workflow_models=args.workflow_model,
        workflow_devices=workflow_device_specs,
    )
    source_hashes_before = _source_hashes()
    input_manifest = {
        "water": {"path": str(source), "sha256": sha256_file(source)},
        "ammonia": {
            "path": str(ammonia_source),
            "sha256": sha256_file(ammonia_source),
        },
    }
    checkpoint_manifest = {model: _checkpoint_record(model) for model in models}

    frozen = {
        "effective_protocol": json_safe(protocol),
        "inputs": input_manifest,
        "checkpoints": checkpoint_manifest,
        "source_hashes": source_hashes_before,
    }
    write_json_atomic(output_dir / "protocol-freeze.json", frozen)
    records = []
    for model in models:
        for device_spec in device_specs:
            device = device_spec["canonical"]
            slug = device.replace(":", "-")
            records.append(
                run_efh_case(
                    source,
                    model=model,
                    device=device,
                    requested_device=device_spec["requested"],
                    openmm_platform=args.openmm_platform,
                    attempt_dir=output_dir / f"efh-{model}-{slug}",
                )
            )

    workflow_records = []
    for model in args.workflow_model:
        for device_spec in workflow_device_specs:
            device = device_spec["canonical"]
            slug = device.replace(":", "-")
            workflow_records.append(
                run_workflows(
                    inputs_dir=inputs_dir,
                    model=model,
                    device=device,
                    requested_device=device_spec["requested"],
                    openmm_platform=args.openmm_platform,
                    attempt_dir=output_dir / f"workflow-{model}-{slug}",
                )
            )

    source_hashes_after = _source_hashes()
    source_unchanged = source_hashes_before == source_hashes_after
    input_manifest_after = {
        name: {"path": item["path"], "sha256": sha256_file(item["path"])}
        for name, item in input_manifest.items()
    }
    checkpoint_manifest_after = {
        model: _checkpoint_record(model) for model in models
    }
    artifacts_unchanged = bool(
        input_manifest == input_manifest_after
        and checkpoint_manifest == checkpoint_manifest_after
    )
    case_artifacts_consistent = all(
        record.get("input", {}).get("sha256") == input_manifest["water"]["sha256"]
        and record.get("checkpoint", {}).get("sha256")
        == checkpoint_manifest[record["model"]]["sha256"]
        for record in records
    ) and all(
        record.get("inputs") == input_manifest
        and record.get("checkpoint", {}).get("sha256")
        == checkpoint_manifest[record["model"]]["sha256"]
        for record in workflow_records
    )
    requested_workflow_count = len(args.workflow_model) * len(workflow_device_specs)
    execution_summary = summarize_execution(
        records,
        workflow_records,
        requested_case_count=len(models) * len(device_specs),
        requested_workflow_case_count=requested_workflow_count,
        source_unchanged=source_unchanged,
        artifacts_unchanged=artifacts_unchanged,
        case_artifacts_consistent=case_artifacts_consistent,
    )
    convergence = {
        "opt_freq_single_step_spectrum_minimum_diagnostic_count": sum(
            record.get("opt_freq", {})
            .get("single_step_spectrum_diagnostic", {})
            .get("is_minimum")
            is True
            for record in workflow_records
        ),
        "prfo_single_step_spectrum_first_order_saddle_diagnostic_count": sum(
            record.get("prfo", {})
            .get("single_step_spectrum_diagnostic", {})
            .get("is_first_order_saddle")
            is True
            for record in workflow_records
        ),
        "note": (
            "These are single-step symmetrized-spectrum diagnostics, not qualified "
            "stationary-point counts; raw Hessian refinement is outside this canary. "
            "They are separate from whether the requested task actually executed."
        ),
    }
    result = {
        "schema_version": 1,
        "artifact_type": "route1-mlip-device-canary",
        "effective_protocol": json_safe(protocol),
        "command_provenance": command_provenance(
            __file__,
            vars(args),
            repository_root=ROOT,
            environment_variables=(
                "MAPLE_OFFLINE",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "OPENMM_CPU_THREADS",
            ),
        ),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pid": os.getpid(),
        },
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "source_unchanged_during_run": source_unchanged,
        "inputs": input_manifest,
        "inputs_after": input_manifest_after,
        "checkpoints": checkpoint_manifest,
        "checkpoints_after": checkpoint_manifest_after,
        "artifacts_unchanged_during_run": artifacts_unchanged,
        "case_artifacts_consistent": case_artifacts_consistent,
        **execution_summary,
        "cases": records,
        "device_comparisons": compare_device_pairs(records),
        "workflow_cases": workflow_records,
        "convergence": convergence,
        "limitations": [
            "Only explicitly named local models are exercised; the wider registry is not validated.",
            "OpenMM CPU or Reference means CUDA MLIP plus CPU solvent is heterogeneous, not all-GPU.",
            "One water E/F/H case and tiny workflows are interface canaries, not accuracy or speed benchmarks.",
            "Raw device differences are reported without a universal cross-model precision threshold.",
            "Unsupported, missing-checkpoint, nonconverged, and native failures remain in the denominator.",
        ],
    }
    write_json_atomic(output_dir / "result.json", json_safe(result))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--device", action="append", default=[])
    parser.add_argument("--openmm-platform", choices=("CPU", "Reference"), default="CPU")
    parser.add_argument(
        "--workflow-model",
        action="append",
        default=[],
        help="Opt in to bounded OPT->FREQ and PRFO execution for this selected model.",
    )
    parser.add_argument(
        "--workflow-device",
        action="append",
        default=[],
        help="Device for opted-in workflow models (required with --workflow-model).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.device:
        args.device = ["cpu", "cuda:0"]
    if bool(args.workflow_model) != bool(args.workflow_device):
        raise ValueError(
            "--workflow-model and --workflow-device must be supplied together"
        )
    result = run(args)
    print(json.dumps(json_safe(result), indent=2))
    return 0 if result["actual_execution_all_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
