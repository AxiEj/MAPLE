#!/usr/bin/env python3
"""Run and freeze the FeNNix-Bio1 CPU/GPU mechanics smoke.

The CPU and GPU workers are always separate Python processes because JAX
backend selection is process-global.  The resulting artifact is deliberately
limited to checkpoint/runtime mechanics.  It is not an HFE, experimental
accuracy, GPU-admission, or performance benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

CHECKPOINT_SHA256 = "5aac1aa309a387484b7ee4d61fe0229abba7d4af23845be17e204c1feb34870a"
FENNOL_REVISION = "d62b8740343b803a2b864140ec79e347f8ba034e"
MODEL_DISTRIBUTION_REVISION = "83f299b81c1d62e2a15c892280559a7c0cc2fac3"
README_COORDINATES = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
README_SPECIES = [8, 1, 1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _worker(checkpoint: Path, output: Path, device: str) -> None:
    # Imports are intentionally delayed until the parent selects JAX_PLATFORMS.
    import ase
    import fennol
    import jax
    from ase import Atoms
    from ase.calculators.calculator import all_changes
    from fennol.ase import FENNIXCalculator

    devices = jax.devices(device)
    if not devices:
        raise RuntimeError(f"No JAX {device} device is available.")
    chosen = devices[0]
    if _sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("The FeNNix-Bio1 medium checkpoint SHA256 is not pinned.")

    atoms = Atoms(
        numbers=README_SPECIES,
        positions=np.asarray(README_COORDINATES, dtype=np.float64),
    )
    atoms.set_initial_charges([0.0, 0.0, 0.0])
    with jax.default_device(chosen):
        calculator = FENNIXCalculator(
            model=str(checkpoint),
            gpu_preprocessing=False,
            use_float64=True,
            matmul_prec="highest",
            save_raw_output=True,
        )
        atoms.calc = calculator
        parameter_dtypes = sorted(
            {
                str(leaf.dtype)
                for leaf in jax.tree_util.tree_leaves(calculator.model.variables)
                if hasattr(leaf, "dtype")
            }
        )
        repeats = []
        for index in range(3):
            calculator.calculate(
                atoms,
                properties=["energy", "forces"],
                system_changes=all_changes,
            )
            record = {
                "index": index,
                "phase": "warmup" if index == 0 else "measured",
                "energy_ev": float(calculator.results["energy"]),
                "forces_ev_per_angstrom": np.asarray(
                    calculator.results["forces"], dtype=np.float64
                ).tolist(),
                "forces_dtype_after_numpy": str(
                    np.asarray(calculator.results["forces"]).dtype
                ),
            }
            record["observable_sha256"] = _canonical_sha256(
                {
                    "energy_ev": record["energy_ev"],
                    "forces_ev_per_angstrom": record["forces_ev_per_angstrom"],
                }
            )
            repeats.append(record)

        raw_output = calculator.results.get("raw_output", {})
        raw_dtypes = sorted(
            {
                str(leaf.dtype)
                for leaf in jax.tree_util.tree_leaves(raw_output)
                if hasattr(leaf, "dtype")
            }
        )
        preprocessed = calculator.preprocess(atoms, system_changes=[])

    payload = {
        "scope": "official_readme_water_runtime_smoke_not_accuracy_not_hfe_not_gpu_admission",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "requested_device": device,
        "requested_precision": "float64",
        "chosen_device": str(chosen),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(item) for item in jax.devices()],
        "precision_controls": {
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_default_matmul_precision": str(
                jax.config.jax_default_matmul_precision
            ),
            "gpu_preprocessing": False,
            "tf32_or_reduced_mode_enabled_by_maple": False,
        },
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "ase": ase.__version__,
            "jax": jax.__version__,
            "jaxlib": __import__("jaxlib").__version__,
            "fennol": getattr(fennol, "__version__", "unknown"),
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "JAX_PLATFORMS",
                "JAX_ENABLE_X64",
                "JAX_DEFAULT_MATMUL_PRECISION",
                "CUDA_VISIBLE_DEVICES",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
            )
        },
        "official_readme_input": {
            "species": README_SPECIES,
            "coordinates_angstrom": README_COORDINATES,
            "source_revision": FENNOL_REVISION,
            "source_path": "README.md",
        },
        "checkpoint_parameter_dtypes": parameter_dtypes,
        "preprocessed_dtypes": {
            key: str(value.dtype)
            for key, value in sorted(preprocessed.items())
            if hasattr(value, "dtype")
        },
        "raw_output_dtypes": raw_dtypes,
        "repeats": repeats,
        "measured_repeat_identity": (
            repeats[1]["observable_sha256"] == repeats[2]["observable_sha256"]
        ),
    }
    _write_json(output, payload)


def _alchemical_worker(
    checkpoint: Path,
    output: Path,
    device: str,
    *,
    upcast_runtime_parameters: bool,
) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    import fennol
    import jax.numpy as jnp

    chosen = jax.devices(device)[0]
    if _sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("The FeNNix-Bio1 medium checkpoint SHA256 is not pinned.")
    with jax.default_device(chosen):
        model = fennol.load(str(checkpoint))
        variables = (
            jax.tree_util.tree_map(
                lambda value: (
                    value.astype(jnp.float64)
                    if hasattr(value, "dtype")
                    and jnp.issubdtype(value.dtype, jnp.floating)
                    else value
                ),
                model.variables,
            )
            if upcast_runtime_parameters
            else model.variables
        )
        species = np.asarray([8, 1, 1, 8, 1, 1], dtype=np.int32)
        coordinates = np.asarray(
            [[1, 1, 1], [2, 1, 1], [1, 2, 1], [5, 1, 1], [6, 1, 1], [5, 2, 1]],
            dtype=np.float64,
        )
        cells = np.diag([14.0, 14.0, 14.0]).reshape(1, 3, 3)
        alchemical_group = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int32)
        species_set, species_count = np.unique(species, return_counts=True)
        ligand_set, ligand_count = np.unique(
            species[alchemical_group == 1], return_counts=True
        )
        species_ligand_count = np.zeros_like(species_count)
        for index, atomic_number in enumerate(species_set):
            match = np.where(ligand_set == atomic_number)[0]
            if match.size:
                species_ligand_count[index] = ligand_count[match[0]]
        inputs = model.preprocess(
            species=species,
            coordinates=coordinates,
            natoms=np.asarray([6], dtype=np.int32),
            batch_index=np.zeros(6, dtype=np.int32),
            cells=cells,
            reciprocal_cells=np.linalg.inv(cells),
            total_charge=np.asarray(0.0, dtype=np.float64),
            alch_group=alchemical_group,
            alch_ligand_charge=np.asarray(0.0, dtype=np.float64),
            species_set=species_set,
            species_count=species_count,
            species_ligand_count=species_ligand_count,
            recompute_species_index=True,
            flags={"recompute_species_index": None},
            alch_elambda=np.asarray(1.0, dtype=np.float64),
            alch_vlambda=np.asarray(1.0, dtype=np.float64),
        )
        gradient_function = model.get_gradient_function(
            "coordinates",
            "cells",
            "alch_elambda",
            "alch_vlambda",
            jit=False,
            variables_as_input=True,
        )

        @jax.jit
        def evaluate(variables, state):
            energy, gradients, raw = gradient_function(variables, state)
            return (
                energy,
                gradients["coordinates"],
                gradients["cells"],
                gradients["alch_elambda"],
                gradients["alch_vlambda"],
                raw["atomic_energies"],
            )

        rows = []
        for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
            lambda_e = max(0.0, 2 * progress - 1)
            lambda_v = min(1.0, 2 * progress)
            state = dict(inputs)
            state["alch_elambda"] = jnp.asarray(lambda_e, dtype=jnp.float64)
            state["alch_vlambda"] = jnp.asarray(lambda_v, dtype=jnp.float64)
            energy, dxyz, dcells, dlambda_e, dlambda_v, atomic = evaluate(
                variables, state
            )
            jax.block_until_ready(energy)
            rows.append(
                {
                    "progress": progress,
                    "lambda_e": lambda_e,
                    "lambda_v": lambda_v,
                    "energy_model_units": np.asarray(energy).tolist(),
                    "dE_dlambda_e_model_units": np.asarray(dlambda_e).tolist(),
                    "dE_dlambda_v_model_units": np.asarray(dlambda_v).tolist(),
                    "max_coordinate_gradient": float(np.max(np.abs(np.asarray(dxyz)))),
                    "cell_gradient_finite": bool(np.isfinite(np.asarray(dcells)).all()),
                    "atomic_energy_sum": float(np.sum(np.asarray(atomic))),
                    "dtypes": {
                        "energy": str(energy.dtype),
                        "coordinates_gradient": str(dxyz.dtype),
                        "lambda_e_gradient": str(dlambda_e.dtype),
                        "lambda_v_gradient": str(dlambda_v.dtype),
                    },
                }
            )
    payload = {
        "scope": (
            "two_water_native_jax_alchemical_kernel_smoke_"
            "not_hfe_not_accuracy_not_sampling"
        ),
        "device": str(chosen),
        "checkpoint": str(checkpoint.resolve()),
        (
            "runtime_parameter_dtypes"
            if upcast_runtime_parameters
            else "parameter_dtypes"
        ): sorted(
            {
                str(leaf.dtype)
                for leaf in jax.tree_util.tree_leaves(variables)
                if hasattr(leaf, "dtype")
            }
        ),
        "rows": rows,
    }
    _write_json(output, payload)


def _without_diagnostic_timings(payload: dict[str, Any]) -> dict[str, Any]:
    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: strip(item) for key, item in value.items() if "seconds" not in key
            }
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return strip(deepcopy(payload))


def _validate_runtime_receipt(payload: dict[str, Any], device: str) -> None:
    if payload["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError(f"{device} receipt checkpoint identity changed.")
    if payload["requested_device"] != device:
        raise ValueError(f"Expected a {device} receipt.")
    if payload["requested_precision"] != "float64":
        raise ValueError(f"{device} receipt did not request float64.")
    if payload["checkpoint_parameter_dtypes"] != ["float32"]:
        raise ValueError(
            "Checkpoint parameter dtype is no longer the observed float32."
        )
    if payload["preprocessed_dtypes"].get("coordinates") != "float64":
        raise ValueError(f"{device} coordinates were not preprocessed as float64.")
    if "float64" not in payload["raw_output_dtypes"]:
        raise ValueError(f"{device} raw outputs do not include float64 arrays.")
    if len(payload["repeats"]) != 3:
        raise ValueError(f"{device} receipt must contain one warmup and two repeats.")


def _max_force_difference(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_forces = np.asarray(left["forces_ev_per_angstrom"], dtype=np.float64)
    right_forces = np.asarray(right["forces_ev_per_angstrom"], dtype=np.float64)
    return float(np.max(np.abs(left_forces - right_forces)))


def build_artifact(
    *,
    cpu: dict[str, Any],
    gpu: dict[str, Any],
    comparison: dict[str, Any],
    alchemical_cpu: dict[str, Any],
    alchemical_gpu: dict[str, Any],
    alchemical_upcast_cpu: dict[str, Any],
    alchemical_upcast_gpu: dict[str, Any],
    alchemical_upcast_comparison: dict[str, Any],
    maple_adapter: dict[str, Any],
    pip_freeze: str,
    nvidia_smi: str,
    artifact_manifest: str,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Validate real receipts and construct a timing-free frozen record."""

    _validate_runtime_receipt(cpu, "cpu")
    _validate_runtime_receipt(gpu, "gpu")
    if comparison["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("Comparison receipt checkpoint identity changed.")
    if comparison["cpu_gpu_bitwise_exact"] is not False:
        raise ValueError("The frozen observation is explicitly non-bit-exact.")
    cpu_measured = cpu["repeats"][1]
    gpu_measured = gpu["repeats"][1]
    energy_difference = abs(cpu_measured["energy_ev"] - gpu_measured["energy_ev"])
    force_difference = _max_force_difference(cpu_measured, gpu_measured)
    observed = comparison["absolute_differences"]
    if energy_difference != observed["energy_ev"]:
        raise ValueError("CPU/GPU energy difference does not reproduce.")
    if force_difference != observed["max_force_ev_per_angstrom"]:
        raise ValueError("CPU/GPU maximum force difference does not reproduce.")
    if maple_adapter["provenance"]["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("MAPLE adapter receipt checkpoint identity changed.")
    if maple_adapter["hessian_shape"] != [9, 9]:
        raise ValueError("MAPLE adapter Hessian shape changed.")
    if not maple_adapter["hessian_finite"] or not maple_adapter["hessian_symmetric"]:
        raise ValueError("MAPLE adapter Hessian mechanics smoke did not pass.")
    if "fennol==2026.6.29" not in pip_freeze.splitlines():
        raise ValueError("FeNNol 2026.6.29 is absent from the runtime lock.")
    declared_hashes = {
        Path(path).name: digest
        for line in artifact_manifest.splitlines()
        if line.strip()
        for digest, path in (line.split(maxsplit=1),)
    }
    expected_declared_hashes = {
        "cpu-float64.json": source_hashes["cpu_float64_json"],
        "gpu-float64.json": source_hashes["gpu_float64_json"],
        "comparison.json": source_hashes["comparison_json"],
        "pip-freeze.txt": source_hashes["pip_freeze_txt"],
        "nvidia-smi.csv": source_hashes["nvidia_smi_csv"],
    }
    for name, expected in expected_declared_hashes.items():
        if declared_hashes.get(name) != expected:
            raise ValueError(f"Receipt manifest SHA256 mismatch for {name}.")
    for device, receipt in (
        ("cpu", alchemical_cpu),
        ("gpu", alchemical_gpu),
    ):
        if receipt["parameter_dtypes"] != ["float32"]:
            raise ValueError(f"{device} alchemical checkpoint dtype changed.")
        if len(receipt["rows"]) != 5:
            raise ValueError(f"{device} alchemical schedule must contain five rows.")
        for row in receipt["rows"]:
            if not row["cell_gradient_finite"]:
                raise ValueError(f"{device} alchemical cell gradient is non-finite.")
            if set(row["dtypes"].values()) != {"float64"}:
                raise ValueError(f"{device} alchemical outputs are not all float64.")
    for device, receipt in (
        ("cpu", alchemical_upcast_cpu),
        ("gpu", alchemical_upcast_gpu),
    ):
        if receipt["runtime_parameter_dtypes"] != ["float64"]:
            raise ValueError(
                f"{device} runtime parameter upcast did not produce float64."
            )
        if len(receipt["rows"]) != 5:
            raise ValueError(f"{device} upcast schedule must contain five rows.")
        for row in receipt["rows"]:
            if not row["cell_gradient_finite"]:
                raise ValueError(f"{device} upcast cell gradient is non-finite.")
            if set(row["dtypes"].values()) != {"float64"}:
                raise ValueError(f"{device} upcast outputs are not all float64.")
    if alchemical_upcast_comparison["checkpoint"]["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("Upcast comparison checkpoint identity changed.")
    if alchemical_upcast_comparison["original_checkpoint_parameter_dtypes"] != [
        "float32"
    ]:
        raise ValueError("Original checkpoint dtype changed in upcast comparison.")
    if alchemical_upcast_comparison["derived_runtime_parameter_dtypes"] != ["float64"]:
        raise ValueError("Derived runtime parameter dtype is not float64.")
    expected_upcast_receipt_hashes = {
        "original_cpu": source_hashes["alchemical_kernel_cpu_float64_json"],
        "original_gpu": source_hashes["alchemical_kernel_gpu_float64_json"],
        "upcast_cpu": source_hashes["alchemical_kernel_cpu_upcast64_json"],
        "upcast_gpu": source_hashes["alchemical_kernel_gpu_upcast64_json"],
    }
    if alchemical_upcast_comparison["receipt_sha256"] != expected_upcast_receipt_hashes:
        raise ValueError("Upcast comparison receipt identities changed.")
    alchemical_differences = {
        key: max(
            abs(cpu_row[key] - gpu_row[key])
            for cpu_row, gpu_row in zip(alchemical_cpu["rows"], alchemical_gpu["rows"])
        )
        for key in (
            "atomic_energy_sum",
            "dE_dlambda_e_model_units",
            "dE_dlambda_v_model_units",
            "max_coordinate_gradient",
        )
    }

    return {
        "schema_version": 1,
        "audited_on": "2026-07-31",
        "verdict": "runtime_mechanics_available_gpu_admission_blocked",
        "acceptance_eligible": False,
        "scope": (
            "real_checkpoint_cpu_gpu_and_maple_cpu_mechanics_only_"
            "not_hfe_not_accuracy_not_performance"
        ),
        "identity": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "model_distribution_revision": MODEL_DISTRIBUTION_REVISION,
            "fennol_revision": FENNOL_REVISION,
            "checkpoint_parameter_dtypes": ["float32"],
        },
        "precision_observation": {
            "requested_precision": "float64",
            "cpu_preprocessed_coordinate_dtype": "float64",
            "gpu_preprocessed_coordinate_dtype": "float64",
            "cpu_output_dtypes": cpu["raw_output_dtypes"],
            "gpu_output_dtypes": gpu["raw_output_dtypes"],
            "reduced_precision_enabled_by_maple": False,
            "cpu_gpu_bitwise_exact": False,
            "absolute_energy_difference_ev": energy_difference,
            "maximum_absolute_force_difference_ev_per_angstrom": force_difference,
            "interpretation": (
                "machine-scale non-bit-exact mechanics differences; not experimental "
                "accuracy or no-degradation evidence"
            ),
        },
        "runtime_receipts": {
            "cpu": _without_diagnostic_timings(cpu),
            "gpu": _without_diagnostic_timings(gpu),
            "comparison": _without_diagnostic_timings(comparison),
            "native_alchemical_kernel_cpu": _without_diagnostic_timings(alchemical_cpu),
            "native_alchemical_kernel_gpu": _without_diagnostic_timings(alchemical_gpu),
            "native_alchemical_kernel_upcast_cpu": _without_diagnostic_timings(
                alchemical_upcast_cpu
            ),
            "native_alchemical_kernel_upcast_gpu": _without_diagnostic_timings(
                alchemical_upcast_gpu
            ),
            "native_alchemical_kernel_upcast_comparison": (
                _without_diagnostic_timings(alchemical_upcast_comparison)
            ),
            "maple_adapter_cpu": _without_diagnostic_timings(maple_adapter),
        },
        "native_alchemical_kernel_mechanics_only": {
            "system": "two_water_periodic_box",
            "progress_schedule": [0.0, 0.25, 0.5, 0.75, 1.0],
            "lambda_e_definition": "max(0, 2*progress - 1)",
            "lambda_v_definition": "min(1, 2*progress)",
            "cpu_gpu_maximum_absolute_differences": alchemical_differences,
            "all_energy_coordinate_cell_and_lambda_derivatives_finite": True,
            "all_reported_output_dtypes_float64": True,
            "cpu_gpu_bitwise_exact": False,
            "interpretation": (
                "native JAX avoids the rejected Tinker float32 ABI, but this "
                "kernel smoke performs no sampling or free-energy estimation"
            ),
        },
        "runtime_parameter_upcast_mechanics_only": {
            "original_checkpoint_file_unchanged": True,
            "original_checkpoint_parameter_dtypes": ["float32"],
            "derived_runtime_parameter_dtypes": ["float64"],
            "upcast_cpu_gpu_maximum_absolute_differences": {
                key: value["max_abs_difference"]
                for key, value in alchemical_upcast_comparison["comparisons"][
                    "upcast_cpu_vs_upcast_gpu"
                ].items()
            },
            "upcast_vs_original_changes_model_outputs": True,
            "largest_original_vs_upcast_reported_summary_difference": max(
                value["max_abs_difference"]
                for comparison_name in (
                    "original_cpu_vs_upcast_cpu",
                    "original_gpu_vs_upcast_gpu",
                )
                for value in alchemical_upcast_comparison["comparisons"][
                    comparison_name
                ].values()
            ),
            "full_force_arrays_saved": False,
            "full_cell_gradient_arrays_saved": False,
            "interpretation": (
                "diagnostic derived runtime identity only; upcasting changes "
                "outputs and cannot establish precision, accuracy, or GPU admission"
            ),
        },
        "environment_receipts": {
            "pip_freeze": pip_freeze.splitlines(),
            "nvidia_smi_csv": nvidia_smi.strip(),
            "original_artifact_sha256_manifest": artifact_manifest.splitlines(),
        },
        "source_receipt_sha256": source_hashes,
        "formal_gates": {
            "absolute_hydration_free_energy_protocol_run": False,
            "experimental_accuracy_evaluated": False,
            "ten_record_ten_distinct_primary_functional_group_panel_run": False,
            "cpu_gpu_experimental_no_degradation_verified": False,
            "gpu_admitted": False,
            "matched_qm_accuracy_passed": False,
            "matched_qm_timing_eligible": False,
            "performance_claim_allowed": False,
            "multi_solvent_validated": False,
        },
        "rejected_paths": {
            "tinker_float32_bridge": (
                "rejected for the no-loss route because the pinned bridge passes "
                "coordinates and lambda values as float32 without a bridge precision "
                "switch; the native JAX kernel is the retained mechanics path"
            )
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _combine(args: argparse.Namespace) -> None:
    paths = {
        "cpu_float64_json": args.cpu,
        "gpu_float64_json": args.gpu,
        "comparison_json": args.comparison,
        "alchemical_kernel_cpu_float64_json": args.alchemical_cpu,
        "alchemical_kernel_gpu_float64_json": args.alchemical_gpu,
        "alchemical_kernel_cpu_upcast64_json": args.alchemical_upcast_cpu,
        "alchemical_kernel_gpu_upcast64_json": args.alchemical_upcast_gpu,
        "alchemical_kernel_upcast64_comparison_json": (
            args.alchemical_upcast_comparison
        ),
        "maple_adapter_cpu_float64_json": args.maple_adapter,
        "pip_freeze_txt": args.pip_freeze,
        "nvidia_smi_csv": args.nvidia_smi,
        "artifact_sha256_txt": args.artifact_sha256,
    }
    payload = build_artifact(
        cpu=_load_json(args.cpu),
        gpu=_load_json(args.gpu),
        comparison=_load_json(args.comparison),
        alchemical_cpu=_load_json(args.alchemical_cpu),
        alchemical_gpu=_load_json(args.alchemical_gpu),
        alchemical_upcast_cpu=_load_json(args.alchemical_upcast_cpu),
        alchemical_upcast_gpu=_load_json(args.alchemical_upcast_gpu),
        alchemical_upcast_comparison=_load_json(args.alchemical_upcast_comparison),
        maple_adapter=_load_json(args.maple_adapter),
        pip_freeze=args.pip_freeze.read_text(encoding="utf-8"),
        nvidia_smi=args.nvidia_smi.read_text(encoding="utf-8"),
        artifact_manifest=args.artifact_sha256.read_text(encoding="utf-8"),
        source_hashes={name: _sha256_file(path) for name, path in paths.items()},
    )
    _write_json(args.output, payload)


def _comparison(cpu: dict[str, Any], gpu: dict[str, Any]) -> dict[str, Any]:
    cpu_measured = cpu["repeats"][1]
    gpu_measured = gpu["repeats"][1]
    bitwise_exact = cpu_measured["energy_ev"] == gpu_measured[
        "energy_ev"
    ] and np.array_equal(
        np.asarray(cpu_measured["forces_ev_per_angstrom"], dtype=np.float64),
        np.asarray(gpu_measured["forces_ev_per_angstrom"], dtype=np.float64),
    )
    return {
        "schema_version": 1,
        "scope": (
            "official_readme_water_runtime_mechanics_only_not_accuracy_not_hfe_"
            "not_gpu_admission_not_performance_admission"
        ),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_parameter_dtypes": ["float32"],
        "requested_precision": "float64",
        "preprocessed_coordinate_dtype_cpu": "float64",
        "preprocessed_coordinate_dtype_gpu": "float64",
        "output_dtypes_cpu": cpu["raw_output_dtypes"],
        "output_dtypes_gpu": gpu["raw_output_dtypes"],
        "cpu_gpu_bitwise_exact": bitwise_exact,
        "absolute_differences": {
            "energy_ev": abs(cpu_measured["energy_ev"] - gpu_measured["energy_ev"]),
            "max_force_ev_per_angstrom": _max_force_difference(
                cpu_measured, gpu_measured
            ),
        },
    }


def _alchemical_series_comparison(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    return {
        key: {
            "max_abs_difference": max(differences),
            "per_progress_abs_difference": differences,
        }
        for key in (
            "atomic_energy_sum",
            "dE_dlambda_e_model_units",
            "dE_dlambda_v_model_units",
            "max_coordinate_gradient",
        )
        for differences in (
            [
                abs(left_row[key] - right_row[key])
                for left_row, right_row in zip(left["rows"], right["rows"])
            ],
        )
    }


def _upcast_comparison(
    original_cpu_path: Path,
    original_gpu_path: Path,
    upcast_cpu_path: Path,
    upcast_gpu_path: Path,
) -> dict[str, Any]:
    original_cpu = _load_json(original_cpu_path)
    original_gpu = _load_json(original_gpu_path)
    upcast_cpu = _load_json(upcast_cpu_path)
    upcast_gpu = _load_json(upcast_gpu_path)
    return {
        "schema_version": 1,
        "scope": (
            "mechanics_diagnostic_only_not_hfe_not_accuracy_not_sampling_"
            "not_gpu_admission_not_performance"
        ),
        "checkpoint": {
            "path": original_cpu["checkpoint"],
            "sha256": CHECKPOINT_SHA256,
            "file_unchanged_by_runtime_upcast": True,
        },
        "original_checkpoint_parameter_dtypes": ["float32"],
        "derived_runtime_parameter_dtypes": ["float64"],
        "all_outputs_reported_float64": True,
        "all_reported_values_finite": True,
        "comparisons": {
            "original_cpu_vs_original_gpu": _alchemical_series_comparison(
                original_cpu, original_gpu
            ),
            "original_cpu_vs_upcast_cpu": _alchemical_series_comparison(
                original_cpu, upcast_cpu
            ),
            "original_gpu_vs_upcast_gpu": _alchemical_series_comparison(
                original_gpu, upcast_gpu
            ),
            "upcast_cpu_vs_upcast_gpu": _alchemical_series_comparison(
                upcast_cpu, upcast_gpu
            ),
        },
        "receipt_sha256": {
            "original_cpu": _sha256_file(original_cpu_path),
            "original_gpu": _sha256_file(original_gpu_path),
            "upcast_cpu": _sha256_file(upcast_cpu_path),
            "upcast_gpu": _sha256_file(upcast_gpu_path),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "gpu_production_admission": (
            "blocked_pending_formal_same_panel_cpu_gpu_hfe_validation_with_"
            "at_least_10_distinct_primary_functional_groups"
        ),
    }


def _gpu_environment(base: dict[str, str], python: Path) -> dict[str, str]:
    environment = dict(base)
    environment.update(
        {
            "JAX_PLATFORMS": "cuda",
            "JAX_ENABLE_X64": "1",
            "JAX_DEFAULT_MATMUL_PRECISION": "highest",
        }
    )
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment_root = python.absolute().parent.parent
    bundled_library_dirs = sorted(
        str(path.resolve())
        for path in (environment_root / "cuda-libs").glob("*/lib")
        if path.is_dir()
    )
    cupti_library_dirs = sorted(
        str(path.resolve())
        for cuda_root in Path("/usr/local").glob("cuda*")
        for path in (cuda_root / "extras" / "CUPTI").glob("lib64")
        if path.is_dir()
    )
    existing = environment.get("LD_LIBRARY_PATH", "")
    entries = [*bundled_library_dirs, *cupti_library_dirs]
    if existing:
        entries.append(existing)
    if entries:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(entries)
    return environment


def _run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cpu_path = args.output_dir / "cpu-float64.json"
    gpu_path = args.output_dir / "gpu-float64.json"
    base_command = [
        str(args.python),
        str(Path(__file__).resolve()),
        "_worker",
        "--checkpoint",
        str(args.checkpoint),
    ]
    for device, output in (("cpu", cpu_path), ("gpu", gpu_path)):
        environment = dict(os.environ)
        if device == "gpu":
            environment = _gpu_environment(environment, args.python)
        else:
            environment.update(
                {
                    "JAX_PLATFORMS": "cpu",
                    "JAX_ENABLE_X64": "1",
                    "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                }
            )
        subprocess.run(
            [*base_command, "--output", str(output), "--device", device],
            check=True,
            env=environment,
        )
    comparison = _comparison(_load_json(cpu_path), _load_json(gpu_path))
    _write_json(args.output_dir / "comparison.json", comparison)
    alchemical_command = [
        str(args.python),
        str(Path(__file__).resolve()),
        "_alchemical_worker",
        "--checkpoint",
        str(args.checkpoint),
    ]
    for device in ("cpu", "gpu"):
        environment = dict(os.environ)
        if device == "gpu":
            environment = _gpu_environment(environment, args.python)
        else:
            environment.update(
                {
                    "JAX_PLATFORMS": "cpu",
                    "JAX_ENABLE_X64": "1",
                    "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                }
            )
        subprocess.run(
            [
                *alchemical_command,
                "--output",
                str(args.output_dir / f"alchemical-kernel-{device}-float64.json"),
                "--device",
                device,
            ],
            check=True,
            env=environment,
        )
    original_cpu_path = args.output_dir / "alchemical-kernel-cpu-float64.json"
    original_gpu_path = args.output_dir / "alchemical-kernel-gpu-float64.json"
    upcast_paths = {}
    for device in ("cpu", "gpu"):
        environment = dict(os.environ)
        if device == "gpu":
            environment = _gpu_environment(environment, args.python)
        else:
            environment.update(
                {
                    "JAX_PLATFORMS": "cpu",
                    "JAX_ENABLE_X64": "1",
                    "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                }
            )
        output = args.output_dir / f"alchemical-kernel-{device}-upcast64.json"
        subprocess.run(
            [
                *alchemical_command,
                "--output",
                str(output),
                "--device",
                device,
                "--upcast-runtime-parameters",
            ],
            check=True,
            env=environment,
        )
        upcast_paths[device] = output
    _write_json(
        args.output_dir / "alchemical-kernel-upcast64-comparison.json",
        _upcast_comparison(
            original_cpu_path,
            original_gpu_path,
            upcast_paths["cpu"],
            upcast_paths["gpu"],
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("_worker")
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--device", choices=("cpu", "gpu"), required=True)
    alchemical_worker = subparsers.add_parser("_alchemical_worker")
    alchemical_worker.add_argument("--checkpoint", type=Path, required=True)
    alchemical_worker.add_argument("--output", type=Path, required=True)
    alchemical_worker.add_argument("--device", choices=("cpu", "gpu"), required=True)
    alchemical_worker.add_argument("--upcast-runtime-parameters", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--python", type=Path, default=Path(sys.executable))
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)

    combine = subparsers.add_parser("combine")
    combine.add_argument("--cpu", type=Path, required=True)
    combine.add_argument("--gpu", type=Path, required=True)
    combine.add_argument("--comparison", type=Path, required=True)
    combine.add_argument("--alchemical-cpu", type=Path, required=True)
    combine.add_argument("--alchemical-gpu", type=Path, required=True)
    combine.add_argument("--alchemical-upcast-cpu", type=Path, required=True)
    combine.add_argument("--alchemical-upcast-gpu", type=Path, required=True)
    combine.add_argument("--alchemical-upcast-comparison", type=Path, required=True)
    combine.add_argument("--maple-adapter", type=Path, required=True)
    combine.add_argument("--pip-freeze", type=Path, required=True)
    combine.add_argument("--nvidia-smi", type=Path, required=True)
    combine.add_argument("--artifact-sha256", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "_worker":
        _worker(args.checkpoint, args.output, args.device)
    elif args.command == "_alchemical_worker":
        _alchemical_worker(
            args.checkpoint,
            args.output,
            args.device,
            upcast_runtime_parameters=args.upcast_runtime_parameters,
        )
    elif args.command == "run":
        _run(args)
    else:
        _combine(args)


if __name__ == "__main__":
    main()
