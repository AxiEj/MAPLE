#!/usr/bin/env python3
"""Audit AniSolv compact CPU/CUDA parity without admitting the GPU path.

This is a physical-hardware rejection audit, not an accelerator approval
artifact.  It runs the exact pinned upstream six-example water panel on CPU
and CUDA at the same float32 and float64 dtypes with reduced-precision and
fast modes disabled.  Literal scalar equality is required.  The six-row panel
does not calculate experimental accuracy and can reject, but never unlock, an
accelerator.  Timing is diagnostic only and can never override failure.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SIBLING_AUDIT = Path(__file__).with_name("run_anisolv_compact_sample_audit.py")
_SPEC = importlib.util.spec_from_file_location(
    "maple_anisolv_sample_audit",
    SIBLING_AUDIT,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load source-binding helpers from {SIBLING_AUDIT}.")
SOURCE_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(SOURCE_AUDIT)

AUDITED_ON = "2026-07-30"
AUDIT_SCHEMA_VERSION = 1
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
TIMING_REPEATS = 3
DTYPE_NAMES = ("float64", "float32")
REFERENCE_SAMPLE_ARTIFACT = (
    "benchmarks/anisolv-compact-upstream-sample-audit-2026-07-30.json"
)
REFERENCE_SAMPLE_ARTIFACT_SHA256 = (
    "9018364bef21b7d672f0a1404954b26702a096d0be4da266af719195349b07f6"
)


def _output_sha256(
    record_id: str,
    role: str,
    energy_ev: float,
) -> str:
    payload = {
        "observable": "energy",
        "record_id": record_id,
        "role": role,
        "values": [energy_ev],
    }
    return SOURCE_AUDIT._sha256_bytes(SOURCE_AUDIT._canonical_bytes(payload))


def _configuration_sha256(
    item: dict[str, Any],
    *,
    dtype_name: str,
) -> str:
    payload = {
        "atomic_numbers": item["atomic_numbers"],
        "charge": 0,
        "dtype": dtype_name,
        "inference_settings": "default",
        "multiplicity": 1,
        "positions_angstrom": item["positions_angstrom"],
        "solvent": "water",
    }
    return SOURCE_AUDIT._sha256_bytes(SOURCE_AUDIT._canonical_bytes(payload))


def _finite_sequence(values: Any, *, label: str) -> list[float]:
    result = [
        SOURCE_AUDIT._finite_float(value, label=f"{label}[{index}]")
        for index, value in enumerate(values)
    ]
    if not result:
        raise ValueError(f"{label} must not be empty.")
    return result


def _compare_lane(
    *,
    dtype_name: str,
    panel: list[dict[str, Any]],
    reference_energies_ev: list[float],
    accelerator_energies_ev: list[float],
    reference_timing_seconds: list[float],
    accelerator_timing_seconds: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference_energies_ev = _finite_sequence(
        reference_energies_ev,
        label=f"{dtype_name} CPU energies",
    )
    accelerator_energies_ev = _finite_sequence(
        accelerator_energies_ev,
        label=f"{dtype_name} CUDA energies",
    )
    if len(panel) != len(reference_energies_ev) or len(panel) != len(
        accelerator_energies_ev
    ):
        raise ValueError("CPU/CUDA outputs do not cover the complete frozen panel.")

    paired_outputs = []
    manifest_records = []
    absolute_energy_differences = []
    for item, reference_energy, accelerator_energy in zip(
        panel, reference_energies_ev, accelerator_energies_ev
    ):
        record_id = f"{dtype_name}:{item['g2_name']}"
        reference_sha256 = _output_sha256(
            record_id,
            "reference",
            reference_energy,
        )
        accelerator_sha256 = _output_sha256(
            record_id,
            "accelerator",
            accelerator_energy,
        )
        absolute_difference = abs(accelerator_energy - reference_energy)
        absolute_energy_differences.append(absolute_difference)
        paired_outputs.append(
            {
                "accelerator_energy_ev": accelerator_energy,
                "accelerator_sha256": accelerator_sha256,
                "absolute_energy_difference_ev": absolute_difference,
                "energy_exactly_equal": accelerator_energy == reference_energy,
                "label": item["label"],
                "record_id": record_id,
                "reference_energy_ev": reference_energy,
                "reference_sha256": reference_sha256,
            }
        )
        manifest_records.append(
            {
                "configuration_sha256": _configuration_sha256(
                    item,
                    dtype_name=dtype_name,
                ),
                "observable_output_sha256s": {
                    "energy": {
                        "accelerator": accelerator_sha256,
                        "reference": reference_sha256,
                    }
                },
                "panel_id": SOURCE_AUDIT.PANEL_IDENTITY_SHA256,
                "record_id": record_id,
            }
        )

    energy_exact_parity = all(
        comparison["energy_exactly_equal"] for comparison in paired_outputs
    )
    lane_passed = energy_exact_parity

    reference_timing_seconds = _finite_sequence(
        reference_timing_seconds,
        label=f"{dtype_name} CPU timings",
    )
    accelerator_timing_seconds = _finite_sequence(
        accelerator_timing_seconds,
        label=f"{dtype_name} CUDA timings",
    )
    reference_median = statistics.median(reference_timing_seconds)
    accelerator_median = statistics.median(accelerator_timing_seconds)
    if reference_median <= 0.0 or accelerator_median <= 0.0:
        raise ValueError("CPU/CUDA timing observations must be positive.")
    speedup = reference_median / accelerator_median

    return (
        {
            "energy_parity": {
                "comparison_count": len(paired_outputs),
                "nonzero_difference_count": sum(
                    not comparison["energy_exactly_equal"]
                    for comparison in paired_outputs
                ),
                "observed_maximum_abs_difference_ev": max(absolute_energy_differences),
                "paired_outputs": paired_outputs,
                "passed_exact_equality": energy_exact_parity,
                "threshold_ev": 0.0,
            },
            "no_loss_passed": lane_passed,
            "precision_controls": {
                "accelerator_dtype": dtype_name,
                "autocast": False,
                "bfloat16": False,
                "fast_math": False,
                "float16": False,
                "matmul_precision": "highest",
                "reduced_matmul": False,
                "reference_dtype": dtype_name,
                "relaxed_convergence": False,
                "same_scalar_precision": True,
                "shortened_sampling": False,
                "tf32": False,
            },
            "timing_diagnostic_not_an_admission_observable": {
                "accelerator_median_seconds": accelerator_median,
                "accelerator_repetitions_seconds": accelerator_timing_seconds,
                "reference_median_seconds": reference_median,
                "reference_repetitions_seconds": reference_timing_seconds,
                "speedup_reference_over_accelerator": speedup,
                "useful_acceleration_observed": speedup > 1.0,
            },
            "verdict": "pass_no_loss" if lane_passed else "fail_no_loss",
        },
        manifest_records,
    )


def _evaluate_panel(
    *,
    sample: Any,
    torch: Any,
    checkpoint: str,
    panel: list[dict[str, Any]],
    device: str,
    dtype: Any,
) -> tuple[list[float], list[float]]:
    def evaluate_once() -> list[float]:
        energies = []
        for item in panel:
            energy_ev, _forces = sample.predict_solvation_energy(
                (item["atomic_numbers"], item["positions_angstrom"]),
                charge=0,
                spin=1,
                solvent="water",
                checkpoint=checkpoint,
                device=device,
                dtype=dtype,
                inference_settings="default",
            )
            energies.append(
                SOURCE_AUDIT._finite_float(
                    energy_ev,
                    label=f"{device} {dtype} {item['label']} energy",
                )
            )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        return energies

    evaluate_once()
    observations = []
    timings = []
    for _ in range(TIMING_REPEATS):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        energies = evaluate_once()
        elapsed = time.perf_counter() - started
        if elapsed <= 0.0 or not math.isfinite(elapsed):
            raise RuntimeError("Observed a non-positive or non-finite timing.")
        observations.append(energies)
        timings.append(elapsed)
    reference = observations[0]
    if any(values != reference for values in observations[1:]):
        raise RuntimeError(
            f"{device} {dtype} outputs changed across deterministic repetitions."
        )
    return reference, timings


def _validate_precision_state(state: dict[str, Any]) -> None:
    expected_false = (
        "autocast_cpu",
        "autocast_cuda",
        "bf16_reduced_precision_reduction",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "fp16_accumulation",
        "fp16_reduced_precision_reduction",
    )
    if state.get("float32_matmul_precision") != "highest" or any(
        state.get(name) is not False for name in expected_false
    ):
        raise RuntimeError("Reduced-precision CUDA controls were not fully disabled.")


def _driver_fingerprint(torch: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,driver_version,name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("The GPU audit requires a working nvidia-smi.") from exc
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU records.")
    selected = None
    expected_uuid = f"GPU-{properties.uuid}"
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 6:
            raise RuntimeError("nvidia-smi returned an unexpected CSV shape.")
        if fields[1] == expected_uuid:
            selected = fields
            break
    if selected is None:
        raise RuntimeError("Torch CUDA device 0 was not found in nvidia-smi output.")
    return {
        "compute_capability": f"{properties.major}.{properties.minor}",
        "cuda_device_index": int(selected[0]),
        "driver_version": selected[2],
        "name": properties.name,
        "pci_bus_id": properties.pci_bus_id,
        "pci_device_id": properties.pci_device_id,
        "total_memory_bytes": properties.total_memory,
        "uuid": selected[1],
    }


def _cpu_model_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _loaded_model_fingerprints(source_root: Path) -> list[dict[str, Any]]:
    predict_module = importlib.import_module("anisolv.predict")
    cache = getattr(predict_module, "_MODEL_CACHE", None)
    if not isinstance(cache, dict):
        raise RuntimeError("AniSolv official model cache is unavailable.")
    expected_checkpoint = str((source_root / SOURCE_AUDIT.CHECKPOINT["path"]).resolve())
    expected_pairs = {
        ("cpu", "torch.float32"),
        ("cpu", "torch.float64"),
        ("cuda:0", "torch.float32"),
        ("cuda:0", "torch.float64"),
    }
    fingerprints = []
    observed_pairs = set()
    for cache_key, model in cache.items():
        if not isinstance(cache_key, tuple) or len(cache_key) < 4:
            raise RuntimeError("AniSolv model cache key has an unexpected shape.")
        if cache_key[0] != expected_checkpoint:
            raise RuntimeError("AniSolv GPU audit loaded an unpinned checkpoint.")
        pair = (str(cache_key[1]), str(cache_key[2]))
        observed_pairs.add(pair)
        settings = str(cache_key[3])
        for required in (
            "tf32=False",
            "compile=False",
            "merge_mole=False",
            "execution_mode='general'",
        ):
            if required not in settings:
                raise RuntimeError(
                    f"AniSolv inference settings lost required token {required!r}."
                )
        parameter_devices = sorted({str(value.device) for value in model.parameters()})
        parameter_dtypes = sorted({str(value.dtype) for value in model.parameters()})
        buffer_devices = sorted({str(value.device) for value in model.buffers()})
        buffer_dtypes = sorted({str(value.dtype) for value in model.buffers()})
        expected_device = "cpu" if pair[0] == "cpu" else "cuda:0"
        if parameter_devices != [expected_device] or parameter_dtypes != [pair[1]]:
            raise RuntimeError(
                "AniSolv cached model device/dtype does not match its key."
            )
        if buffer_devices != [expected_device] or any(
            dtype not in {pair[1], "torch.int64"} for dtype in buffer_dtypes
        ):
            raise RuntimeError(
                "AniSolv cached model buffers do not match the audited device/dtype."
            )
        fingerprints.append(
            {
                "buffer_devices": buffer_devices,
                "buffer_dtypes": buffer_dtypes,
                "device": pair[0],
                "dtype": pair[1],
                "inference_settings": settings,
                "parameter_devices": parameter_devices,
                "parameter_dtypes": parameter_dtypes,
            }
        )
    if observed_pairs != expected_pairs:
        raise RuntimeError(
            f"AniSolv GPU audit expected model lanes {sorted(expected_pairs)!r}, "
            f"got {sorted(observed_pairs)!r}."
        )
    return sorted(
        fingerprints,
        key=lambda item: (item["dtype"], item["device"]),
    )


def audit_gpu_no_loss(upstream_root: Path) -> dict[str, Any]:
    """Run the pinned CPU/CUDA audit in a fresh, source-isolated process."""

    if "torch" in sys.modules:
        raise RuntimeError(
            "AniSolv GPU audit requires a fresh interpreter so deterministic "
            "CUDA controls are set before Torch initialization."
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

    import ase
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("AniSolv GPU audit requires a physical CUDA device.")
    torch.cuda.set_device(0)
    upstream_root = upstream_root.expanduser().resolve()
    SOURCE_AUDIT.verify_upstream_identity(upstream_root)
    dependency_origins = {
        "ase": SOURCE_AUDIT._verify_dependency_origin(ase, label="ASE"),
        "numpy": SOURCE_AUDIT._verify_dependency_origin(np, label="NumPy"),
        "torch": SOURCE_AUDIT._verify_dependency_origin(torch, label="PyTorch"),
    }
    original_path = list(sys.path)
    original_pycache_prefix = sys.pycache_prefix

    with tempfile.TemporaryDirectory(prefix="maple-anisolv-gpu-audit-") as temporary:
        isolated_parent = Path(temporary)
        source_root = isolated_parent / "anisolv"
        SOURCE_AUDIT._materialize_pinned_tree(upstream_root, source_root)
        pycache_root = isolated_parent / "pycache"
        pycache_root.mkdir()
        sys.pycache_prefix = str(pycache_root)
        try:
            precision_state = SOURCE_AUDIT._configure_no_reduced_precision(torch)
            _validate_precision_state(precision_state)
            sample = SOURCE_AUDIT._load_upstream_sample(
                source_root,
                isolated_parent,
            )
            panel, panel_sha256 = SOURCE_AUDIT._panel_identity(sample)
            checkpoint = str((source_root / SOURCE_AUDIT.CHECKPOINT["path"]).resolve())

            lanes = {}
            manifest_records = []
            for dtype_name in DTYPE_NAMES:
                dtype = getattr(torch, dtype_name)
                reference_energies, reference_timings = _evaluate_panel(
                    sample=sample,
                    torch=torch,
                    checkpoint=checkpoint,
                    panel=panel,
                    device="cpu",
                    dtype=dtype,
                )
                accelerator_energies, accelerator_timings = _evaluate_panel(
                    sample=sample,
                    torch=torch,
                    checkpoint=checkpoint,
                    panel=panel,
                    device="cuda:0",
                    dtype=dtype,
                )
                lane, lane_manifest_records = _compare_lane(
                    dtype_name=dtype_name,
                    panel=panel,
                    reference_energies_ev=reference_energies,
                    accelerator_energies_ev=accelerator_energies,
                    reference_timing_seconds=reference_timings,
                    accelerator_timing_seconds=accelerator_timings,
                )
                lanes[dtype_name] = lane
                manifest_records.extend(lane_manifest_records)

            comparison_manifest = {
                "checkpoint_sha256s": [SOURCE_AUDIT.CHECKPOINT["sha256"]],
                "comparison_manifest_id": (
                    "anisolv-compact-upstream-water-six-cpu-cuda-v1"
                ),
                "model_id": "anisolv-compact",
                "model_version": "0.2.0-66f36dc",
                "observable_counts": {"energy": len(manifest_records)},
                "records": manifest_records,
                "schema_version": 1,
                "task_modes": ["sp:default"],
            }
            comparison_manifest_sha256 = SOURCE_AUDIT._sha256_bytes(
                SOURCE_AUDIT._canonical_bytes(comparison_manifest)
            )
            all_lanes_passed = all(lane["no_loss_passed"] for lane in lanes.values())
            module_origins = SOURCE_AUDIT._verified_anisolv_module_origins(source_root)
            return {
                "acceptance_eligible": False,
                "artifact_kind": (
                    "negative_physical_hardware_observation_not_gpu_admission_evidence"
                ),
                "audited_on": AUDITED_ON,
                "comparison_manifest": comparison_manifest,
                "comparison_manifest_sha256": comparison_manifest_sha256,
                "gpu_admission": {
                    "accuracy_panel_eligible": False,
                    "allowed_inference_modes": [],
                    "allowed_tasks": [],
                    "card_evidence_must_remain_null": True,
                    "no_loss_parity_verified": False,
                    "smaller_panel_can_reject_but_never_unlock": True,
                    "status": (
                        "blocked_ineligible_six_record_accuracy_panel"
                        if all_lanes_passed
                        else "blocked_nonzero_cpu_cuda_scalar_difference"
                    ),
                },
                "identity": {
                    "checkpoint": {
                        **SOURCE_AUDIT.CHECKPOINT,
                        "verified": True,
                    },
                    "repository": SOURCE_AUDIT.UPSTREAM_REPOSITORY,
                    "revision": SOURCE_AUDIT.UPSTREAM_REVISION,
                    "tree": SOURCE_AUDIT.UPSTREAM_TREE,
                },
                "lanes": lanes,
                "model_id": "anisolv-compact",
                "model_runtime_fingerprints": _loaded_model_fingerprints(source_root),
                "model_version": "0.2.0-66f36dc",
                "panel": {
                    "accuracy_metric_reporting_allowed": False,
                    "functional_group_accuracy_gate": {
                        "functional_group_taxonomy": None,
                        "minimum_distinct_functional_groups": 10,
                        "minimum_record_count": 10,
                        "observed_distinct_functional_groups": 0,
                        "observed_record_count": len(panel),
                        "passes": False,
                        "record_assignments_predeclared": False,
                    },
                    "geometry_and_source_row_sha256": panel_sha256,
                    "experimental_label_values_used_for_identity": False,
                    "record_count_per_dtype": len(panel),
                    "reference_sample_artifact": REFERENCE_SAMPLE_ARTIFACT,
                    "reference_sample_artifact_sha256": (
                        REFERENCE_SAMPLE_ARTIFACT_SHA256
                    ),
                    "training_overlap": "unknown",
                    "water_only": True,
                },
                "precision_backend_state": precision_state,
                "runtime": {
                    "accelerator_backend": "cuda",
                    "ase_version": ase.__version__,
                    "cublas_workspace_config": os.environ.get(
                        "CUBLAS_WORKSPACE_CONFIG"
                    ),
                    "cuda_runtime_version": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version(),
                    "dependency_origins_within_active_environment": dependency_origins,
                    "deterministic_algorithms": (
                        torch.are_deterministic_algorithms_enabled()
                    ),
                    "gpu": _driver_fingerprint(torch),
                    "host": {
                        "cpu_model": _cpu_model_name(),
                        "machine": platform.machine(),
                        "platform": platform.platform(),
                    },
                    "numpy_version": np.__version__,
                    "python_version": ".".join(map(str, sys.version_info[:3])),
                    "torch_force_no_weights_only_load": (
                        os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") == "1"
                    ),
                    "torch_version": torch.__version__,
                },
                "schema_version": AUDIT_SCHEMA_VERSION,
                "scientific_scope": {
                    "accuracy_metric_reporting_allowed": False,
                    "experimental_labels_used": False,
                    "independent_experimental_holdout": False,
                    "no_fitting_or_calibration_performed": True,
                    "no_training_or_fine_tuning_performed": True,
                    "performance_can_override_scientific_failure": False,
                    "strict_literal_zero_loss_required": True,
                    "upstream_forces_admitted": False,
                },
                "source_execution_isolation": {
                    "git_archive_from_verified_tree": True,
                    "isolated_import_parent": True,
                    "isolated_pycache_root": True,
                    "loaded_anisolv_module_origins": module_origins,
                    "loaded_anisolv_module_origins_sha256": (
                        SOURCE_AUDIT._sha256_bytes(
                            SOURCE_AUDIT._canonical_bytes(module_origins)
                        )
                    ),
                },
                "verdict": (
                    "panel_pass_no_loss_not_admission"
                    if all_lanes_passed
                    else "fail_no_loss"
                ),
            }
        finally:
            sys.path[:] = original_path
            sys.pycache_prefix = original_pycache_prefix
            for name in list(sys.modules):
                if name == "anisolv" or name.startswith("anisolv."):
                    del sys.modules[name]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        required=True,
        help="Clean checkout of the exact pinned Ant-on-knee/anisolv revision.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = audit_gpu_no_loss(args.upstream_root)
    SOURCE_AUDIT._write_json_atomic(args.output, payload)
    summary = {
        name: {
            "max_abs_energy_difference_ev": lane["energy_parity"][
                "observed_maximum_abs_difference_ev"
            ],
            "no_loss_passed": lane["no_loss_passed"],
            "speedup": lane["timing_diagnostic_not_an_admission_observable"][
                "speedup_reference_over_accelerator"
            ],
        }
        for name, lane in payload["lanes"].items()
    }
    print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
    print(f"GPU admission verdict: {payload['verdict']}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
