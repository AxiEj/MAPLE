#!/usr/bin/env python3
"""Reproduce the MACE-OFF23-SC CPU/CUDA no-loss rejection audit.

The single official water box is sufficient to reject literal parity after a
nonzero difference is observed.  It is not an experimental-accuracy panel and
can never admit GPU execution or support a solvation-accuracy claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

CHECKPOINT_REPOSITORY = "https://github.com/jharrymoore/MACE-OFF23-SC"
CHECKPOINT_REVISION = "efbb20c930462bc37a247998a27814dff35d474f"
CHECKPOINT_TREE = "25c1aa63e7960b71c7a5c0b2d7be4739cf621c94"
CHECKPOINT_RELATIVE_PATH = "MACE-OFF23-SC_swa.model"
CHECKPOINT_SIZE = 10_360_353
CHECKPOINT_SHA256 = "32c9fb51704f96da855c67e0cdc9894f3e41694e98b7a0ed8813388b9f21db33"

PROTOCOL_REPOSITORY = "https://github.com/jharrymoore/mace-md"
PROTOCOL_REVISION = "c3056287622ba18f9b905e9df45affdff46cb147"
PROTOCOL_TREE = "a0f401a102c5f1614174a6f7cd0e03d470c955f9"
WATERBOX_RELATIVE_PATH = "examples/example_data/waterbox.xyz"
WATERBOX_SIZE = 30_371
WATERBOX_SHA256 = "a052257f5f9c068884ec7527d6dd41d05a7c3705b729e7d9703543890931ec61"

AUDITED_ON = "2026-07-30"
ARTIFACT_SCHEMA_VERSION = 1
COMPARISON_MANIFEST_ID = "mace-off23-sc-official-waterbox-float64-cpu-cuda-v1"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
TIMING_REPETITIONS = 3


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _normalized_repository(value: str) -> str:
    normalized = value.strip().removesuffix("/").removesuffix(".git")
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.split(":", 1)[1]
    return normalized


def _verify_checkout(
    root: Path,
    *,
    repository: str,
    revision: str,
    tree: str,
) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"Verified upstream checkout does not exist: {root}")
    if _git(root, "rev-parse", "HEAD") != revision:
        raise ValueError(f"Upstream checkout revision mismatch: {root}")
    if _git(root, "rev-parse", "HEAD^{tree}") != tree:
        raise ValueError(f"Upstream checkout tree mismatch: {root}")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"Upstream checkout must be clean: {root}")
    if _normalized_repository(_git(root, "remote", "get-url", "origin")) != (
        _normalized_repository(repository)
    ):
        raise ValueError(f"Upstream checkout origin mismatch: {root}")


def _verify_file(path: Path, *, size: int, sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Audited file does not exist: {path}")
    if path.stat().st_size != size:
        raise ValueError(f"Audited file size mismatch: {path}")
    actual = _sha256_file(path)
    if actual != sha256:
        raise ValueError(
            f"Audited file SHA256 mismatch for {path}: expected {sha256}, got {actual}."
        )


def _set_precision_controls(torch: Any) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
    return {
        "accelerator_dtype": "float64",
        "autocast": False,
        "bfloat16": False,
        "fast_math": False,
        "float16": False,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "reduced_matmul": False,
        "reference_dtype": "float64",
        "relaxed_convergence": False,
        "same_scalar_precision": True,
        "shortened_sampling": False,
        "tf32": False,
    }


def _finite_scalar(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _evaluate(
    *,
    atoms: Any,
    checkpoint: Path,
    device: str,
    torch: Any,
    calculator_class: Any,
) -> dict[str, Any]:
    import numpy as np
    from ase.calculators.calculator import all_changes

    calculator = calculator_class(
        model_paths=str(checkpoint),
        device=device,
        default_dtype="float64",
    )

    def evaluate_once() -> tuple[float, np.ndarray]:
        calculator.calculate(
            atoms,
            properties=["energy", "forces"],
            system_changes=all_changes,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        energy = _finite_scalar(
            calculator.results["energy"],
            label=f"{device} energy",
        )
        forces = np.asarray(calculator.results["forces"], dtype=np.float64)
        if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
            raise ValueError(f"{device} forces must be finite with shape (N, 3).")
        return energy, forces

    evaluate_once()
    observations = []
    timings = []
    for _ in range(TIMING_REPETITIONS):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        energy, forces = evaluate_once()
        elapsed = time.perf_counter() - started
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise RuntimeError("Timing observation must be finite and positive.")
        observations.append((energy, forces))
        timings.append(elapsed)

    reference_energy, reference_forces = observations[0]
    for energy, forces in observations[1:]:
        if energy != reference_energy or not np.array_equal(forces, reference_forces):
            raise RuntimeError(f"{device} output changed across repetitions.")
    return {
        "energy_ev": reference_energy,
        "forces_ev_per_angstrom": reference_forces.reshape(-1).tolist(),
        "timings_seconds": timings,
    }


def _output_sha256(
    *,
    observable: str,
    record_id: str,
    role: str,
    values: list[float],
) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "observable": observable,
                "record_id": record_id,
                "role": role,
                "values": values,
            }
        )
    )


def _runtime_metadata(torch: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "accelerator_backend": "cuda",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_runtime_version": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "gpu": {
            "compute_capability": f"{properties.major}.{properties.minor}",
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "uuid": f"GPU-{properties.uuid}",
        },
        "package_versions": {
            "ase": importlib.metadata.version("ase"),
            "mace-torch": importlib.metadata.version("mace-torch"),
            "numpy": importlib.metadata.version("numpy"),
            "torch": torch.__version__,
        },
        "reference_backend": "cpu",
    }


def run_audit(
    *,
    checkpoint_root: Path,
    protocol_root: Path,
) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != CUBLAS_WORKSPACE_CONFIG:
        raise ValueError(
            f"CUBLAS_WORKSPACE_CONFIG must equal {CUBLAS_WORKSPACE_CONFIG}."
        )

    _verify_checkout(
        checkpoint_root,
        repository=CHECKPOINT_REPOSITORY,
        revision=CHECKPOINT_REVISION,
        tree=CHECKPOINT_TREE,
    )
    _verify_checkout(
        protocol_root,
        repository=PROTOCOL_REPOSITORY,
        revision=PROTOCOL_REVISION,
        tree=PROTOCOL_TREE,
    )
    checkpoint = checkpoint_root / CHECKPOINT_RELATIVE_PATH
    waterbox = protocol_root / WATERBOX_RELATIVE_PATH
    _verify_file(
        checkpoint,
        size=CHECKPOINT_SIZE,
        sha256=CHECKPOINT_SHA256,
    )
    _verify_file(
        waterbox,
        size=WATERBOX_SIZE,
        sha256=WATERBOX_SHA256,
    )

    import numpy as np
    import torch
    from ase.io import read
    from mace.calculators import MACECalculator

    if not torch.cuda.is_available():
        raise RuntimeError("MACE-OFF23-SC GPU audit requires physical CUDA.")
    precision_controls = _set_precision_controls(torch)

    atoms = read(waterbox)
    if (
        len(atoms) != 192
        or atoms.get_chemical_formula() != "H128O64"
        or not np.asarray(atoms.pbc, dtype=bool).all()
        or not np.allclose(
            atoms.cell.lengths(),
            [12.442877769470215] * 3,
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise ValueError(
            "Official water-box identity does not match the audit contract."
        )

    reference = _evaluate(
        atoms=atoms,
        checkpoint=checkpoint,
        device="cpu",
        torch=torch,
        calculator_class=MACECalculator,
    )
    accelerator = _evaluate(
        atoms=atoms,
        checkpoint=checkpoint,
        device="cuda",
        torch=torch,
        calculator_class=MACECalculator,
    )

    reference_forces = np.asarray(
        reference["forces_ev_per_angstrom"],
        dtype=np.float64,
    )
    accelerator_forces = np.asarray(
        accelerator["forces_ev_per_angstrom"],
        dtype=np.float64,
    )
    force_differences = np.abs(accelerator_forces - reference_forces)
    energy_difference = abs(accelerator["energy_ev"] - reference["energy_ev"])
    energy_exact = accelerator["energy_ev"] == reference["energy_ev"]
    forces_exact = np.array_equal(accelerator_forces, reference_forces)
    record_id = "float64:official-waterbox-192"

    energy_reference_sha = _output_sha256(
        observable="energy",
        record_id=record_id,
        role="reference",
        values=[reference["energy_ev"]],
    )
    energy_accelerator_sha = _output_sha256(
        observable="energy",
        record_id=record_id,
        role="accelerator",
        values=[accelerator["energy_ev"]],
    )
    forces_reference_sha = _output_sha256(
        observable="forces",
        record_id=record_id,
        role="reference",
        values=reference["forces_ev_per_angstrom"],
    )
    forces_accelerator_sha = _output_sha256(
        observable="forces",
        record_id=record_id,
        role="accelerator",
        values=accelerator["forces_ev_per_angstrom"],
    )
    configuration_sha = _sha256_bytes(
        _canonical_bytes(
            {
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "compute_dtype": "float64",
                "input_payload_sha256": WATERBOX_SHA256,
                "pbc": [True, True, True],
                "record_id": record_id,
            }
        )
    )
    manifest = {
        "checkpoint_sha256s": [CHECKPOINT_SHA256],
        "comparison_manifest_id": COMPARISON_MANIFEST_ID,
        "model_id": "mace-off23-sc",
        "model_version": "MACE-OFF23-SC_swa",
        "observable_counts": {"energy": 1, "forces": 576},
        "records": [
            {
                "configuration_sha256": configuration_sha,
                "observable_output_sha256s": {
                    "energy": {
                        "accelerator": energy_accelerator_sha,
                        "reference": energy_reference_sha,
                    },
                    "forces": {
                        "accelerator": forces_accelerator_sha,
                        "reference": forces_reference_sha,
                    },
                },
                "panel_id": WATERBOX_SHA256,
                "record_id": record_id,
            }
        ],
        "schema_version": 1,
        "task_modes": ["sp:default", "opt:default", "md:default"],
    }
    reference_median = statistics.median(reference["timings_seconds"])
    accelerator_median = statistics.median(accelerator["timings_seconds"])
    no_loss = energy_exact and forces_exact

    return {
        "acceptance_eligible": False,
        "accuracy_evaluation_performed": False,
        "artifact_kind": (
            "negative_physical_hardware_observation_not_gpu_admission_evidence"
        ),
        "audited_on": AUDITED_ON,
        "comparison_manifest": manifest,
        "comparison_manifest_sha256": _sha256_bytes(_canonical_bytes(manifest)),
        "gpu_admission": {
            "accuracy_panel_eligible": False,
            "allowed_inference_modes": [],
            "allowed_tasks": [],
            "card_evidence_must_remain_null": True,
            "no_loss_parity_verified": False,
            "smaller_panel_can_reject_but_never_unlock": True,
            "status": "blocked_nonzero_cpu_cuda_energy_and_force_differences",
        },
        "identity": {
            "checkpoint": {
                "path": CHECKPOINT_RELATIVE_PATH,
                "sha256": CHECKPOINT_SHA256,
                "size_bytes": CHECKPOINT_SIZE,
            },
            "checkpoint_repository": CHECKPOINT_REPOSITORY,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "checkpoint_tree": CHECKPOINT_TREE,
            "protocol_repository": PROTOCOL_REPOSITORY,
            "protocol_revision": PROTOCOL_REVISION,
            "protocol_tree": PROTOCOL_TREE,
            "waterbox": {
                "atom_count": 192,
                "cell_lengths_angstrom": atoms.cell.lengths().tolist(),
                "formula": atoms.get_chemical_formula(),
                "path": WATERBOX_RELATIVE_PATH,
                "pbc": atoms.pbc.tolist(),
                "sha256": WATERBOX_SHA256,
                "size_bytes": WATERBOX_SIZE,
            },
        },
        "observables": {
            "energy": {
                "accelerator_sha256": energy_accelerator_sha,
                "accelerator_values_ev": [accelerator["energy_ev"]],
                "comparison_count": 1,
                "nonzero_difference_count": int(not energy_exact),
                "observed_maximum_abs_difference_ev": energy_difference,
                "passed_exact_equality": energy_exact,
                "reference_sha256": energy_reference_sha,
                "reference_values_ev": [reference["energy_ev"]],
                "threshold_ev": 0.0,
            },
            "forces": {
                "accelerator_sha256": forces_accelerator_sha,
                "accelerator_values_ev_per_angstrom": accelerator[
                    "forces_ev_per_angstrom"
                ],
                "comparison_count": int(force_differences.size),
                "nonzero_difference_count": int(np.count_nonzero(force_differences)),
                "observed_maximum_abs_difference_ev_per_angstrom": float(
                    force_differences.max()
                ),
                "passed_exact_equality": forces_exact,
                "reference_sha256": forces_reference_sha,
                "reference_values_ev_per_angstrom": reference["forces_ev_per_angstrom"],
                "threshold_ev_per_angstrom": 0.0,
            },
        },
        "panel": {
            "accuracy_metric_reporting_allowed": False,
            "experimental_labels_used": False,
            "functional_group_accuracy_gate": {
                "minimum_distinct_functional_groups": 10,
                "minimum_record_count": 10,
                "observed_distinct_functional_groups": 0,
                "observed_record_count": 0,
                "passes": False,
                "record_assignments_predeclared": False,
            },
            "purpose": "numerical_cpu_cuda_rejection_only",
            "record_count": 1,
        },
        "precision_controls": precision_controls,
        "runtime": _runtime_metadata(torch),
        "scientific_scope": {
            "absolute_solvation_free_energy_evaluated": False,
            "experimental_accuracy_evaluated": False,
            "no_fitting_or_calibration_performed": True,
            "no_training_or_fine_tuning_performed": True,
            "performance_can_override_scientific_failure": False,
            "strict_literal_zero_loss_required": True,
        },
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "timing_diagnostic_not_an_admission_observable": {
            "accelerator_median_seconds": accelerator_median,
            "accelerator_repetitions_seconds": accelerator["timings_seconds"],
            "reference_median_seconds": reference_median,
            "reference_repetitions_seconds": reference["timings_seconds"],
            "speedup_reference_over_accelerator": (
                reference_median / accelerator_median
            ),
        },
        "verdict": "pass_no_loss" if no_loss else "fail_no_loss",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_audit(
        checkpoint_root=args.checkpoint_root.expanduser().resolve(),
        protocol_root=args.protocol_root.expanduser().resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
