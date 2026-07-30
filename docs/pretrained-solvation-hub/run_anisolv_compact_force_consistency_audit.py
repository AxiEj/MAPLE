#!/usr/bin/env python3
"""Reproduce AniSolv compact's orientation-dependent upstream force defect."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

SIBLING_AUDIT = Path(__file__).with_name("run_anisolv_compact_sample_audit.py")
_SPEC = importlib.util.spec_from_file_location(
    "maple_anisolv_sample_audit", SIBLING_AUDIT
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load source-binding helpers from {SIBLING_AUDIT}.")
SOURCE_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(SOURCE_AUDIT)

AUDITED_ON = "2026-07-30"
FINITE_DIFFERENCE_STEP_ANGSTROM = 1.0e-5
FORCE_CONSISTENCY_TOLERANCE_EV_PER_ANGSTROM = 1.0e-6
ENERGY_ROTATION_TOLERANCE_EV = 1.0e-12
ROTATION_AXIS = [1.0, 2.0, 3.0]
ROTATION_ANGLE_RAD = 0.731
CASES = ("H2O", "CH3OH")


def _rotation_matrix(np):
    axis = np.asarray(ROTATION_AXIS, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cosine = math.cos(ROTATION_ANGLE_RAD)
    sine = math.sin(ROTATION_ANGLE_RAD)
    complement = 1.0 - cosine
    return np.asarray(
        [
            [
                cosine + x * x * complement,
                x * y * complement - z * sine,
                x * z * complement + y * sine,
            ],
            [
                y * x * complement + z * sine,
                cosine + y * y * complement,
                y * z * complement - x * sine,
            ],
            [
                z * x * complement - y * sine,
                z * y * complement + x * sine,
                cosine + z * z * complement,
            ],
        ],
        dtype=np.float64,
    )


def _evaluate(sample, torch, np, checkpoint: str, numbers, positions):
    energy, forces = sample.predict_solvation_energy(
        (numbers, positions.tolist()),
        charge=0,
        spin=1,
        solvent="water",
        checkpoint=checkpoint,
        device="cpu",
        dtype=torch.float64,
        inference_settings="default",
    )
    energy = SOURCE_AUDIT._finite_float(energy, label="force-probe energy")
    forces = np.asarray(forces, dtype=np.float64)
    if forces.shape != positions.shape or not np.isfinite(forces).all():
        raise ValueError("AniSolv force-probe output is non-finite or has wrong shape.")
    return energy, forces


def _finite_difference_forces(sample, torch, np, checkpoint, numbers, positions):
    forces = np.empty_like(positions, dtype=np.float64)
    for atom_index in range(positions.shape[0]):
        for axis_index in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom_index, axis_index] += FINITE_DIFFERENCE_STEP_ANGSTROM
            minus[atom_index, axis_index] -= FINITE_DIFFERENCE_STEP_ANGSTROM
            energy_plus, _ = _evaluate(sample, torch, np, checkpoint, numbers, plus)
            energy_minus, _ = _evaluate(sample, torch, np, checkpoint, numbers, minus)
            forces[atom_index, axis_index] = -(energy_plus - energy_minus) / (
                2.0 * FINITE_DIFFERENCE_STEP_ANGSTROM
            )
    if not np.isfinite(forces).all():
        raise ValueError("AniSolv finite-difference force probe is non-finite.")
    return forces


def _array_record(array, np) -> dict[str, Any]:
    rows = np.asarray(array, dtype=np.float64).tolist()
    return {
        "sha256": SOURCE_AUDIT._sha256_bytes(SOURCE_AUDIT._canonical_bytes(rows)),
        "values": rows,
    }


def _float64_model_fingerprint(source_root: Path) -> dict[str, Any]:
    predict_module = importlib.import_module("anisolv.predict")
    cache = getattr(predict_module, "_MODEL_CACHE", None)
    if not isinstance(cache, dict) or len(cache) != 1:
        raise RuntimeError("Expected one AniSolv float64 force-probe model.")
    cache_key, model = next(iter(cache.items()))
    expected_checkpoint = str((source_root / SOURCE_AUDIT.CHECKPOINT["path"]).resolve())
    if (
        not isinstance(cache_key, tuple)
        or cache_key[0] != expected_checkpoint
        or cache_key[1:3] != ("cpu", "torch.float64")
    ):
        raise RuntimeError("AniSolv force probe did not use pinned CPU float64.")
    parameter_devices = sorted({str(value.device) for value in model.parameters()})
    parameter_dtypes = sorted({str(value.dtype) for value in model.parameters()})
    if parameter_devices != ["cpu"] or parameter_dtypes != ["torch.float64"]:
        raise RuntimeError("AniSolv force-probe parameters are not CPU float64.")
    return {
        "cache_key": {
            "checkpoint": SOURCE_AUDIT.CHECKPOINT["path"],
            "device": str(cache_key[1]),
            "dtype": str(cache_key[2]),
            "inference_settings": str(cache_key[3]),
        },
        "parameter_devices": parameter_devices,
        "parameter_dtypes": parameter_dtypes,
    }


def _force_probe(sample, torch, np, checkpoint, item, rotation):
    numbers = list(item["atomic_numbers"])
    positions = np.asarray(item["positions_angstrom"], dtype=np.float64)
    rotated_positions = positions @ rotation.T

    energy, autograd_forces = _evaluate(
        sample, torch, np, checkpoint, numbers, positions
    )
    finite_difference_forces = _finite_difference_forces(
        sample, torch, np, checkpoint, numbers, positions
    )
    rotated_energy, rotated_autograd_forces = _evaluate(
        sample, torch, np, checkpoint, numbers, rotated_positions
    )
    rotated_finite_difference_forces = _finite_difference_forces(
        sample, torch, np, checkpoint, numbers, rotated_positions
    )
    rotated_forces_back = rotated_autograd_forces @ rotation

    original_difference = np.abs(autograd_forces - finite_difference_forces)
    rotated_difference = np.abs(
        rotated_autograd_forces - rotated_finite_difference_forces
    )
    covariance_difference = np.abs(autograd_forces - rotated_forces_back)
    original_worst = np.unravel_index(
        int(np.argmax(original_difference)), original_difference.shape
    )
    rotated_worst = np.unravel_index(
        int(np.argmax(rotated_difference)), rotated_difference.shape
    )
    covariance_worst = np.unravel_index(
        int(np.argmax(covariance_difference)), covariance_difference.shape
    )

    return {
        "atomic_numbers": numbers,
        "energy_ev": energy,
        "energy_rotation_abs_drift_ev": SOURCE_AUDIT._finite_float(
            abs(rotated_energy - energy),
            label=f"{item['label']} rotation energy drift",
        ),
        "formula": item["formula"],
        "g2_name": item["g2_name"],
        "label": item["label"],
        "original": {
            "autograd_forces_ev_per_angstrom": _array_record(autograd_forces, np),
            "finite_difference_forces_ev_per_angstrom": _array_record(
                finite_difference_forces, np
            ),
            "max_abs_force_difference_ev_per_angstrom": SOURCE_AUDIT._finite_float(
                original_difference[original_worst],
                label=f"{item['label']} original force difference",
            ),
            "worst_component_zero_based": [int(value) for value in original_worst],
        },
        "positions_angstrom": _array_record(positions, np),
        "rotated": {
            "autograd_forces_ev_per_angstrom": _array_record(
                rotated_autograd_forces, np
            ),
            "energy_ev": rotated_energy,
            "finite_difference_forces_ev_per_angstrom": _array_record(
                rotated_finite_difference_forces, np
            ),
            "max_abs_force_difference_ev_per_angstrom": SOURCE_AUDIT._finite_float(
                rotated_difference[rotated_worst],
                label=f"{item['label']} rotated force difference",
            ),
            "positions_angstrom": _array_record(rotated_positions, np),
            "worst_component_zero_based": [int(value) for value in rotated_worst],
        },
        "rotation_covariance": {
            "max_abs_force_difference_ev_per_angstrom": SOURCE_AUDIT._finite_float(
                covariance_difference[covariance_worst],
                label=f"{item['label']} force covariance difference",
            ),
            "rotated_forces_mapped_back_ev_per_angstrom": _array_record(
                rotated_forces_back, np
            ),
            "worst_component_zero_based": [int(value) for value in covariance_worst],
        },
    }


def audit_force_consistency(upstream_root: Path) -> dict[str, Any]:
    import ase
    import numpy as np
    import torch

    upstream_root = upstream_root.expanduser().resolve()
    SOURCE_AUDIT.verify_upstream_identity(upstream_root)
    dependency_origins = {
        "ase": SOURCE_AUDIT._verify_dependency_origin(ase, label="ASE"),
        "numpy": SOURCE_AUDIT._verify_dependency_origin(np, label="NumPy"),
        "torch": SOURCE_AUDIT._verify_dependency_origin(torch, label="PyTorch"),
    }
    original_path = list(sys.path)
    original_pycache_prefix = sys.pycache_prefix

    with tempfile.TemporaryDirectory(prefix="maple-anisolv-force-audit-") as temporary:
        isolated_parent = Path(temporary)
        source_root = isolated_parent / "anisolv"
        SOURCE_AUDIT._materialize_pinned_tree(upstream_root, source_root)
        pycache_root = isolated_parent / "pycache"
        pycache_root.mkdir()
        sys.pycache_prefix = str(pycache_root)
        try:
            sample = SOURCE_AUDIT._load_upstream_sample(source_root, isolated_parent)
            precision_state = SOURCE_AUDIT._configure_no_reduced_precision(torch)
            panel, panel_sha256 = SOURCE_AUDIT._panel_identity(sample)
            checkpoint = str((source_root / SOURCE_AUDIT.CHECKPOINT["path"]).resolve())
            rotation = _rotation_matrix(np)
            records = [
                _force_probe(
                    sample,
                    torch,
                    np,
                    checkpoint,
                    next(item for item in panel if item["g2_name"] == case),
                    rotation,
                )
                for case in CASES
            ]
            module_origins = SOURCE_AUDIT._verified_anisolv_module_origins(source_root)
            model_fingerprint = _float64_model_fingerprint(source_root)
        finally:
            sys.path[:] = original_path
            sys.pycache_prefix = original_pycache_prefix
            for name in list(sys.modules):
                if name == "anisolv" or name.startswith("anisolv."):
                    del sys.modules[name]

    original_maximum = max(
        record["original"]["max_abs_force_difference_ev_per_angstrom"]
        for record in records
    )
    rotated_maximum = max(
        record["rotated"]["max_abs_force_difference_ev_per_angstrom"]
        for record in records
    )
    energy_maximum = max(record["energy_rotation_abs_drift_ev"] for record in records)
    original_passed = original_maximum <= FORCE_CONSISTENCY_TOLERANCE_EV_PER_ANGSTROM
    rotated_passed = rotated_maximum <= FORCE_CONSISTENCY_TOLERANCE_EV_PER_ANGSTROM
    if (
        original_passed
        or not rotated_passed
        or energy_maximum > ENERGY_ROTATION_TOLERANCE_EV
    ):
        raise ValueError(
            "Pinned AniSolv force-defect signature changed; review upstream identity "
            "and the fail-closed energy-only disposition."
        )

    return {
        "acceptance_eligible": False,
        "audited_on": AUDITED_ON,
        "disposition": {
            "maple_runtime_scope": "single_point_scalar_energy_only",
            "optimization_frequency_md_forbidden": True,
            "upstream_force_admission": "rejected",
        },
        "finite_difference": {
            "formula": "F_i=-[E(R+h*e_i)-E(R-h*e_i)]/(2h)",
            "step_angstrom": FINITE_DIFFERENCE_STEP_ANGSTROM,
        },
        "identity": {
            "checkpoint": {**SOURCE_AUDIT.CHECKPOINT, "verified": True},
            "panel_geometry_and_source_row_sha256": panel_sha256,
            "experimental_label_values_used_for_identity": False,
            "repository": SOURCE_AUDIT.UPSTREAM_REPOSITORY,
            "revision": SOURCE_AUDIT.UPSTREAM_REVISION,
            "tree": SOURCE_AUDIT.UPSTREAM_TREE,
        },
        "metrics": {
            "energy_rotation_max_abs_drift_ev": energy_maximum,
            "energy_rotation_tolerance_ev": ENERGY_ROTATION_TOLERANCE_EV,
            "energy_rotation_tolerance_passed": (
                energy_maximum <= ENERGY_ROTATION_TOLERANCE_EV
            ),
            "force_consistency_tolerance_ev_per_angstrom": (
                FORCE_CONSISTENCY_TOLERANCE_EV_PER_ANGSTROM
            ),
            "original_orientation_max_abs_difference_ev_per_angstrom": (
                original_maximum
            ),
            "original_orientation_passed": original_passed,
            "rotated_orientation_max_abs_difference_ev_per_angstrom": (rotated_maximum),
            "rotated_orientation_passed": rotated_passed,
        },
        "model_runtime_fingerprint": model_fingerprint,
        "records": records,
        "rotation": {
            "angle_rad": ROTATION_ANGLE_RAD,
            "applied_as": "positions @ matrix.T",
            "axis_before_normalization": ROTATION_AXIS,
            "center": "origin_without_recentering",
            "matrix": _array_record(rotation, np),
        },
        "runtime": {
            "ase_version": ase.__version__,
            "dependency_origins_within_active_environment": dependency_origins,
            "device": "cpu",
            "dtype": "torch.float64",
            "inference_settings": "default",
            "numpy_version": np.__version__,
            "precision_controls": precision_state,
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "torch_version": torch.__version__,
        },
        "source_execution_isolation": {
            "git_archive_from_verified_tree": True,
            "ignored_checkout_files_excluded_from_execution": True,
            "isolated_import_parent": True,
            "isolated_pycache_root": True,
            "loaded_source_modules": module_origins,
            "loaded_source_modules_sha256": SOURCE_AUDIT._sha256_bytes(
                SOURCE_AUDIT._canonical_bytes(module_origins)
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = audit_force_consistency(args.upstream_root)
    SOURCE_AUDIT._write_json_atomic(args.output, payload)
    print(
        SOURCE_AUDIT.json.dumps(
            payload["metrics"],
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
