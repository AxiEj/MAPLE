#!/usr/bin/env python3
"""Run an arbitrary multiple of four CPKS modes via the frozen 4-mode core.

The frozen PySCF implementation is invoked unchanged for each four-mode group.
All groups evaluate MEP response on the same full exterior surface, so the
complete cross-group source-response matrix is reconstructed from those MEP
values rather than approximated or omitted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pyscf


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SELF_REPO_PATH = "tools/route2_release/run_vqm24_grouped_static_cpks_response.py"
FROZEN_CORE_REPO_PATH = "tools/route2_release/run_vqm24_static_cpks_response.py"
ARTIFACT = "route2-vqm24-grouped-static-cpks-observable-response-v1"
GROUP_SIZE = 4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(
        f"{array.dtype.str}:{array.shape}".encode()
        + b"\0"
        + array.tobytes(order="C")
    ).hexdigest()


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--modes", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-memory-mb", type=int, default=8000)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser.parse_args()


def assemble_grouped_observables(
    *,
    induced_mep_parts: list[np.ndarray],
    induced_dipole_parts: list[np.ndarray],
    curvature_parts: list[np.ndarray],
    source_surface_indices: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Concatenate groups and reconstruct the complete source response."""

    if not induced_mep_parts or not (
        len(induced_mep_parts)
        == len(induced_dipole_parts)
        == len(curvature_parts)
    ):
        raise ValueError("Grouped observable parts must be nonempty and aligned.")
    induced_mep = np.concatenate(induced_mep_parts, axis=0)
    induced_dipole = np.concatenate(induced_dipole_parts, axis=0)
    curvature = np.concatenate(curvature_parts, axis=0)
    indices = np.asarray(source_surface_indices)
    mode_count = len(induced_mep)
    if (
        induced_mep.ndim != 2
        or induced_dipole.shape != (mode_count, 3)
        or curvature.shape != (mode_count,)
        or indices.shape != (mode_count,)
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(indices >= induced_mep.shape[1])
        or not np.all(np.isfinite(induced_mep))
        or not np.all(np.isfinite(induced_dipole))
        or not np.all(np.isfinite(curvature))
    ):
        raise ValueError("Grouped observable shapes or values are invalid.")
    source_response = induced_mep[:, indices]
    symmetric_response = 0.5 * (source_response + source_response.T)
    reciprocity = float(
        np.linalg.norm(source_response - source_response.T)
        / max(np.linalg.norm(source_response), np.finfo(float).tiny)
    )
    return {
        "induced_mep": induced_mep,
        "induced_dipole": induced_dipole,
        "curvature": curvature,
        "source_response": source_response,
        "reciprocity": reciprocity,
        "symmetric_eigenvalues": np.linalg.eigvalsh(symmetric_response),
    }


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    surface_path = args.surface.expanduser().resolve(strict=True)
    modes_path = args.modes.expanduser().resolve(strict=True)
    output_json = args.output_json.expanduser().resolve()
    output_npz = args.output_npz.expanduser().resolve()
    if output_json.exists() or output_npz.exists():
        raise FileExistsError(output_json if output_json.exists() else output_npz)
    if args.threads != 8 or args.max_memory_mb != 8000:
        raise RuntimeError("Grouped CPKS preserves the frozen 8-thread/8-GB core.")
    core = (SOURCE_ROOT / FROZEN_CORE_REPO_PATH).resolve(strict=True)
    with np.load(modes_path, allow_pickle=False) as state:
        source_points = np.asarray(state["source_points_bohr"], dtype=np.float64)
        source_indices = np.asarray(
            state["source_surface_indices"], dtype=np.int64
        )
        source_partitions = (
            np.asarray(state["source_partition_indices"], dtype=np.int64)
            if "source_partition_indices" in state
            else np.zeros(len(source_points), dtype=np.int64)
        )
    if (
        source_points.ndim != 2
        or source_points.shape[1] != 3
        or len(source_points) <= GROUP_SIZE
        or len(source_points) % GROUP_SIZE != 0
        or source_indices.shape != (len(source_points),)
        or source_partitions.shape != (len(source_points),)
        or not np.all(np.isfinite(source_points))
    ):
        raise RuntimeError("Grouped CPKS modes must be a finite multiple of four.")
    with np.load(surface_path, allow_pickle=False) as state:
        surface_points = np.asarray(state["surface_points_bohr"], dtype=np.float64)
        quadrature_weights = np.asarray(state["quadrature_weights"], dtype=np.float64)
        parent_atom_indices = np.asarray(state["parent_atom_indices"], dtype=np.int64)
    if (
        np.any(source_indices < 0)
        or np.any(source_indices >= len(surface_points))
        or not np.allclose(
            source_points,
            surface_points[source_indices],
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise RuntimeError("Grouped source modes no longer index the full surface.")

    started = time.perf_counter()
    group_root = output_json.parent / f"{output_json.stem}.groups"
    if group_root.exists():
        raise FileExistsError(group_root)
    group_root.mkdir(parents=True, exist_ok=False)
    group_results: list[dict[str, Any]] = []
    induced_mep_parts = []
    induced_dipole_parts = []
    curvature_parts = []
    zero_source_parts = []
    zero_surface_reference = None
    zero_dipole_reference = None
    checkpoint_reference = None
    for group_index, start in enumerate(range(0, len(source_points), GROUP_SIZE)):
        stop = start + GROUP_SIZE
        directory = group_root / f"group-{group_index:02d}"
        directory.mkdir()
        group_modes = directory / "modes.npz"
        np.savez(
            group_modes,
            source_points_bohr=source_points[start:stop],
            source_surface_indices=source_indices[start:stop],
            source_partition_indices=source_partitions[start:stop],
        )
        group_json = directory / "cpks.json"
        group_npz = directory / "cpks.npz"
        command = [
            sys.executable,
            str(core),
            "--checkpoint",
            str(checkpoint),
            "--surface",
            str(surface_path),
            "--modes",
            str(group_modes),
            "--threads",
            str(args.threads),
            "--max-memory-mb",
            str(args.max_memory_mb),
            "--output-json",
            str(group_json),
            "--output-npz",
            str(group_npz),
        ]
        completed = subprocess.run(
            command,
            cwd=SOURCE_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (directory / "stdout.log").write_text(completed.stdout)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Frozen CPKS core failed for group {group_index}: {completed.returncode}."
            )
        result = json.loads(group_json.read_text())
        if result.get("status") != "success" or _sha256(group_npz) != result[
            "output"
        ]["npz_sha256"]:
            raise RuntimeError(f"Frozen CPKS group {group_index} is invalid.")
        with np.load(group_npz, allow_pickle=False) as state:
            induced_mep_parts.append(
                np.asarray(
                    state["induced_surface_mep_hartree_per_e_per_source_e"],
                    dtype=np.float64,
                )
            )
            induced_dipole_parts.append(
                np.asarray(
                    state["induced_dipole_e_bohr_per_source_e"],
                    dtype=np.float64,
                )
            )
            curvature_parts.append(
                np.asarray(state["energy_curvature_hartree_per_e2"], dtype=np.float64)
            )
            zero_surface = np.asarray(
                state["zero_total_surface_mep_hartree_per_e"], dtype=np.float64
            )
            zero_source_parts.append(
                np.asarray(
                    state["zero_total_source_mep_hartree_per_e"], dtype=np.float64
                )
            )
            zero_dipole = np.asarray(state["zero_total_dipole_e_bohr"], dtype=np.float64)
        if zero_surface_reference is None:
            zero_surface_reference = zero_surface
            zero_dipole_reference = zero_dipole
            checkpoint_reference = result["checkpoint_consistency"]
        else:
            if not np.array_equal(zero_surface_reference, zero_surface):
                raise RuntimeError("Grouped CPKS zero-field surface MEP changed.")
            if not np.array_equal(zero_dipole_reference, zero_dipole):
                raise RuntimeError("Grouped CPKS zero-field dipole changed.")
            if result["checkpoint_consistency"] != checkpoint_reference:
                raise RuntimeError("Grouped CPKS checkpoint consistency changed.")
        group_results.append(
            {
                "group_index": group_index,
                "mode_start": start,
                "mode_stop": stop,
                "json_sha256": _sha256(group_json),
                "npz_sha256": _sha256(group_npz),
                "modes_sha256": _sha256(group_modes),
                "response": result["response"],
                "runtime": result["runtime"],
            }
        )
    if zero_surface_reference is None or zero_dipole_reference is None:
        raise RuntimeError("Grouped CPKS produced no groups.")
    assembled = assemble_grouped_observables(
        induced_mep_parts=induced_mep_parts,
        induced_dipole_parts=induced_dipole_parts,
        curvature_parts=curvature_parts,
        source_surface_indices=source_indices,
    )
    induced_mep = assembled["induced_mep"]
    induced_dipole = assembled["induced_dipole"]
    curvature = assembled["curvature"]
    zero_source = np.concatenate(zero_source_parts, axis=0)
    source_response = assembled["source_response"]
    reciprocity = float(assembled["reciprocity"])
    eigenvalues = assembled["symmetric_eigenvalues"]
    with output_npz.open("xb") as handle:
        np.savez(
            handle,
            source_points_bohr=source_points,
            source_surface_indices=source_indices,
            surface_points_bohr=surface_points,
            quadrature_weights=quadrature_weights,
            parent_atom_indices=parent_atom_indices,
            zero_total_surface_mep_hartree_per_e=zero_surface_reference,
            zero_total_source_mep_hartree_per_e=zero_source,
            zero_total_dipole_e_bohr=zero_dipole_reference,
            induced_surface_mep_hartree_per_e_per_source_e=induced_mep,
            induced_dipole_e_bohr_per_source_e=induced_dipole,
            energy_curvature_hartree_per_e2=curvature,
            source_response_hartree_per_e2=source_response,
        )
    maximum_residual_frobenius = max(
        max(float(value) for value in group["response"]["final_residual_relative_frobenius"])
        for group in group_results
    )
    maximum_residual_infinity = max(
        max(float(value) for value in group["response"]["final_residual_relative_infinity"])
        for group in group_results
    )
    maximum_electron_number = max(
        max(abs(float(value)) for value in group["response"]["electron_number_derivative_e_per_source_e"])
        for group in group_results
    )
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "success",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "checkpoint_sha256": _sha256(checkpoint),
            "surface_npz_sha256": _sha256(surface_path),
            "modes_npz_sha256": _sha256(modes_path),
            "source_points_sha256": _array_sha256(source_points),
            "surface_points_sha256": _array_sha256(surface_points),
        },
        "source": {
            "runner_path": SELF_REPO_PATH,
            "runner_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "frozen_core_path": FROZEN_CORE_REPO_PATH,
            "frozen_core_sha256": _sha256(core),
        },
        "method": {
            "group_size": GROUP_SIZE,
            "group_count": len(group_results),
            "mode_count": len(source_points),
            "cross_group_response_reconstructed_from_full_surface_mep": True,
        },
        "checkpoint_consistency": checkpoint_reference,
        "response": {
            "mode_count": len(source_points),
            "maximum_group_residual_relative_frobenius": maximum_residual_frobenius,
            "maximum_group_residual_relative_infinity": maximum_residual_infinity,
            "maximum_electron_number_derivative_abs": maximum_electron_number,
            "source_response_reciprocity_relative_frobenius": reciprocity,
            "source_response_symmetric_eigenvalues_hartree_per_e2": eigenvalues.tolist(),
            "energy_curvature_diagnostic_only": True,
        },
        "groups": group_results,
        "output": {
            "npz_sha256": _sha256(output_npz),
            "induced_surface_mep_sha256": _array_sha256(induced_mep),
            "induced_dipole_sha256": _array_sha256(induced_dipole),
            "source_response_sha256": _array_sha256(source_response),
        },
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "pyscf": pyscf.__version__,
            "threads_per_group": args.threads,
            "total_elapsed_seconds": time.perf_counter() - started,
        },
        "claim_boundary": {
            "q_to_zero_static_response_computed": True,
            "ao_density_response_released_as_training_label": False,
            "energy_curvature_training_target": False,
            "fit_or_training_performed": False,
            "experimental_solvation_target_read": False,
            "pcm_or_cavity_used": False,
            "validation_or_blind_formula_opened": False,
            "capability_admitted": False,
        },
    }
    _write_exclusive_json(output_json, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
