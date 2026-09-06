#!/usr/bin/env python3
"""Generate and validate one twelve-mode sector-balanced CPKS response record."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SELF_REPO_PATH = "tools/route2_release/run_vqm24_sector_balanced_cpks_record.py"
ARTIFACT = "route2-vqm24-sector-balanced-twelve-mode-cpks-record-v1"
PREREGISTRATION_ARTIFACT = (
    "route2-vqm24-sector-balanced-twelve-mode-cpks-prereg-v1"
)
GROUPED_ARTIFACT = "route2-vqm24-grouped-static-cpks-observable-response-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _symmetric_relative(left: np.ndarray, right: np.ndarray) -> float:
    numerator = 2.0 * float(np.linalg.norm(left - right))
    denominator = float(np.linalg.norm(left) + np.linalg.norm(right))
    return numerator / max(denominator, np.finfo(float).tiny)


def run(args: argparse.Namespace) -> dict[str, Any]:
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    output_directory = args.output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    preregistration = json.loads(preregistration_path.read_text())
    if (
        preregistration.get("artifact") != PREREGISTRATION_ARTIFACT
        or preregistration.get("status") != "locked-before-twelve-mode-execution"
    ):
        raise ValueError("Twelve-mode CPKS preregistration is invalid.")
    if _sha256(SOURCE_ROOT / SELF_REPO_PATH) != preregistration["source"][
        "record_runner_sha256"
    ]:
        raise RuntimeError("Twelve-mode record runner changed after freeze.")
    grouped_runner = SOURCE_ROOT / preregistration["source"]["grouped_runner_path"]
    if _sha256(grouped_runner) != preregistration["source"]["grouped_runner_sha256"]:
        raise RuntimeError("Grouped CPKS runner changed after freeze.")
    records = {record["record_id"]: record for record in preregistration["records"]}
    if args.record_id not in records:
        raise KeyError(f"Unknown twelve-mode record: {args.record_id}")
    record = records[args.record_id]
    inputs = {name: Path(value["path"]) for name, value in record["inputs"].items()}
    for name, path in inputs.items():
        if _sha256(path) != record["inputs"][name]["sha256"]:
            raise RuntimeError(f"Twelve-mode input changed: {args.record_id}/{name}")
    with np.load(inputs["twelve_modes"], allow_pickle=False) as state:
        twelve_points = np.asarray(state["source_points_bohr"], dtype=np.float64)
        twelve_indices = np.asarray(state["source_surface_indices"], dtype=np.int64)
        prefix_count = int(np.asarray(state["frozen_prefix_count"]))
    with np.load(inputs["four_modes"], allow_pickle=False) as state:
        four_points = np.asarray(state["source_points_bohr"], dtype=np.float64)
        four_indices = np.asarray(state["source_surface_indices"], dtype=np.int64)
    if (
        twelve_points.shape != (12, 3)
        or twelve_indices.shape != (12,)
        or prefix_count != 4
        or not np.array_equal(twelve_points[:4], four_points)
        or not np.array_equal(twelve_indices[:4], four_indices)
    ):
        raise RuntimeError("Twelve-mode record does not preserve the four-mode prefix.")
    output_directory.mkdir(parents=True, exist_ok=False)
    grouped_json = output_directory / "cpks.json"
    grouped_npz = output_directory / "cpks.npz"
    command = [
        sys.executable,
        str(grouped_runner),
        "--checkpoint",
        str(inputs["gas_checkpoint"]),
        "--surface",
        str(inputs["surface"]),
        "--modes",
        str(inputs["twelve_modes"]),
        "--threads",
        "8",
        "--max-memory-mb",
        "8000",
        "--output-json",
        str(grouped_json),
        "--output-npz",
        str(grouped_npz),
    ]
    completed = subprocess.run(
        command,
        cwd=SOURCE_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output_directory / "cpks.stdout.log").write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Grouped CPKS failed for {args.record_id}: {completed.returncode}."
        )
    grouped = json.loads(grouped_json.read_text())
    if grouped.get("artifact") != GROUPED_ARTIFACT or grouped.get("status") != "success":
        raise RuntimeError("Grouped CPKS output has the wrong identity or status.")
    if _sha256(grouped_npz) != grouped["output"]["npz_sha256"]:
        raise RuntimeError("Grouped CPKS NPZ digest failed.")
    with np.load(grouped_npz, allow_pickle=False) as state:
        induced_mep = np.asarray(
            state["induced_surface_mep_hartree_per_e_per_source_e"],
            dtype=np.float64,
        )
        induced_dipole = np.asarray(
            state["induced_dipole_e_bohr_per_source_e"], dtype=np.float64
        )
        source_response = np.asarray(
            state["source_response_hartree_per_e2"], dtype=np.float64
        )
    with np.load(inputs["finite_field_observable"], allow_pickle=False) as state:
        finite_mep = np.asarray(
            state["induced_surface_mep_hartree_per_e_per_source_e"],
            dtype=np.float64,
        )
        finite_dipole = np.asarray(
            state["induced_dipole_e_bohr_per_source_e"], dtype=np.float64
        )
    if (
        induced_mep.shape[0] != 12
        or induced_dipole.shape != (12, 3)
        or source_response.shape != (12, 12)
        or finite_mep.shape != induced_mep[:4].shape
        or finite_dipole.shape != induced_dipole[:4].shape
    ):
        raise RuntimeError("Twelve-mode/finite-field observable shapes differ.")
    response = grouped["response"]
    eigenvalues = np.asarray(
        response["source_response_symmetric_eigenvalues_hartree_per_e2"],
        dtype=np.float64,
    )
    metrics = {
        "prefix_mep_symmetric_relative": _symmetric_relative(
            induced_mep[:4], finite_mep
        ),
        "prefix_dipole_symmetric_relative": _symmetric_relative(
            induced_dipole[:4], finite_dipole
        ),
        "maximum_group_residual_relative_frobenius": float(
            response["maximum_group_residual_relative_frobenius"]
        ),
        "maximum_group_residual_relative_infinity": float(
            response["maximum_group_residual_relative_infinity"]
        ),
        "maximum_electron_number_derivative_abs": float(
            response["maximum_electron_number_derivative_abs"]
        ),
        "source_response_reciprocity_relative_frobenius": float(
            response["source_response_reciprocity_relative_frobenius"]
        ),
        "maximum_source_response_eigenvalue_hartree_per_e2": float(
            np.max(eigenvalues)
        ),
        "minimum_source_response_eigenvalue_hartree_per_e2": float(
            np.min(eigenvalues)
        ),
        "total_elapsed_seconds": float(grouped["runtime"]["total_elapsed_seconds"]),
    }
    gates = preregistration["gates"]
    record_gates = {
        "four_mode_prefix_mep": metrics["prefix_mep_symmetric_relative"]
        <= gates["prefix_mep_symmetric_relative_maximum"],
        "four_mode_prefix_dipole": metrics["prefix_dipole_symmetric_relative"]
        <= gates["prefix_dipole_symmetric_relative_maximum"],
        "cpks_residual_frobenius": metrics[
            "maximum_group_residual_relative_frobenius"
        ]
        <= gates["direct_cpks_residual_relative_frobenius_maximum"],
        "cpks_residual_infinity": metrics[
            "maximum_group_residual_relative_infinity"
        ]
        <= gates["direct_cpks_residual_relative_infinity_maximum"],
        "electron_number": metrics["maximum_electron_number_derivative_abs"]
        <= gates["electron_number_derivative_abs_maximum"],
        "reciprocity": metrics[
            "source_response_reciprocity_relative_frobenius"
        ]
        <= gates["reciprocity_relative_frobenius_maximum"],
        "passivity": metrics["maximum_source_response_eigenvalue_hartree_per_e2"]
        <= gates["passivity_maximum_eigenvalue_hartree_per_e2"],
    }
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "pass" if all(record_gates.values()) else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "record_id": args.record_id,
        "metrics": metrics,
        "gates": record_gates,
        "source": {
            "runner_path": SELF_REPO_PATH,
            "runner_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "preregistration_file_sha256": _sha256(preregistration_path),
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "grouped_runner_sha256": _sha256(grouped_runner),
            "cpks_json_sha256": _sha256(grouped_json),
            "cpks_npz_sha256": _sha256(grouped_npz),
        },
        "claim_boundary": {
            "independent_twelve_mode_QM_response_generated": True,
            "energy_curvature_training_target": False,
            "model_fit_or_accuracy_measured": False,
            "experimental_solvation_target_read": False,
            "pcm_or_cavity_used": False,
            "validation_or_blind_formula_opened": False,
            "maple_capability_admitted": False,
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    result_path = output_directory / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    result_path.chmod(0o444)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
