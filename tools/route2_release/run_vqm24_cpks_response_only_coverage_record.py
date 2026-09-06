#!/usr/bin/env python3
"""Run and assess one dense four-mode response-only CPKS coverage record."""

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
SELF_REPO_PATH = (
    "tools/route2_release/run_vqm24_cpks_response_only_coverage_record.py"
)
ARTIFACT = "route2-vqm24-cpks-response-only-coverage-record-v1"
PREREGISTRATION_ARTIFACT = (
    "route2-vqm24-cpks-response-only-coverage-prereg-v1"
)
CPKS_ARTIFACT = "route2-vqm24-static-cpks-observable-response-level3-v1"


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


def _mode_maximum(left: np.ndarray, right: np.ndarray) -> float:
    return max(
        _symmetric_relative(left[index], right[index])
        for index in range(len(left))
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    output_directory = args.output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    preregistration = json.loads(preregistration_path.read_text())
    if (
        preregistration.get("artifact") != PREREGISTRATION_ARTIFACT
        or preregistration.get("status") != "locked-before-coverage-execution"
    ):
        raise ValueError("Response-only CPKS coverage preregistration is invalid.")
    if _sha256(SOURCE_ROOT / SELF_REPO_PATH) != preregistration["source"][
        "record_runner_sha256"
    ]:
        raise RuntimeError("Coverage record runner changed after freeze.")
    cpks_runner = SOURCE_ROOT / preregistration["source"]["cpks_runner_path"]
    if _sha256(cpks_runner) != preregistration["source"]["cpks_runner_sha256"]:
        raise RuntimeError("Frozen CPKS runner changed before coverage.")
    records = {
        record["record_id"]: record for record in preregistration["records"]
    }
    if args.record_id not in records:
        raise KeyError(f"Unknown coverage record: {args.record_id}")
    record = records[args.record_id]
    inputs = {name: Path(value["path"]) for name, value in record["inputs"].items()}
    for name, path in inputs.items():
        if _sha256(path) != record["inputs"][name]["sha256"]:
            raise RuntimeError(f"Coverage input changed: {args.record_id}/{name}")
    output_directory.mkdir(parents=True, exist_ok=False)
    cpks_json = output_directory / "cpks.json"
    cpks_npz = output_directory / "cpks.npz"
    command = [
        sys.executable,
        str(cpks_runner),
        "--checkpoint",
        str(inputs["gas_checkpoint"]),
        "--surface",
        str(inputs["surface"]),
        "--modes",
        str(inputs["modes"]),
        "--threads",
        "8",
        "--max-memory-mb",
        "8000",
        "--output-json",
        str(cpks_json),
        "--output-npz",
        str(cpks_npz),
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
            f"CPKS record failed for {args.record_id} with {completed.returncode}."
        )
    cpks = json.loads(cpks_json.read_text())
    if cpks.get("artifact") != CPKS_ARTIFACT or cpks.get("status") != "success":
        raise RuntimeError("CPKS record returned the wrong artifact or status.")
    if _sha256(cpks_npz) != cpks["output"]["npz_sha256"]:
        raise RuntimeError("CPKS NPZ digest does not match its result JSON.")
    with np.load(cpks_npz, allow_pickle=False) as state:
        cpks_mep = np.asarray(
            state["induced_surface_mep_hartree_per_e_per_source_e"],
            dtype=np.float64,
        )
        cpks_dipole = np.asarray(
            state["induced_dipole_e_bohr_per_source_e"], dtype=np.float64
        )
    with np.load(inputs["finite_field_observable"], allow_pickle=False) as state:
        finite_mep = np.asarray(
            state["induced_surface_mep_hartree_per_e_per_source_e"],
            dtype=np.float64,
        )
        finite_dipole = np.asarray(
            state["induced_dipole_e_bohr_per_source_e"], dtype=np.float64
        )
    if cpks_mep.shape != finite_mep.shape or cpks_dipole.shape != finite_dipole.shape:
        raise RuntimeError("CPKS/finite-field observable shapes differ.")
    response = cpks["response"]
    metrics = {
        "mep_symmetric_relative": _symmetric_relative(cpks_mep, finite_mep),
        "mep_mode_symmetric_relative_maximum": _mode_maximum(
            cpks_mep, finite_mep
        ),
        "dipole_symmetric_relative": _symmetric_relative(
            cpks_dipole, finite_dipole
        ),
        "dipole_mode_symmetric_relative_maximum": _mode_maximum(
            cpks_dipole, finite_dipole
        ),
        "maximum_cpks_residual_relative_frobenius": max(
            float(value) for value in response["final_residual_relative_frobenius"]
        ),
        "maximum_cpks_residual_relative_infinity": max(
            float(value) for value in response["final_residual_relative_infinity"]
        ),
        "checkpoint_canonical_fock_residual_relative": abs(
            float(
                cpks["checkpoint_consistency"][
                    "canonical_fock_residual_relative_frobenius"
                ]
            )
        ),
        "checkpoint_rebuilt_energy_abs_hartree": abs(
            float(cpks["checkpoint_consistency"]["rebuilt_energy_error_hartree"])
        ),
        "maximum_energy_identity_abs_hartree_per_e2": max(
            abs(float(value))
            for value in response["energy_identity_error_hartree_per_e2"]
        ),
        "maximum_electron_number_derivative_abs": max(
            abs(float(value))
            for value in response["electron_number_derivative_e_per_source_e"]
        ),
        "reciprocity_relative_frobenius": float(
            response["susceptibility_reciprocity_relative_frobenius"]
        ),
        "maximum_passivity_eigenvalue_hartree_per_e2": max(
            float(value)
            for value in response[
                "susceptibility_symmetric_eigenvalues_hartree_per_e2"
            ]
        ),
        "cpks_total_elapsed_seconds": float(cpks["runtime"]["total_elapsed_seconds"]),
    }
    gates = preregistration["gates"]
    record_gates = {
        "mep_global": metrics["mep_symmetric_relative"]
        <= gates["mep_symmetric_relative_maximum"],
        "mep_mode": metrics["mep_mode_symmetric_relative_maximum"]
        <= gates["mep_symmetric_relative_maximum"],
        "dipole_global": metrics["dipole_symmetric_relative"]
        <= gates["dipole_symmetric_relative_maximum"],
        "dipole_mode": metrics["dipole_mode_symmetric_relative_maximum"]
        <= gates["dipole_symmetric_relative_maximum"],
        "cpks_residual_frobenius": metrics[
            "maximum_cpks_residual_relative_frobenius"
        ]
        <= gates["direct_cpks_residual_relative_frobenius_maximum"],
        "cpks_residual_infinity": metrics[
            "maximum_cpks_residual_relative_infinity"
        ]
        <= gates["direct_cpks_residual_relative_infinity_maximum"],
        "checkpoint_canonical_fock": metrics[
            "checkpoint_canonical_fock_residual_relative"
        ]
        <= gates["checkpoint_canonical_fock_residual_relative_maximum"],
        "checkpoint_rebuilt_energy": metrics[
            "checkpoint_rebuilt_energy_abs_hartree"
        ]
        <= gates["checkpoint_rebuilt_energy_abs_hartree_maximum"],
        "energy_identity": metrics[
            "maximum_energy_identity_abs_hartree_per_e2"
        ]
        <= gates["energy_identity_abs_hartree_per_e2_maximum"],
        "electron_number": metrics["maximum_electron_number_derivative_abs"]
        <= gates["electron_number_derivative_abs_maximum"],
        "reciprocity": metrics["reciprocity_relative_frobenius"]
        <= gates["reciprocity_relative_frobenius_maximum"],
        "passivity": metrics["maximum_passivity_eigenvalue_hartree_per_e2"]
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
            "cpks_runner_sha256": _sha256(cpks_runner),
            "cpks_json_sha256": _sha256(cpks_json),
            "cpks_npz_sha256": _sha256(cpks_npz),
            "finite_field_observable_sha256": _sha256(
                inputs["finite_field_observable"]
            ),
        },
        "claim_boundary": {
            "independent_QM_response_compared": True,
            "energy_curvature_used_as_target_or_gate": False,
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
