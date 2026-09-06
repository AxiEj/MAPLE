#!/usr/bin/env python3
"""Seal the dense 32-record observable-supervision QM batch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SELF_REPO_PATH = "tools/route2_release/seal_vqm24_observable_training_batch.py"
PREREGISTRATION_ARTIFACT = "route2-vqm24-observable-training-batch-dense-prereg-v2"
OBSERVABLE_ARTIFACT = "route2-vqm24-observable-training-record-level3-v1"
ARTIFACT = "route2-vqm24-observable-training-batch-result-v1"


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


def seal(args: argparse.Namespace) -> dict[str, Any]:
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    output_directory = args.output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    preregistration = json.loads(preregistration_path.read_text())
    if (
        preregistration.get("artifact") != PREREGISTRATION_ARTIFACT
        or preregistration.get("status")
        != "locked-before-dense-observable-qm-execution"
    ):
        raise ValueError("Dense observable preregistration is invalid.")
    runner_path = SOURCE_ROOT / preregistration["source"]["observable_runner_path"]
    if _sha256(runner_path) != preregistration["source"]["observable_runner_sha256"]:
        raise RuntimeError("Observable runner changed after batch freeze.")

    records = []
    nonlinearity_mep = []
    nonlinearity_dipole = []
    for preregistered in preregistration["records"]:
        record_id = preregistered["record_id"]
        json_path = Path(preregistered["outputs"]["observable_json"])
        npz_path = Path(preregistered["outputs"]["observable_npz"])
        result = json.loads(json_path.read_text())
        if (
            result.get("artifact") != OBSERVABLE_ARTIFACT
            or result.get("status") != "success"
        ):
            raise RuntimeError(f"Observable helper failed for {record_id}.")
        if _sha256(npz_path) != result["output"]["npz_sha256"]:
            raise RuntimeError(f"Observable NPZ drifted for {record_id}.")
        if result["source"]["runner_sha256"] != preregistration["source"][
            "observable_runner_sha256"
        ]:
            raise RuntimeError(f"Observable runner identity differs for {record_id}.")
        with np.load(preregistered["inputs"]["surface"]["path"], allow_pickle=False) as surface:
            partitions = np.asarray(surface["partition_indices"], dtype=np.int64)
        with np.load(npz_path, allow_pickle=False) as state:
            weights = np.asarray(state["quadrature_weights"], dtype=np.float64)
            zero_mep = np.asarray(
                state["zero_total_surface_mep_hartree_per_e"],
                dtype=np.float64,
            )
            perturbed_mep = np.asarray(
                state["perturbed_total_surface_mep_hartree_per_e"],
                dtype=np.float64,
            )
            zero_dipole = np.asarray(state["zero_total_dipole_e_bohr"], dtype=np.float64)
            perturbed_dipole = np.asarray(
                state["perturbed_total_dipole_e_bohr"],
                dtype=np.float64,
            )
        if (
            weights.shape != zero_mep.shape
            or partitions.shape != zero_mep.shape
            or perturbed_mep.shape != (4, 2, len(zero_mep))
            or zero_dipole.shape != (3,)
            or perturbed_dipole.shape != (4, 2, 3)
        ):
            raise RuntimeError(f"Observable arrays have invalid shapes for {record_id}.")
        record_mep = []
        record_dipole = []
        for mode in range(4):
            even_mep = 0.5 * (
                perturbed_mep[mode, 1] + perturbed_mep[mode, 0]
            ) - zero_mep
            odd_mep = 0.5 * (
                perturbed_mep[mode, 1] - perturbed_mep[mode, 0]
            )
            mep_ratio = float(
                np.sqrt(np.sum(weights * even_mep**2))
                / max(
                    float(np.sqrt(np.sum(weights * odd_mep**2))),
                    np.finfo(float).tiny,
                )
            )
            even_dipole = 0.5 * (
                perturbed_dipole[mode, 1] + perturbed_dipole[mode, 0]
            ) - zero_dipole
            odd_dipole = 0.5 * (
                perturbed_dipole[mode, 1] - perturbed_dipole[mode, 0]
            )
            dipole_ratio = float(
                np.linalg.norm(even_dipole)
                / max(float(np.linalg.norm(odd_dipole)), np.finfo(float).tiny)
            )
            record_mep.append(mep_ratio)
            record_dipole.append(dipole_ratio)
            nonlinearity_mep.append(mep_ratio)
            nonlinearity_dipole.append(dipole_ratio)
        records.append(
            {
                "record_id": record_id,
                "stratum": preregistered["selection_record"]["stratum"],
                "formula": preregistered["selection_record"][
                    "chemical_formula_hill"
                ],
                "observable_json_sha256": _sha256(json_path),
                "observable_npz_sha256": _sha256(npz_path),
                "fit_point_count": int(np.count_nonzero(partitions == 0)),
                "audit_point_count": int(np.count_nonzero(partitions == 1)),
                "maximum_electron_count_error_e": float(
                    result["numerical_checks"]["maximum_electron_count_error_e"]
                ),
                "enthalpy_slope_vs_zero_source_mep_relative": float(
                    result["numerical_checks"][
                        "enthalpy_slope_vs_zero_source_mep_relative"
                    ]
                ),
                "mep_nonlinearity_by_mode": record_mep,
                "dipole_nonlinearity_by_mode": record_dipole,
                "scf_cycle_range": [
                    int(result["numerical_checks"]["minimum_scf_cycles"]),
                    int(result["numerical_checks"]["maximum_scf_cycles"]),
                ],
                "elapsed_seconds": float(result["runtime"]["total_elapsed_seconds"]),
                "numerical_energy_curvature_label_emitted": bool(
                    result["finite_field_protocol"][
                        "numerical_energy_curvature_label_emitted"
                    ]
                ),
            }
        )
    records.sort(key=lambda record: record["record_id"])
    maximum_electron_error = max(
        record["maximum_electron_count_error_e"] for record in records
    )
    maximum_conjugacy_error = max(
        record["enthalpy_slope_vs_zero_source_mep_relative"]
        for record in records
    )
    gates = {
        "record_count": len(records) == 32,
        "electron_count": maximum_electron_error <= 1.0e-10,
        "enthalpy_mep_conjugacy": maximum_conjugacy_error <= 1.0e-5,
        "mep_nonlinearity_mean": float(np.mean(nonlinearity_mep)) <= 0.02,
        "mep_nonlinearity_maximum": max(nonlinearity_mep) <= 0.05,
        "dipole_nonlinearity_mean": float(np.mean(nonlinearity_dipole)) <= 0.02,
        "dipole_nonlinearity_maximum": max(nonlinearity_dipole) <= 0.05,
        "no_curvature_labels": not any(
            record["numerical_energy_curvature_label_emitted"] for record in records
        ),
    }
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "pass-observable-training-batch" if all(gates.values()) else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "sealer_path": SELF_REPO_PATH,
            "sealer_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "preregistration_path": str(preregistration_path.relative_to(SOURCE_ROOT)),
            "preregistration_file_sha256": _sha256(preregistration_path),
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "observable_runner_sha256": preregistration["source"][
                "observable_runner_sha256"
            ],
        },
        "records": records,
        "aggregate": {
            "record_count": len(records),
            "maximum_electron_count_error_e": maximum_electron_error,
            "maximum_enthalpy_mep_conjugacy_relative": maximum_conjugacy_error,
            "mean_mep_nonlinearity": float(np.mean(nonlinearity_mep)),
            "maximum_mep_nonlinearity": max(nonlinearity_mep),
            "mean_dipole_nonlinearity": float(np.mean(nonlinearity_dipole)),
            "maximum_dipole_nonlinearity": max(nonlinearity_dipole),
            "total_elapsed_seconds": sum(
                record["elapsed_seconds"] for record in records
            ),
        },
        "gates": gates,
        "claim_boundary": {
            "independent_qm_observable_training_batch_generated": True,
            "quadratic_response_adequacy_passed": all(gates.values()),
            "density_or_partition_coefficient_label_emitted": False,
            "fit_or_training_performed": False,
            "model_accuracy_measured": False,
            "experimental_solvation_target_read": False,
            "vqm24_energy_target_used": False,
            "pcm_or_cavity_used": False,
            "capability_admitted": False,
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    output_directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(preregistration_path, output_directory / "preregistration.json")
    result_path = output_directory / "result.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    result_path.chmod(0o444)
    (output_directory / "README.md").write_text(
        "# VQM24 dense observable-training QM batch\n\n"
        f"Status: `{payload['status']}`.\n\n"
        "This seal validates numerical enthalpy/MEP conjugacy, electron count, "
        "and the adequacy of a first-order quadratic response description. It "
        "contains no model fit, coefficient label, PCM target, or solvation label.\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seal(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
