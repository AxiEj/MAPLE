#!/usr/bin/env python3
"""Validate and seal the first VQM24 localized-field QM data record."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SELF_REPO_PATH = "tools/route2_release/seal_vqm24_nonuniform_qm_gate0.py"
ARTIFACT = "route2-vqm24-localized-field-qm-gate0-result-v1"
INHERITED_STEP_CONSISTENCY_MAXIMUM = 0.02
ELECTRON_COUNT_ERROR_MAXIMUM = 1.0e-7


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


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(left - right)
        / max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1.0e-30)
    )


def seal(args: argparse.Namespace) -> dict[str, Any]:
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    run_root = args.run_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    preregistration = json.loads(preregistration_path.read_text())
    if (
        preregistration.get("artifact")
        != "route2-vqm24-localized-field-qm-gate0-prereg-v1"
        or preregistration.get("status")
        != "locked-before-first-vqm24-qm-gate0-execution"
    ):
        raise ValueError("VQM24 QM Gate-0 preregistration is invalid.")
    for record in preregistration["input_files"].values():
        path = Path(str(record["path"])).resolve(strict=True)
        if _sha256(path) != record["sha256"]:
            raise ValueError("VQM24 QM Gate-0 input drifted.")
    gas_json_path = run_root / "gas/gas.json"
    gas_checkpoint_path = run_root / "gas/gas.chk"
    response_json_path = run_root / "qm-response/qm-response.json"
    response_npz_path = run_root / "qm-response/qm-response.npz"
    gas = json.loads(gas_json_path.read_text())
    response = json.loads(response_json_path.read_text())
    if gas.get("status") != "pass" or response.get("status") != "success":
        raise RuntimeError("VQM24 QM Gate-0 calculation did not pass its helper.")
    if _sha256(gas_checkpoint_path) != gas["result"]["checkpoint"]["sha256"]:
        raise RuntimeError("VQM24 gas checkpoint hash differs from its ledger.")
    if _sha256(response_npz_path) != response["output"]["npz_sha256"]:
        raise RuntimeError("VQM24 response NPZ hash differs from its ledger.")
    with np.load(response_npz_path, allow_pickle=False) as state:
        steps = np.asarray(state["field_steps_e"], dtype=np.float64)
        mep = np.asarray(
            state["induced_surface_mep_hartree_per_e_per_source_e"],
            dtype=np.float64,
        )
        dipole = np.asarray(
            state["induced_dipole_e_bohr_per_source_e"],
            dtype=np.float64,
        )
    if (
        tuple(steps.tolist()) != (3.0e-4, 1.0e-3)
        or mep.ndim != 3
        or mep.shape[:2] != (2, 4)
        or dipole.shape != (2, 4, 3)
    ):
        raise RuntimeError("VQM24 response arrays violate the frozen protocol.")
    mep_step = _relative(mep[0], mep[1])
    dipole_step = _relative(dipole[0], dipole[1])
    maximum_electron_error = max(
        float(record["electron_count_error_e"])
        for record in response["perturbation_records"]
    )
    cycles = [int(record["scf_cycles"]) for record in response["perturbation_records"]]
    records = {
        (
            int(record["mode_index"]),
            float(record["step_e"]),
            int(record["sign"]),
        ): record
        for record in response["perturbation_records"]
    }
    zero_energy = float(gas["result"]["energy_hartree"])
    energy_diagnostics = []
    all_curvatures_negative = True
    for mode_index in range(4):
        values = []
        for step in steps:
            negative = float(records[(mode_index, float(step), -1)]["energy_hartree"])
            positive = float(records[(mode_index, float(step), 1)]["energy_hartree"])
            slope = (positive - negative) / (2.0 * float(step))
            curvature = (
                positive + negative - 2.0 * zero_energy
            ) / float(step * step)
            all_curvatures_negative &= curvature < 0.0
            values.append(
                {
                    "step_e": float(step),
                    "central_slope_hartree_per_e": slope,
                    "central_curvature_hartree_per_e2": curvature,
                }
            )
        energy_diagnostics.append(
            {
                "mode_index": mode_index,
                "by_step": values,
                "slope_step_relative_difference": abs(
                    values[0]["central_slope_hartree_per_e"]
                    - values[1]["central_slope_hartree_per_e"]
                )
                / max(
                    abs(values[0]["central_slope_hartree_per_e"]),
                    abs(values[1]["central_slope_hartree_per_e"]),
                    1.0e-30,
                ),
                "curvature_step_relative_difference": abs(
                    values[0]["central_curvature_hartree_per_e2"]
                    - values[1]["central_curvature_hartree_per_e2"]
                )
                / max(
                    abs(values[0]["central_curvature_hartree_per_e2"]),
                    abs(values[1]["central_curvature_hartree_per_e2"]),
                    1.0e-30,
                ),
            }
        )
    gates = {
        "mep_step_consistency": mep_step <= INHERITED_STEP_CONSISTENCY_MAXIMUM,
        "dipole_step_consistency": (
            dipole_step <= INHERITED_STEP_CONSISTENCY_MAXIMUM
        ),
        "electron_count": maximum_electron_error <= ELECTRON_COUNT_ERROR_MAXIMUM,
        "all_scf_converged": len(response["perturbation_records"]) == 16,
        "all_energy_curvatures_negative": all_curvatures_negative,
    }
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "pass-numerical-gate0" if all(gates.values()) else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "sealer_path": SELF_REPO_PATH,
            "sealer_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "preregistration_path": str(preregistration_path),
            "preregistration_file_sha256": _sha256(preregistration_path),
            "preregistration_sha256": preregistration["preregistration_sha256"],
        },
        "files": {
            "gas_json": {"path": str(gas_json_path), "sha256": _sha256(gas_json_path)},
            "gas_checkpoint": {
                "path": str(gas_checkpoint_path),
                "sha256": _sha256(gas_checkpoint_path),
            },
            "response_json": {
                "path": str(response_json_path),
                "sha256": _sha256(response_json_path),
            },
            "response_npz": {
                "path": str(response_npz_path),
                "sha256": _sha256(response_npz_path),
            },
        },
        "gas": {
            "energy_hartree": zero_energy,
            "scf_cycles": int(gas["result"]["scf_cycles"]),
            "elapsed_seconds": float(gas["result"]["elapsed_seconds"]),
            "density_binding_residual_inf": float(
                gas["result"]["checkpoint_density_binding_residual_inf"]
            ),
        },
        "response": {
            "mode_count": 4,
            "field_steps_e": steps.tolist(),
            "surface_point_count": int(mep.shape[2]),
            "mep_step_relative_difference": mep_step,
            "dipole_step_relative_difference": dipole_step,
            "maximum_electron_count_error_e": maximum_electron_error,
            "minimum_scf_cycles": min(cycles),
            "maximum_scf_cycles": max(cycles),
            "energy_diagnostics": energy_diagnostics,
        },
        "gates": gates,
        "claim_boundary": {
            "independent_qm_training_datum_generated": True,
            "fit_or_training_performed": False,
            "experimental_solvation_target_read": False,
            "pcm_or_cavity_used": False,
            "capability_admitted": False,
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    output.chmod(0o444)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seal(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
