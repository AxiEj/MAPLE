#!/usr/bin/env python3
"""Seal the 32-record gas checkpoint stage of observable training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SELF_REPO_PATH = "tools/route2_release/seal_vqm24_observable_training_gas.py"
PREREGISTRATION_ARTIFACT = "route2-vqm24-observable-training-batch-dense-prereg-v2"
ARTIFACT = "route2-vqm24-observable-training-gas-result-v1"


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
        raise ValueError("Dense observable training preregistration is invalid.")
    records = []
    for preregistered in preregistration["records"]:
        gas_directory = Path(preregistered["outputs"]["gas_directory"])
        gas_json_path = gas_directory / "gas.json"
        checkpoint_path = gas_directory / "gas.chk"
        gas = json.loads(gas_json_path.read_text())
        if gas.get("status") != "pass":
            raise RuntimeError(
                f"Gas helper failed for {preregistered['record_id']}."
            )
        if gas["input"]["mol2_sha256"] != preregistered["inputs"]["mol2"]["sha256"]:
            raise RuntimeError("Gas helper used a different frozen geometry.")
        checkpoint_sha256 = _sha256(checkpoint_path)
        if checkpoint_sha256 != gas["result"]["checkpoint"]["sha256"]:
            raise RuntimeError("Gas checkpoint differs from its helper ledger.")
        if (
            gas["numerics"]["nonlocal_grid_profile"] != "level"
            or int(gas["numerics"]["nonlocal_grid_level"]) != 3
        ):
            raise RuntimeError("Gas checkpoint does not use the frozen level-3 NLC grid.")
        records.append(
            {
                "record_id": preregistered["record_id"],
                "stratum": preregistered["selection_record"]["stratum"],
                "formula": preregistered["selection_record"][
                    "chemical_formula_hill"
                ],
                "gas_json_sha256": _sha256(gas_json_path),
                "checkpoint_sha256": checkpoint_sha256,
                "energy_hartree": float(gas["result"]["energy_hartree"]),
                "scf_cycles": int(gas["result"]["scf_cycles"]),
                "elapsed_seconds": float(gas["result"]["elapsed_seconds"]),
                "density_binding_residual_inf": float(
                    gas["result"]["checkpoint_density_binding_residual_inf"]
                ),
                "nonlocal_grid_point_count": int(
                    gas["result"]["nonlocal_grid_point_count"]
                ),
            }
        )
    records.sort(key=lambda record: record["record_id"])
    maximum_density_residual = max(
        record["density_binding_residual_inf"] for record in records
    )
    gates = {
        "record_count": len(records) == 32,
        "density_binding": maximum_density_residual <= 1.0e-12,
        "all_scf_cycles_positive": all(record["scf_cycles"] > 0 for record in records),
    }
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "pass-observable-training-gas" if all(gates.values()) else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "sealer_path": SELF_REPO_PATH,
            "sealer_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "preregistration_path": str(preregistration_path.relative_to(SOURCE_ROOT)),
            "preregistration_file_sha256": _sha256(preregistration_path),
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "gas_runner_sha256": preregistration["source"]["gas_runner"]["sha256"],
        },
        "records": records,
        "aggregate": {
            "record_count": len(records),
            "minimum_scf_cycles": min(record["scf_cycles"] for record in records),
            "maximum_scf_cycles": max(record["scf_cycles"] for record in records),
            "total_elapsed_seconds": sum(
                record["elapsed_seconds"] for record in records
            ),
            "maximum_density_binding_residual_inf": maximum_density_residual,
        },
        "gates": gates,
        "claim_boundary": {
            "independent_qm_gas_checkpoints_generated": True,
            "observable_response_generated": False,
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
        "# VQM24 observable-training gas checkpoints\n\n"
        f"Status: `{payload['status']}`.\n\n"
        "All 32 frozen train geometries were solved with the common level-3 "
        "RKS omegaB97M-V/def2-TZVPD protocol. This stage contains no field "
        "response, fit, solvation label, or capability admission.\n"
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
