#!/usr/bin/env python3
"""Seal the complete 32-record response-only CPKS coverage gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SELF_REPO_PATH = "tools/route2_release/seal_vqm24_cpks_response_only_coverage.py"
ARTIFACT = "route2-vqm24-cpks-response-only-coverage-result-v1"
RECORD_ARTIFACT = "route2-vqm24-cpks-response-only-coverage-record-v1"
PREREGISTRATION_ARTIFACT = (
    "route2-vqm24-cpks-response-only-coverage-prereg-v1"
)


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
        raise ValueError("Coverage preregistration is invalid.")
    if _sha256(SOURCE_ROOT / SELF_REPO_PATH) != preregistration["source"][
        "sealer_sha256"
    ]:
        raise RuntimeError("Coverage sealer changed after preregistration.")
    records = []
    copied = []
    for frozen in preregistration["records"]:
        source_directory = Path(frozen["output_directory"])
        result_path = (source_directory / "result.json").resolve(strict=True)
        result = json.loads(result_path.read_text())
        if (
            result.get("artifact") != RECORD_ARTIFACT
            or result.get("record_id") != frozen["record_id"]
            or result["source"]["preregistration_sha256"]
            != preregistration["preregistration_sha256"]
        ):
            raise RuntimeError(f"Coverage result identity failed: {frozen['record_id']}")
        if _canonical_sha256(
            {key: value for key, value in result.items() if key != "result_sha256"}
        ) != result["result_sha256"]:
            raise RuntimeError(f"Coverage payload digest failed: {frozen['record_id']}")
        records.append(
            {
                "record_id": frozen["record_id"],
                "status": result["status"],
                "metrics": result["metrics"],
                "gates": result["gates"],
                "result_sha256": _sha256(result_path),
                "result_payload_sha256": result["result_sha256"],
            }
        )
        destination = output_directory / "records" / frozen["record_id"]
        copied.extend(
            [
                (result_path, destination / "result.json"),
                (source_directory / "cpks.json", destination / "cpks.json"),
                (source_directory / "cpks.npz", destination / "cpks.npz"),
            ]
        )
    all_pass = len(records) == 32 and all(record["status"] == "pass" for record in records)
    metric_names = (
        "mep_symmetric_relative",
        "mep_mode_symmetric_relative_maximum",
        "dipole_symmetric_relative",
        "dipole_mode_symmetric_relative_maximum",
        "maximum_cpks_residual_relative_frobenius",
        "maximum_cpks_residual_relative_infinity",
        "maximum_energy_identity_abs_hartree_per_e2",
        "maximum_electron_number_derivative_abs",
        "reciprocity_relative_frobenius",
        "maximum_passivity_eigenvalue_hartree_per_e2",
    )
    aggregate = {
        "record_count": len(records),
        "pass_count": sum(record["status"] == "pass" for record in records),
        **{
            f"maximum_{name}": max(record["metrics"][name] for record in records)
            for name in metric_names
        },
        "total_cpks_elapsed_seconds": sum(
            record["metrics"]["cpks_total_elapsed_seconds"] for record in records
        ),
    }
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "pass-response-only-cpks-coverage" if all_pass else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "sealer_path": SELF_REPO_PATH,
            "sealer_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "preregistration_file_sha256": _sha256(preregistration_path),
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "record_runner_sha256": preregistration["source"][
                "record_runner_sha256"
            ],
            "cpks_runner_sha256": preregistration["source"]["cpks_runner_sha256"],
        },
        "records": sorted(records, key=lambda record: record["record_id"]),
        "aggregate": aggregate,
        "decision": {
            "cpks_response_only_generator_admitted_for_train_domain": all_pass,
            "twelve_mode_generation_authorized": all_pass,
            "energy_curvature_target_authorized": False,
            "model_training_or_maple_capability_admitted": False,
        },
        "claim_boundary": {
            "train_split_response_generator_coverage_complete": all_pass,
            "only_existing_four_mode_finite_field_targets_compared": True,
            "new_twelve_mode_QM_response_generated": False,
            "model_fit_or_accuracy_measured": False,
            "experimental_solvation_target_read": False,
            "pcm_or_cavity_used": False,
            "validation_or_blind_formula_opened": False,
            "maple_capability_admitted": False,
        },
        "runtime": {"python_executable": sys.executable},
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    output_directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(preregistration_path, output_directory / "preregistration.json")
    for source, destination in copied:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    result_path = output_directory / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    result_path.chmod(0o444)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
