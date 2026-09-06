#!/usr/bin/env python3
"""Seal all 32 twelve-mode sector-balanced CPKS response records."""

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
SELF_REPO_PATH = "tools/route2_release/seal_vqm24_sector_balanced_cpks.py"
ARTIFACT = "route2-vqm24-sector-balanced-twelve-mode-cpks-result-v1"
RECORD_ARTIFACT = "route2-vqm24-sector-balanced-twelve-mode-cpks-record-v1"
PREREGISTRATION_ARTIFACT = (
    "route2-vqm24-sector-balanced-twelve-mode-cpks-prereg-v1"
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
        or preregistration.get("status") != "locked-before-twelve-mode-execution"
    ):
        raise ValueError("Twelve-mode preregistration is invalid.")
    if _sha256(SOURCE_ROOT / SELF_REPO_PATH) != preregistration["source"][
        "sealer_sha256"
    ]:
        raise RuntimeError("Twelve-mode sealer changed after freeze.")
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
            raise RuntimeError(f"Twelve-mode result identity failed: {frozen['record_id']}")
        records.append(
            {
                "record_id": frozen["record_id"],
                "status": result["status"],
                "metrics": result["metrics"],
                "gates": result["gates"],
                "result_file_sha256": _sha256(result_path),
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
        "prefix_mep_symmetric_relative",
        "prefix_dipole_symmetric_relative",
        "maximum_group_residual_relative_frobenius",
        "maximum_group_residual_relative_infinity",
        "maximum_electron_number_derivative_abs",
        "source_response_reciprocity_relative_frobenius",
        "maximum_source_response_eigenvalue_hartree_per_e2",
    )
    aggregate = {
        "record_count": len(records),
        "pass_count": sum(record["status"] == "pass" for record in records),
        **{
            f"maximum_{name}": max(record["metrics"][name] for record in records)
            for name in metric_names
        },
        "minimum_source_response_eigenvalue_hartree_per_e2": min(
            record["metrics"]["minimum_source_response_eigenvalue_hartree_per_e2"]
            for record in records
        ),
        "total_elapsed_seconds": sum(
            record["metrics"]["total_elapsed_seconds"] for record in records
        ),
    }
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "pass-twelve-mode-response-generation" if all_pass else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "sealer_path": SELF_REPO_PATH,
            "sealer_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "preregistration_file_sha256": _sha256(preregistration_path),
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "record_runner_sha256": preregistration["source"][
                "record_runner_sha256"
            ],
            "grouped_runner_sha256": preregistration["source"][
                "grouped_runner_sha256"
            ],
        },
        "records": sorted(records, key=lambda record: record["record_id"]),
        "aggregate": aggregate,
        "decision": {
            "twelve_mode_response_data_admitted_for_train_development": all_pass,
            "energy_curvature_target_authorized": False,
            "validation_or_blind_opened": False,
            "model_or_maple_capability_admitted": False,
        },
        "claim_boundary": {
            "independent_train_split_QM_response_generation_complete": all_pass,
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
