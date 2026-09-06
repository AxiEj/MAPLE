#!/usr/bin/env python3
"""Aggregate the five preregistered VQM24 rare-element QM Gate-A records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SELF_REPO_PATH = "tools/route2_release/aggregate_vqm24_nonuniform_qm_gate_a.py"
RECORD_ARTIFACT = "route2-vqm24-localized-field-qm-record-level3-result-v1"
ARTIFACT = "route2-vqm24-localized-field-qm-gate-a-summary-v1"
EXPECTED_STRATA = ("bromine", "chlorine", "fluorine", "phosphorus", "sulfur")


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


def _relative_to_source_root(path: Path) -> str:
    try:
        return str(path.relative_to(SOURCE_ROOT))
    except ValueError:
        return str(path)


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    output_directory = args.output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    result_paths = [
        path.expanduser().resolve(strict=True) for path in args.record_result
    ]
    if len(result_paths) != len(EXPECTED_STRATA):
        raise ValueError("Gate-A aggregation requires exactly five record results.")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in result_paths:
        result = json.loads(path.read_text())
        if result.get("artifact") != RECORD_ARTIFACT:
            raise ValueError(f"Unexpected Gate-A record artifact: {path}.")
        stratum = str(result["selection"]["stratum"])
        if stratum in seen:
            raise ValueError(f"Duplicate Gate-A stratum: {stratum}.")
        seen.add(stratum)
        for file_record in result["files"].values():
            file_path = Path(str(file_record["path"]))
            if not file_path.is_absolute():
                file_path = SOURCE_ROOT / file_path
            file_path = file_path.resolve(strict=True)
            if _sha256(file_path) != file_record["sha256"]:
                raise RuntimeError(f"Gate-A record file drifted: {file_path}.")
        records.append(
            {
                "stratum": stratum,
                "status": result["status"],
                "formula": result["selection"]["chemical_formula_hill"],
                "dataset_index": int(result["selection"]["dataset_index"]),
                "record_sha256": result["selection"]["record_sha256"],
                "result_path": _relative_to_source_root(path),
                "result_file_sha256": _sha256(path),
                "result_sha256": result["result_sha256"],
                "gas_scf_cycles": int(result["gas"]["scf_cycles"]),
                "response_scf_cycle_range": [
                    int(result["response"]["minimum_scf_cycles"]),
                    int(result["response"]["maximum_scf_cycles"]),
                ],
                "mep_step_relative_difference": float(
                    result["response"]["mep_step_relative_difference"]
                ),
                "dipole_step_relative_difference": float(
                    result["response"]["dipole_step_relative_difference"]
                ),
                "maximum_electron_count_error_e": float(
                    result["response"]["maximum_electron_count_error_e"]
                ),
                "all_gates_pass": all(bool(value) for value in result["gates"].values()),
            }
        )
    if tuple(sorted(seen)) != EXPECTED_STRATA:
        raise ValueError("Gate-A record strata differ from the frozen set.")
    records.sort(key=lambda item: item["stratum"])
    all_records_pass = all(
        record["status"] == "pass-numerical-gate-a-record"
        and record["all_gates_pass"]
        for record in records
    )
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "pass-numerical-gate-a" if all_records_pass else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "aggregator_path": SELF_REPO_PATH,
            "aggregator_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
        },
        "protocol": {
            "record_count": len(records),
            "strata": list(EXPECTED_STRATA),
            "electronic_structure": "RKS omegaB97M-V/def2-TZVPD",
            "semilocal_grid": "PySCF level 3",
            "nonlocal_grid": "PySCF level 3",
            "localized_mode_count_per_record": 4,
            "field_steps_e": [3.0e-4, 1.0e-3],
        },
        "records": records,
        "aggregate": {
            "maximum_mep_step_relative_difference": max(
                record["mep_step_relative_difference"] for record in records
            ),
            "maximum_dipole_step_relative_difference": max(
                record["dipole_step_relative_difference"] for record in records
            ),
            "maximum_electron_count_error_e": max(
                record["maximum_electron_count_error_e"] for record in records
            ),
            "all_records_pass": all_records_pass,
        },
        "claim_boundary": {
            "rare_element_numerical_protocol_validated": all_records_pass,
            "independent_qm_training_data_generated": all_records_pass,
            "fit_or_training_performed": False,
            "experimental_solvation_target_read": False,
            "vqm24_energy_target_used": False,
            "pcm_or_cavity_used": False,
            "model_accuracy_measured": False,
            "capability_admitted": False,
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    output_directory.mkdir(parents=True, exist_ok=False)
    result_path = output_directory / "summary.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    result_path.chmod(0o444)
    (output_directory / "README.md").write_text(
        "# VQM24 nonuniform-field QM Gate-A\n\n"
        f"Status: `{payload['status']}`.\n\n"
        "Five frozen train records cover F, P, S, Cl, and Br under one common "
        "RKS omegaB97M-V/def2-TZVPD and PySCF level-3 grid protocol. The gate "
        "checks numerical response consistency only. It does not train or "
        "evaluate a model, read VQM24 energy labels or experimental solvation "
        "targets, use PCM/cavity labels, or admit a MAPLE capability.\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-result", type=Path, action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
