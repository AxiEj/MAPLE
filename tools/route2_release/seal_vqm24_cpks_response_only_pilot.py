#!/usr/bin/env python3
"""Reassess the frozen Gate-B CPKS evidence for response-only targets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SELF_REPO_PATH = "tools/route2_release/seal_vqm24_cpks_response_only_pilot.py"
ARTIFACT = "route2-vqm24-cpks-response-only-pilot-result-v1"
PROTOCOL_ARTIFACT = "route2-vqm24-cpks-response-only-pilot-protocol-v1"


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
    protocol_path = args.protocol.expanduser().resolve(strict=True)
    gate_b_path = args.gate_b_result.expanduser().resolve(strict=True)
    output_directory = args.output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    protocol = json.loads(protocol_path.read_text())
    gate_b = json.loads(gate_b_path.read_text())
    if (
        protocol.get("artifact") != PROTOCOL_ARTIFACT
        or protocol.get("status") != "locked-before-response-only-assessment"
    ):
        raise ValueError("Response-only CPKS pilot protocol is invalid.")
    if _sha256(SOURCE_ROOT / SELF_REPO_PATH) != protocol["source"][
        "sealer_sha256"
    ]:
        raise RuntimeError("Response-only CPKS sealer changed after freeze.")
    if _sha256(gate_b_path) != protocol["parent"]["file_sha256"]:
        raise RuntimeError("Gate-B evidence changed before response-only assessment.")
    required = tuple(protocol["required_gate_names"])
    excluded = tuple(protocol["excluded_gate_names"])
    if set(required) & set(excluded) or excluded != ("curvature_absolute",):
        raise ValueError("Response-only gate partition is inconsistent.")
    records = []
    for record in gate_b["records"]:
        unknown = set(record["gates"]).difference(required, excluded)
        if unknown:
            raise RuntimeError(f"Unclassified Gate-B checks: {sorted(unknown)}")
        required_results = {name: bool(record["gates"][name]) for name in required}
        passed = all(required_results.values())
        records.append(
            {
                "stratum": record["stratum"],
                "status": "pass" if passed else "fail",
                "required_gates": required_results,
                "excluded_diagnostic": {
                    name: bool(record["gates"][name]) for name in excluded
                },
                "metrics": {
                    name: value
                    for name, value in record["metrics"].items()
                    if "curvature" not in name
                },
                "gate_b_record_sha256": _canonical_sha256(record),
            }
        )
    all_pass = all(record["status"] == "pass" for record in records)
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "pass-response-only-pilot" if all_pass else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "sealer_path": SELF_REPO_PATH,
            "sealer_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "protocol_file_sha256": _sha256(protocol_path),
            "protocol_sha256": protocol["protocol_sha256"],
            "parent_gate_b_file_sha256": _sha256(gate_b_path),
            "parent_gate_b_payload_sha256": gate_b["result_sha256"],
        },
        "records": sorted(records, key=lambda record: record["stratum"]),
        "aggregate": {
            "record_count": len(records),
            "pass_count": sum(record["status"] == "pass" for record in records),
            "maximum_mep_symmetric_relative": max(
                record["metrics"]["mep_symmetric_relative"] for record in records
            ),
            "maximum_dipole_symmetric_relative": max(
                record["metrics"]["dipole_symmetric_relative"]
                for record in records
            ),
            "maximum_cpks_residual_relative_frobenius": max(
                record["metrics"]["maximum_cpks_residual_relative_frobenius"]
                for record in records
            ),
        },
        "decision": {
            "response_only_pilot_passed": all_pass,
            "full_32_record_coverage_preregistration_authorized": all_pass,
            "cpks_generator_admitted": False,
            "energy_curvature_target_authorized": False,
            "new_QM_response_generated": False,
            "model_training_authorized": False,
        },
        "claim_boundary": {
            "existing_opened_gate_b_evidence_reassessed": True,
            "new_QM_calculation_run": False,
            "energy_curvature_excluded_because_not_a_training_target": True,
            "response_only_generator_coverage_complete": False,
            "experimental_solvation_target_read": False,
            "pcm_or_cavity_used": False,
            "model_fit_or_accuracy_measured": False,
            "validation_or_blind_formula_opened": False,
            "maple_capability_admitted": False,
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    output_directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(protocol_path, output_directory / "protocol.json")
    result_path = output_directory / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    result_path.chmod(0o444)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--gate-b-result", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
