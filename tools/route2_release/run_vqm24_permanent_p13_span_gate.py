#!/usr/bin/env python3
"""Run the preregistered point-P13 permanent-source representation gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maple.solvation.reference.p13_permanent_span import (
    evaluate_p13_permanent_span,
)


SELF_REPO_PATH = "tools/route2_release/run_vqm24_permanent_p13_span_gate.py"
PREREGISTRATION_ARTIFACT = "route2-vqm24-permanent-p13-span-gate-prereg-v1"
PARENT_ARTIFACT = "route2-vqm24-permanent-p13-zero-field-mep-prereg-v1"
ARTIFACT = "route2-vqm24-permanent-p13-span-gate-result-v1"


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
        or preregistration.get("status") != "locked-before-p13-span-evaluation"
    ):
        raise ValueError("P13 span preregistration is invalid.")
    parent_path = SOURCE_ROOT / preregistration["parent"]["path"]
    if _sha256(parent_path) != preregistration["parent"]["file_sha256"]:
        raise RuntimeError("P13 zero-field parent changed after gate freeze.")
    parent = json.loads(parent_path.read_text())
    if parent.get("artifact") != PARENT_ARTIFACT:
        raise ValueError("P13 zero-field parent has the wrong artifact.")
    for name in ("span_module", "radial_coupling", "quadrupole_coupling"):
        path = SOURCE_ROOT / preregistration["source"][f"{name}_path"]
        if _sha256(path) != preregistration["source"][f"{name}_sha256"]:
            raise RuntimeError(f"P13 gate source changed after freeze: {name}.")
    expected_ids = set(preregistration["record_ids"])
    records = []
    gates = preregistration["gates"]
    for record in parent["records"]:
        record_id = record["record_id"]
        if record_id not in expected_ids:
            raise RuntimeError("P13 zero-field record is outside the frozen gate.")
        surface_path = Path(record["inputs"]["surface"]["path"])
        output_json_path = Path(record["outputs"]["json"])
        output_npz_path = Path(record["outputs"]["npz"])
        zero_field = json.loads(output_json_path.read_text())
        if zero_field.get("status") != "success":
            raise RuntimeError(f"P13 zero-field helper failed for {record_id}.")
        if _sha256(output_npz_path) != zero_field["output"]["npz_sha256"]:
            raise RuntimeError(f"P13 zero-field NPZ drifted for {record_id}.")
        with np.load(surface_path, allow_pickle=False) as surface:
            positions = np.asarray(surface["atom_positions_angstrom"], dtype=np.float64)
            points = np.asarray(surface["surface_points_bohr"], dtype=np.float64)
            weights = np.asarray(surface["quadrature_weights"], dtype=np.float64)
            partitions = np.asarray(surface["partition_indices"], dtype=np.int64)
        with np.load(output_npz_path, allow_pickle=False) as state:
            target = np.asarray(
                state["total_surface_mep_hartree_per_e"],
                dtype=np.float64,
            )
        result = evaluate_p13_permanent_span(
            atom_positions_angstrom=positions,
            surface_points_bohr=points,
            quadrature_weights=weights,
            partition_indices=partitions,
            total_surface_mep_hartree_per_e=target,
            total_charge_e=0.0,
        )
        metrics = asdict(result)
        record_gates = {
            "fit_operator_rank": (
                result.weighted_fit_operator_rank == result.reduced_dimension
            ),
            "audit_relative_mep": (
                result.audit_relative_mep_error
                <= float(gates["audit_relative_mep_error_maximum"])
            ),
            "audit_absolute_mep": (
                result.audit_max_absolute_hartree_per_e
                <= float(gates["audit_max_absolute_hartree_per_e"])
            ),
            "audit_minus_fit": (
                result.audit_minus_fit_relative
                <= float(gates["audit_minus_fit_relative_maximum"])
            ),
            "audit_to_fit_ratio": (
                result.audit_to_fit_ratio
                <= float(gates["audit_to_fit_ratio_maximum"])
            ),
            "charge_constraint": (
                result.charge_constraint_residual_e
                <= float(gates["charge_constraint_residual_e_maximum"])
            ),
            "no_coefficient_label": not result.coefficient_label_emitted,
            "no_model_fit": not result.model_fit_performed,
        }
        records.append(
            {
                "record_id": record_id,
                "status": "pass" if all(record_gates.values()) else "fail",
                "metrics": metrics,
                "gates": record_gates,
                "zero_field_json_sha256": _sha256(output_json_path),
                "zero_field_npz_sha256": _sha256(output_npz_path),
            }
        )
    if {record["record_id"] for record in records} != expected_ids:
        raise RuntimeError("P13 span gate did not consume the frozen record set.")
    records.sort(key=lambda record: record["record_id"])
    all_pass = all(record["status"] == "pass" for record in records)
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "pass-permanent-p13-span" if all_pass else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "runner_path": SELF_REPO_PATH,
            "runner_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "preregistration_path": str(preregistration_path.relative_to(SOURCE_ROOT)),
            "preregistration_file_sha256": _sha256(preregistration_path),
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "parent_file_sha256": _sha256(parent_path),
        },
        "records": records,
        "aggregate": {
            "record_count": len(records),
            "all_records_pass": all_pass,
            "maximum_audit_relative_mep_error": max(
                record["metrics"]["audit_relative_mep_error"] for record in records
            ),
            "maximum_audit_minus_fit_relative": max(
                record["metrics"]["audit_minus_fit_relative"] for record in records
            ),
            "maximum_audit_to_fit_ratio": max(
                record["metrics"]["audit_to_fit_ratio"] for record in records
            ),
        },
        "claim_boundary": {
            "point_p13_permanent_span_admitted_for_head_training": all_pass,
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
        "# VQM24 point-P13 permanent source span gate\n\n"
        f"Status: `{payload['status']}`.\n\n"
        "Every molecule is fit on the frozen 86-point fit frame and evaluated "
        "on the rotated audit frame. Coefficients are discarded and are never "
        "used as labels. This is a representation gate, not model training.\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
