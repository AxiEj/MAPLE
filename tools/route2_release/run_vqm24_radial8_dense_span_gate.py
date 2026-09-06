#!/usr/bin/env python3
"""Run the preregistered dense fit/audit gate for the 8N radial source span."""

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

from maple.solvation.reference.radial_gto_observable_span import (
    evaluate_radial_gto_observable_span,
)


SELF_REPO_PATH = "tools/route2_release/run_vqm24_radial8_dense_span_gate.py"
PREREGISTRATION_ARTIFACT = (
    "route2-vqm24-radial8-dense-observable-span-gate-prereg-v1"
)
PARENT_ARTIFACT = "route2-vqm24-observable-training-batch-dense-prereg-v2"
ARTIFACT = "route2-vqm24-radial8-dense-observable-span-gate-result-v1"


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
        or preregistration.get("status")
        != "locked-before-remaining-span-evaluation"
    ):
        raise ValueError("Radial8 span preregistration is invalid.")
    parent_path = SOURCE_ROOT / preregistration["parent"]["path"]
    if _sha256(parent_path) != preregistration["parent"]["file_sha256"]:
        raise RuntimeError("Dense observable parent changed after span freeze.")
    parent = json.loads(parent_path.read_text())
    if parent.get("artifact") != PARENT_ARTIFACT:
        raise ValueError("Dense observable parent has the wrong artifact.")
    for name in ("span_module", "coupling"):
        path = SOURCE_ROOT / preregistration["source"][f"{name}_path"]
        if _sha256(path) != preregistration["source"][f"{name}_sha256"]:
            raise RuntimeError(f"Span-gate source changed after freeze: {name}.")

    gate_values = preregistration["gates"]
    opened = set(preregistration["opened_pilots"]["record_ids"])
    validation = set(preregistration["validation_record_ids"])
    records: list[dict[str, Any]] = []
    for record in parent["records"]:
        record_id = record["record_id"]
        surface_path = Path(record["inputs"]["surface"]["path"])
        modes_path = Path(record["inputs"]["modes"]["path"])
        observable_json_path = Path(record["outputs"]["observable_json"])
        observable_npz_path = Path(record["outputs"]["observable_npz"])
        observable = json.loads(observable_json_path.read_text())
        if observable.get("status") != "success":
            raise RuntimeError(f"Observable record failed for {record_id}.")
        if _sha256(observable_npz_path) != observable["output"]["npz_sha256"]:
            raise RuntimeError(f"Observable NPZ drifted for {record_id}.")
        with np.load(surface_path, allow_pickle=False) as surface:
            positions = np.asarray(
                surface["atom_positions_angstrom"],
                dtype=np.float64,
            )
            points = np.asarray(surface["surface_points_bohr"], dtype=np.float64)
            weights = np.asarray(surface["quadrature_weights"], dtype=np.float64)
            partitions = np.asarray(surface["partition_indices"], dtype=np.int64)
        with np.load(modes_path, allow_pickle=False) as modes:
            source_indices = np.asarray(
                modes["source_surface_indices"],
                dtype=np.int64,
            )
        with np.load(observable_npz_path, allow_pickle=False) as state:
            zero_mep = np.asarray(
                state["zero_total_surface_mep_hartree_per_e"],
                dtype=np.float64,
            )
            response_mep = np.asarray(
                state["induced_surface_mep_hartree_per_e_per_source_e"],
                dtype=np.float64,
            )
        result = evaluate_radial_gto_observable_span(
            atom_positions_angstrom=positions,
            surface_points_bohr=points,
            quadrature_weights=weights,
            partition_indices=partitions,
            source_surface_indices=source_indices,
            zero_total_surface_mep_hartree_per_e=zero_mep,
            induced_surface_mep_hartree_per_e_per_source_e=response_mep,
            total_charge_e=0.0,
        )
        metrics = asdict(result)
        record_gates = {
            "weighted_fit_operator_rank": (
                result.weighted_fit_operator_rank == result.reduced_dimension
            ),
            "zero_mep_audit_relative": (
                result.zero_mep_audit_relative
                <= float(gate_values["zero_mep_audit_relative_maximum"])
            ),
            "zero_mep_audit_absolute": (
                result.zero_mep_audit_max_absolute_hartree_per_e
                <= float(
                    gate_values["zero_mep_audit_max_absolute_hartree_per_e"]
                )
            ),
            "response_mep_audit_relative": (
                result.response_mep_audit_relative
                <= float(gate_values["response_mep_audit_relative_maximum"])
            ),
            "response_mep_audit_absolute": (
                result.response_mep_audit_max_absolute_hartree_per_e_per_source_e
                <= float(
                    gate_values[
                        "response_mep_audit_max_absolute_hartree_per_e_per_source_e"
                    ]
                )
            ),
            "charge_constraint": (
                result.maximum_charge_constraint_residual_e
                <= float(gate_values["maximum_charge_constraint_residual_e"])
            ),
            "source_response": (
                result.source_response_relative_error
                <= float(gate_values["source_response_relative_error_maximum"])
            ),
            "source_response_reciprocity": (
                result.source_response_reciprocity_relative
                <= float(
                    gate_values["source_response_reciprocity_relative_maximum"]
                )
            ),
            "source_response_passivity": (
                max(result.source_response_symmetric_eigenvalues_hartree_per_e2)
                <= float(
                    gate_values[
                        "source_response_maximum_symmetric_eigenvalue_hartree_per_e2"
                    ]
                )
            ),
            "response_direction_rank": (
                result.response_direction_rank
                == int(gate_values["response_direction_rank"])
            ),
            "positive_hessian_extension": (
                result.positive_hessian_extension_minimum_gram_eigenvalue
                >= float(
                    gate_values[
                        "positive_hessian_extension_minimum_gram_eigenvalue"
                    ]
                )
            ),
            "no_coefficient_label": not result.coefficient_label_emitted,
            "no_model_fit": not result.model_fit_performed,
        }
        role = "opened-pilot" if record_id in opened else "validation"
        if role == "validation" and record_id not in validation:
            raise RuntimeError("Span-gate record is outside both frozen partitions.")
        records.append(
            {
                "record_id": record_id,
                "role": role,
                "status": "pass" if all(record_gates.values()) else "fail",
                "metrics": metrics,
                "gates": record_gates,
                "observable_json_sha256": _sha256(observable_json_path),
                "observable_npz_sha256": _sha256(observable_npz_path),
            }
        )
    records.sort(key=lambda record: record["record_id"])
    validation_records = [record for record in records if record["role"] == "validation"]
    all_validation_pass = all(record["status"] == "pass" for record in validation_records)
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "pass-radial8-observable-span" if all_validation_pass else "fail",
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
            "opened_pilot_count": len(records) - len(validation_records),
            "validation_record_count": len(validation_records),
            "all_validation_pass": all_validation_pass,
            "maximum_zero_mep_audit_relative": max(
                record["metrics"]["zero_mep_audit_relative"]
                for record in validation_records
            ),
            "maximum_response_mep_audit_relative": max(
                record["metrics"]["response_mep_audit_relative"]
                for record in validation_records
            ),
            "maximum_source_response_relative_error": max(
                record["metrics"]["source_response_relative_error"]
                for record in validation_records
            ),
        },
        "claim_boundary": {
            "radial8_source_span_admitted_for_observable_head_training": (
                all_validation_pass
            ),
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
        "# VQM24 radial8 dense observable span gate\n\n"
        f"Status: `{payload['status']}`.\n\n"
        "Every validation molecule is fit only on its dense fit partition and "
        "evaluated on the independent frozen audit partition. No fitted source "
        "coefficient is stored or used as a label. This gate tests representation "
        "feasibility only; it does not train a head or admit MAPLE capabilities.\n"
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
