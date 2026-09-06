#!/usr/bin/env python3
"""Freeze the 32-record zero-field P13 permanent-MEP probe batch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maple.solvation.reference.permanent_quadrupole_probes import (  # noqa: E402
    PERMANENT_P13_LEBEDEV_ORDER,
    PERMANENT_P13_LEBEDEV_POINTS,
    build_permanent_p13_probe_partition,
)
from tools.route2_release import run_vqm24_zero_field_mep_record as zero_mep  # noqa: E402


SELF_REPO_PATH = "tools/route2_release/prepare_vqm24_permanent_p13_mep_batch.py"
BUILDER_REPO_PATH = "maple/solvation/reference/permanent_quadrupole_probes.py"
ARTIFACT = "route2-vqm24-permanent-p13-zero-field-mep-prereg-v1"
PARENT_ARTIFACT = "route2-vqm24-observable-training-batch-dense-prereg-v2"


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


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    parent_path = args.parent.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    parent = json.loads(parent_path.read_text())
    if (
        parent.get("artifact") != PARENT_ARTIFACT
        or parent.get("status") != "locked-before-dense-observable-qm-execution"
    ):
        raise ValueError("Dense observable parent is invalid.")
    builder_path = SOURCE_ROOT / BUILDER_REPO_PATH
    runner_path = SOURCE_ROOT / zero_mep.SELF_REPO_PATH
    records = []
    for parent_record in parent["records"]:
        sparse_surface_path = Path(parent_record["inputs"]["surface"]["path"])
        with np.load(sparse_surface_path, allow_pickle=False) as state:
            positions = np.asarray(state["atom_positions_angstrom"], dtype=np.float64)
            radii = np.asarray(state["ase_vdw_radii_angstrom"], dtype=np.float64)
            atomic_numbers = np.asarray(state["atomic_numbers"], dtype=np.int64)
        probes = build_permanent_p13_probe_partition(
            positions,
            radii,
            clearance_angstrom=float(parent["probe_protocol"]["clearance_angstrom"]),
        )
        fit_count = int(np.count_nonzero(probes.partition_indices == 0))
        audit_count = int(np.count_nonzero(probes.partition_indices == 1))
        minimum = 13 * len(positions) - 1
        if fit_count <= minimum or audit_count <= minimum:
            raise RuntimeError("P13 permanent probes are not overdetermined.")
        record_root = sparse_surface_path.parent
        surface_path = record_root / "permanent-p13-surface-v1.npz"
        output_json = record_root / "permanent-p13-zero-mep/result.json"
        output_npz = record_root / "permanent-p13-zero-mep/result.npz"
        if surface_path.exists() or output_json.exists() or output_npz.exists():
            raise FileExistsError(surface_path)
        np.savez(
            surface_path,
            surface_points_bohr=probes.surface_points_bohr,
            quadrature_weights=probes.quadrature_weights,
            parent_atom_indices=probes.parent_atom_indices,
            partition_indices=probes.partition_indices,
            atomic_numbers=atomic_numbers,
            atom_positions_angstrom=positions,
            ase_vdw_radii_angstrom=radii,
            lebedev_order=np.asarray(PERMANENT_P13_LEBEDEV_ORDER),
            lebedev_points_per_frame=np.asarray(PERMANENT_P13_LEBEDEV_POINTS),
        )
        surface_path.chmod(0o444)
        checkpoint_path = Path(parent_record["outputs"]["gas_directory"]) / "gas.chk"
        gas_json_path = checkpoint_path.with_name("gas.json")
        gas = json.loads(gas_json_path.read_text())
        checkpoint_sha256 = _sha256(checkpoint_path)
        if checkpoint_sha256 != gas["result"]["checkpoint"]["sha256"]:
            raise RuntimeError("Gas checkpoint differs from its ledger.")
        records.append(
            {
                "record_id": parent_record["record_id"],
                "selection_record": parent_record["selection_record"],
                "inputs": {
                    "checkpoint": {
                        "path": str(checkpoint_path),
                        "sha256": checkpoint_sha256,
                    },
                    "surface": {
                        "path": str(surface_path),
                        "sha256": _sha256(surface_path),
                        "fit_point_count": fit_count,
                        "audit_point_count": audit_count,
                    },
                },
                "outputs": {
                    "json": str(output_json),
                    "npz": str(output_npz),
                },
            }
        )
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-permanent-p13-zero-field-mep-execution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent": {
            "path": str(parent_path.relative_to(SOURCE_ROOT)),
            "file_sha256": _sha256(parent_path),
            "preregistration_sha256": parent["preregistration_sha256"],
        },
        "selection": parent["selection"],
        "records": records,
        "source": {
            "creator_path": SELF_REPO_PATH,
            "creator_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "builder_path": BUILDER_REPO_PATH,
            "builder_sha256": _sha256(builder_path),
            "runner_path": zero_mep.SELF_REPO_PATH,
            "runner_sha256": _sha256(runner_path),
        },
        "protocol": {
            "purpose": "zero-field full molecular MEP for P13 permanent source",
            "fit_angular_grid": "PySCF 86-point Lebedev order 15",
            "audit_angular_grid": "same rule under frozen audit rotation",
            "each_partition_exceeds_13N_minus_1": True,
            "field_response_generated": False,
            "gas_scf_reused_without_rerun": True,
            "cavity_or_solvent_used": False,
        },
        "claim_boundary": {
            "fit_or_training_performed": False,
            "density_or_partition_coefficient_label_emitted": False,
            "experimental_solvation_target_read": False,
            "vqm24_energy_target_used": False,
            "pcm_or_cavity_used": False,
            "model_accuracy_measured": False,
            "capability_admitted": False,
        },
    }
    payload["preregistration_sha256"] = _canonical_sha256(payload)
    with output_path.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    output_path.chmod(0o444)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
