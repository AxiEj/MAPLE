#!/usr/bin/env python3
"""Supersede sparse observable probes with dense fit/audit partitions."""

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

from maple.solvation.reference.exterior_probe_partition import (  # noqa: E402
    DENSE_EXTERIOR_AUDIT_ROTATION,
    DENSE_EXTERIOR_LEBEDEV_ORDER,
    DENSE_EXTERIOR_LEBEDEV_POINTS,
    DENSE_EXTERIOR_PARTITION_NAMES,
    build_dense_exterior_probe_partition,
)
from tools.route2_release import run_vqm24_observable_training_record as observable  # noqa: E402


SELF_REPO_PATH = "tools/route2_release/create_vqm24_observable_training_dense_v2.py"
DENSE_BUILDER_REPO_PATH = "maple/solvation/reference/exterior_probe_partition.py"
ARTIFACT = "route2-vqm24-observable-training-batch-dense-prereg-v2"
PARENT_ARTIFACT = "route2-vqm24-observable-training-batch-prereg-v1"
PARENT_STATUS = "locked-before-observable-training-qm-execution"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


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


def create(args: argparse.Namespace) -> dict[str, Any]:
    from maple.function.calculator.extra_correction.implicit.route2_nonuniform_response import (
        select_farthest_exterior_point_charge_modes,
    )

    parent_path = args.parent.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    parent = json.loads(parent_path.read_text())
    if (
        parent.get("artifact") != PARENT_ARTIFACT
        or parent.get("status") != PARENT_STATUS
    ):
        raise ValueError("Sparse observable-training parent is invalid.")
    runner = SOURCE_ROOT / observable.SELF_REPO_PATH
    dense_builder = SOURCE_ROOT / DENSE_BUILDER_REPO_PATH
    records: list[dict[str, Any]] = []
    for parent_record in parent["records"]:
        for name in ("observable_json", "observable_npz"):
            if Path(parent_record["outputs"][name]).exists():
                raise RuntimeError(
                    "Sparse observable execution started before dense-v2 freeze."
                )
        sparse_surface_path = Path(parent_record["inputs"]["surface"]["path"])
        if _sha256(sparse_surface_path) != parent_record["inputs"]["surface"]["sha256"]:
            raise RuntimeError("Sparse parent surface input drifted.")
        with np.load(sparse_surface_path, allow_pickle=False) as state:
            positions = np.asarray(state["atom_positions_angstrom"], dtype=np.float64)
            radii = np.asarray(state["ase_vdw_radii_angstrom"], dtype=np.float64)
            atomic_numbers = np.asarray(state["atomic_numbers"], dtype=np.int64)
        probes = build_dense_exterior_probe_partition(
            positions,
            radii,
            clearance_angstrom=float(parent["probe_protocol"]["clearance_angstrom"]),
        )
        fit_mask = probes.mask("fit")
        audit_mask = probes.mask("audit")
        source_dimension = 8 * len(positions)
        if (
            int(np.count_nonzero(fit_mask)) <= source_dimension
            or int(np.count_nonzero(audit_mask)) <= source_dimension
        ):
            raise RuntimeError(
                "Dense fit/audit probes must each exceed the 8N source dimension."
            )
        fit_global_indices = np.flatnonzero(fit_mask)
        modes = select_farthest_exterior_point_charge_modes(
            probes.surface_points_bohr[fit_mask],
            positions,
            mode_count=int(parent["probe_protocol"]["mode_count"]),
        )
        source_global_indices = fit_global_indices[modes.source_surface_indices]
        record_root = sparse_surface_path.parent
        dense_surface_path = record_root / "dense-surface-v2.npz"
        dense_modes_path = record_root / "dense-localized-modes-v2.npz"
        if dense_surface_path.exists() or dense_modes_path.exists():
            raise FileExistsError(
                dense_surface_path if dense_surface_path.exists() else dense_modes_path
            )
        np.savez(
            dense_surface_path,
            surface_points_bohr=probes.surface_points_bohr,
            quadrature_weights=probes.quadrature_weights,
            parent_atom_indices=probes.parent_atom_indices,
            partition_indices=probes.partition_indices,
            partition_names=np.asarray(DENSE_EXTERIOR_PARTITION_NAMES),
            atomic_numbers=atomic_numbers,
            atom_positions_angstrom=positions,
            ase_vdw_radii_angstrom=radii,
            lebedev_order=np.asarray(DENSE_EXTERIOR_LEBEDEV_ORDER),
            lebedev_points_per_frame=np.asarray(DENSE_EXTERIOR_LEBEDEV_POINTS),
            audit_rotation=DENSE_EXTERIOR_AUDIT_ROTATION,
        )
        np.savez(
            dense_modes_path,
            source_points_bohr=probes.surface_points_bohr[source_global_indices],
            source_surface_indices=source_global_indices,
            source_partition_indices=probes.partition_indices[source_global_indices],
        )
        for path in (dense_surface_path, dense_modes_path):
            path.chmod(0o444)
        records.append(
            {
                "record_id": parent_record["record_id"],
                "selection_record": parent_record["selection_record"],
                "inputs": {
                    "mol2": parent_record["inputs"]["mol2"],
                    "surface": {
                        "path": str(dense_surface_path),
                        "sha256": _sha256(dense_surface_path),
                        "point_count": int(len(probes.surface_points_bohr)),
                        "fit_point_count": int(np.count_nonzero(fit_mask)),
                        "audit_point_count": int(np.count_nonzero(audit_mask)),
                        "surface_points_sha256": _sha256_array(
                            probes.surface_points_bohr
                        ),
                    },
                    "modes": {
                        "path": str(dense_modes_path),
                        "sha256": _sha256(dense_modes_path),
                        "source_points_sha256": _sha256_array(
                            probes.surface_points_bohr[source_global_indices]
                        ),
                    },
                },
                "outputs": {
                    "gas_directory": parent_record["outputs"]["gas_directory"],
                    "observable_json": str(
                        record_root / "observable-dense-v2/result.json"
                    ),
                    "observable_npz": str(
                        record_root / "observable-dense-v2/result.npz"
                    ),
                },
            }
        )

    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 2,
        "status": "locked-before-dense-observable-qm-execution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent": {
            "path": str(parent_path.relative_to(SOURCE_ROOT)),
            "file_sha256": _sha256(parent_path),
            "preregistration_sha256": parent["preregistration_sha256"],
            "disposition": (
                "gas execution retained; sparse observable surface superseded "
                "before any observable record execution"
            ),
        },
        "selection": parent["selection"],
        "records": records,
        "source": {
            "creator_path": SELF_REPO_PATH,
            "creator_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "dense_builder_path": DENSE_BUILDER_REPO_PATH,
            "dense_builder_sha256": _sha256(dense_builder),
            "observable_runner_path": observable.SELF_REPO_PATH,
            "observable_runner_sha256": _sha256(runner),
            "shared_finite_field_runner_path": observable.SHARED_RUNNER_REPO_PATH,
            "shared_finite_field_runner_sha256": _sha256(
                SOURCE_ROOT / observable.SHARED_RUNNER_REPO_PATH
            ),
            "gas_runner": parent["source"]["gas_runner"],
        },
        "probe_protocol": {
            "atomic_radius": parent["probe_protocol"]["atomic_radius"],
            "clearance_angstrom": parent["probe_protocol"]["clearance_angstrom"],
            "fit_angular_grid": "PySCF 50-point Lebedev order 11",
            "audit_angular_grid": (
                "same 50-point rule under frozen quaternion (1,2,3,4) rotation"
            ),
            "fit_and_audit_each_exceed_8N_source_dimension": True,
            "field_step_e": parent["probe_protocol"]["field_step_e"],
            "mode_count": parent["probe_protocol"]["mode_count"],
            "source_modes_selected_only_from_fit_partition": True,
            "cavity_or_solvent_used": False,
            "sampling_is_not_a_continuum_operator": True,
        },
        "qm_protocol": parent["qm_protocol"],
        "target_contract": {
            **parent["target_contract"],
            "dense_mep_fit_partition": True,
            "dense_mep_audit_partition": True,
            "sparse_26_direction_mep_target": False,
        },
        "claim_boundary": parent["claim_boundary"],
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
    print(json.dumps(create(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
