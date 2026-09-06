#!/usr/bin/env python3
"""Generate target-independent 12-mode radial/l2-balanced response inputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

from ase.units import Hartree
import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maple.solvation.coupling.exact_gto import (  # noqa: E402
    FixedSurfaceGeometry,
    MACEPolarRadialGTOCoupling,
)
from maple.solvation.coupling.gaussian_quadrupole import (  # noqa: E402
    gaussian_traceless_quadrupole_surface_operator,
)
from maple.solvation.reference.sector_balanced_response_modes import (  # noqa: E402
    select_sector_balanced_response_modes,
)


SELF_REPO_PATH = (
    "tools/route2_release/prepare_vqm24_sector_balanced_response_modes.py"
)
ARTIFACT = "route2-vqm24-sector-balanced-p13-response-mode-batch-v1"
PREREGISTRATION_ARTIFACT = (
    "route2-vqm24-sector-balanced-p13-response-mode-prereg-v1"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(
        f"{array.dtype.str}:{array.shape}".encode()
        + b"\0"
        + array.tobytes(order="C")
    ).hexdigest()


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


def run(args: argparse.Namespace) -> dict[str, object]:
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    output_directory = args.output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    preregistration = json.loads(preregistration_path.read_text())
    if (
        preregistration.get("artifact") != PREREGISTRATION_ARTIFACT
        or preregistration.get("status") != "locked-before-mode-generation"
    ):
        raise ValueError("Sector-balanced mode preregistration is invalid.")
    if _sha256(SOURCE_ROOT / SELF_REPO_PATH) != preregistration["source"][
        "runner_sha256"
    ]:
        raise RuntimeError("Mode-generation runner changed after freeze.")
    selector = preregistration["source"]["selector"]
    if _sha256(SOURCE_ROOT / selector["path"]) != selector["sha256"]:
        raise RuntimeError("Sector-balanced selector changed after freeze.")
    parent_path = SOURCE_ROOT / preregistration["parent"]["path"]
    if _sha256(parent_path) != preregistration["parent"]["file_sha256"]:
        raise RuntimeError("Dense observable parent changed after mode freeze.")
    parent = json.loads(parent_path.read_text())
    mode_count = int(preregistration["protocol"]["mode_count"])
    output_directory.mkdir(parents=True, exist_ok=False)
    records = []
    for record in parent["records"]:
        record_id = record["record_id"]
        surface_path = Path(record["inputs"]["surface"]["path"])
        prefix_path = Path(record["inputs"]["modes"]["path"])
        if (
            _sha256(surface_path) != record["inputs"]["surface"]["sha256"]
            or _sha256(prefix_path) != record["inputs"]["modes"]["sha256"]
        ):
            raise RuntimeError(f"Frozen surface/mode input drifted for {record_id}.")
        with np.load(surface_path, allow_pickle=False) as surface:
            points = np.asarray(surface["surface_points_bohr"], dtype=np.float64)
            positions = np.asarray(
                surface["atom_positions_angstrom"], dtype=np.float64
            )
            partitions = np.asarray(surface["partition_indices"], dtype=np.int64)
        with np.load(prefix_path, allow_pickle=False) as prefix_state:
            prefix = np.asarray(
                prefix_state["source_surface_indices"], dtype=np.int64
            )
        geometry = FixedSurfaceGeometry(positions, points)
        radial = MACEPolarRadialGTOCoupling().surface_operator(geometry) / Hartree
        l2 = gaussian_traceless_quadrupole_surface_operator(
            points_bohr=points,
            centers_angstrom=positions,
            sigma_angstrom=float(preregistration["protocol"]["l2_sigma_angstrom"]),
        )
        selection = select_sector_balanced_response_modes(
            radial_field_operator=radial,
            l2_field_operator=l2,
            fit_surface_indices=np.flatnonzero(partitions == 0),
            frozen_prefix_indices=prefix,
            mode_count=mode_count,
            relative_rank_tolerance=float(
                preregistration["protocol"]["relative_rank_tolerance"]
            ),
        )
        metrics = asdict(selection)
        gates = {
            "rank": selection.balanced_rank == mode_count,
            "condition": selection.balanced_condition_number
            <= float(preregistration["gates"]["condition_number_maximum"]),
            "greedy_residual": selection.minimum_greedy_residual_norm
            >= float(preregistration["gates"]["minimum_greedy_residual_norm"]),
            "radial_fraction_lower": selection.radial_fraction_of_selected_balanced_norm
            >= float(preregistration["gates"]["radial_fraction_minimum"]),
            "radial_fraction_upper": selection.radial_fraction_of_selected_balanced_norm
            <= float(preregistration["gates"]["radial_fraction_maximum"]),
            "prefix_preserved": tuple(selection.source_surface_indices[: len(prefix)])
            == tuple(int(value) for value in prefix),
            "target_free": selection.target_used is False,
        }
        directory = output_directory / record_id
        directory.mkdir()
        modes_path = directory / "sector-balanced-modes.npz"
        selected = np.asarray(selection.source_surface_indices, dtype=np.int64)
        np.savez(
            modes_path,
            source_points_bohr=points[selected],
            source_surface_indices=selected,
            source_partition_indices=partitions[selected],
            frozen_prefix_count=np.asarray(selection.frozen_prefix_count),
            radial_gauge_reduced_rms=np.asarray(
                selection.radial_gauge_reduced_rms
            ),
            l2_rms=np.asarray(selection.l2_rms),
            balanced_singular_values=np.asarray(
                selection.balanced_singular_values, dtype=np.float64
            ),
        )
        modes_path.chmod(0o444)
        records.append(
            {
                "record_id": record_id,
                "status": "pass" if all(gates.values()) else "fail",
                "metrics": metrics,
                "gates": gates,
                "surface_path": str(surface_path),
                "surface_sha256": _sha256(surface_path),
                "modes_path": str(modes_path),
                "modes_sha256": _sha256(modes_path),
                "source_points_sha256": _array_sha256(points[selected]),
                "selection_record_sha256": record["selection_record"][
                    "record_sha256"
                ],
            }
        )
    records.sort(key=lambda item: item["record_id"])
    all_pass = all(record["status"] == "pass" for record in records)
    payload: dict[str, object] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "pass-target-independent-mode-generation" if all_pass else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "runner_path": SELF_REPO_PATH,
            "runner_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "preregistration_file_sha256": _sha256(preregistration_path),
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "parent_file_sha256": _sha256(parent_path),
        },
        "records": records,
        "aggregate": {
            "record_count": len(records),
            "pass_count": sum(record["status"] == "pass" for record in records),
            "maximum_condition_number": max(
                record["metrics"]["balanced_condition_number"]
                for record in records
            ),
            "minimum_greedy_residual_norm": min(
                record["metrics"]["minimum_greedy_residual_norm"]
                for record in records
            ),
            "minimum_radial_fraction": min(
                record["metrics"]["radial_fraction_of_selected_balanced_norm"]
                for record in records
            ),
            "maximum_radial_fraction": max(
                record["metrics"]["radial_fraction_of_selected_balanced_norm"]
                for record in records
            ),
        },
        "claim_boundary": {
            "target_independent_input_modes_generated": True,
            "qm_response_or_energy_read": False,
            "model_prediction_read": False,
            "pcm_or_cavity_used": False,
            "experimental_solvation_target_read": False,
            "response_data_generated": False,
            "model_fit_or_selection_performed": False,
            "validation_or_blind_formula_opened": False,
            "maple_capability_admitted": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    shutil.copy2(preregistration_path, output_directory / "preregistration.json")
    result_path = output_directory / "manifest.json"
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
