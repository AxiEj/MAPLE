#!/usr/bin/env python3
"""Freeze the VQM24 train split for observable-supervised scalar learning."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import ase
import ase.data
from ase.data import vdw_radii
import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tools.route2_release import prepare_vqm24_nonuniform_qm_gate0 as base  # noqa: E402
from tools.route2_release import run_vqm24_observable_training_record as observable  # noqa: E402


SELF_REPO_PATH = "tools/route2_release/prepare_vqm24_observable_training_batch.py"
ARTIFACT = "route2-vqm24-observable-training-batch-prereg-v1"
SELECTION_ARTIFACT = "route2-vqm24-nonuniform-qm-geometry-selection-v1"
SELECTION_STATUS = "locked-before-any-maple-external-field-qm-calculation"


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


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.chmod(0o444)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    from maple.function.calculator.extra_correction.implicit.route2_nonuniform_response import (
        select_farthest_exterior_point_charge_modes,
    )
    from maple.function.calculator.extra_correction.implicit.route2_static_surface_mep import (
        build_route2_smd_exterior_probe_surface,
    )

    selection_path = args.selection.expanduser().resolve(strict=True)
    selection = json.loads(selection_path.read_text())
    if (
        selection.get("artifact") != SELECTION_ARTIFACT
        or selection.get("status") != SELECTION_STATUS
    ):
        raise ValueError("Frozen VQM24 geometry selection is invalid.")
    records = [record for record in selection["records"] if record["split"] == "train"]
    if len(records) != 32:
        raise RuntimeError("Observable batch requires the frozen 32-record train split.")
    if len({record["chemical_formula_hill"] for record in records}) != len(records):
        raise RuntimeError("Observable train formulas are not unique.")

    output_root = args.output_root.expanduser().resolve()
    preregistration_path = args.preregistration.expanduser().resolve()
    source_snapshot = args.source_snapshot.expanduser().resolve(strict=True)
    if output_root.exists():
        raise FileExistsError(output_root)
    if preregistration_path.exists():
        raise FileExistsError(preregistration_path)
    if subprocess.check_output(
        ["git", "-C", str(source_snapshot), "status", "--porcelain"],
        text=True,
    ).strip():
        raise RuntimeError("QM gas-runner source snapshot must be clean.")
    source_head = subprocess.check_output(
        ["git", "-C", str(source_snapshot), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    gas_runner = source_snapshot / (
        "docs/implicit-solvation/benchmarks/run_route2_qm_gas_checkpoint.py"
    )
    observable_runner = SOURCE_ROOT / observable.SELF_REPO_PATH
    shared_response_runner = SOURCE_ROOT / observable.SHARED_RUNNER_REPO_PATH

    output_root.mkdir(parents=True)
    frozen_records: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item["record_sha256"]):
        record_id = (
            f"{record['stratum']}-{int(record['dataset_index'])}-"
            f"{record['record_sha256'][:12]}"
        )
        record_root = output_root / record_id
        record_root.mkdir()
        numbers = np.asarray(record["atomic_numbers"], dtype=np.int64)
        positions = np.asarray(record["positions_angstrom"], dtype=np.float64)
        radii_angstrom = np.asarray(
            [vdw_radii[int(number)] for number in numbers],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(radii_angstrom)) or np.any(radii_angstrom <= 0.0):
            raise RuntimeError(f"ASE van der Waals radius is unavailable for {record_id}.")
        surface = build_route2_smd_exterior_probe_surface(
            positions,
            radii_angstrom,
            clearance_angstrom=base.SURFACE_CLEARANCE_ANGSTROM,
        )
        modes = select_farthest_exterior_point_charge_modes(
            surface.surface_points_bohr,
            positions,
            mode_count=base.MODE_COUNT,
        )
        mol2_path = record_root / "geometry.mol2"
        surface_path = record_root / "surface.npz"
        modes_path = record_root / "localized-modes.npz"
        base._write_mol2(
            mol2_path,
            numbers=numbers,
            positions=positions,
            molecule_name=str(record["names"][0]),
        )
        np.savez(
            surface_path,
            surface_points_bohr=surface.surface_points_bohr,
            quadrature_weights=surface.quadrature_weights,
            parent_atom_indices=surface.parent_atom_indices,
            atomic_numbers=numbers,
            atom_positions_angstrom=positions,
            ase_vdw_radii_angstrom=radii_angstrom,
        )
        np.savez(
            modes_path,
            source_points_bohr=modes.source_points_bohr,
            source_surface_indices=modes.source_surface_indices,
        )
        for path in (surface_path, modes_path):
            path.chmod(0o444)
        frozen_records.append(
            {
                "record_id": record_id,
                "selection_record": record,
                "inputs": {
                    "mol2": {"path": str(mol2_path), "sha256": _sha256(mol2_path)},
                    "surface": {
                        "path": str(surface_path),
                        "sha256": _sha256(surface_path),
                    },
                    "modes": {"path": str(modes_path), "sha256": _sha256(modes_path)},
                },
                "outputs": {
                    "gas_directory": str(record_root / "gas"),
                    "observable_json": str(record_root / "observable/result.json"),
                    "observable_npz": str(record_root / "observable/result.npz"),
                },
            }
        )

    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-observable-training-qm-execution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "path": str(selection_path),
            "file_sha256": _sha256(selection_path),
            "selection_sha256": selection["selection_sha256"],
            "split": "train",
            "record_count": len(frozen_records),
        },
        "records": frozen_records,
        "source": {
            "creator_path": SELF_REPO_PATH,
            "creator_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "surface_builder_path": (
                "maple/function/calculator/extra_correction/implicit/"
                "route2_static_surface_mep.py"
            ),
            "surface_builder_sha256": _sha256(
                SOURCE_ROOT
                / "maple/function/calculator/extra_correction/implicit/"
                "route2_static_surface_mep.py"
            ),
            "mode_selector_path": (
                "maple/function/calculator/extra_correction/implicit/"
                "route2_nonuniform_response.py"
            ),
            "mode_selector_sha256": _sha256(
                SOURCE_ROOT
                / "maple/function/calculator/extra_correction/implicit/"
                "route2_nonuniform_response.py"
            ),
            "gas_runner": {"path": str(gas_runner), "sha256": _sha256(gas_runner)},
            "observable_runner": {
                "path": observable.SELF_REPO_PATH,
                "sha256": _sha256(observable_runner),
            },
            "shared_finite_field_runner": {
                "path": observable.SHARED_RUNNER_REPO_PATH,
                "sha256": _sha256(shared_response_runner),
            },
            "clean_gas_source_snapshot": str(source_snapshot),
            "clean_gas_source_git_head": source_head,
            "ase_vdw_radius_source": {
                "path": str(Path(ase.data.__file__).resolve(strict=True)),
                "sha256": _sha256(Path(ase.data.__file__).resolve(strict=True)),
                "ase_version": ase.__version__,
            },
        },
        "probe_protocol": {
            "atomic_radius": "ASE standard van der Waals radii",
            "clearance_angstrom": base.SURFACE_CLEARANCE_ANGSTROM,
            "candidate_directions": 26,
            "mode_count": base.MODE_COUNT,
            "field_step_e": observable.FIELD_STEP_E,
            "signs": [-1, 1],
            "cavity_or_solvent_used": False,
        },
        "qm_protocol": {
            "method": "omegaB97M-V",
            "basis": "def2-tzvpd",
            "reference": "RKS density fitting",
            "semilocal_grid_level": 3,
            "nonlocal_grid_profile": "PySCF level 3",
            "threads": 8,
            "maximum_memory_mb": 8000,
        },
        "target_contract": {
            "zero_field_energy": True,
            "zero_field_full_molecular_mep": True,
            "zero_field_molecular_dipole": True,
            "perturbed_total_external_enthalpy": True,
            "external_nuclear_coupling_included": True,
            "perturbed_full_molecular_mep": True,
            "perturbed_molecular_dipole": True,
            "density_or_partition_coefficient": False,
            "numerical_energy_curvature": False,
            "pcm_or_cavity_quantity": False,
            "experimental_solvation_quantity": False,
            "vqm24_energy_or_atomization_quantity": False,
        },
        "claim_boundary": {
            "vqm24_energy_target_used": False,
            "experimental_solvation_target_read": False,
            "pcm_or_cavity_label_used": False,
            "fit_or_training_performed": False,
            "model_accuracy_measured": False,
            "capability_admitted": False,
        },
    }
    payload["preregistration_sha256"] = _canonical_sha256(payload)
    _write_json_exclusive(preregistration_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
