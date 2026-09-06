#!/usr/bin/env python3
"""Freeze one selected VQM24 localized-field QM input record."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from ase.data import vdw_radii
import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
SELF_REPO_PATH = "tools/route2_release/prepare_vqm24_nonuniform_qm_record.py"
ARTIFACT = "route2-vqm24-localized-field-qm-record-prereg-v1"

from tools.route2_release import prepare_vqm24_nonuniform_qm_gate0 as base  # noqa: E402


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
        handle.write(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    path.chmod(0o444)


def create(args: argparse.Namespace) -> dict[str, Any]:
    import ase
    import ase.data

    from maple.function.calculator.extra_correction.implicit.route2_nonuniform_response import (
        select_farthest_exterior_point_charge_modes,
    )
    from maple.function.calculator.extra_correction.implicit.route2_static_surface_mep import (
        build_route2_smd_exterior_probe_surface,
    )

    selection_path = args.selection.expanduser().resolve(strict=True)
    selection = json.loads(selection_path.read_text())
    record = next(
        item
        for item in selection["records"]
        if item["record_sha256"] == args.record_sha256
    )
    output_dir = args.output_dir.expanduser().resolve()
    preregistration = args.preregistration.expanduser().resolve()
    source_snapshot = args.source_snapshot.expanduser().resolve(strict=True)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if preregistration.exists():
        raise FileExistsError(preregistration)
    if subprocess.check_output(
        ["git", "-C", str(source_snapshot), "status", "--porcelain"],
        text=True,
    ).strip():
        raise RuntimeError("QM source snapshot must be clean.")
    source_head = subprocess.check_output(
        ["git", "-C", str(source_snapshot), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    output_dir.mkdir(parents=True)
    numbers = np.asarray(record["atomic_numbers"], dtype=np.int64)
    positions = np.asarray(record["positions_angstrom"], dtype=np.float64)
    radii_angstrom = np.asarray([vdw_radii[int(number)] for number in numbers])
    if not np.all(np.isfinite(radii_angstrom)) or np.any(radii_angstrom <= 0.0):
        raise RuntimeError("ASE van der Waals radius is unavailable for this record.")
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
    mol2 = output_dir / "geometry.mol2"
    surface_path = output_dir / "surface.npz"
    modes_path = output_dir / "localized-modes.npz"
    base._write_mol2(
        mol2,
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
    gas_runner = source_snapshot / (
        "docs/implicit-solvation/benchmarks/run_route2_qm_gas_checkpoint.py"
    )
    response_runner = source_snapshot / (
        "docs/implicit-solvation/benchmarks/"
        "route2_qm_localized_point_charge_response.py"
    )
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-record-qm-execution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "path": str(selection_path),
            "sha256": _sha256(selection_path),
            "selection_sha256": selection["selection_sha256"],
            "record": record,
        },
        "input_files": {
            "mol2": {"path": str(mol2), "sha256": _sha256(mol2)},
            "surface": {"path": str(surface_path), "sha256": _sha256(surface_path)},
            "modes": {"path": str(modes_path), "sha256": _sha256(modes_path)},
            "ase_vdw_radius_source": {
                "path": str(Path(ase.data.__file__).resolve(strict=True)),
                "sha256": _sha256(Path(ase.data.__file__).resolve(strict=True)),
                "ase_version": ase.__version__,
            },
        },
        "source": {
            "creator_path": SELF_REPO_PATH,
            "creator_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "shared_gate0_creator_path": base.SELF_REPO_PATH,
            "shared_gate0_creator_sha256": _sha256(SOURCE_ROOT / base.SELF_REPO_PATH),
            "clean_snapshot": str(source_snapshot),
            "clean_snapshot_git_head": source_head,
            "gas_runner": {"path": str(gas_runner), "sha256": _sha256(gas_runner)},
            "response_runner": {
                "path": str(response_runner),
                "sha256": _sha256(response_runner),
            },
        },
        "probe_protocol": {
            "atomic_radius": "ASE standard van der Waals radii",
            "clearance_angstrom": base.SURFACE_CLEARANCE_ANGSTROM,
            "candidate_directions": 26,
            "mode_count": base.MODE_COUNT,
            "field_steps_e": list(base.FIELD_STEPS_E),
            "cavity_or_solvent_used": False,
        },
        "qm_protocol": {
            "method": "omegaB97M-V",
            "basis": "def2-tzvpd",
            "reference": "RKS density fitting",
            "semilocal_grid_level": 3,
            "nonlocal_grid": [50, 194],
            "nonlocal_prune": "sg1_prune",
            "threads": 8,
            "maximum_memory_mb": 8000,
        },
        "outputs": {
            "gas_directory": str(output_dir / "gas"),
            "response_directory": str(output_dir / "qm-response"),
        },
        "claim_boundary": {
            "vqm24_energy_target_used": False,
            "experimental_solvation_target_read": False,
            "pcm_or_cavity_label_used": False,
            "fit_or_training_performed": False,
            "capability_admitted": False,
        },
    }
    payload["preregistration_sha256"] = _canonical_sha256(payload)
    _write_json_exclusive(preregistration, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--record-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
