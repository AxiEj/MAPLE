#!/usr/bin/env python3
"""Freeze the first VQM24 localized-field QM Gate-0 inputs before execution."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from ase.data import chemical_symbols
from ase.units import Bohr
import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
SELF_REPO_PATH = "tools/route2_release/prepare_vqm24_nonuniform_qm_gate0.py"
ARTIFACT = "route2-vqm24-localized-field-qm-gate0-prereg-v1"
SELECTION = SOURCE_ROOT / (
    "docs/route2/preregistrations/"
    "vqm24-nonuniform-qm-geometry-selection-v1.json"
)
ATOMIC_TABLE = SOURCE_ROOT / (
    "docs/implicit-solvation/benchmarks/"
    "route2-rhodrop-atomic-reference-gaussian-mixture-v1.npz"
)
ATOMIC_MANIFEST = SOURCE_ROOT / (
    "docs/implicit-solvation/benchmarks/"
    "route2-rhodrop-atomic-reference-gaussian-mixture-v1.json"
)
PROMOLECULAR_ISOVALUE_E_PER_BOHR3 = 1.0e-4
SURFACE_CLEARANCE_ANGSTROM = 1.0
MODE_COUNT = 4
FIELD_STEPS_E = (3.0e-4, 1.0e-3)


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


def _atomic_isodensity_radius_bohr(mixture: Any) -> float:
    counts = np.asarray(mixture.electron_counts, dtype=np.float64)
    exponents = np.asarray(mixture.gaussian_exponents_bohr2, dtype=np.float64)

    def density(radius: float) -> float:
        return float(
            np.sum(
                counts
                * (exponents / np.pi) ** 1.5
                * np.exp(-exponents * radius * radius)
            )
        )

    lower = 0.0
    upper = 2.0
    while density(upper) > PROMOLECULAR_ISOVALUE_E_PER_BOHR3:
        upper *= 2.0
        if upper > 128.0:
            raise RuntimeError("atomic reference density did not reach its isovalue.")
    for _iteration in range(100):
        midpoint = 0.5 * (lower + upper)
        if density(midpoint) > PROMOLECULAR_ISOVALUE_E_PER_BOHR3:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _write_mol2(
    path: Path,
    *,
    numbers: np.ndarray,
    positions: np.ndarray,
    molecule_name: str,
) -> None:
    lines = [
        "@<TRIPOS>MOLECULE",
        molecule_name,
        f"{len(numbers)} 0 0 0 0",
        "SMALL",
        "NO_CHARGES",
        "",
        "@<TRIPOS>ATOM",
    ]
    for index, (number, position) in enumerate(
        zip(numbers, positions, strict=True),
        start=1,
    ):
        symbol = chemical_symbols[int(number)]
        atom_type = symbol if symbol in {"Br", "Cl"} else f"{symbol}.3"
        lines.append(
            f"{index:7d} {symbol}{index:<4d} "
            f"{position[0]: .10f} {position[1]: .10f} {position[2]: .10f} "
            f"{atom_type:<6s} 1 MOL 0.000000"
        )
    lines.extend(("@<TRIPOS>BOND", ""))
    path.write_text("\n".join(lines))
    path.chmod(0o444)


def create(args: argparse.Namespace) -> dict[str, Any]:
    from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
        load_atomic_reference_density_asset,
    )
    from maple.function.calculator.extra_correction.implicit.route2_nonuniform_response import (
        select_farthest_exterior_point_charge_modes,
    )
    from maple.function.calculator.extra_correction.implicit.route2_static_surface_mep import (
        build_route2_smd_exterior_probe_surface,
    )

    selection = json.loads(SELECTION.read_text())
    record = next(item for item in selection["records"] if item["split"] == "train")
    output_dir = args.output_dir.expanduser().resolve()
    preregistration = args.preregistration.expanduser().resolve()
    source_snapshot = args.source_snapshot.expanduser().resolve(strict=True)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if preregistration.exists():
        raise FileExistsError(preregistration)
    output_dir.mkdir(parents=True)
    numbers = np.asarray(record["atomic_numbers"], dtype=np.int64)
    positions = np.asarray(record["positions_angstrom"], dtype=np.float64)
    asset = load_atomic_reference_density_asset(
        table_path=ATOMIC_TABLE,
        manifest_path=ATOMIC_MANIFEST,
    )
    radii_bohr = np.asarray(
        [
            _atomic_isodensity_radius_bohr(asset.mixture(int(number)))
            for number in numbers
        ]
    )
    surface = build_route2_smd_exterior_probe_surface(
        positions,
        radii_bohr * Bohr,
        clearance_angstrom=SURFACE_CLEARANCE_ANGSTROM,
    )
    modes = select_farthest_exterior_point_charge_modes(
        surface.surface_points_bohr,
        positions,
        mode_count=MODE_COUNT,
    )
    molecule_name = str(record["names"][0])
    mol2 = output_dir / "geometry.mol2"
    surface_path = output_dir / "surface.npz"
    modes_path = output_dir / "localized-modes.npz"
    _write_mol2(
        mol2,
        numbers=numbers,
        positions=positions,
        molecule_name=molecule_name,
    )
    np.savez(
        surface_path,
        surface_points_bohr=surface.surface_points_bohr,
        quadrature_weights=surface.quadrature_weights,
        parent_atom_indices=surface.parent_atom_indices,
        atomic_numbers=numbers,
        atom_positions_angstrom=positions,
        promolecular_isodensity_radii_bohr=radii_bohr,
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
    response_output_dir = output_dir / "qm-response"
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-first-vqm24-qm-gate0-execution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "path": str(SELECTION),
            "sha256": _sha256(SELECTION),
            "selection_sha256": selection["selection_sha256"],
            "record": record,
        },
        "input_files": {
            "mol2": {"path": str(mol2), "sha256": _sha256(mol2)},
            "surface": {
                "path": str(surface_path),
                "sha256": _sha256(surface_path),
            },
            "modes": {"path": str(modes_path), "sha256": _sha256(modes_path)},
            "atomic_reference_table": {
                "path": str(ATOMIC_TABLE),
                "sha256": _sha256(ATOMIC_TABLE),
            },
            "atomic_reference_manifest": {
                "path": str(ATOMIC_MANIFEST),
                "sha256": _sha256(ATOMIC_MANIFEST),
            },
        },
        "source": {
            "creator_path": SELF_REPO_PATH,
            "creator_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "clean_snapshot": str(source_snapshot),
            "clean_snapshot_git_head": "0e64f187748bd3aabab9eb2797a7f7affab47c8b",
            "gas_runner": {"path": str(gas_runner), "sha256": _sha256(gas_runner)},
            "response_runner": {
                "path": str(response_runner),
                "sha256": _sha256(response_runner),
            },
        },
        "probe_protocol": {
            "atomic_radius": (
                "isolated frozen atomic-reference density radius at "
                f"{PROMOLECULAR_ISOVALUE_E_PER_BOHR3} e/bohr^3"
            ),
            "clearance_angstrom": SURFACE_CLEARANCE_ANGSTROM,
            "candidate_directions": 26,
            "mode_count": MODE_COUNT,
            "mode_selection": "geometry-only farthest exterior points",
            "field_steps_e": list(FIELD_STEPS_E),
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
            "response_directory": str(response_output_dir),
        },
        "claim_boundary": {
            "experimental_solvation_target_read": False,
            "vqm24_energy_target_used": False,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
