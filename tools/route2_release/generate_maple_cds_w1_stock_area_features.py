#!/usr/bin/env python3
"""Generate target-blind aqueous SMD stock-area design features.

This is an explicitly energy-only diagnostic surface.  It reconstructs the
published 18-column aqueous SMD tension design on PySCF's highest available
Lebedev/SWIG SASA grid and compares the stock-coefficient contraction with the
compiled legacy SMD CDS energy.  The resulting rows are suitable for asking a
bounded statistical question about the frozen SMD tension family; they are not
the final smooth, structurally SO(3)-equivariant CDS force model.

The licensed MNSol loader parses the complete frozen table to recover and
validate geometries.  Target attributes and hybrid prediction records are not
accessed by feature computation and are never emitted.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
from ase.data import chemical_symbols

_SCRIPT_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SCRIPT_SOURCE_ROOT))

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2,
    SMD_WATER_TENSION_PARAMETER_NAMES,
    aqueous_cds_tension_design_row,
)
from maple.solvation.release.evidence import (
    collect_loaded_repository_sources,
    runtime_record,
)

ARTIFACT = "route2-maple-cds-w1-water-stock-area-feature-record-v1"
PREREGISTRATION_ARTIFACT = "route2-maple-cds-w1-water-stock-area-features-prereg-v1"
EXPECTED_WATER_RECORD_COUNT = 306
PYSCF_VERSION = "2.13.1"
SASA_LEBEDEV_ORDER = 131
SASA_GRID_POINTS_PER_ATOM = 5810
SASA_PROBE_RADIUS_ANGSTROM = 0.4
STOCK_ENERGY_PARITY_TOLERANCE_KCAL_MOL = 0.05
ROTATION_CONTROL_TOLERANCE_KCAL_MOL = 0.05
ROTATION_CONTROL_MATRIX = (
    (0.36, -0.48, 0.8),
    (0.8, 0.6, 0.0),
    (-0.48, 0.64, 0.6),
)
INPUT_FILE_NAMES = (
    "MNSolDatabase_v2012.zip",
    "route2-mnsol-development-selection-v1.private.json",
    "route2-mnsol-pilot-selection-v1.json",
    "route2-mnsol-protocol-v1.json",
)
REQUIRED_SOURCE_FILE_NAMES = (
    "GOAL.md",
    "docs/implicit-solvation/benchmarks/benchmark_core.py",
    "docs/implicit-solvation/benchmarks/mnsol_dataset.py",
    "docs/implicit-solvation/benchmarks/mnsol_partition.py",
    "docs/implicit-solvation/benchmarks/mnsol_pilot.py",
    "docs/route2/MAPLE_CDS_W1.md",
    "maple/function/calculator/extra_correction/implicit/smd_cds.py",
    "maple/solvation/release/evidence.py",
    "tests/route2_vnext/test_maple_cds_w1_stock_area_features.py",
    "tools/route2_release/create_maple_cds_w1_stock_area_preregistration.py",
    "tools/route2_release/generate_maple_cds_w1_stock_area_features.py",
)
PROHIBITED_OUTPUT_KEYS = frozenset(
    {
        "delta_g_kcal_mol",
        "experimental_delta_g_kcal_mol",
        "continuum_polarization_kcal_mol",
        "predicted_delta_g_kcal_mol",
        "target_kcal_mol",
    }
)
PARENT_ARTIFACT = "route2-hybrid-smd-development-prereg-v3"
PARENT_INPUT_HASH_FIELDS = {
    "MNSolDatabase_v2012.zip": "dataset_zip_sha256",
    "route2-mnsol-development-selection-v1.private.json": (
        "development_selection_sha256"
    ),
    "route2-mnsol-pilot-selection-v1.json": "pilot_selection_sha256",
    "route2-mnsol-protocol-v1.json": "protocol_sha256",
}
PREREGISTRATION_STATUS = (
    "locked-before-first-complete-306-stock-area-generation-and-before-any-"
    "target-coupled-analysis"
)


def claim_boundary() -> str:
    return (
        "Energy-only target-blind diagnostic of the frozen aqueous SMD "
        "18-column family under a high-order PySCF SWIG SASA approximation. "
        "It does not admit a force or transfer coefficients to a future "
        "positive-parent smooth-area profile."
    )


def preregistration_schema_keys() -> frozenset[str]:
    return frozenset(
        {
            "artifact",
            "schema_version",
            "status",
            "locked_at_utc",
            "partition",
            "water_record_count",
            "water_identity_sha256",
            "dataset_loader_parses_experimental_targets",
            "experimental_targets_used_by_feature_computation",
            "experimental_targets_emitted",
            "hybrid_prediction_records_read_by_feature_generator",
            "confirmation_selection_manifest_opened",
            "confirmation_records_selected_or_emitted",
            "fitting_or_calibration_permitted",
            "checkpoint_or_method_selection_permitted",
            "geometry_only_unit_canaries_precede_lock",
            "energy_only_diagnostic",
            "force_capability",
            "source_root",
            "source_git_head",
            "source_git_tree",
            "source_files_sha256",
            "feature_generator_sha256",
            "runtime_identity",
            "runtime_identity_sha256",
            "pyscf_runtime_assets",
            "pyscf_runtime_assets_sha256",
            "input_root",
            "input_files_sha256",
            "parent_hybrid_preregistration_path",
            "parent_hybrid_preregistration_sha256",
            "parent_hybrid_source_git_head",
            "feature_output_dir",
            "preregistration_path",
            "area_definition",
            "linear_basis",
            "numerical_choice_rationale",
            "output_contract",
            "claim_boundary",
            "self_sha256",
        }
    )


class StockAreaFeatureError(RuntimeError):
    """Raised when stock-area diagnostic evidence is not reproducible."""


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
        ).encode("utf-8")
    ).hexdigest()


def stock_area_contract() -> dict[str, object]:
    return {
        "contract": "pyscf-smd-experimental-swig-sasa-energy-diagnostic-v1",
        "pyscf_version": PYSCF_VERSION,
        "surface_builder": "pyscf.solvent.pcm.gen_surface",
        "surface_discretization": "SWIG",
        "lebedev_order": SASA_LEBEDEV_ORDER,
        "grid_points_per_atom": SASA_GRID_POINTS_PER_ATOM,
        "radii": "pyscf.data.radii.VDW-plus-0.4-angstrom-probe",
        "basis_used_only_to-initialize-molecule": "sto-3g",
        "stock_energy_parity_tolerance_kcal_mol": (
            STOCK_ENERGY_PARITY_TOLERANCE_KCAL_MOL
        ),
        "rotation_control_tolerance_kcal_mol": (ROTATION_CONTROL_TOLERANCE_KCAL_MOL),
        "rotation_control_matrix": [list(row) for row in ROTATION_CONTROL_MATRIX],
        "laboratory_fixed_grid": True,
        "energy_only_diagnostic": True,
        "force_capability": False,
    }


def linear_basis_contract() -> dict[str, object]:
    coefficients = SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2.tolist()
    return {
        "contract": "published-aqueous-smd-linear-18-column-v1",
        "parameter_names": list(SMD_WATER_TENSION_PARAMETER_NAMES),
        "stock_coefficients_cal_mol_angstrom2": coefficients,
        "stock_coefficients_sha256": _canonical_sha256(coefficients),
    }


def numerical_choice_rationale() -> dict[str, object]:
    return {
        "grid_choice": "highest-Lebedev-grid-shipped-by-bound-PySCF",
        "selected_from_geometry_and-runtime_only": True,
        "target_values_used": False,
        "hybrid_predictions_used": False,
        "accuracy_residuals_used": False,
    }


def output_contract() -> dict[str, object]:
    return {
        "one_exclusive_json_per_water_record": True,
        "coordinates_emitted": False,
        "experimental_targets_emitted": False,
        "hybrid_outputs_emitted": False,
        "fit_or_calibration_emitted": False,
    }


def _runtime_identity() -> tuple[dict[str, object], str]:
    identity = runtime_record()
    identity.pop("generated_at_utc", None)
    return identity, _canonical_sha256(identity)


def _pyscf_runtime_assets() -> tuple[dict[str, object], str]:
    try:
        import pyscf
        from pyscf.data import radii
        from pyscf.dft import gen_grid
        from pyscf.solvent import pcm, smd
    except ImportError as exc:  # pragma: no cover - optional runtime boundary
        raise StockAreaFeatureError("PySCF stock-area runtime is unavailable.") from exc
    if pyscf.__version__ != PYSCF_VERSION:
        raise StockAreaFeatureError(
            f"PySCF {PYSCF_VERSION} is required, got {pyscf.__version__}."
        )
    raw_paths = {
        "pyscf_init_py": getattr(pyscf, "__file__", None),
        "pyscf_pcm_py": getattr(pcm, "__file__", None),
        "pyscf_smd_py": getattr(smd, "__file__", None),
        "pyscf_radii_py": getattr(radii, "__file__", None),
        "pyscf_gen_grid_py": getattr(gen_grid, "__file__", None),
        "pyscf_libsolvent_so": getattr(getattr(smd, "libsolvent", None), "_name", None),
        "pyscf_libdft_so": getattr(getattr(gen_grid, "libdft", None), "_name", None),
    }
    files: dict[str, dict[str, str]] = {}
    for label, raw in raw_paths.items():
        if not isinstance(raw, str) or not raw:
            raise StockAreaFeatureError(f"PySCF asset {label!r} is unavailable.")
        path = Path(raw).expanduser().resolve(strict=True)
        if not path.is_file():
            raise StockAreaFeatureError(f"PySCF asset {label!r} is not a file.")
        files[label] = {"realpath": str(path), "sha256": _sha256(path)}
    assets: dict[str, object] = {
        "contract": "pyscf-smd-stock-area-runtime-assets-v1",
        "pyscf_version": PYSCF_VERSION,
        "files": files,
    }
    return assets, _canonical_sha256(assets)


def _hex_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StockAreaFeatureError(f"{name} must be a lowercase SHA256.")
    return value


def _validate_parent_input_hashes(
    parent: Mapping[str, Any], input_hashes: Mapping[str, Any]
) -> None:
    if set(input_hashes) != set(INPUT_FILE_NAMES):
        raise StockAreaFeatureError("Child frozen input hashes are incomplete.")
    for filename, parent_field in PARENT_INPUT_HASH_FIELDS.items():
        parent_hash = _hex_digest(parent.get(parent_field), name=parent_field)
        child_hash = _hex_digest(input_hashes.get(filename), name=filename)
        if parent_hash != child_hash:
            raise StockAreaFeatureError(
                f"Parent input hash {parent_field!r} disagrees with child input."
            )


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        diagnostic = result.stderr.strip() or result.stdout.strip()
        raise StockAreaFeatureError(f"Git command {arguments!r} failed: {diagnostic}")
    return result.stdout.strip()


def _relative_file(root: Path, raw: object, *, name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise StockAreaFeatureError(f"{name} must be a non-empty relative path.")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise StockAreaFeatureError(f"{name} must stay inside its bound root.")
    resolved = (root / relative).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise StockAreaFeatureError(f"{name} resolves outside its bound root.") from exc
    if not resolved.is_file():
        raise StockAreaFeatureError(f"{name} must resolve to a regular file.")
    return resolved


def _outside_root(path: Path, root: Path, *, name: str) -> None:
    try:
        path.relative_to(root)
    except ValueError:
        return
    raise StockAreaFeatureError(f"{name} must be outside the source checkout.")


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise StockAreaFeatureError(f"{name} must be a JSON object.")
    return payload


def _load_selection(
    *, source_root: Path, input_root: Path
) -> tuple[object, tuple[object, ...]]:
    benchmark_root = source_root / "docs/implicit-solvation/benchmarks"
    module_names = (
        "benchmark_core",
        "mnsol_dataset",
        "mnsol_pilot",
        "mnsol_partition",
    )
    previous_modules = {name: sys.modules.pop(name, None) for name in module_names}
    sys.path.insert(0, str(benchmark_root))
    try:
        dataset_module = importlib.import_module("mnsol_dataset")
        partition_module = importlib.import_module("mnsol_partition")
        for module in (dataset_module, partition_module):
            module_path = Path(module.__file__).resolve(strict=True)
            if module_path.parent != benchmark_root:
                raise StockAreaFeatureError(
                    "MNSol benchmark modules came from another checkout."
                )
        protocol = dataset_module.load_mnsol_protocol(
            input_root / "route2-mnsol-protocol-v1.json"
        )
        dataset = dataset_module.load_mnsol_v2012(
            input_root / "MNSolDatabase_v2012.zip",
            protocol,
        )
        selection = _load_json_object(
            input_root / "route2-mnsol-development-selection-v1.private.json",
            name="development selection",
        )
        pilot = _load_json_object(
            input_root / "route2-mnsol-pilot-selection-v1.json",
            name="pilot selection",
        )
        rows = partition_module.validate_frozen_mnsol_partition_selection(
            selection,
            dataset,
            protocol,
            pilot,
        )
    finally:
        sys.path.pop(0)
        for name in reversed(module_names):
            sys.modules.pop(name, None)
            previous = previous_modules[name]
            if previous is not None:
                sys.modules[name] = previous
    return protocol, rows


def _water_records(
    rows: tuple[object, ...],
) -> tuple[tuple[tuple[int, object], ...], str]:
    water: list[tuple[int, object]] = []
    identities: list[dict[str, object]] = []
    for selection_index, item in enumerate(rows):
        if getattr(item, "canonical_solvent", None) != "water":
            continue
        eligible = getattr(item, "eligible_record")
        if getattr(eligible, "partition", None) != "development":
            raise StockAreaFeatureError("Frozen water row is not development data.")
        geometry = eligible.geometry
        opaque_record_id = _hex_digest(
            getattr(item, "opaque_record_id", None), name="opaque_record_id"
        )
        geometry_sha256 = _hex_digest(
            getattr(geometry, "sha256", None), name="geometry_sha256"
        )
        atomic_numbers = tuple(getattr(geometry, "atomic_numbers"))
        if not atomic_numbers:
            raise StockAreaFeatureError("Frozen water geometry has no atoms.")
        water.append((selection_index, item))
        identities.append(
            {
                "water_ordinal": len(water) - 1,
                "selection_index": selection_index,
                "opaque_record_id": opaque_record_id,
                "geometry_sha256": geometry_sha256,
                "atom_count": len(atomic_numbers),
            }
        )
    if len(water) != EXPECTED_WATER_RECORD_COUNT:
        raise StockAreaFeatureError("Expected exactly 306 frozen water rows.")
    return tuple(water), _canonical_sha256(identities)


def _validate_rotation_matrix() -> np.ndarray:
    rotation = np.asarray(ROTATION_CONTROL_MATRIX, dtype=np.float64)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-15, rtol=0.0):
        raise StockAreaFeatureError("Rotation control is not orthogonal.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2.0e-15):
        raise StockAreaFeatureError("Rotation control must be proper.")
    return rotation


def _pyscf_stock_area_row(
    symbols: tuple[str, ...], positions_angstrom: np.ndarray
) -> tuple[np.ndarray, float]:
    try:
        import pyscf
        from pyscf import gto
        from pyscf.data import radii
        from pyscf.dft import gen_grid
        from pyscf.solvent import pcm, smd
    except ImportError as exc:  # pragma: no cover - optional runtime boundary
        raise StockAreaFeatureError("PySCF stock-area runtime is unavailable.") from exc
    if pyscf.__version__ != PYSCF_VERSION:
        raise StockAreaFeatureError(
            f"PySCF {PYSCF_VERSION} is required, got {pyscf.__version__}."
        )
    if gen_grid.LEBEDEV_ORDER.get(SASA_LEBEDEV_ORDER) != SASA_GRID_POINTS_PER_ATOM:
        raise StockAreaFeatureError("PySCF Lebedev-order mapping drifted.")
    if max(gen_grid.LEBEDEV_ORDER.values()) != SASA_GRID_POINTS_PER_ATOM:
        raise StockAreaFeatureError("Configured SASA grid is not PySCF's maximum.")
    positions = np.asarray(positions_angstrom, dtype=np.float64)
    if positions.shape != (len(symbols), 3) or not np.all(np.isfinite(positions)):
        raise StockAreaFeatureError("Coordinates must be finite with shape (N,3).")
    mol = gto.M(
        atom=list(zip(symbols, positions.tolist(), strict=True)),
        unit="Angstrom",
        basis="sto-3g",
        charge=0,
        spin=0,
        verbose=0,
    )
    surface = pcm.gen_surface(
        mol,
        ng=SASA_GRID_POINTS_PER_ATOM,
        rad=radii.VDW + SASA_PROBE_RADIUS_ANGSTROM / radii.BOHR,
    )
    areas = (
        np.asarray(
            [
                np.sum(surface["area"][start:stop])
                for start, stop in surface["gslice_by_atom"]
            ],
            dtype=np.float64,
        )
        * float(radii.BOHR) ** 2
    )
    if areas.shape != (len(symbols),) or not np.all(np.isfinite(areas)):
        raise StockAreaFeatureError("PySCF returned invalid per-atom SASA.")
    if np.any(areas < 0.0):
        raise StockAreaFeatureError("PySCF returned a negative per-atom SASA.")
    atom_radii_angstrom = (
        np.asarray(radii.VDW[mol.atom_charges()], dtype=np.float64) * float(radii.BOHR)
        + SASA_PROBE_RADIUS_ANGSTROM
    )
    maximum_areas = 4.0 * np.pi * atom_radii_angstrom**2
    if np.any(areas > maximum_areas * (1.0 + 1.0e-12) + 1.0e-10):
        raise StockAreaFeatureError("PySCF returned an overfull per-atom SASA.")
    design = aqueous_cds_tension_design_row(symbols, positions, areas)
    design = np.asarray(design, dtype=np.float64)
    if design.shape != (len(SMD_WATER_TENSION_PARAMETER_NAMES),) or not np.all(
        np.isfinite(design)
    ):
        raise StockAreaFeatureError("SMD stock-area design row is invalid.")
    smd_object = smd.SMD(mol, solvent="water")
    legacy_kcal_mol = float(smd.get_cds_legacy(smd_object)[0] * smd.hartree2kcal)
    if not math.isfinite(legacy_kcal_mol):
        raise StockAreaFeatureError("Compiled legacy SMD control is non-finite.")
    return design, legacy_kcal_mol


def _feature_payload(
    *,
    water_ordinal: int,
    selection_index: int,
    item: object,
    preregistration_sha256: str,
    runtime_identity_sha256: str,
) -> dict[str, object]:
    if (
        not isinstance(water_ordinal, int)
        or isinstance(water_ordinal, bool)
        or water_ordinal < 0
    ):
        raise StockAreaFeatureError("water_ordinal must be a non-negative int.")
    if (
        not isinstance(selection_index, int)
        or isinstance(selection_index, bool)
        or selection_index < 0
    ):
        raise StockAreaFeatureError("selection_index must be a non-negative int.")
    preregistration_sha256 = _hex_digest(
        preregistration_sha256, name="preregistration_sha256"
    )
    runtime_identity_sha256 = _hex_digest(
        runtime_identity_sha256, name="runtime_identity_sha256"
    )
    if getattr(item, "canonical_solvent", None) != "water":
        raise StockAreaFeatureError("Stock-area rows must be water records.")
    eligible = getattr(item, "eligible_record")
    if getattr(eligible, "partition", None) != "development":
        raise StockAreaFeatureError("Stock-area rows must be development records.")
    geometry = eligible.geometry
    if getattr(geometry, "charge", None) != 0:
        raise StockAreaFeatureError(
            "Stock-area diagnostic requires neutral geometries."
        )
    if getattr(geometry, "multiplicity", None) != 1:
        raise StockAreaFeatureError(
            "Stock-area diagnostic requires singlet geometries."
        )
    symbols = tuple(chemical_symbols[int(number)] for number in geometry.atomic_numbers)
    positions = np.asarray(geometry.coordinates_angstrom, dtype=np.float64)
    design, legacy = _pyscf_stock_area_row(symbols, positions)
    stock = float(design @ SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2)
    parity_error = stock - legacy
    if abs(parity_error) > STOCK_ENERGY_PARITY_TOLERANCE_KCAL_MOL:
        raise StockAreaFeatureError("Stock-area energy does not close legacy SMD.")
    centered = positions - np.mean(positions, axis=0, keepdims=True)
    rotated = centered @ _validate_rotation_matrix().T
    rotated_design, rotated_legacy = _pyscf_stock_area_row(symbols, rotated)
    rotated_stock = float(
        rotated_design @ SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2
    )
    rotation_error = rotated_stock - stock
    if abs(rotation_error) > ROTATION_CONTROL_TOLERANCE_KCAL_MOL:
        raise StockAreaFeatureError("Stock-area rotation control exceeded tolerance.")
    legacy_rotation_error = rotated_legacy - legacy
    if abs(legacy_rotation_error) > ROTATION_CONTROL_TOLERANCE_KCAL_MOL:
        raise StockAreaFeatureError("Legacy SMD rotation control exceeded tolerance.")
    rotated_parity_error = rotated_stock - rotated_legacy
    if abs(rotated_parity_error) > STOCK_ENERGY_PARITY_TOLERANCE_KCAL_MOL:
        raise StockAreaFeatureError(
            "Rotated stock-area energy does not close legacy SMD."
        )
    payload: dict[str, object] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "pass",
        "do_not_commit": True,
        "partition": "development",
        "canonical_solvent": "water",
        "dataset_loader_parsed_experimental_targets": True,
        "experimental_targets_used_by_feature_computation": False,
        "experimental_targets_emitted": False,
        "hybrid_prediction_records_read": False,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_selected_or_emitted": False,
        "fitting_or_calibration_performed": False,
        "energy_only_diagnostic": True,
        "force_capability": False,
        "water_ordinal": water_ordinal,
        "selection_index": selection_index,
        "opaque_record_id": _hex_digest(
            getattr(item, "opaque_record_id", None), name="opaque_record_id"
        ),
        "geometry_sha256": _hex_digest(
            getattr(geometry, "sha256", None), name="geometry_sha256"
        ),
        "atom_count": len(symbols),
        "symbols": list(symbols),
        "preregistration_sha256": preregistration_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "design_parameter_names": list(SMD_WATER_TENSION_PARAMETER_NAMES),
        "design_row_angstrom2_div_1000": design.tolist(),
        "stock_smd_reconstruction_kcal_mol": stock,
        "compiled_legacy_smd_kcal_mol": legacy,
        "stock_minus_compiled_legacy_kcal_mol": parity_error,
        "rotation_control_stock_error_kcal_mol": rotation_error,
        "rotation_control_legacy_error_kcal_mol": legacy_rotation_error,
        "rotated_stock_minus_compiled_legacy_kcal_mol": rotated_parity_error,
        "claim_boundary": claim_boundary(),
    }
    if PROHIBITED_OUTPUT_KEYS.intersection(payload):
        raise StockAreaFeatureError("Feature payload contains a prohibited target.")
    payload["feature_sha256"] = _canonical_sha256(payload)
    return payload


def _validate_preregistration(
    *,
    preregistration: Mapping[str, Any],
    preregistration_path: Path,
    source_root: Path,
    input_root: Path,
) -> None:
    expected_fields = {
        "artifact": PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": PREREGISTRATION_STATUS,
        "partition": "development-water-only",
        "water_record_count": EXPECTED_WATER_RECORD_COUNT,
        "dataset_loader_parses_experimental_targets": True,
        "experimental_targets_used_by_feature_computation": False,
        "experimental_targets_emitted": False,
        "hybrid_prediction_records_read_by_feature_generator": False,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_selected_or_emitted": False,
        "fitting_or_calibration_permitted": False,
        "checkpoint_or_method_selection_permitted": False,
        "geometry_only_unit_canaries_precede_lock": True,
        "energy_only_diagnostic": True,
        "force_capability": False,
    }
    if set(preregistration) != preregistration_schema_keys():
        raise StockAreaFeatureError("Preregistration top-level schema drifted.")
    for key, value in expected_fields.items():
        if preregistration.get(key) != value:
            raise StockAreaFeatureError(f"Preregistration {key!r} drifted.")
    if preregistration.get("source_root") != str(source_root):
        raise StockAreaFeatureError("Preregistration source root drifted.")
    if preregistration.get("input_root") != str(input_root):
        raise StockAreaFeatureError("Preregistration input root drifted.")
    if preregistration.get("preregistration_path") != str(preregistration_path):
        raise StockAreaFeatureError("Preregistration path binding drifted.")
    if preregistration.get("claim_boundary") != claim_boundary():
        raise StockAreaFeatureError("Preregistration claim boundary drifted.")
    locked_at = preregistration.get("locked_at_utc")
    if not isinstance(locked_at, str):
        raise StockAreaFeatureError("Preregistration lock time is invalid.")
    try:
        parsed_locked_at = datetime.fromisoformat(locked_at)
    except ValueError as exc:
        raise StockAreaFeatureError("Preregistration lock time is invalid.") from exc
    if parsed_locked_at.tzinfo is None:
        raise StockAreaFeatureError("Preregistration lock time lacks a timezone.")
    if _git(source_root, "rev-parse", "HEAD") != preregistration.get("source_git_head"):
        raise StockAreaFeatureError("Preregistered source HEAD drifted.")
    if _git(source_root, "rev-parse", "HEAD^{tree}") != preregistration.get(
        "source_git_tree"
    ):
        raise StockAreaFeatureError("Preregistered source tree drifted.")
    if _git(source_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StockAreaFeatureError("Feature generation requires a clean checkout.")
    if preregistration.get("feature_generator_sha256") != _sha256(Path(__file__)):
        raise StockAreaFeatureError("Feature-generator hash drifted.")
    without_self = dict(preregistration)
    observed_self = without_self.pop("self_sha256", None)
    if observed_self != _canonical_sha256(without_self):
        raise StockAreaFeatureError("Preregistration self hash drifted.")
    if preregistration_path.stat().st_mode & 0o222:
        raise StockAreaFeatureError("Preregistration must be read-only.")
    parent_path = (
        Path(str(preregistration.get("parent_hybrid_preregistration_path")))
        .expanduser()
        .resolve(strict=True)
    )
    if preregistration.get("parent_hybrid_preregistration_sha256") != _sha256(
        parent_path
    ):
        raise StockAreaFeatureError("Parent hybrid preregistration drifted.")
    if parent_path.stat().st_mode & 0o222:
        raise StockAreaFeatureError("Parent hybrid preregistration must be read-only.")
    parent = _load_json_object(parent_path, name="parent hybrid preregistration")
    for key, value in {
        "artifact_id": PARENT_ARTIFACT,
        "schema_version": 3,
        "status": "locked-before-first-v3-hybrid-evaluation",
        "partition": "development",
        "record_count": 505,
        "confirmation_partition_opened": False,
        "fitting_or_calibration_permitted": False,
    }.items():
        if parent.get(key) != value:
            raise StockAreaFeatureError(f"Parent field {key!r} drifted.")
    if preregistration.get("parent_hybrid_source_git_head") != parent.get(
        "source_git_head"
    ):
        raise StockAreaFeatureError("Parent source binding drifted.")
    if parent.get("input_root") != str(input_root):
        raise StockAreaFeatureError("Parent frozen input root drifted.")
    input_hashes = preregistration.get("input_files_sha256")
    if not isinstance(input_hashes, dict) or set(input_hashes) != set(INPUT_FILE_NAMES):
        raise StockAreaFeatureError("Preregistration input hashes are incomplete.")
    for relative, expected in input_hashes.items():
        if _sha256(_relative_file(input_root, relative, name="input file")) != expected:
            raise StockAreaFeatureError(f"Input file {relative!r} drifted.")
    _validate_parent_input_hashes(parent, input_hashes)
    source_hashes = preregistration.get("source_files_sha256")
    if not isinstance(source_hashes, dict) or not set(
        REQUIRED_SOURCE_FILE_NAMES
    ).issubset(source_hashes):
        raise StockAreaFeatureError("Preregistration source hashes are incomplete.")
    for relative, expected in source_hashes.items():
        if (
            _sha256(_relative_file(source_root, relative, name="source file"))
            != expected
        ):
            raise StockAreaFeatureError(f"Source file {relative!r} drifted.")
    runtime_identity, runtime_sha = _runtime_identity()
    if preregistration.get("runtime_identity") != runtime_identity:
        raise StockAreaFeatureError("Numerical runtime drifted.")
    if preregistration.get("runtime_identity_sha256") != runtime_sha:
        raise StockAreaFeatureError("Runtime digest drifted.")
    pyscf_assets, pyscf_assets_sha = _pyscf_runtime_assets()
    if preregistration.get("pyscf_runtime_assets") != pyscf_assets:
        raise StockAreaFeatureError("PySCF runtime assets drifted.")
    if preregistration.get("pyscf_runtime_assets_sha256") != pyscf_assets_sha:
        raise StockAreaFeatureError("PySCF runtime asset digest drifted.")
    if preregistration.get("area_definition") != stock_area_contract():
        raise StockAreaFeatureError("Preregistered area definition drifted.")
    if preregistration.get("linear_basis") != linear_basis_contract():
        raise StockAreaFeatureError("Preregistered linear basis drifted.")
    if (
        preregistration.get("numerical_choice_rationale")
        != numerical_choice_rationale()
    ):
        raise StockAreaFeatureError("Numerical-choice rationale drifted.")
    if preregistration.get("output_contract") != output_contract():
        raise StockAreaFeatureError("Output contract drifted.")


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)


def generate(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    if source_root != _SCRIPT_SOURCE_ROOT:
        raise StockAreaFeatureError("Generator must run from its source checkout.")
    input_root = args.input_root.expanduser().resolve(strict=True)
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    _outside_root(preregistration_path, source_root, name="preregistration")
    _outside_root(output_dir, source_root, name="feature output directory")
    preregistration = _load_json_object(
        preregistration_path, name="stock-area preregistration"
    )
    _validate_preregistration(
        preregistration=preregistration,
        preregistration_path=preregistration_path,
        source_root=source_root,
        input_root=input_root,
    )
    if preregistration.get("feature_output_dir") != str(output_dir):
        raise StockAreaFeatureError("Feature output directory binding drifted.")
    _protocol, rows = _load_selection(source_root=source_root, input_root=input_root)
    water, identity_sha = _water_records(rows)
    if preregistration.get("water_identity_sha256") != identity_sha:
        raise StockAreaFeatureError("Frozen water identity drifted.")
    loaded = collect_loaded_repository_sources(
        source_root,
        required_paths=tuple(
            path for path in REQUIRED_SOURCE_FILE_NAMES if path.endswith(".py")
        ),
    )
    if not set(loaded).issubset(preregistration["source_files_sha256"]):
        raise StockAreaFeatureError("Preregistration omits loaded source files.")
    start = int(args.start)
    stop = len(water) if args.stop is None else int(args.stop)
    if not 0 <= start < stop <= len(water):
        raise StockAreaFeatureError("Feature shard bounds are invalid.")
    preregistration_sha = _sha256(preregistration_path)
    generated: list[str] = []
    for ordinal in range(start, stop):
        selection_index, item = water[ordinal]
        payload = _feature_payload(
            water_ordinal=ordinal,
            selection_index=selection_index,
            item=item,
            preregistration_sha256=preregistration_sha,
            runtime_identity_sha256=preregistration["runtime_identity_sha256"],
        )
        destination = output_dir / f"water-{ordinal:03d}.json"
        _write_json_exclusive(destination, payload)
        generated.append(destination.name)
    summary: dict[str, object] = {
        "artifact": "route2-maple-cds-w1-water-stock-area-feature-shard-v1",
        "status": "complete",
        "start": start,
        "stop": stop,
        "generated_count": len(generated),
        "generated_files": generated,
        "preregistration_sha256": preregistration_sha,
        "runtime_identity_sha256": preregistration["runtime_identity_sha256"],
        "experimental_targets_used_by_feature_computation": False,
        "experimental_targets_emitted": False,
        "hybrid_prediction_records_read": False,
        "confirmation_selection_manifest_opened": False,
        "fitting_or_calibration_performed": False,
        "energy_only_diagnostic": True,
        "force_capability": False,
    }
    summary["summary_sha256"] = _canonical_sha256(summary)
    if _git(source_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StockAreaFeatureError("Source checkout changed during generation.")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args()
    print(json.dumps(generate(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
