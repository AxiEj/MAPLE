#!/usr/bin/env python3
"""Generate experiment-blind MAPLE-CDS-W1 water geometry features.

This tool is deliberately target-blind at the feature-computation boundary.
The licensed MNSol loader necessarily parses the table, including its target
column, because the same archive is the frozen source of molecular geometries.
No target attribute is subsequently accessed by the feature computation or
copied into an output.  The tool validates the frozen development selection,
selects its preregistered water rows, and evaluates only the content-addressed
smooth harmonic area and fixed 18-column aqueous SMD surface-tension design
row.  It never opens hybrid prediction records or a confirmation-selection
manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

_SCRIPT_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SCRIPT_SOURCE_ROOT))

from ase import Atoms
from ase.data import chemical_symbols
import numpy as np

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2,
    SMD_WATER_TENSION_PARAMETER_NAMES,
    smd_sasa_radii,
)
from maple.solvation.continuum.harmonic_cds_area import (
    SMOOTH_HARMONIC_EXPOSURE_AREA_CONTRACT_ID,
    SmoothHarmonicExposureArea,
)
from maple.solvation.harmonic_cds import (
    MAPLE_CDS_W1_HARMONIC_LINEAR_PROFILE_ID,
    SmoothHarmonicAqueousLinearCDSTerm,
)
from maple.solvation.release.evidence import (
    collect_loaded_repository_sources,
    runtime_record,
)

ARTIFACT = "route2-maple-cds-w1-water-feature-record-v1"
PREREGISTRATION_ARTIFACT = "route2-maple-cds-w1-water-features-prereg-v1"
EXPECTED_WATER_RECORD_COUNT = 306
AREA_TRANSITION_WIDTH_ANGSTROM2 = 0.18
AREA_EXPOSURE_LMAX = 4
AREA_RADIAL_QUADRATURE_ORDER = 192
AREA_DTYPE = "torch.float64"
AREA_DEVICE = "cpu"
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
    "docs/route2/evidence/HARMONIC_CDS_AREA_PRO_AUDIT_2026-08-16.md",
    "maple/function/calculator/extra_correction/implicit/smd_cds.py",
    "maple/function/route2_smd_profiles.py",
    "maple/function/route2_solvents.py",
    "maple/solvation/api/units.py",
    "maple/solvation/continuum/harmonic_cds_area.py",
    "maple/solvation/continuum/harmonic_coefficients.py",
    "maple/solvation/continuum/harmonic_exposure.py",
    "maple/solvation/continuum/harmonic_single_layer.py",
    "maple/solvation/continuum/harmonic_torch_primitives.py",
    "maple/solvation/coupling/operator.py",
    "maple/solvation/coupling/spaces.py",
    "maple/solvation/coupling/state_equation.py",
    "maple/solvation/harmonic_cds.py",
    "maple/solvation/release/evidence.py",
    "maple/solvation/solvent_terms.py",
    "tests/route2_vnext/test_maple_cds_w1_feature_generation.py",
    "tools/route2_release/create_maple_cds_w1_feature_preregistration.py",
    "tools/route2_release/generate_maple_cds_w1_features.py",
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


class W1FeatureGenerationError(RuntimeError):
    """Raised when target-free feature provenance is incomplete or inconsistent."""


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


def _runtime_identity() -> tuple[dict[str, object], str]:
    identity = runtime_record()
    identity.pop("generated_at_utc", None)
    return identity, _canonical_sha256(identity)


def _hex_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise W1FeatureGenerationError(f"{name} must be a lowercase SHA256.")
    return value


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        diagnostic = result.stderr.strip() or result.stdout.strip()
        raise W1FeatureGenerationError(
            f"Git command {arguments!r} failed: {diagnostic}"
        )
    return result.stdout.strip()


def _relative_file(root: Path, raw: object, *, name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise W1FeatureGenerationError(f"{name} must be a non-empty relative path.")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise W1FeatureGenerationError(f"{name} must stay inside its bound root.")
    resolved = (root / relative).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise W1FeatureGenerationError(
            f"{name} resolves outside its bound root."
        ) from exc
    if not resolved.is_file():
        raise W1FeatureGenerationError(f"{name} must resolve to a regular file.")
    return resolved


def _outside_root(path: Path, root: Path, *, name: str) -> None:
    try:
        path.relative_to(root)
    except ValueError:
        return
    raise W1FeatureGenerationError(f"{name} must be outside the source checkout.")


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise W1FeatureGenerationError(f"{name} must be a JSON object.")
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
                raise W1FeatureGenerationError(
                    "MNSol benchmark modules were imported from another checkout."
                )

        protocol_path = input_root / "route2-mnsol-protocol-v1.json"
        dataset_path = input_root / "MNSolDatabase_v2012.zip"
        selection_path = (
            input_root / "route2-mnsol-development-selection-v1.private.json"
        )
        pilot_path = input_root / "route2-mnsol-pilot-selection-v1.json"
        protocol = dataset_module.load_mnsol_protocol(protocol_path)
        dataset = dataset_module.load_mnsol_v2012(dataset_path, protocol)
        selection = _load_json_object(selection_path, name="development selection")
        pilot = _load_json_object(pilot_path, name="pilot selection")
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
            raise W1FeatureGenerationError(
                "Frozen water identity contains a non-development record."
            )
        geometry = eligible.geometry
        opaque_record_id = _hex_digest(
            getattr(item, "opaque_record_id", None),
            name="opaque_record_id",
        )
        geometry_sha256 = _hex_digest(
            getattr(geometry, "sha256", None),
            name="geometry_sha256",
        )
        atomic_numbers = tuple(getattr(geometry, "atomic_numbers"))
        if not atomic_numbers:
            raise W1FeatureGenerationError("Frozen water geometry has no atoms.")
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
        raise W1FeatureGenerationError(
            "Frozen development selection does not contain exactly 306 water rows."
        )
    return tuple(water), _canonical_sha256(identities)


def _maximum_active_pair_factors(
    water: tuple[tuple[int, object], ...],
) -> int:
    """Reproduce the harmonic product-degree bound from geometry only."""

    maximum = 0
    for _selection_index, item in water:
        geometry = item.eligible_record.geometry
        numbers = tuple(int(value) for value in geometry.atomic_numbers)
        symbols = tuple(chemical_symbols[number] for number in numbers)
        radii = np.asarray(smd_sasa_radii(symbols), dtype=float)
        positions = np.asarray(geometry.coordinates_angstrom, dtype=float)
        if positions.shape != (len(numbers), 3) or not np.all(np.isfinite(positions)):
            raise W1FeatureGenerationError("Frozen water geometry is invalid.")
        for atom_i, radius_i in enumerate(radii):
            active = 0
            buried = False
            for atom_j, radius_j in enumerate(radii):
                if atom_i == atom_j:
                    continue
                distance = float(np.linalg.norm(positions[atom_j] - positions[atom_i]))
                if not math.isfinite(distance) or distance <= 1.0e-12:
                    raise W1FeatureGenerationError(
                        "Frozen water geometry has coincident atom centres."
                    )
                z_minimum = (distance - radius_i) ** 2 - radius_j**2
                z_maximum = (distance + radius_i) ** 2 - radius_j**2
                if z_minimum >= AREA_TRANSITION_WIDTH_ANGSTROM2:
                    continue
                if z_maximum <= -AREA_TRANSITION_WIDTH_ANGSTROM2:
                    buried = True
                    break
                active += 1
            if not buried:
                maximum = max(maximum, active)
    return maximum


def _validate_preregistration(
    *,
    preregistration: Mapping[str, Any],
    preregistration_path: Path,
    source_root: Path,
    input_root: Path,
) -> None:
    if preregistration.get("artifact") != PREREGISTRATION_ARTIFACT:
        raise W1FeatureGenerationError("Unknown W1 feature preregistration.")
    for key, expected in (
        ("schema_version", 1),
        ("status", "locked-before-first-feature-evaluation"),
        ("partition", "development-water-only"),
        ("water_record_count", EXPECTED_WATER_RECORD_COUNT),
        ("dataset_loader_parses_experimental_targets", True),
        ("experimental_targets_used_by_feature_computation", False),
        ("experimental_targets_emitted", False),
        ("hybrid_prediction_records_read_by_feature_generator", False),
        ("confirmation_selection_manifest_opened", False),
        ("confirmation_records_selected_or_emitted", False),
        ("fitting_or_calibration_permitted", False),
        ("checkpoint_or_method_selection_permitted", False),
    ):
        if preregistration.get(key) != expected:
            raise W1FeatureGenerationError(f"Preregistration {key} drifted.")
    if preregistration.get("source_root") != str(source_root):
        raise W1FeatureGenerationError("Preregistration source_root drifted.")
    if preregistration.get("input_root") != str(input_root):
        raise W1FeatureGenerationError("Preregistration input_root drifted.")
    if preregistration.get("preregistration_path") != str(preregistration_path):
        raise W1FeatureGenerationError("Preregistration path binding drifted.")
    expected_head = preregistration.get("source_git_head")
    expected_tree = preregistration.get("source_git_tree")
    if _git(source_root, "rev-parse", "HEAD") != expected_head:
        raise W1FeatureGenerationError("Preregistered source HEAD drifted.")
    if _git(source_root, "rev-parse", "HEAD^{tree}") != expected_tree:
        raise W1FeatureGenerationError("Preregistered source tree drifted.")
    if _git(source_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise W1FeatureGenerationError(
            "W1 feature generation requires a clean source worktree."
        )
    if preregistration.get("feature_generator_sha256") != _sha256(Path(__file__)):
        raise W1FeatureGenerationError("Feature-generator source hash drifted.")
    expected_preregistration_sha256 = preregistration.get("self_sha256")
    without_self = dict(preregistration)
    without_self.pop("self_sha256", None)
    if expected_preregistration_sha256 != _canonical_sha256(without_self):
        raise W1FeatureGenerationError("Preregistration self hash drifted.")
    if preregistration_path.stat().st_mode & 0o222:
        raise W1FeatureGenerationError("Preregistration must be read-only.")
    parent_path_raw = preregistration.get("parent_hybrid_preregistration_path")
    parent_path = Path(str(parent_path_raw)).expanduser().resolve(strict=True)
    if preregistration.get("parent_hybrid_preregistration_sha256") != _sha256(
        parent_path
    ):
        raise W1FeatureGenerationError("Parent hybrid preregistration drifted.")
    parent = _load_json_object(parent_path, name="parent hybrid preregistration")
    for key, expected in (
        ("artifact_id", "route2-hybrid-smd-development-prereg-v3"),
        ("schema_version", 3),
        ("status", "locked-before-first-v3-hybrid-evaluation"),
        ("partition", "development"),
        ("record_count", 505),
        ("confirmation_partition_opened", False),
        ("fitting_or_calibration_permitted", False),
    ):
        if parent.get(key) != expected:
            raise W1FeatureGenerationError(
                f"Parent hybrid preregistration {key!r} drifted."
            )
    if preregistration.get("parent_hybrid_source_git_head") != parent.get(
        "source_git_head"
    ):
        raise W1FeatureGenerationError("Parent hybrid source binding drifted.")
    input_hashes = preregistration.get("input_files_sha256")
    if not isinstance(input_hashes, dict) or tuple(sorted(input_hashes)) != tuple(
        sorted(INPUT_FILE_NAMES)
    ):
        raise W1FeatureGenerationError("Preregistration input hashes are missing.")
    for relative, expected in input_hashes.items():
        path = _relative_file(input_root, relative, name="input file")
        if _sha256(path) != expected:
            raise W1FeatureGenerationError(f"Input file {relative!r} drifted.")
    source_hashes = preregistration.get("source_files_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise W1FeatureGenerationError("Preregistration source hashes are missing.")
    missing_required = sorted(set(REQUIRED_SOURCE_FILE_NAMES) - set(source_hashes))
    if missing_required:
        raise W1FeatureGenerationError(
            "Preregistration omits required source files: "
            + ", ".join(missing_required)
        )
    for relative, expected in source_hashes.items():
        path = _relative_file(source_root, relative, name="source file")
        if _sha256(path) != expected:
            raise W1FeatureGenerationError(f"Source file {relative!r} drifted.")
    area = preregistration.get("area_definition")
    expected_area = {
        "contract_id": SMOOTH_HARMONIC_EXPOSURE_AREA_CONTRACT_ID,
        "profile_id": MAPLE_CDS_W1_HARMONIC_LINEAR_PROFILE_ID,
        "radii": "published-smd-sasa-radii-including-0.4-A-probe",
        "transition_width_angstrom2": AREA_TRANSITION_WIDTH_ANGSTROM2,
        "exposure_lmax": AREA_EXPOSURE_LMAX,
        "radial_quadrature_order": AREA_RADIAL_QUADRATURE_ORDER,
        "dtype": AREA_DTYPE,
        "device": AREA_DEVICE,
        "area_measure": "a_i^2-integral-e_i-domega",
    }
    if area != expected_area:
        raise W1FeatureGenerationError("Preregistered area definition drifted.")
    linear_basis = preregistration.get("linear_basis")
    expected_stock = SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2.tolist()
    expected_linear_basis = {
        "contract": "published-aqueous-smd-linear-18-column-v1",
        "parameter_names": list(SMD_WATER_TENSION_PARAMETER_NAMES),
        "stock_coefficients_cal_mol_angstrom2": expected_stock,
        "stock_coefficients_sha256": _canonical_sha256(expected_stock),
    }
    if linear_basis != expected_linear_basis:
        raise W1FeatureGenerationError("Preregistered linear basis drifted.")
    expected_output_contract = {
        "one_exclusive_json_per_water_record": True,
        "coordinates_emitted": False,
        "experimental_targets_emitted": False,
        "hybrid_outputs_emitted": False,
        "fit_or_calibration_emitted": False,
    }
    if preregistration.get("output_contract") != expected_output_contract:
        raise W1FeatureGenerationError("Preregistered output contract drifted.")
    runtime_identity, runtime_identity_sha256 = _runtime_identity()
    if preregistration.get("runtime_identity") != runtime_identity:
        raise W1FeatureGenerationError("Preregistered numerical runtime drifted.")
    if preregistration.get("runtime_identity_sha256") != runtime_identity_sha256:
        raise W1FeatureGenerationError("Preregistered runtime digest drifted.")


def _feature_payload(
    *,
    water_ordinal: int,
    selection_index: int,
    item: object,
    preregistration_sha256: str,
    runtime_identity_sha256: str,
) -> dict[str, object]:
    if (
        isinstance(water_ordinal, bool)
        or not isinstance(water_ordinal, int)
        or water_ordinal < 0
    ):
        raise W1FeatureGenerationError("water_ordinal must be a non-negative int.")
    if (
        isinstance(selection_index, bool)
        or not isinstance(selection_index, int)
        or selection_index < 0
    ):
        raise W1FeatureGenerationError("selection_index must be a non-negative int.")
    preregistration_sha256 = _hex_digest(
        preregistration_sha256,
        name="preregistration_sha256",
    )
    runtime_identity_sha256 = _hex_digest(
        runtime_identity_sha256,
        name="runtime_identity_sha256",
    )
    if getattr(item, "canonical_solvent", None) != "water":
        raise W1FeatureGenerationError("W1 feature rows must be water records.")
    eligible = getattr(item, "eligible_record")
    if getattr(eligible, "partition", None) != "development":
        raise W1FeatureGenerationError("W1 feature rows must be development records.")
    geometry = eligible.geometry
    opaque_record_id = _hex_digest(
        getattr(item, "opaque_record_id", None),
        name="opaque_record_id",
    )
    geometry_digest = _hex_digest(
        getattr(geometry, "sha256", None),
        name="geometry_sha256",
    )
    symbols = tuple(chemical_symbols[number] for number in geometry.atomic_numbers)
    atoms = Atoms(symbols, positions=np.asarray(geometry.coordinates_angstrom))
    radii = tuple(float(value) for value in smd_sasa_radii(symbols))
    import torch

    area = SmoothHarmonicExposureArea(
        atomic_numbers=tuple(int(value) for value in geometry.atomic_numbers),
        radii_angstrom=radii,
        transition_width_angstrom2=AREA_TRANSITION_WIDTH_ANGSTROM2,
        exposure_lmax=AREA_EXPOSURE_LMAX,
        radial_quadrature_order=AREA_RADIAL_QUADRATURE_ORDER,
        dtype=torch.float64,
        device=AREA_DEVICE,
    )
    term = SmoothHarmonicAqueousLinearCDSTerm(
        symbols=symbols,
        area=area,
        coefficients_cal_mol_angstrom2=(
            SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2
        ),
    )
    design = term.design_row_kcal_mol(atoms)
    if design.shape != (18,) or not np.all(np.isfinite(design)):
        raise W1FeatureGenerationError("W1 design row is invalid.")
    stock_control = float(
        np.dot(design, SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2)
    )
    if not math.isfinite(stock_control):
        raise W1FeatureGenerationError("W1 stock-control reconstruction is invalid.")
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
        "water_ordinal": water_ordinal,
        "selection_index": selection_index,
        "opaque_record_id": opaque_record_id,
        "geometry_sha256": geometry_digest,
        "atom_count": len(symbols),
        "symbols": list(symbols),
        "preregistration_sha256": preregistration_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "area_state_identity": area.state_identity(atoms),
        "stock_control_configuration_sha256": term.configuration_sha256(),
        "design_parameter_names": list(SMD_WATER_TENSION_PARAMETER_NAMES),
        "design_row_angstrom2_div_1000": design.tolist(),
        "stock_smd_control_kcal_mol": stock_control,
        "claim_boundary": (
            "Target-blind geometry feature computation only; the dataset loader "
            "parsed the source table, but no target or hybrid output was used or "
            "emitted and no fit, calibration, confirmation selection, or accuracy "
            "claim occurred."
        ),
    }
    if PROHIBITED_OUTPUT_KEYS.intersection(payload):
        raise W1FeatureGenerationError("Feature payload contains a prohibited target.")
    payload["feature_sha256"] = _canonical_sha256(payload)
    return payload


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)


def generate(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    if _SCRIPT_SOURCE_ROOT != source_root:
        raise W1FeatureGenerationError(
            "Feature generator must be executed from its bound source root."
        )
    input_root = args.input_root.expanduser().resolve(strict=True)
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    _outside_root(preregistration_path, source_root, name="preregistration")
    _outside_root(output_dir, source_root, name="feature output directory")
    preregistration = _load_json_object(
        preregistration_path,
        name="W1 feature preregistration",
    )
    _validate_preregistration(
        preregistration=preregistration,
        preregistration_path=preregistration_path,
        source_root=source_root,
        input_root=input_root,
    )
    if preregistration.get("feature_output_dir") != str(output_dir):
        raise W1FeatureGenerationError("Feature output directory binding drifted.")
    _protocol, rows = _load_selection(source_root=source_root, input_root=input_root)
    water, water_identity_sha256 = _water_records(rows)
    if preregistration.get("water_identity_sha256") != water_identity_sha256:
        raise W1FeatureGenerationError("Frozen water identity binding drifted.")
    maximum_active = _maximum_active_pair_factors(water)
    finite_product_degree = (maximum_active + 1) * AREA_EXPOSURE_LMAX
    expected_rationale = {
        "maximum_observed_active_pair_factors": maximum_active,
        "finite_product_degree_formula": "(active_pair_factors+1)*lmax",
        "finite_product_degree": finite_product_degree,
        "implementation_degree_limit": 128,
        "selected_from_geometry_only": True,
        "accuracy_results_used": False,
    }
    if preregistration.get("numerical_choice_rationale") != expected_rationale:
        raise W1FeatureGenerationError(
            "Preregistered geometry-only degree rationale drifted."
        )
    if finite_product_degree > 128:
        raise W1FeatureGenerationError(
            "Preregistered harmonic exposure exceeds the finite degree bound."
        )
    loaded_sources = collect_loaded_repository_sources(
        source_root,
        required_paths=tuple(
            name for name in REQUIRED_SOURCE_FILE_NAMES if name.endswith(".py")
        ),
    )
    source_hashes = preregistration["source_files_sha256"]
    missing_loaded = sorted(set(loaded_sources) - set(source_hashes))
    if missing_loaded:
        raise W1FeatureGenerationError(
            "Preregistration omits loaded repository sources: "
            + ", ".join(missing_loaded)
        )
    start = int(args.start)
    stop = len(water) if args.stop is None else int(args.stop)
    if not 0 <= start < stop <= len(water):
        raise W1FeatureGenerationError("Feature shard bounds are invalid.")
    preregistration_sha256 = _sha256(preregistration_path)
    generated: list[str] = []
    for water_ordinal in range(start, stop):
        selection_index, item = water[water_ordinal]
        payload = _feature_payload(
            water_ordinal=water_ordinal,
            selection_index=selection_index,
            item=item,
            preregistration_sha256=preregistration_sha256,
            runtime_identity_sha256=preregistration["runtime_identity_sha256"],
        )
        destination = output_dir / f"water-{water_ordinal:03d}.json"
        _write_json_exclusive(destination, payload)
        generated.append(destination.name)
    summary = {
        "artifact": "route2-maple-cds-w1-water-feature-shard-v1",
        "status": "complete",
        "start": start,
        "stop": stop,
        "generated_count": len(generated),
        "generated_files": generated,
        "preregistration_sha256": preregistration_sha256,
        "source_git_head": preregistration["source_git_head"],
        "runtime_identity_sha256": preregistration["runtime_identity_sha256"],
        "dataset_loader_parsed_experimental_targets": True,
        "experimental_targets_used_by_feature_computation": False,
        "experimental_targets_emitted": False,
        "hybrid_prediction_records_read": False,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_selected_or_emitted": False,
        "fitting_or_calibration_performed": False,
    }
    summary["summary_sha256"] = _canonical_sha256(summary)
    if _git(source_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise W1FeatureGenerationError(
            "Source worktree changed during W1 feature generation."
        )
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
