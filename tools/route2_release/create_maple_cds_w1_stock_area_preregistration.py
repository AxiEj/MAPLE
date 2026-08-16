#!/usr/bin/env python3
"""Freeze target-blind PySCF stock-area features for the M3 diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

_SCRIPT_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SCRIPT_SOURCE_ROOT))

from maple.solvation.release.evidence import (
    RepositorySnapshot,
    canonical_json_sha256,
    collect_loaded_repository_sources,
    committed_source_hashes,
    sha256_file,
)

from generate_maple_cds_w1_stock_area_features import (
    EXPECTED_WATER_RECORD_COUNT,
    INPUT_FILE_NAMES,
    PARENT_ARTIFACT,
    PREREGISTRATION_ARTIFACT,
    PREREGISTRATION_STATUS,
    REQUIRED_SOURCE_FILE_NAMES,
    StockAreaFeatureError,
    _load_selection,
    _pyscf_runtime_assets,
    _runtime_identity,
    _validate_parent_input_hashes,
    _water_records,
    claim_boundary,
    linear_basis_contract,
    numerical_choice_rationale,
    output_contract,
    preregistration_schema_keys,
    stock_area_contract,
)


def _outside_source(path: Path, source_root: Path, *, name: str) -> None:
    try:
        path.relative_to(source_root)
    except ValueError:
        return
    raise StockAreaFeatureError(f"{name} must be outside source checkout.")


def _load_parent(path: Path) -> dict[str, object]:
    if path.stat().st_mode & 0o222:
        raise StockAreaFeatureError("Parent hybrid preregistration must be read-only.")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise StockAreaFeatureError("Parent hybrid preregistration is invalid.")
    for key, value in {
        "artifact_id": PARENT_ARTIFACT,
        "schema_version": 3,
        "status": "locked-before-first-v3-hybrid-evaluation",
        "partition": "development",
        "record_count": 505,
        "confirmation_partition_opened": False,
        "fitting_or_calibration_permitted": False,
    }.items():
        if payload.get(key) != value:
            raise StockAreaFeatureError(f"Parent field {key!r} drifted.")
    return payload


def create(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    if source_root != _SCRIPT_SOURCE_ROOT:
        raise StockAreaFeatureError("Creator must run for its own source checkout.")
    input_root = args.input_root.expanduser().resolve(strict=True)
    parent_path = args.parent_hybrid_preregistration.expanduser().resolve(strict=True)
    output_dir = args.feature_output_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    _outside_source(output, source_root, name="preregistration")
    _outside_source(output_dir, source_root, name="feature output directory")
    if output.exists():
        raise FileExistsError(output)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise StockAreaFeatureError("Feature output directory must be empty.")

    snapshot = RepositorySnapshot.capture(source_root)
    parent = _load_parent(parent_path)
    if parent.get("input_root") != str(input_root):
        raise StockAreaFeatureError("Parent uses a different frozen input root.")
    input_hashes = {name: sha256_file(input_root / name) for name in INPUT_FILE_NAMES}
    _validate_parent_input_hashes(parent, input_hashes)
    _protocol, rows = _load_selection(source_root=source_root, input_root=input_root)
    water, water_identity_sha256 = _water_records(rows)
    loaded = collect_loaded_repository_sources(
        source_root,
        required_paths=tuple(
            path for path in REQUIRED_SOURCE_FILE_NAMES if path.endswith(".py")
        ),
    )
    source_hashes = committed_source_hashes(
        snapshot,
        tuple(sorted(set(loaded).union(REQUIRED_SOURCE_FILE_NAMES))),
    )
    runtime_identity, runtime_sha = _runtime_identity()
    pyscf_assets, pyscf_assets_sha = _pyscf_runtime_assets()
    generator_relative = (
        "tools/route2_release/generate_maple_cds_w1_stock_area_features.py"
    )
    payload: dict[str, object] = {
        "artifact": PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": PREREGISTRATION_STATUS,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "development-water-only",
        "water_record_count": len(water),
        "water_identity_sha256": water_identity_sha256,
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
        "source_root": str(source_root),
        "source_git_head": snapshot.head,
        "source_git_tree": snapshot.tree,
        "source_files_sha256": source_hashes,
        "feature_generator_sha256": source_hashes[generator_relative],
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": runtime_sha,
        "pyscf_runtime_assets": pyscf_assets,
        "pyscf_runtime_assets_sha256": pyscf_assets_sha,
        "input_root": str(input_root),
        "input_files_sha256": input_hashes,
        "parent_hybrid_preregistration_path": str(parent_path),
        "parent_hybrid_preregistration_sha256": sha256_file(parent_path),
        "parent_hybrid_source_git_head": parent["source_git_head"],
        "feature_output_dir": str(output_dir),
        "preregistration_path": str(output),
        "area_definition": stock_area_contract(),
        "linear_basis": linear_basis_contract(),
        "numerical_choice_rationale": numerical_choice_rationale(),
        "output_contract": output_contract(),
        "claim_boundary": claim_boundary(),
    }
    if len(water) != EXPECTED_WATER_RECORD_COUNT:
        raise StockAreaFeatureError("Frozen water count drifted.")
    payload["self_sha256"] = canonical_json_sha256(payload)
    if set(payload) != preregistration_schema_keys():
        raise StockAreaFeatureError("Creator emitted an unexpected schema.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output.chmod(0o444)
    snapshot.assert_unchanged()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--parent-hybrid-preregistration", type=Path, required=True)
    parser.add_argument("--feature-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = create(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(sha256_file(args.output.expanduser().resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
