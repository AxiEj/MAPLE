#!/usr/bin/env python3
"""Aggregate the complete target-blind stock-area MAPLE-CDS-W1 matrix.

The licensed MNSol loader parses the frozen development table only to recover
and validate geometry identities.  This tool never accesses target attributes
or hybrid prediction records.  It regenerates every stock-area feature from
the frozen geometry before accepting it, then emits only target-free matrix
rank, conditioning, support, and PySCF parity diagnostics.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

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
    PROHIBITED_OUTPUT_KEYS,
    REQUIRED_SOURCE_FILE_NAMES,
    SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2,
    SMD_WATER_TENSION_PARAMETER_NAMES,
    StockAreaFeatureError,
    _canonical_sha256,
    _feature_payload,
    _load_json_object,
    _load_selection,
    _pyscf_runtime_assets,
    _runtime_identity,
    _sha256,
    _validate_parent_input_hashes,
    _water_records,
    claim_boundary,
    linear_basis_contract,
    numerical_choice_rationale,
    output_contract,
    preregistration_schema_keys,
    stock_area_contract,
)

AGGREGATE_ARTIFACT = "route2-maple-cds-w1-water-stock-area-matrix-v1"


class StockAreaAggregationError(RuntimeError):
    """Raised when stock-area feature evidence is incomplete or inconsistent."""


def _outside_source(path: Path, source_root: Path, *, name: str) -> None:
    try:
        path.relative_to(source_root)
    except ValueError:
        return
    raise StockAreaAggregationError(f"{name} must be outside source checkout.")


def _load_object(path: Path, *, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise StockAreaAggregationError(f"{name} must be a JSON object.")
    return payload


def _lower_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StockAreaAggregationError(f"{name} must be a lowercase SHA256.")
    return value


def _lower_git_object(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StockAreaAggregationError(f"{name} must be a full lowercase Git ID.")
    return value


def _git(root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=not binary,
    )
    if result.returncode != 0:
        diagnostic = result.stderr if binary else (result.stderr or result.stdout)
        if isinstance(diagnostic, bytes):
            diagnostic = diagnostic.decode("utf-8", errors="replace")
        raise StockAreaAggregationError(
            f"Git command {arguments!r} failed: {diagnostic.strip()}"
        )
    if binary:
        if not isinstance(result.stdout, bytes):  # pragma: no cover
            raise StockAreaAggregationError("Binary Git read returned text.")
        return result.stdout
    if not isinstance(result.stdout, str):  # pragma: no cover
        raise StockAreaAggregationError("Text Git read returned bytes.")
    return result.stdout.strip()


def _relative_file(root: Path, raw: object, *, name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise StockAreaAggregationError(f"{name} must be a non-empty relative path.")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise StockAreaAggregationError(f"{name} must stay inside its bound root.")
    resolved = (root / relative).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise StockAreaAggregationError(
            f"{name} resolves outside its bound root."
        ) from exc
    if not resolved.is_file():
        raise StockAreaAggregationError(f"{name} must resolve to a regular file.")
    return resolved


def _feature_content_sha256(payload: Mapping[str, object]) -> str:
    without_digest = dict(payload)
    observed = without_digest.pop("feature_sha256", None)
    expected = _canonical_sha256(without_digest)
    if observed != expected:
        raise StockAreaAggregationError("Feature content hash drifted.")
    return expected


def _validate_feature_record(
    *,
    payload: Mapping[str, object],
    path: Path,
    water_ordinal: int,
    selection_index: int,
    item: object,
    preregistration_sha256: str,
    runtime_identity_sha256: str,
) -> tuple[np.ndarray, dict[str, float]]:
    if path.stat().st_mode & 0o222:
        raise StockAreaAggregationError(f"Feature {water_ordinal} is not read-only.")
    if PROHIBITED_OUTPUT_KEYS.intersection(payload):
        raise StockAreaAggregationError(
            f"Feature {water_ordinal} contains a prohibited target key."
        )
    _feature_content_sha256(payload)
    expected = _feature_payload(
        water_ordinal=water_ordinal,
        selection_index=selection_index,
        item=item,
        preregistration_sha256=preregistration_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
    )
    if dict(payload) != expected:
        raise StockAreaAggregationError(
            f"Feature {water_ordinal} does not exactly reproduce its frozen geometry."
        )
    row = np.asarray(expected["design_row_angstrom2_div_1000"], dtype=float)
    if row.shape != (len(SMD_WATER_TENSION_PARAMETER_NAMES),) or not np.all(
        np.isfinite(row)
    ):
        raise StockAreaAggregationError(f"Feature {water_ordinal} row is invalid.")
    controls = {
        key: float(expected[key])
        for key in (
            "stock_minus_compiled_legacy_kcal_mol",
            "rotation_control_stock_error_kcal_mol",
            "rotation_control_legacy_error_kcal_mol",
            "rotated_stock_minus_compiled_legacy_kcal_mol",
        )
    }
    if not all(np.isfinite(tuple(controls.values()))):
        raise StockAreaAggregationError(
            f"Feature {water_ordinal} control is non-finite."
        )
    return np.array(row, copy=True), controls


def _validate_preregistration(
    *,
    preregistration: Mapping[str, object],
    preregistration_path: Path,
    source_root: Path,
    input_root: Path,
    records_dir: Path,
) -> tuple[str, str]:
    if set(preregistration) != preregistration_schema_keys():
        raise StockAreaAggregationError("Stock-area preregistration schema drifted.")
    expected = {
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
        "source_root": str(source_root),
        "input_root": str(input_root),
        "feature_output_dir": str(records_dir),
        "preregistration_path": str(preregistration_path),
        "claim_boundary": claim_boundary(),
    }
    for key, value in expected.items():
        if preregistration.get(key) != value:
            raise StockAreaAggregationError(
                f"Stock-area preregistration {key!r} drifted."
            )
    locked_at = preregistration.get("locked_at_utc")
    if not isinstance(locked_at, str):
        raise StockAreaAggregationError("Preregistration lock time is invalid.")
    try:
        parsed_locked_at = datetime.fromisoformat(locked_at)
    except ValueError as exc:
        raise StockAreaAggregationError(
            "Preregistration lock time is invalid."
        ) from exc
    if parsed_locked_at.tzinfo is None:
        raise StockAreaAggregationError("Preregistration lock time lacks timezone.")
    if preregistration_path.stat().st_mode & 0o222:
        raise StockAreaAggregationError("Preregistration is not read-only.")
    without_self = dict(preregistration)
    observed_self = without_self.pop("self_sha256", None)
    if observed_self != _canonical_sha256(without_self):
        raise StockAreaAggregationError("Preregistration self hash drifted.")
    preregistration_sha256 = _sha256(preregistration_path)

    runtime_identity_sha256 = _lower_sha256(
        preregistration.get("runtime_identity_sha256"),
        name="runtime_identity_sha256",
    )
    runtime_identity, observed_runtime_sha256 = _runtime_identity()
    if preregistration.get("runtime_identity") != runtime_identity:
        raise StockAreaAggregationError("Aggregation numerical runtime drifted.")
    if observed_runtime_sha256 != runtime_identity_sha256:
        raise StockAreaAggregationError("Aggregation runtime digest drifted.")
    pyscf_assets, pyscf_assets_sha256 = _pyscf_runtime_assets()
    if preregistration.get("pyscf_runtime_assets") != pyscf_assets:
        raise StockAreaAggregationError("PySCF runtime assets drifted.")
    if preregistration.get("pyscf_runtime_assets_sha256") != pyscf_assets_sha256:
        raise StockAreaAggregationError("PySCF runtime asset digest drifted.")

    input_hashes = preregistration.get("input_files_sha256")
    if not isinstance(input_hashes, dict) or set(input_hashes) != set(INPUT_FILE_NAMES):
        raise StockAreaAggregationError("Frozen input hash manifest drifted.")
    for relative in INPUT_FILE_NAMES:
        if input_hashes[relative] != sha256_file(
            _relative_file(input_root, relative, name="input file")
        ):
            raise StockAreaAggregationError(f"Input file {relative!r} drifted.")

    parent_path = (
        Path(str(preregistration.get("parent_hybrid_preregistration_path")))
        .expanduser()
        .resolve(strict=True)
    )
    if parent_path.stat().st_mode & 0o222:
        raise StockAreaAggregationError("Parent preregistration is not read-only.")
    if preregistration.get("parent_hybrid_preregistration_sha256") != sha256_file(
        parent_path
    ):
        raise StockAreaAggregationError("Parent preregistration drifted.")
    parent = _load_object(parent_path, name="parent hybrid preregistration")
    for key, value in {
        "artifact_id": PARENT_ARTIFACT,
        "schema_version": 3,
        "status": "locked-before-first-v3-hybrid-evaluation",
        "partition": "development",
        "record_count": 505,
        "confirmation_partition_opened": False,
        "fitting_or_calibration_permitted": False,
        "input_root": str(input_root),
    }.items():
        if parent.get(key) != value:
            raise StockAreaAggregationError(f"Parent field {key!r} drifted.")
    if preregistration.get("parent_hybrid_source_git_head") != parent.get(
        "source_git_head"
    ):
        raise StockAreaAggregationError("Parent source binding drifted.")
    try:
        _validate_parent_input_hashes(parent, input_hashes)
    except StockAreaFeatureError as exc:
        raise StockAreaAggregationError("Parent frozen input binding drifted.") from exc

    source_head = _lower_git_object(
        preregistration.get("source_git_head"), name="source_git_head"
    )
    source_tree = _lower_git_object(
        preregistration.get("source_git_tree"), name="source_git_tree"
    )
    if _git(source_root, "rev-parse", f"{source_head}^{{tree}}") != source_tree:
        raise StockAreaAggregationError("Historical source tree drifted.")
    source_hashes = preregistration.get("source_files_sha256")
    if not isinstance(source_hashes, dict) or not set(
        REQUIRED_SOURCE_FILE_NAMES
    ).issubset(source_hashes):
        raise StockAreaAggregationError("Preregistered source manifest is incomplete.")
    for relative, value in source_hashes.items():
        expected_sha256 = _lower_sha256(
            value, name=f"source_files_sha256[{relative!r}]"
        )
        historical = _git(
            source_root,
            "show",
            f"{source_head}:{relative}",
            binary=True,
        )
        if not isinstance(historical, bytes):  # pragma: no cover
            raise StockAreaAggregationError("Historical source read returned text.")
        historical_sha256 = hashlib.sha256(historical).hexdigest()
        current_sha256 = sha256_file(
            _relative_file(source_root, relative, name="source file")
        )
        if expected_sha256 != historical_sha256 or expected_sha256 != current_sha256:
            raise StockAreaAggregationError(
                f"Preregistered source {relative!r} drifted."
            )
    generator_relative = (
        "tools/route2_release/generate_maple_cds_w1_stock_area_features.py"
    )
    if preregistration.get("feature_generator_sha256") != source_hashes.get(
        generator_relative
    ):
        raise StockAreaAggregationError("Feature-generator binding drifted.")
    if preregistration.get("area_definition") != stock_area_contract():
        raise StockAreaAggregationError("Stock-area definition drifted.")
    if preregistration.get("linear_basis") != linear_basis_contract():
        raise StockAreaAggregationError("Linear basis drifted.")
    if (
        preregistration.get("numerical_choice_rationale")
        != numerical_choice_rationale()
    ):
        raise StockAreaAggregationError("Numerical-choice rationale drifted.")
    if preregistration.get("output_contract") != output_contract():
        raise StockAreaAggregationError("Output contract drifted.")
    return preregistration_sha256, runtime_identity_sha256


def _matrix_diagnostics(matrix: np.ndarray) -> dict[str, object]:
    values = np.asarray(matrix, dtype=float)
    expected_shape = (
        EXPECTED_WATER_RECORD_COUNT,
        len(SMD_WATER_TENSION_PARAMETER_NAMES),
    )
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        raise StockAreaAggregationError("Complete stock-area matrix is invalid.")
    singular_values = np.linalg.svd(values, compute_uv=False)
    tolerance = float(
        max(values.shape) * np.finfo(values.dtype).eps * singular_values[0]
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    support_tolerance = 64.0 * np.finfo(values.dtype).eps
    support = np.count_nonzero(np.abs(values) > support_tolerance, axis=0)
    norms = np.linalg.norm(values, axis=0)
    active = norms > support_tolerance
    if not np.any(active):
        raise StockAreaAggregationError("Stock-area matrix has no active columns.")
    standardized = values[:, active] / norms[active]
    standardized_singular_values = np.linalg.svd(standardized, compute_uv=False)
    standardized_tolerance = float(
        max(standardized.shape)
        * np.finfo(values.dtype).eps
        * standardized_singular_values[0]
    )
    standardized_rank = int(
        np.count_nonzero(standardized_singular_values > standardized_tolerance)
    )
    standardized_condition = (
        float(standardized_singular_values[0] / standardized_singular_values[-1])
        if standardized_rank == standardized.shape[1]
        else None
    )
    return {
        "shape": list(values.shape),
        "raw_rank": rank,
        "raw_rank_tolerance": tolerance,
        "raw_singular_values": singular_values.tolist(),
        "nonzero_record_count_by_column": support.tolist(),
        "column_l2_norm": norms.tolist(),
        "active_column_indices": np.flatnonzero(active).tolist(),
        "standardization": "divide-each-nonzero-column-by-target-blind-l2-norm",
        "standardized_rank": standardized_rank,
        "standardized_rank_tolerance": standardized_tolerance,
        "standardized_singular_values": standardized_singular_values.tolist(),
        "standardized_condition_number_if_full_rank": standardized_condition,
        "claim_boundary": (
            "Target-free identifiability diagnostic only; rank, support, and "
            "conditioning do not select coefficients or fit a model."
        ),
    }


def _control_diagnostics(rows: list[Mapping[str, float]]) -> dict[str, object]:
    if not rows:
        raise StockAreaAggregationError("No stock-area controls were provided.")
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows):
        raise StockAreaAggregationError("Stock-area control schema drifted.")
    result: dict[str, object] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        if not np.all(np.isfinite(values)):
            raise StockAreaAggregationError(f"Control {key!r} is non-finite.")
        result[key] = {
            "maximum_absolute_kcal_mol": float(np.max(np.abs(values))),
            "rms_kcal_mol": float(np.sqrt(np.mean(values**2))),
            "mean_kcal_mol": float(np.mean(values)),
        }
    return result


def _write_external_json_exclusive(
    snapshot: RepositorySnapshot, path: Path, payload: object
) -> None:
    _outside_source(path, snapshot.root, name="aggregate output")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x") as handle:
        handle.write(serialized)
    path.chmod(0o444)


def aggregate(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    if source_root != _SCRIPT_SOURCE_ROOT:
        raise StockAreaAggregationError(
            "Stock-area aggregator must execute for its own source checkout."
        )
    input_root = args.input_root.expanduser().resolve(strict=True)
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    records_dir = args.records_dir.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    _outside_source(output, source_root, name="aggregate output")
    snapshot = RepositorySnapshot.capture(source_root)
    preregistration = _load_json_object(
        preregistration_path, name="stock-area preregistration"
    )
    preregistration_sha256, runtime_identity_sha256 = _validate_preregistration(
        preregistration=preregistration,
        preregistration_path=preregistration_path,
        source_root=source_root,
        input_root=input_root,
        records_dir=records_dir,
    )

    expected_names = [
        f"water-{ordinal:03d}.json" for ordinal in range(EXPECTED_WATER_RECORD_COUNT)
    ]
    actual_names = sorted(path.name for path in records_dir.iterdir() if path.is_file())
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))
        extra = sorted(set(actual_names) - set(expected_names))
        raise StockAreaAggregationError(
            f"Stock-area matrix is incomplete; missing={missing}, extra={extra}."
        )

    _protocol, rows = _load_selection(source_root=source_root, input_root=input_root)
    water, water_identity_sha256 = _water_records(rows)
    if len(water) != EXPECTED_WATER_RECORD_COUNT:
        raise StockAreaAggregationError("Frozen water count drifted.")
    if preregistration.get("water_identity_sha256") != water_identity_sha256:
        raise StockAreaAggregationError("Frozen water identity drifted.")

    matrix_rows: list[np.ndarray] = []
    control_rows: list[dict[str, float]] = []
    record_rows: list[dict[str, object]] = []
    feature_files_sha256: dict[str, str] = {}
    for water_ordinal, (selection_index, item) in enumerate(water):
        path = records_dir / expected_names[water_ordinal]
        payload = _load_object(path, name=f"feature {water_ordinal}")
        row, controls = _validate_feature_record(
            payload=payload,
            path=path,
            water_ordinal=water_ordinal,
            selection_index=selection_index,
            item=item,
            preregistration_sha256=preregistration_sha256,
            runtime_identity_sha256=runtime_identity_sha256,
        )
        matrix_rows.append(row)
        control_rows.append(controls)
        feature_files_sha256[path.name] = sha256_file(path)
        record_rows.append(
            {
                "water_ordinal": water_ordinal,
                "selection_index": selection_index,
                "opaque_record_id": item.opaque_record_id,
                "geometry_sha256": item.eligible_record.geometry.sha256,
                "design_row_angstrom2_div_1000": row.tolist(),
                "stock_smd_reconstruction_kcal_mol": float(
                    row @ SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2
                ),
                "feature_sha256": payload["feature_sha256"],
                "feature_file_sha256": feature_files_sha256[path.name],
            }
        )
    matrix = np.stack(matrix_rows, axis=0)
    loaded_sources = collect_loaded_repository_sources(
        source_root,
        required_paths=(
            "docs/implicit-solvation/benchmarks/benchmark_core.py",
            "docs/implicit-solvation/benchmarks/mnsol_dataset.py",
            "docs/implicit-solvation/benchmarks/mnsol_partition.py",
            "docs/implicit-solvation/benchmarks/mnsol_pilot.py",
            "tools/route2_release/aggregate_maple_cds_w1_stock_area_features.py",
            "tools/route2_release/generate_maple_cds_w1_stock_area_features.py",
        ),
    )
    payload: dict[str, object] = {
        "artifact": AGGREGATE_ARTIFACT,
        "schema_version": 1,
        "status": "complete",
        "do_not_commit": True,
        "record_count": len(record_rows),
        "partition": "development-water-only",
        "dataset_loader_parsed_experimental_targets": True,
        "experimental_targets_used_by_feature_aggregation": False,
        "experimental_targets_emitted": False,
        "hybrid_prediction_records_read": False,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_selected_or_emitted": False,
        "fitting_or_calibration_performed": False,
        "energy_only_diagnostic": True,
        "force_capability": False,
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "preregistered_feature_source_git_head": preregistration["source_git_head"],
        "aggregation_source_git_head": snapshot.head,
        "aggregation_source_git_tree": snapshot.tree,
        "aggregation_source_files_sha256": committed_source_hashes(
            snapshot, loaded_sources
        ),
        "runtime_identity_sha256": runtime_identity_sha256,
        "water_identity_sha256": water_identity_sha256,
        "design_parameter_names": list(SMD_WATER_TENSION_PARAMETER_NAMES),
        "records": record_rows,
        "matrix_sha256": canonical_json_sha256(matrix.tolist()),
        "feature_files_sha256": feature_files_sha256,
        "matrix_diagnostics": _matrix_diagnostics(matrix),
        "control_diagnostics": _control_diagnostics(control_rows),
        "claim_boundary": (
            "Complete target-blind stock-area geometry design matrix. It is not "
            "a fit, an accuracy result, a force-capable area model, a CDS "
            "admission, or a Route-2 public capability."
        ),
    }
    payload["aggregate_sha256"] = canonical_json_sha256(payload)
    _write_external_json_exclusive(snapshot, output, payload)
    snapshot.assert_unchanged()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate(args)
    print(
        json.dumps(
            {
                "artifact": payload["artifact"],
                "status": payload["status"],
                "record_count": payload["record_count"],
                "matrix_sha256": payload["matrix_sha256"],
                "aggregate_sha256": payload["aggregate_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
