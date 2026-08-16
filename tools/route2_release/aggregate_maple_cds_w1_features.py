#!/usr/bin/env python3
"""Aggregate the complete target-blind MAPLE-CDS-W1 design matrix.

This tool consumes only the external W1 preregistration, its 306 geometry
feature records, and the already-frozen MNSol inputs needed to recompute record
identity.  The dataset loader parses the licensed table, but no experimental
target attribute or hybrid prediction record is used.  The aggregate is a
content-addressed geometry design matrix, not a fit or accuracy result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

_SCRIPT_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SCRIPT_SOURCE_ROOT))

import numpy as np

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2,
    SMD_WATER_TENSION_PARAMETER_NAMES,
)
from maple.solvation.continuum.harmonic_cds_area import (
    POSITIVE_BERNSTEIN_HARMONIC_EXPOSURE_AREA_CONTRACT_ID,
)
from maple.solvation.continuum.harmonic_positive_exposure import (
    POSITIVE_BERNSTEIN_EXPOSURE_CONTRACT_ID,
    POSITIVE_BERNSTEIN_MAXIMUM_INTEGRAND_DEGREE,
    POSITIVE_BERNSTEIN_MAXIMUM_TRANSITION_FACTORS,
    POSITIVE_BERNSTEIN_PAIR_DEGREE,
)
from maple.solvation.harmonic_cds import (
    MAPLE_CDS_W1_POSITIVE_BERNSTEIN_PROFILE_ID,
)
from maple.solvation.release.evidence import (
    RepositorySnapshot,
    canonical_json_sha256,
    collect_loaded_repository_sources,
    committed_source_hashes,
    sha256_file,
)

from generate_maple_cds_w1_features import (
    AREA_DEVICE,
    AREA_DTYPE,
    AREA_SURFACE_LMAX,
    AREA_TRANSITION_WIDTH_ANGSTROM2,
    EXPECTED_WATER_RECORD_COUNT,
    INPUT_FILE_NAMES,
    PREREGISTRATION_ARTIFACT,
    PROHIBITED_OUTPUT_KEYS,
    REQUIRED_SOURCE_FILE_NAMES,
    _feature_payload,
    _load_selection,
    _maximum_active_pair_factors,
    _runtime_identity,
    _water_records,
)

AGGREGATE_ARTIFACT = "route2-maple-cds-w1-positive-parent-water-feature-matrix-v2"


class W1FeatureAggregationError(RuntimeError):
    """Raised when W1 geometry-feature evidence is incomplete or inconsistent."""


_PREREGISTRATION_KEYS = frozenset(
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
        "source_root",
        "source_git_head",
        "source_git_tree",
        "source_files_sha256",
        "feature_generator_sha256",
        "runtime_identity",
        "runtime_identity_sha256",
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


def _outside_source(path: Path, source_root: Path, *, name: str) -> None:
    try:
        path.relative_to(source_root)
    except ValueError:
        return
    raise W1FeatureAggregationError(f"{name} must be outside source checkout.")


def _load_object(path: Path, *, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise W1FeatureAggregationError(f"{name} must be a JSON object.")
    return payload


def _lower_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise W1FeatureAggregationError(f"{name} must be a lowercase SHA256.")
    return value


def _lower_git_object(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise W1FeatureAggregationError(f"{name} must be a full lowercase Git ID.")
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
        raise W1FeatureAggregationError(
            f"Git command {arguments!r} failed: {diagnostic.strip()}"
        )
    if binary:
        if not isinstance(
            result.stdout, bytes
        ):  # pragma: no cover - subprocess contract
            raise W1FeatureAggregationError(
                "Binary Git read unexpectedly returned text."
            )
        return result.stdout
    if not isinstance(result.stdout, str):  # pragma: no cover - subprocess contract
        raise W1FeatureAggregationError("Text Git read unexpectedly returned bytes.")
    return result.stdout.strip()


def _feature_content_sha256(payload: Mapping[str, object]) -> str:
    without_digest = dict(payload)
    observed = without_digest.pop("feature_sha256", None)
    expected = canonical_json_sha256(without_digest)
    if observed != expected:
        raise W1FeatureAggregationError("Feature content hash drifted.")
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
) -> np.ndarray:
    if path.stat().st_mode & 0o222:
        raise W1FeatureAggregationError(f"Feature {water_ordinal} is not read-only.")
    if PROHIBITED_OUTPUT_KEYS.intersection(payload):
        raise W1FeatureAggregationError(
            f"Feature {water_ordinal} contains a prohibited target key."
        )
    _feature_content_sha256(payload)
    expected_payload = _feature_payload(
        water_ordinal=water_ordinal,
        selection_index=selection_index,
        item=item,
        preregistration_sha256=preregistration_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
    )
    if dict(payload) != expected_payload:
        raise W1FeatureAggregationError(
            f"Feature {water_ordinal} does not exactly reproduce its frozen geometry."
        )
    row = np.asarray(expected_payload["design_row_angstrom2_div_1000"], dtype=float)
    return np.array(row, copy=True)


def _relative_file(root: Path, raw: object, *, name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise W1FeatureAggregationError(f"{name} must be a non-empty relative path.")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise W1FeatureAggregationError(f"{name} must stay inside its bound root.")
    resolved = (root / relative).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise W1FeatureAggregationError(
            f"{name} resolves outside its bound root."
        ) from exc
    if not resolved.is_file():
        raise W1FeatureAggregationError(f"{name} must resolve to a regular file.")
    return resolved


def _validate_preregistration(
    *,
    preregistration: Mapping[str, object],
    preregistration_path: Path,
    input_root: Path,
    records_dir: Path,
) -> tuple[str, str]:
    if set(preregistration) != _PREREGISTRATION_KEYS:
        raise W1FeatureAggregationError("W1 preregistration schema keys drifted.")
    expected_scalars = {
        "artifact": PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-first-positive-parent-feature-evaluation",
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
        "feature_output_dir": str(records_dir),
        "input_root": str(input_root),
        "preregistration_path": str(preregistration_path),
    }
    for key, expected in expected_scalars.items():
        if preregistration.get(key) != expected:
            raise W1FeatureAggregationError(f"W1 preregistration {key!r} drifted.")
    locked_at = preregistration.get("locked_at_utc")
    if not isinstance(locked_at, str) or not locked_at:
        raise W1FeatureAggregationError("W1 preregistration lock time is invalid.")
    if preregistration_path.stat().st_mode & 0o222:
        raise W1FeatureAggregationError("W1 preregistration is not read-only.")
    without_self = dict(preregistration)
    observed_self = without_self.pop("self_sha256", None)
    if observed_self != canonical_json_sha256(without_self):
        raise W1FeatureAggregationError("W1 preregistration self hash drifted.")
    preregistration_sha256 = sha256_file(preregistration_path)

    runtime_identity_sha256 = _lower_sha256(
        preregistration.get("runtime_identity_sha256"),
        name="runtime_identity_sha256",
    )
    runtime_identity, observed_runtime_sha256 = _runtime_identity()
    if preregistration.get("runtime_identity") != runtime_identity:
        raise W1FeatureAggregationError("W1 aggregation numerical runtime drifted.")
    if observed_runtime_sha256 != runtime_identity_sha256:
        raise W1FeatureAggregationError("W1 aggregation runtime digest drifted.")

    input_hashes = preregistration.get("input_files_sha256")
    if not isinstance(input_hashes, dict) or tuple(sorted(input_hashes)) != tuple(
        sorted(INPUT_FILE_NAMES)
    ):
        raise W1FeatureAggregationError("W1 input hash manifest drifted.")
    for relative in INPUT_FILE_NAMES:
        if input_hashes[relative] != sha256_file(
            _relative_file(input_root, relative, name="input file")
        ):
            raise W1FeatureAggregationError(f"W1 input file {relative!r} drifted.")

    parent_path = (
        Path(str(preregistration.get("parent_hybrid_preregistration_path")))
        .expanduser()
        .resolve(strict=True)
    )
    if preregistration.get("parent_hybrid_preregistration_sha256") != sha256_file(
        parent_path
    ):
        raise W1FeatureAggregationError("W1 parent preregistration drifted.")
    parent = _load_object(parent_path, name="parent hybrid preregistration")
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
            raise W1FeatureAggregationError(
                f"W1 parent preregistration {key!r} drifted."
            )
    if preregistration.get("parent_hybrid_source_git_head") != parent.get(
        "source_git_head"
    ):
        raise W1FeatureAggregationError("W1 parent source binding drifted.")
    if parent.get("input_root") != str(input_root):
        raise W1FeatureAggregationError("W1 parent input-root binding drifted.")
    parent_input_fields = {
        "MNSolDatabase_v2012.zip": "dataset_zip_sha256",
        "route2-mnsol-development-selection-v1.private.json": (
            "development_selection_sha256"
        ),
        "route2-mnsol-pilot-selection-v1.json": "pilot_selection_sha256",
        "route2-mnsol-protocol-v1.json": "protocol_sha256",
    }
    for relative, parent_key in parent_input_fields.items():
        if parent.get(parent_key) != input_hashes[relative]:
            raise W1FeatureAggregationError(
                f"W1 parent input hash {parent_key!r} drifted."
            )

    preregistered_sources = preregistration.get("source_files_sha256")
    preregistered_root = Path(str(preregistration.get("source_root"))).resolve(
        strict=True
    )
    source_head = _lower_git_object(
        preregistration.get("source_git_head"),
        name="source_git_head",
    )
    source_tree = _lower_git_object(
        preregistration.get("source_git_tree"),
        name="source_git_tree",
    )
    if _git(preregistered_root, "rev-parse", f"{source_head}^{{tree}}") != source_tree:
        raise W1FeatureAggregationError("W1 historical source tree drifted.")
    if not isinstance(preregistered_sources, dict) or not preregistered_sources:
        raise W1FeatureAggregationError("W1 source hash manifest is missing.")
    missing_required = sorted(
        set(REQUIRED_SOURCE_FILE_NAMES) - set(preregistered_sources)
    )
    if missing_required:
        raise W1FeatureAggregationError(
            "W1 preregistration omits required source files: "
            + ", ".join(missing_required)
        )
    for relative, expected in preregistered_sources.items():
        _lower_sha256(expected, name=f"source_files_sha256[{relative!r}]")
        path = _relative_file(preregistered_root, relative, name="source file")
        historical_bytes = _git(
            preregistered_root,
            "show",
            f"{source_head}:{relative}",
            binary=True,
        )
        historical_sha256 = hashlib.sha256(historical_bytes).hexdigest()
        if expected != historical_sha256 or expected != sha256_file(path):
            raise W1FeatureAggregationError(
                f"W1 preregistered source {relative!r} drifted."
            )
    generator_relative = "tools/route2_release/generate_maple_cds_w1_features.py"
    if preregistration.get("feature_generator_sha256") != preregistered_sources.get(
        generator_relative
    ):
        raise W1FeatureAggregationError("W1 feature-generator binding drifted.")

    expected_area = {
        "contract_id": POSITIVE_BERNSTEIN_HARMONIC_EXPOSURE_AREA_CONTRACT_ID,
        "exposure_contract_id": POSITIVE_BERNSTEIN_EXPOSURE_CONTRACT_ID,
        "profile_id": MAPLE_CDS_W1_POSITIVE_BERNSTEIN_PROFILE_ID,
        "radii": "published-smd-sasa-radii-including-0.4-A-probe",
        "transition_width_angstrom2": AREA_TRANSITION_WIDTH_ANGSTROM2,
        "surface_lmax": AREA_SURFACE_LMAX,
        "retained_parent_moment_lmax": 2 * AREA_SURFACE_LMAX,
        "positive_parent_pair_degree": POSITIVE_BERNSTEIN_PAIR_DEGREE,
        "maximum_transition_factors": (POSITIVE_BERNSTEIN_MAXIMUM_TRANSITION_FACTORS),
        "maximum_integrand_degree": POSITIVE_BERNSTEIN_MAXIMUM_INTEGRAND_DEGREE,
        "dtype": AREA_DTYPE,
        "device": AREA_DEVICE,
        "area_measure": "a_i^2-integral-e_i-domega",
        "reconstructed_low_band_used_as_mask": False,
    }
    if preregistration.get("area_definition") != expected_area:
        raise W1FeatureAggregationError("W1 area definition drifted.")
    expected_stock = SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2.tolist()
    expected_linear_basis = {
        "contract": "published-aqueous-smd-linear-18-column-v1",
        "parameter_names": list(SMD_WATER_TENSION_PARAMETER_NAMES),
        "stock_coefficients_cal_mol_angstrom2": expected_stock,
        "stock_coefficients_sha256": canonical_json_sha256(expected_stock),
    }
    if preregistration.get("linear_basis") != expected_linear_basis:
        raise W1FeatureAggregationError("W1 linear basis drifted.")
    expected_output_contract = {
        "one_exclusive_json_per_water_record": True,
        "coordinates_emitted": False,
        "experimental_targets_emitted": False,
        "hybrid_outputs_emitted": False,
        "fit_or_calibration_emitted": False,
    }
    if preregistration.get("output_contract") != expected_output_contract:
        raise W1FeatureAggregationError("W1 output contract drifted.")
    return preregistration_sha256, runtime_identity_sha256


def _write_external_json_exclusive(
    snapshot: RepositorySnapshot,
    path: Path,
    payload: object,
) -> None:
    _outside_source(path, snapshot.root, name="aggregate output")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x") as handle:
        handle.write(serialized)
    path.chmod(0o444)


def _matrix_diagnostics(matrix: np.ndarray) -> dict[str, object]:
    values = np.asarray(matrix, dtype=float)
    if values.shape != (EXPECTED_WATER_RECORD_COUNT, 18) or not np.all(
        np.isfinite(values)
    ):
        raise W1FeatureAggregationError("Complete W1 design matrix is invalid.")
    singular_values = np.linalg.svd(values, compute_uv=False)
    tolerance = float(
        max(values.shape) * np.finfo(values.dtype).eps * singular_values[0]
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == values.shape[1]
        else None
    )
    support_tolerance = 64.0 * np.finfo(values.dtype).eps
    return {
        "shape": list(values.shape),
        "rank": rank,
        "rank_tolerance": tolerance,
        "singular_values": singular_values.tolist(),
        "condition_number_if_full_rank": condition,
        "nonzero_record_count_by_column": np.count_nonzero(
            np.abs(values) > support_tolerance,
            axis=0,
        ).tolist(),
        "column_l2_norm": np.linalg.norm(values, axis=0).tolist(),
        "claim_boundary": (
            "Target-free identifiability diagnostic only; rank/support do not "
            "select coefficients, regularization, or a solvation model."
        ),
    }


def aggregate(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    if source_root != _SCRIPT_SOURCE_ROOT:
        raise W1FeatureAggregationError(
            "Feature aggregator must execute for its own source checkout."
        )
    input_root = args.input_root.expanduser().resolve(strict=True)
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    records_dir = args.records_dir.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    _outside_source(output, source_root, name="aggregate output")
    snapshot = RepositorySnapshot.capture(source_root)
    preregistration = _load_object(preregistration_path, name="W1 preregistration")
    preregistration_sha256, runtime_identity_sha256 = _validate_preregistration(
        preregistration=preregistration,
        preregistration_path=preregistration_path,
        input_root=input_root,
        records_dir=records_dir,
    )

    actual_names = sorted(path.name for path in records_dir.iterdir() if path.is_file())
    expected_names = [
        f"water-{ordinal:03d}.json" for ordinal in range(EXPECTED_WATER_RECORD_COUNT)
    ]
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))
        extra = sorted(set(actual_names) - set(expected_names))
        raise W1FeatureAggregationError(
            f"W1 feature matrix is incomplete; missing={missing}, extra={extra}."
        )

    _protocol, rows = _load_selection(source_root=source_root, input_root=input_root)
    water, water_identity_sha256 = _water_records(rows)
    if len(water) != EXPECTED_WATER_RECORD_COUNT:
        raise W1FeatureAggregationError("W1 selected water count drifted.")
    if preregistration.get("water_identity_sha256") != water_identity_sha256:
        raise W1FeatureAggregationError("W1 water identity binding drifted.")
    maximum_active = _maximum_active_pair_factors(water)
    finite_product_degree = (
        POSITIVE_BERNSTEIN_PAIR_DEGREE * maximum_active + 2 * AREA_SURFACE_LMAX
    )
    expected_rationale = {
        "maximum_observed_active_pair_factors": maximum_active,
        "finite_product_degree_formula": (
            "pair_degree*active_pair_factors+2*surface_lmax"
        ),
        "positive_parent_pair_degree": POSITIVE_BERNSTEIN_PAIR_DEGREE,
        "maximum_integrand_degree": finite_product_degree,
        "maximum_transition_factor_cap": (
            POSITIVE_BERNSTEIN_MAXIMUM_TRANSITION_FACTORS
        ),
        "implementation_degree_limit": (POSITIVE_BERNSTEIN_MAXIMUM_INTEGRAND_DEGREE),
        "selected_from_geometry_only": True,
        "accuracy_results_used": False,
    }
    if preregistration.get("numerical_choice_rationale") != expected_rationale:
        raise W1FeatureAggregationError("W1 numerical-choice rationale drifted.")
    if (
        maximum_active > POSITIVE_BERNSTEIN_MAXIMUM_TRANSITION_FACTORS
        or finite_product_degree > POSITIVE_BERNSTEIN_MAXIMUM_INTEGRAND_DEGREE
    ):
        raise W1FeatureAggregationError(
            "W1 positive parent exceeds the frozen structural bound."
        )
    matrix_rows: list[np.ndarray] = []
    record_rows: list[dict[str, object]] = []
    feature_file_sha256: dict[str, str] = {}
    for water_ordinal, (selection_index, item) in enumerate(water):
        path = records_dir / expected_names[water_ordinal]
        payload = _load_object(path, name=f"feature {water_ordinal}")
        row = _validate_feature_record(
            payload=payload,
            path=path,
            water_ordinal=water_ordinal,
            selection_index=selection_index,
            item=item,
            preregistration_sha256=preregistration_sha256,
            runtime_identity_sha256=runtime_identity_sha256,
        )
        matrix_rows.append(row)
        feature_file_sha256[path.name] = sha256_file(path)
        record_rows.append(
            {
                "water_ordinal": water_ordinal,
                "selection_index": selection_index,
                "opaque_record_id": item.opaque_record_id,
                "geometry_sha256": item.eligible_record.geometry.sha256,
                "design_row_angstrom2_div_1000": row.tolist(),
                "stock_smd_control_kcal_mol": float(
                    row @ SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2
                ),
                "feature_sha256": payload["feature_sha256"],
                "feature_file_sha256": feature_file_sha256[path.name],
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
            "tools/route2_release/aggregate_maple_cds_w1_features.py",
            "tools/route2_release/generate_maple_cds_w1_features.py",
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
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "preregistered_feature_source_git_head": preregistration["source_git_head"],
        "aggregation_source_git_head": snapshot.head,
        "aggregation_source_git_tree": snapshot.tree,
        "aggregation_source_files_sha256": committed_source_hashes(
            snapshot,
            loaded_sources,
        ),
        "runtime_identity_sha256": runtime_identity_sha256,
        "water_identity_sha256": water_identity_sha256,
        "design_parameter_names": list(SMD_WATER_TENSION_PARAMETER_NAMES),
        "records": record_rows,
        "matrix_sha256": canonical_json_sha256(matrix.tolist()),
        "feature_files_sha256": feature_file_sha256,
        "matrix_diagnostics": _matrix_diagnostics(matrix),
        "claim_boundary": (
            "Complete target-blind W1 geometry design matrix. It is not a fit, "
            "an accuracy result, a CDS admission, or a Route-2 capability."
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
