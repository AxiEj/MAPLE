#!/usr/bin/env python3
"""Freeze the target-blind three-parameter MAPLE-CDS-W1 M3 candidate.

This creator consumes only the immutable stock-area geometry design matrix,
the frozen development-selection identities, and geometry members of the
licensed MNSol archive.  It never opens the MNSol target table, hybrid
prediction records, or confirmation data.  The resulting artifact freezes:

* a unit-reparameterization-invariant 18 -> 3 coefficient subspace;
* deterministic target-blind molecular families and ten OOF folds;
* the later LAD numerical and admission contracts before any M3 target use.

M3 remains a development-only energy candidate.  This preregistration neither
fits coefficients nor admits a production CDS surface or force.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from zipfile import ZipFile, ZipInfo

import numpy as np

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))
_BENCHMARK_ROOT = _SOURCE_ROOT / "docs/implicit-solvation/benchmarks"
sys.path.insert(0, str(_BENCHMARK_ROOT))

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2,
    SMD_WATER_TENSION_PARAMETER_NAMES,
)
from maple.solvation.release.evidence import (
    RepositorySnapshot,
    canonical_json_sha256,
    committed_source_hashes,
    runtime_record,
    sha256_file,
)
from mnsol_dataset import (
    MNSOL_MAX_GEOMETRY_BYTES,
    MNSOL_MAX_GEOMETRY_COUNT,
    MNSOL_MAX_SOURCE_ARCHIVE_BYTES,
    MNSOL_MAX_TOTAL_GEOMETRY_BYTES,
    MNSOL_MAX_ZIP_COMPRESSION_RATIO,
    MNSOL_MAX_ZIP_MEMBERS,
    MNSOL_MAX_ZIP_UNCOMPRESSED_BYTES,
    MNSolGeometry,
    _parse_geometry,
)

PREREGISTRATION_ARTIFACT = "route2-maple-cds-w1-m3-prereg-v1"
STOCK_AGGREGATE_ARTIFACT = "route2-maple-cds-w1-water-stock-area-matrix-v1"
STOCK_PREREGISTRATION_ARTIFACT = (
    "route2-maple-cds-w1-water-stock-area-features-prereg-v1"
)
CANDIDATE_PROFILE_ID = "maple-cds-w1-stock-area-target-blind-3d-lad-water-v1"
EXPECTED_WATER_COUNT = 306
EXPECTED_PARAMETER_COUNT = 18
OOF_FOLD_COUNT = 10

PRIMARY_MAE_THRESHOLD_KCAL_MOL = 1.5
MINIMUM_MAE_IMPROVEMENT_KCAL_MOL = 0.05
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_816
HIGHS_PRIMAL_FEASIBILITY_TOLERANCE = 1.0e-9
HIGHS_DUAL_FEASIBILITY_TOLERANCE = 1.0e-9
LAD_OBJECTIVE_CAP_SCALE = 1.0e-8
LAD_COEFFICIENT_RANGE_RELATIVE_TOLERANCE = 1.0e-6

_IDENTITY_ABSOLUTE_TOLERANCE = 2.0e-12
_SINGULAR_GAP_RELATIVE_TOLERANCE = 1.0e-10

_SOURCE_FILES = (
    "docs/implicit-solvation/benchmarks/mnsol_dataset.py",
    "docs/implicit-solvation/benchmarks/route2-mnsol-protocol-v1.json",
    "docs/route2/MAPLE_CDS_W1.md",
    "docs/route2/evidence/M3_PRO_AUDIT_2026-08-16.md",
    "maple/function/calculator/extra_correction/implicit/smd_cds.py",
    "maple/solvation/release/evidence.py",
    "tests/route2_vnext/test_maple_cds_w1_m3_preregistration.py",
    "tools/route2_release/aggregate_maple_cds_w1_stock_area_features.py",
    "tools/route2_release/create_maple_cds_w1_m3_preregistration.py",
    "tools/route2_release/fit_maple_cds_w1_m3_lad.py",
)

_FROZEN_INPUT_FILE_NAMES = (
    "MNSolDatabase_v2012.zip",
    "route2-mnsol-development-selection-v1.private.json",
    "route2-mnsol-pilot-selection-v1.json",
    "route2-mnsol-protocol-v1.json",
)

_STOCK_RECORD_KEYS = frozenset(
    {
        "water_ordinal",
        "selection_index",
        "opaque_record_id",
        "geometry_sha256",
        "design_row_angstrom2_div_1000",
        "stock_smd_reconstruction_kcal_mol",
        "feature_sha256",
        "feature_file_sha256",
    }
)

PREREGISTRATION_KEYS = frozenset(
    {
        "artifact",
        "schema_version",
        "status",
        "locked_at_utc",
        "candidate_profile_id",
        "partition",
        "water_record_count",
        "source_root",
        "source_git_head",
        "source_git_tree",
        "source_files_sha256",
        "runtime_identity",
        "runtime_identity_sha256",
        "input_root",
        "input_files_sha256",
        "stock_aggregate_path",
        "stock_aggregate_file_sha256",
        "stock_aggregate_self_sha256",
        "stock_matrix_sha256",
        "stock_preregistration_path",
        "stock_preregistration_file_sha256",
        "parent_hybrid_preregistration_path",
        "parent_hybrid_preregistration_file_sha256",
        "parent_hybrid_records_path",
        "parent_hybrid_evidence_root",
        "water_identity_sha256",
        "experimental_targets_read_by_m3_preregistration",
        "experimental_targets_used_by_subspace_or_folds",
        "mnsol_target_table_member_opened_by_m3_preregistration",
        "hybrid_prediction_records_read_by_m3_preregistration",
        "confirmation_selection_manifest_opened",
        "confirmation_records_opened",
        "fitting_or_calibration_performed",
        "prior_development_information_disclosed",
        "external_pro_math_reviews",
        "subspace",
        "grouped_oof",
        "fit_contract",
        "development_decision_rule",
        "deployment_policy",
        "nonaqueous_policy",
        "confirmation_policy",
        "claim_boundary",
        "self_sha256",
    }
)

_PRO_REVIEWS = (
    {
        "topic": "unit-invariant-target-blind-three-dimensional-subspace",
        "answer_sha256": (
            "abe858853901b412f8f57a8851bf6f3fa691e1f9eaf31d49caa6115e510dc2df"
        ),
        "local_verdict": "accepted-after-independent-numerical-rederivation",
    },
    {
        "topic": "practical-target-blind-grouping-and-scipy-lad-critique",
        "answer_sha256": (
            "06aa532bdfa064fe5bd4b65a0dbafc9f57968b8bb9874dbd4d32ad131f3b88b1"
        ),
        "local_verdict": (
            "partially-accepted-single-link-family-rule-rejected-by-"
            "176-of-306-giant-component-canary"
        ),
    },
    {
        "topic": "giant-component-counterexample-and-final-family-choice",
        "answer_sha256": (
            "b72cd200c8e72a99bf500bf5c527485ef24378d2239d8db48e1f41714976b590"
        ),
        "local_verdict": (
            "accepted-use-exact-heavy-element-count-equivalence-with-narrow-"
            "non-scaffold-claim"
        ),
    },
    {
        "topic": "lad-numerical-point-identification-policy",
        "answer_sha256": (
            "42db96fbb32daba444a2eacffa011733a79695ca1576b8966d62f175a264a414"
        ),
        "local_verdict": (
            "accepted-use-primary-lad-only-and-reject-numerically-nonunique-folds"
        ),
    },
)


class M3PreregistrationError(RuntimeError):
    """Raised when the target-blind M3 contract cannot be frozen."""


def frozen_fit_contract() -> dict[str, object]:
    """Return the exact target-visible M3 estimator contract."""

    return {
        "model": (
            "continuum_polarization_kcal_mol + X @ " "(alpha*theta_stock + W@beta)"
        ),
        "parameter_count": 3,
        "intercept": False,
        "validation": "ten-fold-family-grouped-out-of-fold",
        "training_scaling": "per-fold-target-blind-column-l2",
        "response_scaling": "divide-by-max(1,max-absolute-training-response)",
        "objective": "unweighted-least-absolute-deviation",
        "solver": "scipy.optimize.linprog-method-highs-ds",
        "highs_primal_feasibility_tolerance": HIGHS_PRIMAL_FEASIBILITY_TOLERANCE,
        "highs_dual_feasibility_tolerance": HIGHS_DUAL_FEASIBILITY_TOLERANCE,
        "objective_cap": "recomputed_L1 + 1e-8*max(recomputed_L1,training_count)",
        "objective_cap_scale": LAD_OBJECTIVE_CAP_SCALE,
        "nonuniqueness_policy": (
            "primary-LAD-only; six min/max scaled-coefficient LPs inside "
            "the objective cap; reject any fold whose coefficient range "
            "exceeds the frozen relative tolerance"
        ),
        "coefficient_range_relative_tolerance": (
            LAD_COEFFICIENT_RANGE_RELATIVE_TOLERANCE
        ),
        "secondary_or_anchor_objective": False,
        "standard_state": "1M-ideal-gas-to-1M-ideal-solution",
        "standard_state_correction_kcal_mol": 0.0,
    }


def frozen_development_decision_rule() -> dict[str, object]:
    """Return the exact development-only pass/fail rule."""

    return {
        "primary_oof_mae_threshold_kcal_mol": PRIMARY_MAE_THRESHOLD_KCAL_MOL,
        "primary_mixed_505_mae_threshold_kcal_mol": PRIMARY_MAE_THRESHOLD_KCAL_MOL,
        "minimum_oof_mae_improvement_over_better_of_m0_m1_kcal_mol": (
            MINIMUM_MAE_IMPROVEMENT_KCAL_MOL
        ),
        "cluster_bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "cluster_bootstrap_seed": BOOTSTRAP_SEED,
        "one_sided_95_percent_oof_mae_ucb": "report-only-on-development",
        "required_conditions": (
            "finite-complete-OOF-water-MAE<=1.5, mixed-505-water-OOF-plus-"
            "nonaqueous-M1-MAE<=1.5, and OOF-water-MAE improvement>=0.05 over "
            "the better frozen M0/M1 water baseline"
        ),
        "tail_and_family_influence_metrics": (
            "report-only-on-development-and-fail-closed-if-nonfinite"
        ),
    }


def _normalized_runtime_identity() -> tuple[dict[str, object], str]:
    identity = runtime_record()
    identity.pop("generated_at_utc", None)
    return identity, canonical_json_sha256(identity)


def _outside_source(path: Path, source_root: Path, *, name: str) -> None:
    try:
        path.relative_to(source_root)
    except ValueError:
        return
    raise M3PreregistrationError(f"{name} must be outside the source checkout.")


def _lower_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M3PreregistrationError(f"{name} must be a lowercase SHA256.")
    return value


def _read_json_object(path: Path, *, name: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise M3PreregistrationError(f"{name} must be a JSON object.")
    return payload, hashlib.sha256(raw).hexdigest()


def _validate_self_hash(
    payload: Mapping[str, Any], *, digest_key: str, name: str
) -> None:
    unsigned = dict(payload)
    observed = unsigned.pop(digest_key, None)
    if observed != canonical_json_sha256(unsigned):
        raise M3PreregistrationError(f"{name} self hash drifted.")


def _numeric_rank(values: np.ndarray) -> tuple[int, float, np.ndarray]:
    singular = np.linalg.svd(values, compute_uv=False)
    if singular.size == 0 or not np.all(np.isfinite(singular)):
        raise M3PreregistrationError(
            "A target-blind matrix has invalid singular values."
        )
    tolerance = float(max(values.shape) * np.finfo(values.dtype).eps * singular[0])
    return int(np.count_nonzero(singular > tolerance)), tolerance, singular


def _orient_mode(mode: np.ndarray) -> tuple[np.ndarray, int]:
    result = np.array(mode, dtype=float, copy=True)
    magnitudes = np.abs(result)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise M3PreregistrationError("A target-blind mode is invalid.")
    maximum = float(np.max(magnitudes))
    if maximum <= 0.0:
        raise M3PreregistrationError("A target-blind mode has zero norm.")
    pivot = int(np.flatnonzero(magnitudes == maximum)[0])
    if result[pivot] < 0.0:
        result *= -1.0
    return result, pivot


def target_blind_subspace(
    matrix: np.ndarray, stock_coefficients: np.ndarray
) -> dict[str, object]:
    """Return the frozen stock direction plus two target-blind geometry modes."""

    values = np.asarray(matrix, dtype=float)
    stock = np.asarray(stock_coefficients, dtype=float)
    if (
        values.ndim != 2
        or stock.shape != (values.shape[1],)
        or values.shape[0] < values.shape[1]
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(stock))
    ):
        raise M3PreregistrationError("M3 matrix or stock coefficients are invalid.")

    rank, rank_tolerance, raw_singular = _numeric_rank(values)
    if rank != values.shape[1]:
        raise M3PreregistrationError("M3 target-blind matrix is rank deficient.")
    column_norms = np.linalg.norm(values, axis=0)
    if not np.all(np.isfinite(column_norms)) or np.any(column_norms <= 0.0):
        raise M3PreregistrationError("M3 matrix has an inactive design column.")

    standardized = values / column_norms
    metric_stock = column_norms * stock
    metric_stock_norm = float(np.linalg.norm(metric_stock))
    if not math.isfinite(metric_stock_norm) or metric_stock_norm <= 0.0:
        raise M3PreregistrationError("Stock coefficients define no M3 direction.")
    stock_unit = metric_stock / metric_stock_norm
    projector = np.eye(values.shape[1]) - np.outer(stock_unit, stock_unit)
    projected = standardized @ projector
    _left, projected_singular, right_transpose = np.linalg.svd(
        projected, full_matrices=False
    )
    projected_rank = int(
        np.count_nonzero(
            projected_singular
            > max(projected.shape) * np.finfo(values.dtype).eps * projected_singular[0]
        )
    )
    if projected_rank != values.shape[1] - 1:
        raise M3PreregistrationError(
            "Stock-orthogonal target-blind subspace has unexpected rank."
        )
    if projected_singular.size < 3:
        raise M3PreregistrationError("M3 requires at least three projected modes.")
    first_gap = float(projected_singular[0] - projected_singular[1])
    second_gap = float(projected_singular[1] - projected_singular[2])
    gap_scale = max(1.0, float(projected_singular[0]))
    if (
        first_gap <= _SINGULAR_GAP_RELATIVE_TOLERANCE * gap_scale
        or second_gap <= _SINGULAR_GAP_RELATIVE_TOLERANCE * gap_scale
    ):
        raise M3PreregistrationError(
            "Leading target-blind geometry modes are not uniquely separated."
        )

    modes = np.array(right_transpose[:2].T, copy=True)
    pivots: list[int] = []
    for column in range(2):
        modes[:, column], pivot = _orient_mode(modes[:, column])
        pivots.append(pivot)
    coefficient_modes = modes / column_norms[:, None]
    reduced_design = np.column_stack((values @ stock, values @ coefficient_modes))

    gram = coefficient_modes.T @ np.diag(column_norms**2) @ coefficient_modes
    stock_orthogonality = stock @ np.diag(column_norms**2) @ coefficient_modes
    if not np.allclose(gram, np.eye(2), atol=_IDENTITY_ABSOLUTE_TOLERANCE, rtol=0.0):
        raise M3PreregistrationError("M3 coefficient modes lost metric orthonormality.")
    if not np.allclose(
        stock_orthogonality,
        np.zeros(2),
        atol=_IDENTITY_ABSOLUTE_TOLERANCE,
        rtol=0.0,
    ):
        raise M3PreregistrationError("M3 coefficient modes are not stock-orthogonal.")
    reduced_rank, reduced_tolerance, reduced_singular = _numeric_rank(reduced_design)
    if reduced_rank != 3:
        raise M3PreregistrationError("M3 reduced design is rank deficient.")

    # Independent deterministic positive unit reparameterization canary.
    unit_scale = np.exp(np.linspace(-2.0, 2.0, values.shape[1]))
    rescaled = values / unit_scale
    rescaled_stock = stock * unit_scale
    rescaled_norms = np.linalg.norm(rescaled, axis=0)
    rescaled_standardized = rescaled / rescaled_norms
    rescaled_metric_stock = rescaled_norms * rescaled_stock
    rescaled_unit = rescaled_metric_stock / np.linalg.norm(rescaled_metric_stock)
    rescaled_projector = np.eye(values.shape[1]) - np.outer(
        rescaled_unit, rescaled_unit
    )
    _u2, _s2, vh2 = np.linalg.svd(
        rescaled_standardized @ rescaled_projector, full_matrices=False
    )
    rescaled_modes = np.array(vh2[:2].T, copy=True)
    for column in range(2):
        rescaled_modes[:, column], _ = _orient_mode(rescaled_modes[:, column])
    rescaled_coefficient_modes = rescaled_modes / rescaled_norms[:, None]
    rescaled_reduced = np.column_stack(
        (rescaled @ rescaled_stock, rescaled @ rescaled_coefficient_modes)
    )
    invariance_error = float(np.max(np.abs(rescaled_reduced - reduced_design)))
    if invariance_error > _IDENTITY_ABSOLUTE_TOLERANCE:
        raise M3PreregistrationError(
            "M3 subspace changed under positive parameter-unit reparameterization."
        )

    return {
        "construction": (
            "theta(alpha,beta)=alpha*theta_stock+D^-1*[v1,v2]*beta; "
            "v1,v2 are the first two right singular vectors of "
            "(X*D^-1)*(I-q0*q0^T), q0=D*theta_stock/||D*theta_stock||"
        ),
        "parameter_count": 3,
        "intercept": False,
        "column_l2_norms": column_norms.tolist(),
        "stock_coefficients": stock.tolist(),
        "stock_metric_norm": metric_stock_norm,
        "stock_unit_vector": stock_unit.tolist(),
        "geometry_modes_standardized_coordinates": modes.tolist(),
        "geometry_mode_sign_pivots": pivots,
        "geometry_modes_coefficient_coordinates": coefficient_modes.tolist(),
        "reduced_design": reduced_design.tolist(),
        "reduced_design_sha256": canonical_json_sha256(reduced_design.tolist()),
        "raw_rank": rank,
        "raw_rank_tolerance": rank_tolerance,
        "raw_singular_values": raw_singular.tolist(),
        "projected_rank": projected_rank,
        "projected_singular_values": projected_singular.tolist(),
        "first_singular_gap": first_gap,
        "second_singular_gap": second_gap,
        "reduced_rank": reduced_rank,
        "reduced_rank_tolerance": reduced_tolerance,
        "reduced_singular_values": reduced_singular.tolist(),
        "metric_orthonormality_max_abs_error": float(np.max(np.abs(gram - np.eye(2)))),
        "stock_metric_orthogonality_max_abs_error": float(
            np.max(np.abs(stock_orthogonality))
        ),
        "positive_unit_reparameterization_prediction_max_abs_error": (invariance_error),
        "target_values_read_or_used": False,
    }


def _safe_geometry_members(archive: ZipFile) -> list[tuple[PurePosixPath, ZipInfo]]:
    infos = archive.infolist()
    if len(infos) > MNSOL_MAX_ZIP_MEMBERS:
        raise M3PreregistrationError("MNSol archive has too many members.")
    if sum(max(0, info.file_size) for info in infos) > MNSOL_MAX_ZIP_UNCOMPRESSED_BYTES:
        raise M3PreregistrationError("MNSol archive is too large when uncompressed.")
    if any(
        info.file_size > 0
        and (
            info.compress_size <= 0
            or info.file_size / info.compress_size > MNSOL_MAX_ZIP_COMPRESSION_RATIO
        )
        for info in infos
    ):
        raise M3PreregistrationError("MNSol archive compression ratio is unsafe.")
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise M3PreregistrationError("MNSol archive has duplicate member names.")

    result: list[tuple[PurePosixPath, ZipInfo]] = []
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise M3PreregistrationError("MNSol archive contains an unsafe path.")
        if (
            "__MACOSX" not in path.parts
            and path.suffix.lower() == ".xyz"
            and path.parent.name == "all_solutes"
        ):
            result.append((path, info))
    if len(result) > MNSOL_MAX_GEOMETRY_COUNT:
        raise M3PreregistrationError("MNSol archive has too many geometries.")
    if (
        sum(max(0, info.file_size) for _, info in result)
        > MNSOL_MAX_TOTAL_GEOMETRY_BYTES
    ):
        raise M3PreregistrationError("MNSol geometry payload is too large.")
    stems = [path.stem for path, _ in result]
    if len(stems) != len(set(stems)):
        raise M3PreregistrationError("MNSol archive has duplicate geometry handles.")
    return result


def _read_geometry_member(archive: ZipFile, info: ZipInfo, *, handle: str) -> bytes:
    if info.file_size < 0 or info.file_size > MNSOL_MAX_GEOMETRY_BYTES:
        raise M3PreregistrationError(f"Geometry {handle!r} exceeds its size limit.")
    with archive.open(info, "r") as stream:
        payload = stream.read(MNSOL_MAX_GEOMETRY_BYTES + 1)
    if len(payload) > MNSOL_MAX_GEOMETRY_BYTES:
        raise M3PreregistrationError(f"Geometry {handle!r} exceeds its size limit.")
    return payload


def load_selected_geometries_without_table(
    archive_path: Path, geometry_sha256: Iterable[str]
) -> dict[str, MNSolGeometry]:
    """Load selected geometry members without opening the MNSol target table."""

    if not stat.S_ISREG(archive_path.lstat().st_mode):
        raise M3PreregistrationError("MNSol archive must be a regular file.")
    if archive_path.stat().st_size > MNSOL_MAX_SOURCE_ARCHIVE_BYTES:
        raise M3PreregistrationError("MNSol archive exceeds its compressed size limit.")
    wanted = {_lower_sha256(value, name="geometry_sha256") for value in geometry_sha256}
    if not wanted:
        raise M3PreregistrationError("No selected geometry identities were supplied.")
    found: dict[str, MNSolGeometry] = {}
    with ZipFile(archive_path) as archive:
        for logical_path, info in _safe_geometry_members(archive):
            payload = _read_geometry_member(archive, info, handle=logical_path.stem)
            digest = hashlib.sha256(payload).hexdigest()
            if digest not in wanted:
                continue
            if digest in found:
                raise M3PreregistrationError(
                    "A selected geometry digest occurs more than once in the archive."
                )
            geometry = _parse_geometry(logical_path.stem, payload)
            if geometry.sha256 != digest:
                raise M3PreregistrationError("Geometry parser digest drifted.")
            found[digest] = geometry
    missing = sorted(wanted - set(found))
    if missing:
        raise M3PreregistrationError(
            f"Selected geometries are absent from the archive: {missing}."
        )
    return found


def heavy_element_formula_family_descriptor(
    geometry: MNSolGeometry,
) -> dict[str, object]:
    atomic_numbers = [int(value) for value in geometry.atomic_numbers if value > 1]
    scope = "heavy-atoms"
    if not atomic_numbers:
        atomic_numbers = [int(value) for value in geometry.atomic_numbers]
        scope = "all-atoms-no-heavy-atom-fallback"
    counts = sorted(Counter(atomic_numbers).items())
    return {
        "contract": "exact-element-count-family-v1",
        "scope": scope,
        "atomic_number_counts": [list(item) for item in counts],
    }


def heavy_element_formula_family_key(geometry: MNSolGeometry) -> str:
    """Return a target-blind equivalence key with no transitive-chain merging."""

    return canonical_json_sha256(heavy_element_formula_family_descriptor(geometry))


def deterministic_family_folds(
    *,
    stable_row_ids: Sequence[str],
    family_keys: Sequence[str],
    fold_count: int = OOF_FOLD_COUNT,
) -> dict[str, object]:
    if len(stable_row_ids) != len(family_keys) or not stable_row_ids:
        raise M3PreregistrationError("M3 family inputs are inconsistent.")
    if fold_count < 2:
        raise M3PreregistrationError("M3 OOF requires at least two folds.")
    if len(set(stable_row_ids)) != len(stable_row_ids):
        raise M3PreregistrationError("M3 stable row identities are not unique.")

    members_by_key: dict[str, list[int]] = {}
    for index, raw_key in enumerate(family_keys):
        key = _lower_sha256(raw_key, name="family_key")
        members_by_key.setdefault(key, []).append(index)
    if len(members_by_key) < fold_count:
        raise M3PreregistrationError("M3 has fewer target-blind families than folds.")

    families: list[dict[str, object]] = []
    for key, members in members_by_key.items():
        family_stable_key = min(stable_row_ids[index] for index in members)
        families.append(
            {
                "family_key": key,
                "family_stable_key": family_stable_key,
                "member_indices": sorted(members),
            }
        )
    families.sort(
        key=lambda item: (
            -len(item["member_indices"]),
            item["family_stable_key"],
            item["family_key"],
        )
    )

    loads = [0] * fold_count
    row_folds = [-1] * len(stable_row_ids)
    emitted_families: list[dict[str, object]] = []
    for family in families:
        fold = min(range(fold_count), key=lambda index: (loads[index], index))
        members = list(family["member_indices"])
        loads[fold] += len(members)
        for member in members:
            if row_folds[member] != -1:
                raise M3PreregistrationError("An M3 row belongs to two families.")
            row_folds[member] = fold
        emitted_families.append({**family, "fold": fold})
    if any(value < 0 for value in row_folds) or any(value == 0 for value in loads):
        raise M3PreregistrationError("M3 fold assignment is incomplete.")
    for family in emitted_families:
        if {row_folds[index] for index in family["member_indices"]} != {family["fold"]}:
            raise M3PreregistrationError("An M3 family was split across folds.")

    return {
        "fold_count": fold_count,
        "family_count": len(emitted_families),
        "maximum_family_size": max(
            len(family["member_indices"]) for family in emitted_families
        ),
        "fold_record_counts": loads,
        "row_fold_assignments": row_folds,
        "families": emitted_families,
        "assignment": (
            "sort families by decreasing size then stable family/identity key; "
            "assign to the currently least-loaded fold, lowest fold index on ties"
        ),
    }


def _validate_fold_designs(
    reduced_design: np.ndarray, row_folds: Sequence[int]
) -> list[dict[str, object]]:
    values = np.asarray(reduced_design, dtype=float)
    folds = np.asarray(row_folds, dtype=int)
    if values.shape != (len(folds), 3) or not np.all(np.isfinite(values)):
        raise M3PreregistrationError("M3 fold design has an invalid shape.")
    diagnostics: list[dict[str, object]] = []
    for fold in range(OOF_FOLD_COUNT):
        training = values[folds != fold]
        validation = values[folds == fold]
        if training.size == 0 or validation.size == 0:
            raise M3PreregistrationError("An M3 OOF fold is empty.")
        scales = np.linalg.norm(training, axis=0)
        if np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
            raise M3PreregistrationError("An M3 fold has an inactive predictor.")
        standardized = training / scales
        rank, tolerance, singular = _numeric_rank(standardized)
        if rank != 3:
            raise M3PreregistrationError("An M3 training fold is rank deficient.")
        condition = float(singular[0] / singular[-1])
        if not math.isfinite(condition) or condition > 1.0e6:
            raise M3PreregistrationError("An M3 training fold is ill-conditioned.")
        diagnostics.append(
            {
                "fold": fold,
                "training_count": len(training),
                "validation_count": len(validation),
                "training_column_l2_scales": scales.tolist(),
                "standardized_rank": rank,
                "standardized_rank_tolerance": tolerance,
                "standardized_singular_values": singular.tolist(),
                "standardized_condition_number": condition,
            }
        )
    return diagnostics


def _write_json_exclusive_atomic(
    *, output: Path, payload: dict[str, object], snapshot: RepositorySnapshot
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        snapshot.assert_unchanged()
        os.link(temporary, output)
        directory_fd = os.open(
            output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_stock_inputs(
    *, aggregate_path: Path, stock_preregistration_path: Path, input_root: Path
) -> tuple[dict[str, Any], str, dict[str, Any], str, np.ndarray]:
    if aggregate_path.stat().st_mode & 0o222:
        raise M3PreregistrationError("Stock-area aggregate must be read-only.")
    aggregate, aggregate_file_sha = _read_json_object(
        aggregate_path, name="stock-area aggregate"
    )
    _validate_self_hash(
        aggregate,
        digest_key="aggregate_sha256",
        name="stock-area aggregate",
    )
    for key, value in {
        "artifact": STOCK_AGGREGATE_ARTIFACT,
        "schema_version": 1,
        "status": "complete",
        "record_count": EXPECTED_WATER_COUNT,
        "partition": "development-water-only",
        "experimental_targets_used_by_feature_aggregation": False,
        "experimental_targets_emitted": False,
        "hybrid_prediction_records_read": False,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_selected_or_emitted": False,
        "fitting_or_calibration_performed": False,
    }.items():
        if aggregate.get(key) != value:
            raise M3PreregistrationError(f"Stock aggregate field {key!r} drifted.")
    if aggregate.get("design_parameter_names") != list(
        SMD_WATER_TENSION_PARAMETER_NAMES
    ):
        raise M3PreregistrationError("Stock aggregate parameter ordering drifted.")

    records = aggregate.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_WATER_COUNT:
        raise M3PreregistrationError("Stock aggregate records are incomplete.")
    rows: list[np.ndarray] = []
    stable_ids: set[str] = set()
    geometry_ids: set[str] = set()
    for ordinal, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or set(record) != _STOCK_RECORD_KEYS
            or record.get("water_ordinal") != ordinal
        ):
            raise M3PreregistrationError("Stock aggregate record ordering drifted.")
        stable_id = _lower_sha256(
            record.get("opaque_record_id"), name="opaque_record_id"
        )
        geometry_id = _lower_sha256(
            record.get("geometry_sha256"), name="geometry_sha256"
        )
        stable_ids.add(stable_id)
        geometry_ids.add(geometry_id)
        row = np.asarray(record.get("design_row_angstrom2_div_1000"), dtype=float)
        if row.shape != (EXPECTED_PARAMETER_COUNT,) or not np.all(np.isfinite(row)):
            raise M3PreregistrationError("Stock aggregate contains an invalid row.")
        expected_stock = float(
            row @ SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2
        )
        observed_stock = float(record.get("stock_smd_reconstruction_kcal_mol"))
        if not math.isclose(
            expected_stock, observed_stock, abs_tol=2.0e-12, rel_tol=0.0
        ):
            raise M3PreregistrationError("Stock reconstruction control drifted.")
        rows.append(row)
    if len(stable_ids) != EXPECTED_WATER_COUNT:
        raise M3PreregistrationError(
            "Stock aggregate stable identities are not unique."
        )
    if len(geometry_ids) != EXPECTED_WATER_COUNT:
        raise M3PreregistrationError("Water development geometries are not unique.")
    matrix = np.stack(rows)
    if aggregate.get("matrix_sha256") != canonical_json_sha256(matrix.tolist()):
        raise M3PreregistrationError("Stock aggregate matrix digest drifted.")

    resolved_stock_prereg = stock_preregistration_path.resolve(strict=True)
    if (
        Path(str(aggregate.get("preregistration_path"))).resolve()
        != resolved_stock_prereg
    ):
        raise M3PreregistrationError(
            "Stock aggregate belongs to another preregistration."
        )
    if aggregate.get("preregistration_sha256") != sha256_file(resolved_stock_prereg):
        raise M3PreregistrationError("Stock preregistration digest drifted.")
    if resolved_stock_prereg.stat().st_mode & 0o222:
        raise M3PreregistrationError("Stock preregistration must be read-only.")
    stock_prereg, stock_prereg_file_sha = _read_json_object(
        resolved_stock_prereg, name="stock preregistration"
    )
    _validate_self_hash(
        stock_prereg,
        digest_key="self_sha256",
        name="stock preregistration",
    )
    if stock_prereg.get("artifact") != STOCK_PREREGISTRATION_ARTIFACT:
        raise M3PreregistrationError("Stock preregistration identity drifted.")
    if Path(str(stock_prereg.get("input_root"))).resolve() != input_root:
        raise M3PreregistrationError("Stock preregistration input root drifted.")
    input_hashes = stock_prereg.get("input_files_sha256")
    if not isinstance(input_hashes, dict) or set(input_hashes) != set(
        _FROZEN_INPUT_FILE_NAMES
    ):
        raise M3PreregistrationError("Stock preregistration input manifest drifted.")
    for name, digest in input_hashes.items():
        path = (input_root / name).resolve(strict=True)
        try:
            path.relative_to(input_root)
        except ValueError as exc:
            raise M3PreregistrationError(
                "A stock input path escaped its root."
            ) from exc
        if _lower_sha256(digest, name=f"input_files_sha256[{name!r}]") != sha256_file(
            path
        ):
            raise M3PreregistrationError(f"Frozen input {name!r} drifted.")
    return (
        aggregate,
        aggregate_file_sha,
        stock_prereg,
        stock_prereg_file_sha,
        matrix,
    )


def _validate_selection_identities(
    *, selection_path: Path, aggregate_records: Sequence[Mapping[str, Any]]
) -> None:
    selection, _sha = _read_json_object(selection_path, name="development selection")
    if (
        selection.get("partition") != "development"
        or selection.get("record_count") != 505
    ):
        raise M3PreregistrationError("Development selection identity drifted.")
    selected = selection.get("selected_records")
    if not isinstance(selected, list) or len(selected) != 505:
        raise M3PreregistrationError("Development selection records are incomplete.")
    by_index = {
        item.get("selection_index"): item for item in selected if isinstance(item, dict)
    }
    if len(by_index) != 505:
        raise M3PreregistrationError("Development selection indices are invalid.")
    for record in aggregate_records:
        index = record.get("selection_index")
        item = by_index.get(index)
        if item is None or any(
            item.get(key) != record.get(key)
            for key in ("opaque_record_id", "geometry_sha256")
        ):
            raise M3PreregistrationError("Stock and development identities disagree.")
        if item.get("canonical_solvent") != "water":
            raise M3PreregistrationError("M3 received a non-water development row.")


def create(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    if source_root != _SOURCE_ROOT:
        raise M3PreregistrationError("M3 creator must run for its own checkout.")
    input_root = args.input_root.expanduser().resolve(strict=True)
    aggregate_path = args.stock_aggregate.expanduser().resolve(strict=True)
    stock_preregistration_path = args.stock_preregistration.expanduser().resolve(
        strict=True
    )
    output = args.output.expanduser().resolve()
    _outside_source(output, source_root, name="M3 preregistration")
    if output.exists():
        raise FileExistsError(output)

    snapshot = RepositorySnapshot.capture(source_root)
    aggregate, aggregate_file_sha, stock_prereg, stock_prereg_file_sha, matrix = (
        _validate_stock_inputs(
            aggregate_path=aggregate_path,
            stock_preregistration_path=stock_preregistration_path,
            input_root=input_root,
        )
    )
    parent_hybrid_path = Path(
        str(stock_prereg.get("parent_hybrid_preregistration_path"))
    ).resolve(strict=True)
    if parent_hybrid_path.stat().st_mode & 0o222:
        raise M3PreregistrationError("Parent hybrid preregistration must be read-only.")
    parent_hybrid, parent_hybrid_file_sha = _read_json_object(
        parent_hybrid_path, name="parent hybrid preregistration"
    )
    if (
        stock_prereg.get("parent_hybrid_preregistration_sha256")
        != parent_hybrid_file_sha
    ):
        raise M3PreregistrationError("Parent hybrid preregistration digest drifted.")
    for key, value in {
        "artifact_id": "route2-hybrid-smd-development-prereg-v3",
        "schema_version": 3,
        "status": "locked-before-first-v3-hybrid-evaluation",
        "partition": "development",
        "record_count": 505,
        "confirmation_partition_opened": False,
        "fitting_or_calibration_permitted": False,
        "input_root": str(input_root),
        "preregistration_path": str(parent_hybrid_path),
    }.items():
        if parent_hybrid.get(key) != value:
            raise M3PreregistrationError(f"Parent hybrid field {key!r} drifted.")
    hybrid_records_path = Path(str(parent_hybrid.get("records_path"))).resolve(
        strict=True
    )
    if hybrid_records_path != parent_hybrid_path.parent / "records":
        raise M3PreregistrationError("Parent hybrid record path drifted.")
    records = list(aggregate["records"])
    selection_path = input_root / "route2-mnsol-development-selection-v1.private.json"
    _validate_selection_identities(
        selection_path=selection_path,
        aggregate_records=records,
    )
    archive_path = input_root / "MNSolDatabase_v2012.zip"
    geometries = load_selected_geometries_without_table(
        archive_path,
        [str(record["geometry_sha256"]) for record in records],
    )

    subspace = target_blind_subspace(
        matrix,
        SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2,
    )
    stable_ids = [str(record["opaque_record_id"]) for record in records]
    family_descriptors = [
        heavy_element_formula_family_descriptor(
            geometries[str(record["geometry_sha256"])]
        )
        for record in records
    ]
    family_keys = [canonical_json_sha256(item) for item in family_descriptors]
    folds = deterministic_family_folds(
        stable_row_ids=stable_ids,
        family_keys=family_keys,
    )
    descriptor_by_key = {
        key: descriptor
        for key, descriptor in zip(family_keys, family_descriptors, strict=True)
    }
    for family in folds["families"]:
        family["family_descriptor"] = descriptor_by_key[family["family_key"]]
    row_ledger = [
        {
            "water_ordinal": ordinal,
            "selection_index": record["selection_index"],
            "opaque_record_id": record["opaque_record_id"],
            "geometry_sha256": record["geometry_sha256"],
            "family_key": family_keys[ordinal],
            "fold": folds["row_fold_assignments"][ordinal],
        }
        for ordinal, record in enumerate(records)
    ]
    fold_diagnostics = _validate_fold_designs(
        np.asarray(subspace["reduced_design"], dtype=float),
        folds["row_fold_assignments"],
    )

    runtime_identity, runtime_identity_sha256 = _normalized_runtime_identity()
    source_hashes = committed_source_hashes(snapshot, _SOURCE_FILES)
    payload: dict[str, object] = {
        "artifact": PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-first-m3-target-use-or-fit",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_profile_id": CANDIDATE_PROFILE_ID,
        "partition": "development-water-only",
        "water_record_count": EXPECTED_WATER_COUNT,
        "source_root": str(source_root),
        "source_git_head": snapshot.head,
        "source_git_tree": snapshot.tree,
        "source_files_sha256": source_hashes,
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": runtime_identity_sha256,
        "input_root": str(input_root),
        "input_files_sha256": dict(stock_prereg["input_files_sha256"]),
        "stock_aggregate_path": str(aggregate_path),
        "stock_aggregate_file_sha256": aggregate_file_sha,
        "stock_aggregate_self_sha256": aggregate["aggregate_sha256"],
        "stock_matrix_sha256": aggregate["matrix_sha256"],
        "stock_preregistration_path": str(stock_preregistration_path),
        "stock_preregistration_file_sha256": stock_prereg_file_sha,
        "parent_hybrid_preregistration_path": str(parent_hybrid_path),
        "parent_hybrid_preregistration_file_sha256": parent_hybrid_file_sha,
        "parent_hybrid_records_path": str(hybrid_records_path),
        "parent_hybrid_evidence_root": str(parent_hybrid_path.parent),
        "water_identity_sha256": aggregate["water_identity_sha256"],
        "experimental_targets_read_by_m3_preregistration": False,
        "experimental_targets_used_by_subspace_or_folds": False,
        "mnsol_target_table_member_opened_by_m3_preregistration": False,
        "hybrid_prediction_records_read_by_m3_preregistration": False,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_opened": False,
        "fitting_or_calibration_performed": False,
        "prior_development_information_disclosed": {
            "m2_water_mae_kcal_mol": 1.6265953472,
            "m2_failed_primary_threshold": True,
            "m3_attempt_was_selected_after_m2_failure": True,
            "subspace_and_folds_use_no_target_or_hybrid_values": True,
        },
        "external_pro_math_reviews": list(_PRO_REVIEWS),
        "subspace": subspace,
        "grouped_oof": {
            "family_contract": (
                "exact heavy-element-count multiset; all-element-count fallback "
                "only for molecules without a heavy atom"
            ),
            "family_claim": (
                "No identical operational element-count family crosses folds. "
                "This is not a scaffold, MCS, bond-order, or homolog-series "
                "generalization guarantee."
            ),
            **folds,
            "row_ledger": row_ledger,
            "fold_design_diagnostics": fold_diagnostics,
        },
        "fit_contract": frozen_fit_contract(),
        "development_decision_rule": frozen_development_decision_rule(),
        "deployment_policy": (
            "No deployable coefficient is created by preregistration. If OOF "
            "admission passes, one separately identified all-306 coefficient is "
            "fit under the same frozen solver; OOF predictions remain the only "
            "development validation predictions."
        ),
        "nonaqueous_policy": "retain-stock-smd-cds-m1-unchanged",
        "confirmation_policy": "remain-sealed-until-m3-development-decision-is-frozen",
        "claim_boundary": (
            "Target-blind stock-area M3 preregistration only. The selected "
            "subspace uses the complete development covariate matrix and is "
            "therefore transductive with respect to geometry covariates, but no "
            "development target or hybrid prediction value. It is not an "
            "accuracy result, a production smooth-area fit, a force model, or "
            "a public Route-2 capability."
        ),
    }
    payload["self_sha256"] = canonical_json_sha256(payload)
    _write_json_exclusive_atomic(output=output, payload=payload, snapshot=snapshot)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--stock-aggregate", type=Path, required=True)
    parser.add_argument("--stock-preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    payload = create(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
