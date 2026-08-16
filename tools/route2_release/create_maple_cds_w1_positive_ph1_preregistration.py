#!/usr/bin/env python3
"""Freeze the final positive-parent PH1.0 lane before target access.

The creator consumes only the target-blind positive-parent design aggregate and
historical target-free fold metadata.  It constructs every fold-specific and
all-development deployment subspace before any experimental or hybrid result is
joined to the design matrix.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))

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
from tools.route2_release.create_maple_cds_w1_m3_preregistration import (
    OOF_FOLD_COUNT,
)

AGGREGATE_ARTIFACT = "route2-maple-cds-w1-positive-parent-water-feature-matrix-v2"
PREREGISTRATION_ARTIFACT = "route2-maple-cds-w1-positive-parent-ph1-prereg-v1"
CANDIDATE_PROFILE_ID = "maple-cds-w1-positive-parent-target-blind-3d-ph1-water-v1"
RESULT_ARTIFACT = "route2-maple-cds-w1-positive-parent-ph1-grouped-oof-v1"
EXPECTED_WATER_COUNT = 306
EXPECTED_MIXED_COUNT = 505

M3_PREREGISTRATION_FILE_SHA256 = (
    "0302635aa2186f326cf73ffccf284b0f0ba07aa7d51c4432c13cc03e80ba365b"
)
M3_PREREGISTRATION_SELF_SHA256 = (
    "a79d551ac7b8040444a7a4168c3e872b43bcd84f2b481695c7a77fd371c26092"
)
M4_PREREGISTRATION_FILE_SHA256 = (
    "0e844d937cd43eb864966d28d6468599607cd27ad9cafb889dcfce75b2211444"
)
M4_PREREGISTRATION_SELF_SHA256 = (
    "e3f893061f1f45a68f6ce506942683e7e75fcd48d0ff8c04e6cd85afd3615760"
)
M4_FAILURE_FILE_SHA256 = (
    "36de2acfbd3389ef117ed15266d56789104c796c85cef52d663c4ad728469d8b"
)
M4_FAILURE_SELF_SHA256 = (
    "a12a22425e40b26dc1637713e8c20690901aa1737b1f5aa139690fc945a17495"
)

PRO_Q10_PROMPT_SHA256 = (
    "872ccc0d1cbecf31231758bd8cd1f6f529dd4f84aad6d507b9a5f71f550fb0a8"
)
PRO_Q10_ANSWER_SHA256 = (
    "c156fb5c5fdf5764a20b46b04856cae93a3e510ea4a518295215726134d3d7d5"
)
PRO_Q11_PROMPT_SHA256 = (
    "9bcf28b31a0ed1eb62de31445f473c9fa1b52c6fec50b962e0645d5b83fb1992"
)
PRO_Q11_ANSWER_SHA256 = (
    "20668f9c800462173f5c0630abcc8122703aa184a10e5adf31fc22558d31347c"
)
PRO_Q12_PROMPT_SHA256 = (
    "c4a09e46142a8aef205c209ed0e16c48d60574e5c9212753f8ae8f63e5d18073"
)
PRO_Q12_SUBMISSION_SHA256 = (
    "3db4077018f2468b528164a90e2b84bd9aa8f711d3ceaf1a8a927eaa399998f5"
)
PRO_Q12_ANSWER_SHA256 = (
    "c76a2eca4300f6019acbbb1c16801c46904bd652eda905919a756fe1b62cc965"
)

PSEUDO_HUBER_DELTA_KCAL_MOL = 1.0
PREDICTION_CERTIFICATE_KCAL_MOL = 1.0e-6
PRIMARY_MAE_THRESHOLD_KCAL_MOL = 1.5
MIXED_MAE_THRESHOLD_KCAL_MOL = 1.5
MAXIMUM_MATRIX_CONDITION = 1.0e8
RELATIVE_RANK_FLOOR = 1.0e-12
RELATIVE_MODE_FLOOR = 1.0e-8
RELATIVE_MODE_GAP = 1.0e-8
PROJECTOR_TOLERANCE = 1.0e-10
PIVOT_TIE_TOLERANCE = 1.0e-12
REPRESENTATION_TOLERANCE_KCAL_MOL = 1.0e-10
NEWTON_ARMIJO_C1 = 1.0e-4
NEWTON_BACKTRACK_FACTOR = 0.5
NEWTON_MAXIMUM_ACCEPTED_ITERATIONS = 200
NEWTON_MAXIMUM_ARMIJO_TRIALS = 60
NEWTON_DAMPING_EXPONENTS = tuple(range(-12, 13))
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_817

_SOURCE_FILES = (
    "GOAL.md",
    "docs/route2/MAPLE_CDS_W1.md",
    "docs/route2/evidence/M4_TERMINAL_FAILURE_2026-08-17.md",
    "docs/route2/evidence/POSITIVE_BERNSTEIN_PARENT_PRO_AUDIT_2026-08-17.md",
    "docs/route2/evidence/POSITIVE_PH1_PRO_AUDIT_2026-08-17.md",
    "docs/route2/evidence/SMOOTH_AREA_TERMINAL_LANE_PRO_DECISION_2026-08-17.md",
    "maple/function/calculator/extra_correction/implicit/smd_cds.py",
    "maple/solvation/release/evidence.py",
    "tests/route2_vnext/test_maple_cds_w1_positive_ph1.py",
    "tools/route2_release/aggregate_maple_cds_w1_features.py",
    "tools/route2_release/analyze_hybrid_smd_components_v3.py",
    "tools/route2_release/create_maple_cds_w1_m3_preregistration.py",
    "tools/route2_release/create_maple_cds_w1_positive_ph1_preregistration.py",
    "tools/route2_release/fit_maple_cds_w1_positive_ph1.py",
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
        "mixed_record_count",
        "source_root",
        "source_git_head",
        "source_git_tree",
        "source_files_sha256",
        "runtime_identity",
        "runtime_identity_sha256",
        "feature_aggregate_path",
        "feature_aggregate_file_sha256",
        "feature_aggregate_self_sha256",
        "feature_matrix_sha256",
        "feature_water_identity_sha256",
        "m3_preregistration_path",
        "m3_preregistration_file_sha256",
        "m3_preregistration_self_sha256",
        "m4_preregistration_path",
        "m4_preregistration_file_sha256",
        "m4_preregistration_self_sha256",
        "m4_failure_path",
        "m4_failure_file_sha256",
        "m4_failure_self_sha256",
        "m4_oof_mae_computed",
        "m4_terminal_rule_changed_after_failure",
        "parent_hybrid_evidence_root",
        "parent_hybrid_records_path",
        "parent_hybrid_integrity_audit_path",
        "parent_hybrid_integrity_audit_sha256",
        "row_ledger",
        "row_fold_assignments",
        "fold_record_counts",
        "family_count",
        "maximum_family_size",
        "stock_coefficients_cal_mol_angstrom2",
        "stock_coefficients_sha256",
        "design_parameter_names",
        "fold_subspaces",
        "deployment_subspace",
        "fold_subspaces_sha256",
        "deployment_subspace_sha256",
        "experimental_targets_read_by_preregistration",
        "hybrid_prediction_records_read_by_preregistration",
        "confirmation_selection_manifest_opened",
        "confirmation_records_opened",
        "fitting_or_calibration_performed",
        "fit_contract",
        "development_decision_rule",
        "external_pro_math_review",
        "claim_boundary",
        "self_sha256",
    }
)


class PositivePH1PreregistrationError(RuntimeError):
    """Raised when the final target-blind PH1.0 contract cannot be frozen."""


def normalized_runtime_identity() -> tuple[dict[str, object], str]:
    identity = runtime_record()
    identity.pop("generated_at_utc", None)
    return identity, canonical_json_sha256(identity)


def _read_object(path: Path, *, name: str) -> tuple[dict[str, Any], str]:
    if path.stat().st_mode & 0o222:
        raise PositivePH1PreregistrationError(f"{name} must be read-only.")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise PositivePH1PreregistrationError(f"{name} must be a JSON object.")
    return payload, hashlib.sha256(raw).hexdigest()


def _validate_self_hash(payload: Mapping[str, object], *, key: str, name: str) -> None:
    unsigned = dict(payload)
    observed = unsigned.pop(key, None)
    if observed != canonical_json_sha256(unsigned):
        raise PositivePH1PreregistrationError(f"{name} self hash drifted.")


def _finite_matrix(value: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise PositivePH1PreregistrationError(f"{name} is invalid.")
    return array


def _spectral_norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=float), ord=2))


def _canonical_frame(
    projector: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int], dict[str, float]]:
    q = np.asarray(projector, dtype=float)
    candidates: list[tuple[float, int, int]] = []
    for p in range(q.shape[0] - 1):
        for r in range(p + 1, q.shape[0]):
            determinant = float(q[p, p] * q[r, r] - q[p, r] * q[r, p])
            candidates.append((determinant, p, r))
    maximum = max(item[0] for item in candidates)
    tied = [
        item
        for item in candidates
        if maximum - item[0] <= PIVOT_TIE_TOLERANCE * max(1.0, maximum)
    ]
    _determinant, p, r = min(tied, key=lambda item: (item[1], item[2]))
    selector = np.eye(q.shape[0], dtype=float)[:, [p, r]]
    pivot = selector.T @ q @ selector
    pivot = 0.5 * (pivot + pivot.T)
    condition = float(np.linalg.cond(pivot))
    if not math.isfinite(condition) or condition > MAXIMUM_MATRIX_CONDITION:
        raise PositivePH1PreregistrationError(
            "Canonical projector pivot is ill-conditioned."
        )
    try:
        lower = np.linalg.cholesky(pivot)
    except np.linalg.LinAlgError as exc:
        raise PositivePH1PreregistrationError(
            "Canonical projector pivot is not positive definite."
        ) from exc
    projected = q @ selector
    frame = np.linalg.solve(lower, projected.T).T
    diagnostics = {
        "maximum_principal_minor": maximum,
        "selected_principal_minor": float(q[p, p] * q[r, r] - q[p, r] * q[r, p]),
        "pivot_condition_number": condition,
    }
    return frame, (p, r), diagnostics


def build_target_blind_subspace(
    design: np.ndarray,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    label: str,
) -> dict[str, object]:
    """Construct and certify one target-blind stock-plus-two-mode subspace."""

    values = np.asarray(design, dtype=float)
    training = np.asarray(training_indices, dtype=int)
    validation = np.asarray(validation_indices, dtype=int)
    if (
        values.shape != (EXPECTED_WATER_COUNT, 18)
        or not np.all(np.isfinite(values))
        or training.ndim != 1
        or validation.ndim != 1
        or len(training) <= 18
        or set(training.tolist()).intersection(validation.tolist())
        or sorted(np.concatenate((training, validation)).tolist())
        != list(range(EXPECTED_WATER_COUNT))
    ):
        raise PositivePH1PreregistrationError(f"{label} row partition is invalid.")

    stock = np.asarray(
        SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2,
        dtype=float,
    )
    train_values = values[training]
    column_norms = np.linalg.norm(train_values, axis=0)
    if np.any(column_norms <= 0.0) or not np.all(np.isfinite(column_norms)):
        raise PositivePH1PreregistrationError(f"{label} has an inactive column.")
    standardized = train_values / column_norms
    standardized_singular = np.linalg.svd(standardized, compute_uv=False)
    standardized_condition = float(standardized_singular[0] / standardized_singular[-1])
    if (
        standardized_condition > MAXIMUM_MATRIX_CONDITION
        or standardized_singular[-1] <= RELATIVE_RANK_FLOOR * standardized_singular[0]
    ):
        raise PositivePH1PreregistrationError(
            f"{label} standardized geometry is rank-deficient or ill-conditioned."
        )

    stock_metric = column_norms * stock
    stock_metric_norm = float(np.linalg.norm(stock_metric))
    if not math.isfinite(stock_metric_norm) or stock_metric_norm <= 0.0:
        raise PositivePH1PreregistrationError(f"{label} stock direction is null.")
    stock_unit = stock_metric / stock_metric_norm
    complement = np.eye(18, dtype=float) - np.outer(stock_unit, stock_unit)
    operator = complement @ standardized.T @ standardized @ complement
    operator = 0.5 * (operator + operator.T)
    eigenvalues_ascending, eigenvectors_ascending = np.linalg.eigh(operator)
    order = np.argsort(eigenvalues_ascending)[::-1]
    eigenvalues = eigenvalues_ascending[order]
    eigenvectors = eigenvectors_ascending[:, order]
    leading = float(eigenvalues[0])
    if not math.isfinite(leading) or leading <= 0.0:
        raise PositivePH1PreregistrationError(f"{label} spectral operator is null.")
    if float(eigenvalues[-1]) < -RELATIVE_RANK_FLOOR * leading:
        raise PositivePH1PreregistrationError(
            f"{label} spectral operator has material negative curvature."
        )
    if float(eigenvalues[1]) <= RELATIVE_MODE_FLOOR * leading:
        raise PositivePH1PreregistrationError(f"{label} second mode is null.")
    if float(eigenvalues[1] - eigenvalues[2]) <= RELATIVE_MODE_GAP * leading:
        raise PositivePH1PreregistrationError(
            f"{label} rank-two spectral subspace is not separated."
        )

    raw_modes = eigenvectors[:, :2]
    projector = raw_modes @ raw_modes.T
    projector = 0.5 * (projector + projector.T)
    projector_idempotence = _spectral_norm(projector @ projector - projector)
    projector_trace_error = abs(float(np.trace(projector)) - 2.0)
    projector_stock_error = float(np.linalg.norm(projector @ stock_unit))
    if (
        projector_idempotence > PROJECTOR_TOLERANCE
        or projector_trace_error > PROJECTOR_TOLERANCE
        or projector_stock_error > PROJECTOR_TOLERANCE
    ):
        raise PositivePH1PreregistrationError(f"{label} projector check failed.")

    frame, pivot_pair, pivot_diagnostics = _canonical_frame(projector)
    frame_orthonormality = _spectral_norm(frame.T @ frame - np.eye(2))
    frame_stock_error = float(np.linalg.norm(frame.T @ stock_unit))
    frame_range_error = _spectral_norm(projector @ frame - frame)
    if (
        frame_orthonormality > PROJECTOR_TOLERANCE
        or frame_stock_error > PROJECTOR_TOLERANCE
        or frame_range_error > PROJECTOR_TOLERANCE
    ):
        raise PositivePH1PreregistrationError(f"{label} canonical frame check failed.")

    coefficient_modes = frame / column_norms[:, None]
    all_row_physical = np.column_stack((values @ stock, values @ coefficient_modes))
    optimizer_scales = np.linalg.norm(all_row_physical[training], axis=0)
    if np.any(optimizer_scales <= 0.0) or not np.all(np.isfinite(optimizer_scales)):
        raise PositivePH1PreregistrationError(
            f"{label} optimizer design has an inactive column."
        )
    all_row_optimizer = all_row_physical / optimizer_scales
    optimizer_singular = np.linalg.svd(all_row_optimizer[training], compute_uv=False)
    optimizer_condition = float(optimizer_singular[0] / optimizer_singular[-1])
    if (
        optimizer_condition > MAXIMUM_MATRIX_CONDITION
        or optimizer_singular[-1] <= RELATIVE_RANK_FLOOR * optimizer_singular[0]
    ):
        raise PositivePH1PreregistrationError(
            f"{label} optimizer design is rank-deficient or ill-conditioned."
        )

    return {
        "label": label,
        "training_row_indices": training.tolist(),
        "validation_row_indices": validation.tolist(),
        "column_l2_norms": column_norms.tolist(),
        "standardized_geometry_singular_values": standardized_singular.tolist(),
        "standardized_geometry_condition_number": standardized_condition,
        "stock_metric_vector": stock_metric.tolist(),
        "stock_metric_norm": stock_metric_norm,
        "stock_unit_vector": stock_unit.tolist(),
        "spectral_operator_eigenvalues_descending": eigenvalues.tolist(),
        "rank_two_projector": projector.tolist(),
        "rank_two_projector_sha256": canonical_json_sha256(projector.tolist()),
        "projector_idempotence_spectral_error": projector_idempotence,
        "projector_trace_error": projector_trace_error,
        "projector_stock_l2_error": projector_stock_error,
        "canonical_pivot_pair_zero_based": list(pivot_pair),
        **pivot_diagnostics,
        "canonical_frame": frame.tolist(),
        "canonical_frame_sha256": canonical_json_sha256(frame.tolist()),
        "canonical_frame_orthonormality_spectral_error": frame_orthonormality,
        "canonical_frame_stock_l2_error": frame_stock_error,
        "canonical_frame_range_spectral_error": frame_range_error,
        "physical_coefficient_modes": coefficient_modes.tolist(),
        "all_row_physical_design": all_row_physical.tolist(),
        "all_row_physical_design_sha256": canonical_json_sha256(
            all_row_physical.tolist()
        ),
        "training_optimizer_column_l2_scales": optimizer_scales.tolist(),
        "all_row_optimizer_design": all_row_optimizer.tolist(),
        "all_row_optimizer_design_sha256": canonical_json_sha256(
            all_row_optimizer.tolist()
        ),
        "training_optimizer_singular_values": optimizer_singular.tolist(),
        "training_optimizer_condition_number": optimizer_condition,
        "target_values_read_or_used": False,
        "validation_geometry_used_to_construct_subspace_or_scaling": False,
    }


def frozen_fit_contract() -> dict[str, object]:
    return {
        "model": "frozen_polarization_kcal_mol + X_positive_parent @ theta",
        "residual_target": "experimental_delta_g - frozen_polarization",
        "parameter_count": 3,
        "intercept": False,
        "regularizer": False,
        "loss": "pseudo-huber",
        "pseudo_huber_delta_kcal_mol": PSEUDO_HUBER_DELTA_KCAL_MOL,
        "fold_subspace": "training-geometry-only-stock-plus-two-canonical-modes",
        "deployment_subspace": "all-306-geometry-only-same-construction",
        "optimizer": "deterministic-safeguarded-newton-cholesky-armijo",
        "optimizer_starts": ["zero", "thin-svd-least-squares"],
        "armijo_c1": NEWTON_ARMIJO_C1,
        "backtrack_factor": NEWTON_BACKTRACK_FACTOR,
        "maximum_accepted_iterations": NEWTON_MAXIMUM_ACCEPTED_ITERATIONS,
        "maximum_armijo_trials": NEWTON_MAXIMUM_ARMIJO_TRIALS,
        "damping_exponents": list(NEWTON_DAMPING_EXPONENTS),
        "success_criterion": "two-independent-strict-minimizer-ball-certificates",
        "prediction_certificate_kcal_mol": PREDICTION_CERTIFICATE_KCAL_MOL,
        "fallback_estimator": False,
        "standard_state": "1M-ideal-gas-to-1M-ideal-solution",
        "standard_state_correction_kcal_mol": 0.0,
    }


def frozen_decision_rule() -> dict[str, object]:
    return {
        "water_oof_exact_minimizer_mae_upper_bound_kcal_mol": (
            PRIMARY_MAE_THRESHOLD_KCAL_MOL
        ),
        "mixed_505_water_oof_nonaqueous_m1_mae_threshold_kcal_mol": (
            MIXED_MAE_THRESHOLD_KCAL_MOL
        ),
        "water_numerical_upper_bound": (
            "endpoint_mae + tau_pred + max_representation_error + "
            "64*eps*max(delta,max_abs_error)"
        ),
        "mixed_505_numerical_upper_bound": (
            "endpoint_mae + (306/505)*(tau_pred+max_representation_error) + "
            "64*eps*max(delta,max_abs_error)"
        ),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_ucb": "report-only-development-diagnostic",
        "confirmation_partition_opened": False,
        "aspirational_one_kcal_has_branching_consequence": False,
        "terminal_policy": (
            "any pretarget gate, fold certificate, replay, reconstruction, OOF, "
            "mixed-505, or deployment failure terminates this hybrid accuracy route; "
            "no further CDS candidate"
        ),
    }


def frozen_pro_review() -> dict[str, object]:
    return {
        "verified_model": "Pro, 5 of 5.",
        "q10_prompt_sha256": PRO_Q10_PROMPT_SHA256,
        "q10_answer_sha256": PRO_Q10_ANSWER_SHA256,
        "q10_marker": "MOVE TO SMOOTH-AREA",
        "q11_prompt_sha256": PRO_Q11_PROMPT_SHA256,
        "q11_answer_sha256": PRO_Q11_ANSWER_SHA256,
        "q11_marker": "USE POSITIVE BERNSTEIN PARENT",
        "q12_conversation": "6a822a40-5ee4-83e8-84b8-8d2bbd86b973",
        "q12_prompt_sha256": PRO_Q12_PROMPT_SHA256,
        "q12_submission_evidence_sha256": PRO_Q12_SUBMISSION_SHA256,
        "q12_answer_sha256": PRO_Q12_ANSWER_SHA256,
        "q12_marker": "FREEZE PH1.0 CONTRACT",
        "local_verdict": "accepted-after-independent-derivation-and-pretarget-canaries",
    }


def _validate_feature_aggregate(
    path: Path,
) -> tuple[dict[str, Any], str, np.ndarray]:
    aggregate, file_sha256 = _read_object(path, name="positive-parent aggregate")
    _validate_self_hash(
        aggregate, key="aggregate_sha256", name="positive-parent aggregate"
    )
    expected = {
        "artifact": AGGREGATE_ARTIFACT,
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
    }
    for key, value in expected.items():
        if aggregate.get(key) != value:
            raise PositivePH1PreregistrationError(
                f"Positive-parent aggregate field {key!r} drifted."
            )
    records = aggregate.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_WATER_COUNT:
        raise PositivePH1PreregistrationError("Positive-parent aggregate rows drifted.")
    design = _finite_matrix(
        [record.get("design_row_angstrom2_div_1000") for record in records],
        shape=(EXPECTED_WATER_COUNT, 18),
        name="positive-parent design",
    )
    if aggregate.get("matrix_sha256") != canonical_json_sha256(design.tolist()):
        raise PositivePH1PreregistrationError("Positive-parent matrix digest drifted.")
    if aggregate.get("design_parameter_names") != list(
        SMD_WATER_TENSION_PARAMETER_NAMES
    ):
        raise PositivePH1PreregistrationError("Positive-parent basis names drifted.")
    return aggregate, file_sha256, design


def _validate_m3_preregistration(path: Path) -> tuple[dict[str, Any], str]:
    payload, file_sha256 = _read_object(path, name="M3 preregistration")
    _validate_self_hash(payload, key="self_sha256", name="M3 preregistration")
    for key, value in {
        "artifact": "route2-maple-cds-w1-m3-prereg-v1",
        "schema_version": 1,
        "status": "locked-before-first-m3-target-use-or-fit",
        "water_record_count": EXPECTED_WATER_COUNT,
        "experimental_targets_read_by_m3_preregistration": False,
        "experimental_targets_used_by_subspace_or_folds": False,
        "hybrid_prediction_records_read_by_m3_preregistration": False,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_opened": False,
        "fitting_or_calibration_performed": False,
        "self_sha256": M3_PREREGISTRATION_SELF_SHA256,
    }.items():
        if payload.get(key) != value:
            raise PositivePH1PreregistrationError(
                f"M3 preregistration field {key!r} drifted."
            )
    if file_sha256 != M3_PREREGISTRATION_FILE_SHA256:
        raise PositivePH1PreregistrationError("M3 preregistration file drifted.")
    return payload, file_sha256


def _validate_m4_terminal(
    preregistration_path: Path, failure_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg, prereg_sha = _read_object(preregistration_path, name="M4 preregistration")
    failure, failure_sha = _read_object(failure_path, name="M4 failure")
    _validate_self_hash(prereg, key="self_sha256", name="M4 preregistration")
    _validate_self_hash(failure, key="self_sha256", name="M4 failure")
    if (
        prereg_sha != M4_PREREGISTRATION_FILE_SHA256
        or prereg.get("self_sha256") != M4_PREREGISTRATION_SELF_SHA256
        or prereg.get("artifact") != "route2-maple-cds-w1-m4-ph1-prereg-v1"
        or prereg.get("status") != "locked-before-first-m4-target-use-or-fit"
    ):
        raise PositivePH1PreregistrationError("M4 preregistration drifted.")
    if (
        failure_sha != M4_FAILURE_FILE_SHA256
        or failure.get("self_sha256") != M4_FAILURE_SELF_SHA256
        or failure.get("artifact") != "route2-maple-cds-w1-m4-ph1-terminal-failure-v1"
        or failure.get("status") != "development-fail-numerical-certificate"
        or failure.get("oof_mae_computed") is not False
        or failure.get("confirmation_partition_opened") is not False
        or failure.get("threshold_solver_or_rule_changed_after_failure") is not False
    ):
        raise PositivePH1PreregistrationError("M4 terminal failure drifted.")
    return prereg, failure


def _write_json_exclusive_atomic(
    *, output: Path, payload: dict[str, object], snapshot: RepositorySnapshot
) -> None:
    try:
        output.relative_to(snapshot.root)
    except ValueError:
        pass
    else:
        raise PositivePH1PreregistrationError("Preregistration must be outside source.")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o400)
        try:
            os.link(temporary, output)
        except FileExistsError:
            raise
        output.chmod(0o400)
    finally:
        temporary.unlink(missing_ok=True)
    snapshot.assert_unchanged()


def create(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    if source_root != _SOURCE_ROOT:
        raise PositivePH1PreregistrationError(
            "PH1.0 creator must execute for its own source checkout."
        )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    snapshot = RepositorySnapshot.capture(source_root)
    runtime_identity, runtime_identity_sha256 = normalized_runtime_identity()

    aggregate_path = args.feature_aggregate.expanduser().resolve(strict=True)
    aggregate, aggregate_file_sha256, design = _validate_feature_aggregate(
        aggregate_path
    )
    m3_path = args.m3_preregistration.expanduser().resolve(strict=True)
    m3, m3_file_sha256 = _validate_m3_preregistration(m3_path)
    m4_path = args.m4_preregistration.expanduser().resolve(strict=True)
    m4_failure_path = args.m4_failure.expanduser().resolve(strict=True)
    _m4, m4_failure = _validate_m4_terminal(m4_path, m4_failure_path)

    grouped = m3.get("grouped_oof")
    if not isinstance(grouped, dict):
        raise PositivePH1PreregistrationError("M3 grouped folds are missing.")
    row_ledger = grouped.get("row_ledger")
    row_folds = np.asarray(grouped.get("row_fold_assignments"), dtype=int)
    if (
        not isinstance(row_ledger, list)
        or len(row_ledger) != EXPECTED_WATER_COUNT
        or row_folds.shape != (EXPECTED_WATER_COUNT,)
        or set(row_folds.tolist()) != set(range(OOF_FOLD_COUNT))
        or grouped.get("fold_record_counts") != [31, 31, 31, 31, 31, 31, 30, 30, 30, 30]
    ):
        raise PositivePH1PreregistrationError("M3 grouped fold identity drifted.")
    aggregate_records = aggregate["records"]
    for ordinal, (ledger, record) in enumerate(zip(row_ledger, aggregate_records)):
        if (
            not isinstance(ledger, dict)
            or not isinstance(record, dict)
            or ledger.get("water_ordinal") != ordinal
            or any(
                ledger.get(key) != record.get(key)
                for key in ("selection_index", "opaque_record_id", "geometry_sha256")
            )
            or ledger.get("fold") != int(row_folds[ordinal])
        ):
            raise PositivePH1PreregistrationError(
                "Positive-parent rows and frozen grouped folds disagree."
            )

    folds: list[dict[str, object]] = []
    all_indices = np.arange(EXPECTED_WATER_COUNT, dtype=int)
    for fold in range(OOF_FOLD_COUNT):
        validation = all_indices[row_folds == fold]
        training = all_indices[row_folds != fold]
        folds.append(
            build_target_blind_subspace(
                design,
                training,
                validation,
                label=f"fold-{fold}",
            )
        )
    deployment = build_target_blind_subspace(
        design,
        all_indices,
        np.empty(0, dtype=int),
        label="all-306-deployment",
    )

    evidence_root = Path(str(m3.get("parent_hybrid_evidence_root"))).resolve(
        strict=True
    )
    records_path = Path(str(m3.get("parent_hybrid_records_path"))).resolve(strict=True)
    integrity_audit_path = (
        evidence_root / "audits/independent-integrity-audit.json"
    ).resolve(strict=True)
    source_files = committed_source_hashes(snapshot, _SOURCE_FILES)
    stock = SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2.tolist()
    payload: dict[str, object] = {
        "artifact": PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-first-positive-ph1-target-use-or-fit",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_profile_id": CANDIDATE_PROFILE_ID,
        "partition": "development-water-only",
        "water_record_count": EXPECTED_WATER_COUNT,
        "mixed_record_count": EXPECTED_MIXED_COUNT,
        "source_root": str(source_root),
        "source_git_head": snapshot.head,
        "source_git_tree": snapshot.tree,
        "source_files_sha256": source_files,
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": runtime_identity_sha256,
        "feature_aggregate_path": str(aggregate_path),
        "feature_aggregate_file_sha256": aggregate_file_sha256,
        "feature_aggregate_self_sha256": aggregate["aggregate_sha256"],
        "feature_matrix_sha256": aggregate["matrix_sha256"],
        "feature_water_identity_sha256": aggregate["water_identity_sha256"],
        "m3_preregistration_path": str(m3_path),
        "m3_preregistration_file_sha256": m3_file_sha256,
        "m3_preregistration_self_sha256": m3["self_sha256"],
        "m4_preregistration_path": str(m4_path),
        "m4_preregistration_file_sha256": M4_PREREGISTRATION_FILE_SHA256,
        "m4_preregistration_self_sha256": M4_PREREGISTRATION_SELF_SHA256,
        "m4_failure_path": str(m4_failure_path),
        "m4_failure_file_sha256": M4_FAILURE_FILE_SHA256,
        "m4_failure_self_sha256": m4_failure["self_sha256"],
        "m4_oof_mae_computed": False,
        "m4_terminal_rule_changed_after_failure": False,
        "parent_hybrid_evidence_root": str(evidence_root),
        "parent_hybrid_records_path": str(records_path),
        "parent_hybrid_integrity_audit_path": str(integrity_audit_path),
        "parent_hybrid_integrity_audit_sha256": sha256_file(integrity_audit_path),
        "row_ledger": row_ledger,
        "row_fold_assignments": row_folds.tolist(),
        "fold_record_counts": grouped["fold_record_counts"],
        "family_count": grouped["family_count"],
        "maximum_family_size": grouped["maximum_family_size"],
        "stock_coefficients_cal_mol_angstrom2": stock,
        "stock_coefficients_sha256": canonical_json_sha256(stock),
        "design_parameter_names": list(SMD_WATER_TENSION_PARAMETER_NAMES),
        "fold_subspaces": folds,
        "deployment_subspace": deployment,
        "fold_subspaces_sha256": canonical_json_sha256(folds),
        "deployment_subspace_sha256": canonical_json_sha256(deployment),
        "experimental_targets_read_by_preregistration": False,
        "hybrid_prediction_records_read_by_preregistration": False,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_opened": False,
        "fitting_or_calibration_performed": False,
        "fit_contract": frozen_fit_contract(),
        "development_decision_rule": frozen_decision_rule(),
        "external_pro_math_review": frozen_pro_review(),
        "claim_boundary": (
            "Final target-blind positive-parent PH1.0 development candidate. "
            "All fold and deployment geometry subspaces are frozen before target "
            "joining. This is not an accuracy result, confirmation, force, or "
            "public Route-2 capability."
        ),
    }
    payload["self_sha256"] = canonical_json_sha256(payload)
    if set(payload) != PREREGISTRATION_KEYS:
        raise PositivePH1PreregistrationError(
            "Internal PH1.0 preregistration schema drifted."
        )
    _write_json_exclusive_atomic(output=output, payload=payload, snapshot=snapshot)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--feature-aggregate", type=Path, required=True)
    parser.add_argument("--m3-preregistration", type=Path, required=True)
    parser.add_argument("--m4-preregistration", type=Path, required=True)
    parser.add_argument("--m4-failure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    payload = create(parser.parse_args())
    print(
        json.dumps(
            {
                "artifact": payload["artifact"],
                "status": payload["status"],
                "candidate_profile_id": payload["candidate_profile_id"],
                "feature_matrix_sha256": payload["feature_matrix_sha256"],
                "fold_subspaces_sha256": payload["fold_subspaces_sha256"],
                "deployment_subspace_sha256": payload["deployment_subspace_sha256"],
                "self_sha256": payload["self_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
