#!/usr/bin/env python3
"""Run the preregistered M4-PH1.0 grouped-OOF development evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import minimize

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))

from maple.solvation.release.evidence import (
    RepositorySnapshot,
    canonical_json_sha256,
    sha256_file,
)
from tools.route2_release.analyze_hybrid_smd_components_v3 import (
    EXPECTED_RECORD_COUNT,
    _error_metrics,
    _load_bound_records,
)
from tools.route2_release.create_maple_cds_w1_m3_preregistration import (
    EXPECTED_WATER_COUNT,
    OOF_FOLD_COUNT,
)
from tools.route2_release.create_maple_cds_w1_m4_preregistration import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CANDIDATE_PROFILE_ID,
    CURVATURE_RELATIVE_FLOOR,
    INDEPENDENT_SOLUTION_RELATIVE_TOLERANCE,
    MAXIMUM_SOLVER_ITERATIONS,
    M3_FAILURE_FILE_SHA256,
    M3_FAILURE_SELF_SHA256,
    M3_PREREGISTRATION_FILE_SHA256,
    M3_PREREGISTRATION_SELF_SHA256,
    MINIMUM_MAE_IMPROVEMENT_KCAL_MOL,
    OBJECTIVE_REPRODUCIBILITY_RELATIVE_TOLERANCE,
    PREDICTION_REPRODUCIBILITY_KCAL_MOL,
    PREREGISTRATION_ARTIFACT,
    PREREGISTRATION_KEYS,
    PRIMARY_MAE_THRESHOLD_KCAL_MOL,
    PSEUDO_HUBER_DELTA_KCAL_MOL,
    RESULT_ARTIFACT,
    STANDARDIZED_CONDITION_LIMIT,
    STATIONARITY_INFINITY_TOLERANCE,
    TRUST_EXACT_GRADIENT_TOLERANCE,
    _SOURCE_FILES,
    frozen_development_decision_rule,
    frozen_fit_contract,
    frozen_prior_development_disclosure,
    frozen_pro_review,
    normalized_runtime_identity,
    validate_m3_terminal_parent,
)
from tools.route2_release.fit_maple_cds_w1_m3_lad import (
    _cluster_bootstrap_mae,
    _write_json_exclusive_atomic,
)


class M4FitError(RuntimeError):
    """Raised when M4 fitting or a numerical certificate fails closed."""


def _finite_array(value: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise M4FitError(f"{name} is invalid.")
    return result


def _validate_self_hash(payload: dict[str, Any], *, key: str, name: str) -> None:
    unsigned = dict(payload)
    observed = unsigned.pop(key, None)
    if observed != canonical_json_sha256(unsigned):
        raise M4FitError(f"{name} self hash drifted.")


def _pseudo_huber_value_gradient_hessian(
    coefficients: np.ndarray,
    design: np.ndarray,
    response_over_delta: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate mean dimensionless pseudo-Huber loss and exact derivatives."""

    values = np.asarray(design, dtype=float)
    target = np.asarray(response_over_delta, dtype=float)
    point = np.asarray(coefficients, dtype=float)
    if (
        values.ndim != 2
        or point.shape != (values.shape[1],)
        or target.shape != (values.shape[0],)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(point))
    ):
        raise M4FitError("Pseudo-Huber evaluation inputs are invalid.")
    q = target - values @ point
    h = np.hypot(1.0, q)
    loss_terms = np.empty_like(q)
    safe_square = np.abs(q) <= math.sqrt(np.finfo(float).max) / 2.0
    loss_terms[safe_square] = q[safe_square] * q[safe_square] / (h[safe_square] + 1.0)
    loss_terms[~safe_square] = h[~safe_square] - 1.0
    sample_count = float(values.shape[0])
    score = q / h
    weights = 1.0 / (h * h * h)
    objective = float(np.sum(loss_terms) / sample_count)
    gradient = -(values.T @ score) / sample_count
    hessian = values.T @ (weights[:, None] * values) / sample_count
    if (
        not math.isfinite(objective)
        or not np.all(np.isfinite(gradient))
        or not np.all(np.isfinite(hessian))
    ):
        raise M4FitError("Pseudo-Huber evaluation is non-finite.")
    return objective, gradient, hessian


def _solve_from_start(
    *, design: np.ndarray, response_over_delta: np.ndarray, start: np.ndarray, name: str
) -> dict[str, object]:
    def evaluate(point: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        return _pseudo_huber_value_gradient_hessian(point, design, response_over_delta)

    result = minimize(
        fun=lambda point: evaluate(point)[0],
        x0=np.asarray(start, dtype=float),
        jac=lambda point: evaluate(point)[1],
        hess=lambda point: evaluate(point)[2],
        method="trust-exact",
        options={
            "gtol": TRUST_EXACT_GRADIENT_TOLERANCE,
            "maxiter": MAXIMUM_SOLVER_ITERATIONS,
        },
    )
    if not result.success or result.x is None:
        raise M4FitError(f"Pseudo-Huber {name} solve failed: {result.message}")
    point = np.asarray(result.x, dtype=float)
    objective, gradient, hessian = evaluate(point)
    if abs(float(result.fun) - objective) > 1.0e-12 * max(1.0, abs(objective)):
        raise M4FitError(f"Pseudo-Huber {name} objective replay drifted.")
    gradient_norm = float(np.max(np.abs(gradient)))
    if gradient_norm > STATIONARITY_INFINITY_TOLERANCE:
        raise M4FitError(f"Pseudo-Huber {name} stationarity failed.")
    eigenvalues = np.linalg.eigvalsh(hessian)
    if not np.all(np.isfinite(eigenvalues)):
        raise M4FitError(f"Pseudo-Huber {name} curvature is non-finite.")
    curvature_floor = CURVATURE_RELATIVE_FLOOR * max(1.0, float(eigenvalues[-1]))
    if float(eigenvalues[0]) < curvature_floor:
        raise M4FitError(f"Pseudo-Huber {name} curvature is unresolved.")
    try:
        np.linalg.cholesky(hessian)
    except np.linalg.LinAlgError as exc:
        raise M4FitError(
            f"Pseudo-Huber {name} Hessian is not positive definite."
        ) from exc
    return {
        "name": name,
        "scaled_coefficients": point.tolist(),
        "normalized_objective": objective,
        "gradient_infinity_norm": gradient_norm,
        "hessian_eigenvalues": eigenvalues.tolist(),
        "hessian_curvature_floor": curvature_floor,
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "gradient_evaluations": int(result.njev),
        "hessian_evaluations": int(result.nhev),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
    }


def fit_unique_pseudo_huber(
    design: np.ndarray,
    response: np.ndarray,
    *,
    validation_design: np.ndarray | None = None,
) -> dict[str, object]:
    """Fit M4-PH1.0 and certify the unique floating-point solution."""

    values = np.asarray(design, dtype=float)
    target = np.asarray(response, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or values.shape[0] <= values.shape[1]
        or target.shape != (values.shape[0],)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(target))
    ):
        raise M4FitError("Pseudo-Huber training data are invalid.")
    held_out = None
    if validation_design is not None:
        held_out = np.asarray(validation_design, dtype=float)
        if (
            held_out.ndim != 2
            or held_out.shape[1] != 3
            or not np.all(np.isfinite(held_out))
        ):
            raise M4FitError("Pseudo-Huber validation design is invalid.")

    column_scales = np.linalg.norm(values, axis=0)
    if np.any(column_scales <= 0.0) or not np.all(np.isfinite(column_scales)):
        raise M4FitError("Pseudo-Huber design has an inactive column.")
    standardized = values / column_scales
    singular = np.linalg.svd(standardized, compute_uv=False)
    rank_floor = (
        100.0 * np.finfo(float).eps * max(standardized.shape) * float(singular[0])
    )
    if float(singular[-1]) <= rank_floor:
        raise M4FitError("Pseudo-Huber standardized design is rank deficient.")
    condition = float(singular[0] / singular[-1])
    if condition > STANDARDIZED_CONDITION_LIMIT:
        raise M4FitError("Pseudo-Huber standardized design is ill-conditioned.")

    normalized_target = target / PSEUDO_HUBER_DELTA_KCAL_MOL
    zero = np.zeros(3, dtype=float)
    ols = np.linalg.lstsq(standardized, normalized_target, rcond=None)[0]
    zero_result = _solve_from_start(
        design=standardized,
        response_over_delta=normalized_target,
        start=zero,
        name="zero-start",
    )
    ols_result = _solve_from_start(
        design=standardized,
        response_over_delta=normalized_target,
        start=ols,
        name="svd-ols-start",
    )
    scaled_zero = np.asarray(zero_result["scaled_coefficients"], dtype=float)
    scaled_ols = np.asarray(ols_result["scaled_coefficients"], dtype=float)
    coefficient_difference = float(np.max(np.abs(scaled_zero - scaled_ols)))
    allowed_coefficient_difference = INDEPENDENT_SOLUTION_RELATIVE_TOLERANCE * max(
        1.0,
        float(np.max(np.abs(scaled_zero))),
        float(np.max(np.abs(scaled_ols))),
    )
    if coefficient_difference > allowed_coefficient_difference:
        raise M4FitError("Pseudo-Huber independent starts disagree in coefficient.")

    coefficients_zero = PSEUDO_HUBER_DELTA_KCAL_MOL * scaled_zero / column_scales
    coefficients_ols = PSEUDO_HUBER_DELTA_KCAL_MOL * scaled_ols / column_scales
    training_prediction_difference = float(
        np.max(np.abs(values @ coefficients_zero - values @ coefficients_ols))
    )
    validation_prediction_difference = 0.0
    if held_out is not None and len(held_out):
        validation_prediction_difference = float(
            np.max(np.abs(held_out @ coefficients_zero - held_out @ coefficients_ols))
        )
    if (
        training_prediction_difference > PREDICTION_REPRODUCIBILITY_KCAL_MOL
        or validation_prediction_difference > PREDICTION_REPRODUCIBILITY_KCAL_MOL
    ):
        raise M4FitError("Pseudo-Huber independent-start predictions disagree.")
    objective_difference = abs(
        float(zero_result["normalized_objective"])
        - float(ols_result["normalized_objective"])
    )
    objective_allowed = OBJECTIVE_REPRODUCIBILITY_RELATIVE_TOLERANCE * max(
        1.0,
        abs(float(zero_result["normalized_objective"])),
        abs(float(ols_result["normalized_objective"])),
    )
    if objective_difference > objective_allowed:
        raise M4FitError("Pseudo-Huber independent-start objectives disagree.")

    prediction = values @ coefficients_zero
    if not np.all(np.isfinite(coefficients_zero)) or not np.all(
        np.isfinite(prediction)
    ):
        raise M4FitError("Pseudo-Huber coefficient or prediction is non-finite.")
    return {
        "reduced_coefficients": coefficients_zero.tolist(),
        "scaled_coefficients": scaled_zero.tolist(),
        "pseudo_huber_delta_kcal_mol": PSEUDO_HUBER_DELTA_KCAL_MOL,
        "training_column_l2_scales": column_scales.tolist(),
        "training_standardized_singular_values": singular.tolist(),
        "training_standardized_rank_floor": rank_floor,
        "training_standardized_condition_number": condition,
        "training_objective_sum_kcal_mol_equivalent": (
            float(zero_result["normalized_objective"])
            * values.shape[0]
            * PSEUDO_HUBER_DELTA_KCAL_MOL**2
        ),
        "zero_start_certificate": zero_result,
        "svd_ols_start_certificate": ols_result,
        "independent_start_scaled_coefficient_max_abs_difference": (
            coefficient_difference
        ),
        "independent_start_allowed_scaled_coefficient_difference": (
            allowed_coefficient_difference
        ),
        "independent_start_training_prediction_max_abs_difference_kcal_mol": (
            training_prediction_difference
        ),
        "independent_start_validation_prediction_max_abs_difference_kcal_mol": (
            validation_prediction_difference
        ),
        "independent_start_normalized_objective_difference": objective_difference,
        "independent_start_allowed_normalized_objective_difference": (
            objective_allowed
        ),
    }


def _read_preregistration(path: Path, source_root: Path) -> tuple[dict[str, Any], str]:
    if path.stat().st_mode & 0o222:
        raise M4FitError("M4 preregistration must be read-only.")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise M4FitError("M4 preregistration is not an object.")
    _validate_self_hash(payload, key="self_sha256", name="M4 preregistration")
    if set(payload) != PREREGISTRATION_KEYS:
        raise M4FitError("M4 preregistration schema drifted.")
    for key, value in {
        "artifact": PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-first-m4-target-use-or-fit",
        "candidate_profile_id": CANDIDATE_PROFILE_ID,
        "partition": "development-water-only",
        "water_record_count": EXPECTED_WATER_COUNT,
        "source_root": str(source_root),
        "experimental_targets_read_by_m4_preregistration": False,
        "hybrid_prediction_records_read_by_m4_preregistration": False,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_opened": False,
        "fitting_or_calibration_performed": False,
        "m3_oof_mae_computed": False,
        "m3_threshold_or_rule_changed_after_failure": False,
    }.items():
        if payload.get(key) != value:
            raise M4FitError(f"M4 preregistration field {key!r} drifted.")
    if payload.get("fit_contract") != frozen_fit_contract():
        raise M4FitError("M4 preregistration fit contract drifted.")
    if payload.get("development_decision_rule") != frozen_development_decision_rule():
        raise M4FitError("M4 preregistration decision rule drifted.")
    if (
        payload.get("prior_development_information_disclosed")
        != frozen_prior_development_disclosure()
    ):
        raise M4FitError("M4 preregistration prior disclosure drifted.")
    if payload.get("external_pro_math_review") != frozen_pro_review():
        raise M4FitError("M4 preregistration Pro-review binding drifted.")
    for key, value in {
        "m3_preregistration_file_sha256": M3_PREREGISTRATION_FILE_SHA256,
        "m3_preregistration_self_sha256": M3_PREREGISTRATION_SELF_SHA256,
        "m3_failure_file_sha256": M3_FAILURE_FILE_SHA256,
        "m3_failure_self_sha256": M3_FAILURE_SELF_SHA256,
        "m3_terminal_error": "Primary LAD coefficient 0 is numerically non-unique.",
    }.items():
        if payload.get(key) != value:
            raise M4FitError(f"M4 preregistration field {key!r} drifted.")
    source_files = payload.get("source_files_sha256")
    if not isinstance(source_files, dict) or set(source_files) != set(_SOURCE_FILES):
        raise M4FitError("M4 preregistration source manifest drifted.")
    return payload, hashlib.sha256(raw).hexdigest()


def fit(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    if source_root != _SOURCE_ROOT:
        raise M4FitError("M4 fitter must run for its own checkout.")
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise M4FitError("M4 fit output must be outside source.")
    if output.exists():
        raise FileExistsError(output)

    snapshot = RepositorySnapshot.capture(source_root)
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    preregistration, preregistration_file_sha = _read_preregistration(
        preregistration_path, source_root
    )
    if preregistration.get("source_git_head") != snapshot.head:
        raise M4FitError("M4 preregistration Git head drifted.")
    if preregistration.get("source_git_tree") != snapshot.tree:
        raise M4FitError("M4 preregistration Git tree drifted.")
    for relative, digest in dict(preregistration["source_files_sha256"]).items():
        if sha256_file(source_root / relative) != digest:
            raise M4FitError(f"M4 source file {relative!r} drifted.")
    runtime_identity, runtime_identity_sha256 = normalized_runtime_identity()
    if preregistration.get("runtime_identity") != runtime_identity:
        raise M4FitError("M4 numerical runtime identity drifted.")
    if preregistration.get("runtime_identity_sha256") != runtime_identity_sha256:
        raise M4FitError("M4 numerical runtime digest drifted.")

    m3_preregistration_path = Path(
        str(preregistration["m3_preregistration_path"])
    ).resolve(strict=True)
    m3_failure_path = Path(str(preregistration["m3_failure_path"])).resolve(strict=True)
    m3_preregistration, _failure = validate_m3_terminal_parent(
        m3_preregistration_path=m3_preregistration_path,
        m3_failure_path=m3_failure_path,
    )

    evidence_root = Path(
        str(m3_preregistration["parent_hybrid_evidence_root"])
    ).resolve(strict=True)
    input_dir = args.input_dir.expanduser().resolve(strict=True)
    if (
        input_dir
        != Path(str(m3_preregistration["parent_hybrid_records_path"])).resolve()
    ):
        raise M4FitError("M4 records are outside the bound hybrid evidence root.")
    audit_path = args.integrity_audit.expanduser().resolve(strict=True)
    if audit_path != evidence_root / "audits/independent-integrity-audit.json":
        raise M4FitError("M4 audit is outside the bound hybrid evidence root.")
    audit_raw = audit_path.read_bytes()
    audit = json.loads(audit_raw)
    if not isinstance(audit, dict):
        raise M4FitError("Hybrid integrity audit is invalid.")
    audit_unsigned = dict(audit)
    observed_verification = audit_unsigned.pop("verification_sha256", None)
    if observed_verification != canonical_json_sha256(audit_unsigned):
        raise M4FitError("Hybrid integrity-audit self hash drifted.")
    if audit.get("preregistration_sha256") != m3_preregistration.get(
        "parent_hybrid_preregistration_file_sha256"
    ):
        raise M4FitError("Hybrid audit is not bound to the M4 parent profile.")
    records = _load_bound_records(
        input_dir=input_dir,
        audit=audit,
        expected_count=EXPECTED_RECORD_COUNT,
    )

    grouped = dict(m3_preregistration["grouped_oof"])
    row_ledger = grouped.get("row_ledger")
    if not isinstance(row_ledger, list) or len(row_ledger) != EXPECTED_WATER_COUNT:
        raise M4FitError("M4 row ledger is incomplete.")
    water_records: list[dict[str, Any]] = []
    for ordinal, ledger in enumerate(row_ledger):
        if not isinstance(ledger, dict) or ledger.get("water_ordinal") != ordinal:
            raise M4FitError("M4 row ledger ordering drifted.")
        selection_index = ledger.get("selection_index")
        if isinstance(selection_index, bool) or not isinstance(selection_index, int):
            raise M4FitError("M4 selection index is invalid.")
        record = records[selection_index]
        if (
            any(
                record.get(key) != ledger.get(key)
                for key in ("selection_index", "opaque_record_id", "geometry_sha256")
            )
            or record.get("canonical_solvent") != "water"
        ):
            raise M4FitError("M4 row ledger and hybrid record disagree.")
        water_records.append(record)

    subspace = dict(m3_preregistration["subspace"])
    design = _finite_array(
        subspace.get("reduced_design"),
        shape=(EXPECTED_WATER_COUNT, 3),
        name="M4 reduced design",
    )
    if subspace.get("reduced_design_sha256") != canonical_json_sha256(design.tolist()):
        raise M4FitError("M4 reduced-design digest drifted.")
    row_folds = np.asarray(grouped.get("row_fold_assignments"), dtype=int)
    if row_folds.shape != (EXPECTED_WATER_COUNT,) or set(row_folds.tolist()) != set(
        range(OOF_FOLD_COUNT)
    ):
        raise M4FitError("M4 fold assignments drifted.")

    experimental = np.asarray(
        [float(row["experimental_delta_g_kcal_mol"]) for row in water_records]
    )
    electrostatic = np.asarray(
        [float(row["continuum_polarization_kcal_mol"]) for row in water_records]
    )
    stock = np.asarray([float(row["smd_cds_kcal_mol"]) for row in water_records])
    target = experimental - electrostatic
    if not all(
        np.all(np.isfinite(values))
        for values in (experimental, electrostatic, stock, target)
    ):
        raise M4FitError("M4 target ledger is non-finite.")

    oof_cds = np.full(EXPECTED_WATER_COUNT, np.nan)
    fold_results: list[dict[str, object]] = []
    for fold in range(OOF_FOLD_COUNT):
        training = row_folds != fold
        validation = row_folds == fold
        result = fit_unique_pseudo_huber(
            design[training],
            target[training],
            validation_design=design[validation],
        )
        coefficients = np.asarray(result["reduced_coefficients"], dtype=float)
        oof_cds[validation] = design[validation] @ coefficients
        fold_results.append(
            {
                "fold": fold,
                "training_count": int(np.count_nonzero(training)),
                "validation_count": int(np.count_nonzero(validation)),
                "fit": result,
            }
        )
    if not np.all(np.isfinite(oof_cds)):
        raise M4FitError("M4 did not produce one finite OOF prediction per row.")

    m0_prediction = electrostatic
    m1_prediction = electrostatic + stock
    m4_prediction = electrostatic + oof_cds
    m0 = _error_metrics(m0_prediction.tolist(), experimental.tolist())
    m1 = _error_metrics(m1_prediction.tolist(), experimental.tolist())
    m4 = _error_metrics(m4_prediction.tolist(), experimental.tolist())
    best_baseline_mae = min(
        float(m0["mean_absolute_error_kcal_mol"]),
        float(m1["mean_absolute_error_kcal_mol"]),
    )
    improvement = best_baseline_mae - float(m4["mean_absolute_error_kcal_mol"])

    families = grouped.get("families")
    if not isinstance(families, list):
        raise M4FitError("M4 family ledger is invalid.")
    family_members = [family["member_indices"] for family in families]
    absolute_errors = np.abs(m4_prediction - experimental)
    bootstrap = _cluster_bootstrap_mae(
        absolute_errors=absolute_errors,
        families=family_members,
    )
    if (
        bootstrap.get("replicates") != BOOTSTRAP_REPLICATES
        or bootstrap.get("seed") != BOOTSTRAP_SEED
    ):
        raise M4FitError("M4 clustered-bootstrap contract drifted.")
    overall_mae = float(np.mean(absolute_errors))
    family_influence: list[dict[str, object]] = []
    for family in families:
        members = np.asarray(family["member_indices"], dtype=int)
        keep = np.ones(EXPECTED_WATER_COUNT, dtype=bool)
        keep[members] = False
        without = float(np.mean(absolute_errors[keep]))
        family_influence.append(
            {
                "family_key": family["family_key"],
                "family_size": len(members),
                "mae_without_family_kcal_mol": without,
                "absolute_mae_change_kcal_mol": abs(without - overall_mae),
            }
        )
    family_influence.sort(
        key=lambda item: (-item["absolute_mae_change_kcal_mol"], item["family_key"])
    )

    oof_by_selection = {
        int(ledger["selection_index"]): float(m4_prediction[ordinal])
        for ordinal, ledger in enumerate(row_ledger)
    }
    mixed_prediction: list[float] = []
    mixed_experimental: list[float] = []
    for index, record in enumerate(records):
        if index in oof_by_selection:
            mixed_prediction.append(oof_by_selection[index])
        else:
            mixed_prediction.append(
                float(record["continuum_polarization_kcal_mol"])
                + float(record["smd_cds_kcal_mol"])
            )
        mixed_experimental.append(float(record["experimental_delta_g_kcal_mol"]))
    mixed_metrics = _error_metrics(mixed_prediction, mixed_experimental)
    passed = (
        float(m4["mean_absolute_error_kcal_mol"]) <= PRIMARY_MAE_THRESHOLD_KCAL_MOL
        and float(mixed_metrics["mean_absolute_error_kcal_mol"])
        <= PRIMARY_MAE_THRESHOLD_KCAL_MOL
        and improvement >= MINIMUM_MAE_IMPROVEMENT_KCAL_MOL
    )

    deployable_fit = None
    if passed:
        deployable_fit = fit_unique_pseudo_huber(design, target)
        reduced_coefficients = np.asarray(
            deployable_fit["reduced_coefficients"], dtype=float
        )
        stock_coefficients = np.asarray(subspace["stock_coefficients"], dtype=float)
        coefficient_modes = np.asarray(
            subspace["geometry_modes_coefficient_coordinates"], dtype=float
        )
        deployable_fit["atomic_surface_tension_coefficients_cal_mol_angstrom2"] = (
            reduced_coefficients[0] * stock_coefficients
            + coefficient_modes @ reduced_coefficients[1:]
        ).tolist()
        deployable_fit["claim_boundary"] = (
            "All-306 development refit for future candidate implementation; "
            "not an OOF validation prediction and not public admission."
        )

    result_payload: dict[str, object] = {
        "artifact": RESULT_ARTIFACT,
        "schema_version": 1,
        "status": "development-pass" if passed else "development-fail",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_profile_id": CANDIDATE_PROFILE_ID,
        "partition": "development-water-only",
        "fitting_or_calibration_performed": True,
        "fitted_parameter_count": 3,
        "fitted_intercept": False,
        "pseudo_huber_delta_kcal_mol": PSEUDO_HUBER_DELTA_KCAL_MOL,
        "confirmation_partition_opened": False,
        "source_git_head": snapshot.head,
        "source_git_tree": snapshot.tree,
        "runtime_identity_sha256": runtime_identity_sha256,
        "preregistration_path": str(preregistration_path),
        "preregistration_file_sha256": preregistration_file_sha,
        "m3_preregistration_file_sha256": preregistration[
            "m3_preregistration_file_sha256"
        ],
        "m3_failure_file_sha256": preregistration["m3_failure_file_sha256"],
        "integrity_audit_path": str(audit_path),
        "integrity_audit_file_sha256": hashlib.sha256(audit_raw).hexdigest(),
        "record_files_manifest_sha256": audit["record_files_manifest_sha256"],
        "water_baselines": {"m0": m0, "m1": m1},
        "water_m4_grouped_oof": m4,
        "water_oof_mae_improvement_over_better_baseline_kcal_mol": improvement,
        "mixed_505_water_m4_oof_nonaqueous_m1": mixed_metrics,
        "development_decision": "pass" if passed else "fail",
        "fold_results": fold_results,
        "cluster_bootstrap": bootstrap,
        "maximum_family_influence": family_influence[0],
        "family_influence": family_influence,
        "all_306_deployable_candidate_fit": deployable_fit,
        "claim_boundary": (
            "Development-only grouped OOF stock-area M4-PH1.0 energy result. "
            "It is not M3, confirmation, a production smooth-area profile, "
            "a force, or a public Route-2 capability."
        ),
    }
    result_payload["self_sha256"] = canonical_json_sha256(result_payload)
    _write_json_exclusive_atomic(
        output=output, payload=result_payload, snapshot=snapshot
    )
    return result_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--integrity-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    payload = fit(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
