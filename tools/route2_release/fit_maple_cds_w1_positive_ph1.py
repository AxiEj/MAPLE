#!/usr/bin/env python3
"""Run the one terminal positive-parent PH1.0 grouped-OOF development fit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2,
)
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
from tools.route2_release.create_maple_cds_w1_positive_ph1_preregistration import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CANDIDATE_PROFILE_ID,
    EXPECTED_MIXED_COUNT,
    EXPECTED_WATER_COUNT,
    MAXIMUM_MATRIX_CONDITION,
    MIXED_MAE_THRESHOLD_KCAL_MOL,
    NEWTON_ARMIJO_C1,
    NEWTON_BACKTRACK_FACTOR,
    NEWTON_DAMPING_EXPONENTS,
    NEWTON_MAXIMUM_ACCEPTED_ITERATIONS,
    NEWTON_MAXIMUM_ARMIJO_TRIALS,
    OOF_FOLD_COUNT,
    PREDICTION_CERTIFICATE_KCAL_MOL,
    PREREGISTRATION_ARTIFACT,
    PREREGISTRATION_KEYS,
    PRIMARY_MAE_THRESHOLD_KCAL_MOL,
    PSEUDO_HUBER_DELTA_KCAL_MOL,
    REPRESENTATION_TOLERANCE_KCAL_MOL,
    RESULT_ARTIFACT,
    _SOURCE_FILES,
    _validate_feature_aggregate,
    _validate_m3_preregistration,
    _validate_m4_terminal,
    build_target_blind_subspace,
    frozen_decision_rule,
    frozen_fit_contract,
    frozen_pro_review,
    normalized_runtime_identity,
)
from tools.route2_release.fit_maple_cds_w1_m3_lad import (
    _write_json_exclusive_atomic,
)


class PositivePH1FitError(RuntimeError):
    """Raised when the terminal PH1.0 fit or certificate fails closed."""


def _validate_self_hash(payload: Mapping[str, object], *, key: str, name: str) -> None:
    unsigned = dict(payload)
    observed = unsigned.pop(key, None)
    if observed != canonical_json_sha256(unsigned):
        raise PositivePH1FitError(f"{name} self hash drifted.")


def _finite_array(value: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise PositivePH1FitError(f"{name} is invalid.")
    return array


def _dot3(row: np.ndarray, point: np.ndarray) -> float:
    return math.fsum(float(row[index]) * float(point[index]) for index in range(3))


def _evaluate_pseudo_huber(
    point: np.ndarray,
    design: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(design, dtype=float)
    coefficients = np.asarray(point, dtype=float)
    response = np.asarray(target, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or coefficients.shape != (3,)
        or response.shape != (values.shape[0],)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(coefficients))
        or not np.all(np.isfinite(response))
    ):
        raise PositivePH1FitError("Pseudo-Huber evaluation inputs are invalid.")
    delta = PSEUDO_HUBER_DELTA_KCAL_MOL
    residuals = np.asarray(
        [_dot3(row, coefficients) - float(y) for row, y in zip(values, response)],
        dtype=float,
    )
    hypot = np.hypot(1.0, residuals / delta)
    loss_terms = (residuals * residuals) / (hypot + 1.0)
    scores = residuals / hypot
    weights = 1.0 / (hypot * hypot * hypot)
    objective = math.fsum(float(value) for value in loss_terms)
    gradient = np.asarray(
        [
            math.fsum(
                float(values[row, column]) * float(scores[row])
                for row in range(values.shape[0])
            )
            for column in range(3)
        ],
        dtype=float,
    )
    hessian = np.empty((3, 3), dtype=float)
    for left in range(3):
        for right in range(left, 3):
            value = math.fsum(
                float(values[row, left])
                * float(weights[row])
                * float(values[row, right])
                for row in range(values.shape[0])
            )
            hessian[left, right] = value
            hessian[right, left] = value
    if (
        not math.isfinite(objective)
        or not np.all(np.isfinite(gradient))
        or not np.all(np.isfinite(hessian))
        or not np.all(np.isfinite(residuals))
    ):
        raise PositivePH1FitError("Pseudo-Huber evaluation became non-finite.")
    return objective, gradient, hessian, residuals


def _weighted_gram(design: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(design, dtype=float)
    factors = np.asarray(weights, dtype=float)
    result = np.empty((3, 3), dtype=float)
    for left in range(3):
        for right in range(left, 3):
            value = math.fsum(
                float(values[row, left])
                * float(factors[row])
                * float(values[row, right])
                for row in range(values.shape[0])
            )
            result[left, right] = value
            result[right, left] = value
    return result


def _minimizer_certificate(
    *,
    point: np.ndarray,
    training_design: np.ndarray,
    all_row_design: np.ndarray,
    target: np.ndarray,
) -> tuple[bool, dict[str, object]]:
    objective, gradient, hessian, residuals = _evaluate_pseudo_huber(
        point,
        training_design,
        target,
    )
    epsilon = np.finfo(float).eps
    all_row_norms = np.linalg.norm(all_row_design, axis=1)
    q_plus_all = (1.0 + 32.0 * epsilon) * all_row_norms
    q_max = float(np.max(q_plus_all))
    if not math.isfinite(q_max) or q_max <= 0.0:
        raise PositivePH1FitError("Prediction certificate row norm is invalid.")
    radius = PREDICTION_CERTIFICATE_KCAL_MOL / q_max
    q_plus_training = (1.0 + 32.0 * epsilon) * np.linalg.norm(
        training_design,
        axis=1,
    )
    upper_residual = np.abs(residuals) + q_plus_training * radius
    lower_weights = (1.0 - 32.0 * epsilon) * (
        1.0 + (upper_residual / PSEUDO_HUBER_DELTA_KCAL_MOL) ** 2
    ) ** (-1.5)
    if np.any(lower_weights <= 0.0) or not np.all(np.isfinite(lower_weights)):
        raise PositivePH1FitError("Prediction certificate weights are invalid.")
    lower_hessian = _weighted_gram(training_design, lower_weights)
    lower_eigenvalues = np.linalg.eigvalsh(lower_hessian)
    endpoint_eigenvalues = np.linalg.eigvalsh(hessian)
    if not np.all(np.isfinite(lower_eigenvalues)) or not np.all(
        np.isfinite(endpoint_eigenvalues)
    ):
        raise PositivePH1FitError("Prediction certificate curvature is non-finite.")
    lower_minimum = float(lower_eigenvalues[0])
    lower_condition = (
        float(lower_eigenvalues[-1] / lower_minimum)
        if lower_minimum > 0.0
        else math.inf
    )
    endpoint_minimum = float(endpoint_eigenvalues[0])
    endpoint_condition = (
        float(endpoint_eigenvalues[-1] / endpoint_minimum)
        if endpoint_minimum > 0.0
        else math.inf
    )
    lower_cholesky = True
    endpoint_cholesky = True
    try:
        np.linalg.cholesky(lower_hessian)
    except np.linalg.LinAlgError:
        lower_cholesky = False
    try:
        np.linalg.cholesky(hessian)
    except np.linalg.LinAlgError:
        endpoint_cholesky = False
    gradient_norm = float(np.linalg.norm(gradient))
    strict_gradient_bound = 0.5 * lower_minimum * radius
    passed = bool(
        lower_minimum > 0.0
        and lower_cholesky
        and endpoint_cholesky
        and lower_condition <= MAXIMUM_MATRIX_CONDITION
        and endpoint_condition <= MAXIMUM_MATRIX_CONDITION
        and gradient_norm < strict_gradient_bound
    )
    return passed, {
        "objective_sum_kcal_mol2": objective,
        "gradient_l2_norm": gradient_norm,
        "prediction_q_max": q_max,
        "certified_coefficient_radius": radius,
        "strict_gradient_bound": strict_gradient_bound,
        "lower_hessian_eigenvalues": lower_eigenvalues.tolist(),
        "lower_hessian_condition_number": lower_condition,
        "endpoint_hessian_eigenvalues": endpoint_eigenvalues.tolist(),
        "endpoint_hessian_condition_number": endpoint_condition,
        "lower_hessian_cholesky": lower_cholesky,
        "endpoint_hessian_cholesky": endpoint_cholesky,
        "certificate_passed": passed,
    }


def _thin_svd_least_squares(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    left, singular, right_transpose = np.linalg.svd(design, full_matrices=False)
    if singular.shape != (3,) or singular[-1] <= 0.0:
        raise PositivePH1FitError("Least-squares start is rank deficient.")
    return right_transpose.T @ ((left.T @ target) / singular)


def _solve_one_start(
    *,
    training_design: np.ndarray,
    all_row_design: np.ndarray,
    target: np.ndarray,
    start: np.ndarray,
    name: str,
) -> dict[str, object]:
    point = np.asarray(start, dtype=float).copy()
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise PositivePH1FitError(f"{name} start is invalid.")
    epsilon = np.finfo(float).eps
    accepted_iterations = 0
    function_evaluations = 0
    damping_history: list[float] = []
    backtrack_history: list[int] = []

    while True:
        objective, gradient, hessian, _residuals = _evaluate_pseudo_huber(
            point,
            training_design,
            target,
        )
        function_evaluations += 1
        certified, certificate = _minimizer_certificate(
            point=point,
            training_design=training_design,
            all_row_design=all_row_design,
            target=target,
        )
        function_evaluations += 1
        if certified:
            return {
                "name": name,
                "endpoint": point.tolist(),
                "accepted_iterations": accepted_iterations,
                "function_evaluations": function_evaluations,
                "damping_history": damping_history,
                "armijo_backtrack_history": backtrack_history,
                "certificate": certificate,
            }
        if accepted_iterations >= NEWTON_MAXIMUM_ACCEPTED_ITERATIONS:
            raise PositivePH1FitError(
                f"{name} did not certify within the accepted-iteration limit."
            )

        eigenvalues = np.linalg.eigvalsh(hessian)
        if not np.all(np.isfinite(eigenvalues)):
            raise PositivePH1FitError(f"{name} Hessian is non-finite.")
        hessian_scale = max(1.0, float(eigenvalues[-1]))
        damping_candidates = [0.0] + [
            (10.0**exponent) * hessian_scale for exponent in NEWTON_DAMPING_EXPONENTS
        ]
        direction = None
        selected_damping = None
        directional_derivative = None
        for damping in damping_candidates:
            damped = hessian + damping * np.eye(3)
            try:
                lower = np.linalg.cholesky(damped)
            except np.linalg.LinAlgError:
                continue
            trial_direction = np.linalg.solve(
                lower.T,
                np.linalg.solve(lower, -gradient),
            )
            trial_derivative = float(np.dot(gradient, trial_direction))
            descent_floor = (
                -100.0
                * epsilon
                * float(np.linalg.norm(gradient))
                * float(np.linalg.norm(trial_direction))
            )
            if (
                np.all(np.isfinite(trial_direction))
                and math.isfinite(trial_derivative)
                and trial_derivative < descent_floor
            ):
                direction = trial_direction
                selected_damping = float(damping)
                directional_derivative = trial_derivative
                break
        if (
            direction is None
            or selected_damping is None
            or directional_derivative is None
        ):
            raise PositivePH1FitError(f"{name} found no certified descent direction.")

        accepted_point = None
        accepted_trial = None
        for trial in range(NEWTON_MAXIMUM_ARMIJO_TRIALS):
            step = NEWTON_BACKTRACK_FACTOR**trial
            candidate = point + step * direction
            candidate_objective, _g, _h, _r = _evaluate_pseudo_huber(
                candidate,
                training_design,
                target,
            )
            function_evaluations += 1
            if candidate_objective <= (
                objective + NEWTON_ARMIJO_C1 * step * directional_derivative
            ):
                accepted_point = candidate
                accepted_trial = trial
                break
        if accepted_point is None or accepted_trial is None:
            raise PositivePH1FitError(f"{name} Armijo line search failed.")

        step_norm = float(np.linalg.norm(accepted_point - point))
        previous_norm = float(np.linalg.norm(point))
        point = accepted_point
        accepted_iterations += 1
        damping_history.append(selected_damping)
        backtrack_history.append(accepted_trial)
        if step_norm <= 32.0 * epsilon * max(
            PSEUDO_HUBER_DELTA_KCAL_MOL,
            previous_norm,
        ):
            certified, certificate = _minimizer_certificate(
                point=point,
                training_design=training_design,
                all_row_design=all_row_design,
                target=target,
            )
            function_evaluations += 1
            if certified:
                return {
                    "name": name,
                    "endpoint": point.tolist(),
                    "accepted_iterations": accepted_iterations,
                    "function_evaluations": function_evaluations,
                    "damping_history": damping_history,
                    "armijo_backtrack_history": backtrack_history,
                    "certificate": certificate,
                }
            raise PositivePH1FitError(f"{name} stagnated without certification.")


def fit_certified_pseudo_huber(
    *,
    all_row_design: np.ndarray,
    training_indices: np.ndarray,
    target_all_rows: np.ndarray,
) -> dict[str, object]:
    values = np.asarray(all_row_design, dtype=float)
    training = np.asarray(training_indices, dtype=int)
    target = np.asarray(target_all_rows, dtype=float)
    if (
        values.shape != (EXPECTED_WATER_COUNT, 3)
        or target.shape != (EXPECTED_WATER_COUNT,)
        or training.ndim != 1
        or len(training) <= 3
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(target))
    ):
        raise PositivePH1FitError("Certified pseudo-Huber fit inputs are invalid.")
    training_design = values[training]
    training_target = target[training]
    singular = np.linalg.svd(training_design, compute_uv=False)
    condition = float(singular[0] / singular[-1])
    if (
        singular[-1] <= 0.0
        or condition > MAXIMUM_MATRIX_CONDITION
        or singular[-1] <= 1.0e-12 * singular[0]
    ):
        raise PositivePH1FitError("Certified optimizer design gate failed.")
    starts = {
        "zero-start": np.zeros(3, dtype=float),
        "svd-least-squares-start": _thin_svd_least_squares(
            training_design,
            training_target,
        ),
    }
    results = {
        name: _solve_one_start(
            training_design=training_design,
            all_row_design=values,
            target=training_target,
            start=start,
            name=name,
        )
        for name, start in starts.items()
    }
    zero = np.asarray(results["zero-start"]["endpoint"], dtype=float)
    least_squares = np.asarray(
        results["svd-least-squares-start"]["endpoint"],
        dtype=float,
    )
    radius = float(results["zero-start"]["certificate"]["certified_coefficient_radius"])
    replay_distance = float(np.linalg.norm(zero - least_squares))
    replay_allowance = 2.0 * radius + 100.0 * np.finfo(float).eps * max(
        1.0,
        float(np.linalg.norm(zero)),
        float(np.linalg.norm(least_squares)),
    )
    if replay_distance > replay_allowance:
        raise PositivePH1FitError("Two-start certified minimizer balls do not overlap.")
    return {
        "authoritative_scaled_endpoint": zero.tolist(),
        "training_optimizer_singular_values": singular.tolist(),
        "training_optimizer_condition_number": condition,
        "zero_start": results["zero-start"],
        "svd_least_squares_start": results["svd-least-squares-start"],
        "two_start_endpoint_l2_distance": replay_distance,
        "two_start_replay_allowance": replay_allowance,
    }


def _read_preregistration(
    path: Path,
    source_root: Path,
) -> tuple[dict[str, Any], str]:
    if path.stat().st_mode & 0o222:
        raise PositivePH1FitError("PH1.0 preregistration must be read-only.")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise PositivePH1FitError("PH1.0 preregistration must be an object.")
    _validate_self_hash(payload, key="self_sha256", name="PH1.0 preregistration")
    if set(payload) != PREREGISTRATION_KEYS:
        raise PositivePH1FitError("PH1.0 preregistration schema drifted.")
    expected = {
        "artifact": PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-first-positive-ph1-target-use-or-fit",
        "candidate_profile_id": CANDIDATE_PROFILE_ID,
        "partition": "development-water-only",
        "water_record_count": EXPECTED_WATER_COUNT,
        "mixed_record_count": EXPECTED_MIXED_COUNT,
        "source_root": str(source_root),
        "experimental_targets_read_by_preregistration": False,
        "hybrid_prediction_records_read_by_preregistration": False,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_opened": False,
        "fitting_or_calibration_performed": False,
        "m4_oof_mae_computed": False,
        "m4_terminal_rule_changed_after_failure": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PositivePH1FitError(f"PH1.0 preregistration {key!r} drifted.")
    if payload.get("fit_contract") != frozen_fit_contract():
        raise PositivePH1FitError("PH1.0 fit contract drifted.")
    if payload.get("development_decision_rule") != frozen_decision_rule():
        raise PositivePH1FitError("PH1.0 decision rule drifted.")
    if payload.get("external_pro_math_review") != frozen_pro_review():
        raise PositivePH1FitError("PH1.0 Pro-review binding drifted.")
    source_files = payload.get("source_files_sha256")
    if not isinstance(source_files, dict) or set(source_files) != set(_SOURCE_FILES):
        raise PositivePH1FitError("PH1.0 source manifest drifted.")
    return payload, hashlib.sha256(raw).hexdigest()


def _subspace_to_theta_and_prediction(
    *,
    subspace: Mapping[str, object],
    scaled_endpoint: np.ndarray,
    design: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    stock = _finite_array(
        SMD_STOCK,
        shape=(18,),
        name="stock coefficients",
    )
    modes = _finite_array(
        subspace.get("physical_coefficient_modes"),
        shape=(18, 2),
        name="physical coefficient modes",
    )
    scales = _finite_array(
        subspace.get("training_optimizer_column_l2_scales"),
        shape=(3,),
        name="optimizer scales",
    )
    point = np.asarray(scaled_endpoint, dtype=float) / scales
    theta = point[0] * stock + modes @ point[1:]
    physical_prediction = design @ theta
    optimizer_design = _finite_array(
        subspace.get("all_row_optimizer_design"),
        shape=(EXPECTED_WATER_COUNT, 3),
        name="optimizer design",
    )
    optimizer_prediction = optimizer_design @ scaled_endpoint
    representation_error = float(
        np.max(np.abs(physical_prediction - optimizer_prediction))
    )
    if representation_error > REPRESENTATION_TOLERANCE_KCAL_MOL:
        raise PositivePH1FitError("Physical and optimizer predictions disagree.")
    return theta, physical_prediction, representation_error, point


SMD_STOCK = np.asarray(
    SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2,
    dtype=float,
)


def _cluster_bootstrap_mae(
    *, absolute_errors: np.ndarray, families: list[list[int]]
) -> dict[str, object]:
    errors = np.asarray(absolute_errors, dtype=float)
    blocks = [np.asarray(members, dtype=int) for members in families]
    if (
        errors.shape != (EXPECTED_WATER_COUNT,)
        or not np.all(np.isfinite(errors))
        or not blocks
        or sorted(np.concatenate(blocks).tolist()) != list(range(EXPECTED_WATER_COUNT))
    ):
        raise PositivePH1FitError("PH1.0 clustered-bootstrap ledger is invalid.")
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        choices = generator.integers(0, len(blocks), size=len(blocks))
        sampled = np.concatenate([blocks[index] for index in choices])
        estimates[replicate] = float(np.mean(errors[sampled]))
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "sampling_unit": "preregistered-exact-heavy-element-count-family",
        "refits_performed": False,
        "mean_kcal_mol": float(np.mean(estimates)),
        "standard_deviation_kcal_mol": float(np.std(estimates, ddof=1)),
        "q025_kcal_mol": float(np.quantile(estimates, 0.025, method="linear")),
        "q950_kcal_mol": float(np.quantile(estimates, 0.95, method="linear")),
        "q975_kcal_mol": float(np.quantile(estimates, 0.975, method="linear")),
        "claim_boundary": (
            "Fixed-OOF-error clustered sampling diagnostic; not full training-"
            "procedure uncertainty and not a confirmation result."
        ),
    }


def fit(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    if source_root != _SOURCE_ROOT:
        raise PositivePH1FitError("PH1.0 fitter must run for its own checkout.")
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise PositivePH1FitError("PH1.0 result must be outside source.")
    if output.exists():
        raise FileExistsError(output)

    snapshot = RepositorySnapshot.capture(source_root)
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    preregistration, preregistration_file_sha256 = _read_preregistration(
        preregistration_path,
        source_root,
    )
    if preregistration.get("source_git_head") != snapshot.head:
        raise PositivePH1FitError("PH1.0 preregistration Git head drifted.")
    if preregistration.get("source_git_tree") != snapshot.tree:
        raise PositivePH1FitError("PH1.0 preregistration Git tree drifted.")
    for relative, digest in dict(preregistration["source_files_sha256"]).items():
        if sha256_file(source_root / relative) != digest:
            raise PositivePH1FitError(f"PH1.0 source {relative!r} drifted.")
    runtime_identity, runtime_identity_sha256 = normalized_runtime_identity()
    if preregistration.get("runtime_identity") != runtime_identity:
        raise PositivePH1FitError("PH1.0 numerical runtime drifted.")
    if preregistration.get("runtime_identity_sha256") != runtime_identity_sha256:
        raise PositivePH1FitError("PH1.0 runtime digest drifted.")

    aggregate_path = Path(str(preregistration["feature_aggregate_path"])).resolve(
        strict=True
    )
    aggregate, aggregate_file_sha256, design = _validate_feature_aggregate(
        aggregate_path
    )
    for key, value in {
        "feature_aggregate_file_sha256": aggregate_file_sha256,
        "feature_aggregate_self_sha256": aggregate["aggregate_sha256"],
        "feature_matrix_sha256": aggregate["matrix_sha256"],
        "feature_water_identity_sha256": aggregate["water_identity_sha256"],
    }.items():
        if preregistration.get(key) != value:
            raise PositivePH1FitError(f"PH1.0 aggregate binding {key!r} drifted.")

    m3_path = Path(str(preregistration["m3_preregistration_path"])).resolve(strict=True)
    m3, m3_file_sha256 = _validate_m3_preregistration(m3_path)
    if preregistration.get("m3_preregistration_file_sha256") != m3_file_sha256:
        raise PositivePH1FitError("PH1.0 M3 binding drifted.")
    _validate_m4_terminal(
        Path(str(preregistration["m4_preregistration_path"])).resolve(strict=True),
        Path(str(preregistration["m4_failure_path"])).resolve(strict=True),
    )

    row_ledger = preregistration.get("row_ledger")
    row_folds = np.asarray(preregistration.get("row_fold_assignments"), dtype=int)
    if (
        not isinstance(row_ledger, list)
        or len(row_ledger) != EXPECTED_WATER_COUNT
        or row_folds.shape != (EXPECTED_WATER_COUNT,)
    ):
        raise PositivePH1FitError("PH1.0 row ledger drifted.")
    all_indices = np.arange(EXPECTED_WATER_COUNT, dtype=int)
    expected_folds = []
    for fold in range(OOF_FOLD_COUNT):
        expected_folds.append(
            build_target_blind_subspace(
                design,
                all_indices[row_folds != fold],
                all_indices[row_folds == fold],
                label=f"fold-{fold}",
            )
        )
    expected_deployment = build_target_blind_subspace(
        design,
        all_indices,
        np.empty(0, dtype=int),
        label="all-306-deployment",
    )
    if preregistration.get("fold_subspaces") != expected_folds:
        raise PositivePH1FitError("PH1.0 frozen fold subspaces drifted.")
    if preregistration.get("deployment_subspace") != expected_deployment:
        raise PositivePH1FitError("PH1.0 frozen deployment subspace drifted.")

    input_dir = args.input_dir.expanduser().resolve(strict=True)
    if input_dir != Path(str(preregistration["parent_hybrid_records_path"])).resolve(
        strict=True
    ):
        raise PositivePH1FitError("PH1.0 records path drifted.")
    audit_path = args.integrity_audit.expanduser().resolve(strict=True)
    if audit_path != Path(
        str(preregistration["parent_hybrid_integrity_audit_path"])
    ).resolve(strict=True):
        raise PositivePH1FitError("PH1.0 integrity-audit path drifted.")
    audit_raw = audit_path.read_bytes()
    if (
        preregistration.get("parent_hybrid_integrity_audit_sha256")
        != hashlib.sha256(audit_raw).hexdigest()
    ):
        raise PositivePH1FitError("PH1.0 integrity-audit digest drifted.")
    audit = json.loads(audit_raw)
    if not isinstance(audit, dict):
        raise PositivePH1FitError("PH1.0 integrity audit is invalid.")
    audit_unsigned = dict(audit)
    observed_verification = audit_unsigned.pop("verification_sha256", None)
    if observed_verification != canonical_json_sha256(audit_unsigned):
        raise PositivePH1FitError("PH1.0 integrity-audit self hash drifted.")
    if audit.get("preregistration_sha256") != m3.get(
        "parent_hybrid_preregistration_file_sha256"
    ):
        raise PositivePH1FitError("PH1.0 audit belongs to another hybrid parent.")
    records = _load_bound_records(
        input_dir=input_dir,
        audit=audit,
        expected_count=EXPECTED_RECORD_COUNT,
    )

    water_records: list[dict[str, Any]] = []
    for ordinal, ledger in enumerate(row_ledger):
        if not isinstance(ledger, dict) or ledger.get("water_ordinal") != ordinal:
            raise PositivePH1FitError("PH1.0 water ledger ordering drifted.")
        selection_index = ledger.get("selection_index")
        if isinstance(selection_index, bool) or not isinstance(selection_index, int):
            raise PositivePH1FitError("PH1.0 selection index is invalid.")
        record = records[selection_index]
        if (
            any(
                record.get(key) != ledger.get(key)
                for key in ("selection_index", "opaque_record_id", "geometry_sha256")
            )
            or record.get("canonical_solvent") != "water"
        ):
            raise PositivePH1FitError("PH1.0 feature and hybrid rows disagree.")
        water_records.append(record)

    experimental = np.asarray(
        [float(record["experimental_delta_g_kcal_mol"]) for record in water_records],
        dtype=float,
    )
    polarization = np.asarray(
        [float(record["continuum_polarization_kcal_mol"]) for record in water_records],
        dtype=float,
    )
    target = experimental - polarization
    if not np.all(np.isfinite(target)):
        raise PositivePH1FitError("PH1.0 residual target is non-finite.")

    oof_cds = np.full(EXPECTED_WATER_COUNT, np.nan, dtype=float)
    fold_results: list[dict[str, object]] = []
    maximum_representation_error = 0.0
    for fold, subspace in enumerate(expected_folds):
        training = all_indices[row_folds != fold]
        validation = all_indices[row_folds == fold]
        optimizer_design = _finite_array(
            subspace["all_row_optimizer_design"],
            shape=(EXPECTED_WATER_COUNT, 3),
            name=f"fold {fold} optimizer design",
        )
        certificate = fit_certified_pseudo_huber(
            all_row_design=optimizer_design,
            training_indices=training,
            target_all_rows=target,
        )
        endpoint = np.asarray(
            certificate["authoritative_scaled_endpoint"],
            dtype=float,
        )
        theta, physical_prediction, representation_error, physical_point = (
            _subspace_to_theta_and_prediction(
                subspace=subspace,
                scaled_endpoint=endpoint,
                design=design,
            )
        )
        maximum_representation_error = max(
            maximum_representation_error,
            representation_error,
        )
        oof_cds[validation] = physical_prediction[validation]
        fold_results.append(
            {
                "fold": fold,
                "training_count": len(training),
                "validation_count": len(validation),
                "physical_parameters_alpha_beta": physical_point.tolist(),
                "atomic_surface_tension_coefficients_cal_mol_angstrom2": (
                    theta.tolist()
                ),
                "representation_max_abs_error_kcal_mol": representation_error,
                "certificate": certificate,
            }
        )
    if not np.all(np.isfinite(oof_cds)):
        raise PositivePH1FitError("PH1.0 OOF coverage is incomplete.")

    oof_total = polarization + oof_cds
    absolute_errors = np.abs(oof_total - experimental)
    oof_mae = math.fsum(float(value) for value in absolute_errors) / float(
        EXPECTED_WATER_COUNT
    )
    epsilon = np.finfo(float).eps
    accumulation_allowance = (
        64.0
        * epsilon
        * max(
            PSEUDO_HUBER_DELTA_KCAL_MOL,
            float(np.max(absolute_errors)),
        )
    )
    water_mae_upper = (
        oof_mae
        + PREDICTION_CERTIFICATE_KCAL_MOL
        + maximum_representation_error
        + accumulation_allowance
    )

    oof_by_selection = {
        int(ledger["selection_index"]): float(oof_total[ordinal])
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
    mixed_accumulation_allowance = (
        64.0
        * epsilon
        * max(
            PSEUDO_HUBER_DELTA_KCAL_MOL,
            max(
                abs(prediction - reference)
                for prediction, reference in zip(mixed_prediction, mixed_experimental)
            ),
        )
    )
    mixed_mae_upper = (
        float(mixed_metrics["mean_absolute_error_kcal_mol"])
        + (EXPECTED_WATER_COUNT / EXPECTED_MIXED_COUNT)
        * (PREDICTION_CERTIFICATE_KCAL_MOL + maximum_representation_error)
        + (mixed_accumulation_allowance)
    )
    passed = bool(
        water_mae_upper <= PRIMARY_MAE_THRESHOLD_KCAL_MOL
        and mixed_mae_upper <= MIXED_MAE_THRESHOLD_KCAL_MOL
    )

    deployment = None
    if passed:
        deployment_optimizer = _finite_array(
            expected_deployment["all_row_optimizer_design"],
            shape=(EXPECTED_WATER_COUNT, 3),
            name="deployment optimizer design",
        )
        deployment_certificate = fit_certified_pseudo_huber(
            all_row_design=deployment_optimizer,
            training_indices=all_indices,
            target_all_rows=target,
        )
        deployment_endpoint = np.asarray(
            deployment_certificate["authoritative_scaled_endpoint"],
            dtype=float,
        )
        theta, _prediction, representation_error, physical_point = (
            _subspace_to_theta_and_prediction(
                subspace=expected_deployment,
                scaled_endpoint=deployment_endpoint,
                design=design,
            )
        )
        deployment = {
            "physical_parameters_alpha_beta": physical_point.tolist(),
            "atomic_surface_tension_coefficients_cal_mol_angstrom2": theta.tolist(),
            "representation_max_abs_error_kcal_mol": representation_error,
            "certificate": deployment_certificate,
            "claim_boundary": (
                "All-306 development refit after the grouped-OOF pass. It is not "
                "an OOF prediction, confirmation evidence, force admission, or "
                "public Route-2 capability."
            ),
        }

    m0 = _error_metrics(polarization.tolist(), experimental.tolist())
    m1 = _error_metrics(
        [
            float(record["continuum_polarization_kcal_mol"])
            + float(record["smd_cds_kcal_mol"])
            for record in water_records
        ],
        experimental.tolist(),
    )
    families = m3["grouped_oof"]["families"]
    if not isinstance(families, list):
        raise PositivePH1FitError("PH1.0 family ledger is invalid.")
    bootstrap = _cluster_bootstrap_mae(
        absolute_errors=absolute_errors,
        families=[family["member_indices"] for family in families],
    )

    payload: dict[str, object] = {
        "artifact": RESULT_ARTIFACT,
        "schema_version": 1,
        "status": "development-pass" if passed else "development-fail",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_profile_id": CANDIDATE_PROFILE_ID,
        "partition": "development-water-only",
        "fitting_or_calibration_performed": True,
        "confirmation_partition_opened": False,
        "source_git_head": snapshot.head,
        "source_git_tree": snapshot.tree,
        "runtime_identity_sha256": runtime_identity_sha256,
        "preregistration_path": str(preregistration_path),
        "preregistration_file_sha256": preregistration_file_sha256,
        "feature_aggregate_file_sha256": aggregate_file_sha256,
        "feature_matrix_sha256": aggregate["matrix_sha256"],
        "integrity_audit_path": str(audit_path),
        "integrity_audit_file_sha256": hashlib.sha256(audit_raw).hexdigest(),
        "record_files_manifest_sha256": audit["record_files_manifest_sha256"],
        "water_baselines": {"m0": m0, "m1": m1},
        "water_positive_ph1_grouped_oof_endpoint_metrics": _error_metrics(
            oof_total.tolist(),
            experimental.tolist(),
        ),
        "water_positive_ph1_oof_mae_math_fsum_kcal_mol": oof_mae,
        "water_positive_ph1_oof_exact_minimizer_mae_upper_bound_kcal_mol": (
            water_mae_upper
        ),
        "mixed_505_water_positive_ph1_oof_nonaqueous_m1": mixed_metrics,
        "mixed_505_exact_minimizer_mae_upper_bound_kcal_mol": mixed_mae_upper,
        "prediction_certificate_kcal_mol": PREDICTION_CERTIFICATE_KCAL_MOL,
        "maximum_representation_error_kcal_mol": maximum_representation_error,
        "water_accumulation_allowance_kcal_mol": accumulation_allowance,
        "development_decision": "pass" if passed else "fail",
        "fold_results": fold_results,
        "cluster_bootstrap": bootstrap,
        "all_306_deployment_candidate_fit": deployment,
        "terminal_policy_applies": True,
        "claim_boundary": (
            "Development-only grouped-OOF positive-parent PH1.0 energy result. "
            "It is not independent confirmation, force/Hessian admission, strict "
            "Tier V, or a public Route-2 capability."
        ),
    }
    payload["self_sha256"] = canonical_json_sha256(payload)
    _write_json_exclusive_atomic(output=output, payload=payload, snapshot=snapshot)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--integrity-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    payload = fit(parser.parse_args())
    print(
        json.dumps(
            {
                "artifact": payload["artifact"],
                "status": payload["status"],
                "water_mae_upper_kcal_mol": payload[
                    "water_positive_ph1_oof_exact_minimizer_mae_upper_bound_kcal_mol"
                ],
                "mixed_505_mae_upper_kcal_mol": payload[
                    "mixed_505_exact_minimizer_mae_upper_bound_kcal_mol"
                ],
                "development_decision": payload["development_decision"],
                "self_sha256": payload["self_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
