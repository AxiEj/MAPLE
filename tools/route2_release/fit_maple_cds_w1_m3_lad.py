#!/usr/bin/env python3
"""Run the preregistered grouped-OOF three-parameter M3 LAD evaluation."""

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
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))

from maple.solvation.release.evidence import (
    RepositorySnapshot,
    canonical_json_sha256,
    sha256_file,
)

try:
    from tools.route2_release.analyze_hybrid_smd_components_v3 import (
        EXPECTED_RECORD_COUNT,
        _error_metrics,
        _load_bound_records,
    )
    from tools.route2_release.create_maple_cds_w1_m3_preregistration import (
        BOOTSTRAP_REPLICATES,
        BOOTSTRAP_SEED,
        CANDIDATE_PROFILE_ID,
        EXPECTED_WATER_COUNT,
        HIGHS_DUAL_FEASIBILITY_TOLERANCE,
        HIGHS_PRIMAL_FEASIBILITY_TOLERANCE,
        LAD_COEFFICIENT_RANGE_RELATIVE_TOLERANCE,
        LAD_OBJECTIVE_CAP_SCALE,
        MINIMUM_MAE_IMPROVEMENT_KCAL_MOL,
        OOF_FOLD_COUNT,
        PREREGISTRATION_KEYS,
        PREREGISTRATION_ARTIFACT,
        PRIMARY_MAE_THRESHOLD_KCAL_MOL,
        _normalized_runtime_identity,
        _PRO_REVIEWS,
        _SOURCE_FILES,
        frozen_development_decision_rule,
        frozen_fit_contract,
    )
except ModuleNotFoundError:
    from analyze_hybrid_smd_components_v3 import (
        EXPECTED_RECORD_COUNT,
        _error_metrics,
        _load_bound_records,
    )
    from create_maple_cds_w1_m3_preregistration import (
        BOOTSTRAP_REPLICATES,
        BOOTSTRAP_SEED,
        CANDIDATE_PROFILE_ID,
        EXPECTED_WATER_COUNT,
        HIGHS_DUAL_FEASIBILITY_TOLERANCE,
        HIGHS_PRIMAL_FEASIBILITY_TOLERANCE,
        LAD_COEFFICIENT_RANGE_RELATIVE_TOLERANCE,
        LAD_OBJECTIVE_CAP_SCALE,
        MINIMUM_MAE_IMPROVEMENT_KCAL_MOL,
        OOF_FOLD_COUNT,
        PREREGISTRATION_KEYS,
        PREREGISTRATION_ARTIFACT,
        PRIMARY_MAE_THRESHOLD_KCAL_MOL,
        _normalized_runtime_identity,
        _PRO_REVIEWS,
        _SOURCE_FILES,
        frozen_development_decision_rule,
        frozen_fit_contract,
    )


RESULT_ARTIFACT = "route2-maple-cds-w1-m3-grouped-oof-lad-v1"
_CERTIFICATE_RELATIVE_TOLERANCE = 1.0e-7


class M3FitError(RuntimeError):
    """Raised when M3 fitting or its numerical certificate fails closed."""


def _finite_array(value: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise M3FitError(f"{name} is invalid.")
    return result


def _validate_self_hash(payload: Mapping[str, Any], *, key: str, name: str) -> None:
    unsigned = dict(payload)
    observed = unsigned.pop(key, None)
    if observed != canonical_json_sha256(unsigned):
        raise M3FitError(f"{name} self hash drifted.")


def _lp_problem(design: np.ndarray, response: np.ndarray):
    count, parameter_count = design.shape
    identity = np.eye(count)
    a_ub = np.block(
        [
            [design, -identity],
            [-design, -identity],
        ]
    )
    b_ub = np.concatenate((response, -response))
    objective = np.concatenate((np.zeros(parameter_count), np.ones(count)))
    bounds = [(None, None)] * parameter_count + [(0.0, None)] * count
    return a_ub, b_ub, objective, bounds


def _solve_lp(
    *,
    objective: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    bounds: Sequence[tuple[float | None, float | None]],
    name: str,
):
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs-ds",
        options={
            "presolve": True,
            "primal_feasibility_tolerance": HIGHS_PRIMAL_FEASIBILITY_TOLERANCE,
            "dual_feasibility_tolerance": HIGHS_DUAL_FEASIBILITY_TOLERANCE,
        },
    )
    if not result.success or result.status != 0 or result.x is None:
        raise M3FitError(f"{name} did not terminate optimally: {result.message}")
    if not np.all(np.isfinite(result.x)) or not math.isfinite(float(result.fun)):
        raise M3FitError(f"{name} returned non-finite values.")
    maximum_violation = float(np.max(a_ub @ result.x - b_ub))
    tolerance = _CERTIFICATE_RELATIVE_TOLERANCE * max(1.0, float(np.max(np.abs(b_ub))))
    if maximum_violation > tolerance:
        raise M3FitError(f"{name} failed its recomputed primal certificate.")
    return result, maximum_violation


def _primary_dual_certificate(
    *, result: Any, objective: np.ndarray, a_ub: np.ndarray, b_ub: np.ndarray
) -> dict[str, float]:
    marginals = np.asarray(result.ineqlin.marginals, dtype=float)
    slacks = np.asarray(result.ineqlin.residual, dtype=float)
    lower_marginals = np.asarray(result.lower.marginals, dtype=float)
    upper_marginals = np.asarray(result.upper.marginals, dtype=float)
    if not all(
        np.all(np.isfinite(values))
        for values in (marginals, slacks, lower_marginals, upper_marginals)
    ):
        raise M3FitError("Primary LAD dual certificate is non-finite.")
    dual_objective = float(b_ub @ marginals)
    duality_gap = abs(float(result.fun) - dual_objective)
    stationarity = objective - a_ub.T @ marginals
    stationarity -= lower_marginals + upper_marginals
    stationarity_error = float(np.max(np.abs(stationarity)))
    complementarity_error = float(np.max(np.abs(marginals * slacks)))
    scale = max(1.0, abs(float(result.fun)), abs(dual_objective))
    tolerance = _CERTIFICATE_RELATIVE_TOLERANCE * scale
    if (
        duality_gap > tolerance
        or stationarity_error > tolerance
        or complementarity_error > tolerance
        or np.max(marginals) > tolerance
    ):
        raise M3FitError("Primary LAD dual/KKT certificate is inconclusive.")
    return {
        "dual_objective": dual_objective,
        "duality_gap": duality_gap,
        "stationarity_max_abs": stationarity_error,
        "complementarity_max_abs": complementarity_error,
        "certificate_tolerance": tolerance,
    }


def fit_numerically_unique_lad(
    design: np.ndarray, response: np.ndarray
) -> dict[str, object]:
    """Fit pure LAD and reject a numerically non-point-identified coefficient."""

    values = np.asarray(design, dtype=float)
    target = np.asarray(response, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or target.shape != (values.shape[0],)
        or values.shape[0] <= values.shape[1]
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(target))
    ):
        raise M3FitError("LAD training data are invalid.")
    column_scales = np.linalg.norm(values, axis=0)
    if np.any(column_scales <= 0.0) or not np.all(np.isfinite(column_scales)):
        raise M3FitError("LAD training design has an inactive column.")
    standardized = values / column_scales
    singular = np.linalg.svd(standardized, compute_uv=False)
    rank_tolerance = (
        max(standardized.shape) * np.finfo(standardized.dtype).eps * singular[0]
    )
    if int(np.count_nonzero(singular > rank_tolerance)) != 3:
        raise M3FitError("LAD training design is rank deficient.")

    response_scale = max(1.0, float(np.max(np.abs(target))))
    normalized_target = target / response_scale
    a_ub, b_ub, objective, bounds = _lp_problem(standardized, normalized_target)
    primary, maximum_violation = _solve_lp(
        objective=objective,
        a_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        name="primary LAD LP",
    )
    scaled_coefficients = np.asarray(primary.x[:3], dtype=float)
    normalized_residual = normalized_target - standardized @ scaled_coefficients
    recomputed_l1 = float(np.sum(np.abs(normalized_residual)))
    objective_tolerance = _CERTIFICATE_RELATIVE_TOLERANCE * max(
        1.0, recomputed_l1, abs(float(primary.fun))
    )
    if abs(float(primary.fun) - recomputed_l1) > objective_tolerance:
        raise M3FitError("Primary LAD objective does not equal recomputed L1 loss.")
    dual = _primary_dual_certificate(
        result=primary,
        objective=objective,
        a_ub=a_ub,
        b_ub=b_ub,
    )

    objective_cap = recomputed_l1 + LAD_OBJECTIVE_CAP_SCALE * max(
        recomputed_l1, float(values.shape[0])
    )
    cap_a_ub = np.vstack((a_ub, objective))
    cap_b_ub = np.concatenate((b_ub, [objective_cap]))
    ranges: list[dict[str, float]] = []
    for coordinate in range(3):
        extrema: list[float] = []
        cap_losses: list[float] = []
        for direction, label in ((1.0, "minimum"), (-1.0, "maximum")):
            range_objective = np.zeros_like(objective)
            range_objective[coordinate] = direction
            extremum, _violation = _solve_lp(
                objective=range_objective,
                a_ub=cap_a_ub,
                b_ub=cap_b_ub,
                bounds=bounds,
                name=f"coefficient-{coordinate}-{label} range LP",
            )
            gamma = np.asarray(extremum.x[:3], dtype=float)
            loss = float(np.sum(np.abs(normalized_target - standardized @ gamma)))
            cap_tolerance = _CERTIFICATE_RELATIVE_TOLERANCE * max(1.0, objective_cap)
            if loss > objective_cap + cap_tolerance:
                raise M3FitError("A coefficient extremizer violates the LAD cap.")
            extrema.append(float(gamma[coordinate]))
            cap_losses.append(loss)
        minimum = extrema[0]
        maximum = extrema[1]
        if minimum > maximum:
            minimum, maximum = maximum, minimum
            cap_losses.reverse()
        width = maximum - minimum
        allowed = LAD_COEFFICIENT_RANGE_RELATIVE_TOLERANCE * max(
            1.0, abs(minimum), abs(maximum)
        )
        if width > allowed:
            raise M3FitError(
                f"Primary LAD coefficient {coordinate} is numerically non-unique."
            )
        ranges.append(
            {
                "coordinate": coordinate,
                "minimum_scaled_coefficient": minimum,
                "maximum_scaled_coefficient": maximum,
                "range": width,
                "allowed_range": allowed,
                "minimum_extremizer_recomputed_l1": cap_losses[0],
                "maximum_extremizer_recomputed_l1": cap_losses[1],
            }
        )

    coefficients = scaled_coefficients * response_scale / column_scales
    prediction = values @ coefficients
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(prediction)):
        raise M3FitError("LAD coefficient or prediction is non-finite.")
    return {
        "reduced_coefficients": coefficients.tolist(),
        "scaled_coefficients": scaled_coefficients.tolist(),
        "training_column_l2_scales": column_scales.tolist(),
        "response_scale": response_scale,
        "training_l1_kcal_mol": float(np.sum(np.abs(target - prediction))),
        "normalized_recomputed_l1": recomputed_l1,
        "normalized_reported_lp_objective": float(primary.fun),
        "normalized_objective_cap": objective_cap,
        "primary_maximum_constraint_violation": maximum_violation,
        "primary_dual_certificate": dual,
        "coefficient_range_certificates": ranges,
        "training_standardized_singular_values": singular.tolist(),
        "training_standardized_condition_number": float(singular[0] / singular[-1]),
    }


def _cluster_bootstrap_mae(
    *, absolute_errors: np.ndarray, families: Sequence[Sequence[int]]
) -> dict[str, object]:
    if absolute_errors.ndim != 1 or not np.all(np.isfinite(absolute_errors)):
        raise M3FitError("OOF errors are invalid for bootstrap.")
    blocks = [np.asarray(members, dtype=int) for members in families]
    if not blocks or sorted(np.concatenate(blocks).tolist()) != list(
        range(len(absolute_errors))
    ):
        raise M3FitError("Bootstrap families do not partition the OOF rows.")
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        choices = generator.integers(0, len(blocks), size=len(blocks))
        sampled = np.concatenate([blocks[index] for index in choices])
        estimates[replicate] = float(np.mean(absolute_errors[sampled]))
    if not np.all(np.isfinite(estimates)):
        raise M3FitError("Cluster bootstrap produced a non-finite estimate.")
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


def _read_preregistration(path: Path, source_root: Path):
    if path.stat().st_mode & 0o222:
        raise M3FitError("M3 preregistration must be read-only.")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise M3FitError("M3 preregistration is not an object.")
    _validate_self_hash(payload, key="self_sha256", name="M3 preregistration")
    if set(payload) != PREREGISTRATION_KEYS:
        raise M3FitError("M3 preregistration schema drifted.")
    for key, value in {
        "artifact": PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-first-m3-target-use-or-fit",
        "candidate_profile_id": CANDIDATE_PROFILE_ID,
        "partition": "development-water-only",
        "water_record_count": EXPECTED_WATER_COUNT,
        "experimental_targets_read_by_m3_preregistration": False,
        "experimental_targets_used_by_subspace_or_folds": False,
        "mnsol_target_table_member_opened_by_m3_preregistration": False,
        "hybrid_prediction_records_read_by_m3_preregistration": False,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_opened": False,
        "fitting_or_calibration_performed": False,
    }.items():
        if payload.get(key) != value:
            raise M3FitError(f"M3 preregistration field {key!r} drifted.")
    if Path(str(payload.get("source_root"))).resolve() != source_root:
        raise M3FitError("M3 preregistration source root drifted.")
    if payload.get("fit_contract") != frozen_fit_contract():
        raise M3FitError("M3 preregistration fit contract drifted.")
    if payload.get("development_decision_rule") != frozen_development_decision_rule():
        raise M3FitError("M3 preregistration development decision rule drifted.")
    if payload.get("external_pro_math_reviews") != list(_PRO_REVIEWS):
        raise M3FitError("M3 preregistration Pro-review ledger drifted.")
    source_files = payload.get("source_files_sha256")
    if not isinstance(source_files, dict) or set(source_files) != set(_SOURCE_FILES):
        raise M3FitError("M3 preregistration source manifest drifted.")
    return payload, hashlib.sha256(raw).hexdigest()


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


def fit(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    if source_root != _SOURCE_ROOT:
        raise M3FitError("M3 fitter must run for its own source checkout.")
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise M3FitError("M3 fit output must be outside the source checkout.")
    if output.exists():
        raise FileExistsError(output)

    snapshot = RepositorySnapshot.capture(source_root)
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    preregistration, preregistration_file_sha = _read_preregistration(
        preregistration_path, source_root
    )
    if preregistration.get("source_git_head") != snapshot.head:
        raise M3FitError("M3 preregistration Git head drifted.")
    if preregistration.get("source_git_tree") != snapshot.tree:
        raise M3FitError("M3 preregistration Git tree drifted.")
    for relative, digest in dict(preregistration["source_files_sha256"]).items():
        if sha256_file(source_root / relative) != digest:
            raise M3FitError(f"M3 source file {relative!r} drifted.")
    runtime_identity, runtime_identity_sha256 = _normalized_runtime_identity()
    if preregistration.get("runtime_identity") != runtime_identity:
        raise M3FitError("M3 numerical runtime identity drifted.")
    if preregistration.get("runtime_identity_sha256") != runtime_identity_sha256:
        raise M3FitError("M3 numerical runtime digest drifted.")

    aggregate_path = Path(str(preregistration["stock_aggregate_path"])).resolve(
        strict=True
    )
    if aggregate_path.stat().st_mode & 0o222:
        raise M3FitError("Bound stock-area aggregate must be read-only.")
    if sha256_file(aggregate_path) != preregistration.get(
        "stock_aggregate_file_sha256"
    ):
        raise M3FitError("Bound stock-area aggregate drifted.")
    aggregate = json.loads(aggregate_path.read_text())
    if not isinstance(aggregate, dict):
        raise M3FitError("Bound stock-area aggregate is invalid.")
    _validate_self_hash(aggregate, key="aggregate_sha256", name="stock-area aggregate")
    if aggregate.get("matrix_sha256") != preregistration.get("stock_matrix_sha256"):
        raise M3FitError("Bound stock-area matrix identity drifted.")
    stock_preregistration_path = Path(
        str(preregistration["stock_preregistration_path"])
    ).resolve(strict=True)
    if stock_preregistration_path.stat().st_mode & 0o222:
        raise M3FitError("Bound stock-area preregistration must be read-only.")
    if sha256_file(stock_preregistration_path) != preregistration.get(
        "stock_preregistration_file_sha256"
    ):
        raise M3FitError("Bound stock-area preregistration drifted.")
    parent_hybrid_path = Path(
        str(preregistration["parent_hybrid_preregistration_path"])
    ).resolve(strict=True)
    if parent_hybrid_path.stat().st_mode & 0o222:
        raise M3FitError("Bound parent hybrid preregistration must be read-only.")
    if sha256_file(parent_hybrid_path) != preregistration.get(
        "parent_hybrid_preregistration_file_sha256"
    ):
        raise M3FitError("Bound parent hybrid preregistration drifted.")

    evidence_root = Path(str(preregistration["parent_hybrid_evidence_root"])).resolve(
        strict=True
    )
    input_dir = args.input_dir.expanduser().resolve(strict=True)
    if input_dir != Path(str(preregistration["parent_hybrid_records_path"])).resolve():
        raise M3FitError("M3 records are outside the bound hybrid evidence root.")
    audit_path = args.integrity_audit.expanduser().resolve(strict=True)
    if audit_path != evidence_root / "audits/independent-integrity-audit.json":
        raise M3FitError("M3 audit is outside the bound hybrid evidence root.")
    audit_raw = audit_path.read_bytes()
    audit = json.loads(audit_raw)
    if not isinstance(audit, dict):
        raise M3FitError("Hybrid integrity audit is invalid.")
    audit_unsigned = dict(audit)
    observed_verification = audit_unsigned.pop("verification_sha256", None)
    if observed_verification != canonical_json_sha256(audit_unsigned):
        raise M3FitError("Hybrid integrity-audit self hash drifted.")
    if audit.get("preregistration_sha256") != preregistration.get(
        "parent_hybrid_preregistration_file_sha256"
    ):
        raise M3FitError("Hybrid audit is not bound to the M3 parent profile.")
    records = _load_bound_records(
        input_dir=input_dir,
        audit=audit,
        expected_count=EXPECTED_RECORD_COUNT,
    )

    grouped = dict(preregistration["grouped_oof"])
    row_ledger = grouped.get("row_ledger")
    if not isinstance(row_ledger, list) or len(row_ledger) != EXPECTED_WATER_COUNT:
        raise M3FitError("M3 row ledger is incomplete.")
    water_records: list[dict[str, Any]] = []
    for ordinal, ledger in enumerate(row_ledger):
        if not isinstance(ledger, dict) or ledger.get("water_ordinal") != ordinal:
            raise M3FitError("M3 row ledger ordering drifted.")
        selection_index = ledger.get("selection_index")
        if isinstance(selection_index, bool) or not isinstance(selection_index, int):
            raise M3FitError("M3 row ledger selection index is invalid.")
        record = records[selection_index]
        if (
            any(
                record.get(key) != ledger.get(key)
                for key in ("selection_index", "opaque_record_id", "geometry_sha256")
            )
            or record.get("canonical_solvent") != "water"
        ):
            raise M3FitError("M3 row ledger and hybrid record disagree.")
        water_records.append(record)

    subspace = dict(preregistration["subspace"])
    design = _finite_array(
        subspace.get("reduced_design"),
        shape=(EXPECTED_WATER_COUNT, 3),
        name="M3 reduced design",
    )
    if subspace.get("reduced_design_sha256") != canonical_json_sha256(design.tolist()):
        raise M3FitError("M3 reduced-design digest drifted.")
    row_folds = np.asarray(grouped.get("row_fold_assignments"), dtype=int)
    if row_folds.shape != (EXPECTED_WATER_COUNT,) or set(row_folds.tolist()) != set(
        range(OOF_FOLD_COUNT)
    ):
        raise M3FitError("M3 OOF fold assignments drifted.")

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
        raise M3FitError("M3 target ledger contains non-finite values.")

    oof_cds = np.full(EXPECTED_WATER_COUNT, np.nan)
    fold_results: list[dict[str, object]] = []
    for fold in range(OOF_FOLD_COUNT):
        training = row_folds != fold
        validation = row_folds == fold
        result = fit_numerically_unique_lad(design[training], target[training])
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
        raise M3FitError(
            "M3 did not produce exactly one finite OOF prediction per row."
        )

    m0_prediction = electrostatic
    m1_prediction = electrostatic + stock
    m3_prediction = electrostatic + oof_cds
    m0 = _error_metrics(m0_prediction.tolist(), experimental.tolist())
    m1 = _error_metrics(m1_prediction.tolist(), experimental.tolist())
    m3 = _error_metrics(m3_prediction.tolist(), experimental.tolist())
    best_baseline_mae = min(
        float(m0["mean_absolute_error_kcal_mol"]),
        float(m1["mean_absolute_error_kcal_mol"]),
    )
    improvement = best_baseline_mae - float(m3["mean_absolute_error_kcal_mol"])

    families = grouped.get("families")
    if not isinstance(families, list):
        raise M3FitError("M3 family ledger is invalid.")
    family_members = [family["member_indices"] for family in families]
    absolute_errors = np.abs(m3_prediction - experimental)
    bootstrap = _cluster_bootstrap_mae(
        absolute_errors=absolute_errors,
        families=family_members,
    )
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

    mixed_prediction: list[float] = []
    oof_by_selection = {
        int(ledger["selection_index"]): float(m3_prediction[ordinal])
        for ordinal, ledger in enumerate(row_ledger)
    }
    for index, record in enumerate(records):
        if index in oof_by_selection:
            mixed_prediction.append(oof_by_selection[index])
        else:
            mixed_prediction.append(
                float(record["continuum_polarization_kcal_mol"])
                + float(record["smd_cds_kcal_mol"])
            )
    mixed_experimental = [
        float(record["experimental_delta_g_kcal_mol"]) for record in records
    ]
    mixed_metrics = _error_metrics(mixed_prediction, mixed_experimental)
    passed = (
        float(m3["mean_absolute_error_kcal_mol"]) <= PRIMARY_MAE_THRESHOLD_KCAL_MOL
        and float(mixed_metrics["mean_absolute_error_kcal_mol"])
        <= PRIMARY_MAE_THRESHOLD_KCAL_MOL
        and improvement >= MINIMUM_MAE_IMPROVEMENT_KCAL_MOL
    )

    deployable_fit = None
    if passed:
        deployable_fit = fit_numerically_unique_lad(design, target)
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
            "All-306 development refit for a future candidate implementation; "
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
        "confirmation_partition_opened": False,
        "source_git_head": snapshot.head,
        "source_git_tree": snapshot.tree,
        "runtime_identity_sha256": runtime_identity_sha256,
        "preregistration_path": str(preregistration_path),
        "preregistration_file_sha256": preregistration_file_sha,
        "integrity_audit_path": str(audit_path),
        "integrity_audit_file_sha256": hashlib.sha256(audit_raw).hexdigest(),
        "record_files_manifest_sha256": audit["record_files_manifest_sha256"],
        "water_baselines": {"m0": m0, "m1": m1},
        "water_m3_grouped_oof": m3,
        "water_oof_mae_improvement_over_better_baseline_kcal_mol": improvement,
        "mixed_505_water_m3_oof_nonaqueous_m1": mixed_metrics,
        "development_decision": "pass" if passed else "fail",
        "fold_results": fold_results,
        "cluster_bootstrap": bootstrap,
        "maximum_family_influence": family_influence[0],
        "family_influence": family_influence,
        "all_306_deployable_candidate_fit": deployable_fit,
        "claim_boundary": (
            "Development-only grouped OOF stock-area M3 energy result. The "
            "family rule is elemental-composition grouping, not scaffold "
            "generalization. The production smooth-area profile, force, public "
            "API, and confirmation capabilities remain unadmitted."
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
