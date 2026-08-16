#!/usr/bin/env python3
"""Freeze M4-PH1.0 after the terminal, nonunique M3 LAD result.

The creator reads only the target-blind M3 preregistration and its target-free
terminal failure record.  It does not open hybrid prediction records, MNSol
targets, or confirmation data.  M4 is a new estimator, not a repair of M3.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))

from maple.solvation.release.evidence import (
    RepositorySnapshot,
    canonical_json_sha256,
    committed_source_hashes,
    runtime_record,
)
from tools.route2_release.create_maple_cds_w1_m3_preregistration import (
    EXPECTED_WATER_COUNT,
    OOF_FOLD_COUNT,
    PREREGISTRATION_ARTIFACT as M3_PREREGISTRATION_ARTIFACT,
)

PREREGISTRATION_ARTIFACT = "route2-maple-cds-w1-m4-ph1-prereg-v1"
CANDIDATE_PROFILE_ID = "maple-cds-w1-stock-area-target-blind-3d-ph1-water-v1"
RESULT_ARTIFACT = "route2-maple-cds-w1-m4-ph1-grouped-oof-v1"

M3_PREREGISTRATION_FILE_SHA256 = (
    "0302635aa2186f326cf73ffccf284b0f0ba07aa7d51c4432c13cc03e80ba365b"
)
M3_PREREGISTRATION_SELF_SHA256 = (
    "a79d551ac7b8040444a7a4168c3e872b43bcd84f2b481695c7a77fd371c26092"
)
M3_SOURCE_GIT_HEAD = "44ce23189fe0494876dafcee61a0d591e0a63773"
M3_FAILURE_ARTIFACT = "route2-maple-cds-w1-m3-terminal-failure-v1"
M3_FAILURE_FILE_SHA256 = (
    "0085d34463e73e9ae39314853a57e3ee515463e208706d4c942aa57f38a5f24d"
)
M3_FAILURE_SELF_SHA256 = (
    "e1ae0588b10b550ea38b68884dc64972a6c5871f24239a59ff78d7527b205570"
)
PRO_Q9_PROMPT_SHA256 = (
    "59e2c37cf80848e04aa61f4a172b7b0a428d04f9583241ab7d0c5bc01866ce8c"
)
PRO_Q9_SUBMISSION_SHA256 = (
    "36d4ee9a99f051409b7fc393daedcf97e536571040ad3f6ca3dd8943421e5283"
)
PRO_Q9_ANSWER_SHA256 = (
    "4494687a9d7fa957d3fbd9bcdfc330c7e2651aa31809fda9c2a4bc59139cb720"
)

PSEUDO_HUBER_DELTA_KCAL_MOL = 1.0
PRIMARY_MAE_THRESHOLD_KCAL_MOL = 1.5
MINIMUM_MAE_IMPROVEMENT_KCAL_MOL = 0.05
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_816
TRUST_EXACT_GRADIENT_TOLERANCE = 1.0e-12
STATIONARITY_INFINITY_TOLERANCE = 1.0e-10
STANDARDIZED_CONDITION_LIMIT = 5.0
CURVATURE_RELATIVE_FLOOR = 1.0e-12
INDEPENDENT_SOLUTION_RELATIVE_TOLERANCE = 1.0e-8
PREDICTION_REPRODUCIBILITY_KCAL_MOL = 1.0e-8
OBJECTIVE_REPRODUCIBILITY_RELATIVE_TOLERANCE = 1.0e-12
MAXIMUM_SOLVER_ITERATIONS = 1_000

_SOURCE_FILES = (
    "docs/route2/MAPLE_CDS_W1.md",
    "docs/route2/evidence/M3_TERMINAL_FAILURE_2026-08-16.md",
    "docs/route2/evidence/M4_PRO_AUDIT_2026-08-16.md",
    "maple/solvation/release/evidence.py",
    "tests/route2_vnext/test_maple_cds_w1_m4_pseudohuber.py",
    "tools/route2_release/analyze_hybrid_smd_components_v3.py",
    "tools/route2_release/create_maple_cds_w1_m3_preregistration.py",
    "tools/route2_release/create_maple_cds_w1_m4_preregistration.py",
    "tools/route2_release/fit_maple_cds_w1_m3_lad.py",
    "tools/route2_release/fit_maple_cds_w1_m4_pseudohuber.py",
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
        "m3_preregistration_path",
        "m3_preregistration_file_sha256",
        "m3_preregistration_self_sha256",
        "m3_failure_path",
        "m3_failure_file_sha256",
        "m3_failure_self_sha256",
        "m3_terminal_error",
        "m3_oof_mae_computed",
        "m3_threshold_or_rule_changed_after_failure",
        "experimental_targets_read_by_m4_preregistration",
        "hybrid_prediction_records_read_by_m4_preregistration",
        "confirmation_selection_manifest_opened",
        "confirmation_records_opened",
        "fitting_or_calibration_performed",
        "prior_development_information_disclosed",
        "external_pro_math_review",
        "fit_contract",
        "development_decision_rule",
        "deployment_policy",
        "nonaqueous_policy",
        "confirmation_policy",
        "claim_boundary",
        "self_sha256",
    }
)


class M4PreregistrationError(RuntimeError):
    """Raised when the M4 preregistration cannot be frozen exactly."""


def normalized_runtime_identity() -> tuple[dict[str, object], str]:
    identity = runtime_record()
    identity.pop("generated_at_utc", None)
    return identity, canonical_json_sha256(identity)


def frozen_fit_contract() -> dict[str, object]:
    return {
        "model": (
            "continuum_polarization_kcal_mol + X @ " "(alpha*theta_stock + W@beta)"
        ),
        "parameter_count": 3,
        "intercept": False,
        "validation": "same-ten-fold-family-grouped-out-of-fold-as-m3",
        "loss": "strictly-convex-pseudo-huber",
        "pseudo_huber_delta_kcal_mol": PSEUDO_HUBER_DELTA_KCAL_MOL,
        "regularizer": False,
        "training_design_scaling": "per-fold-column-l2",
        "response_scaling": "divide-by-fixed-physical-delta",
        "solver": "scipy.optimize.minimize-method-trust-exact",
        "solver_starts": ["zero", "svd-ols"],
        "solver_gradient_tolerance": TRUST_EXACT_GRADIENT_TOLERANCE,
        "maximum_solver_iterations": MAXIMUM_SOLVER_ITERATIONS,
        "stationarity_infinity_tolerance": STATIONARITY_INFINITY_TOLERANCE,
        "standardized_condition_limit": STANDARDIZED_CONDITION_LIMIT,
        "curvature_relative_floor": CURVATURE_RELATIVE_FLOOR,
        "independent_solution_relative_tolerance": (
            INDEPENDENT_SOLUTION_RELATIVE_TOLERANCE
        ),
        "prediction_reproducibility_kcal_mol": (PREDICTION_REPRODUCIBILITY_KCAL_MOL),
        "objective_reproducibility_relative_tolerance": (
            OBJECTIVE_REPRODUCIBILITY_RELATIVE_TOLERANCE
        ),
        "fallback_estimator": False,
        "standard_state": "1M-ideal-gas-to-1M-ideal-solution",
        "standard_state_correction_kcal_mol": 0.0,
    }


def frozen_development_decision_rule() -> dict[str, object]:
    return {
        "primary_oof_water_mae_threshold_kcal_mol": (PRIMARY_MAE_THRESHOLD_KCAL_MOL),
        "primary_mixed_505_mae_threshold_kcal_mol": (PRIMARY_MAE_THRESHOLD_KCAL_MOL),
        "minimum_oof_water_mae_improvement_over_better_m0_m1_kcal_mol": (
            MINIMUM_MAE_IMPROVEMENT_KCAL_MOL
        ),
        "cluster_bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "cluster_bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_ucb": "report-only-on-development",
        "required_conditions": (
            "all-ten-fold-numerical-certificates-pass; complete finite OOF; "
            "water MAE<=1.5; mixed-505 water-M4 plus nonaqueous-M1 MAE<=1.5; "
            "water improvement>=0.05 over the better frozen M0/M1 baseline"
        ),
        "terminal_policy": (
            "if M4 misses or a fold certificate fails, stop this target-blind "
            "three-dimensional stock-area estimator lane; do not tune delta"
        ),
    }


def frozen_prior_development_disclosure() -> dict[str, object]:
    return {
        "m2_water_mae_kcal_mol": 1.6265953472,
        "m2_failed_primary_threshold": True,
        "m3_terminal_numerical_nonuniqueness": True,
        "m3_oof_mae_was_not_computed": True,
        "m4_was_selected_after_m3_failure": True,
    }


def frozen_pro_review() -> dict[str, object]:
    return {
        "verified_model": "Pro, 5 of 5.",
        "conversation": "6a8208a3-d5d8-83e8-a7fb-de8235d1b59e",
        "prompt_sha256": PRO_Q9_PROMPT_SHA256,
        "submission_evidence_sha256": PRO_Q9_SUBMISSION_SHA256,
        "answer_sha256": PRO_Q9_ANSWER_SHA256,
        "answer_marker": "USE PSEUDO-HUBER",
        "local_verdict": (
            "accepted-after-independent-strict-convexity-and-solver-canaries"
        ),
    }


def _validate_self_hash(payload: Mapping[str, Any], *, key: str, name: str) -> None:
    unsigned = dict(payload)
    observed = unsigned.pop(key, None)
    if observed != canonical_json_sha256(unsigned):
        raise M4PreregistrationError(f"{name} self hash drifted.")


def _read_object(path: Path, *, name: str) -> tuple[dict[str, Any], str]:
    if path.stat().st_mode & 0o222:
        raise M4PreregistrationError(f"{name} must be read-only.")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise M4PreregistrationError(f"{name} is not a JSON object.")
    return payload, hashlib.sha256(raw).hexdigest()


def validate_m3_terminal_parent(
    *, m3_preregistration_path: Path, m3_failure_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg, prereg_file_sha = _read_object(
        m3_preregistration_path, name="M3 preregistration"
    )
    _validate_self_hash(prereg, key="self_sha256", name="M3 preregistration")
    for key, value in {
        "artifact": M3_PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-first-m3-target-use-or-fit",
        "source_git_head": M3_SOURCE_GIT_HEAD,
        "water_record_count": EXPECTED_WATER_COUNT,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_opened": False,
        "fitting_or_calibration_performed": False,
        "self_sha256": M3_PREREGISTRATION_SELF_SHA256,
    }.items():
        if prereg.get(key) != value:
            raise M4PreregistrationError(f"M3 preregistration field {key!r} drifted.")
    if prereg_file_sha != M3_PREREGISTRATION_FILE_SHA256:
        raise M4PreregistrationError("M3 preregistration file identity drifted.")
    grouped = prereg.get("grouped_oof")
    if (
        not isinstance(grouped, dict)
        or grouped.get("fold_count") != OOF_FOLD_COUNT
        or grouped.get("family_count") != 202
        or grouped.get("maximum_family_size") != 7
        or grouped.get("fold_record_counts") != [31, 31, 31, 31, 31, 31, 30, 30, 30, 30]
    ):
        raise M4PreregistrationError("M3 grouped-OOF identity drifted.")

    failure, failure_file_sha = _read_object(m3_failure_path, name="M3 failure")
    _validate_self_hash(failure, key="self_sha256", name="M3 failure")
    for key, value in {
        "artifact": M3_FAILURE_ARTIFACT,
        "schema_version": 1,
        "status": "development-fail-numerically-nonunique-lad",
        "source_git_head": M3_SOURCE_GIT_HEAD,
        "preregistration_file_sha256": M3_PREREGISTRATION_FILE_SHA256,
        "preregistration_self_sha256": M3_PREREGISTRATION_SELF_SHA256,
        "command_exit_code": 1,
        "terminal_error": "Primary LAD coefficient 0 is numerically non-unique.",
        "oof_prediction_completed": False,
        "oof_mae_computed": False,
        "threshold_or_uniqueness_rule_changed_after_failure": False,
        "confirmation_partition_opened": False,
        "self_sha256": M3_FAILURE_SELF_SHA256,
    }.items():
        if failure.get(key) != value:
            raise M4PreregistrationError(f"M3 failure field {key!r} drifted.")
    if failure_file_sha != M3_FAILURE_FILE_SHA256:
        raise M4PreregistrationError("M3 failure file identity drifted.")
    if Path(str(failure.get("preregistration_path"))).resolve() != (
        m3_preregistration_path
    ):
        raise M4PreregistrationError("M3 failure belongs to another preregistration.")
    return prereg, failure


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


def create(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    if source_root != _SOURCE_ROOT:
        raise M4PreregistrationError("M4 creator must run for its own checkout.")
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise M4PreregistrationError("M4 preregistration must be outside source.")
    if output.exists():
        raise FileExistsError(output)

    m3_preregistration_path = args.m3_preregistration.expanduser().resolve(strict=True)
    m3_failure_path = args.m3_failure.expanduser().resolve(strict=True)
    validate_m3_terminal_parent(
        m3_preregistration_path=m3_preregistration_path,
        m3_failure_path=m3_failure_path,
    )

    snapshot = RepositorySnapshot.capture(source_root)
    runtime_identity, runtime_identity_sha256 = normalized_runtime_identity()
    source_hashes = committed_source_hashes(snapshot, _SOURCE_FILES)
    payload: dict[str, object] = {
        "artifact": PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-first-m4-target-use-or-fit",
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
        "m3_preregistration_path": str(m3_preregistration_path),
        "m3_preregistration_file_sha256": M3_PREREGISTRATION_FILE_SHA256,
        "m3_preregistration_self_sha256": M3_PREREGISTRATION_SELF_SHA256,
        "m3_failure_path": str(m3_failure_path),
        "m3_failure_file_sha256": M3_FAILURE_FILE_SHA256,
        "m3_failure_self_sha256": M3_FAILURE_SELF_SHA256,
        "m3_terminal_error": "Primary LAD coefficient 0 is numerically non-unique.",
        "m3_oof_mae_computed": False,
        "m3_threshold_or_rule_changed_after_failure": False,
        "experimental_targets_read_by_m4_preregistration": False,
        "hybrid_prediction_records_read_by_m4_preregistration": False,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_opened": False,
        "fitting_or_calibration_performed": False,
        "prior_development_information_disclosed": (
            frozen_prior_development_disclosure()
        ),
        "external_pro_math_review": frozen_pro_review(),
        "fit_contract": frozen_fit_contract(),
        "development_decision_rule": frozen_development_decision_rule(),
        "deployment_policy": (
            "Only if grouped OOF passes, fit one separately labeled all-306 "
            "development coefficient under the same estimator; it is not an "
            "OOF prediction or public admission."
        ),
        "nonaqueous_policy": "retain-stock-smd-cds-m1-unchanged",
        "confirmation_policy": (
            "remain-sealed-until-m4-development-decision-and-coefficients-are-frozen"
        ),
        "claim_boundary": (
            "New development-only M4-PH1.0 estimator on the M3 target-blind "
            "subspace and folds. It is not M3, confirmation, a production "
            "smooth-area CDS, a force, or a public capability."
        ),
    }
    if set(payload) | {"self_sha256"} != PREREGISTRATION_KEYS:
        raise M4PreregistrationError("Internal M4 preregistration schema drifted.")
    payload["self_sha256"] = canonical_json_sha256(payload)
    _write_json_exclusive_atomic(output=output, payload=payload, snapshot=snapshot)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--m3-preregistration", type=Path, required=True)
    parser.add_argument("--m3-failure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    payload = create(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
