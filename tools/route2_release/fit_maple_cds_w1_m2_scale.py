#!/usr/bin/env python3
"""Fit the preregistered one-parameter water CDS scale on complete v3 data."""

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

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))

from maple.solvation.release.evidence import (
    RepositorySnapshot,
    canonical_json_sha256,
    sha256_file,
)

try:  # package import used by tests and installed entrypoints
    from tools.route2_release.analyze_hybrid_smd_components_v3 import (
        EXPECTED_RECORD_COUNT,
        _error_metrics,
        _load_bound_records,
    )
    from tools.route2_release.create_maple_cds_w1_m2_preregistration import (
        ALPHA_MAXIMUM,
        ALPHA_MINIMUM,
        EXPECTED_NONAQUEOUS_COUNT,
        EXPECTED_RECORD_COUNT as PREREGISTERED_RECORD_COUNT,
        EXPECTED_WATER_COUNT,
        MINIMUM_MAE_IMPROVEMENT_KCAL_MOL,
        PREREGISTRATION_ARTIFACT,
        PRIMARY_MAE_THRESHOLD_KCAL_MOL,
        normalized_runtime_identity,
    )
except ModuleNotFoundError:  # direct ``python tools/route2_release/...`` execution
    from analyze_hybrid_smd_components_v3 import (
        EXPECTED_RECORD_COUNT,
        _error_metrics,
        _load_bound_records,
    )
    from create_maple_cds_w1_m2_preregistration import (
        ALPHA_MAXIMUM,
        ALPHA_MINIMUM,
        EXPECTED_NONAQUEOUS_COUNT,
        EXPECTED_RECORD_COUNT as PREREGISTERED_RECORD_COUNT,
        EXPECTED_WATER_COUNT,
        MINIMUM_MAE_IMPROVEMENT_KCAL_MOL,
        PREREGISTRATION_ARTIFACT,
        PRIMARY_MAE_THRESHOLD_KCAL_MOL,
        normalized_runtime_identity,
    )


RESULT_ARTIFACT = "route2-maple-cds-w1-m2-fit-v1"


class M2FitError(RuntimeError):
    """Raised when the M2 fit is not exactly bound to its preregistration."""


def _validate_self_hash(payload: Mapping[str, Any], *, name: str) -> None:
    expected = payload.get("self_sha256")
    unsigned = dict(payload)
    unsigned.pop("self_sha256", None)
    if expected != canonical_json_sha256(unsigned):
        raise M2FitError(f"{name} self hash drifted.")


def _load_preregistration(path: Path, source_root: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise M2FitError("M2 preregistration is not an object.")
    _validate_self_hash(payload, name="M2 preregistration")
    expected = {
        "artifact": PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-terminal-v3-evidence-and-first-m2-fit",
        "partition": "development-water-only",
        "expected_water_record_count": EXPECTED_WATER_COUNT,
        "expected_nonaqueous_record_count": EXPECTED_NONAQUEOUS_COUNT,
        "experimental_targets_read_by_preregistration": False,
        "hybrid_prediction_records_read_by_preregistration": False,
        "hybrid_record_file_names_observed_by_preregistration": True,
        "confirmation_partition_opened": False,
        "nonaqueous_policy": "retain-stock-smd-cds-m1-unchanged",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise M2FitError(f"M2 preregistration field {key!r} drifted.")
    if Path(str(payload.get("source_root"))).resolve() != source_root:
        raise M2FitError("M2 preregistration source root drifted.")
    fit = payload.get("fit_contract")
    expected_fit = {
        "target": "experimental_delta_g_kcal_mol-continuum_polarization_kcal_mol",
        "predictor": "stock_smd_cds_kcal_mol",
        "model": "continuum_polarization_kcal_mol+alpha*stock_smd_cds_kcal_mol",
        "objective": "unweighted-ordinary-least-squares",
        "closed_form": "alpha=sum(x*y)/sum(x*x)",
        "intercept": False,
        "record_scope": "all-and-only-306-water-development-records",
        "alpha_minimum_inclusive": ALPHA_MINIMUM,
        "alpha_maximum_inclusive": ALPHA_MAXIMUM,
        "out_of_domain_action": "reject-candidate-without-clipping",
        "standard_state": "1M-ideal-gas-to-1M-ideal-solution",
        "standard_state_correction_kcal_mol": 0.0,
    }
    if fit != expected_fit:
        raise M2FitError("M2 fit contract drifted.")
    decision = payload.get("development_decision_rule")
    expected_decision = {
        "primary_mae_threshold_kcal_mol": PRIMARY_MAE_THRESHOLD_KCAL_MOL,
        "minimum_mae_improvement_over_better_of_m0_m1_kcal_mol": (
            MINIMUM_MAE_IMPROVEMENT_KCAL_MOL
        ),
        "both_conditions_required": True,
        "tail_metrics": "report-only-not-post-hoc-gates",
    }
    if decision != expected_decision:
        raise M2FitError("M2 development decision rule drifted.")
    state = payload.get("hybrid_evidence_state_at_lock")
    if not isinstance(state, dict):
        raise M2FitError("M2 preregistration lacks its preterminal evidence state.")
    expected_terminal = {
        "audits/preregistered-full-verification.json": True,
        "audits/independent-integrity-audit.json": True,
        "exits/aggregate.exit": True,
        "exits/integrity-audit.exit": True,
    }
    for key, expected in (
        ("expected_record_count", PREREGISTERED_RECORD_COUNT),
        ("record_file_names_observed", True),
        ("record_contents_read", False),
        ("terminal_evidence_absent", expected_terminal),
    ):
        if state.get(key) != expected:
            raise M2FitError(f"M2 preterminal evidence field {key!r} drifted.")
    record_count = state.get("record_file_count_at_lock")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or not 0 <= record_count < PREREGISTERED_RECORD_COUNT
    ):
        raise M2FitError("M2 was not locked before all v3 records existed.")
    names_sha = state.get("record_file_names_sha256_at_lock")
    if (
        not isinstance(names_sha, str)
        or len(names_sha) != 64
        or any(character not in "0123456789abcdef" for character in names_sha)
    ):
        raise M2FitError("M2 preterminal record-name digest is invalid.")
    return payload, hashlib.sha256(raw).hexdigest()


def _read_json_object_with_sha256(
    path: Path, *, name: str
) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise M2FitError(f"{name} is not an object.")
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
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        snapshot.assert_unchanged()
        os.link(temporary, output)
        directory_fd = os.open(
            output.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_parent_audit_binding(
    *, preregistration: Mapping[str, Any], audit: Mapping[str, Any]
) -> None:
    if audit.get("preregistration_sha256") != preregistration.get(
        "parent_hybrid_preregistration_sha256"
    ):
        raise M2FitError(
            "Hybrid-v3 integrity audit is not bound to the preregistered parent."
        )


def _fit_water_scale(records: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    water = [row for row in records if row.get("canonical_solvent") == "water"]
    nonaqueous = [row for row in records if row.get("canonical_solvent") != "water"]
    if (
        len(water) != EXPECTED_WATER_COUNT
        or len(nonaqueous) != EXPECTED_NONAQUEOUS_COUNT
    ):
        raise M2FitError("Water/nonaqueous record counts drifted.")

    experimental = [float(row["experimental_delta_g_kcal_mol"]) for row in water]
    electrostatic = [float(row["continuum_polarization_kcal_mol"]) for row in water]
    stock = [float(row["smd_cds_kcal_mol"]) for row in water]
    target = [
        reference - polarization
        for reference, polarization in zip(experimental, electrostatic, strict=True)
    ]
    denominator = sum(value * value for value in stock)
    if not math.isfinite(denominator) or denominator <= 1.0e-20:
        raise M2FitError("Stock CDS predictor has no finite scale information.")
    alpha = (
        sum(
            predictor * response
            for predictor, response in zip(stock, target, strict=True)
        )
        / denominator
    )
    if not math.isfinite(alpha):
        raise M2FitError("Fitted alpha is non-finite.")

    m0_prediction = electrostatic
    m1_prediction = [
        polarization + cds
        for polarization, cds in zip(electrostatic, stock, strict=True)
    ]
    m2_prediction = [
        polarization + alpha * cds
        for polarization, cds in zip(electrostatic, stock, strict=True)
    ]
    m0 = _error_metrics(m0_prediction, experimental)
    m1 = _error_metrics(m1_prediction, experimental)
    m2 = _error_metrics(m2_prediction, experimental)
    alpha_in_domain = ALPHA_MINIMUM <= alpha <= ALPHA_MAXIMUM
    best_baseline_mae = min(
        float(m0["mean_absolute_error_kcal_mol"]),
        float(m1["mean_absolute_error_kcal_mol"]),
    )
    improvement = best_baseline_mae - float(m2["mean_absolute_error_kcal_mol"])
    primary_pass = (
        alpha_in_domain
        and float(m2["mean_absolute_error_kcal_mol"]) <= PRIMARY_MAE_THRESHOLD_KCAL_MOL
        and improvement >= MINIMUM_MAE_IMPROVEMENT_KCAL_MOL
    )

    all_experimental = [float(row["experimental_delta_g_kcal_mol"]) for row in records]
    mixed_prediction = [
        float(row["continuum_polarization_kcal_mol"])
        + (
            alpha * float(row["smd_cds_kcal_mol"])
            if row.get("canonical_solvent") == "water"
            else float(row["smd_cds_kcal_mol"])
        )
        for row in records
    ]
    return {
        "alpha": alpha,
        "alpha_in_preregistered_domain": alpha_in_domain,
        "water_m0": m0,
        "water_m1": m1,
        "water_m2": m2,
        "water_m2_mae_improvement_over_better_baseline_kcal_mol": improvement,
        "water_development_decision": "pass" if primary_pass else "fail",
        "mixed_505_water_m2_nonaqueous_m1": _error_metrics(
            mixed_prediction, all_experimental
        ),
    }


def fit(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    if source_root != _SOURCE_ROOT:
        raise M2FitError("M2 fitter must execute for its own source checkout.")
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise M2FitError("M2 fit output must be outside the checkout.")
    if output.exists():
        raise FileExistsError(output)

    snapshot = RepositorySnapshot.capture(source_root)
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    preregistration, preregistration_raw_sha256 = _load_preregistration(
        preregistration_path,
        source_root,
    )
    if preregistration.get("source_git_head") != snapshot.head:
        raise M2FitError("M2 preregistration Git head drifted.")
    if preregistration.get("source_git_tree") != snapshot.tree:
        raise M2FitError("M2 preregistration Git tree drifted.")
    for relative, expected in dict(preregistration["source_files_sha256"]).items():
        if sha256_file(source_root / relative) != expected:
            raise M2FitError(f"M2 source file {relative!r} drifted.")
    runtime_identity, runtime_identity_sha256 = normalized_runtime_identity()
    if preregistration.get("runtime_identity") != runtime_identity:
        raise M2FitError("M2 numerical runtime identity drifted.")
    if preregistration.get("runtime_identity_sha256") != runtime_identity_sha256:
        raise M2FitError("M2 numerical runtime digest drifted.")
    parent_path = Path(
        str(preregistration["parent_hybrid_preregistration_path"])
    ).resolve(strict=True)
    parent_raw = parent_path.read_bytes()
    if hashlib.sha256(parent_raw).hexdigest() != preregistration.get(
        "parent_hybrid_preregistration_sha256"
    ):
        raise M2FitError("Parent hybrid-v3 preregistration drifted.")

    state = dict(preregistration["hybrid_evidence_state_at_lock"])
    evidence_root = Path(str(state["hybrid_evidence_root"])).resolve(strict=True)
    expected_records_path = evidence_root / "records"
    input_dir = args.input_dir.expanduser().resolve(strict=True)
    if input_dir != expected_records_path:
        raise M2FitError("M2 input records are outside the bound hybrid evidence root.")

    audit_path = args.integrity_audit.expanduser().resolve(strict=True)
    if audit_path != evidence_root / "audits/independent-integrity-audit.json":
        raise M2FitError("M2 integrity audit is outside the bound evidence root.")
    audit, audit_raw_sha256 = _read_json_object_with_sha256(
        audit_path,
        name="Hybrid-v3 integrity audit",
    )
    audit_verification = audit.get("verification_sha256")
    audit_unsigned = dict(audit)
    audit_unsigned.pop("verification_sha256", None)
    if audit_verification != canonical_json_sha256(audit_unsigned):
        raise M2FitError("Hybrid-v3 integrity-audit self hash drifted.")
    _validate_parent_audit_binding(
        preregistration=preregistration,
        audit=audit,
    )
    records = _load_bound_records(
        input_dir=input_dir,
        audit=audit,
        expected_count=EXPECTED_RECORD_COUNT,
    )
    result = _fit_water_scale(records)
    payload: dict[str, object] = {
        "artifact": RESULT_ARTIFACT,
        "schema_version": 1,
        "status": (
            "development-pass"
            if result["water_development_decision"] == "pass"
            else "development-fail"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_profile_id": preregistration["candidate_profile_id"],
        "partition": "development-water-only",
        "fitting_or_calibration_performed": True,
        "fitted_parameter_count": 1,
        "fitted_intercept": False,
        "confirmation_partition_opened": False,
        "nonaqueous_policy": preregistration["nonaqueous_policy"],
        "source_git_head": snapshot.head,
        "source_git_tree": snapshot.tree,
        "runtime_identity_sha256": runtime_identity_sha256,
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": preregistration_raw_sha256,
        "integrity_audit_path": str(audit_path),
        "integrity_audit_sha256": audit_raw_sha256,
        "record_files_manifest_sha256": audit["record_files_manifest_sha256"],
        "fit_contract": preregistration["fit_contract"],
        "development_decision_rule": preregistration["development_decision_rule"],
        "result": result,
        "claim_boundary": (
            "Development-only one-parameter water CDS fit. Confirmation remains "
            "sealed, nonaqueous records retain M1, and no force or production "
            "capability is admitted."
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
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
