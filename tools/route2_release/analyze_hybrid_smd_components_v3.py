from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

EXPECTED_RECORD_COUNT = 505
EXPECTED_RECORD_ARTIFACT = "route2-hybrid-smd-development-record-v3"
EXPECTED_AUDIT_ARTIFACT = "route2-hybrid-smd-development-integrity-audit-v1"
ALLOWED_TERMINAL_AUDIT_STATUSES = frozenset({"pass", "accuracy-failure"})
LEDGER_ABSOLUTE_TOLERANCE_KCAL_MOL = 1.0e-12


class HybridComponentAnalysisError(RuntimeError):
    """Raised when component analysis is attempted on unbound evidence."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise HybridComponentAnalysisError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HybridComponentAnalysisError(f"{name} must be finite.") from exc
    if not math.isfinite(result):
        raise HybridComponentAnalysisError(f"{name} must be finite.")
    return result


def _quantile_type7(values: Sequence[float], probability: float) -> float:
    """Return the explicitly defined R/NumPy type-7 sample quantile."""

    if not values:
        raise HybridComponentAnalysisError("A quantile requires at least one value.")
    if not 0.0 <= probability <= 1.0:
        raise HybridComponentAnalysisError("Quantile probability is outside [0, 1].")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    fraction = position - lower
    if fraction == 0.0:
        return ordered[lower]
    return ordered[lower] + fraction * (ordered[lower + 1] - ordered[lower])


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise HybridComponentAnalysisError(
            "Pearson inputs must be nonempty and paired."
        )
    left_mean = fmean(left)
    right_mean = fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_norm = math.sqrt(sum(value * value for value in left_centered))
    right_norm = math.sqrt(sum(value * value for value in right_centered))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered, strict=True)
    ) / (left_norm * right_norm)


def _error_metrics(
    predicted: Sequence[float], experimental: Sequence[float]
) -> dict[str, object]:
    if len(predicted) != len(experimental) or not predicted:
        raise HybridComponentAnalysisError(
            "Energy vectors must be nonempty and paired."
        )
    signed = [
        prediction - reference
        for prediction, reference in zip(predicted, experimental, strict=True)
    ]
    absolute = [abs(value) for value in signed]
    return {
        "record_count": len(signed),
        "mean_signed_error_kcal_mol": fmean(signed),
        "mean_absolute_error_kcal_mol": fmean(absolute),
        "root_mean_square_error_kcal_mol": math.sqrt(
            fmean(value * value for value in signed)
        ),
        "q95_absolute_error_kcal_mol": _quantile_type7(absolute, 0.95),
        "maximum_absolute_error_kcal_mol": max(absolute),
        "ge_1_0_count": sum(value >= 1.0 for value in absolute),
        "ge_1_5_count": sum(value >= 1.5 for value in absolute),
        "ge_3_0_count": sum(value >= 3.0 for value in absolute),
    }


def _component_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    if not records:
        raise HybridComponentAnalysisError("A component group cannot be empty.")
    experimental = [
        _finite(row["experimental_delta_g_kcal_mol"], name="experimental energy")
        for row in records
    ]
    electrostatic = [
        _finite(
            row["continuum_polarization_kcal_mol"],
            name="continuum polarization",
        )
        for row in records
    ]
    stock_cds = [
        _finite(row["smd_cds_kcal_mol"], name="stock SMD-CDS") for row in records
    ]
    stock_total = [
        polarization + cds
        for polarization, cds in zip(electrostatic, stock_cds, strict=True)
    ]
    effective_target = [
        reference - polarization
        for reference, polarization in zip(experimental, electrostatic, strict=True)
    ]
    m0 = _error_metrics(electrostatic, experimental)
    m1 = _error_metrics(stock_total, experimental)
    m0_absolute = [
        abs(prediction - reference)
        for prediction, reference in zip(electrostatic, experimental, strict=True)
    ]
    m1_absolute = [
        abs(prediction - reference)
        for prediction, reference in zip(stock_total, experimental, strict=True)
    ]
    helped = 0
    worsened = 0
    tied = 0
    for before, after in zip(m0_absolute, m1_absolute, strict=True):
        delta = after - before
        if delta < -LEDGER_ABSOLUTE_TOLERANCE_KCAL_MOL:
            helped += 1
        elif delta > LEDGER_ABSOLUTE_TOLERANCE_KCAL_MOL:
            worsened += 1
        else:
            tied += 1
    return {
        "record_count": len(records),
        "m0_hybrid_electrostatic_only": m0,
        "m1_hybrid_electrostatic_plus_stock_smd_cds": m1,
        "paired_stock_cds_effect": {
            "mean_absolute_error_change_m1_minus_m0_kcal_mol": (
                float(m1["mean_absolute_error_kcal_mol"])
                - float(m0["mean_absolute_error_kcal_mol"])
            ),
            "helped_count": helped,
            "worsened_count": worsened,
            "tied_count": tied,
        },
        "effective_cds_diagnostic": {
            "mean_target_kcal_mol": fmean(effective_target),
            "mean_stock_smd_cds_kcal_mol": fmean(stock_cds),
            "pearson_stock_smd_cds_vs_effective_target": _pearson(
                stock_cds, effective_target
            ),
            "interpretation_boundary": (
                "The target is an effective residual under the frozen energy and "
                "standard-state convention, not an observable CDS component."
            ),
        },
    }


def _load_bound_records(
    *,
    input_dir: Path,
    audit: Mapping[str, Any],
    expected_count: int,
) -> list[dict[str, Any]]:
    if audit.get("artifact") != EXPECTED_AUDIT_ARTIFACT:
        raise HybridComponentAnalysisError("Unknown integrity-audit identity.")
    if audit.get("evidence_contract_version") != 3:
        raise HybridComponentAnalysisError("Integrity audit is not for v3 evidence.")
    if audit.get("status") not in ALLOWED_TERMINAL_AUDIT_STATUSES:
        raise HybridComponentAnalysisError(
            "Component analysis requires a complete terminal integrity audit."
        )
    for key, expected in (
        ("expected_record_count", expected_count),
        ("record_count", expected_count),
        ("success_count", expected_count),
        ("failure_count", 0),
        ("partition", "development"),
        ("confirmation_partition_opened", False),
    ):
        if audit.get(key) != expected:
            raise HybridComponentAnalysisError(f"Integrity-audit {key} drifted.")
    if audit.get("missing_selection_indices") != []:
        raise HybridComponentAnalysisError("Integrity audit reports missing records.")
    if audit.get("failed_selection_indices") != []:
        raise HybridComponentAnalysisError("Integrity audit reports provider failures.")

    expected_names = {f"index-{index:03d}.json" for index in range(expected_count)}
    paths = {path.name: path for path in input_dir.glob("index-*.json")}
    if set(paths) != expected_names:
        raise HybridComponentAnalysisError(
            "Record directory is incomplete or contains unexpected record files."
        )

    records: list[dict[str, Any]] = []
    manifest: list[tuple[int, str]] = []
    expected_bindings = {
        "preregistration_sha256": audit.get("preregistration_sha256"),
        "runner_sha256": audit.get("runner_sha256"),
        "mdp_checkpoint_sha256": audit.get("mdp_checkpoint_sha256"),
        "polar_checkpoint_sha256": audit.get("polar_checkpoint_sha256"),
        "hybrid_configuration_sha256": audit.get("hybrid_configuration_sha256"),
    }
    for index in range(expected_count):
        path = paths[f"index-{index:03d}.json"]
        row = json.loads(path.read_text())
        if not isinstance(row, dict):
            raise HybridComponentAnalysisError(f"Record {index:03d} is not an object.")
        if row.get("artifact") != EXPECTED_RECORD_ARTIFACT:
            raise HybridComponentAnalysisError(f"Record {index:03d} artifact drifted.")
        if row.get("selection_index") != index:
            raise HybridComponentAnalysisError(f"Record {index:03d} index drifted.")
        if row.get("status") != "pass":
            raise HybridComponentAnalysisError(f"Record {index:03d} did not pass.")
        if row.get("partition") != "development":
            raise HybridComponentAnalysisError(f"Record {index:03d} partition drifted.")
        if row.get("confirmation_partition_opened") is not False:
            raise HybridComponentAnalysisError(
                f"Record {index:03d} opened confirmation."
            )
        if row.get("do_not_commit") is not True:
            raise HybridComponentAnalysisError(
                f"Record {index:03d} commit guard drifted."
            )
        for key, expected in expected_bindings.items():
            if row.get(key) != expected:
                raise HybridComponentAnalysisError(f"Record {index:03d} {key} drifted.")
        solvent = row.get("canonical_solvent")
        if not isinstance(solvent, str) or not solvent:
            raise HybridComponentAnalysisError(
                f"Record {index:03d} solvent identity is invalid."
            )
        continuum = _finite(
            row.get("continuum_polarization_kcal_mol"),
            name=f"record {index:03d} continuum polarization",
        )
        cds = _finite(
            row.get("smd_cds_kcal_mol"),
            name=f"record {index:03d} stock SMD-CDS",
        )
        predicted = _finite(
            row.get("predicted_delta_g_kcal_mol"),
            name=f"record {index:03d} predicted energy",
        )
        _finite(
            row.get("experimental_delta_g_kcal_mol"),
            name=f"record {index:03d} experimental energy",
        )
        if not math.isclose(
            predicted,
            continuum + cds,
            rel_tol=0.0,
            abs_tol=LEDGER_ABSOLUTE_TOLERANCE_KCAL_MOL,
        ):
            raise HybridComponentAnalysisError(
                f"Record {index:03d} component ledger does not close."
            )
        records.append(row)
        manifest.append((index, _sha256(path)))
    if _canonical_sha256(manifest) != audit.get("record_files_manifest_sha256"):
        raise HybridComponentAnalysisError("Record-file manifest drifted from audit.")
    return records


def analyze_hybrid_components(
    *,
    input_dir: Path,
    integrity_audit_path: Path,
    expected_count: int = EXPECTED_RECORD_COUNT,
) -> dict[str, object]:
    audit = json.loads(integrity_audit_path.read_text())
    if not isinstance(audit, dict):
        raise HybridComponentAnalysisError("Integrity audit is not a JSON object.")
    audit_verification = audit.get("verification_sha256")
    audit_without_verification = dict(audit)
    audit_without_verification.pop("verification_sha256", None)
    if audit_verification != _canonical_sha256(audit_without_verification):
        raise HybridComponentAnalysisError("Integrity-audit self-hash drifted.")
    records = _load_bound_records(
        input_dir=input_dir,
        audit=audit,
        expected_count=expected_count,
    )
    water = [row for row in records if row["canonical_solvent"] == "water"]
    nonaqueous = [row for row in records if row["canonical_solvent"] != "water"]
    per_solvent: dict[str, object] = {}
    for solvent in sorted({str(row["canonical_solvent"]) for row in records}):
        per_solvent[solvent] = _component_metrics(
            [row for row in records if row["canonical_solvent"] == solvent]
        )
    payload: dict[str, object] = {
        "artifact": "route2-hybrid-smd-component-diagnostic-v1",
        "status": "complete",
        "do_not_commit": True,
        "partition": "development",
        "confirmation_partition_opened": False,
        "fitting_or_calibration_performed": False,
        "record_count": len(records),
        "integrity_audit_sha256": _sha256(integrity_audit_path),
        "integrity_audit_verification_sha256": audit.get("verification_sha256"),
        "record_files_manifest_sha256": audit.get("record_files_manifest_sha256"),
        "preregistration_sha256": audit.get("preregistration_sha256"),
        "runner_sha256": audit.get("runner_sha256"),
        "mdp_checkpoint_sha256": audit.get("mdp_checkpoint_sha256"),
        "polar_checkpoint_sha256": audit.get("polar_checkpoint_sha256"),
        "hybrid_configuration_sha256": audit.get("hybrid_configuration_sha256"),
        "all_records": _component_metrics(records),
        "water_records": _component_metrics(water),
        "nonaqueous_records": _component_metrics(nonaqueous),
        "per_solvent": per_solvent,
        "claim_boundary": (
            "Read-only paired M0/M1 component diagnosis on complete frozen v3 "
            "development evidence. It performs no fit, calibration, confirmation "
            "access, source validation, force admission, or production admission."
        ),
    }
    payload["analysis_sha256"] = _canonical_sha256(payload)
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--integrity-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze_hybrid_components(
        input_dir=args.input_dir,
        integrity_audit_path=args.integrity_audit,
    )
    _write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
