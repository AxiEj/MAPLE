#!/usr/bin/env python3
"""Re-score frozen Route-2 endpoints with a PCM-half-coupling-only ledger.

This script does not run a model, tune a parameter, or select records from
experimental errors.  It re-composes already frozen exact-GTO endpoint
components as

    Delta G = G_PCM(= 1/2 <c, f_reac>) + G_CDS + G_standard_state

and writes the excluded field-conditioned MACE energy change explicitly.  It
is therefore an algebraic comparison of two ledgers on identical SCF density,
continuum, geometry, and CDS endpoints, not a new physical validation run.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS = REPO_ROOT / "docs" / "implicit-solvation" / "benchmarks"
DEFAULT_HISTORICAL = (
    BENCHMARKS / "route2-legacy-exact-gto-historical-freesolv12-diagnostic-v1.json"
)
DEFAULT_CONFIRMATION = (
    BENCHMARKS / "route2-legacy-exact-gto-confirmation123-diagnostic-v1.json"
)
DEFAULT_HISTORICAL_LOCK = (
    BENCHMARKS / "route2-v0-historical-freesolv10-regression-v1.json"
)
DEFAULT_OUTPUT = BENCHMARKS / "route2-direct-pcm-half-coupling-reledger-v1.json"
LEDGER_VERSION = "pcm-half-coupling-only-v1"
KCAL_TOLERANCE = 1.0e-10


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return loaded


def _require_finite(value: Any, *, label: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite; received {value!r}.")
    return converted


def _successful_reledger_row(record: dict[str, Any]) -> dict[str, Any] | None:
    diagnostic = record.get("legacy_exact_gto_diagnostic")
    if not isinstance(diagnostic, dict):
        raise TypeError(
            f"{record.get('compound_id')}: missing legacy_exact_gto_diagnostic."
        )
    if "status" in diagnostic:
        return None

    components = diagnostic.get("components_kcal_mol")
    if not isinstance(components, dict):
        raise TypeError(f"{record.get('compound_id')}: missing component ledger.")
    required = {
        "solute_polarization",
        "pcm_polarization",
        "cds",
        "standard_state",
        "delta_g_solv",
    }
    missing = required.difference(components)
    if missing:
        raise ValueError(
            f"{record.get('compound_id')}: component ledger is missing "
            + ", ".join(sorted(missing))
            + "."
        )
    legacy_prediction = _require_finite(
        diagnostic.get("predicted_kcal_mol"),
        label=f"{record.get('compound_id')} legacy prediction",
    )
    legacy_recomposition = sum(
        _require_finite(components[key], label=f"legacy {key}")
        for key in (
            "solute_polarization",
            "pcm_polarization",
            "cds",
            "standard_state",
        )
    )
    if abs(legacy_prediction - legacy_recomposition) > KCAL_TOLERANCE:
        raise ValueError(
            f"{record.get('compound_id')}: frozen legacy ledger does not "
            "recompose to predicted_kcal_mol."
        )
    stored_total = _require_finite(
        components["delta_g_solv"],
        label=f"{record.get('compound_id')} legacy delta_g_solv",
    )
    if abs(legacy_prediction - stored_total) > KCAL_TOLERANCE:
        raise ValueError(
            f"{record.get('compound_id')}: frozen legacy total differs from "
            "predicted_kcal_mol."
        )

    direct_prediction = sum(
        _require_finite(components[key], label=f"direct {key}")
        for key in ("pcm_polarization", "cds", "standard_state")
    )
    direct_components = {
        "solute_polarization": 0.0,
        "pcm_polarization": float(components["pcm_polarization"]),
        "electrostatic": float(components["pcm_polarization"]),
        "cds": float(components["cds"]),
        "standard_state": float(components["standard_state"]),
        "delta_g_solv": direct_prediction,
    }
    if (
        abs(
            direct_prediction
            - (
                direct_components["electrostatic"]
                + direct_components["cds"]
                + direct_components["standard_state"]
            )
        )
        > KCAL_TOLERANCE
    ):
        raise AssertionError("Direct PCM reledger failed to recompose.")

    experimental = _require_finite(
        record.get("experimental_kcal_mol"),
        label=f"{record.get('compound_id')} experimental value",
    )
    return {
        "compound_id": record["compound_id"],
        "name": record["name"],
        "functional_groups": list(record.get("functional_groups", [])),
        "element_class": record.get("element_class"),
        "heteroatom_bin": record.get("heteroatom_bin"),
        "size_bin": record.get("size_bin"),
        "flexibility_bin": record.get("flexibility_bin"),
        "experimental_kcal_mol": experimental,
        "legacy_prediction_kcal_mol": legacy_prediction,
        "legacy_signed_error_kcal_mol": legacy_prediction - experimental,
        "legacy_absolute_error_kcal_mol": abs(legacy_prediction - experimental),
        "direct_pcm_prediction_kcal_mol": direct_prediction,
        "direct_pcm_signed_error_kcal_mol": direct_prediction - experimental,
        "direct_pcm_absolute_error_kcal_mol": abs(direct_prediction - experimental),
        "excluded_field_conditioned_mace_energy_change_kcal_mol": float(
            components["solute_polarization"]
        ),
        "direct_pcm_components_kcal_mol": direct_components,
        "source_legacy_components_kcal_mol": dict(components),
    }


def _failure_row(record: dict[str, Any]) -> dict[str, Any]:
    diagnostic = record["legacy_exact_gto_diagnostic"]
    if not isinstance(diagnostic, dict) or "status" not in diagnostic:
        raise AssertionError("Expected a rejected provider record.")
    return {
        "compound_id": record["compound_id"],
        "name": record["name"],
        "functional_groups": list(record.get("functional_groups", [])),
        "status": diagnostic["status"],
        "reason": diagnostic.get("reason"),
        "reledger_status": "not-computable-without-a-published-source-endpoint",
    }


def _metrics(
    rows: list[dict[str, Any]],
    *,
    prediction_key: str,
    expected_record_count: int,
    failure_count: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot calculate metrics with no successful records.")
    errors = [row[f"{prediction_key}_signed_error_kcal_mol"] for row in rows]
    absolute_errors = [abs(error) for error in errors]
    worst = max(rows, key=lambda row: row[f"{prediction_key}_absolute_error_kcal_mol"])
    threshold = 1.5
    return {
        "expected_record_count": expected_record_count,
        "success_count": len(rows),
        "failure_count": failure_count,
        "coverage_fraction": len(rows) / expected_record_count,
        "mae_kcal_mol": sum(absolute_errors) / len(absolute_errors),
        "rmse_kcal_mol": math.sqrt(
            sum(error * error for error in errors) / len(errors)
        ),
        "mean_signed_error_kcal_mol": sum(errors) / len(errors),
        "maximum_absolute_error_kcal_mol": max(absolute_errors),
        "worst_compound_id": worst["compound_id"],
        "worst_name": worst["name"],
        "records_strictly_below_1_5_kcal_mol": sum(
            error < threshold for error in absolute_errors
        ),
        "records_at_or_above_1_5_kcal_mol": sum(
            error >= threshold for error in absolute_errors
        ),
        "all_successful_records_strictly_below_1_5_kcal_mol": all(
            error < threshold for error in absolute_errors
        ),
    }


def _functional_group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        labels = row["functional_groups"] or ["unlabelled"]
        for label in labels:
            groups[str(label) or "unlabelled"].append(row)
    result: dict[str, Any] = {}
    for label, members in sorted(groups.items()):
        direct_abs = [
            member["direct_pcm_absolute_error_kcal_mol"] for member in members
        ]
        result[label] = {
            "count": len(members),
            "mae_kcal_mol": sum(direct_abs) / len(direct_abs),
            "maximum_absolute_error_kcal_mol": max(direct_abs),
        }
    return result


def _reledger_panel(
    source: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = source.get("records")
    if not isinstance(records, list):
        raise TypeError("Source diagnostic must contain a records list.")
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for source_record in records:
        if not isinstance(source_record, dict):
            raise TypeError("Source record must be a JSON object.")
        row = _successful_reledger_row(source_record)
        if row is None:
            failures.append(_failure_row(source_record))
        else:
            successes.append(row)
    return successes, failures


def _locked_historical_gate(
    lock: dict[str, Any],
    historical_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    locked_records = lock.get("locked_records")
    if not isinstance(locked_records, list):
        raise TypeError("Historical lock must contain locked_records.")
    rows_by_id = {row["compound_id"]: row for row in historical_rows}
    selected: list[dict[str, Any]] = []
    for locked in locked_records:
        compound_id = locked["compound_id"]
        try:
            row = rows_by_id[compound_id]
        except KeyError as exc:
            raise ValueError(
                f"Mandatory historical record {compound_id} is absent from "
                "the reledger source."
            ) from exc
        experimental = _require_finite(
            locked["experimental_kcal_mol"],
            label=f"locked {compound_id} experimental value",
        )
        if abs(experimental - row["experimental_kcal_mol"]) > KCAL_TOLERANCE:
            raise ValueError(
                f"Mandatory historical record {compound_id} changed its "
                "experimental value."
            )
        selected.append(
            {
                **row,
                "chemical_class": locked["chemical_class"],
                "locked_ordinal": locked["ordinal"],
            }
        )
    gate = lock.get("acceptance_gate")
    if not isinstance(gate, dict):
        raise TypeError("Historical lock must contain acceptance_gate.")
    threshold = _require_finite(
        gate["maximum_absolute_error_threshold_kcal_mol"],
        label="historical maximum-error threshold",
    )
    historic_max = lock.get("historical_observed_maximum_error")
    if not isinstance(historic_max, dict):
        raise TypeError("Historical lock must name the old maximum-error record.")
    historic_max_id = historic_max["compound_id"]
    if historic_max_id not in {row["compound_id"] for row in selected}:
        raise ValueError("The historic 7.041-kcal/mol record is missing.")
    distinct_classes = sorted({row["chemical_class"] for row in selected})
    all_individually_below = all(
        row["direct_pcm_absolute_error_kcal_mol"] < threshold for row in selected
    )
    required_classes = int(
        gate["minimum_distinct_functional_group_or_scaffold_classes"]
    )
    return {
        "panel_id": lock["panel_id"],
        "expected_record_count": int(gate["expected_record_count"]),
        "actual_record_count": len(selected),
        "maximum_absolute_error_threshold_kcal_mol": threshold,
        "historic_7_041_kcal_mol_record": {
            "compound_id": historic_max_id,
            "name": historic_max["name"],
            "historical_absolute_error_kcal_mol": historic_max[
                "absolute_error_kcal_mol"
            ],
            "direct_pcm_absolute_error_kcal_mol": next(
                row["direct_pcm_absolute_error_kcal_mol"]
                for row in selected
                if row["compound_id"] == historic_max_id
            ),
        },
        "distinct_chemical_class_count": len(distinct_classes),
        "minimum_distinct_chemical_class_count": required_classes,
        "all_locked_records_individually_below_1_5_kcal_mol": (all_individually_below),
        "gate_satisfied_for_this_reledger_only": (
            len(selected) == int(gate["expected_record_count"])
            and historic_max_id in {row["compound_id"] for row in selected}
            and len(distinct_classes) >= required_classes
            and all_individually_below
        ),
        "records": selected,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--historical-lock", type=Path, default=DEFAULT_HISTORICAL_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    historical_path = args.historical.resolve()
    confirmation_path = args.confirmation.resolve()
    lock_path = args.historical_lock.resolve()
    output_path = args.output.resolve()
    historical = _load_json(historical_path)
    confirmation = _load_json(confirmation_path)
    lock = _load_json(lock_path)

    historical_successes, historical_failures = _reledger_panel(historical)
    confirmation_successes, confirmation_failures = _reledger_panel(confirmation)
    if {row["compound_id"] for row in historical_successes}.intersection(
        row["compound_id"] for row in confirmation_successes
    ):
        raise ValueError("Historical and confirmation panels unexpectedly overlap.")

    all_successes = historical_successes + confirmation_successes
    historical_expected = int(historical["panel"]["expected_record_count"])
    confirmation_expected = int(confirmation["panel"]["expected_record_count"])
    output = {
        "schema_version": 1,
        "artifact": "route2-direct-pcm-half-coupling-reledger-v1",
        "status": "retrospective-frozen-endpoint-diagnostic-not-route2v-acceptance",
        "electrostatic_energy_ledger": LEDGER_VERSION,
        "formula": (
            "delta_G_solv = 0.5*<c_MACE-POLAR, f_reac_PCM> + G_CDS "
            "+ G_standard_state; the field-conditioned MACE energy change "
            "is excluded from the reported leaf ledger"
        ),
        "claim_boundary": (
            "This is a deterministic re-composition of frozen legacy exact-GTO "
            "SCF endpoints. It neither reruns MACE/PCM nor repairs the known "
            "nonvariational MACE fixed point, reciprocity defect, force "
            "inconsistency, point-l<=1 permanent-source limitation, or 46 "
            "PCMSolver provider rejections. It must not be called a common "
            "stationary free energy or a chemical-accuracy certification."
        ),
        "no_target_policy": {
            "experimental_values_used_only_for_post_hoc_metrics": True,
            "experimental_solvation_fit": False,
            "post_training": False,
            "fine_tuning": False,
            "map_or_uq_calibration": False,
            "radius_adjustment": False,
            "half_coupling_adjustment": False,
            "record_exclusion": False,
            "reledger_rule_selected_before_reading_errors": True,
        },
        "source_artifacts": {
            "historical_freesolv12": {
                "path": str(historical_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(historical_path),
                "source_artifact": historical.get("artifact"),
            },
            "confirmation123": {
                "path": str(confirmation_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(confirmation_path),
                "source_artifact": confirmation.get("artifact"),
            },
            "mandatory_historical10_lock": {
                "path": str(lock_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(lock_path),
                "source_panel_id": lock.get("panel_id"),
            },
        },
        "summary": {
            "historical_freesolv12": {
                "legacy": _metrics(
                    historical_successes,
                    prediction_key="legacy",
                    expected_record_count=historical_expected,
                    failure_count=len(historical_failures),
                ),
                "direct_pcm": _metrics(
                    historical_successes,
                    prediction_key="direct_pcm",
                    expected_record_count=historical_expected,
                    failure_count=len(historical_failures),
                ),
            },
            "confirmation123": {
                "legacy": _metrics(
                    confirmation_successes,
                    prediction_key="legacy",
                    expected_record_count=confirmation_expected,
                    failure_count=len(confirmation_failures),
                ),
                "direct_pcm": _metrics(
                    confirmation_successes,
                    prediction_key="direct_pcm",
                    expected_record_count=confirmation_expected,
                    failure_count=len(confirmation_failures),
                ),
            },
            "combined_nonoverlapping_successful_endpoints": {
                "legacy": _metrics(
                    all_successes,
                    prediction_key="legacy",
                    expected_record_count=(historical_expected + confirmation_expected),
                    failure_count=(
                        len(historical_failures) + len(confirmation_failures)
                    ),
                ),
                "direct_pcm": _metrics(
                    all_successes,
                    prediction_key="direct_pcm",
                    expected_record_count=(historical_expected + confirmation_expected),
                    failure_count=(
                        len(historical_failures) + len(confirmation_failures)
                    ),
                ),
            },
        },
        "mandatory_historical10_gate": _locked_historical_gate(
            lock,
            historical_successes,
        ),
        "functional_group_coverage": {
            "historical_freesolv12": _functional_group_metrics(historical_successes),
            "confirmation123_successes_only": _functional_group_metrics(
                confirmation_successes
            ),
        },
        "panels": {
            "historical_freesolv12": {
                "successful_records": historical_successes,
                "provider_rejections": historical_failures,
            },
            "confirmation123": {
                "successful_records": confirmation_successes,
                "provider_rejections": confirmation_failures,
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
