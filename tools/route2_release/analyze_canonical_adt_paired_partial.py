#!/usr/bin/env python3
"""Pair an in-progress canonical-ADT run with the frozen hybrid-v3 baseline.

This is a development-only diagnostic.  It performs no fit, parameter choice,
or confirmation-set access.  Its purpose is to distinguish a genuine change in
the electrostatic ledger (M0) from cancellation against the unchanged stock
SMD-CDS term (M1) before the expensive 505-record run is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

BASELINE_ARTIFACT = "route2-hybrid-smd-development-record-v3"
CANDIDATE_ARTIFACT = "route2-mace-mdp-polar-canonical-adt-ddx-development-record-v1"
IDENTITY_FIELDS = (
    "selection_index",
    "dataset_row_sha256",
    "geometry_sha256",
    "opaque_record_id",
    "selection_score_sha256",
    "canonical_solvent",
    "experimental_delta_g_kcal_mol",
    "mdp_checkpoint_sha256",
    "polar_checkpoint_sha256",
)


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


def _read_indexed_records(directory: Path) -> dict[int, tuple[Path, dict[str, Any]]]:
    records: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(directory.glob("index-*.json")):
        payload = json.loads(path.read_text())
        index = payload.get("selection_index")
        if type(index) is not int or index < 0:
            raise ValueError(f"Invalid selection index in {path}.")
        if index in records:
            raise ValueError(f"Duplicate selection index {index}.")
        records[index] = (path, payload)
    if not records:
        raise ValueError(f"No index records found in {directory}.")
    return records


def _metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    signed = predicted - target
    absolute = np.abs(signed)
    return {
        "record_count": int(predicted.size),
        "mean_signed_error_kcal_mol": float(np.mean(signed)),
        "mean_absolute_error_kcal_mol": float(np.mean(absolute)),
        "root_mean_square_error_kcal_mol": float(np.sqrt(np.mean(np.square(signed)))),
        "q95_absolute_error_kcal_mol": float(np.quantile(absolute, 0.95)),
        "maximum_absolute_error_kcal_mol": float(np.max(absolute)),
    }


def _paired_counts(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, int]:
    tolerance = 1.0e-12
    delta = np.abs(baseline) - np.abs(candidate)
    helped = int(np.count_nonzero(delta > tolerance))
    worsened = int(np.count_nonzero(delta < -tolerance))
    return {
        "helped_count": helped,
        "worsened_count": worsened,
        "tied_count": int(delta.size - helped - worsened),
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(value):
        raise ValueError("Nonfinite paired Pearson correlation.")
    return value


def analyze(baseline_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    baseline_records = _read_indexed_records(baseline_dir)
    candidate_records = _read_indexed_records(candidate_dir)
    matched: list[tuple[Path, dict[str, Any], Path, dict[str, Any]]] = []
    candidate_failures: list[int] = []

    for index, (candidate_path, candidate) in sorted(candidate_records.items()):
        if candidate.get("artifact") != CANDIDATE_ARTIFACT:
            raise ValueError(f"Candidate artifact drift in {candidate_path}.")
        if candidate.get("partition") != "development":
            raise ValueError(f"Candidate partition drift in {candidate_path}.")
        if candidate.get("confirmation_partition_opened") is not False:
            raise ValueError(f"Candidate confirmation flag drift in {candidate_path}.")
        if candidate.get("status") != "pass":
            candidate_failures.append(index)
            continue
        try:
            baseline_path, baseline = baseline_records[index]
        except KeyError as exc:
            raise ValueError(f"Missing baseline record {index}.") from exc
        if baseline.get("artifact") != BASELINE_ARTIFACT:
            raise ValueError(f"Baseline artifact drift in {baseline_path}.")
        if baseline.get("status") != "pass":
            raise ValueError(f"Baseline record {index} did not pass.")
        for field in IDENTITY_FIELDS:
            if baseline.get(field) != candidate.get(field):
                raise ValueError(f"Paired identity mismatch for {field} at {index}.")
        baseline_cds = float(baseline["smd_cds_kcal_mol"])
        candidate_cds = float(candidate["smd_cds_kcal_mol"])
        if not np.isclose(baseline_cds, candidate_cds, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Stock SMD-CDS drift at {index}.")
        candidate_m0 = float(candidate["continuum_polarization_kcal_mol"])
        candidate_m1 = float(
            candidate["m1_electrostatic_plus_stock_smd_cds"][
                "predicted_delta_g_kcal_mol"
            ]
        )
        if not np.isclose(
            candidate_m1, candidate_m0 + candidate_cds, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(f"Candidate M1 ledger identity drift at {index}.")
        baseline_m0 = float(baseline["continuum_polarization_kcal_mol"])
        baseline_m1 = float(baseline["predicted_delta_g_kcal_mol"])
        if not np.isclose(
            baseline_m1, baseline_m0 + baseline_cds, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(f"Baseline M1 ledger identity drift at {index}.")
        matched.append((baseline_path, baseline, candidate_path, candidate))

    if not matched:
        raise ValueError("No passing candidate records could be paired.")

    target = np.asarray(
        [
            float(candidate["experimental_delta_g_kcal_mol"])
            for _, _, _, candidate in matched
        ]
    )
    baseline_m0 = np.asarray(
        [
            float(baseline["continuum_polarization_kcal_mol"])
            for _, baseline, _, _ in matched
        ]
    )
    baseline_m1 = np.asarray(
        [float(baseline["predicted_delta_g_kcal_mol"]) for _, baseline, _, _ in matched]
    )
    candidate_m0 = np.asarray(
        [
            float(candidate["continuum_polarization_kcal_mol"])
            for _, _, _, candidate in matched
        ]
    )
    candidate_m1 = np.asarray(
        [
            float(
                candidate["m1_electrostatic_plus_stock_smd_cds"][
                    "predicted_delta_g_kcal_mol"
                ]
            )
            for _, _, _, candidate in matched
        ]
    )
    baseline_m0_error = baseline_m0 - target
    baseline_m1_error = baseline_m1 - target
    candidate_m0_error = candidate_m0 - target
    candidate_m1_error = candidate_m1 - target
    polarization_shift = candidate_m0 - baseline_m0
    baseline_m0_metrics = _metrics(baseline_m0, target)
    candidate_m0_metrics = _metrics(candidate_m0, target)
    baseline_m1_metrics = _metrics(baseline_m1, target)
    candidate_m1_metrics = _metrics(candidate_m1, target)

    payload: dict[str, Any] = {
        "artifact": "route2-canonical-adt-vs-hybrid-v3-paired-interim-v1",
        "status": "interim-development-diagnostic",
        "do_not_commit_result_records": True,
        "confirmation_partition_opened": False,
        "matched_record_count": len(matched),
        "candidate_record_count_seen": len(candidate_records),
        "candidate_failure_indices": candidate_failures,
        "matched_selection_indices": [
            int(candidate["selection_index"]) for _, _, _, candidate in matched
        ],
        "baseline_m0_metrics": baseline_m0_metrics,
        "candidate_m0_metrics": candidate_m0_metrics,
        "baseline_m1_metrics": baseline_m1_metrics,
        "candidate_m1_metrics": candidate_m1_metrics,
        "paired_change": {
            "m0_mae_improvement_kcal_mol": float(
                baseline_m0_metrics["mean_absolute_error_kcal_mol"]
                - candidate_m0_metrics["mean_absolute_error_kcal_mol"]
            ),
            "m1_mae_improvement_kcal_mol": float(
                baseline_m1_metrics["mean_absolute_error_kcal_mol"]
                - candidate_m1_metrics["mean_absolute_error_kcal_mol"]
            ),
            "m0_absolute_error_counts": _paired_counts(
                candidate_m0_error, baseline_m0_error
            ),
            "m1_absolute_error_counts": _paired_counts(
                candidate_m1_error, baseline_m1_error
            ),
            "polarization_shift_mean_kcal_mol": float(np.mean(polarization_shift)),
            "polarization_shift_median_kcal_mol": float(np.median(polarization_shift)),
            "polarization_shift_minimum_kcal_mol": float(np.min(polarization_shift)),
            "polarization_shift_maximum_kcal_mol": float(np.max(polarization_shift)),
            "m0_prediction_pearson_r": _pearson(baseline_m0, candidate_m0),
            "m1_prediction_pearson_r": _pearson(baseline_m1, candidate_m1),
            "stock_smd_cds_maximum_absolute_difference_kcal_mol": float(
                np.max(
                    np.abs((baseline_m1 - baseline_m0) - (candidate_m1 - candidate_m0))
                )
            ),
        },
        "input_records_sha256": {
            "baseline": {path.name: _sha256(path) for path, _, _, _ in matched},
            "candidate": {path.name: _sha256(path) for _, _, path, _ in matched},
        },
        "claim_boundary": (
            "Interim paired diagnostic over currently completed development records. "
            "It performs no fit or candidate selection and opens no confirmation "
            "record. It may reject a claimed early improvement but cannot establish "
            "the terminal 505-record accuracy of the still-running candidate."
        ),
    }
    payload["analysis_sha256"] = _canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.baseline_dir, args.candidate_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
