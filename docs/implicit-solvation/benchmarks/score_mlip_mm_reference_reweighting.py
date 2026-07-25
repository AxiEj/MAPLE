#!/usr/bin/env python3
"""Score a sealed MM-reference/MLIP-reweighting artifact on development labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_DIR.parents[2]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_core import (  # noqa: E402
    artifact_content_sha256,
    load_json,
    seal_artifact,
    sha256_file,
    write_json_atomic,
)


def _load_sealed(path: str | Path, *, expected_type: str) -> dict[str, Any]:
    artifact = load_json(path)
    if artifact.get("artifact_type") != expected_type:
        raise ValueError(f"Unexpected artifact type in {path}.")
    if artifact.get("content_sha256") != artifact_content_sha256(artifact):
        raise ValueError(f"Artifact self-hash changed: {path}.")
    return artifact


def _metrics(predicted: np.ndarray, experimental: np.ndarray) -> dict[str, Any]:
    error = predicted - experimental
    return {
        "n": int(len(error)),
        "mean_signed_error_kcal_mol": float(error.mean()),
        "mae_kcal_mol": float(np.abs(error).mean()),
        "rmse_kcal_mol": float(np.sqrt(np.square(error).mean())),
        "maximum_absolute_error_kcal_mol": float(np.abs(error).max()),
    }


def score(
    analysis_path: str | Path,
    label_summary_path: str | Path,
) -> dict[str, Any]:
    analysis_path = Path(analysis_path).resolve()
    label_summary_path = Path(label_summary_path).resolve()
    analysis = _load_sealed(
        analysis_path,
        expected_type="route1-mmgb-reference-mlip-reweighting-summary",
    )
    label_summary = _load_sealed(
        label_summary_path,
        expected_type="route1-multi-mlip-force-consistent-obc2-ti-development",
    )
    if analysis["label_boundary"]["experimental_labels_read"] is not False:
        raise ValueError("The energy analysis was not sealed label-free.")
    if analysis["decision"]["promotion_allowed"] is not False:
        raise ValueError("The reference-reweighting diagnostic is promotable.")

    labels: dict[tuple[str, str], dict[str, Any]] = {}
    for row in label_summary["case_results"]:
        identity = (row["model"], row["compound_id"])
        if identity in labels:
            raise ValueError(f"Duplicate development-label identity: {identity}.")
        labels[identity] = row
    identities = {(row["model"], row["compound_id"]) for row in analysis["records"]}
    if set(labels) != identities:
        raise ValueError("Analysis and development-label identities do not match.")

    method_fields = {
        "reference_mm_mbar": "reference_solvation_mbar_kcal_mol",
        "reference_only_accelerated": ("reference_only_accelerated_delta_g_kcal_mol"),
        "bidirectional_indirect_mbar": "bidirectional_mbar_delta_g_kcal_mol",
        "direct_target_multistate_mbar": ("direct_target_multistate_mbar_kcal_mol"),
    }
    model_summaries: dict[str, Any] = {}
    case_results: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in analysis["records"]}):
        rows = [row for row in analysis["records"] if row["model"] == model]
        experimental = np.asarray(
            [
                labels[(model, row["compound_id"])]["experimental_kcal_mol"]
                for row in rows
            ],
            dtype=np.float64,
        )
        fixed_geometry = np.asarray(
            [
                labels[(model, row["compound_id"])]["fixed_geometry_obc2_ace_kcal_mol"]
                for row in rows
            ],
            dtype=np.float64,
        )
        metrics = {"fixed_geometry_obc2_ace": _metrics(fixed_geometry, experimental)}
        for method, field in method_fields.items():
            predicted = np.asarray(
                [row[field] for row in rows],
                dtype=np.float64,
            )
            metrics[method] = _metrics(predicted, experimental)
        fixed_mae = metrics["fixed_geometry_obc2_ace"]["mae_kcal_mol"]
        model_summaries[model] = {
            "case_count": len(rows),
            **metrics,
            "reference_only_accelerated_minus_fixed_geometry_mae_kcal_mol": (
                metrics["reference_only_accelerated"]["mae_kcal_mol"] - fixed_mae
            ),
            "bidirectional_indirect_minus_direct_target_mae_kcal_mol": (
                metrics["bidirectional_indirect_mbar"]["mae_kcal_mol"]
                - metrics["direct_target_multistate_mbar"]["mae_kcal_mol"]
            ),
            "promotion_allowed": False,
        }
        for row, observed, fixed in zip(
            rows,
            experimental,
            fixed_geometry,
            strict=True,
        ):
            predictions = {
                method: float(row[field]) for method, field in method_fields.items()
            }
            case_results.append(
                {
                    "model": model,
                    "compound_id": row["compound_id"],
                    "name": row["name"],
                    "experimental_kcal_mol": float(observed),
                    "fixed_geometry_obc2_ace_kcal_mol": float(fixed),
                    "predictions_kcal_mol": predictions,
                    "absolute_errors_kcal_mol": {
                        "fixed_geometry_obc2_ace": float(abs(fixed - observed)),
                        **{
                            method: float(abs(value - observed))
                            for method, value in predictions.items()
                        },
                    },
                }
            )

    result = {
        "schema_version": 1,
        "artifact_type": "route1-mmgb-reference-mlip-reweighting-development-score",
        "analysis_artifact": {
            "path": analysis_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "file_sha256": sha256_file(analysis_path),
            "content_sha256": analysis["content_sha256"],
        },
        "label_source": {
            "path": label_summary_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "file_sha256": sha256_file(label_summary_path),
            "content_sha256": label_summary["content_sha256"],
            "partition": "development",
        },
        "case_results": case_results,
        "model_summaries": model_summaries,
        "decision": {
            "result": "negative_development_diagnostic",
            "promotion_allowed": False,
            "reason": (
                "The reference-only estimate worsens three-case development "
                "MAE for every MLIP, and the sealed label-free analysis passes "
                "zero complete numerical gate sets."
            ),
        },
        "label_use_boundary": {
            "analysis_sealed_before_scoring": True,
            "labels_read_only_by_scoring_phase": True,
            "labels_changed_neither_protocol_nor_energy_estimates": True,
            "confirmation_partition": False,
        },
        "claim_boundary": {
            "chemical_accuracy_established": False,
            "equilibrated_sampling_proven": False,
            "reference_only_production_validated": False,
            "product_solvation_free_energy_established": False,
        },
        "implementation_provenance": {
            "scorer": Path(__file__).resolve().relative_to(REPOSITORY_ROOT).as_posix(),
            "scorer_sha256": sha256_file(__file__),
        },
    }
    return seal_artifact(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        default=str(BENCHMARK_DIR / "route1-mm-reference-reweighting-2026-07-25.json"),
    )
    parser.add_argument(
        "--labels",
        default=str(BENCHMARK_DIR / "route1-multi-mlip-obc2-ti-2026-07-25.json"),
    )
    parser.add_argument(
        "--output",
        default=str(
            BENCHMARK_DIR / "route1-mm-reference-reweighting-score-2026-07-25.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = score(args.analysis, args.labels)
    write_json_atomic(args.output, result)
    print(json.dumps(result["model_summaries"], sort_keys=True))
    print(result["decision"]["result"])


if __name__ == "__main__":
    main()
