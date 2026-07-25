#!/usr/bin/env python3
"""Score sealed nonequilibrium-switching estimates on development labels."""

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


def _index(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        identity = (row["model"], row["compound_id"])
        if identity in indexed:
            raise ValueError(f"Duplicate model/case identity: {identity}.")
        indexed[identity] = row
    return indexed


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
    endpoint_analysis_path: str | Path,
    label_summary_path: str | Path,
) -> dict[str, Any]:
    analysis_path = Path(analysis_path).resolve()
    endpoint_analysis_path = Path(endpoint_analysis_path).resolve()
    label_summary_path = Path(label_summary_path).resolve()
    analysis = _load_sealed(
        analysis_path,
        expected_type="route1-mmgb-to-mlipgb-nonequilibrium-switching-summary",
    )
    endpoint_analysis = _load_sealed(
        endpoint_analysis_path,
        expected_type="route1-mmgb-reference-mlip-reweighting-summary",
    )
    label_summary = _load_sealed(
        label_summary_path,
        expected_type="route1-multi-mlip-force-consistent-obc2-ti-development",
    )
    if analysis["label_boundary"]["experimental_labels_read"] is not False:
        raise ValueError("The switching analysis was not sealed label-free.")
    if analysis["decision"]["promotion_allowed"] is not False:
        raise ValueError("The switching diagnostic is promotable.")
    if analysis["matrix_complete"] is not True:
        raise ValueError("The switching matrix is incomplete.")

    switching = _index(analysis["records"])
    endpoint = _index(endpoint_analysis["records"])
    labels = _index(label_summary["case_results"])
    if set(switching) != set(endpoint) or set(switching) != set(labels):
        raise ValueError("Switching, endpoint, and label identities do not match.")

    case_results: list[dict[str, Any]] = []
    for identity in sorted(switching):
        switch_row = switching[identity]
        endpoint_row = endpoint[identity]
        label_row = labels[identity]
        predictions = {
            "fixed_geometry_obc2_ace": float(
                label_row["fixed_geometry_obc2_ace_kcal_mol"]
            ),
            "reference_mm_mbar": float(
                endpoint_row["reference_solvation_mbar_kcal_mol"]
            ),
            "endpoint_reweighting_bidirectional": float(
                endpoint_row["bidirectional_mbar_delta_g_kcal_mol"]
            ),
            "nonequilibrium_switching_20fs": float(
                switch_row["indirect_target_solvation_kcal_mol"]
            ),
            "direct_target_multistate_mbar": float(
                switch_row["direct_target_multistate_mbar_kcal_mol"]
            ),
        }
        observed = float(label_row["experimental_kcal_mol"])
        case_results.append(
            {
                "model": identity[0],
                "compound_id": identity[1],
                "name": switch_row["name"],
                "experimental_kcal_mol": observed,
                "predictions_kcal_mol": predictions,
                "absolute_errors_kcal_mol": {
                    method: abs(value - observed)
                    for method, value in predictions.items()
                },
                "switching_minus_direct_target_mbar_kcal_mol": float(
                    switch_row["indirect_minus_direct_mbar_kcal_mol"]
                ),
                "switching_numerical_gate_pass": bool(
                    switch_row["all_numerical_checks_pass"]
                ),
            }
        )

    methods = tuple(case_results[0]["predictions_kcal_mol"])
    model_summaries: dict[str, Any] = {}
    for model in sorted({row["model"] for row in case_results}):
        rows = [row for row in case_results if row["model"] == model]
        experimental = np.asarray(
            [row["experimental_kcal_mol"] for row in rows],
            dtype=np.float64,
        )
        metrics = {
            method: _metrics(
                np.asarray(
                    [row["predictions_kcal_mol"][method] for row in rows],
                    dtype=np.float64,
                ),
                experimental,
            )
            for method in methods
        }
        model_summaries[model] = {
            "case_count": len(rows),
            **metrics,
            "mean_absolute_switching_minus_direct_target_mbar_kcal_mol": float(
                np.mean(
                    np.abs(
                        [
                            row["switching_minus_direct_target_mbar_kcal_mol"]
                            for row in rows
                        ]
                    )
                )
            ),
            "maximum_absolute_switching_minus_direct_target_mbar_kcal_mol": float(
                np.max(
                    np.abs(
                        [
                            row["switching_minus_direct_target_mbar_kcal_mol"]
                            for row in rows
                        ]
                    )
                )
            ),
            "nonequilibrium_minus_fixed_geometry_mae_kcal_mol": (
                metrics["nonequilibrium_switching_20fs"]["mae_kcal_mol"]
                - metrics["fixed_geometry_obc2_ace"]["mae_kcal_mol"]
            ),
            "nonequilibrium_minus_endpoint_reweighting_mae_kcal_mol": (
                metrics["nonequilibrium_switching_20fs"]["mae_kcal_mol"]
                - metrics["endpoint_reweighting_bidirectional"]["mae_kcal_mol"]
            ),
            "nonequilibrium_minus_direct_target_mbar_mae_kcal_mol": (
                metrics["nonequilibrium_switching_20fs"]["mae_kcal_mol"]
                - metrics["direct_target_multistate_mbar"]["mae_kcal_mol"]
            ),
            "switching_numerical_gate_pass_count": sum(
                row["switching_numerical_gate_pass"] for row in rows
            ),
            "promotion_allowed": False,
        }

    overall_experimental = np.asarray(
        [row["experimental_kcal_mol"] for row in case_results],
        dtype=np.float64,
    )
    overall_metrics = {
        method: _metrics(
            np.asarray(
                [row["predictions_kcal_mol"][method] for row in case_results],
                dtype=np.float64,
            ),
            overall_experimental,
        )
        for method in methods
    }
    overall_switching_direct_difference = np.abs(
        [row["switching_minus_direct_target_mbar_kcal_mol"] for row in case_results]
    )
    overall_metrics["switching_vs_direct_target_mbar"] = {
        "n": len(case_results),
        "mean_absolute_difference_kcal_mol": float(
            np.mean(overall_switching_direct_difference)
        ),
        "maximum_absolute_difference_kcal_mol": float(
            np.max(overall_switching_direct_difference)
        ),
    }
    improved_vs_fixed = sum(
        summary["nonequilibrium_minus_fixed_geometry_mae_kcal_mol"] < 0.0
        for summary in model_summaries.values()
    )
    improved_vs_endpoint = sum(
        summary["nonequilibrium_minus_endpoint_reweighting_mae_kcal_mol"] < 0.0
        for summary in model_summaries.values()
    )
    numerical_passes = analysis["aggregate_diagnostics"]["numerical_gate_pass_count"]
    result = {
        "schema_version": 1,
        "artifact_type": (
            "route1-mmgb-to-mlipgb-nonequilibrium-switching-development-score"
        ),
        "analysis_artifact": {
            "path": analysis_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "file_sha256": sha256_file(analysis_path),
            "content_sha256": analysis["content_sha256"],
        },
        "endpoint_analysis_artifact": {
            "path": endpoint_analysis_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "file_sha256": sha256_file(endpoint_analysis_path),
            "content_sha256": endpoint_analysis["content_sha256"],
        },
        "label_source": {
            "path": label_summary_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "file_sha256": sha256_file(label_summary_path),
            "content_sha256": label_summary["content_sha256"],
            "partition": "development",
        },
        "case_results": case_results,
        "model_summaries": model_summaries,
        "overall_metrics": overall_metrics,
        "decision": {
            "result": "numerically_unvalidated_development_diagnostic",
            "promotion_allowed": False,
            "reason": (
                f"{numerical_passes}/{len(case_results)} complete switching "
                "numerical gate sets passed; 20 fs switching improved "
                f"three-case MAE versus fixed geometry for {improved_vs_fixed}/"
                f"{len(model_summaries)} MLIPs and versus endpoint reweighting "
                f"for {improved_vs_endpoint}/{len(model_summaries)} MLIPs. "
                "Development labels cannot promote this diagnostic."
            ),
        },
        "cost_observations": {
            "target_energy_force_evaluations": analysis["aggregate_diagnostics"][
                "total_target_energy_force_evaluations"
            ],
            "switching_wall_seconds": analysis["aggregate_diagnostics"][
                "total_switching_wall_seconds"
            ],
            "bare_classical_mm_speed_claimed": False,
            "production_acceleration_established": False,
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
            "independent_switching_work_proven": False,
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
        default=str(
            BENCHMARK_DIR / "route1-mm-mlip-nonequilibrium-switching-2026-07-25.json"
        ),
    )
    parser.add_argument(
        "--endpoint-analysis",
        default=str(BENCHMARK_DIR / "route1-mm-reference-reweighting-2026-07-25.json"),
    )
    parser.add_argument(
        "--labels",
        default=str(BENCHMARK_DIR / "route1-multi-mlip-obc2-ti-2026-07-25.json"),
    )
    parser.add_argument(
        "--output",
        default=str(
            BENCHMARK_DIR
            / "route1-mm-mlip-nonequilibrium-switching-score-2026-07-25.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = score(args.analysis, args.endpoint_analysis, args.labels)
    write_json_atomic(args.output, result)
    print(json.dumps(result["model_summaries"], sort_keys=True))
    print(result["decision"]["result"])


if __name__ == "__main__":
    main()
