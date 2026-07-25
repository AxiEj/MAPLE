#!/usr/bin/env python
"""Score a sealed Route 1 MBAR artifact against development labels."""

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
        expected_type="route1-multi-mlip-obc2-mbar-analysis",
    )
    label_summary = _load_sealed(
        label_summary_path,
        expected_type="route1-multi-mlip-force-consistent-obc2-ti-development",
    )
    if analysis["label_boundary"]["experimental_labels_read"] is not False:
        raise ValueError("MBAR analysis was not sealed behind a label-free boundary.")
    if analysis["decision"]["promotion_allowed"] is not False:
        raise ValueError("The short MBAR artifact cannot be promotion-eligible.")

    labels: dict[tuple[str, str], dict[str, Any]] = {}
    for row in label_summary["case_results"]:
        identity = (row["model"], row["compound_id"])
        if identity in labels:
            raise ValueError(f"Duplicate label identity: {identity}.")
        labels[identity] = row
    analysis_identities = {
        (row["model"], row["compound_id"]) for row in analysis["records"]
    }
    if set(labels) != analysis_identities:
        raise ValueError("MBAR and development-label identities do not match.")

    cases: list[dict[str, Any]] = []
    model_names = sorted({row["model"] for row in analysis["records"]})
    model_summaries: dict[str, Any] = {}
    for model in model_names:
        model_rows = [
            row for row in analysis["records"] if row["model"] == model
        ]
        predicted = np.asarray(
            [row["mbar"]["endpoint_delta_g_kcal_mol"] for row in model_rows],
            dtype=np.float64,
        )
        experimental = np.asarray(
            [
                labels[(model, row["compound_id"])]["experimental_kcal_mol"]
                for row in model_rows
            ],
            dtype=np.float64,
        )
        fixed_geometry = np.asarray(
            [
                labels[(model, row["compound_id"])][
                    "fixed_geometry_obc2_ace_kcal_mol"
                ]
                for row in model_rows
            ],
            dtype=np.float64,
        )
        mbar_metrics = _metrics(predicted, experimental)
        fixed_metrics = _metrics(fixed_geometry, experimental)
        model_summaries[model] = {
            "case_count": len(model_rows),
            "fixed_geometry": fixed_metrics,
            "mbar": mbar_metrics,
            "mbar_minus_fixed_geometry_mae_kcal_mol": float(
                mbar_metrics["mae_kcal_mol"]
                - fixed_metrics["mae_kcal_mol"]
            ),
            "promotion_allowed": False,
        }
        for row, observed, baseline in zip(
            model_rows,
            experimental,
            fixed_geometry,
            strict=True,
        ):
            prediction = row["mbar"]["endpoint_delta_g_kcal_mol"]
            cases.append(
                {
                    "model": model,
                    "compound_id": row["compound_id"],
                    "name": row["name"],
                    "experimental_kcal_mol": float(observed),
                    "fixed_geometry_obc2_ace_kcal_mol": float(baseline),
                    "mbar_kcal_mol": float(prediction),
                    "mbar_absolute_error_kcal_mol": float(
                        abs(prediction - observed)
                    ),
                }
            )

    result = {
        "schema_version": 1,
        "artifact_type": "route1-multi-mlip-obc2-mbar-development-score",
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
        "case_results": cases,
        "model_summaries": model_summaries,
        "decision": {
            "result": "negative_development_diagnostic",
            "promotion_allowed": False,
            "reason": (
                "All three MBAR estimates worsen three-case development MAE, "
                "and the sealed analysis fails equilibrium and independent-"
                "sample gates."
            ),
        },
        "label_use_boundary": {
            "analysis_sealed_before_scoring": True,
            "labels_read_only_by_scoring_phase": True,
            "confirmation_partition": False,
        },
        "implementation_provenance": {
            "scorer": Path(__file__)
            .resolve()
            .relative_to(REPOSITORY_ROOT)
            .as_posix(),
            "scorer_sha256": sha256_file(__file__),
        },
        "claim_boundary": {
            "chemical_accuracy_established": False,
            "equilibrated_sampling_proven": False,
            "product_solvation_free_energy_established": False,
        },
    }
    return seal_artifact(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        default=str(
            BENCHMARK_DIR / "route1-multi-mlip-obc2-mbar-2026-07-25.json"
        ),
    )
    parser.add_argument(
        "--labels",
        default=str(
            BENCHMARK_DIR / "route1-multi-mlip-obc2-ti-2026-07-25.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            BENCHMARK_DIR
            / "route1-multi-mlip-obc2-mbar-score-2026-07-25.json"
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
