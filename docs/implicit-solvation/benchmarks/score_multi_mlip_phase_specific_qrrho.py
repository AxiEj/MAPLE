#!/usr/bin/env python3
"""Score a complete sealed qRRHO artifact against FreeSolv labels.

This program is the only qRRHO stage that opens the label-bearing prepared
FreeSolv artifact. It refuses to do so until all 18 preregistered model-case
records and the aggregate energy artifact pass their hashes and completeness
checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core import (  # noqa: E402
    canonical_json_bytes,
    command_provenance,
    load_json,
    seal_artifact,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from run_multi_mlip_phase_specific_qrrho import (  # noqa: E402
    _contains_forbidden_label_key,
    _resolve_repository_file,
    _validate_self_hash,
    load_qrrho_protocol,
    validate_model_case_record,
)


def _error_metrics(
    prediction_kcal_mol: np.ndarray,
    experimental_kcal_mol: np.ndarray,
) -> dict[str, float | int]:
    prediction = np.asarray(prediction_kcal_mol, dtype=np.float64)
    experimental = np.asarray(experimental_kcal_mol, dtype=np.float64)
    if (
        prediction.shape != experimental.shape
        or prediction.ndim != 1
        or not len(prediction)
        or not np.isfinite(prediction).all()
        or not np.isfinite(experimental).all()
    ):
        raise ValueError("Scoring arrays must be finite, non-empty, and paired.")
    errors = prediction - experimental
    return {
        "n": int(len(errors)),
        "mse_kcal_mol": float(errors.mean()),
        "mae_kcal_mol": float(np.abs(errors).mean()),
        "rmse_kcal_mol": float(np.sqrt(np.square(errors).mean())),
        "maximum_absolute_error_kcal_mol": float(np.abs(errors).max()),
    }


def paired_bootstrap_mae_improvement(
    *,
    baseline_kcal_mol: np.ndarray,
    qrrho_kcal_mol: np.ndarray,
    experimental_kcal_mol: np.ndarray,
    resamples: int,
    random_seed: int,
) -> dict[str, Any]:
    """Bootstrap paired MAE(baseline)-MAE(qRRHO) by compound."""

    baseline = np.asarray(baseline_kcal_mol, dtype=np.float64)
    qrrho = np.asarray(qrrho_kcal_mol, dtype=np.float64)
    experimental = np.asarray(experimental_kcal_mol, dtype=np.float64)
    if (
        baseline.shape != qrrho.shape
        or baseline.shape != experimental.shape
        or baseline.ndim != 1
        or not len(baseline)
        or not np.isfinite(baseline).all()
        or not np.isfinite(qrrho).all()
        or not np.isfinite(experimental).all()
        or resamples <= 0
    ):
        raise ValueError("Invalid paired bootstrap inputs.")
    baseline_absolute_error = np.abs(baseline - experimental)
    qrrho_absolute_error = np.abs(qrrho - experimental)
    point = float(baseline_absolute_error.mean() - qrrho_absolute_error.mean())
    generator = np.random.default_rng(int(random_seed))
    indices = generator.integers(
        0,
        len(baseline),
        size=(int(resamples), len(baseline)),
    )
    distribution = baseline_absolute_error[indices].mean(axis=1) - qrrho_absolute_error[
        indices
    ].mean(axis=1)
    lower, upper = np.quantile(distribution, [0.025, 0.975])
    return {
        "point_improvement_kcal_mol": point,
        "bootstrap_lower_kcal_mol": float(lower),
        "bootstrap_upper_kcal_mol": float(upper),
        "bootstrap_mean_kcal_mol": float(distribution.mean()),
        "resamples": int(resamples),
        "random_seed": int(random_seed),
        "distribution_sha256": sha256_bytes(
            canonical_json_bytes(distribution.tolist())
        ),
    }


def validate_sealed_energy_artifact(
    *,
    protocol: dict[str, Any],
    fingerprint: str,
    source_manifest: dict[str, Any],
    artifact_path: Path,
) -> dict[str, Any]:
    """Validate every prediction before returning a label-access token."""

    if artifact_path.resolve().parent != SCRIPT_DIR:
        raise ValueError(
            "The qRRHO aggregate must reside in the frozen benchmark directory."
        )
    artifact = load_json(artifact_path)
    _validate_self_hash(artifact, name=str(artifact_path))
    expected = int(protocol["energy_artifact_gates"]["expected_model_case_count"])
    if (
        artifact.get("schema_version") != 1
        or artifact.get("artifact_type")
        != "route1-multi-mlip-phase-specific-qrrho-energy-artifact"
        or artifact.get("protocol_id") != protocol["protocol_id"]
        or artifact.get("protocol_fingerprint") != fingerprint
        or artifact.get("status") != "sealed-before-label-scoring"
        or artifact.get("source_partition") != "development"
        or artifact.get("durable_record_schema_version") != 1
        or artifact.get("source_manifest_content_sha256")
        != source_manifest["content_sha256"]
        or int(artifact.get("model_case_count", -1)) != expected
        or len(artifact.get("records", [])) != expected
        or artifact.get("claim_boundary") != protocol["claim_boundary"]
        or _contains_forbidden_label_key(artifact)
    ):
        raise ValueError("The qRRHO energy artifact is incomplete or crosses labels.")
    raw_root_value = artifact.get("durable_raw_root")
    if (
        not isinstance(raw_root_value, str)
        or not raw_root_value
        or "\\" in raw_root_value
    ):
        raise ValueError("The durable qRRHO evidence root is invalid.")
    raw_root_relative = Path(raw_root_value)
    if raw_root_relative.is_absolute() or ".." in raw_root_relative.parts:
        raise ValueError("The durable qRRHO evidence root escapes the repository.")
    raw_root = (REPOSITORY_ROOT / raw_root_relative).resolve()
    try:
        raw_root.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError("The durable qRRHO evidence root escapes by symlink.") from exc
    if not raw_root.is_dir():
        raise ValueError("The durable qRRHO evidence root is missing.")
    expected_raw_root = (
        artifact_path.resolve().parent / f"{artifact_path.resolve().stem}-raw"
    )
    if raw_root != expected_raw_root:
        raise ValueError(
            "The durable qRRHO evidence root is not the aggregate's fixed sibling."
        )

    observed_order = [
        (entry["model"], entry["compound_id"]) for entry in artifact["records"]
    ]
    expected_order = [
        (model["name"], case["compound_id"])
        for model in protocol["models"]
        for case in protocol["cases"]
    ]
    if observed_order != expected_order:
        raise ValueError("The sealed model-case order differs from the protocol.")
    models = {model["name"]: model for model in protocol["models"]}
    cases = {case["compound_id"]: case for case in protocol["cases"]}
    for entry in artifact["records"]:
        record_path = _resolve_repository_file(
            entry["record"],
            name="sealed model-case record path",
        )
        try:
            record_path.relative_to(raw_root)
        except ValueError as exc:
            raise ValueError("A model-case record escaped the durable root.") from exc
        if sha256_file(record_path) != entry["record_file_sha256"]:
            raise ValueError(f"Model-case record file changed: {record_path}.")
        record = load_json(record_path)
        try:
            validate_model_case_record(
                record,
                protocol=protocol,
                fingerprint=fingerprint,
                source_manifest=source_manifest,
                model=models[entry["model"]],
                protocol_case=cases[entry["compound_id"]],
                evidence_root=raw_root,
                require_durable_evidence=True,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid sealed model-case record: {record_path}."
            ) from exc
        if record.get("content_sha256") != entry["record_content_sha256"] or (
            canonical_json_bytes(record["prediction"])
            != canonical_json_bytes(entry["prediction"])
        ):
            raise ValueError(f"Invalid sealed model-case record: {record_path}.")
    return artifact


def _load_labels_after_seal(
    *,
    source_manifest: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    source = source_manifest["source_evidence"]
    prepared_path = REPOSITORY_ROOT / source["prepared_input"]
    if sha256_file(prepared_path) != source["prepared_input_sha256"]:
        raise ValueError("The pinned label-bearing prepared artifact changed.")
    prepared = load_json(prepared_path)
    candidates = {
        candidate["compound_id"]: candidate for candidate in prepared["candidates"]
    }
    selected = {}
    for case in protocol["cases"]:
        candidate = candidates.get(case["compound_id"])
        if candidate is None or candidate.get("partition") != "development":
            raise ValueError(f"Missing development label for {case['compound_id']}.")
        selected[case["compound_id"]] = candidate
    return selected, {
        "prepared_artifact": source["prepared_input"],
        "prepared_artifact_sha256": source["prepared_input_sha256"],
        "dataset_commit": prepared["dataset_commit"],
        "dataset_artifact_sha256": prepared["dataset_artifact_sha256"],
    }


def score_predictions(
    *,
    protocol: dict[str, Any],
    energy_artifact: dict[str, Any],
    labels: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply only the gates preregistered in the protocol."""

    scoring = protocol["post_seal_scoring"]
    case_ids = [case["compound_id"] for case in protocol["cases"]]
    experimental = np.asarray(
        [labels[compound_id]["experimental_kcal_mol"] for compound_id in case_ids],
        dtype=np.float64,
    )
    records = {
        (entry["model"], entry["compound_id"]): entry
        for entry in energy_artifact["records"]
    }
    baseline = np.asarray(
        [
            records[(protocol["models"][0]["name"], compound_id)]["prediction"][
                "single_reference_endpoint_baseline_kcal_mol"
            ]
            for compound_id in case_ids
        ],
        dtype=np.float64,
    )
    for model in protocol["models"][1:]:
        observed = np.asarray(
            [
                records[(model["name"], compound_id)]["prediction"][
                    "single_reference_endpoint_baseline_kcal_mol"
                ]
                for compound_id in case_ids
            ],
            dtype=np.float64,
        )
        if not np.array_equal(observed, baseline):
            raise ValueError("The fixed single-reference baseline changed by MLIP.")

    model_results = {}
    corrections = {}
    positive_count = 0
    significant_degradations = 0
    all_stable = True
    for model in protocol["models"]:
        name = model["name"]
        qrrho = np.asarray(
            [
                records[(name, compound_id)]["prediction"][
                    "primary_hydration_prediction_kcal_mol"
                ]
                for compound_id in case_ids
            ],
            dtype=np.float64,
        )
        bootstrap = paired_bootstrap_mae_improvement(
            baseline_kcal_mol=baseline,
            qrrho_kcal_mol=qrrho,
            experimental_kcal_mol=experimental,
            resamples=int(scoring["bootstrap"]["resamples"]),
            random_seed=int(scoring["bootstrap"]["random_seed"]),
        )
        positive = bool(
            bootstrap["point_improvement_kcal_mol"]
            >= float(
                scoring["positive_model_gate"]["minimum_point_improvement_kcal_mol"]
            )
            and bootstrap["bootstrap_lower_kcal_mol"]
            > float(
                scoring["positive_model_gate"][
                    "bootstrap_lower_bound_must_exceed_kcal_mol"
                ]
            )
        )
        significant_degradation = bool(
            bootstrap["point_improvement_kcal_mol"]
            <= -float(
                scoring["positive_model_gate"]["minimum_point_improvement_kcal_mol"]
            )
            and bootstrap["bootstrap_upper_kcal_mol"] < 0.0
        )
        positive_count += int(positive)
        significant_degradations += int(significant_degradation)

        unstable_cases = []
        per_case = []
        for index, compound_id in enumerate(case_ids):
            variants = records[(name, compound_id)]["prediction"]["variants"]
            primary = float(variants["primary"]["hydration_prediction_kcal_mol"])
            deviations = {
                variant_id: abs(float(value["hydration_prediction_kcal_mol"]) - primary)
                for variant_id, value in variants.items()
                if variant_id != "primary"
            }
            maximum = max(deviations.values())
            unstable = maximum > float(
                scoring["low_frequency_stability_gate"][
                    "per_case_instability_threshold_kcal_mol"
                ]
            )
            if unstable:
                unstable_cases.append(compound_id)
            per_case.append(
                {
                    "compound_id": compound_id,
                    "experimental_kcal_mol": float(experimental[index]),
                    "baseline_kcal_mol": float(baseline[index]),
                    "qrrho_primary_kcal_mol": float(qrrho[index]),
                    "qrrho_correction_from_baseline_kcal_mol": float(
                        qrrho[index] - baseline[index]
                    ),
                    "maximum_sensitivity_deviation_kcal_mol": maximum,
                    "sensitivity_deviation_kcal_mol": deviations,
                    "low_frequency_unstable": unstable,
                }
            )
        stable = len(unstable_cases) <= int(
            scoring["low_frequency_stability_gate"]["maximum_unstable_cases_per_model"]
        )
        all_stable = all_stable and stable
        corrections[name] = qrrho - baseline
        model_results[name] = {
            "baseline_metrics": _error_metrics(baseline, experimental),
            "qrrho_primary_metrics": _error_metrics(qrrho, experimental),
            "paired_mae_improvement": bootstrap,
            "positive_model_gate_passed": positive,
            "significant_degradation": significant_degradation,
            "low_frequency_stability": {
                "passed": stable,
                "unstable_case_count": len(unstable_cases),
                "unstable_cases": unstable_cases,
            },
            "cases": per_case,
        }

    zero_tolerance = float(
        scoring["cross_model_direction_gate"]["zero_tolerance_kcal_mol"]
    )
    direction_cases = []
    direction_agreement_count = 0
    model_names = [model["name"] for model in protocol["models"]]
    for index, compound_id in enumerate(case_ids):
        signs = {
            name: (
                1
                if corrections[name][index] > zero_tolerance
                else -1 if corrections[name][index] < -zero_tolerance else 0
            )
            for name in model_names
        }
        positive = sum(value == 1 for value in signs.values())
        negative = sum(value == -1 for value in signs.values())
        agrees = max(positive, negative) >= 2
        direction_agreement_count += int(agrees)
        direction_cases.append(
            {
                "compound_id": compound_id,
                "model_signs": signs,
                "same_nonzero_sign_in_at_least_two_models": agrees,
            }
        )
    direction_passed = direction_agreement_count >= int(
        scoring["cross_model_direction_gate"][
            "minimum_cases_with_same_nonzero_sign_in_at_least_two_models"
        ]
    )
    accuracy_passed = positive_count >= 2 and significant_degradations == 0
    route_passed = bool(accuracy_passed and direction_passed and all_stable)
    return {
        "case_ids": case_ids,
        "model_results": model_results,
        "route_gates": {
            "accuracy": {
                "passed": accuracy_passed,
                "positive_model_count": positive_count,
                "significantly_degraded_model_count": significant_degradations,
            },
            "cross_model_direction": {
                "passed": direction_passed,
                "agreement_case_count": direction_agreement_count,
                "cases": direction_cases,
            },
            "low_frequency_stability": {
                "passed": all_stable,
            },
        },
        "route_passed_all_preregistered_development_gates": route_passed,
        "decision": (
            "eligible-only-for-preregistered-larger-heldout-study"
            if route_passed
            else "development-falsification-failed-no-promotion"
        ),
    }


def score_command(args: argparse.Namespace) -> None:
    protocol, fingerprint, source_manifest = load_qrrho_protocol(args.protocol)
    artifact_path = Path(args.energy_artifact).resolve()
    energy_artifact = validate_sealed_energy_artifact(
        protocol=protocol,
        fingerprint=fingerprint,
        source_manifest=source_manifest,
        artifact_path=artifact_path,
    )
    # Label access occurs only after every completeness/hash check above passes.
    labels, label_provenance = _load_labels_after_seal(
        source_manifest=source_manifest,
        protocol=protocol,
    )
    result = score_predictions(
        protocol=protocol,
        energy_artifact=energy_artifact,
        labels=labels,
    )
    output = {
        "schema_version": 1,
        "artifact_type": "route1-multi-mlip-phase-specific-qrrho-score",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "source_partition": "development",
        "energy_artifact": str(artifact_path),
        "energy_artifact_file_sha256": sha256_file(artifact_path),
        "energy_artifact_content_sha256": energy_artifact["content_sha256"],
        "label_provenance": label_provenance,
        "post_seal_scoring": protocol["post_seal_scoring"],
        "result": result,
        "claim_boundary": protocol["claim_boundary"],
        "command_provenance": command_provenance(
            __file__,
            {
                "protocol": args.protocol,
                "energy_artifact": args.energy_artifact,
                "output": args.output,
            },
            repository_root=REPOSITORY_ROOT,
        ),
    }
    seal_artifact(output)
    write_json_atomic(args.output, output)
    print(
        f"Wrote post-seal development score to {Path(args.output).resolve()}: "
        f"{result['decision']}."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default=str(SCRIPT_DIR / "multi_mlip_phase_specific_qrrho_protocol.json"),
    )
    parser.add_argument(
        "--energy-artifact",
        default=str(
            SCRIPT_DIR / "route1-multi-mlip-phase-specific-qrrho-2026-07-25.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            SCRIPT_DIR / "route1-multi-mlip-phase-specific-qrrho-score-2026-07-25.json"
        ),
    )
    parser.set_defaults(handler=score_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
