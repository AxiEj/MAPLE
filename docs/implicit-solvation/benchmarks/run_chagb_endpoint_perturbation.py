#!/usr/bin/env python3
"""Apply a label-separated CHA-GB endpoint correction to OBC-II ensembles."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core import (  # noqa: E402
    artifact_content_sha256,
    canonical_json_bytes,
    command_provenance,
    load_json,
    seal_artifact,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from run_chagb_nonpolar import (  # noqa: E402
    load_evaluation_protocol,
    resolve_executables,
)
from run_mlip_chagb_metropolis import (  # noqa: E402
    TargetEnergyEvaluator,
    _prepare_topology,
)
from run_route1_task_matrix import read_xyz_frames  # noqa: E402

from maple.function.free_energy import (  # noqa: E402
    analyze_one_sided_perturbation,
)
from maple.function.free_energy import perturbation as perturbation_module  # noqa: E402

PROTOCOL_ID = "maple-route1-obc2-to-chagb-endpoint-perturbation-v1"
KCAL_PER_HARTREE = 627.5094740631
FORBIDDEN_LABEL_KEYS = {
    "experimental_kcal_mol",
    "experimental_reference",
    "experimental_uncertainty_kcal_mol",
}


def _contains_forbidden_label_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key in FORBIDDEN_LABEL_KEYS or _contains_forbidden_label_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_label_key(item) for item in value)
    return False


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = load_json(path)
    if protocol.get("schema_version") != 1:
        raise ValueError("Only endpoint-perturbation protocol schema 1 is supported.")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected endpoint-perturbation protocol ID.")
    route = protocol.get("route_contract", {})
    if (
        route.get("gas_phase_mm_energy") is not False
        or route.get("hydration_label_residual") is not False
        or route.get("mlip_retraining") is not False
        or float(route.get("gas_correction_kcal_mol", math.nan)) != 0.0
    ):
        raise ValueError("Endpoint perturbation violates the Route 1 boundary.")
    boundary = protocol.get("execution_boundary", {})
    if boundary.get("energy_phase_reads_experimental_labels") is not False:
        raise ValueError("The energy phase must remain label-free.")
    if boundary.get("promotion_allowed") is not False:
        raise ValueError("The frozen diagnostic cannot be promotable.")
    reuse = protocol.get("sampling_reuse", {})
    if reuse.get("reference_equilibrium_claim") is not False:
        raise ValueError("The short source trajectories cannot claim equilibrium.")
    if reuse.get("target_ensemble_sampled") is not False:
        raise ValueError("The energy-only target ensemble was not sampled.")
    if _contains_forbidden_label_key(protocol):
        raise ValueError("The endpoint-perturbation protocol contains a label.")
    return protocol, sha256_bytes(canonical_json_bytes(protocol))


def _source_path(
    protocol_path: Path,
    protocol: dict[str, Any],
    key: str,
) -> Path:
    evidence = protocol["source_evidence"]
    path = protocol_path.parent / evidence[key]
    observed = sha256_file(path)
    expected = evidence[f"{key}_sha256"]
    if observed != expected:
        raise ValueError(f"Frozen source hash changed for {key}: {path}.")
    return path


def _load_energy_sources(
    protocol_path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    paths = {
        key: _source_path(protocol_path, protocol, key)
        for key in (
            "low_endpoint_record_manifest",
            "low_endpoint_mbar_protocol",
            "low_endpoint_mbar_artifact",
            "parent_sampling_protocol",
            "high_endpoint_protocol",
            "high_endpoint_charge_manifest",
            "prepared_high_endpoint_helper",
        )
    }
    record_manifest = load_json(paths["low_endpoint_record_manifest"])
    low_mbar = load_json(paths["low_endpoint_mbar_artifact"])
    high_manifest = load_json(paths["high_endpoint_charge_manifest"])
    if low_mbar.get("content_sha256") != artifact_content_sha256(low_mbar):
        raise ValueError("Low-endpoint MBAR artifact has an invalid self-hash.")
    expected_content = protocol["source_evidence"][
        "low_endpoint_mbar_artifact_content_sha256"
    ]
    if low_mbar["content_sha256"] != expected_content:
        raise ValueError("Low-endpoint MBAR content hash changed.")
    for name, payload in (
        ("record manifest", record_manifest),
        ("low-endpoint MBAR artifact", low_mbar),
        ("high-endpoint charge manifest", high_manifest),
    ):
        if _contains_forbidden_label_key(payload):
            raise ValueError(f"{name} contains an experimental label.")
    if int(record_manifest.get("record_count", -1)) != int(
        protocol["sampling_reuse"]["record_count"]
    ):
        raise ValueError("Low-endpoint source-record count changed.")
    if int(high_manifest.get("case_count", -1)) <= 0:
        raise ValueError("High-endpoint charge manifest is empty.")
    high_protocol, _ = load_evaluation_protocol(paths["high_endpoint_protocol"])
    return {
        "paths": paths,
        "record_manifest": record_manifest,
        "low_mbar": low_mbar,
        "high_manifest": high_manifest,
        "high_protocol": high_protocol,
    }


def _load_source_records(
    manifest: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in manifest["records"]:
        path = REPOSITORY_ROOT / entry["relative_path"]
        if sha256_file(path) != entry["file_sha256"]:
            raise ValueError(f"Source TI record file hash changed: {path}.")
        record = load_json(path)
        if record.get("content_sha256") != artifact_content_sha256(record):
            raise ValueError(f"Source TI record self-hash changed: {path}.")
        if record["content_sha256"] != entry["content_sha256"]:
            raise ValueError(f"Source TI record content hash changed: {path}.")
        if _contains_forbidden_label_key(record):
            raise ValueError(f"Source TI record contains a label: {path}.")
        identity = (record["model"], record["compound_id"])
        if identity in records:
            raise ValueError(f"Duplicate source TI identity: {identity}.")
        records[identity] = {
            "entry": entry,
            "path": path,
            "record": record,
        }
    return records


def _identity_map(
    records: list[dict[str, Any]],
    *,
    label: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        identity = (record["model"], record["compound_id"])
        if identity in output:
            raise ValueError(f"Duplicate {label} identity: {identity}.")
        output[identity] = record
    return output


def _charge_records(
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    output = {record["compound_id"]: record for record in manifest["records"]}
    if len(output) != len(manifest["records"]):
        raise ValueError("High-endpoint charge manifest has duplicate case IDs.")
    return output


def _lambda_one_window(chain: dict[str, Any]) -> dict[str, Any]:
    windows = [
        window for window in chain["windows"] if float(window["lambda"]) == 1.0
    ]
    if len(windows) != 1:
        raise ValueError("Each source chain must contain exactly one lambda=1 window.")
    return windows[0]


def _production_positions(window: dict[str, Any]) -> list[np.ndarray]:
    trajectory = REPOSITORY_ROOT / window["trajectory"]
    if sha256_file(trajectory) != window["trajectory_sha256"]:
        raise ValueError(f"Source trajectory hash changed: {trajectory}.")
    frames = read_xyz_frames(trajectory)
    discarded = int(window["discarded_frame_count"])
    positions = [
        np.asarray(frame["positions_angstrom"], dtype=np.float64)
        for frame in frames[discarded:]
    ]
    expected = int(window["production_sample_count"])
    if len(positions) != expected or len(window["samples"]) != expected:
        raise ValueError("Source trajectory/sample count changed.")
    return positions


def _evaluate_record(
    *,
    source: dict[str, Any],
    low_mbar: dict[str, Any],
    evaluator: TargetEnergyEvaluator,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    source_record = source["record"]
    calls_before = evaluator.calls
    seconds_before = evaluator.seconds
    chain_results: list[dict[str, Any]] = []
    differences: list[np.ndarray] = []
    for chain in source_record["chains"]:
        window = _lambda_one_window(chain)
        positions = _production_positions(window)
        low_values: list[float] = []
        high_values: list[float] = []
        component_values: list[dict[str, float]] = []
        delta_values: list[float] = []
        for frame, sample in zip(positions, window["samples"], strict=True):
            low = float(sample["solvent_energy_hartree"]) * KCAL_PER_HARTREE
            high, components = evaluator.evaluate(frame)
            delta = float(high - low)
            if not all(math.isfinite(value) for value in (low, high, delta)):
                raise ValueError("Endpoint evaluation returned a non-finite energy.")
            low_values.append(low)
            high_values.append(high)
            component_values.append(
                {key: float(value) for key, value in components.items()}
            )
            delta_values.append(delta)
        differences.append(np.asarray(delta_values, dtype=np.float64))
        chain_results.append(
            {
                "chain_id": chain["chain_id"],
                "source_trajectory": window["trajectory"],
                "source_trajectory_sha256": window["trajectory_sha256"],
                "sample_count": len(delta_values),
                "low_obc2_ace_kcal_mol": low_values,
                "high_chagb_pbsa_kcal_mol": high_values,
                "high_components_kcal_mol": component_values,
                "high_minus_low_kcal_mol": delta_values,
            }
        )

    analysis_options = dict(protocol["analysis"])
    analysis = analyze_one_sided_perturbation(
        differences,
        temperature_kelvin=analysis_options.pop("temperature_kelvin"),
        reference_equilibrium_claim=protocol["sampling_reuse"][
            "reference_equilibrium_claim"
        ],
        **analysis_options,
    )
    correction = float(analysis["combined"]["delta_g_kcal_mol"])
    low_delta_g = float(low_mbar["mbar"]["endpoint_delta_g_kcal_mol"])
    return {
        "model": source_record["model"],
        "compound_id": source_record["compound_id"],
        "name": source_record["name"],
        "flexibility_bin": source_record["flexibility_bin"],
        "source_ti_record": _display_path(source["path"]),
        "source_ti_record_file_sha256": sha256_file(source["path"]),
        "source_ti_record_content_sha256": source_record["content_sha256"],
        "source_low_mbar_delta_g_kcal_mol": low_delta_g,
        "endpoint_correction_kcal_mol": correction,
        "corrected_high_delta_g_kcal_mol": low_delta_g + correction,
        "chains": chain_results,
        "analysis": analysis,
        "high_endpoint_evaluation_count": evaluator.calls - calls_before,
        "high_endpoint_evaluation_seconds": evaluator.seconds - seconds_before,
        "all_numerical_checks_pass": analysis["gates"][
            "numerical_gates_passed"
        ],
        "all_scientific_checks_pass": analysis["gates"][
            "scientific_gates_passed"
        ],
        "promotion_allowed": False,
    }


def run_energy(
    *,
    protocol_path: Path,
    output_path: Path,
    work_dir: Path,
    amber_bin: Path | None,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    protocol, fingerprint = load_protocol(protocol_path)
    sources = _load_energy_sources(protocol_path, protocol)
    source_records = _load_source_records(sources["record_manifest"])
    low_mbar_by_id = _identity_map(
        sources["low_mbar"]["records"],
        label="low-endpoint MBAR",
    )
    charge_by_id = _charge_records(sources["high_manifest"])
    expected_identities = set(source_records)
    if set(low_mbar_by_id) != expected_identities:
        raise ValueError("Low-endpoint MBAR/source-record matrices differ.")

    executables = resolve_executables(
        sources["high_protocol"],
        amber_bin,
    )
    evaluators: dict[str, TargetEnergyEvaluator] = {}
    topology_records: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for identity in sorted(source_records):
        model, compound_id = identity
        source = source_records[identity]
        source_record = source["record"]
        charge_record = charge_by_id[compound_id]
        if compound_id not in evaluators:
            case_dir = work_dir / compound_id
            source_mol2 = REPOSITORY_ROOT / source_record["mol2"]
            topology = _prepare_topology(
                case_dir=case_dir,
                source_mol2=source_mol2,
                source_hash=source_record["mol2_sha256"],
                charges=charge_record["am1bcc_charges_e"],
                chagb_protocol=sources["high_protocol"],
                executables=executables,
            )
            evaluators[compound_id] = TargetEnergyEvaluator(
                case_dir=case_dir,
                topology=topology,
                protocol=sources["high_protocol"],
                executables=executables,
            )
            topology_records[compound_id] = {
                "path": _display_path(topology),
                "sha256": sha256_file(topology),
                "source_mol2": _display_path(source_mol2),
                "source_mol2_sha256": sha256_file(source_mol2),
                "charge_record_sha256": sha256_bytes(
                    canonical_json_bytes(charge_record)
                ),
            }
        records.append(
            _evaluate_record(
                source=source,
                low_mbar=low_mbar_by_id[(model, compound_id)],
                evaluator=evaluators[compound_id],
                protocol=protocol,
            )
        )

    expected_evaluations = int(
        protocol["sampling_reuse"]["expected_high_endpoint_evaluations_total"]
    )
    observed_evaluations = sum(
        record["high_endpoint_evaluation_count"] for record in records
    )
    if observed_evaluations != expected_evaluations:
        raise ValueError(
            f"Expected {expected_evaluations} high-endpoint evaluations, "
            f"observed {observed_evaluations}."
        )
    elapsed = time.perf_counter() - started
    corrections = np.asarray(
        [record["endpoint_correction_kcal_mol"] for record in records],
        dtype=np.float64,
    )
    ess_fractions = np.asarray(
        [
            record["analysis"]["combined"]["effective_sample_fraction"]
            for record in records
        ],
        dtype=np.float64,
    )
    maximum_weights = np.asarray(
        [
            record["analysis"]["combined"]["maximum_normalized_weight"]
            for record in records
        ],
        dtype=np.float64,
    )
    artifact = {
        "schema_version": 1,
        "artifact_type": "route1-obc2-to-chagb-endpoint-perturbation",
        "protocol_id": PROTOCOL_ID,
        "protocol_fingerprint": fingerprint,
        "source_partition": protocol["source_partition"],
        "route_contract": protocol["route_contract"],
        "records": records,
        "aggregate_diagnostics": {
            "record_count": len(records),
            "high_endpoint_evaluation_count": observed_evaluations,
            "high_endpoint_evaluation_seconds": float(
                sum(
                    record["high_endpoint_evaluation_seconds"]
                    for record in records
                )
            ),
            "total_wall_seconds": elapsed,
            "numerical_gate_pass_count": sum(
                record["all_numerical_checks_pass"] for record in records
            ),
            "scientific_gate_pass_count": sum(
                record["all_scientific_checks_pass"] for record in records
            ),
            "minimum_effective_sample_fraction": float(ess_fractions.min()),
            "maximum_normalized_weight": float(maximum_weights.max()),
            "endpoint_correction_kcal_mol": {
                "minimum": float(corrections.min()),
                "median": float(np.median(corrections)),
                "maximum": float(corrections.max()),
            },
        },
        "source_evidence": {
            key: {
                "path": _display_path(path),
                "file_sha256": sha256_file(path),
            }
            for key, path in sources["paths"].items()
        },
        "topology_records": topology_records,
        "provider_executables": executables,
        "implementation_provenance": {
            "analysis_module": _display_path(Path(perturbation_module.__file__)),
            "analysis_module_sha256": sha256_file(perturbation_module.__file__),
            "pymbar_version": importlib.metadata.version("pymbar"),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
        "label_boundary": {
            "experimental_labels_read": False,
            "scoring_performed": False,
            "case_or_parameter_selection_used_labels": False,
        },
        "claim_boundary": {
            "low_endpoint_equilibrium_proven": False,
            "target_ensemble_sampled": False,
            "bidirectional_endpoint_validation_established": False,
            "chemical_accuracy_established": False,
            "product_solvation_free_energy_established": False,
        },
        "decision": {
            "result": "diagnostic_only_not_promotable",
            "promotion_allowed": False,
            "reason": protocol["decision_policy"]["reason"],
        },
        "promotion_allowed": False,
        "command_provenance": command_provenance(
            __file__,
            arguments,
            repository_root=REPOSITORY_ROOT,
        ),
    }
    sealed = seal_artifact(artifact)
    if _contains_forbidden_label_key(sealed):
        raise ValueError("Energy artifact contains an experimental label.")
    write_json_atomic(output_path, sealed)
    return sealed


def _metrics(predicted: np.ndarray, experimental: np.ndarray) -> dict[str, Any]:
    error = predicted - experimental
    return {
        "n": int(len(error)),
        "mean_signed_error_kcal_mol": float(error.mean()),
        "mae_kcal_mol": float(np.abs(error).mean()),
        "rmse_kcal_mol": float(np.sqrt(np.square(error).mean())),
        "maximum_absolute_error_kcal_mol": float(np.abs(error).max()),
    }


def run_score(
    *,
    protocol_path: Path,
    energy_path: Path,
    label_path: Path,
    output_path: Path,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    _, fingerprint = load_protocol(protocol_path)
    energy = load_json(energy_path)
    if energy.get("content_sha256") != artifact_content_sha256(energy):
        raise ValueError("Energy artifact has an invalid self-hash.")
    if energy.get("protocol_fingerprint") != fingerprint:
        raise ValueError("Energy artifact protocol fingerprint changed.")
    labels = load_json(label_path)
    if labels.get("content_sha256") != artifact_content_sha256(labels):
        raise ValueError("Development label artifact has an invalid self-hash.")
    label_by_id = _identity_map(labels["case_results"], label="development label")
    if set(label_by_id) != {
        (record["model"], record["compound_id"]) for record in energy["records"]
    }:
        raise ValueError("Energy and label matrices differ.")

    score_records: list[dict[str, Any]] = []
    for record in energy["records"]:
        identity = (record["model"], record["compound_id"])
        label = label_by_id[identity]
        experimental = float(label["experimental_kcal_mol"])
        predictions = {
            "fixed_geometry_obc2_ace": float(
                label["fixed_geometry_obc2_ace_kcal_mol"]
            ),
            "low_obc2_ace_mbar": float(
                record["source_low_mbar_delta_g_kcal_mol"]
            ),
            "high_chagb_pbsa_endpoint_corrected": float(
                record["corrected_high_delta_g_kcal_mol"]
            ),
        }
        score_records.append(
            {
                "model": record["model"],
                "compound_id": record["compound_id"],
                "name": record["name"],
                "experimental_kcal_mol": experimental,
                "predictions_kcal_mol": predictions,
                "signed_errors_kcal_mol": {
                    name: value - experimental
                    for name, value in predictions.items()
                },
                "endpoint_correction_kcal_mol": record[
                    "endpoint_correction_kcal_mol"
                ],
                "energy_numerical_gates_passed": record[
                    "all_numerical_checks_pass"
                ],
                "energy_scientific_gates_passed": record[
                    "all_scientific_checks_pass"
                ],
            }
        )

    model_summaries: dict[str, Any] = {}
    for model in sorted({record["model"] for record in score_records}):
        rows = [record for record in score_records if record["model"] == model]
        experimental = np.asarray(
            [record["experimental_kcal_mol"] for record in rows],
            dtype=np.float64,
        )
        methods = {}
        for name in rows[0]["predictions_kcal_mol"]:
            predicted = np.asarray(
                [record["predictions_kcal_mol"][name] for record in rows],
                dtype=np.float64,
            )
            methods[name] = _metrics(predicted, experimental)
        corrected_gain_vs_low = (
            methods["low_obc2_ace_mbar"]["mae_kcal_mol"]
            - methods["high_chagb_pbsa_endpoint_corrected"]["mae_kcal_mol"]
        )
        corrected_gain_vs_fixed = (
            methods["fixed_geometry_obc2_ace"]["mae_kcal_mol"]
            - methods["high_chagb_pbsa_endpoint_corrected"]["mae_kcal_mol"]
        )
        model_summaries[model] = {
            "case_count": len(rows),
            "methods": methods,
            "corrected_mae_gain_vs_low_mbar_kcal_mol": corrected_gain_vs_low,
            "corrected_mae_gain_vs_fixed_geometry_kcal_mol": (
                corrected_gain_vs_fixed
            ),
            "numerical_gate_pass_count": sum(
                record["energy_numerical_gates_passed"] for record in rows
            ),
            "scientific_gate_pass_count": sum(
                record["energy_scientific_gates_passed"] for record in rows
            ),
            "promotion_allowed": False,
        }

    artifact = {
        "schema_version": 1,
        "artifact_type": (
            "route1-obc2-to-chagb-endpoint-perturbation-development-score"
        ),
        "protocol_id": PROTOCOL_ID,
        "protocol_fingerprint": fingerprint,
        "source_partition": "development",
        "energy_artifact": {
            "path": _display_path(energy_path),
            "file_sha256": sha256_file(energy_path),
            "content_sha256": energy["content_sha256"],
        },
        "label_artifact": {
            "path": _display_path(label_path),
            "file_sha256": sha256_file(label_path),
            "content_sha256": labels["content_sha256"],
        },
        "records": score_records,
        "model_summaries": model_summaries,
        "label_boundary": {
            "energy_artifact_sealed_before_scoring": True,
            "labels_read_only_by_score_phase": True,
            "labels_changed_neither_cases_parameters_nor_energies": True,
            "confirmation_partition": False,
        },
        "claim_boundary": {
            "chemical_accuracy_established": False,
            "equilibrated_sampling_proven": False,
            "target_ensemble_coverage_proven": False,
            "product_solvation_free_energy_established": False,
        },
        "decision": {
            "result": "development_score_not_promotable",
            "promotion_allowed": False,
            "reason": (
                "The label-free parent trajectories lack equilibrium and "
                "conformer-mixing evidence, so accuracy changes are diagnostic."
            ),
        },
        "promotion_allowed": False,
        "command_provenance": command_provenance(
            __file__,
            arguments,
            repository_root=REPOSITORY_ROOT,
        ),
    }
    sealed = seal_artifact(artifact)
    write_json_atomic(output_path, sealed)
    return sealed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default=str(SCRIPT_DIR / "chagb_endpoint_perturbation_protocol.json"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    energy = subparsers.add_parser("energy")
    energy.add_argument(
        "--output",
        default=str(
            SCRIPT_DIR
            / "route1-multi-mlip-chagb-endpoint-perturbation-2026-07-25.json"
        ),
    )
    energy.add_argument(
        "--work-dir",
        default=str(
            REPOSITORY_ROOT
            / ".omx/benchmarks/route1-chagb-endpoint-perturbation"
        ),
    )
    energy.add_argument("--amber-bin")

    score = subparsers.add_parser("score")
    score.add_argument(
        "--energy",
        default=str(
            SCRIPT_DIR
            / "route1-multi-mlip-chagb-endpoint-perturbation-2026-07-25.json"
        ),
    )
    score.add_argument(
        "--labels",
        default=str(SCRIPT_DIR / "route1-multi-mlip-obc2-ti-2026-07-25.json"),
    )
    score.add_argument(
        "--output",
        default=str(
            SCRIPT_DIR
            / "route1-multi-mlip-chagb-endpoint-perturbation-score-2026-07-25.json"
        ),
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    arguments = {
        key: value
        for key, value in vars(args).items()
    }
    protocol_path = Path(args.protocol).resolve()
    if args.command == "energy":
        result = run_energy(
            protocol_path=protocol_path,
            output_path=Path(args.output).resolve(),
            work_dir=Path(args.work_dir).resolve(),
            amber_bin=(Path(args.amber_bin).resolve() if args.amber_bin else None),
            arguments=arguments,
        )
    else:
        result = run_score(
            protocol_path=protocol_path,
            energy_path=Path(args.energy).resolve(),
            label_path=Path(args.labels).resolve(),
            output_path=Path(args.output).resolve(),
            arguments=arguments,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
