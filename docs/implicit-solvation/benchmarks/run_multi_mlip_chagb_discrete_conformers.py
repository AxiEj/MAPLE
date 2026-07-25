#!/usr/bin/env python3
"""Project a frozen multi-MLIP conformer matrix onto a CHA-GB endpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
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
from run_multi_mlip_discrete_conformers import (  # noqa: E402
    _error_metrics,
    _paired_score,
)

from maple.function.free_energy import (  # noqa: E402
    analyze_discrete_conformer_ensemble,
)

PROTOCOL_ID = "maple-route1-multi-mlip-chagb-discrete-conformer-v1"
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


def _display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _self_hashed(payload: dict[str, Any], *, name: str) -> None:
    if payload.get("content_sha256") != artifact_content_sha256(payload):
        raise ValueError(f"{name} has an invalid content self-hash.")


def load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = load_json(path)
    if protocol.get("schema_version") != 1:
        raise ValueError("Only CHA-GB conformer protocol schema 1 is supported.")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected CHA-GB conformer protocol ID.")
    route = protocol.get("route_contract", {})
    if (
        route.get("gas_phase_mm_energy") is not False
        or route.get("hydration_label_residual") is not False
        or route.get("mlip_retraining") is not False
        or route.get("fixed_am1bcc_charges") is not True
        or route.get("gas_mlip_energies_reused_without_modification") is not True
    ):
        raise ValueError("Protocol violates the Route 1 additive boundary.")
    boundary = protocol.get("execution_boundary", {})
    if (
        boundary.get("energy_phase_reads_experimental_labels") is not False
        or boundary.get("labels_change_neither_cases_parameters_nor_energies")
        is not True
        or boundary.get("promotion_allowed") is not False
    ):
        raise ValueError("Protocol violates the label or promotion boundary.")
    matrix = protocol.get("matrix", {})
    if (
        int(matrix.get("model_count", -1)) < 2
        or int(matrix.get("case_count", -1)) < 1
        or int(matrix.get("unique_state_count", -1)) < 1
        or int(matrix.get("high_endpoint_repeat_count", -1)) < 2
        or int(matrix.get("expected_high_endpoint_evaluation_count", -1))
        != int(matrix["unique_state_count"])
        * int(matrix["high_endpoint_repeat_count"])
    ):
        raise ValueError("Protocol matrix counts are inconsistent.")
    estimator = protocol.get("estimator", {})
    if (
        float(estimator.get("temperature_kelvin", math.nan)) <= 0.0
        or int(estimator.get("minimum_state_count", -1)) < 2
        or float(estimator.get("minimum_effective_conformer_count", math.nan))
        <= 0.0
        or not 0.0
        < float(estimator.get("maximum_dominant_weight", math.nan))
        <= 1.0
        or not 0.0
        < float(estimator.get("minimum_distribution_overlap", math.nan))
        <= 1.0
    ):
        raise ValueError("Protocol estimator settings are invalid.")
    if _contains_forbidden_label_key(protocol):
        raise ValueError("The label-free protocol contains an experimental label.")
    return protocol, sha256_bytes(canonical_json_bytes(protocol))


def _source_path(
    protocol_path: Path,
    protocol: dict[str, Any],
    key: str,
) -> Path:
    source = protocol["source_evidence"]
    path = (protocol_path.parent / source[key]).resolve()
    if sha256_file(path) != source[f"{key}_sha256"]:
        raise ValueError(f"Frozen source hash changed for {key}: {path}.")
    return path


def _load_energy_sources(
    protocol_path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "discrete_protocol",
        "state_manifest",
        "multi_mlip_energy_artifact",
        "high_endpoint_protocol",
        "high_endpoint_charge_manifest",
        "high_endpoint_helper",
        "discrete_runner",
        "discrete_analysis_module",
    )
    paths = {
        key: _source_path(protocol_path, protocol, key)
        for key in keys
    }
    manifest = load_json(paths["state_manifest"])
    source_energy = load_json(paths["multi_mlip_energy_artifact"])
    charges = load_json(paths["high_endpoint_charge_manifest"])
    _self_hashed(manifest, name="State manifest")
    _self_hashed(source_energy, name="Multi-MLIP energy artifact")
    expected = protocol["source_evidence"]
    if manifest["content_sha256"] != expected["state_manifest_content_sha256"]:
        raise ValueError("State-manifest content hash changed.")
    if (
        source_energy["content_sha256"]
        != expected["multi_mlip_energy_artifact_content_sha256"]
    ):
        raise ValueError("Multi-MLIP energy content hash changed.")
    for name, payload in (
        ("state manifest", manifest),
        ("multi-MLIP energy artifact", source_energy),
        ("charge manifest", charges),
    ):
        if _contains_forbidden_label_key(payload):
            raise ValueError(f"{name} contains an experimental label.")
    matrix = protocol["matrix"]
    if (
        len(manifest["cases"]) != int(matrix["case_count"])
        or manifest["selection"]["observed_selected_state_count"]
        != int(matrix["unique_state_count"])
        or len(source_energy["records"])
        != int(matrix["model_case_record_count"])
    ):
        raise ValueError("Frozen source matrix changed.")
    high_protocol, _ = load_evaluation_protocol(
        paths["high_endpoint_protocol"]
    )
    return {
        "paths": paths,
        "manifest": manifest,
        "source_energy": source_energy,
        "charges": charges,
        "high_protocol": high_protocol,
    }


def _identity_map(
    records: list[dict[str, Any]],
    *,
    keys: tuple[str, ...],
    name: str,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    output: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        identity = tuple(record[key] for key in keys)
        if identity in output:
            raise ValueError(f"Duplicate {name} identity: {identity}.")
        output[identity] = record
    return output


def _load_positions(case: dict[str, Any]) -> np.ndarray:
    path = REPOSITORY_ROOT / case["state_file"]
    if sha256_file(path) != case["state_file_sha256"]:
        raise ValueError(f"State hash changed for {case['compound_id']}.")
    positions = np.load(path, allow_pickle=False)
    if (
        list(positions.shape) != case["state_shape"]
        or positions.dtype != np.float64
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError(f"Invalid state array for {case['compound_id']}.")
    return positions


def _array_hash(values: Any) -> str:
    return sha256_bytes(
        canonical_json_bytes(np.asarray(values, dtype=np.float64).tolist())
    )


def _analysis(
    *,
    gas_energy_hartree: list[float],
    solvent_kcal_mol: list[float],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    gas = np.asarray(gas_energy_hartree, dtype=np.float64)
    if gas.shape != np.asarray(solvent_kcal_mol).shape:
        raise ValueError("Gas and solvent state arrays differ.")
    relative = (gas - float(np.min(gas))) * KCAL_PER_HARTREE
    estimator = protocol["estimator"]
    return analyze_discrete_conformer_ensemble(
        gas_energy_kcal_mol=relative,
        solvent_correction_kcal_mol=solvent_kcal_mol,
        temperature_kelvin=estimator["temperature_kelvin"],
        minimum_effective_conformer_count=estimator[
            "minimum_effective_conformer_count"
        ],
        maximum_dominant_weight=estimator["maximum_dominant_weight"],
        minimum_distribution_overlap=estimator[
            "minimum_distribution_overlap"
        ],
    )


def _compact_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "delta_g_discrete_kcal_mol",
        "gas_relative_energy_kcal_mol",
        "gas_weights",
        "solution_weights",
        "gas_effective_conformer_count",
        "solution_effective_conformer_count",
        "gas_maximum_weight",
        "solution_maximum_weight",
        "gas_dominant_state_index",
        "solution_dominant_state_index",
        "distribution_overlap",
        "total_variation_distance",
        "bhattacharyya_coefficient",
        "identities",
        "gates",
        "route_contract",
        "claim_boundary",
    )
    return {key: analysis[key] for key in keys}


def _evaluate_case(
    *,
    case: dict[str, Any],
    charge: dict[str, Any],
    evaluator: TargetEnergyEvaluator,
    repeat_count: int,
) -> dict[str, Any]:
    positions = _load_positions(case)
    energies_by_repeat: list[list[float]] = []
    components_by_repeat: list[list[dict[str, float]]] = []
    seconds_by_repeat: list[float] = []
    for _repeat in range(repeat_count):
        calls_before = evaluator.calls
        seconds_before = evaluator.seconds
        energies: list[float] = []
        components: list[dict[str, float]] = []
        for state in positions:
            # This protocol measures two independent provider evaluations for
            # every frozen state. Disable the helper's adjacent-state cache so
            # single-state cases and exact duplicates are genuinely repeated.
            evaluator._cache_key = None
            evaluator._cache_value = None
            energy, values = evaluator.evaluate(state)
            if not math.isfinite(energy):
                raise ValueError("CHA-GB endpoint returned non-finite energy.")
            energies.append(float(energy))
            components.append(
                {key: float(value) for key, value in values.items()}
            )
        if evaluator.calls - calls_before != len(positions):
            raise ValueError("CHA-GB evaluation count changed.")
        energies_by_repeat.append(energies)
        components_by_repeat.append(components)
        seconds_by_repeat.append(evaluator.seconds - seconds_before)
    primary = np.asarray(energies_by_repeat[0], dtype=np.float64)
    maximum_repeat_difference = max(
        float(
            np.max(
                np.abs(
                    np.asarray(values, dtype=np.float64) - primary
                )
            )
        )
        for values in energies_by_repeat[1:]
    )
    reference_index = int(case["reference_geometry_union"]["reference_index"])
    return {
        "compound_id": case["compound_id"],
        "name": case["name"],
        "flexibility_bin": case["flexibility_bin"],
        "state_count": int(case["state_count"]),
        "state_file": case["state_file"],
        "state_file_sha256": case["state_file_sha256"],
        "source_mol2": case["source"]["mol2"],
        "source_mol2_sha256": case["source"]["mol2_sha256"],
        "charge_record_sha256": sha256_bytes(canonical_json_bytes(charge)),
        "reference_state_index": reference_index,
        "reference_endpoint_kcal_mol": float(primary[reference_index]),
        "high_solvent_kcal_mol_by_repeat": energies_by_repeat,
        "high_components_kcal_mol_by_repeat": components_by_repeat,
        "seconds_by_repeat": seconds_by_repeat,
        "maximum_repeat_difference_kcal_mol": maximum_repeat_difference,
        "primary_high_solvent_sha256": _array_hash(primary),
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
    manifest = sources["manifest"]
    source_energy = sources["source_energy"]
    charge_by_id = {
        record["compound_id"]: record
        for record in sources["charges"]["records"]
    }
    if len(charge_by_id) != len(sources["charges"]["records"]):
        raise ValueError("Charge manifest contains duplicate compound IDs.")
    source_by_id = _identity_map(
        source_energy["records"],
        keys=("model", "compound_id"),
        name="source model/case",
    )
    model_names = list(protocol["matrix"]["models"])
    expected_identities = {
        (model, case["compound_id"])
        for model in model_names
        for case in manifest["cases"]
    }
    if set(source_by_id) != expected_identities:
        raise ValueError("Source model/case matrix changed.")

    executables = resolve_executables(sources["high_protocol"], amber_bin)
    repeat_count = int(protocol["matrix"]["high_endpoint_repeat_count"])
    case_records: list[dict[str, Any]] = []
    evaluators: dict[str, TargetEnergyEvaluator] = {}
    started = time.perf_counter()
    for case in manifest["cases"]:
        compound_id = case["compound_id"]
        charge = charge_by_id[compound_id]
        source_mol2 = REPOSITORY_ROOT / case["source"]["mol2"]
        if sha256_file(source_mol2) != case["source"]["mol2_sha256"]:
            raise ValueError(f"Source MOL2 hash changed for {compound_id}.")
        if charge["source_mol2_sha256"] != case["source"]["mol2_sha256"]:
            raise ValueError(f"Charge/state MOL2 mismatch for {compound_id}.")
        case_dir = work_dir / compound_id
        topology = _prepare_topology(
            case_dir=case_dir,
            source_mol2=source_mol2,
            source_hash=case["source"]["mol2_sha256"],
            charges=charge["am1bcc_charges_e"],
            chagb_protocol=sources["high_protocol"],
            executables=executables,
        )
        evaluator = TargetEnergyEvaluator(
            case_dir=case_dir,
            topology=topology,
            protocol=sources["high_protocol"],
            executables=executables,
        )
        evaluators[compound_id] = evaluator
        record = _evaluate_case(
            case=case,
            charge=charge,
            evaluator=evaluator,
            repeat_count=repeat_count,
        )
        record["topology"] = _display_path(topology)
        record["topology_sha256"] = sha256_file(topology)
        case_records.append(record)

    observed_evaluations = sum(
        evaluator.calls for evaluator in evaluators.values()
    )
    expected_evaluations = int(
        protocol["matrix"]["expected_high_endpoint_evaluation_count"]
    )
    if observed_evaluations != expected_evaluations:
        raise ValueError(
            f"Expected {expected_evaluations} endpoint evaluations, "
            f"observed {observed_evaluations}."
        )
    case_by_id = {record["compound_id"]: record for record in case_records}
    model_records: list[dict[str, Any]] = []
    for model in model_names:
        for case in manifest["cases"]:
            compound_id = case["compound_id"]
            source = source_by_id[(model, compound_id)]
            endpoint = case_by_id[compound_id]
            gas = source["energy_hartree_by_repeat"][0]
            analyses = [
                _analysis(
                    gas_energy_hartree=gas,
                    solvent_kcal_mol=solvent,
                    protocol=protocol,
                )
                for solvent in endpoint["high_solvent_kcal_mol_by_repeat"]
            ]
            delta_g = [
                float(analysis["delta_g_discrete_kcal_mol"])
                for analysis in analyses
            ]
            model_records.append(
                {
                    "model": model,
                    "compound_id": compound_id,
                    "name": case["name"],
                    "flexibility_bin": case["flexibility_bin"],
                    "state_count": int(case["state_count"]),
                    "source_gas_energy_sha256": _array_hash(gas),
                    "source_low_discrete_kcal_mol": float(
                        source["analysis"]["delta_g_discrete_kcal_mol"]
                    ),
                    "source_low_weight_diagnostic_passed": bool(
                        source["analysis"]["gates"][
                            "ensemble_diagnostic_passed"
                        ]
                    ),
                    "high_discrete_kcal_mol_by_repeat": delta_g,
                    "maximum_repeat_delta_g_difference_kcal_mol": max(
                        abs(value - delta_g[0]) for value in delta_g[1:]
                    ),
                    "analysis": _compact_analysis(analyses[0]),
                    "promotion_allowed": False,
                }
            )

    repeat_limit = float(
        protocol["estimator"][
            "maximum_high_endpoint_repeat_difference_kcal_mol"
        ]
    )
    delta_limit = float(
        protocol["estimator"][
            "maximum_repeat_delta_g_difference_kcal_mol"
        ]
    )
    artifact = {
        "schema_version": 1,
        "artifact_type": (
            "route1-multi-mlip-chagb-discrete-conformer-energy"
        ),
        "protocol_id": PROTOCOL_ID,
        "protocol_fingerprint": fingerprint,
        "source_partition": protocol["source_partition"],
        "route_contract": protocol["route_contract"],
        "matrix": protocol["matrix"],
        "source_evidence": {
            key: {
                "path": _display_path(path),
                "file_sha256": sha256_file(path),
            }
            for key, path in sources["paths"].items()
        },
        "provider_executables": executables,
        "case_endpoints": case_records,
        "model_records": model_records,
        "aggregate_diagnostics": {
            "case_count": len(case_records),
            "model_record_count": len(model_records),
            "unique_state_count": sum(
                record["state_count"] for record in case_records
            ),
            "high_endpoint_evaluation_count": observed_evaluations,
            "high_endpoint_seconds_by_repeat": [
                float(
                    sum(
                        record["seconds_by_repeat"][repeat]
                        for record in case_records
                    )
                )
                for repeat in range(repeat_count)
            ],
            "total_wall_seconds": time.perf_counter() - started,
            "maximum_endpoint_repeat_difference_kcal_mol": max(
                record["maximum_repeat_difference_kcal_mol"]
                for record in case_records
            ),
            "maximum_delta_g_repeat_difference_kcal_mol": max(
                record["maximum_repeat_delta_g_difference_kcal_mol"]
                for record in model_records
            ),
            "high_weight_diagnostic_pass_count_by_model": {
                model: sum(
                    record["analysis"]["gates"][
                        "ensemble_diagnostic_passed"
                    ]
                    for record in model_records
                    if record["model"] == model
                )
                for model in model_names
            },
        },
        "engineering_gates": {
            "complete_case_matrix": (
                len(case_records) == int(protocol["matrix"]["case_count"])
            ),
            "complete_model_case_matrix": (
                len(model_records)
                == int(protocol["matrix"]["model_case_record_count"])
            ),
            "complete_high_endpoint_evaluation_count": (
                observed_evaluations == expected_evaluations
            ),
            "endpoint_repeat_difference_within_limit": all(
                record["maximum_repeat_difference_kcal_mol"] <= repeat_limit
                for record in case_records
            ),
            "delta_g_repeat_difference_within_limit": all(
                record["maximum_repeat_delta_g_difference_kcal_mol"]
                <= delta_limit
                for record in model_records
            ),
            "endpoint_repeat_limit_kcal_mol": repeat_limit,
            "delta_g_repeat_limit_kcal_mol": delta_limit,
        },
        "label_boundary": {
            "experimental_labels_read": False,
            "scoring_performed": False,
            "case_or_parameter_selection_used_labels": False,
        },
        "claim_boundary": {
            "conformer_completeness_established": False,
            "basin_measures_included": False,
            "equilibrium_established": False,
            "chemical_accuracy_established": False,
            "public_solvfe_eligible": False,
        },
        "decision": {
            "result": "label_blind_high_endpoint_screen_complete",
            "promotion_allowed": False,
            "long_sampling_decision_requires_separate_score": True,
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


def _model_score(
    *,
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    model_index: int,
) -> dict[str, Any]:
    labels = np.asarray(
        [row["experimental_kcal_mol"] for row in rows],
        dtype=np.float64,
    )
    method_names = tuple(rows[0]["predictions_kcal_mol"])
    predictions = {
        name: np.asarray(
            [row["predictions_kcal_mol"][name] for row in rows],
            dtype=np.float64,
        )
        for name in method_names
    }
    errors = {name: values - labels for name, values in predictions.items()}
    settings = protocol["scoring"]
    comparisons = {}
    for index, baseline in enumerate(
        ("fixed_chagb_pbsa", "discrete_obc2_ace")
    ):
        comparisons[f"high_discrete_vs_{baseline}"] = _paired_score(
            errors[baseline],
            errors["discrete_chagb_pbsa"],
            resamples=settings["bootstrap_resamples"],
            confidence=settings["bootstrap_confidence"],
            seed=settings["bootstrap_seed"] + model_index * 10 + index,
        )
    gain = (
        np.abs(errors["fixed_chagb_pbsa"])
        - np.abs(errors["discrete_chagb_pbsa"])
    )
    positive = np.maximum(gain, 0.0)
    positive_sum = float(np.sum(positive))
    maximum_positive_fraction = (
        float(np.max(positive) / positive_sum)
        if positive_sum > 0.0
        else 1.0
    )
    return {
        "case_count": len(rows),
        "methods": {
            name: _error_metrics(values) for name, values in errors.items()
        },
        "paired_comparisons": comparisons,
        "improved_case_fraction_vs_fixed_chagb": float(
            np.mean(gain > 1.0e-12)
        ),
        "maximum_single_case_positive_gain_fraction": (
            maximum_positive_fraction
        ),
        "promotion_allowed": False,
    }


def run_score(
    *,
    protocol_path: Path,
    energy_path: Path,
    label_path: Path,
    output_path: Path,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    protocol, fingerprint = load_protocol(protocol_path)
    energy = load_json(energy_path)
    _self_hashed(energy, name="CHA-GB conformer energy artifact")
    if energy.get("protocol_fingerprint") != fingerprint:
        raise ValueError("Energy artifact protocol fingerprint changed.")
    labels = load_json(label_path)
    _self_hashed(labels, name="Development label artifact")
    expected = protocol["source_evidence"]
    if sha256_file(label_path) != expected["development_score_artifact_sha256"]:
        raise ValueError("Development label artifact file hash changed.")
    if (
        labels["content_sha256"]
        != expected["development_score_artifact_content_sha256"]
    ):
        raise ValueError("Development label content hash changed.")

    label_by_id = _identity_map(
        labels["cases"],
        keys=("model", "compound_id"),
        name="development label",
    )
    model_by_id = _identity_map(
        energy["model_records"],
        keys=("model", "compound_id"),
        name="high-endpoint model/case",
    )
    endpoint_by_id = {
        record["compound_id"]: record for record in energy["case_endpoints"]
    }
    if set(label_by_id) != set(model_by_id):
        raise ValueError("Energy and label model/case matrices differ.")

    score_rows: list[dict[str, Any]] = []
    for identity in sorted(model_by_id):
        model, compound_id = identity
        source = label_by_id[identity]
        high = model_by_id[identity]
        endpoint = endpoint_by_id[compound_id]
        experimental = float(source["experimental_kcal_mol"])
        predictions = {
            "fixed_obc2_ace": float(source["fixed_geometry_kcal_mol"]),
            "discrete_obc2_ace": float(source["discrete_conformer_kcal_mol"]),
            "fixed_chagb_pbsa": float(
                endpoint["reference_endpoint_kcal_mol"]
            ),
            "discrete_chagb_pbsa": float(
                high["analysis"]["delta_g_discrete_kcal_mol"]
            ),
        }
        score_rows.append(
            {
                "model": model,
                "compound_id": compound_id,
                "name": high["name"],
                "experimental_kcal_mol": experimental,
                "predictions_kcal_mol": predictions,
                "signed_errors_kcal_mol": {
                    name: value - experimental
                    for name, value in predictions.items()
                },
                "high_weight_diagnostic_passed": bool(
                    high["analysis"]["gates"]["ensemble_diagnostic_passed"]
                ),
            }
        )

    model_names = list(protocol["matrix"]["models"])
    summaries = {
        model: _model_score(
            rows=[row for row in score_rows if row["model"] == model],
            protocol=protocol,
            model_index=index,
        )
        for index, model in enumerate(model_names)
    }
    material = float(protocol["scoring"]["material_gain_gate_kcal_mol"])
    for model, summary in summaries.items():
        model_rows = [
            row for row in score_rows if row["model"] == model
        ]
        fixed = summary["paired_comparisons"][
            "high_discrete_vs_fixed_chagb_pbsa"
        ]
        low = summary["paired_comparisons"][
            "high_discrete_vs_discrete_obc2_ace"
        ]
        summary["long_sampling_signal_checks"] = {
            "material_gain_vs_fixed_chagb": (
                fixed["mean_mae_gain_kcal_mol"] >= material
            ),
            "material_gain_vs_discrete_obc2": (
                low["mean_mae_gain_kcal_mol"] >= material
            ),
            "paired_interval_vs_fixed_excludes_zero": (
                fixed["bootstrap_ci_kcal_mol"][0] > 0.0
            ),
            "not_single_case_driven": (
                summary["maximum_single_case_positive_gain_fraction"]
                <= float(
                    protocol["scoring"][
                        "maximum_single_case_positive_gain_fraction"
                    ]
                )
            ),
            "majority_of_cases_improve_vs_fixed": (
                summary["improved_case_fraction_vs_fixed_chagb"]
                >= float(
                    protocol["scoring"][
                        "minimum_improved_case_fraction"
                    ]
                )
            ),
            "all_high_weight_diagnostics_pass": all(
                row["high_weight_diagnostic_passed"]
                for row in model_rows
            ),
        }
        summary["long_sampling_signal_passed"] = all(
            summary["long_sampling_signal_checks"].values()
        )

    all_models_pass = all(
        summary["long_sampling_signal_passed"]
        for summary in summaries.values()
    )
    artifact = {
        "schema_version": 1,
        "artifact_type": (
            "route1-multi-mlip-chagb-discrete-conformer-development-score"
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
        "records": score_rows,
        "model_summaries": summaries,
        "long_sampling_signal_passed_for_all_models": all_models_pass,
        "label_boundary": {
            "energy_artifact_sealed_before_scoring": True,
            "labels_read_only_by_score_phase": True,
            "labels_changed_neither_cases_parameters_nor_energies": True,
            "confirmation_partition": False,
        },
        "claim_boundary": {
            "conformer_completeness_established": False,
            "basin_measures_included": False,
            "equilibrium_established": False,
            "chemical_accuracy_established": False,
            "public_solvfe_eligible": False,
        },
        "decision": {
            "result": (
                "long_sampling_candidate_supported"
                if all_models_pass
                else "long_sampling_candidate_not_supported"
            ),
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
    write_json_atomic(output_path, sealed)
    return sealed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default=str(
            SCRIPT_DIR
            / "multi_mlip_chagb_discrete_conformer_protocol.json"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    energy = commands.add_parser("energy")
    energy.add_argument(
        "--output",
        default=str(
            SCRIPT_DIR
            / "route1-multi-mlip-chagb-discrete-conformer-2026-07-25.json"
        ),
    )
    energy.add_argument(
        "--work-dir",
        default=str(
            REPOSITORY_ROOT
            / ".omx/benchmarks/route1-multi-mlip-chagb-discrete"
        ),
    )
    energy.add_argument("--amber-bin")

    score = commands.add_parser("score")
    score.add_argument(
        "--energy",
        default=str(
            SCRIPT_DIR
            / "route1-multi-mlip-chagb-discrete-conformer-2026-07-25.json"
        ),
    )
    score.add_argument(
        "--labels",
        default=str(
            SCRIPT_DIR
            / "route1-multi-mlip-discrete-conformer-score-2026-07-25.json"
        ),
    )
    score.add_argument(
        "--output",
        default=str(
            SCRIPT_DIR
            / "route1-multi-mlip-chagb-discrete-conformer-score-2026-07-25.json"
        ),
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    arguments = dict(vars(args))
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
