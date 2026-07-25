#!/usr/bin/env python
"""Analyze frozen Route 1 lambda records with upstream PyMBAR."""

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
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core import (  # noqa: E402
    artifact_content_sha256,
    canonical_json_bytes,
    load_json,
    seal_artifact,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from maple.function.free_energy import mbar as mbar_module  # noqa: E402
from maple.function.free_energy import analyze_mbar  # noqa: E402

PROTOCOL_ID = "maple-route1-multi-mlip-obc2-mbar-analysis-v1"


def load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol_path = Path(path)
    protocol = load_json(protocol_path)
    if protocol.get("schema_version") != 1:
        raise ValueError("Only MBAR analysis protocol schema version 1 is supported.")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected MBAR analysis protocol ID.")
    boundary = protocol.get("route_contract", {})
    required_true = (
        "fixed_charge_across_windows",
        "no_gas_phase_mm_energy",
        "no_hydration_residual_model",
        "no_mlip_retraining",
    )
    if not all(boundary.get(key) is True for key in required_true):
        raise ValueError("MBAR analysis protocol violates the Route 1 boundary.")
    if boundary.get("experimental_labels_read") is not False:
        raise ValueError("MBAR energy analysis must remain label-free.")
    if protocol["analysis"].get("equilibrium_claim") is not False:
        raise ValueError("The frozen short records cannot claim equilibrium.")
    if protocol["decision_policy"].get("promotion_allowed") is not False:
        raise ValueError("The post hoc MBAR analysis cannot promote the parent screen.")

    source = protocol["source_evidence"]
    manifest_path = protocol_path.parent / source["manifest"]
    parent_path = protocol_path.parent / source["parent_protocol"]
    if sha256_file(manifest_path) != source["manifest_sha256"]:
        raise ValueError("MBAR source manifest hash changed.")
    if sha256_file(parent_path) != source["parent_protocol_sha256"]:
        raise ValueError("Parent TI protocol hash changed.")
    return protocol, sha256_bytes(canonical_json_bytes(protocol))


def load_source_manifest(
    protocol_path: str | Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    source = protocol["source_evidence"]
    manifest_path = Path(protocol_path).parent / source["manifest"]
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("record_count") != source["record_count"]
        or len(manifest.get("records", [])) != source["record_count"]
    ):
        raise ValueError("Invalid MBAR source manifest.")
    identities: set[tuple[str, str]] = set()
    for row in manifest["records"]:
        identity = (row["model"], row["compound_id"])
        if identity in identities:
            raise ValueError(f"Duplicate MBAR source identity: {identity}.")
        identities.add(identity)
    lowered = json.dumps(manifest, sort_keys=True).lower()
    if "experimental_kcal_mol" in lowered:
        raise ValueError("MBAR source manifest contains an experimental label.")
    return manifest


def load_source_record(row: dict[str, Any]) -> dict[str, Any]:
    path = REPOSITORY_ROOT / row["relative_path"]
    if sha256_file(path) != row["file_sha256"]:
        raise ValueError(f"MBAR source file hash changed: {path}.")
    record = load_json(path)
    if (
        record.get("content_sha256") != row["content_sha256"]
        or artifact_content_sha256(record) != row["content_sha256"]
    ):
        raise ValueError(f"MBAR source content hash changed: {path}.")
    identity = (record.get("model"), record.get("compound_id"))
    if identity != (row["model"], row["compound_id"]):
        raise ValueError(f"MBAR source identity changed: {path}.")
    if "experimental_kcal_mol" in json.dumps(record, sort_keys=True).lower():
        raise ValueError(f"MBAR source record contains an experimental label: {path}.")
    return record


def state_replicates(
    record: dict[str, Any],
    lambda_values: list[float],
) -> list[list[np.ndarray]]:
    if len(record.get("chains", [])) < 2:
        raise ValueError("MBAR analysis requires at least two independent chains.")
    states: list[list[np.ndarray]] = []
    for coupling in lambda_values:
        replicates: list[np.ndarray] = []
        for chain in record["chains"]:
            matches = [
                window
                for window in chain["windows"]
                if np.isclose(window["lambda"], coupling)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Chain {chain['chain_id']} does not contain lambda={coupling}."
                )
            samples = matches[0].get("samples", [])
            matrix = np.asarray(
                [sample["reduced_potentials"] for sample in samples],
                dtype=np.float64,
            )
            replicates.append(matrix)
        states.append(replicates)
    return states


def _analysis_kwargs(protocol: dict[str, Any]) -> dict[str, Any]:
    analysis = protocol["analysis"]
    return {
        "lambda_values": analysis["lambda_values"],
        "temperature_kelvin": analysis["temperature_kelvin"],
        "standard_state": analysis["standard_state"],
        "equilibrium_claim": analysis["equilibrium_claim"],
        "detect_equilibration": analysis["detect_equilibration"],
        "equilibration_nskip": analysis["equilibration_nskip"],
        "minimum_uncorrelated_samples_per_state": (
            analysis["minimum_uncorrelated_samples_per_state"]
        ),
        "minimum_effective_samples_per_state": (
            analysis["minimum_effective_samples_per_state"]
        ),
        "minimum_adjacent_overlap": analysis["minimum_adjacent_overlap"],
        "maximum_iterations": analysis["maximum_iterations"],
        "relative_tolerance": analysis["relative_tolerance"],
    }


def analyze_record(
    record: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    lambdas = [float(value) for value in protocol["analysis"]["lambda_values"]]
    states = state_replicates(record, lambdas)
    kwargs = _analysis_kwargs(protocol)
    combined = analyze_mbar(states, **kwargs)
    repeats = [
        analyze_mbar(
            [[replicates[index]] for replicates in states],
            **kwargs,
        )
        for index in range(len(states[0]))
    ]
    repeat_values = [result["endpoint_delta_g_kcal_mol"] for result in repeats]
    repeat_difference = float(max(repeat_values) - min(repeat_values))
    mbar_ti_difference = float(
        combined["endpoint_delta_g_kcal_mol"] - record["ti_mean_kcal_mol"]
    )
    analysis = protocol["analysis"]
    checks = {
        "upstream_mbar_finite": bool(
            np.isfinite(combined["endpoint_delta_g_kcal_mol"])
            and np.isfinite(combined["endpoint_uncertainty_kcal_mol"])
        ),
        "solver_convergence": combined["gates"]["checks"]["solver_convergence"],
        "minimum_uncorrelated_samples_per_state": combined["gates"]["checks"][
            "minimum_uncorrelated_samples_per_state"
        ],
        "minimum_effective_samples_per_state": combined["gates"]["checks"][
            "minimum_effective_samples_per_state"
        ],
        "adjacent_overlap": combined["gates"]["checks"]["adjacent_overlap"],
        "mbar_ti_agreement": (
            abs(mbar_ti_difference)
            <= analysis["maximum_mbar_ti_absolute_difference_kcal_mol"]
        ),
        "independent_repeat_agreement": (
            repeat_difference
            <= analysis["maximum_independent_repeat_difference_kcal_mol"]
        ),
        "equilibrium_proven": False,
    }
    return {
        "model": record["model"],
        "compound_id": record["compound_id"],
        "name": record["name"],
        "flexibility_bin": record["flexibility_bin"],
        "mbar": combined,
        "independent_repeat_mbar_delta_g_kcal_mol": repeat_values,
        "independent_repeat_difference_kcal_mol": repeat_difference,
        "parent_ti_mean_kcal_mol": float(record["ti_mean_kcal_mol"]),
        "mbar_minus_ti_kcal_mol": mbar_ti_difference,
        "diagnostic_checks": checks,
        "all_diagnostic_checks_pass": all(checks.values()),
        "promotion_allowed": False,
    }


def run(protocol_path: str | Path) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    protocol, fingerprint = load_protocol(protocol_path)
    manifest = load_source_manifest(protocol_path, protocol)
    records = [
        analyze_record(load_source_record(row), protocol) for row in manifest["records"]
    ]
    overlap_values = [
        row["mbar"]["overlap"]["minimum_directional_adjacent_overlap"]
        for row in records
    ]
    uncertainties = [row["mbar"]["endpoint_uncertainty_kcal_mol"] for row in records]
    mbar_ti_differences = [abs(row["mbar_minus_ti_kcal_mol"]) for row in records]
    repeat_differences = [
        row["independent_repeat_difference_kcal_mol"] for row in records
    ]
    result = {
        "schema_version": 1,
        "artifact_type": "route1-multi-mlip-obc2-mbar-analysis",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "source_manifest_sha256": protocol["source_evidence"]["manifest_sha256"],
        "record_count": len(records),
        "records": records,
        "aggregate_diagnostics": {
            "minimum_directional_adjacent_overlap": float(min(overlap_values)),
            "maximum_endpoint_uncertainty_kcal_mol": float(max(uncertainties)),
            "maximum_absolute_mbar_ti_difference_kcal_mol": float(
                max(mbar_ti_differences)
            ),
            "maximum_independent_repeat_difference_kcal_mol": float(
                max(repeat_differences)
            ),
            "diagnostic_pass_count": sum(
                row["all_diagnostic_checks_pass"] for row in records
            ),
            "minimum_uncorrelated_sample_gate_pass_count": sum(
                row["diagnostic_checks"]["minimum_uncorrelated_samples_per_state"]
                for row in records
            ),
            "solver_convergence_gate_pass_count": sum(
                row["diagnostic_checks"]["solver_convergence"] for row in records
            ),
            "minimum_effective_sample_gate_pass_count": sum(
                row["diagnostic_checks"]["minimum_effective_samples_per_state"]
                for row in records
            ),
            "adjacent_overlap_gate_pass_count": sum(
                row["diagnostic_checks"]["adjacent_overlap"] for row in records
            ),
            "mbar_ti_agreement_pass_count": sum(
                row["diagnostic_checks"]["mbar_ti_agreement"] for row in records
            ),
            "independent_repeat_agreement_pass_count": sum(
                row["diagnostic_checks"]["independent_repeat_agreement"]
                for row in records
            ),
        },
        "decision": {
            "result": "diagnostic_only_not_promotable",
            "promotion_allowed": False,
            "reason": protocol["decision_policy"]["reason"],
        },
        "label_boundary": {
            "experimental_labels_read": False,
            "experimental_scoring_performed": False,
        },
        "dependency": protocol["dependency"],
        "implementation_provenance": {
            "runner": Path(__file__).resolve().relative_to(REPOSITORY_ROOT).as_posix(),
            "runner_sha256": sha256_file(__file__),
            "analysis_module": Path(mbar_module.__file__)
            .resolve()
            .relative_to(REPOSITORY_ROOT)
            .as_posix(),
            "analysis_module_sha256": sha256_file(mbar_module.__file__),
            "pymbar_version": records[0]["mbar"]["estimator"]["pymbar_version"],
        },
        "claim_boundary": {
            "upstream_mbar_path_exercised": True,
            "overlap_matrix_reported": True,
            "equilibrated_sampling_proven": False,
            "chemical_accuracy_established": False,
            "product_solvation_free_energy_established": False,
        },
    }
    return seal_artifact(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default=str(BENCHMARK_DIR / "mlip_obc2_mbar_protocol.json"),
    )
    parser.add_argument(
        "--output",
        default=str(BENCHMARK_DIR / "route1-multi-mlip-obc2-mbar-2026-07-25.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args.protocol)
    write_json_atomic(args.output, result)
    print(json.dumps(result["aggregate_diagnostics"], sort_keys=True))
    print(result["decision"]["result"])


if __name__ == "__main__":
    main()
