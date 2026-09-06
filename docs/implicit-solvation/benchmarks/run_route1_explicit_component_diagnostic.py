#!/usr/bin/env python3
"""Diagnose Route 1 continuum-component mismatch against published FreeSolv terms.

This is a retrospective, label-exposed diagnostic.  It never changes a Route 1
energy: the published explicit-solvent calculation is an auxiliary decomposition,
not a replacement target or experimental truth.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # noqa: E402

DEFAULT_PROTOCOL = SCRIPT_DIR / "route1_explicit_component_diagnostic_protocol_v1.json"
DEFAULT_COMPONENT_ARTIFACT = SCRIPT_DIR / "route1-chagb-component-attribution-2026-07-25.json"
DEFAULT_FREESOLV_DATABASE = (
    REPOSITORY_ROOT
    / ".omx/vendor-audits/freesolv-current-20260726-v1/database.json"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "route1-freesolv-explicit-component-diagnostic-2026-07-29.json"
)

ARTIFACT_TYPE = "route1-freesolv-label-exposed-explicit-component-diagnostic"
ENDPOINT = {
    "method_id": "chagb_pbsa_cavity_dispersion",
    "polar_component_key": "chagb_polar",
    "baseline_polar_component_key": "obc2_polar",
    "nonpolar_component_key": "pbsa_cavity_dispersion",
    "published_total_key": "calc",
    "published_charging_key": "calc_charging",
    "published_vdw_key": "calc_vdw",
}
ROUTE1_BOUNDARY = {
    "name": "Additive fixed-charge PB/GB implicit solvation",
    "formula": "E_solution(R)=E_MLIP,gas(R)+G_polar(R,q_fixed)+G_nonpolar(R)",
    "gas_phase_mm_energy": False,
    "hydration_label_residual": False,
    "mlip_retraining": False,
    "fixed_charge": "AM1-BCC",
}
CONCLUSION = "diagnostic_bottleneck_decomposition_not_endpoint_evidence"


def _relative_to_repository(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_object(path: Path) -> dict[str, Any]:
    value = core.load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number.")
    return result


def _sha256(value: object, *, field: str) -> str:
    text = str(value)
    if not core._is_hex(text, 64):
        raise ValueError(f"{field} must be a lowercase SHA256.")
    return text


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty sequence.")
    return math.fsum(values) / len(values)


def _pearson_r(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Pearson inputs must have the same length >= 2.")
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = math.fsum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_scale = math.sqrt(math.fsum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(math.fsum((y - right_mean) ** 2 for y in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return one-based average ranks, deterministically preserving equal ties."""
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[start][1]:
            stop += 1
        rank = (start + 1 + stop) / 2.0
        for position in range(start, stop):
            ranks[ordered[position][0]] = rank
        start = stop
    return ranks


def _spearman_rho(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson_r(_average_ranks(left), _average_ranks(right))


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize empty values.")
    return {
        "mean_signed_kcal_mol": _mean(values),
        "mean_absolute_kcal_mol": _mean([abs(value) for value in values]),
        "rmse_kcal_mol": math.sqrt(_mean([value * value for value in values])),
        "maximum_absolute_kcal_mol": max(abs(value) for value in values),
    }


def load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = _load_object(Path(path).resolve())
    if protocol.get("schema_version") != 1:
        raise ValueError("Only component-diagnostic protocol schema 1 is supported.")
    if protocol.get("protocol_id") != "maple-route1-explicit-component-diagnostic-v1":
        raise ValueError("Unexpected component-diagnostic protocol id.")
    if protocol.get("route1_boundary") != ROUTE1_BOUNDARY:
        raise ValueError("Component diagnostic violates the Route 1 boundary.")
    if protocol.get("endpoint") != ENDPOINT:
        raise ValueError("Component diagnostic endpoint definition changed.")

    design = protocol.get("evaluation_design", {})
    required_design = {
        "historical_label_exposure": True,
        "endpoint_energy_phase_reads_labels": False,
        "diagnostic_phase_reads_labels": True,
        "selection_reads_labels": False,
        "independent_blind_confirmation": False,
        "published_explicit_calculation_is_experimental_truth": False,
        "no_fit": True,
        "no_residual": True,
        "no_endpoint_selection": True,
        "no_threshold_tuning": True,
        "no_force_claim": True,
        "no_ranking_certification": True,
    }
    if any(design.get(key) is not value for key, value in required_design.items()):
        raise ValueError("Component diagnostic weakens its evidence boundary.")

    evidence = protocol.get("source_evidence", {})
    for key in (
        "component_artifact_sha256",
        "component_artifact_content_sha256",
        "freesolv_database_sha256",
    ):
        _sha256(evidence.get(key), field=f"source_evidence.{key}")
    if not core._is_hex(str(evidence.get("freesolv_commit", "")), 40):
        raise ValueError("source_evidence.freesolv_commit must be a Git SHA.")
    expected = evidence.get("expected_case_count")
    if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
        raise ValueError("source_evidence.expected_case_count must be positive.")

    gates = protocol.get("fixed_diagnostic_gates", {})
    if _finite(gates.get("absolute_error_threshold_kcal_mol"), field="threshold") != 1.5:
        raise ValueError("The fixed absolute-error gate changed.")
    if (
        _finite(
            gates.get("published_component_sum_rounding_tolerance_kcal_mol"),
            field="rounding tolerance",
        )
        != 0.0011
    ):
        raise ValueError("The published-component rounding tolerance changed.")
    if _finite(gates.get("component_tie_tolerance_kcal_mol"), field="tie tolerance") != 1e-12:
        raise ValueError("The component tie tolerance changed.")
    if tuple(gates.get("reported_associations", ())) != (
        "pearson_r_route_error_vs_polar_mismatch",
        "spearman_rho_route_error_vs_polar_mismatch",
        "pearson_r_route_error_vs_nonpolar_mismatch",
        "spearman_rho_route_error_vs_nonpolar_mismatch",
    ):
        raise ValueError("Reported association membership or order changed.")
    for key in (
        "p_values_reported",
        "confidence_intervals_reported",
        "regression_fit_reported",
        "bootstrap_performed",
    ):
        if gates.get(key) is not False:
            raise ValueError(f"Unsupported inferential statistic enabled: {key}.")

    rule = protocol.get("pre_registered_decision_rule", {})
    if rule.get("conclusion") != CONCLUSION:
        raise ValueError("Component diagnostic conclusion changed.")
    for key in (
        "endpoint_selection_allowed",
        "charge_method_selection_allowed",
        "energy_correction_allowed",
        "parameter_update_allowed",
        "certified_ranking_allowed",
    ):
        if rule.get(key) is not False:
            raise ValueError(f"Component diagnostic permits unsupported action: {key}.")
    return protocol, core.sha256_bytes(core.canonical_json_bytes(protocol))


def _load_component_artifact(
    protocol: Mapping[str, Any], path: Path
) -> list[dict[str, Any]]:
    evidence = protocol["source_evidence"]
    if core.sha256_file(path) != evidence["component_artifact_sha256"]:
        raise ValueError("Frozen component-artifact file hash mismatch.")
    artifact = _load_object(path)
    if core.artifact_content_sha256(artifact) != artifact.get("content_sha256"):
        raise ValueError("Frozen component-artifact content seal mismatch.")
    if artifact.get("content_sha256") != evidence["component_artifact_content_sha256"]:
        raise ValueError("Frozen component-artifact content hash mismatch.")
    if artifact.get("source_partition") != "development":
        raise ValueError("Component artifact must remain development-only.")
    expected = int(evidence["expected_case_count"])
    records = artifact.get("records")
    if artifact.get("case_count") != expected or not isinstance(records, list) or len(records) != expected:
        raise ValueError("Component artifact coverage changed.")

    ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    endpoint = protocol["endpoint"]
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Component artifact record must be an object.")
        compound_id = str(record.get("compound_id", ""))
        if not compound_id or compound_id in ids:
            raise ValueError("Component artifact IDs must be unique and nonempty.")
        ids.add(compound_id)
        components = record.get("components_kcal_mol")
        predictions = record.get("predictions_kcal_mol")
        if not isinstance(components, dict) or not isinstance(predictions, dict):
            raise ValueError(f"Component record {compound_id} lacks components or predictions.")
        polar = _finite(
            components.get(endpoint["polar_component_key"]),
            field=f"{compound_id}.polar",
        )
        baseline_polar = _finite(
            components.get(endpoint["baseline_polar_component_key"]),
            field=f"{compound_id}.baseline polar",
        )
        nonpolar = _finite(
            components.get(endpoint["nonpolar_component_key"]),
            field=f"{compound_id}.nonpolar",
        )
        route_total = _finite(
            predictions.get(endpoint["method_id"]),
            field=f"{compound_id}.route total",
        )
        if abs(route_total - polar - nonpolar) > 1e-9:
            raise ValueError(f"Route component sum mismatch for {compound_id}.")
        experimental = _finite(
            record.get("experimental_kcal_mol"),
            field=f"{compound_id}.experimental",
        )
        rows.append(
            {
                "compound_id": compound_id,
                "source_record_sha256": _sha256(
                    record.get("source_record_sha256"),
                    field=f"{compound_id}.source_record_sha256",
                ),
                "experimental_kcal_mol": experimental,
                "route_total_kcal_mol": route_total,
                "route_polar_kcal_mol": polar,
                "route_baseline_polar_kcal_mol": baseline_polar,
                "route_nonpolar_kcal_mol": nonpolar,
            }
        )
    return sorted(rows, key=lambda row: row["compound_id"])


def _load_freesolv_components(
    protocol: Mapping[str, Any], path: Path, ids: set[str]
) -> dict[str, dict[str, float]]:
    evidence = protocol["source_evidence"]
    if core.sha256_file(path) != evidence["freesolv_database_sha256"]:
        raise ValueError("Frozen FreeSolv database hash mismatch.")
    database = _load_object(path)
    endpoint = protocol["endpoint"]
    missing = sorted(ids - set(database))
    if missing:
        raise ValueError("FreeSolv database is missing component IDs: " + ", ".join(missing[:5]))
    result: dict[str, dict[str, float]] = {}
    tolerance = protocol["fixed_diagnostic_gates"][
        "published_component_sum_rounding_tolerance_kcal_mol"
    ]
    for compound_id in sorted(ids):
        record = database[compound_id]
        if not isinstance(record, dict):
            raise ValueError(f"FreeSolv entry {compound_id} must be an object.")
        total = _finite(record.get(endpoint["published_total_key"]), field=f"{compound_id}.calc")
        charging = _finite(
            record.get(endpoint["published_charging_key"]),
            field=f"{compound_id}.calc_charging",
        )
        vdw = _finite(record.get(endpoint["published_vdw_key"]), field=f"{compound_id}.calc_vdw")
        decomposition_residual = total - charging - vdw
        if abs(decomposition_residual) > tolerance:
            raise ValueError(f"Published component sum exceeds rounding tolerance for {compound_id}.")
        result[compound_id] = {
            "total_kcal_mol": total,
            "charging_kcal_mol": charging,
            "vdw_kcal_mol": vdw,
            "decomposition_rounding_residual_kcal_mol": decomposition_residual,
        }
    return result


def _component_dominance(rows: Sequence[Mapping[str, float]], tie_tolerance: float) -> dict[str, int]:
    counts = {"polar_larger": 0, "nonpolar_larger": 0, "tied": 0}
    for row in rows:
        polar = abs(float(row["polar_mismatch_kcal_mol"]))
        nonpolar = abs(float(row["nonpolar_mismatch_kcal_mol"]))
        if abs(polar - nonpolar) <= tie_tolerance:
            counts["tied"] += 1
        elif polar > nonpolar:
            counts["polar_larger"] += 1
        else:
            counts["nonpolar_larger"] += 1
    return counts


def _diagnostic_artifact(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    fingerprint: str,
    component_artifact_path: Path,
    freesolv_database_path: Path,
    output: Path,
) -> dict[str, Any]:
    component_rows = _load_component_artifact(protocol, component_artifact_path)
    explicit = _load_freesolv_components(
        protocol,
        freesolv_database_path,
        {str(row["compound_id"]) for row in component_rows},
    )
    threshold = float(protocol["fixed_diagnostic_gates"]["absolute_error_threshold_kcal_mol"])
    tie_tolerance = float(protocol["fixed_diagnostic_gates"]["component_tie_tolerance_kcal_mol"])

    rows: list[dict[str, Any]] = []
    for route in component_rows:
        compound_id = str(route["compound_id"])
        reference = explicit[compound_id]
        polar_mismatch = float(route["route_polar_kcal_mol"]) - reference["charging_kcal_mol"]
        baseline_polar_mismatch = (
            float(route["route_baseline_polar_kcal_mol"])
            - reference["charging_kcal_mol"]
        )
        nonpolar_mismatch = float(route["route_nonpolar_kcal_mol"]) - reference["vdw_kcal_mol"]
        route_minus_explicit = float(route["route_total_kcal_mol"]) - reference["total_kcal_mol"]
        identity_residual = (
            route_minus_explicit
            - polar_mismatch
            - nonpolar_mismatch
            + reference["decomposition_rounding_residual_kcal_mol"]
        )
        if abs(identity_residual) > 1.2e-12:
            raise ValueError(f"Component identity failed for {compound_id}.")
        route_error = float(route["route_total_kcal_mol"]) - float(route["experimental_kcal_mol"])
        explicit_error = reference["total_kcal_mol"] - float(route["experimental_kcal_mol"])
        error_identity_residual = route_error - explicit_error - route_minus_explicit
        if abs(error_identity_residual) > 1.2e-12:
            raise ValueError(f"Error decomposition identity failed for {compound_id}.")
        rows.append(
            {
                "compound_id": compound_id,
                "source_record_sha256": route["source_record_sha256"],
                "polar_mismatch_kcal_mol": polar_mismatch,
                "obc2_polar_mismatch_kcal_mol": baseline_polar_mismatch,
                "nonpolar_mismatch_kcal_mol": nonpolar_mismatch,
                "route_minus_published_explicit_kcal_mol": route_minus_explicit,
                "published_component_rounding_residual_kcal_mol": reference[
                    "decomposition_rounding_residual_kcal_mol"
                ],
                "component_identity_residual_kcal_mol": identity_residual,
                "route_absolute_error_exceeds_fixed_gate": abs(route_error) >= threshold,
                "published_explicit_absolute_error_exceeds_fixed_gate": abs(explicit_error)
                >= threshold,
                "continuum_reference_difference_exceeds_fixed_gate": abs(route_minus_explicit)
                >= threshold,
                "_route_error": route_error,
                "_explicit_error": explicit_error,
            }
        )

    route_errors = [float(row["_route_error"]) for row in rows]
    explicit_errors = [float(row["_explicit_error"]) for row in rows]
    continuum_errors = [float(row["route_minus_published_explicit_kcal_mol"]) for row in rows]
    polar_mismatch = [float(row["polar_mismatch_kcal_mol"]) for row in rows]
    baseline_polar_mismatch = [
        float(row["obc2_polar_mismatch_kcal_mol"]) for row in rows
    ]
    nonpolar_mismatch = [float(row["nonpolar_mismatch_kcal_mol"]) for row in rows]
    rounding_residual = [float(row["published_component_rounding_residual_kcal_mol"]) for row in rows]
    identity_residual = [float(row["component_identity_residual_kcal_mol"]) for row in rows]

    def paired_polar_reduction(selected_rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        reductions = [
            abs(float(row["obc2_polar_mismatch_kcal_mol"]))
            - abs(float(row["polar_mismatch_kcal_mol"]))
            for row in selected_rows
        ]
        return {
            "case_count": len(reductions),
            "mean_absolute_mismatch_reduction_kcal_mol": _mean(reductions),
            "chagb_lower_absolute_mismatch_count": sum(value > 0.0 for value in reductions),
            "obc2_lower_absolute_mismatch_count": sum(value < 0.0 for value in reductions),
            "exact_tie_count": sum(value == 0.0 for value in reductions),
        }

    exposure = {
        "route_and_published_explicit_exceed": 0,
        "route_only_exceeds": 0,
        "published_explicit_only_exceeds": 0,
        "neither_exceeds": 0,
        "route_exceeds_and_continuum_difference_exceeds": 0,
        "route_exceeds_and_continuum_difference_within": 0,
    }
    for row in rows:
        route_exceeds = bool(row["route_absolute_error_exceeds_fixed_gate"])
        explicit_exceeds = bool(row["published_explicit_absolute_error_exceeds_fixed_gate"])
        continuum_exceeds = bool(row["continuum_reference_difference_exceeds_fixed_gate"])
        if route_exceeds and explicit_exceeds:
            exposure["route_and_published_explicit_exceed"] += 1
        elif route_exceeds:
            exposure["route_only_exceeds"] += 1
        elif explicit_exceeds:
            exposure["published_explicit_only_exceeds"] += 1
        else:
            exposure["neither_exceeds"] += 1
        if route_exceeds and continuum_exceeds:
            exposure["route_exceeds_and_continuum_difference_exceeds"] += 1
        elif route_exceeds:
            exposure["route_exceeds_and_continuum_difference_within"] += 1

    public_rows = []
    for row in rows:
        public_rows.append({key: value for key, value in row.items() if not key.startswith("_")})
    tail_rows = [row for row in rows if row["route_absolute_error_exceeds_fixed_gate"]]

    artifact = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": core.sha256_file(protocol_path),
        "protocol_fingerprint": fingerprint,
        "claim_scope": protocol["claim_scope"],
        "route1_boundary": ROUTE1_BOUNDARY,
        "evaluation_design": protocol["evaluation_design"],
        "source_provenance": {
            "component_artifact": _relative_to_repository(component_artifact_path),
            "component_artifact_sha256": core.sha256_file(component_artifact_path),
            "component_artifact_content_sha256": _load_object(component_artifact_path)[
                "content_sha256"
            ],
            "freesolv_database": _relative_to_repository(freesolv_database_path),
            "freesolv_database_sha256": core.sha256_file(freesolv_database_path),
            "freesolv_repository": protocol["source_evidence"]["freesolv_repository"],
            "freesolv_commit": protocol["source_evidence"]["freesolv_commit"],
        },
        "case_count": len(public_rows),
        "fixed_error_gate_kcal_mol": threshold,
        "published_component_definition": {
            "total": "FreeSolv database.json calc",
            "polar_proxy": "FreeSolv database.json calc_charging",
            "nonpolar_proxy": "FreeSolv database.json calc_vdw",
            "caveat": "Published calculated components are a GAFF/AM1-BCC explicit-solvent decomposition and are not experimental labels or a direct replacement target for Route 1.",
        },
        "component_comparison": {
            "obc2_polar_minus_published_charging": _summary(baseline_polar_mismatch),
            "chagb_polar_minus_published_charging": _summary(polar_mismatch),
            "pbsa_cavity_dispersion_minus_published_vdw": _summary(nonpolar_mismatch),
            "route_total_minus_published_explicit_total": _summary(continuum_errors),
            "published_component_rounding_residual": _summary(rounding_residual),
            "component_identity_residual": _summary(identity_residual),
        },
        "polar_model_comparison_against_published_charging": {
            "paired_absolute_mismatch_reduction_chagb_minus_obc2": paired_polar_reduction(rows),
            "route_error_tail_paired_absolute_mismatch_reduction_chagb_minus_obc2": paired_polar_reduction(tail_rows),
            "interpretation_boundary": "This paired component comparison confirms or rejects agreement with one published explicit-solvent charging decomposition. It does not choose an endpoint for Route 1, establish experimental accuracy, or identify a causal missing term.",
        },
        "label_exposed_error_decomposition": {
            "route_total_minus_experiment": _summary(route_errors),
            "published_explicit_total_minus_experiment": _summary(explicit_errors),
            "route_total_minus_published_explicit_total": _summary(continuum_errors),
            "identity": "route_error = published_explicit_error + route_minus_published_explicit",
            "fixed_gate_exposure": exposure,
            "interpretation_boundary": "The four-way gate table is a tail-exposure decomposition, not causal attribution. Both the published force-field/sampling calculation and the continuum endpoint can contribute to a Route 1 error.",
        },
        "component_dominance": {
            "all_cases": _component_dominance(rows, tie_tolerance),
            "route_error_tail_cases": _component_dominance(tail_rows, tie_tolerance),
            "tie_tolerance_kcal_mol": tie_tolerance,
            "interpretation_boundary": "Larger absolute mismatch identifies a descriptive component discrepancy against the published calculation; it does not authorize replacing or scaling that component.",
        },
        "descriptive_associations": {
            "pearson_r_route_error_vs_polar_mismatch": _pearson_r(route_errors, polar_mismatch),
            "spearman_rho_route_error_vs_polar_mismatch": _spearman_rho(route_errors, polar_mismatch),
            "pearson_r_route_error_vs_nonpolar_mismatch": _pearson_r(route_errors, nonpolar_mismatch),
            "spearman_rho_route_error_vs_nonpolar_mismatch": _spearman_rho(route_errors, nonpolar_mismatch),
            "p_values_reported": False,
            "confidence_intervals_reported": False,
            "regression_fit_reported": False,
            "causal_attribution_available": False,
        },
        "records": public_rows,
        "decision": {
            "status": CONCLUSION,
            "endpoint_selection_allowed": False,
            "charge_method_selection_allowed": False,
            "energy_correction_allowed": False,
            "parameter_update_allowed": False,
            "certified_ranking_allowed": False,
            "failure_action": protocol["pre_registered_decision_rule"]["failure_action"],
        },
        "command_provenance": core.command_provenance(
            __file__,
            {
                "phase": "label-exposed-explicit-component-diagnostic",
                "protocol": _relative_to_repository(protocol_path),
                "component_artifact": _relative_to_repository(component_artifact_path),
                "freesolv_database": _relative_to_repository(freesolv_database_path),
                "output": _relative_to_repository(output),
            },
            repository_root=REPOSITORY_ROOT,
        ),
    }
    return core.seal_artifact(artifact)


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = Path(args.protocol).resolve()
    component_artifact_path = Path(args.component_artifact).resolve()
    freesolv_database_path = Path(args.freesolv_database).resolve()
    output = Path(args.output).resolve()
    protocol, fingerprint = load_protocol(protocol_path)
    artifact = _diagnostic_artifact(
        protocol=protocol,
        protocol_path=protocol_path,
        fingerprint=fingerprint,
        component_artifact_path=component_artifact_path,
        freesolv_database_path=freesolv_database_path,
        output=output,
    )
    core.write_json_atomic(output, artifact)
    print(
        json.dumps(
            {
                "output": str(output),
                "file_sha256": core.sha256_file(output),
                "content_sha256": artifact["content_sha256"],
                "case_count": artifact["case_count"],
                "decision": artifact["decision"]["status"],
            },
            indent=2,
        )
    )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--component-artifact", type=Path, default=DEFAULT_COMPONENT_ARTIFACT)
    parser.add_argument("--freesolv-database", type=Path, default=DEFAULT_FREESOLV_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
