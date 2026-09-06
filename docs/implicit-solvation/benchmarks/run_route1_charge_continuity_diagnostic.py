#!/usr/bin/env python3
"""Join frozen Route 1 pair, energy, and score artifacts for one diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # noqa: E402
import ranking_metrics  # noqa: E402


DEFAULT_PROTOCOL = (
    SCRIPT_DIR / "route1_charge_continuity_diagnostic_protocol_v1.json"
)
DEFAULT_SOURCE_MANIFEST = SCRIPT_DIR / "route1_freesolv_reserve_source_manifest.json"
DEFAULT_ENERGY_ARTIFACT = (
    SCRIPT_DIR / "route1-freesolv-reserve-energy-2026-07-25.json"
)
DEFAULT_SCORE_ARTIFACT = (
    SCRIPT_DIR / "route1-freesolv-reserve-score-2026-07-25.json"
)
DEFAULT_PAIR_PROTOCOL = SCRIPT_DIR / "route1_charge_continuity_protocol_v1.json"
DEFAULT_PAIR_ARTIFACT = (
    SCRIPT_DIR
    / "route1-freesolv-reserve-charge-continuity-pairs-2026-07-29.json"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR
    / "route1-freesolv-reserve-charge-continuity-diagnostic-2026-07-29.json"
)

ARTIFACT_TYPE = "route1-freesolv-label-exposed-charge-continuity-diagnostic"
PAIR_ARTIFACT_TYPE = "route1-charge-continuity-label-free-pairs"
ENDPOINTS = (
    "am1bcc_obc2_ace",
    "am1bcc_chagb_pbsa_cavity_dispersion",
)
CHARGE_METRICS = (
    "remote_heavy_atom_charge_l2_e",
    "remote_heavy_atom_charge_rms_e",
    "remote_heavy_atom_charge_max_abs_e",
)
CONCLUSION = "diagnostic_signal_only_insufficient_for_threshold_or_certification"
ROUTE1_BOUNDARY = {
    "name": "Additive fixed-charge PB/GB implicit solvation",
    "formula": "E_solution(R)=E_MLIP,gas(R)+G_polar(R,q_fixed)+G_nonpolar(R)",
    "gas_phase_mm_energy": False,
    "hydration_label_residual": False,
    "mlip_retraining": False,
    "fixed_charge": "AM1-BCC",
}
PAIR_MAPPING_BOUNDARY = {
    "mcs_pattern_enumeration": "single_rdkit_findmcs_smarts_only",
    "distinct_maximum_mcs_patterns_audited": False,
    "mapping_enumeration_scope": (
        "cross_product_of_nonuniquified_embeddings_of_single_rdkit_findmcs_smarts"
    ),
}
_FORBIDDEN_PAIR_FIELDS = frozenset(
    {
        "experimental_kcal_mol",
        "experimental_uncertainty_kcal_mol",
        "signed_errors_kcal_mol",
        "hydration_free_energy",
    }
)
_FORBIDDEN_OUTPUT_FIELDS = _FORBIDDEN_PAIR_FIELDS | frozenset(
    {
        "p_value",
        "p_values",
        "confidence_interval",
        "confidence_intervals",
    }
)


def _relative_to_repository(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _load_object(path: Path) -> dict[str, Any]:
    value = core.load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value)
    if not core._is_hex(text, 64):
        raise ValueError(f"{field} must be a lowercase SHA256.")
    return text


def _finite_number(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite number.")
    return float(value)


def _reject_fields(value: object, forbidden: frozenset[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in forbidden:
                raise ValueError(f"Forbidden field {key!r} crossed an artifact boundary.")
            _reject_fields(nested, forbidden)
    elif isinstance(value, list):
        for nested in value:
            _reject_fields(nested, forbidden)


def load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = _load_object(Path(path).resolve())
    if protocol.get("schema_version") != 1:
        raise ValueError("Only charge-diagnostic protocol schema 1 is supported.")
    if (
        protocol.get("protocol_id")
        != "maple-route1-charge-continuity-diagnostic-v1"
    ):
        raise ValueError("Unexpected charge-continuity diagnostic protocol id.")
    if protocol.get("route1_boundary") != ROUTE1_BOUNDARY:
        raise ValueError("Charge-continuity diagnostic violates the Route 1 boundary.")

    design = protocol.get("evaluation_design", {})
    required_design = {
        "historical_label_exposure": True,
        "pair_phase_reads_labels": False,
        "diagnostic_phase_reads_labels": True,
        "selection_reads_labels": False,
        "pair_edges_are_independent": False,
        "independent_blind_confirmation": False,
        "charge_model_comparison_available": False,
        "certified_ranking_available": False,
        "no_fit": True,
        "no_residual": True,
        "no_endpoint_selection": True,
        "no_threshold_tuning": True,
    }
    if any(design.get(key) is not value for key, value in required_design.items()):
        raise ValueError("Diagnostic protocol weakens its evidence boundary.")
    if protocol.get("pair_mapping_boundary") != PAIR_MAPPING_BOUNDARY:
        raise ValueError("Diagnostic protocol weakens its pair-mapping boundary.")

    evidence = protocol.get("source_evidence", {})
    for field in (
        "source_manifest_sha256",
        "energy_artifact_sha256",
        "energy_artifact_content_sha256",
        "score_artifact_sha256",
        "score_artifact_content_sha256",
        "pair_protocol_sha256",
        "pair_artifact_sha256",
        "pair_artifact_content_sha256",
    ):
        _require_sha256(evidence.get(field), field=f"source_evidence.{field}")
    for field in ("expected_case_count", "expected_pair_count"):
        value = evidence.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"source_evidence.{field} must be a positive integer.")

    if tuple(protocol.get("endpoints", ())) != ENDPOINTS:
        raise ValueError("Diagnostic endpoint membership or order changed.")
    statistics_config = protocol.get("diagnostic_statistics", {})
    if statistics_config.get("lower_value_is_better") is not True:
        raise ValueError("Free-energy ordering must remain lower-is-better.")
    thresholds = list(
        statistics_config.get("fixed_experimental_delta_thresholds_kcal_mol", ())
    )
    if thresholds != [0.5, 1.0, 2.0]:
        raise ValueError("Fixed experimental-difference gates changed.")
    if float(statistics_config.get("experimental_uncertainty_z", 0.0)) != 1.96:
        raise ValueError("The pre-registered uncertainty gate changed.")
    if tuple(statistics_config.get("charge_drift_metrics", ())) != CHARGE_METRICS:
        raise ValueError("Charge-drift metric membership or order changed.")
    if tuple(statistics_config.get("reported_pair_metrics", ())) != (
        "forced_sign_accuracy",
        "answered_sign_accuracy",
        "delta_delta_mae_kcal_mol",
        "delta_delta_rmse_kcal_mol",
    ):
        raise ValueError("Reported pair-metric membership or order changed.")
    if statistics_config.get("reported_descriptive_association") != (
        "spearman_rho_charge_drift_vs_absolute_delta_delta_error"
    ):
        raise ValueError("Reported descriptive association changed.")
    for field in (
        "p_values_reported",
        "confidence_intervals_reported",
        "regression_fit_reported",
        "bootstrap_performed",
    ):
        if statistics_config.get(field) is not False:
            raise ValueError(f"Unsupported inferential statistic enabled: {field}.")

    rule = protocol.get("pre_registered_decision_rule", {})
    if rule.get("conclusion") != CONCLUSION:
        raise ValueError("Diagnostic conclusion changed.")
    for field in (
        "endpoint_selection_allowed",
        "charge_method_selection_allowed",
        "energy_correction_allowed",
        "abstention_threshold_selection_allowed",
        "certified_order_allowed",
    ):
        if rule.get(field) is not False:
            raise ValueError(f"Diagnostic protocol permits unsupported action: {field}.")
    return protocol, core.sha256_bytes(core.canonical_json_bytes(protocol))


def _load_source_manifest(
    protocol: Mapping[str, Any], path: Path
) -> tuple[dict[str, Any], set[str]]:
    evidence = protocol["source_evidence"]
    if core.sha256_file(path) != evidence["source_manifest_sha256"]:
        raise ValueError("Frozen source-manifest hash mismatch.")
    manifest = _load_object(path)
    _reject_fields(manifest, _FORBIDDEN_PAIR_FIELDS)
    records = manifest.get("records", [])
    expected = int(evidence["expected_case_count"])
    if not isinstance(records, list) or len(records) != expected:
        raise ValueError("Source-manifest case count mismatch.")
    ids = [str(row.get("compound_id", "")) for row in records]
    if ids != sorted(ids) or not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Source-manifest IDs must be sorted and unique.")
    if manifest.get("case_count") != expected:
        raise ValueError("Source-manifest declared case count mismatch.")
    return manifest, set(ids)


def _load_energy_artifact(
    protocol: Mapping[str, Any], path: Path, source_ids: set[str]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    evidence = protocol["source_evidence"]
    if core.sha256_file(path) != evidence["energy_artifact_sha256"]:
        raise ValueError("Frozen energy-artifact file hash mismatch.")
    artifact = _load_object(path)
    if core.artifact_content_sha256(artifact) != artifact.get("content_sha256"):
        raise ValueError("Frozen energy-artifact content seal mismatch.")
    if artifact.get("content_sha256") != evidence["energy_artifact_content_sha256"]:
        raise ValueError("Frozen energy-artifact content hash mismatch.")
    if artifact.get("artifact_type") != "route1-freesolv-reserve-label-free-energy":
        raise ValueError("Unexpected energy-artifact type.")
    if artifact.get("source_manifest_sha256") != evidence["source_manifest_sha256"]:
        raise ValueError("Energy artifact is bound to another source manifest.")
    if artifact.get("case_count") != evidence["expected_case_count"]:
        raise ValueError("Energy-artifact case count mismatch.")
    if tuple(artifact.get("endpoint_names", ())) != ENDPOINTS:
        raise ValueError("Energy-artifact endpoint declaration changed.")

    records = artifact.get("records", [])
    if not isinstance(records, list):
        raise ValueError("Energy artifact lacks records.")
    ids = [str(row.get("compound_id", "")) for row in records]
    if len(ids) != len(set(ids)) or set(ids) != source_ids:
        raise ValueError("Energy-artifact IDs do not match the source manifest.")
    rows: dict[str, dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Malformed energy record.")
        compound_id = str(row["compound_id"])
        predictions = row.get("predictions_kcal_mol")
        if not isinstance(predictions, dict) or set(predictions) != set(ENDPOINTS):
            raise ValueError(f"Energy endpoint membership mismatch for {compound_id}.")
        for endpoint in ENDPOINTS:
            _finite_number(
                predictions[endpoint],
                field=f"energy[{compound_id}].{endpoint}",
            )
        rows[compound_id] = row
    return artifact, rows


def _load_score_artifact(
    protocol: Mapping[str, Any],
    path: Path,
    source_ids: set[str],
    energy: Mapping[str, Any],
    energy_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    evidence = protocol["source_evidence"]
    if core.sha256_file(path) != evidence["score_artifact_sha256"]:
        raise ValueError("Frozen score-artifact file hash mismatch.")
    artifact = _load_object(path)
    if core.artifact_content_sha256(artifact) != artifact.get("content_sha256"):
        raise ValueError("Frozen score-artifact content seal mismatch.")
    if artifact.get("content_sha256") != evidence["score_artifact_content_sha256"]:
        raise ValueError("Frozen score-artifact content hash mismatch.")
    if artifact.get("artifact_type") != "route1-freesolv-label-exposed-reserve-score":
        raise ValueError("Unexpected score-artifact type.")
    if artifact.get("case_count") != evidence["expected_case_count"]:
        raise ValueError("Score-artifact case count mismatch.")
    if artifact.get("source_manifest_sha256") != evidence["source_manifest_sha256"]:
        raise ValueError("Score artifact is bound to another source manifest.")
    if artifact.get("sealed_energy_artifact_sha256") != evidence[
        "energy_artifact_sha256"
    ]:
        raise ValueError("Score artifact is bound to another energy file.")
    if artifact.get("sealed_energy_content_sha256") != energy.get("content_sha256"):
        raise ValueError("Score artifact is bound to another energy content seal.")

    records = artifact.get("records", [])
    if not isinstance(records, list):
        raise ValueError("Score artifact lacks records.")
    ids = [str(row.get("compound_id", "")) for row in records]
    if len(ids) != len(set(ids)) or set(ids) != source_ids:
        raise ValueError("Score-artifact IDs do not match the source manifest.")
    rows: dict[str, dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Malformed score record.")
        compound_id = str(row["compound_id"])
        experimental = _finite_number(
            row.get("experimental_kcal_mol"),
            field=f"score[{compound_id}].experimental_kcal_mol",
        )
        uncertainty = _finite_number(
            row.get("experimental_uncertainty_kcal_mol"),
            field=f"score[{compound_id}].experimental_uncertainty_kcal_mol",
        )
        if uncertainty < 0.0:
            raise ValueError(f"Negative experimental uncertainty for {compound_id}.")
        predictions = row.get("predictions_kcal_mol")
        if not isinstance(predictions, dict) or set(predictions) != set(ENDPOINTS):
            raise ValueError(f"Score endpoint membership mismatch for {compound_id}.")
        energy_predictions = energy_rows[compound_id]["predictions_kcal_mol"]
        for endpoint in ENDPOINTS:
            prediction = _finite_number(
                predictions[endpoint],
                field=f"score[{compound_id}].{endpoint}",
            )
            if prediction != float(energy_predictions[endpoint]):
                raise ValueError(
                    f"Score/energy prediction mismatch for {compound_id}, {endpoint}."
                )
        rows[compound_id] = {
            "experimental": experimental,
            "uncertainty": uncertainty,
            "predictions": {
                endpoint: float(predictions[endpoint]) for endpoint in ENDPOINTS
            },
        }
    return artifact, rows


def _load_pair_artifact(
    protocol: Mapping[str, Any],
    pair_protocol_path: Path,
    pair_artifact_path: Path,
    source_ids: set[str],
    energy_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evidence = protocol["source_evidence"]
    if core.sha256_file(pair_protocol_path) != evidence["pair_protocol_sha256"]:
        raise ValueError("Frozen pair-protocol hash mismatch.")
    pair_protocol = _load_object(pair_protocol_path)
    if pair_protocol.get("protocol_id") != "maple-route1-charge-continuity-pairs-v1":
        raise ValueError("Unexpected label-free pair protocol id.")
    if pair_protocol.get("route1_boundary") != ROUTE1_BOUNDARY:
        raise ValueError("Pair protocol violates the Route 1 boundary.")
    expected_pair_design = {
        "historical_label_exposure": True,
        "this_phase_reads_experimental_labels": False,
        "this_phase_reads_score_artifact": False,
        "this_phase_reads_series_artifact": False,
        "no_fit": True,
        "no_residual": True,
        "no_endpoint_selection": True,
        "no_threshold_tuning": True,
        "certified_ranking_available": False,
    }
    if pair_protocol.get("evaluation_design") != expected_pair_design:
        raise ValueError("Pair protocol weakens the label-free phase.")
    if tuple(pair_protocol.get("endpoints", ())) != ENDPOINTS:
        raise ValueError("Pair protocol endpoint declaration changed.")
    expected_pair_rule = {
        "metric_variant_structural_mapping_is_evaluable": False,
        "endpoint_selection_allowed": False,
        "charge_method_selection_allowed": False,
        "energy_correction_allowed": False,
        "abstention_threshold_selection_allowed": False,
        "certified_order_allowed": False,
        "failure_action": (
            "Report the label-free sidecar only; do not infer accuracy or "
            "tune a decision rule."
        ),
    }
    if pair_protocol.get("pre_registered_decision_rule") != expected_pair_rule:
        raise ValueError("Pair protocol permits an unsupported decision.")
    pair_audit_design = pair_protocol.get("pair_audit_design", {})
    if any(
        pair_audit_design.get(key) != value
        for key, value in PAIR_MAPPING_BOUNDARY.items()
    ):
        raise ValueError("Pair protocol weakens the declared mapping boundary.")
    pair_source = pair_protocol.get("source_evidence", {})
    if (
        pair_source.get("source_manifest_sha256")
        != evidence["source_manifest_sha256"]
        or pair_source.get("energy_artifact_sha256")
        != evidence["energy_artifact_sha256"]
        or pair_source.get("energy_artifact_content_sha256")
        != evidence["energy_artifact_content_sha256"]
        or pair_source.get("expected_case_count") != evidence["expected_case_count"]
    ):
        raise ValueError("Pair protocol is bound to different source evidence.")
    pair_fingerprint = core.sha256_bytes(core.canonical_json_bytes(pair_protocol))

    if core.sha256_file(pair_artifact_path) != evidence["pair_artifact_sha256"]:
        raise ValueError("Frozen pair-artifact file hash mismatch.")
    artifact = _load_object(pair_artifact_path)
    if core.artifact_content_sha256(artifact) != artifact.get("content_sha256"):
        raise ValueError("Frozen pair-artifact content seal mismatch.")
    if artifact.get("content_sha256") != evidence["pair_artifact_content_sha256"]:
        raise ValueError("Frozen pair-artifact content hash mismatch.")
    if artifact.get("artifact_type") != PAIR_ARTIFACT_TYPE:
        raise ValueError("Unexpected pair-artifact type.")
    if artifact.get("protocol_id") != pair_protocol["protocol_id"]:
        raise ValueError("Pair artifact protocol id mismatch.")
    if artifact.get("protocol_sha256") != evidence["pair_protocol_sha256"]:
        raise ValueError("Pair artifact protocol file hash mismatch.")
    if artifact.get("protocol_fingerprint") != pair_fingerprint:
        raise ValueError("Pair artifact protocol fingerprint mismatch.")
    if artifact.get("route1_boundary") != ROUTE1_BOUNDARY:
        raise ValueError("Pair artifact violates the Route 1 boundary.")
    if artifact.get("pair_audit_design") != pair_protocol.get(
        "pair_audit_design"
    ):
        raise ValueError("Pair artifact audit design differs from its protocol.")
    if artifact.get("source_manifest_sha256") != evidence["source_manifest_sha256"]:
        raise ValueError("Pair artifact is bound to another source manifest.")
    if artifact.get("energy_artifact_sha256") != evidence["energy_artifact_sha256"]:
        raise ValueError("Pair artifact is bound to another energy file.")
    if artifact.get("energy_artifact_content_sha256") != evidence[
        "energy_artifact_content_sha256"
    ]:
        raise ValueError("Pair artifact is bound to another energy content seal.")
    if artifact.get("case_count") != evidence["expected_case_count"]:
        raise ValueError("Pair-artifact case count mismatch.")
    if artifact.get("evaluable_pair_count") != evidence["expected_pair_count"]:
        raise ValueError("Pair-artifact evaluable-pair count mismatch.")
    expected_label_boundary = {
        "experimental_labels_read": False,
        "score_artifact_read": False,
        "series_artifact_read": False,
        "historical_label_exposure": True,
    }
    if artifact.get("label_boundary") != expected_label_boundary:
        raise ValueError("Pair artifact crossed the experimental-label boundary.")
    expected_pair_decision = {
        "accuracy_claim_available": False,
        "charge_model_comparison_available": False,
        "threshold_selection_allowed": False,
        "endpoint_selection_allowed": False,
        "certified_ranking_available": False,
        "status": "label_free_charge_continuity_sidecar_only",
    }
    if artifact.get("decision") != expected_pair_decision:
        raise ValueError("Pair artifact permits an unsupported decision.")
    _reject_fields(artifact, _FORBIDDEN_PAIR_FIELDS)

    components = artifact.get("components", [])
    if not isinstance(components, list):
        raise ValueError("Pair artifact lacks graph components.")
    if artifact.get("component_count") != len(components):
        raise ValueError("Pair-artifact component count mismatch.")
    component_members: dict[str, set[str]] = {}
    declared_edge_counts: dict[str, int] = {}
    declared_pair_hashes: dict[str, str] = {}
    member_owner: dict[str, str] = {}
    for component in components:
        component_id = str(component.get("component_id", ""))
        members = [str(value) for value in component.get("member_ids", ())]
        if (
            not component_id
            or len(members) < 2
            or members != sorted(members)
            or len(members) != len(set(members))
            or not set(members).issubset(source_ids)
        ):
            raise ValueError("Malformed pair graph component.")
        if component_id in component_members:
            raise ValueError("Duplicate pair graph component id.")
        if component.get("member_count") != len(members):
            raise ValueError("Pair graph component member count mismatch.")
        expected_component_id = "strict-mmp-component-v1:" + core.sha256_bytes(
            core.canonical_json_bytes(members)
        )
        if component_id != expected_component_id:
            raise ValueError("Pair graph component id mismatch.")
        for member in members:
            if member in member_owner:
                raise ValueError("A molecule appears in multiple pair components.")
            member_owner[member] = component_id
        edge_count = component.get("edge_count")
        if isinstance(edge_count, bool) or not isinstance(edge_count, int):
            raise ValueError("Pair graph component edge count is not an integer.")
        pair_ids_sha256 = str(component.get("pair_ids_sha256", ""))
        _require_sha256(
            pair_ids_sha256,
            field=f"component[{component_id}].pair_ids_sha256",
        )
        component_members[component_id] = set(members)
        declared_edge_counts[component_id] = edge_count
        declared_pair_hashes[component_id] = pair_ids_sha256

    pairs = artifact.get("pairs", [])
    if not isinstance(pairs, list) or len(pairs) != evidence["expected_pair_count"]:
        raise ValueError("Pair-artifact record count mismatch.")
    pair_ids: set[str] = set()
    pair_members: set[tuple[str, str]] = set()
    observed_edge_counts = {component_id: 0 for component_id in component_members}
    observed_pair_ids = {component_id: [] for component_id in component_members}
    adjacency = {
        component_id: {member: set() for member in members}
        for component_id, members in component_members.items()
    }
    normalized_pairs: list[dict[str, Any]] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("Malformed pair record.")
        pair_id = str(pair.get("pair_id", ""))
        first_id = str(pair.get("first_id", ""))
        second_id = str(pair.get("second_id", ""))
        component_id = str(pair.get("component_id", ""))
        if (
            not pair_id
            or pair_id in pair_ids
            or first_id >= second_id
            or first_id not in source_ids
            or second_id not in source_ids
        ):
            raise ValueError("Malformed or duplicate strict-MMP pair identity.")
        if pair_id != _pair_id(first_id, second_id):
            raise ValueError("Derived strict-MMP pair id mismatch.")
        member_pair = (first_id, second_id)
        if member_pair in pair_members:
            raise ValueError("Duplicate strict-MMP molecule pair.")
        if component_id not in component_members or not {
            first_id,
            second_id,
        }.issubset(component_members[component_id]):
            raise ValueError("Strict-MMP pair/component membership mismatch.")
        pair_ids.add(pair_id)
        pair_members.add(member_pair)
        observed_edge_counts[component_id] += 1
        observed_pair_ids[component_id].append(pair_id)
        adjacency[component_id][first_id].add(second_id)
        adjacency[component_id][second_id].add(first_id)

        charge_values = {
            metric: _finite_number(pair.get(metric), field=f"{pair_id}.{metric}")
            for metric in CHARGE_METRICS
        }
        if any(value < 0.0 for value in charge_values.values()):
            raise ValueError(f"Negative charge-drift metric for {pair_id}.")
        predicted_delta = pair.get("predicted_delta_kcal_mol")
        if not isinstance(predicted_delta, dict) or set(predicted_delta) != set(
            ENDPOINTS
        ):
            raise ValueError(f"Pair endpoint membership mismatch for {pair_id}.")
        expected_delta = {
            endpoint: float(
                energy_rows[first_id]["predictions_kcal_mol"][endpoint]
            )
            - float(energy_rows[second_id]["predictions_kcal_mol"][endpoint])
            for endpoint in ENDPOINTS
        }
        for endpoint in ENDPOINTS:
            value = _finite_number(
                predicted_delta[endpoint],
                field=f"{pair_id}.{endpoint}",
            )
            if value != expected_delta[endpoint]:
                raise ValueError(f"Pair/energy delta mismatch for {pair_id}, {endpoint}.")
        disagreement = len({_sign(value) for value in expected_delta.values()}) > 1
        if pair.get("endpoint_order_disagreement") is not disagreement:
            raise ValueError(f"Endpoint disagreement flag mismatch for {pair_id}.")
        normalized_pairs.append(
            {
                "pair_id": pair_id,
                "first_id": first_id,
                "second_id": second_id,
                "component_id": component_id,
                "charge_metrics": charge_values,
                "predicted_delta": expected_delta,
                "endpoint_order_disagreement": disagreement,
            }
        )
    if observed_edge_counts != declared_edge_counts:
        raise ValueError("Pair graph component edge counts do not reconcile.")
    for component_id, members in component_members.items():
        expected_pair_hash = core.sha256_bytes(
            core.canonical_json_bytes(sorted(observed_pair_ids[component_id]))
        )
        if declared_pair_hashes[component_id] != expected_pair_hash:
            raise ValueError("Pair graph component pair-id hash mismatch.")
        start = min(members)
        visited = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[component_id][current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if visited != members:
            raise ValueError("Pair graph component is disconnected.")
    if artifact.get("endpoint_order_disagreement_count") != sum(
        pair["endpoint_order_disagreement"] for pair in normalized_pairs
    ):
        raise ValueError("Pair-artifact endpoint-disagreement count mismatch.")
    return artifact, normalized_pairs


def _sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def _pair_id(first_id: str, second_id: str) -> str:
    digest = core.sha256_bytes(core.canonical_json_bytes([first_id, second_id]))
    return f"strict-mmp-v1:{digest}"


def _gate_key(threshold: float) -> str:
    return (
        f"fixed_abs_delta_gte_{threshold:.1f}"
        if float(threshold).is_integer()
        else f"fixed_abs_delta_gte_{threshold:g}"
    )


def _diagnostic_rows(
    pairs: Sequence[Mapping[str, Any]],
    score_rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        first = score_rows[str(pair["first_id"])]
        second = score_rows[str(pair["second_id"])]
        experimental_delta = float(first["experimental"]) - float(
            second["experimental"]
        )
        uncertainty_limit = math.sqrt(
            float(first["uncertainty"]) ** 2 + float(second["uncertainty"]) ** 2
        )
        endpoint_rows: dict[str, dict[str, Any]] = {}
        for endpoint in ENDPOINTS:
            predicted_delta = float(pair["predicted_delta"][endpoint])
            error = predicted_delta - experimental_delta
            endpoint_rows[endpoint] = {
                "predicted_delta": predicted_delta,
                "absolute_error": abs(error),
                "squared_error": error * error,
                "predicted_sign": _sign(predicted_delta),
                "correct": _sign(predicted_delta) == _sign(experimental_delta),
            }
        rows.append(
            {
                "experimental_delta": experimental_delta,
                "uncertainty_limit": uncertainty_limit,
                "charge_metrics": pair["charge_metrics"],
                "endpoint_order_disagreement": pair[
                    "endpoint_order_disagreement"
                ],
                "endpoints": endpoint_rows,
            }
        )
    return rows


def _association(
    rows: Sequence[Mapping[str, Any]], endpoint: str, charge_metric: str
) -> float | None:
    if len(rows) < 2:
        return None
    return ranking_metrics.spearman_rho(
        [float(row["charge_metrics"][charge_metric]) for row in rows],
        [float(row["endpoints"][endpoint]["absolute_error"]) for row in rows],
    )


def _endpoint_summary(
    rows: Sequence[Mapping[str, Any]], endpoint: str
) -> dict[str, Any]:
    count = len(rows)
    if not count:
        return {
            "pair_count": 0,
            "correct_count": 0,
            "wrong_count": 0,
            "predicted_tie_count": 0,
            "forced_sign_accuracy": None,
            "answered_sign_accuracy": None,
            "delta_delta_mae_kcal_mol": None,
            "delta_delta_rmse_kcal_mol": None,
            "descriptive_spearman_charge_drift_vs_absolute_delta_delta_error": {
                metric: None for metric in CHARGE_METRICS
            },
        }
    endpoint_rows = [row["endpoints"][endpoint] for row in rows]
    correct = sum(bool(row["correct"]) for row in endpoint_rows)
    predicted_ties = sum(int(row["predicted_sign"]) == 0 for row in endpoint_rows)
    wrong = count - correct - predicted_ties
    answered = correct + wrong
    absolute_errors = [float(row["absolute_error"]) for row in endpoint_rows]
    squared_errors = [float(row["squared_error"]) for row in endpoint_rows]
    return {
        "pair_count": count,
        "correct_count": correct,
        "wrong_count": wrong,
        "predicted_tie_count": predicted_ties,
        "forced_sign_accuracy": float(correct / count),
        "answered_sign_accuracy": float(correct / answered) if answered else None,
        "delta_delta_mae_kcal_mol": float(statistics.fmean(absolute_errors)),
        "delta_delta_rmse_kcal_mol": math.sqrt(
            float(statistics.fmean(squared_errors))
        ),
        "descriptive_spearman_charge_drift_vs_absolute_delta_delta_error": {
            metric: _association(rows, endpoint, metric)
            for metric in CHARGE_METRICS
        },
    }


def _gate_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    disagreement = sum(bool(row["endpoint_order_disagreement"]) for row in rows)
    cross_tab = {
        "both_correct": 0,
        "both_wrong_or_tied": 0,
        "baseline_only_correct": 0,
        "candidate_only_correct": 0,
    }
    baseline, candidate = ENDPOINTS
    for row in rows:
        baseline_correct = bool(row["endpoints"][baseline]["correct"])
        candidate_correct = bool(row["endpoints"][candidate]["correct"])
        if baseline_correct and candidate_correct:
            cross_tab["both_correct"] += 1
        elif baseline_correct:
            cross_tab["baseline_only_correct"] += 1
        elif candidate_correct:
            cross_tab["candidate_only_correct"] += 1
        else:
            cross_tab["both_wrong_or_tied"] += 1
    return {
        "pair_count": len(rows),
        "endpoint_order_disagreement_count": disagreement,
        "endpoint_order_disagreement_fraction": (
            float(disagreement / len(rows)) if rows else None
        ),
        "endpoint_correctness_cross_tab": cross_tab,
        "methods": {
            endpoint: _endpoint_summary(rows, endpoint) for endpoint in ENDPOINTS
        },
    }


def _metric_summaries(
    pairs: Sequence[Mapping[str, Any]],
    score_rows: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    rows = _diagnostic_rows(pairs, score_rows)
    summaries = {
        "all_experimentally_distinct": _gate_summary(
            [row for row in rows if float(row["experimental_delta"]) != 0.0]
        )
    }
    statistics_config = protocol["diagnostic_statistics"]
    for threshold in statistics_config[
        "fixed_experimental_delta_thresholds_kcal_mol"
    ]:
        numeric = float(threshold)
        summaries[_gate_key(numeric)] = _gate_summary(
            [
                row
                for row in rows
                if abs(float(row["experimental_delta"])) >= numeric
            ]
        )
    z_value = float(statistics_config["experimental_uncertainty_z"])
    summaries[f"uncertainty_z_{z_value:g}"] = _gate_summary(
        [
            row
            for row in rows
            if abs(float(row["experimental_delta"]))
            > z_value * float(row["uncertainty_limit"])
        ]
    )
    return summaries


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = Path(args.protocol).resolve()
    source_manifest_path = Path(args.source_manifest).resolve()
    energy_artifact_path = Path(args.energy_artifact).resolve()
    score_artifact_path = Path(args.score_artifact).resolve()
    pair_protocol_path = Path(args.pair_protocol).resolve()
    pair_artifact_path = Path(args.pair_artifact).resolve()
    output = Path(args.output).resolve()

    protocol, fingerprint = load_protocol(protocol_path)
    _manifest, source_ids = _load_source_manifest(
        protocol, source_manifest_path
    )
    energy, energy_rows = _load_energy_artifact(
        protocol, energy_artifact_path, source_ids
    )
    score, score_rows = _load_score_artifact(
        protocol,
        score_artifact_path,
        source_ids,
        energy,
        energy_rows,
    )
    pair_artifact, pairs = _load_pair_artifact(
        protocol,
        pair_protocol_path,
        pair_artifact_path,
        source_ids,
        energy_rows,
    )
    summaries = _metric_summaries(pairs, score_rows, protocol)
    rule = protocol["pre_registered_decision_rule"]
    artifact = core.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": ARTIFACT_TYPE,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": core.sha256_file(protocol_path),
            "protocol_fingerprint": fingerprint,
            "claim_scope": protocol["claim_scope"],
            "route1_boundary": ROUTE1_BOUNDARY,
            "evaluation_design": protocol["evaluation_design"],
            "pair_mapping_boundary": protocol["pair_mapping_boundary"],
            "provenance_chain": {
                "source_manifest_sha256": core.sha256_file(
                    source_manifest_path
                ),
                "sealed_energy_artifact_sha256": core.sha256_file(
                    energy_artifact_path
                ),
                "sealed_energy_content_sha256": energy["content_sha256"],
                "sealed_score_artifact_sha256": core.sha256_file(
                    score_artifact_path
                ),
                "sealed_score_content_sha256": score["content_sha256"],
                "score_bound_energy_artifact_sha256": score[
                    "sealed_energy_artifact_sha256"
                ],
                "score_bound_energy_content_sha256": score[
                    "sealed_energy_content_sha256"
                ],
                "pair_protocol_sha256": core.sha256_file(pair_protocol_path),
                "sealed_pair_artifact_sha256": core.sha256_file(
                    pair_artifact_path
                ),
                "sealed_pair_content_sha256": pair_artifact["content_sha256"],
            },
            "case_count": int(protocol["source_evidence"]["expected_case_count"]),
            "candidate_pair_count": pair_artifact["candidate_pair_count"],
            "evaluable_pair_count": len(pairs),
            "component_count": pair_artifact["component_count"],
            "component_size_summary": [
                {
                    "member_count": component["member_count"],
                    "edge_count": component["edge_count"],
                }
                for component in pair_artifact["components"]
            ],
            "pair_dependence_boundary": {
                "pair_edges_share_molecules": True,
                "pair_edges_share_graph_components": True,
                "pairs_treated_as_independent_samples": False,
                "p_values_reported": False,
                "confidence_intervals_reported": False,
                "causal_attribution_available": False,
            },
            "fixed_diagnostic_gates": {
                "fixed_experimental_delta_thresholds_kcal_mol": protocol[
                    "diagnostic_statistics"
                ]["fixed_experimental_delta_thresholds_kcal_mol"],
                "experimental_uncertainty_z": protocol["diagnostic_statistics"][
                    "experimental_uncertainty_z"
                ],
            },
            "diagnostic_summaries": summaries,
            "decision": {
                "status": rule["conclusion"],
                "charge_model_comparison_available": False,
                "endpoint_selection_allowed": rule["endpoint_selection_allowed"],
                "charge_method_selection_allowed": rule[
                    "charge_method_selection_allowed"
                ],
                "energy_correction_allowed": rule["energy_correction_allowed"],
                "abstention_threshold_selection_allowed": rule[
                    "abstention_threshold_selection_allowed"
                ],
                "certified_ranking_available": False,
                "failure_action": rule["failure_action"],
            },
            "command_provenance": core.command_provenance(
                __file__,
                {
                    "phase": "label-exposed-charge-continuity-diagnostic",
                    "protocol": _relative_to_repository(protocol_path),
                    "source_manifest": _relative_to_repository(
                        source_manifest_path
                    ),
                    "energy_artifact": _relative_to_repository(
                        energy_artifact_path
                    ),
                    "score_artifact": _relative_to_repository(
                        score_artifact_path
                    ),
                    "pair_protocol": _relative_to_repository(
                        pair_protocol_path
                    ),
                    "pair_artifact": _relative_to_repository(
                        pair_artifact_path
                    ),
                    "output": _relative_to_repository(output),
                },
                repository_root=REPOSITORY_ROOT,
            ),
        }
    )
    _reject_fields(artifact, _FORBIDDEN_OUTPUT_FIELDS)
    core.write_json_atomic(output, artifact)
    print(
        json.dumps(
            {
                "output": str(output),
                "file_sha256": core.sha256_file(output),
                "content_sha256": artifact["content_sha256"],
                "evaluable_pair_count": artifact["evaluable_pair_count"],
                "component_count": artifact["component_count"],
                "decision": artifact["decision"]["status"],
            },
            indent=2,
        )
    )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST
    )
    parser.add_argument(
        "--energy-artifact", type=Path, default=DEFAULT_ENERGY_ARTIFACT
    )
    parser.add_argument(
        "--score-artifact", type=Path, default=DEFAULT_SCORE_ARTIFACT
    )
    parser.add_argument(
        "--pair-protocol", type=Path, default=DEFAULT_PAIR_PROTOCOL
    )
    parser.add_argument(
        "--pair-artifact", type=Path, default=DEFAULT_PAIR_ARTIFACT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
