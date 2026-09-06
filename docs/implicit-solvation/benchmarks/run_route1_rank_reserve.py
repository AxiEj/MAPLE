#!/usr/bin/env python3
"""Score frozen Route 1 energies with series-level ranking diagnostics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # noqa: E402
import ranking_metrics  # noqa: E402


DEFAULT_PROTOCOL = SCRIPT_DIR / "route1_rank_protocol_v1.json"
DEFAULT_SOURCE_MANIFEST = SCRIPT_DIR / "route1_freesolv_reserve_source_manifest.json"
DEFAULT_SCORE_ARTIFACT = SCRIPT_DIR / "route1-freesolv-reserve-score-2026-07-25.json"
DEFAULT_SERIES_ARTIFACT = (
    SCRIPT_DIR / "route1-freesolv-reserve-rank-series-2026-07-29.json"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "route1-freesolv-reserve-rank-diagnostic-2026-07-29.json"
)

ENDPOINTS = (
    "am1bcc_obc2_ace",
    "am1bcc_chagb_pbsa_cavity_dispersion",
)
_ROUTE1_BOUNDARY = {
    "name": "Additive fixed-charge PB/GB implicit solvation",
    "formula": "E_solution(R)=E_MLIP,gas(R)+G_polar(R,q_fixed)+G_nonpolar(R)",
    "gas_phase_mm_energy": False,
    "hydration_label_residual": False,
    "mlip_retraining": False,
    "fixed_charge": "AM1-BCC",
}
_FORBIDDEN_LABEL_FIELDS = frozenset(
    {
        "experimental_kcal_mol",
        "experimental_uncertainty_kcal_mol",
        "signed_errors_kcal_mol",
        "predictions_kcal_mol",
    }
)


def _relative_to_repository(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _raw_protocol(path: str | Path) -> dict[str, Any]:
    protocol = core.load_json(path)
    if protocol.get("schema_version") != 1:
        raise ValueError("Only Route 1 rank protocol schema version 1 is supported.")
    if protocol.get("protocol_id") != "maple-route1-rank-contract-v1":
        raise ValueError("Unexpected Route 1 ranking protocol id.")
    if protocol.get("route1_boundary") != _ROUTE1_BOUNDARY:
        raise ValueError("Ranking protocol violates the Route 1 boundary.")
    design = protocol.get("evaluation_design", {})
    required_design = {
        "historical_label_exposure": True,
        "energy_phase_reads_labels": False,
        "score_phase_reads_labels": True,
        "series_assignment_reads_labels": False,
        "series_disjoint_split": False,
        "congeneric_or_target_series_proven": False,
        "independent_blind_confirmation": False,
        "certified_ranking_available": False,
        "no_fit": True,
        "no_residual": True,
        "no_endpoint_selection": True,
        "reserve_tuning_allowed": False,
    }
    if any(design.get(key) is not value for key, value in required_design.items()):
        raise ValueError("Ranking protocol does not preserve its evidence boundary.")
    if tuple(protocol.get("endpoints", {})) != ENDPOINTS:
        raise ValueError("Ranking endpoint order or membership changed.")
    evidence = protocol.get("source_evidence", {})
    for key in ("source_manifest_sha256", "score_artifact_sha256"):
        value = str(evidence.get(key, ""))
        if len(value) != 64 or set(value) - set("0123456789abcdef"):
            raise ValueError(f"Ranking protocol requires a pinned {key}.")
    if int(evidence.get("expected_case_count", -1)) <= 0:
        raise ValueError("Ranking protocol requires a positive expected case count.")
    derivation = protocol.get("series_derivation", {})
    if derivation.get("kind") != "rdkit-bemis-murcko-from-pinned-mol2-v1":
        raise ValueError("Unexpected ranking series derivation.")
    if int(derivation.get("min_series_size", 0)) < 2:
        raise ValueError("Ranking protocol requires at least two members per series.")
    statistics = protocol.get("ranking_statistics", {})
    if statistics.get("lower_value_is_better") is not True:
        raise ValueError("Route 1 free-energy ranking must use lower-is-better order.")
    if statistics.get("bootstrap_unit") != "complete_series":
        raise ValueError("Ranking bootstrap must resample complete series.")
    if int(statistics.get("bootstrap_resamples", 0)) <= 0:
        raise ValueError("Ranking bootstrap requires positive resamples.")
    confidence = float(statistics.get("bootstrap_confidence", 0.0))
    if not 0.0 < confidence < 1.0:
        raise ValueError("Ranking bootstrap confidence must be in (0, 1).")
    rule = protocol.get("pre_registered_decision_rule", {})
    expected_rule = {
        "endpoint_selection_allowed": False,
        "reserve_tuning_allowed": False,
        "energy_correction_allowed": False,
        "certified_order_allowed": False,
    }
    if any(rule.get(key) is not value for key, value in expected_rule.items()):
        raise ValueError("Ranking protocol permits an unsupported post-score action.")
    return protocol


def load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = _raw_protocol(path)
    return protocol, core.sha256_bytes(core.canonical_json_bytes(protocol))


def _reject_label_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_LABEL_FIELDS:
                raise ValueError(f"Label-free ranking artifact contains label field {key!r}.")
            _reject_label_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_label_fields(nested)


def _load_source_manifest(protocol: Mapping[str, Any], path: Path) -> dict[str, Any]:
    evidence = protocol["source_evidence"]
    if core.sha256_file(path) != evidence["source_manifest_sha256"]:
        raise ValueError("Frozen ranking source-manifest hash mismatch.")
    manifest = core.load_json(path)
    _reject_label_fields(manifest)
    records = manifest.get("records", [])
    expected = int(evidence["expected_case_count"])
    ids = [str(row.get("compound_id", "")) for row in records]
    if len(records) != expected or ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("Frozen ranking source manifest is incomplete or unordered.")
    return manifest


def _load_score_artifact(protocol: Mapping[str, Any], path: Path) -> dict[str, Any]:
    evidence = protocol["source_evidence"]
    if core.sha256_file(path) != evidence["score_artifact_sha256"]:
        raise ValueError("Frozen ranking score-artifact hash mismatch.")
    artifact = core.load_json(path)
    if core.artifact_content_sha256(artifact) != artifact.get("content_sha256"):
        raise ValueError("Frozen ranking score artifact content hash mismatch.")
    if artifact.get("case_count") != evidence["expected_case_count"]:
        raise ValueError("Frozen ranking score artifact case count mismatch.")
    for endpoint in ENDPOINTS:
        if endpoint not in artifact.get("methods", {}):
            raise ValueError(f"Frozen ranking score artifact lacks {endpoint}.")
    return artifact


def _load_series_artifact(
    protocol: Mapping[str, Any],
    fingerprint: str,
    path: Path,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = core.load_json(path)
    if core.artifact_content_sha256(artifact) != artifact.get("content_sha256"):
        raise ValueError("Ranking series artifact content hash mismatch.")
    if artifact.get("artifact_type") != "route1-rank-label-free-structure-series":
        raise ValueError("Unexpected ranking series artifact type.")
    if artifact.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("Ranking series artifact protocol id mismatch.")
    if artifact.get("protocol_fingerprint") != fingerprint:
        raise ValueError("Ranking series artifact protocol fingerprint mismatch.")
    if artifact.get("source_manifest_sha256") != protocol["source_evidence"][
        "source_manifest_sha256"
    ]:
        raise ValueError("Ranking series artifact source manifest mismatch.")
    _reject_label_fields(artifact)
    source_ids = {row["compound_id"] for row in source_manifest["records"]}
    records = artifact.get("records", [])
    if {row.get("compound_id") for row in records} != source_ids:
        raise ValueError("Ranking series artifact records do not reconcile with source.")
    member_to_series: dict[str, str] = {}
    for series in artifact.get("series", []):
        members = list(series.get("member_ids", []))
        if len(members) < int(protocol["series_derivation"]["min_series_size"]):
            raise ValueError("Ranking series contains too few members.")
        if len(members) != len(set(members)) or not set(members).issubset(source_ids):
            raise ValueError("Ranking series contains invalid members.")
        for member in members:
            if member in member_to_series:
                raise ValueError("A ranking member appears in multiple series.")
            member_to_series[member] = str(series["series_id"])
    for row in records:
        assigned = row.get("series_id")
        if assigned != member_to_series.get(row["compound_id"]):
            raise ValueError("Ranking series record/group assignment mismatch.")
    return artifact


def _macro_pair_metrics(
    per_series: Mapping[str, Mapping[str, object]]
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    result: dict[str, dict[str, dict[str, float | int | None]]] = {}
    metric_names = ("forced_sign_accuracy", "delta_delta_mae", "delta_delta_rmse")
    for pair_key in sorted(
        {
            key
            for summary in per_series.values()
            for key in summary["pair_metrics"]  # type: ignore[index]
        }
    ):
        result[pair_key] = {}
        for metric in metric_names:
            values = [
                float(summary["pair_metrics"][pair_key][metric])  # type: ignore[index]
                for _series_id, summary in sorted(per_series.items())
                if summary["pair_metrics"][pair_key][metric] is not None  # type: ignore[index]
            ]
            result[pair_key][metric] = {
                "series_count": len(values),
                "value": float(sum(values) / len(values)) if values else None,
            }
    return result


def _methods_from_score(
    score: Mapping[str, Any], series: Mapping[str, Any], statistics: Mapping[str, Any]
) -> tuple[dict[str, dict[str, dict[str, object]]], dict[str, dict[str, object]]]:
    score_rows = {row["compound_id"]: row for row in score["records"]}
    methods: dict[str, dict[str, dict[str, object]]] = {endpoint: {} for endpoint in ENDPOINTS}
    series_metadata: dict[str, dict[str, object]] = {}
    for definition in series["series"]:
        series_id = str(definition["series_id"])
        member_ids = list(definition["member_ids"])
        if any(member not in score_rows for member in member_ids):
            raise ValueError(f"Ranking series {series_id} has no score record.")
        series_metadata[series_id] = {
            "series_kind": definition["series_kind"],
            "member_count": len(member_ids),
            "scaffold_sha256": definition["scaffold_sha256"],
        }
        for endpoint in ENDPOINTS:
            methods[endpoint][series_id] = ranking_metrics.series_ranking_metrics(
                compound_ids=member_ids,
                experimental=[
                    float(score_rows[member]["experimental_kcal_mol"])
                    for member in member_ids
                ],
                experimental_uncertainty=[
                    float(score_rows[member]["experimental_uncertainty_kcal_mol"])
                    for member in member_ids
                ],
                predicted=[
                    float(score_rows[member]["predictions_kcal_mol"][endpoint])
                    for member in member_ids
                ],
                fixed_delta_thresholds=statistics[
                    "fixed_experimental_delta_thresholds_kcal_mol"
                ],
                uncertainty_z=float(statistics["experimental_uncertainty_z"]),
                top_ks=statistics["top_k"],
                confidence_margin_thresholds=statistics[
                    "confidence_margin_thresholds_kcal_mol"
                ],
            )
    return methods, series_metadata


def _method_summary(per_series: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    macro = ranking_metrics.series_macro_summary(per_series)
    return {
        "series_macro": {
            metric: macro[metric]["value"] for metric in ("kendall_tau_b", "spearman_rho")
        },
        "series_macro_metric_series_count": {
            metric: macro[metric]["series_count"]
            for metric in ("kendall_tau_b", "spearman_rho")
        },
        "series_macro_pair_metrics": _macro_pair_metrics(per_series),
        "per_series": per_series,
    }


def score(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_protocol(protocol_path)
    source_path = Path(args.source_manifest).resolve()
    score_path = Path(args.score_artifact).resolve()
    series_path = Path(args.series_artifact).resolve()
    source_manifest = _load_source_manifest(protocol, source_path)
    score_artifact = _load_score_artifact(protocol, score_path)
    series_artifact = _load_series_artifact(
        protocol, fingerprint, series_path, source_manifest
    )
    methods_by_series, series_metadata = _methods_from_score(
        score_artifact, series_artifact, protocol["ranking_statistics"]
    )
    summaries = {
        endpoint: _method_summary(methods_by_series[endpoint]) for endpoint in ENDPOINTS
    }
    statistics = protocol["ranking_statistics"]
    paired = {
        metric: ranking_metrics.paired_series_macro_bootstrap(
            methods_by_series["am1bcc_obc2_ace"],
            methods_by_series["am1bcc_chagb_pbsa_cavity_dispersion"],
            metric=metric,
            resamples=int(statistics["bootstrap_resamples"]),
            confidence=float(statistics["bootstrap_confidence"]),
            seed=int(statistics["bootstrap_seed"]) + index,
        )
        for index, metric in enumerate(("kendall_tau_b", "spearman_rho"))
    }
    output = Path(args.output).resolve()
    rule = protocol["pre_registered_decision_rule"]
    artifact = core.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-freesolv-label-exposed-rank-diagnostic",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": core.sha256_file(protocol_path),
            "protocol_fingerprint": fingerprint,
            "claim_scope": protocol["claim_scope"],
            "route1_contract": protocol["route1_boundary"],
            "evaluation_design": protocol["evaluation_design"],
            "source_manifest_sha256": core.sha256_file(source_path),
            "sealed_score_artifact_sha256": core.sha256_file(score_path),
            "sealed_score_content_sha256": score_artifact["content_sha256"],
            "series_artifact_sha256": core.sha256_file(series_path),
            "series_content_sha256": series_artifact["content_sha256"],
            "case_count": int(protocol["source_evidence"]["expected_case_count"]),
            "evaluable_series_count": series_artifact["evaluable_series_count"],
            "evaluable_member_count": series_artifact["evaluable_member_count"],
            "series_metadata": series_metadata,
            "methods": summaries,
            "paired_series_macro_change": paired,
            "decision": {
                "certified_ranking_available": False,
                "endpoint_selection_allowed": rule["endpoint_selection_allowed"],
                "reserve_tuning_allowed": rule["reserve_tuning_allowed"],
                "energy_correction_allowed": rule["energy_correction_allowed"],
                "failure_action": rule["failure_action"],
                "future_certification_requires": rule["future_certification_requires"],
            },
            "command_provenance": core.command_provenance(
                __file__,
                {
                    "phase": "label-exposed-ranking-diagnostic",
                    "protocol": _relative_to_repository(protocol_path),
                    "source_manifest": _relative_to_repository(source_path),
                    "score_artifact": _relative_to_repository(score_path),
                    "series_artifact": _relative_to_repository(series_path),
                    "output": _relative_to_repository(output),
                },
                repository_root=REPOSITORY_ROOT,
            ),
        }
    )
    if "experimental_kcal_mol" in json.dumps(artifact, sort_keys=True).lower():
        raise AssertionError("Ranking diagnostic leaked individual experimental labels.")
    core.write_json_atomic(output, artifact)
    print(
        json.dumps(
            {
                "rank_diagnostic": str(output),
                "file_sha256": core.sha256_file(output),
                "content_sha256": artifact["content_sha256"],
                "evaluable_series_count": artifact["evaluable_series_count"],
                "series_macro": {
                    endpoint: summaries[endpoint]["series_macro"]
                    for endpoint in ENDPOINTS
                },
            },
            indent=2,
        )
    )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--score-artifact", type=Path, default=DEFAULT_SCORE_ARTIFACT)
    parser.add_argument("--series-artifact", type=Path, default=DEFAULT_SERIES_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    score(build_parser().parse_args())


if __name__ == "__main__":
    main()
