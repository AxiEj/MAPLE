#!/usr/bin/env python3
"""Run the post-v1 polar-scale sensitivity analysis.

This reviewer-informed diagnostic tests whether the raw surface-field
association survives rank control for the frozen CHA polar-energy magnitude and
surface area.  It fits no energy, changes no prediction, and cannot authorize a
new provider.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # noqa: E402
import build_route1_surface_electrostatics as surface  # noqa: E402
import run_route1_surface_electrostatics_diagnostic as raw  # noqa: E402

DEFAULT_PROTOCOL = (
    SCRIPT_DIR / "route1_surface_electrostatics_scale_sensitivity_protocol_v1.json"
)
DEFAULT_SURFACE_PROTOCOL = SCRIPT_DIR / "route1_surface_electrostatics_protocol_v1.json"
DEFAULT_SURFACE_ARTIFACT = (
    SCRIPT_DIR / "route1-freesolv-surface-electrostatics-label-free-2026-07-29.json"
)
DEFAULT_MISMATCH_ARTIFACT = (
    SCRIPT_DIR / "route1-freesolv-explicit-component-diagnostic-2026-07-29.json"
)
DEFAULT_ENERGY_ARTIFACT = (
    SCRIPT_DIR / "route1-chagb-component-attribution-2026-07-25.json"
)
DEFAULT_PREPARED_IDENTITY = (
    REPOSITORY_ROOT
    / ".omx/benchmarks/neutral-water-freesolv-route1-20260723/prepared.json"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR
    / "route1-freesolv-surface-electrostatics-scale-sensitivity-2026-07-29.json"
)

ARTIFACT_TYPE = "route1-freesolv-label-exposed-surface-electrostatics-scale-sensitivity"
PROFILE_ORDER = surface.PROFILE_ORDER
PROXY_ORDER = ("fn2", "phi2")
PRIMARY_METRICS = (
    "partial_y_fn2_given_z_and_area",
    "partial_y_phi2_given_z_and_area",
)
SECONDARY_METRICS = (
    "partial_y_fn2_given_z",
    "partial_y_phi2_given_z",
)


def _load_object(path: Path) -> dict[str, Any]:
    value = core.load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _relative_to_repository(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = _load_object(Path(path).resolve())
    if protocol.get("schema_version") != 1:
        raise ValueError("Only scale-sensitivity protocol schema 1 is supported.")
    if (
        protocol.get("protocol_id")
        != "maple-route1-surface-electrostatics-scale-sensitivity-v1"
    ):
        raise ValueError("Unexpected scale-sensitivity protocol id.")
    if (
        protocol.get("designation")
        != "post-v1 reviewer-informed adversarial sensitivity analysis"
    ):
        raise ValueError("Scale-sensitivity historical designation changed.")
    boundary = protocol.get("historical_boundary", {})
    required_true = (
        "v1_result_was_seen_before_this_design",
        "source_artifacts_contain_historical_experimental_labels",
    )
    required_false = (
        "independent_confirmation",
        "experimental_hydration_values_referenced_by_runner",
        "published_explicit_charging_is_experimental_truth",
        "radius_profile_agreement_is_independent_replication",
    )
    required_no_action = (
        "no_energy_fit",
        "no_residual_correction",
        "no_endpoint_selection",
        "no_radius_selection",
        "no_provider_implementation",
        "no_threshold_tuning",
        "no_first_shell_causal_claim",
        "no_ranking_certification",
    )
    if any(boundary.get(key) is not True for key in required_true):
        raise ValueError("Scale-sensitivity historical exposure was weakened.")
    if any(boundary.get(key) is not False for key in required_false):
        raise ValueError("Scale-sensitivity label/truth boundary changed.")
    if any(boundary.get(key) is not True for key in required_no_action):
        raise ValueError("Scale-sensitivity no-fit/no-claim boundary changed.")

    evidence = protocol.get("source_evidence", {})
    if (
        evidence.get("expected_case_count") != 526
        or evidence.get("source_partition") != "development"
    ):
        raise ValueError("Scale-sensitivity coverage changed.")
    for key in (
        "surface_artifact_sha256",
        "surface_artifact_content_sha256",
        "component_mismatch_artifact_sha256",
        "component_mismatch_artifact_content_sha256",
        "component_energy_artifact_sha256",
        "component_energy_artifact_content_sha256",
        "prepared_identity_artifact_sha256",
    ):
        if not core._is_hex(str(evidence.get(key, "")), 64):
            raise ValueError(f"source_evidence.{key} must be a SHA256.")

    variables = protocol.get("variables", {})
    if variables != {
        "outcome": "Y_i=abs(G_CHA,polar,i-G_published,charging,i)",
        "primary_scale_control": "Z_i=abs(G_CHA,polar,i)",
        "geometry_control": "A_i=surface_area_angstrom2",
        "candidate_proxies": ["Fn2", "Phi2"],
        "radius_profiles": [
            "primary Bondi-family SAS proxy",
            "control mbondi2 SAS proxy",
        ],
    }:
        raise ValueError("Scale-sensitivity variable definitions changed.")

    statistics = protocol.get("statistics", {})
    if statistics.get("primary") != [
        "partial Spearman(Y,Fn2 | rank(Z),rank(A))",
        "partial Spearman(Y,Phi2 | rank(Z),rank(A))",
    ]:
        raise ValueError("Scale-sensitivity primary statistics changed.")
    if statistics.get("secondary") != [
        "partial Spearman(Y,Fn2 | rank(Z))",
        "partial Spearman(Y,Phi2 | rank(Z))",
    ]:
        raise ValueError("Scale-sensitivity secondary statistics changed.")
    if statistics.get("bootstrap_unit") != "structure_group_sha256":
        raise ValueError("Scale-sensitivity bootstrap unit changed.")
    if statistics.get("bootstrap_resamples") != 5000:
        raise ValueError("Scale-sensitivity bootstrap count changed.")
    if statistics.get("bootstrap_seed") != 2718:
        raise ValueError("Scale-sensitivity bootstrap seed changed.")
    if statistics.get("familywise_confidence_level") != 0.95:
        raise ValueError("Scale-sensitivity confidence level changed.")
    if statistics.get("family_size_per_radius_profile") != 2:
        raise ValueError("Scale-sensitivity statistic family changed.")
    if (
        statistics.get("interval")
        != "two-sided percentile interval with Bonferroni tail probability alpha/(2*2)"
    ):
        raise ValueError("Scale-sensitivity interval definition changed.")
    if (
        statistics.get("p_values_reported") is not False
        or statistics.get("predictive_model_fit") is not False
    ):
        raise ValueError("Unsupported inferential or predictive output enabled.")

    decision = protocol.get("decision_rule", {})
    expected_statuses = {
        "both_primary_partial_intervals_include_zero": (
            "scale_association_only_stop_surface_proxy_to_provider_inference"
        ),
        "any_primary_partial_interval_excludes_zero_but_radius_profiles_disagree": (
            "surface_definition_sensitive_or_unresolved"
        ),
        "same_primary_partial_interval_excludes_zero_with_same_sign_in_both_radius_profiles": (
            "candidate_for_new_pre_registered_independent_confirmation_only"
        ),
    }
    if any(decision.get(key) != value for key, value in expected_statuses.items()):
        raise ValueError("Scale-sensitivity decision statuses changed.")
    for key in (
        "energy_correction_allowed",
        "provider_implementation_allowed",
        "endpoint_selection_allowed",
        "radius_selection_allowed",
        "parameter_update_allowed",
        "ranking_certification_allowed",
        "multisolvent_claim_allowed",
        "maximum_error_claim_allowed",
    ):
        if decision.get(key) is not False:
            raise ValueError(f"Unsupported scale-sensitivity action enabled: {key}.")
    return protocol, core.sha256_bytes(core.canonical_json_bytes(protocol))


def _verify_sealed_source(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_content_sha256: str,
    name: str,
) -> dict[str, Any]:
    if core.sha256_file(path) != expected_file_sha256:
        raise ValueError(f"Frozen {name} file hash mismatch.")
    artifact = _load_object(path)
    if artifact.get("content_sha256") != core.artifact_content_sha256(artifact):
        raise ValueError(f"{name} content seal mismatch.")
    if artifact.get("content_sha256") != expected_content_sha256:
        raise ValueError(f"Frozen {name} content fingerprint mismatch.")
    return artifact


def _load_energy_scale(protocol: Mapping[str, Any], path: Path) -> dict[str, float]:
    evidence = protocol["source_evidence"]
    artifact = _verify_sealed_source(
        path,
        expected_file_sha256=evidence["component_energy_artifact_sha256"],
        expected_content_sha256=evidence["component_energy_artifact_content_sha256"],
        name="component-energy artifact",
    )
    records = artifact.get("records")
    expected = int(evidence["expected_case_count"])
    if (
        artifact.get("case_count") != expected
        or not isinstance(records, list)
        or len(records) != expected
    ):
        raise ValueError("Component-energy coverage changed.")
    output: dict[str, float] = {}
    for record in records:
        compound_id = str(record.get("compound_id", ""))
        components = record.get("components_kcal_mol")
        if not compound_id or compound_id in output or not isinstance(components, dict):
            raise ValueError("Invalid component-energy record.")
        value = components.get("chagb_polar")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid CHA polar component for {compound_id}.")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite CHA polar component for {compound_id}.")
        output[compound_id] = abs(value)
    return output


def _load_mismatches(protocol: Mapping[str, Any], path: Path) -> dict[str, float]:
    evidence = protocol["source_evidence"]
    artifact = _verify_sealed_source(
        path,
        expected_file_sha256=evidence["component_mismatch_artifact_sha256"],
        expected_content_sha256=evidence["component_mismatch_artifact_content_sha256"],
        name="component-mismatch artifact",
    )
    records = artifact.get("records")
    expected = int(evidence["expected_case_count"])
    if (
        artifact.get("case_count") != expected
        or not isinstance(records, list)
        or len(records) != expected
    ):
        raise ValueError("Component-mismatch coverage changed.")
    output: dict[str, float] = {}
    for record in records:
        compound_id = str(record.get("compound_id", ""))
        value = record.get("polar_mismatch_kcal_mol")
        if (
            not compound_id
            or compound_id in output
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("Invalid component-mismatch record.")
        output[compound_id] = abs(float(value))
    return output


def _partial_rank(
    outcome: np.ndarray,
    proxy: np.ndarray,
    controls: Sequence[np.ndarray],
) -> float:
    outcome_rank = raw._average_ranks(np.asarray(outcome, dtype=np.float64))
    proxy_rank = raw._average_ranks(np.asarray(proxy, dtype=np.float64))
    design = np.column_stack(
        [
            np.ones(len(outcome_rank), dtype=np.float64),
            *[
                raw._average_ranks(np.asarray(control, dtype=np.float64))
                for control in controls
            ],
        ]
    )
    outcome_residual = (
        outcome_rank - design @ np.linalg.lstsq(design, outcome_rank, rcond=None)[0]
    )
    proxy_residual = (
        proxy_rank - design @ np.linalg.lstsq(design, proxy_rank, rcond=None)[0]
    )
    denominator = math.sqrt(
        float(np.dot(outcome_residual, outcome_residual))
        * float(np.dot(proxy_residual, proxy_residual))
    )
    if denominator == 0.0:
        raise ValueError("Partial rank correlation is undefined.")
    return float(np.dot(outcome_residual, proxy_residual) / denominator)


def _bootstrap_partial_rank(
    outcome: np.ndarray,
    proxy: np.ndarray,
    controls: Sequence[np.ndarray],
    record_counts: np.ndarray,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    all_counts = np.asarray(record_counts)
    if all_counts.ndim != 2 or all_counts.shape[1] != len(outcome) or batch_size < 1:
        raise ValueError("Bootstrap partial-rank batch input is invalid.")
    output = np.empty(all_counts.shape[0], dtype=np.float64)
    for start in range(0, len(all_counts), batch_size):
        stop = min(start + batch_size, len(all_counts))
        batch_counts = all_counts[start:stop]
        counts = np.asarray(batch_counts, dtype=np.float64)
        outcome_rank = raw._bootstrap_rank_matrix(outcome, batch_counts)
        proxy_rank = raw._bootstrap_rank_matrix(proxy, batch_counts)
        control_ranks = [
            raw._bootstrap_rank_matrix(control, batch_counts) for control in controls
        ]
        batch_count, record_count = counts.shape
        design = np.empty(
            (batch_count, record_count, 1 + len(control_ranks)),
            dtype=np.float64,
        )
        design[:, :, 0] = 1.0
        for index, ranks in enumerate(control_ranks, start=1):
            design[:, :, index] = ranks
        normal = np.einsum("bni,bn,bnj->bij", design, counts, design, optimize=True)
        outcome_rhs = np.einsum(
            "bni,bn,bn->bi", design, counts, outcome_rank, optimize=True
        )
        proxy_rhs = np.einsum(
            "bni,bn,bn->bi", design, counts, proxy_rank, optimize=True
        )
        try:
            outcome_beta = np.linalg.solve(normal, outcome_rhs[..., None])[..., 0]
            proxy_beta = np.linalg.solve(normal, proxy_rhs[..., None])[..., 0]
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Bootstrap partial-rank control matrix is singular."
            ) from exc
        outcome_residual = outcome_rank - np.einsum(
            "bni,bi->bn", design, outcome_beta, optimize=True
        )
        proxy_residual = proxy_rank - np.einsum(
            "bni,bi->bn", design, proxy_beta, optimize=True
        )
        numerator = np.sum(counts * outcome_residual * proxy_residual, axis=1)
        denominator = np.sqrt(
            np.sum(counts * outcome_residual**2, axis=1)
            * np.sum(counts * proxy_residual**2, axis=1)
        )
        if np.any(denominator == 0.0):
            raise ValueError("Bootstrap partial rank correlation is undefined.")
        output[start:stop] = numerator / denominator
    return output


def _stratified_spearman(
    compound_ids: Sequence[str],
    outcome: np.ndarray,
    proxy: np.ndarray,
    scale: np.ndarray,
) -> list[dict[str, Any]]:
    order = np.asarray(
        sorted(
            range(len(compound_ids)),
            key=lambda index: (float(scale[index]), compound_ids[index]),
        ),
        dtype=np.int64,
    )
    output: list[dict[str, Any]] = []
    for stratum_index, indices in enumerate(np.array_split(order, 4)):
        output.append(
            {
                "stratum_index": stratum_index,
                "case_count": len(indices),
                "minimum_abs_cha_polar_kcal_mol": float(np.min(scale[indices])),
                "maximum_abs_cha_polar_kcal_mol": float(np.max(scale[indices])),
                "spearman_rho": raw._spearman(outcome[indices], proxy[indices]),
            }
        )
    return output


def _interval_sign(interval: Sequence[float]) -> int:
    lower, upper = map(float, interval)
    if lower > 0.0:
        return 1
    if upper < 0.0:
        return -1
    return 0


def _decision(
    protocol: Mapping[str, Any], results: Mapping[str, Any]
) -> dict[str, Any]:
    proxy_signs: dict[str, dict[str, int]] = {}
    candidate_proxy: str | None = None
    disagreement = False
    for proxy in PROXY_ORDER:
        metric = f"partial_y_{proxy}_given_z_and_area"
        signs = {
            role: _interval_sign(results[role][metric]["bootstrap_interval"])
            for role in PROFILE_ORDER
        }
        proxy_signs[proxy] = signs
        if signs["primary"] != signs["control"]:
            disagreement = disagreement or any(signs.values())
        elif signs["primary"] != 0:
            candidate_proxy = proxy
    if disagreement:
        status = "surface_definition_sensitive_or_unresolved"
    elif candidate_proxy is not None:
        status = "candidate_for_new_pre_registered_independent_confirmation_only"
    else:
        status = "scale_association_only_stop_surface_proxy_to_provider_inference"
    frozen = protocol["decision_rule"]
    return {
        "status": status,
        "primary_partial_interval_signs": proxy_signs,
        "candidate_proxy": (
            candidate_proxy
            if status
            == "candidate_for_new_pre_registered_independent_confirmation_only"
            else None
        ),
        "interpretation": (
            "After rank control for the frozen CHA polar-energy magnitude and "
            "surface area, no stable extra surface-proxy signal remains."
            if status
            == "scale_association_only_stop_surface_proxy_to_provider_inference"
            else "At most a candidate for a new, separately frozen independent confirmation."
        ),
        **{
            key: frozen[key]
            for key in (
                "energy_correction_allowed",
                "provider_implementation_allowed",
                "endpoint_selection_allowed",
                "radius_selection_allowed",
                "parameter_update_allowed",
                "ranking_certification_allowed",
                "multisolvent_claim_allowed",
                "maximum_error_claim_allowed",
            )
        },
    }


def run(
    *,
    protocol_path: Path,
    surface_protocol_path: Path,
    surface_artifact_path: Path,
    mismatch_artifact_path: Path,
    energy_artifact_path: Path,
    prepared_identity_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    protocol, protocol_sha256 = load_protocol(protocol_path)
    evidence = protocol["source_evidence"]
    surface_protocol, surface_protocol_sha256 = surface.load_protocol(
        surface_protocol_path
    )
    if core.sha256_file(surface_artifact_path) != evidence["surface_artifact_sha256"]:
        raise ValueError("Frozen surface artifact file hash mismatch.")
    surface_records = raw._load_surface_artifact(
        surface_protocol, surface_protocol_sha256, surface_artifact_path
    )
    surface_artifact = _load_object(surface_artifact_path)
    if (
        surface_artifact["content_sha256"]
        != evidence["surface_artifact_content_sha256"]
    ):
        raise ValueError("Frozen surface artifact content fingerprint mismatch.")
    identical_radius_count = sum(
        record["radius_profiles"]["primary"]["radius_vector_sha256"]
        == record["radius_profiles"]["control"]["radius_vector_sha256"]
        for record in surface_records
    )
    mismatches = _load_mismatches(protocol, mismatch_artifact_path)
    energy_scale = _load_energy_scale(protocol, energy_artifact_path)
    structure_groups = raw._load_structure_groups(
        surface_protocol, prepared_identity_path, surface_records
    )

    records = sorted(surface_records, key=lambda row: row["compound_id"])
    compound_ids = [str(record["compound_id"]) for record in records]
    if set(compound_ids) != set(mismatches) or set(compound_ids) != set(energy_scale):
        raise ValueError("Scale-sensitivity source compound-ID sets do not match.")
    outcome = np.asarray([mismatches[key] for key in compound_ids])
    scale = np.asarray([energy_scale[key] for key in compound_ids])
    statistics = protocol["statistics"]
    record_counts, bootstrap_design = raw._bootstrap_counts(
        compound_ids,
        structure_groups,
        resamples=int(statistics["bootstrap_resamples"]),
        seed=int(statistics["bootstrap_seed"]),
    )
    alpha = 1.0 - float(statistics["familywise_confidence_level"])
    family_size = int(statistics["family_size_per_radius_profile"])
    lower_probability = alpha / (2.0 * family_size)
    upper_probability = 1.0 - lower_probability

    results: dict[str, Any] = {}
    for role in PROFILE_ORDER:
        descriptors = [
            record["radius_profiles"][role]["descriptors"] for record in records
        ]
        area = np.asarray([row["surface_area_angstrom2"] for row in descriptors])
        proxies = {
            "fn2": np.asarray([row["fn2_e_per_angstrom2"] for row in descriptors]),
            "phi2": np.asarray([row["phi2_e_per_angstrom"] for row in descriptors]),
        }
        role_results: dict[str, Any] = {}
        for proxy_name, proxy in proxies.items():
            for suffix, controls in (
                ("given_z", [scale]),
                ("given_z_and_area", [scale, area]),
            ):
                metric = f"partial_y_{proxy_name}_{suffix}"
                samples = _bootstrap_partial_rank(
                    outcome, proxy, controls, record_counts
                )
                role_results[metric] = {
                    "partial_spearman_rho": _partial_rank(outcome, proxy, controls),
                    "bootstrap_interval": raw._interval(
                        samples, lower_probability, upper_probability
                    ),
                    "bootstrap_interval_type": (
                        "bonferroni-familywise-95-percent-percentile-bootstrap"
                    ),
                }
            role_results[f"quartile_stratified_y_vs_{proxy_name}"] = (
                _stratified_spearman(compound_ids, outcome, proxy, scale)
            )
        results[role] = role_results

    bootstrap_design["interval_probabilities"] = [
        lower_probability,
        upper_probability,
    ]
    decision = _decision(protocol, results)
    arguments = {
        "protocol": _relative_to_repository(protocol_path),
        "surface_protocol": _relative_to_repository(surface_protocol_path),
        "surface_artifact": _relative_to_repository(surface_artifact_path),
        "mismatch_artifact": _relative_to_repository(mismatch_artifact_path),
        "energy_artifact": _relative_to_repository(energy_artifact_path),
        "prepared_identity": _relative_to_repository(prepared_identity_path),
        "output": _relative_to_repository(output_path),
    }
    artifact = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "designation": protocol["designation"],
        "claim_scope": protocol["claim_scope"],
        "source_partition": "development",
        "case_count": len(compound_ids),
        "historical_boundary": protocol["historical_boundary"],
        "variables": protocol["variables"],
        "statistical_design": {
            **statistics,
            "realized_bootstrap": bootstrap_design,
        },
        "scale_sensitivity_results": results,
        "radius_profile_sensitivity_boundary": {
            "identical_radius_vector_count": identical_radius_count,
            "different_radius_vector_count": len(records) - identical_radius_count,
            "identical_radius_vector_fraction": (identical_radius_count / len(records)),
            "interpretation": (
                "Bondi-family and mbondi2 agreement is a sensitivity check, "
                "not independent replication, because most radius vectors "
                "are identical."
            ),
        },
        "decision": decision,
        "source_provenance": {
            "surface_artifact_sha256": core.sha256_file(surface_artifact_path),
            "surface_artifact_content_sha256": surface_artifact["content_sha256"],
            "component_mismatch_artifact_sha256": core.sha256_file(
                mismatch_artifact_path
            ),
            "component_energy_artifact_sha256": core.sha256_file(energy_artifact_path),
            "prepared_identity_artifact_sha256": core.sha256_file(
                prepared_identity_path
            ),
        },
        "limitations": [
            "The v1 result was seen before this reviewer-informed design; this is not independent confirmation.",
            "Partial rank residualization is a descriptive sensitivity statistic and never modifies an energy.",
            "The published explicit charging component is not experimental truth.",
            "Bondi-family and mbondi2 agreement is not independent replication.",
            "No provider, correction, radius, endpoint, or ranking threshold is selected.",
        ],
        "command_provenance": core.command_provenance(
            __file__, arguments, repository_root=REPOSITORY_ROOT
        ),
    }
    core.seal_artifact(artifact)
    core.write_json_atomic(output_path, artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--surface-protocol", type=Path, default=DEFAULT_SURFACE_PROTOCOL
    )
    parser.add_argument(
        "--surface-artifact", type=Path, default=DEFAULT_SURFACE_ARTIFACT
    )
    parser.add_argument(
        "--mismatch-artifact", type=Path, default=DEFAULT_MISMATCH_ARTIFACT
    )
    parser.add_argument("--energy-artifact", type=Path, default=DEFAULT_ENERGY_ARTIFACT)
    parser.add_argument(
        "--prepared-identity", type=Path, default=DEFAULT_PREPARED_IDENTITY
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = run(
        protocol_path=args.protocol.resolve(),
        surface_protocol_path=args.surface_protocol.resolve(),
        surface_artifact_path=args.surface_artifact.resolve(),
        mismatch_artifact_path=args.mismatch_artifact.resolve(),
        energy_artifact_path=args.energy_artifact.resolve(),
        prepared_identity_path=args.prepared_identity.resolve(),
        output_path=args.output.resolve(),
    )
    print(json.dumps(artifact["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
