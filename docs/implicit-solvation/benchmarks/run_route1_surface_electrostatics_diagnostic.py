#!/usr/bin/env python3
"""Join the sealed surface artifact to frozen polar mismatches retrospectively.

This phase is intentionally label-exposed and descriptive.  It cannot change an
energy, choose a radius profile, select an endpoint, fit a correction, or certify
ranking.  The published FreeSolv charging component is an auxiliary fixed-charge
explicit-solvent calculation, not experimental truth.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

DEFAULT_PROTOCOL = SCRIPT_DIR / "route1_surface_electrostatics_protocol_v1.json"
DEFAULT_SURFACE_ARTIFACT = (
    SCRIPT_DIR
    / "route1-freesolv-surface-electrostatics-label-free-2026-07-29.json"
)
DEFAULT_COMPONENT_ARTIFACT = (
    SCRIPT_DIR / "route1-freesolv-explicit-component-diagnostic-2026-07-29.json"
)
DEFAULT_PREPARED_IDENTITY = (
    REPOSITORY_ROOT
    / ".omx/benchmarks/neutral-water-freesolv-route1-20260723/prepared.json"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR
    / "route1-freesolv-surface-electrostatics-diagnostic-2026-07-29.json"
)

ARTIFACT_TYPE = "route1-freesolv-label-exposed-surface-electrostatics-diagnostic"
ASSOCIATION_ORDER = (
    "absolute_cha_polar_mismatch_vs_fn2",
    "absolute_cha_polar_mismatch_vs_phi2",
    "absolute_cha_polar_mismatch_vs_absolute_gamma_n",
    "absolute_cha_polar_mismatch_vs_surface_area",
    "obc2_to_cha_absolute_mismatch_reduction_vs_fn2",
    "obc2_to_cha_absolute_mismatch_reduction_vs_absolute_gamma_n",
)
DESCRIPTOR_FAMILY = ASSOCIATION_ORDER[:4]
ADDITIONAL_FAMILY = ASSOCIATION_ORDER[4:]


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


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number.")
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Rank input must be one-dimensional, finite, and nonempty.")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        value = array[order[start]]
        while stop < len(order) and array[order[stop]] == value:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have the same length >= 2.")
    left_rank = _average_ranks(np.asarray(left, dtype=np.float64))
    right_rank = _average_ranks(np.asarray(right, dtype=np.float64))
    left_centered = left_rank - float(np.mean(left_rank))
    right_centered = right_rank - float(np.mean(right_rank))
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if denominator == 0.0:
        raise ValueError("Spearman correlation is undefined for a constant input.")
    return float(np.dot(left_centered, right_centered) / denominator)


def _load_surface_artifact(
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    path: Path,
) -> list[dict[str, Any]]:
    artifact = _load_object(path)
    if artifact.get("content_sha256") != core.artifact_content_sha256(artifact):
        raise ValueError("Surface artifact content seal mismatch.")
    if artifact.get("artifact_type") != surface.ARTIFACT_TYPE:
        raise ValueError("Unexpected surface artifact type.")
    if (
        artifact.get("protocol_id") != protocol["protocol_id"]
        or artifact.get("protocol_sha256") != protocol_sha256
    ):
        raise ValueError("Surface artifact protocol fingerprint mismatch.")
    expected = int(protocol["source_evidence"]["expected_case_count"])
    records = artifact.get("records")
    if (
        artifact.get("source_partition") != "development"
        or artifact.get("case_count") != expected
        or not isinstance(records, list)
        or len(records) != expected
    ):
        raise ValueError("Surface artifact coverage changed.")
    decision = artifact.get("decision", {})
    if (
        decision.get("status")
        != "label_free_surface_artifact_valid_for_retrospective_join"
        or decision.get("retrospective_association_allowed") is not True
        or artifact.get("numerical_validation", {}).get("passed") is not True
    ):
        raise ValueError(
            "Surface numerical gates failed; retrospective associations are forbidden."
        )
    ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Surface artifact record must be an object.")
        compound_id = str(record.get("compound_id", ""))
        if not compound_id or compound_id in ids:
            raise ValueError("Surface artifact compound IDs must be unique.")
        ids.add(compound_id)
        if not core._is_hex(str(record.get("source_mol2_sha256", "")), 64):
            raise ValueError(f"Invalid surface source hash for {compound_id}.")
        profiles = record.get("radius_profiles")
        if not isinstance(profiles, dict) or set(profiles) != set(surface.PROFILE_ORDER):
            raise ValueError(f"Surface radius profiles changed for {compound_id}.")
        for role in surface.PROFILE_ORDER:
            descriptors = profiles[role].get("descriptors")
            if (
                not isinstance(descriptors, dict)
                or set(descriptors) != set(surface.DESCRIPTOR_KEYS)
            ):
                raise ValueError(
                    f"Surface descriptor membership changed for {compound_id}/{role}."
                )
            for key in surface.DESCRIPTOR_KEYS:
                _finite(descriptors.get(key), field=f"{compound_id}.{role}.{key}")
    return records


def _load_component_mismatches(
    protocol: Mapping[str, Any],
    path: Path,
) -> dict[str, dict[str, float]]:
    evidence = protocol["source_evidence"]
    if core.sha256_file(path) != evidence["retrospective_component_artifact_sha256"]:
        raise ValueError("Frozen component-diagnostic file hash mismatch.")
    artifact = _load_object(path)
    if artifact.get("content_sha256") != core.artifact_content_sha256(artifact):
        raise ValueError("Component-diagnostic content seal mismatch.")
    if (
        artifact.get("content_sha256")
        != evidence["retrospective_component_artifact_content_sha256"]
    ):
        raise ValueError("Frozen component-diagnostic content fingerprint mismatch.")
    if artifact.get("source_partition") not in (None, "development"):
        raise ValueError("Component diagnostic must remain development-only.")
    expected = int(evidence["expected_case_count"])
    records = artifact.get("records")
    if (
        artifact.get("case_count") != expected
        or not isinstance(records, list)
        or len(records) != expected
    ):
        raise ValueError("Component-diagnostic coverage changed.")
    output: dict[str, dict[str, float]] = {}
    for record in records:
        compound_id = str(record.get("compound_id", ""))
        if not compound_id or compound_id in output:
            raise ValueError("Component-diagnostic compound IDs must be unique.")
        output[compound_id] = {
            "cha": _finite(
                record.get("polar_mismatch_kcal_mol"),
                field=f"{compound_id}.polar_mismatch",
            ),
            "obc2": _finite(
                record.get("obc2_polar_mismatch_kcal_mol"),
                field=f"{compound_id}.obc2_polar_mismatch",
            ),
        }
    return output


def _load_structure_groups(
    protocol: Mapping[str, Any],
    path: Path,
    surface_records: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    evidence = protocol["source_evidence"]
    if core.sha256_file(path) != evidence["prepared_identity_artifact_sha256"]:
        raise ValueError("Frozen prepared-identity artifact hash mismatch.")
    prepared = _load_object(path)
    candidates = prepared.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Prepared identity artifact lacks candidates.")
    surface_by_id = {
        str(record["compound_id"]): record for record in surface_records
    }
    groups: dict[str, str] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("partition") != "development":
            continue
        compound_id = str(candidate.get("compound_id", ""))
        if compound_id not in surface_by_id:
            continue
        group = str(candidate.get("structure_group_sha256", ""))
        source_hash = str(candidate.get("mol2_sha256", ""))
        if not core._is_hex(group, 64):
            raise ValueError(f"Invalid structure-group hash for {compound_id}.")
        if source_hash != surface_by_id[compound_id]["source_mol2_sha256"]:
            raise ValueError(f"Prepared/source MOL2 identity mismatch for {compound_id}.")
        if compound_id in groups:
            raise ValueError("Prepared identity compound IDs must be unique.")
        groups[compound_id] = group
    if set(groups) != set(surface_by_id):
        raise ValueError("Prepared identity coverage does not match surface artifact.")
    return groups


def _association_arrays(
    surface_records: Sequence[Mapping[str, Any]],
    mismatches: Mapping[str, Mapping[str, float]],
) -> tuple[list[str], dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    records = sorted(surface_records, key=lambda row: row["compound_id"])
    compound_ids = [str(record["compound_id"]) for record in records]
    if set(compound_ids) != set(mismatches):
        raise ValueError("Surface/component compound-ID sets do not match.")
    abs_cha = np.asarray(
        [abs(float(mismatches[compound_id]["cha"])) for compound_id in compound_ids],
        dtype=np.float64,
    )
    reduction = np.asarray(
        [
            abs(float(mismatches[compound_id]["obc2"]))
            - abs(float(mismatches[compound_id]["cha"]))
            for compound_id in compound_ids
        ],
        dtype=np.float64,
    )
    output: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for role in surface.PROFILE_ORDER:
        descriptors = [
            record["radius_profiles"][role]["descriptors"] for record in records
        ]
        fn2 = np.asarray(
            [row["fn2_e_per_angstrom2"] for row in descriptors],
            dtype=np.float64,
        )
        phi2 = np.asarray(
            [row["phi2_e_per_angstrom"] for row in descriptors],
            dtype=np.float64,
        )
        gamma = np.abs(
            np.asarray([row["gamma_n"] for row in descriptors], dtype=np.float64)
        )
        area = np.asarray(
            [row["surface_area_angstrom2"] for row in descriptors],
            dtype=np.float64,
        )
        output[role] = {
            "absolute_cha_polar_mismatch_vs_fn2": (abs_cha, fn2),
            "absolute_cha_polar_mismatch_vs_phi2": (abs_cha, phi2),
            "absolute_cha_polar_mismatch_vs_absolute_gamma_n": (
                abs_cha,
                gamma,
            ),
            "absolute_cha_polar_mismatch_vs_surface_area": (abs_cha, area),
            "obc2_to_cha_absolute_mismatch_reduction_vs_fn2": (
                reduction,
                fn2,
            ),
            "obc2_to_cha_absolute_mismatch_reduction_vs_absolute_gamma_n": (
                reduction,
                gamma,
            ),
        }
    return compound_ids, output


def _bootstrap_counts(
    compound_ids: Sequence[str],
    structure_groups: Mapping[str, str],
    *,
    resamples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, compound_id in enumerate(compound_ids):
        grouped[structure_groups[compound_id]].append(index)
    group_ids = sorted(grouped)
    group_indices = [np.asarray(grouped[group], dtype=np.int64) for group in group_ids]
    rng = np.random.default_rng(seed)
    group_draw_counts = np.empty(
        (resamples, len(group_indices)), dtype=np.int16
    )
    for draw_index in range(resamples):
        selection = rng.integers(0, len(group_indices), size=len(group_indices))
        group_draw_counts[draw_index] = np.bincount(
            selection, minlength=len(group_indices)
        )
    record_group_index = np.empty(len(compound_ids), dtype=np.int64)
    for group_index, indices in enumerate(group_indices):
        record_group_index[indices] = group_index
    record_counts = group_draw_counts[:, record_group_index]
    sizes = Counter(len(indices) for indices in group_indices)
    return record_counts, {
        "cluster_count": len(group_indices),
        "cluster_size_counts": {
            str(size): count for size, count in sorted(sizes.items())
        },
        "all_clusters_singletons": all(len(indices) == 1 for indices in group_indices),
        "resamples": resamples,
        "seed": seed,
    }


def _bootstrap_rank_matrix(
    values: np.ndarray, record_counts: np.ndarray
) -> np.ndarray:
    """Return exact average ranks for every count-weighted bootstrap sample."""
    array = np.asarray(values, dtype=np.float64)
    counts = np.asarray(record_counts)
    if (
        array.ndim != 1
        or counts.ndim != 2
        or counts.shape[1] != len(array)
        or not np.isfinite(array).all()
        or np.any(counts < 0)
    ):
        raise ValueError("Bootstrap rank input is invalid.")
    order = np.argsort(array, kind="mergesort")
    group_by_record = np.empty(len(array), dtype=np.int64)
    groups: list[np.ndarray] = []
    start = 0
    while start < len(order):
        stop = start + 1
        value = array[order[start]]
        while stop < len(order) and array[order[stop]] == value:
            stop += 1
        group_index = len(groups)
        members = order[start:stop]
        group_by_record[members] = group_index
        groups.append(members)
        start = stop
    group_counts = np.column_stack(
        [np.sum(counts[:, members], axis=1) for members in groups]
    )
    before = np.cumsum(group_counts, axis=1) - group_counts
    group_ranks = before + 0.5 * (group_counts - 1)
    return group_ranks[:, group_by_record]


def _bootstrap_spearman(
    left: np.ndarray,
    right: np.ndarray,
    record_counts: np.ndarray,
) -> np.ndarray:
    """Vectorized exact Spearman values for count-weighted paired resamples."""
    counts = np.asarray(record_counts, dtype=np.float64)
    left_ranks = _bootstrap_rank_matrix(left, record_counts)
    right_ranks = _bootstrap_rank_matrix(right, record_counts)
    sample_sizes = np.sum(counts, axis=1)
    if np.any(sample_sizes < 2):
        raise ValueError("Bootstrap sample contains fewer than two records.")
    centers = 0.5 * (sample_sizes - 1.0)
    left_centered = left_ranks - centers[:, None]
    right_centered = right_ranks - centers[:, None]
    numerator = np.sum(
        counts * left_centered * right_centered, axis=1
    )
    left_scale = np.sum(counts * left_centered**2, axis=1)
    right_scale = np.sum(counts * right_centered**2, axis=1)
    denominator = np.sqrt(left_scale * right_scale)
    if np.any(denominator == 0.0):
        raise ValueError("Bootstrap Spearman is undefined for a constant sample.")
    return numerator / denominator


def _interval(
    values: Sequence[float], lower_probability: float, upper_probability: float
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Bootstrap interval values are invalid.")
    quantiles = np.quantile(
        array,
        [lower_probability, upper_probability],
        method="linear",
    )
    return [float(quantiles[0]), float(quantiles[1])]


def _interval_sign(interval: Sequence[float]) -> int:
    lower, upper = map(float, interval)
    if lower > 0.0:
        return 1
    if upper < 0.0:
        return -1
    return 0


def _compute_associations(
    protocol: Mapping[str, Any],
    compound_ids: Sequence[str],
    arrays: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]],
    structure_groups: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    statistics = protocol["retrospective_statistics"]
    resamples = int(statistics["bootstrap_resamples"])
    seed = int(statistics["bootstrap_seed"])
    record_counts, bootstrap_design = _bootstrap_counts(
        compound_ids,
        structure_groups,
        resamples=resamples,
        seed=seed,
    )
    point: dict[str, dict[str, float]] = {
        role: {
            name: _spearman(*arrays[role][name])
            for name in ASSOCIATION_ORDER
        }
        for role in surface.PROFILE_ORDER
    }
    samples: dict[str, dict[str, np.ndarray]] = {
        role: {
            name: _bootstrap_spearman(
                arrays[role][name][0],
                arrays[role][name][1],
                record_counts,
            )
            for name in ASSOCIATION_ORDER
        }
        for role in surface.PROFILE_ORDER
    }

    alpha = 1.0 - float(statistics["descriptor_familywise_confidence_level"])
    family_size = int(statistics["descriptor_family_size_per_radius_profile"])
    descriptor_lower = alpha / (2.0 * family_size)
    descriptor_upper = 1.0 - descriptor_lower
    output: dict[str, Any] = {}
    for role in surface.PROFILE_ORDER:
        role_output: dict[str, Any] = {}
        for name in ASSOCIATION_ORDER:
            if name in DESCRIPTOR_FAMILY:
                interval = _interval(
                    samples[role][name], descriptor_lower, descriptor_upper
                )
                interval_type = (
                    "bonferroni-familywise-95-percent-percentile-bootstrap"
                )
            else:
                interval = _interval(samples[role][name], 0.025, 0.975)
                interval_type = "descriptive-95-percent-percentile-bootstrap"
            role_output[name] = {
                "spearman_rho": point[role][name],
                "bootstrap_interval": interval,
                "bootstrap_interval_type": interval_type,
            }
        output[role] = role_output
    bootstrap_design["descriptor_interval_probabilities"] = [
        descriptor_lower,
        descriptor_upper,
    ]
    bootstrap_design["additional_interval_probabilities"] = [0.025, 0.975]
    return output, bootstrap_design


def _decision(
    protocol: Mapping[str, Any], associations: Mapping[str, Any]
) -> dict[str, Any]:
    primary_name = "absolute_cha_polar_mismatch_vs_fn2"
    gamma_name = "absolute_cha_polar_mismatch_vs_absolute_gamma_n"
    primary_interval = associations["primary"][primary_name][
        "bootstrap_interval"
    ]
    control_interval = associations["control"][primary_name][
        "bootstrap_interval"
    ]
    primary_sign = _interval_sign(primary_interval)
    control_sign = _interval_sign(control_interval)
    primary_gamma = float(
        associations["primary"][gamma_name]["spearman_rho"]
    )
    control_gamma = float(
        associations["control"][gamma_name]["spearman_rho"]
    )
    gamma_signs_agree = (
        primary_gamma != 0.0
        and control_gamma != 0.0
        and math.copysign(1.0, primary_gamma)
        == math.copysign(1.0, control_gamma)
    )
    primary_gamma_interval_sign = _interval_sign(
        associations["primary"][gamma_name]["bootstrap_interval"]
    )
    control_gamma_interval_sign = _interval_sign(
        associations["control"][gamma_name]["bootstrap_interval"]
    )
    robust_gamma_signal = (
        primary_gamma_interval_sign != 0
        and primary_gamma_interval_sign == control_gamma_interval_sign
    )
    if primary_sign == 0:
        protocol_rule_status = "reject_surface_field_proxy_for_remaining_mismatch"
    elif control_sign != primary_sign or not gamma_signs_agree:
        protocol_rule_status = "surface_definition_sensitive_or_unresolved"
    else:
        protocol_rule_status = (
            "surface_field_hypothesis_retained_for_independent_test_only"
        )

    # This is an explicit post-result reviewer correction, not a retroactive
    # rewrite of the pre-registered rule.  Two null Gamma intervals having point
    # estimates with the same sign are not replication of a nonlinear signal.
    if (
        protocol_rule_status
        == "surface_field_hypothesis_retained_for_independent_test_only"
        and not robust_gamma_signal
    ):
        status = (
            "surface_electrostatic_scale_association_only_"
            "nonlinear_hypothesis_unresolved"
        )
    else:
        status = protocol_rule_status

    area_name = "absolute_cha_polar_mismatch_vs_surface_area"
    area_confounding = any(
        _interval_sign(associations[role][area_name]["bootstrap_interval"]) != 0
        for role in surface.PROFILE_ORDER
    )
    frozen = protocol["pre_registered_decision_rule"]
    return {
        "status": status,
        "protocol_rule_status": protocol_rule_status,
        "post_result_adversarial_review_override": (
            status != protocol_rule_status
        ),
        "bondi_primary_interval_sign": primary_sign,
        "mbondi2_control_primary_interval_sign": control_sign,
        "absolute_gamma_point_estimate_signs_agree": gamma_signs_agree,
        "bondi_absolute_gamma_interval_sign": primary_gamma_interval_sign,
        "mbondi2_absolute_gamma_interval_sign": control_gamma_interval_sign,
        "robust_absolute_gamma_signal": robust_gamma_signal,
        "surface_area_association_interval_excludes_zero_in_any_profile": area_confounding,
        "interpretation": (
            "The raw field-strength association is retained only as an "
            "electrostatic-scale observation. The sign-heterogeneity proxy is "
            "null, so first-shell/nonlinear mechanism identification remains "
            "unresolved and no provider may be implemented from this result."
            if status
            == (
                "surface_electrostatic_scale_association_only_"
                "nonlinear_hypothesis_unresolved"
            )
            else (
                "At most a development-only physical hypothesis for a separately "
                "frozen independent test; never evidence for a correction or provider."
                if status
                == "surface_field_hypothesis_retained_for_independent_test_only"
                else "The pre-registered proxy is rejected or unresolved; do not implement a first-shell/nonlinear provider from this result."
            )
        ),
        "first_shell_causality_established": frozen[
            "first_shell_causality_established"
        ],
        "energy_correction_allowed": frozen["energy_correction_allowed"],
        "provider_implementation_allowed": frozen[
            "provider_implementation_allowed"
        ],
        "endpoint_selection_allowed": frozen["endpoint_selection_allowed"],
        "radius_selection_allowed": frozen["radius_selection_allowed"],
        "parameter_update_allowed": frozen["parameter_update_allowed"],
        "ranking_certification_allowed": frozen[
            "ranking_certification_allowed"
        ],
        "multisolvent_claim_allowed": frozen["multisolvent_claim_allowed"],
        "maximum_error_claim_allowed": frozen["maximum_error_claim_allowed"],
    }


def _post_result_adversarial_review(
    arrays: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]],
    surface_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    collinearity: dict[str, float] = {}
    for role in surface.PROFILE_ORDER:
        fn2 = arrays[role]["absolute_cha_polar_mismatch_vs_fn2"][1]
        phi2 = arrays[role]["absolute_cha_polar_mismatch_vs_phi2"][1]
        collinearity[role] = _spearman(fn2, phi2)
    identical_profile_count = sum(
        record["radius_profiles"]["primary"]["radius_vector_sha256"]
        == record["radius_profiles"]["control"]["radius_vector_sha256"]
        for record in surface_records
    )
    case_count = len(surface_records)
    return {
        "designation": (
            "post-result adversarial interpretation; not part of the "
            "pre-registered v1 evidence"
        ),
        "fn2_phi2_spearman_collinearity": collinearity,
        "field_strength_proxies_are_highly_collinear": all(
            abs(value) >= 0.9 for value in collinearity.values()
        ),
        "identical_bondi_mbondi2_radius_vector_count": identical_profile_count,
        "different_bondi_mbondi2_radius_vector_count": (
            case_count - identical_profile_count
        ),
        "identical_radius_vector_fraction": identical_profile_count / case_count,
        "interpretation": (
            "Fn2 and Phi2 mostly measure one electrostatic-strength axis, while "
            "Bondi and mbondi2 are identical for most records. Their agreement "
            "is not independent mechanistic replication."
        ),
    }


def run(
    *,
    protocol_path: Path,
    surface_artifact_path: Path,
    component_artifact_path: Path,
    prepared_identity_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    protocol, protocol_sha256 = surface.load_protocol(protocol_path)
    surface_records = _load_surface_artifact(
        protocol, protocol_sha256, surface_artifact_path
    )
    mismatches = _load_component_mismatches(protocol, component_artifact_path)
    structure_groups = _load_structure_groups(
        protocol, prepared_identity_path, surface_records
    )
    compound_ids, arrays = _association_arrays(surface_records, mismatches)
    associations, bootstrap_design = _compute_associations(
        protocol, compound_ids, arrays, structure_groups
    )
    decision = _decision(protocol, associations)
    adversarial_review = _post_result_adversarial_review(
        arrays, surface_records
    )
    arguments = {
        "phase": "retrospective-join",
        "protocol": _relative_to_repository(protocol_path),
        "surface_artifact": _relative_to_repository(surface_artifact_path),
        "component_artifact": _relative_to_repository(component_artifact_path),
        "prepared_identity": _relative_to_repository(prepared_identity_path),
        "output": _relative_to_repository(output_path),
    }
    artifact = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "claim_scope": protocol["claim_scope"],
        "source_partition": "development",
        "case_count": len(compound_ids),
        "label_use_boundary": {
            "historical_label_exposure": True,
            "surface_artifact_was_sealed_before_join": True,
            "surface_phase_read_labels": False,
            "published_component_mismatch_read": True,
            "prepared_identity_source_contains_experimental_values": True,
            "experimental_value_fields_referenced_by_runner": False,
            "published_explicit_component_is_experimental_truth": False,
            "no_fit": True,
            "no_residual": True,
            "no_endpoint_or_radius_selection": True,
        },
        "source_provenance": {
            "surface_artifact": _relative_to_repository(surface_artifact_path),
            "surface_artifact_sha256": core.sha256_file(surface_artifact_path),
            "surface_artifact_content_sha256": _load_object(
                surface_artifact_path
            )["content_sha256"],
            "component_artifact": _relative_to_repository(
                component_artifact_path
            ),
            "component_artifact_sha256": core.sha256_file(
                component_artifact_path
            ),
            "component_artifact_content_sha256": _load_object(
                component_artifact_path
            )["content_sha256"],
            "prepared_identity_artifact": _relative_to_repository(
                prepared_identity_path
            ),
            "prepared_identity_artifact_sha256": core.sha256_file(
                prepared_identity_path
            ),
        },
        "statistical_design": {
            **protocol["retrospective_statistics"],
            "realized_bootstrap": bootstrap_design,
        },
        "descriptive_associations": associations,
        "post_result_adversarial_review": adversarial_review,
        "decision": decision,
        "limitations": [
            "The published charging component is a fixed-charge explicit-solvent calculation, not experimental truth.",
            "All 526 development structure_group_sha256 clusters are reported as realized; singleton groups do not establish congeneric-series generalization.",
            "The SAS descriptors are finite-grid proxies and do not identify a unique first-shell or nonlinear-boundary mechanism.",
            "No experimental residual, energy correction, provider parameter, or ranking threshold is produced.",
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
        "--surface-artifact", type=Path, default=DEFAULT_SURFACE_ARTIFACT
    )
    parser.add_argument(
        "--component-artifact", type=Path, default=DEFAULT_COMPONENT_ARTIFACT
    )
    parser.add_argument(
        "--prepared-identity", type=Path, default=DEFAULT_PREPARED_IDENTITY
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = run(
        protocol_path=args.protocol.resolve(),
        surface_artifact_path=args.surface_artifact.resolve(),
        component_artifact_path=args.component_artifact.resolve(),
        prepared_identity_path=args.prepared_identity.resolve(),
        output_path=args.output.resolve(),
    )
    print(json.dumps(artifact["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
