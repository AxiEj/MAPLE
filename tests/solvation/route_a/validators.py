from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def evaluate_term(term: Mapping[str, object], record: Mapping[str, object]) -> bool:
    field = str(term["field"])
    if field not in record:
        return False
    value = record[field]
    target = term["value"]
    op = term["op"]
    if op == "eq":
        return value == target
    if op == "lt":
        return value < target
    if op == "le":
        return value <= target
    if op == "gt":
        return value > target
    if op == "ge":
        return value >= target
    if op == "abs_le":
        return abs(value) <= target
    raise ValueError(f"unsupported op: {op}")


def _close(left: float, right: float, tolerance: float = 1.0e-12) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def deterministic_bootstrap_indices(spec: Mapping[str, object]) -> list[list[int]]:
    if spec.get("algorithm") != "sha256-counter-modulo-v1":
        raise ValueError("unsupported deterministic bootstrap algorithm")
    seed = spec.get("seed")
    resamples = spec.get("resamples")
    sample_size = spec.get("sample_size")
    if (
        not isinstance(seed, str)
        or not seed
        or not isinstance(resamples, int)
        or resamples < 1000
        or not isinstance(sample_size, int)
        or sample_size < 2
    ):
        raise ValueError("invalid preregistered bootstrap specification")
    return [
        [
            int.from_bytes(
                hashlib.sha256(
                    f"{seed}:{replicate}:{draw}".encode()
                ).digest()[:8],
                "big",
            )
            % sample_size
            for draw in range(sample_size)
        ]
        for replicate in range(resamples)
    ]


def _bootstrap_means(values: list[float], index_rows: list[list[int]]) -> list[float]:
    size = len(values)
    if size == 0 or not index_rows:
        raise ValueError("bootstrap evidence is empty")
    means = []
    for indices in index_rows:
        if len(indices) != size or any(
            not isinstance(index, int) or index < 0 or index >= size
            for index in indices
        ):
            raise ValueError("invalid bootstrap index row")
        means.append(sum(values[index] for index in indices) / size)
    return means


def _status_counts(rows: list[dict]) -> tuple[int, int]:
    return (
        sum(row.get("status") == "fail" for row in rows),
        sum(row.get("status") == "abstain" for row in rows),
    )


def _trusted_scientific_evidence(
    trusted_preregistration: Mapping[str, object],
) -> Mapping[str, object] | None:
    trusted = trusted_preregistration.get("trusted_scientific_evidence")
    return trusted if isinstance(trusted, Mapping) else None


def _derive_absolute_evidence(
    record: Mapping[str, object],
    trusted_preregistration: Mapping[str, object],
) -> dict | None:
    rows = record.get("frozen_absolute_rows")
    trusted = _trusted_scientific_evidence(trusted_preregistration)
    if not isinstance(rows, list) or not rows:
        return None
    if trusted is None:
        return None
    holdout = trusted.get("frozen_holdout")
    if (
        not isinstance(holdout, Mapping)
        or canonical_sha256(holdout) != trusted.get("frozen_holdout_hash")
    ):
        return None
    if canonical_sha256(rows) != record.get("frozen_absolute_rows_hash"):
        return None
    molecule_ids = [row.get("molecule_id") for row in rows]
    expected_ids = holdout.get("molecule_ids")
    experiments = holdout.get("experiment_kcal_by_molecule")
    if (
        not isinstance(expected_ids, list)
        or not isinstance(experiments, Mapping)
        or molecule_ids != expected_ids
        or len(set(molecule_ids)) != len(rows)
    ):
        return None
    if canonical_sha256(molecule_ids) != record.get(
        "frozen_evaluated_molecule_hash"
    ):
        return None
    if any(
        row.get("experiment_kcal") != experiments.get(row.get("molecule_id"))
        for row in rows
    ):
        return None
    fail_count, abstain_count = _status_counts(rows)
    ok = [row for row in rows if row.get("status") == "ok"]
    if len(ok) + fail_count + abstain_count != len(rows) or not ok:
        return None
    try:
        errors = [
            float(row["prediction_kcal"]) - float(row["experiment_kcal"])
            for row in ok
        ]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in errors):
        return None
    count = len(errors)
    return {
        "frozen_mae_kcal": sum(abs(value) for value in errors) / count,
        "frozen_rmse_kcal": math.sqrt(
            sum(value * value for value in errors) / count
        ),
        "frozen_mse_kcal": sum(errors) / count,
        "frozen_fail_count": fail_count,
        "frozen_abstain_count": abstain_count,
    }


def _derive_route2_evidence(
    record: Mapping[str, object],
    trusted_preregistration: Mapping[str, object],
) -> dict | None:
    rows = record.get("route2_paired_rows")
    indices = record.get("route2_bootstrap_indices")
    trusted = _trusted_scientific_evidence(trusted_preregistration)
    if not isinstance(rows, list) or not isinstance(indices, list) or not rows:
        return None
    if trusted is None:
        return None
    if canonical_sha256(rows) != record.get("route2_paired_rows_hash"):
        return None
    if canonical_sha256(indices) != record.get("route2_bootstrap_indices_hash"):
        return None
    try:
        expected_indices = deterministic_bootstrap_indices(
            trusted["route2_bootstrap"]
        )
        holdout = trusted["frozen_holdout"]
        expected_ids = holdout["molecule_ids"]
        experiments = holdout["experiment_kcal_by_molecule"]
        route2_baseline = trusted["route2_baseline_by_molecule"]
        absolute_rows = record["frozen_absolute_rows"]
    except (KeyError, TypeError, ValueError):
        return None
    if indices != expected_indices:
        return None
    molecule_ids = [row.get("molecule_id") for row in rows]
    if molecule_ids != expected_ids or len(set(molecule_ids)) != len(rows):
        return None
    absolute_by_id = {
        row.get("molecule_id"): row.get("prediction_kcal")
        for row in absolute_rows
    }
    if any(
        row.get("experiment_kcal") != experiments.get(row.get("molecule_id"))
        or row.get("route2_prediction_kcal")
        != route2_baseline.get(row.get("molecule_id"))
        or row.get("route_a_prediction_kcal")
        != absolute_by_id.get(row.get("molecule_id"))
        for row in rows
    ):
        return None
    fail_count, abstain_count = _status_counts(rows)
    ok = [row for row in rows if row.get("status") == "ok"]
    if len(ok) + fail_count + abstain_count != len(rows) or not ok:
        return None
    try:
        d_i = [
            abs(float(row["route_a_prediction_kcal"]) - float(row["experiment_kcal"]))
            - abs(float(row["route2_prediction_kcal"]) - float(row["experiment_kcal"]))
            for row in ok
        ]
        means = _bootstrap_means(d_i, indices)
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "route2_delta_ci_95_lower": _percentile(means, 0.025),
        "route2_delta_ci_95_upper": _percentile(means, 0.975),
        "route2_fail_count": fail_count,
        "route2_abstain_count": abstain_count,
        "route2_d_i_abs_error": d_i,
        "route2_sample_count": len(ok),
    }


def _derive_partition_evidence(
    record: Mapping[str, object],
    trusted_preregistration: Mapping[str, object],
) -> dict | None:
    rows = record.get("partition_bias_rows")
    indices = record.get("partition_bias_bootstrap_indices")
    trusted = _trusted_scientific_evidence(trusted_preregistration)
    if not isinstance(rows, list) or not isinstance(indices, list) or not rows:
        return None
    if trusted is None:
        return None
    rows_hash = canonical_sha256(rows)
    if rows_hash != record.get("partition_bias_rows_hash"):
        return None
    if canonical_sha256(indices) != record.get(
        "partition_bias_bootstrap_indices_hash"
    ):
        return None
    try:
        expected_indices = deterministic_bootstrap_indices(
            trusted["partition_bias_bootstrap"]
        )
        expected_references = trusted["full_explicit_reference_by_molecule"]
        absolute_rows = record["frozen_absolute_rows"]
    except (KeyError, TypeError, ValueError):
        return None
    if canonical_sha256(expected_references) != record.get(
        "full_explicit_subset_hash"
    ):
        return None
    if indices != expected_indices:
        return None
    molecule_ids = [row.get("molecule_id") for row in rows]
    expected_ids = list(expected_references)
    if molecule_ids != expected_ids or len(set(molecule_ids)) != len(rows):
        return None
    absolute_by_id = {
        row.get("molecule_id"): row.get("prediction_kcal")
        for row in absolute_rows
    }
    if any(
        row.get("full_explicit_reference_kcal")
        != expected_references.get(row.get("molecule_id"))
        or row.get("route_a_prediction_kcal")
        != absolute_by_id.get(row.get("molecule_id"))
        for row in rows
    ):
        return None
    fail_count, abstain_count = _status_counts(rows)
    ok = [row for row in rows if row.get("status") == "ok"]
    if len(ok) + fail_count + abstain_count != len(rows) or not ok:
        return None
    try:
        signed = [
            float(row["route_a_prediction_kcal"])
            - float(row["full_explicit_reference_kcal"])
            for row in ok
        ]
        means = _bootstrap_means(signed, indices)
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "partition_bias_mean_signed_kcal": sum(signed) / len(signed),
        "partition_bias_ci_95_lower_kcal": _percentile(means, 0.025),
        "partition_bias_ci_95_upper_kcal": _percentile(means, 0.975),
        "partition_bias_fail_count": fail_count,
        "partition_bias_abstain_count": abstain_count,
    }


def _summaries_match(record: Mapping[str, object], derived: Mapping[str, object]) -> bool:
    for field, expected in derived.items():
        if field not in record:
            return False
        actual = record[field]
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or not _close(
                float(actual),
                expected,
            ):
                return False
        elif actual != expected:
            return False
    return True


def derive_reference_cycle_gates(
    contract: Mapping[str, object],
    evidence: Mapping[str, object],
    trusted_preregistration: Mapping[str, object],
    *,
    allow_test_evidence: bool = False,
) -> dict[str, bool] | None:
    trusted_root = contract["trusted_preregistration_root"]
    if canonical_sha256(trusted_preregistration) != trusted_root["canonical_sha256"]:
        return None
    if trusted_preregistration.get("created_before_holdout_open") is not True:
        return None
    evidence_class = trusted_preregistration.get("evidence_class")
    if evidence.get("evidence_class") != evidence_class:
        return None
    production_class = trusted_root["production_allowed_evidence_class"]
    test_class = trusted_root["test_fixture_evidence_class"]
    if evidence_class != production_class and not (
        allow_test_evidence
        and evidence_class == test_class
        and trusted_preregistration.get("test_only") is True
        and evidence.get("test_only") is True
    ):
        return None
    try:
        selection = evidence["selection"]
        selection_policy = evidence["selection_policy"]
        holdout_lock = evidence["holdout_lock"]
        holdout = evidence["holdout"]
        n_by_molecule = evidence["n_by_molecule"]
        nonreference_controls = evidence["nonreference_controls"]
        cycles = evidence["cycles"]
        cycle_artifact_hashes = evidence["cycle_artifact_hashes"]
    except KeyError:
        return None

    selection_hash = canonical_sha256(selection)
    selection_policy_hash = canonical_sha256(selection_policy)
    holdout_hash = canonical_sha256(holdout)
    holdout_lock_hash = canonical_sha256(holdout_lock)
    n_map_hash = canonical_sha256(n_by_molecule)
    controls_hash = canonical_sha256(nonreference_controls)

    if evidence.get("selection_hash") != selection_hash:
        return None
    if evidence.get("selection_policy_hash") != selection_policy_hash:
        return None
    if selection.get("selection_policy_hash") != selection_policy_hash:
        return None
    if evidence.get("holdout_hash") != holdout_hash:
        return None
    if evidence.get("holdout_lock_hash") != holdout_lock_hash:
        return None
    if evidence.get("n_by_molecule_hash") != n_map_hash:
        return None
    if evidence.get("nonreference_controls_hash") != controls_hash:
        return None
    if holdout_lock.get("development_selection_artifact_hash") != selection_hash:
        return None
    if holdout_lock.get("frozen_holdout_hash") != holdout_hash:
        return None
    if holdout_lock.get("opened_for_scoring_after_lock") is not True:
        return None
    if trusted_preregistration.get("selection_policy_artifact_hash") != selection_policy_hash:
        return None
    if trusted_preregistration.get("development_selection_artifact_hash") != selection_hash:
        return None
    if trusted_preregistration.get("holdout_lock_hash") != holdout_lock_hash:
        return None
    if trusted_preregistration.get("frozen_holdout_hash") != holdout_hash:
        return None

    expected_ids = set(contract["cycles"])
    selected_primary = selection.get("selected_primary_cycle")
    if selected_primary not in expected_ids:
        return None
    if evidence.get("primary_cycle") != selected_primary:
        return None

    actual_ids = {row.get("cycle_id") for row in cycles}
    if actual_ids != expected_ids or len(cycles) != len(expected_ids):
        return None

    trusted_science = _trusted_scientific_evidence(
        trusted_preregistration
    )
    if trusted_science is None:
        return None
    trusted_holdout = trusted_science.get("frozen_holdout")
    if (
        not isinstance(trusted_holdout, Mapping)
        or canonical_sha256(trusted_holdout)
        != trusted_science.get("frozen_holdout_hash")
        or holdout != trusted_holdout
    ):
        return None
    expected_molecule_order = holdout.get("molecule_ids")
    if (
        not isinstance(expected_molecule_order, list)
        or list(n_by_molecule) != expected_molecule_order
    ):
        return None
    expected_molecule_ids = set(expected_molecule_order)
    expected_artifact_hashes = {
        "selection_policy_artifact_hash": selection_policy_hash,
        "development_selection_artifact_hash": selection_hash,
        "frozen_holdout_hash": holdout_hash,
        "holdout_lock_hash": holdout_lock_hash,
        "n_by_molecule_artifact_hash": n_map_hash,
        "nonreference_controls_artifact_hash": controls_hash,
    }
    for row in cycles:
        if row.get("frozen_holdout_hash") != holdout_hash:
            return None
        if row.get("n_by_molecule_hash") != n_map_hash:
            return None
        if row.get("nonreference_controls_hash") != controls_hash:
            return None
        if row.get("failure_count") or row.get("abstention_count"):
            return None

        results = row.get("per_molecule_results")
        uncertainty = row.get("uncertainty")
        if not isinstance(results, list) or not isinstance(uncertainty, dict):
            return None
        result_order = [item.get("molecule_id") for item in results]
        result_ids = set(result_order)
        if (
            result_order != expected_molecule_order
            or result_ids != expected_molecule_ids
            or len(results) != len(result_ids)
        ):
            return None
        if any(
            item.get("n") != n_by_molecule.get(item.get("molecule_id"))
            for item in results
        ):
            return None

        result_hash = canonical_sha256(results)
        uncertainty_hash = canonical_sha256(uncertainty)
        if row.get("per_molecule_result_artifact_hash") != result_hash:
            return None
        if row.get("uncertainty_artifact_hash") != uncertainty_hash:
            return None
        declared = cycle_artifact_hashes.get(row["cycle_id"])
        if declared != {
            "per_molecule_result_artifact_hash": result_hash,
            "uncertainty_artifact_hash": uncertainty_hash,
        }:
            return None
        expected_artifact_hashes[
            f"{row['cycle_id']}:per_molecule_result_artifact_hash"
        ] = result_hash
        expected_artifact_hashes[
            f"{row['cycle_id']}:uncertainty_artifact_hash"
        ] = uncertainty_hash

    if evidence.get("artifact_hashes") != expected_artifact_hashes:
        return None
    if evidence.get("per_molecule_both_cycle_results") != cycles:
        return None
    manifest = evidence.get("reference_cycle_ablation_manifest")
    if not isinstance(manifest, dict):
        return None
    manifest_hash = canonical_sha256(manifest)
    if evidence.get("reference_cycle_ablation_manifest_hash") != manifest_hash:
        return None
    if manifest.get("artifact_hashes") != expected_artifact_hashes:
        return None
    if manifest.get("cycle_artifact_hashes") != cycle_artifact_hashes:
        return None
    if manifest.get("primary_cycle") != selected_primary:
        return None
    adjudication = evidence.get("adjudication")
    if not isinstance(adjudication, dict):
        return None
    if canonical_sha256(adjudication) != evidence.get("adjudication_hash"):
        return None
    if adjudication.get("manifest_hash") != manifest_hash:
        return None
    if adjudication.get("primary_cycle") != selected_primary:
        return None

    required_manifest_fields = contract["manifest_required_fields"]
    if any(field not in evidence for field in required_manifest_fields):
        return None

    return {field: True for field in contract["claim_gate_fields"]}


def _canonical_stage(
    record: Mapping[str, object],
    field: str,
) -> Mapping[str, object] | None:
    value = record.get(field)
    if not isinstance(value, Mapping):
        return None
    if canonical_sha256(value) != record.get(f"{field}_hash"):
        return None
    return value


def _derive_stage_gates(
    claim: Mapping[str, object],
    record: Mapping[str, object],
    trusted_preregistration: Mapping[str, object],
    trusted_contract_hashes: Mapping[str, str] | None,
) -> dict[str, bool] | None:
    if trusted_contract_hashes is None:
        return None
    if any(
        record.get(field) != trusted_contract_hashes.get(field)
        for field in claim["hash_fields"]
    ):
        return None

    engineering = _canonical_stage(record, "engineering_evidence")
    closure = _canonical_stage(record, "closure_evidence")
    benchmark = _canonical_stage(record, "benchmark_lock_evidence")
    full_explicit = _canonical_stage(record, "full_explicit_evidence")
    if any(
        evidence is None
        for evidence in (engineering, closure, benchmark, full_explicit)
    ):
        return None

    protocol_hash = trusted_contract_hashes["protocol_hash"]
    expected_evidence_class = trusted_preregistration.get("evidence_class")
    if any(
        evidence.get("evidence_class") != expected_evidence_class
        for evidence in (engineering, closure, benchmark, full_explicit)
    ):
        return None

    engineering_pass = (
        engineering.get("validator_id")
        == "route-a-engineering-stage-validator-v1"
        and engineering.get("protocol_hash") == protocol_hash
        and engineering.get("tests_passed") is True
        and engineering.get("compileall_passed") is True
        and isinstance(engineering.get("test_artifact_hash"), str)
    )
    closure_gates = closure.get("derived_gates")
    required_closure = {
        "internal_closure_ledger",
        "internal_closure_statistics",
        "internal_closure_tail_gates",
        "thermodynamic_state_contract",
    }
    closure_pass = (
        closure.get("validator_id")
        == "route-a-closure-stage-validator-v1"
        and closure.get("protocol_hash") == protocol_hash
        and closure.get("state_contract_hash")
        == trusted_contract_hashes["state_contract_hash"]
        and isinstance(closure_gates, Mapping)
        and all(closure_gates.get(field) is True for field in required_closure)
        and isinstance(closure.get("state_table_artifact_hash"), str)
    )

    trusted = _trusted_scientific_evidence(trusted_preregistration)
    if trusted is None:
        return None
    frozen_holdout_hash = trusted.get("frozen_holdout_hash")
    benchmark_pass = (
        benchmark.get("validator_id")
        == "route-a-benchmark-lock-validator-v1"
        and benchmark.get("protocol_hash") == protocol_hash
        and benchmark.get("frozen_holdout_hash") == frozen_holdout_hash
        and benchmark.get("frozen_reference") is True
        and benchmark.get("protocol_locked_reference_set") is True
        and benchmark.get("holdout_partition_lock") is True
    )
    full_explicit_hash = canonical_sha256(
        trusted["full_explicit_reference_by_molecule"]
    )
    full_explicit_pass = (
        full_explicit.get("validator_id")
        == "route-a-full-explicit-subset-validator-v1"
        and full_explicit.get("protocol_hash") == protocol_hash
        and full_explicit.get("subset_hash") == full_explicit_hash
        and full_explicit.get("complete") is True
    )

    result = {
        "engineering": engineering_pass,
        "frozen_reference": benchmark_pass,
        "protocol_locked_reference_set": benchmark_pass,
        "holdout_partition_lock": benchmark_pass,
        "full_explicit_reference_subset": full_explicit_pass,
    }
    for field in required_closure:
        result[field] = bool(
            closure_pass and closure_gates.get(field)
        )
    return result


def evaluate_claim_status(
    claim: Mapping[str, object],
    status: str,
    raw_record: Mapping[str, object],
    reference_cycle_contract: Mapping[str, object],
    trusted_preregistration: Mapping[str, object],
    trusted_contract_hashes: Mapping[str, str] | None = None,
    *,
    allow_test_evidence: bool = False,
) -> bool:
    record = dict(raw_record)
    if status not in claim["result_order"]:
        return False
    production_gate = claim["production_gate"]
    if not allow_test_evidence and (
        claim["result_order"].index(status)
        > claim["result_order"].index(production_gate["maximum_status"])
    ):
        return False

    stage_gates = _derive_stage_gates(
        claim,
        record,
        trusted_preregistration,
        trusted_contract_hashes,
    )
    if stage_gates is None:
        return False
    record.update(stage_gates)

    required_result_fields = claim["evidence_derivation"]["required_result_fields"]
    requires_scientific_evidence = (
        claim["result_order"].index(status)
        >= claim["result_order"].index("route2-superiority-demonstrated")
    )
    if requires_scientific_evidence and any(
        field not in record for field in required_result_fields
    ):
        return False

    sha_fields = [
        field
        for field in required_result_fields
        if field.endswith("_hash") or field.endswith("_artifact_hash")
    ]
    if requires_scientific_evidence and any(
        not isinstance(record[field], str)
        or re.fullmatch(r"[a-f0-9]{64}", record[field]) is None
        for field in sha_fields
    ):
        return False

    derived_absolute = _derive_absolute_evidence(
        record,
        trusted_preregistration,
    )
    absolute_metric = claim["frozen_absolute_accuracy"]
    absolute_match = (
        derived_absolute is not None
        and _summaries_match(record, derived_absolute)
    )
    absolute_record = {**record, **(derived_absolute or {})}
    record[absolute_metric["gate_field"]] = (
        absolute_match
        and all(
            evaluate_term(term, absolute_record)
            for term in absolute_metric["evaluator"]["terms"]
        )
    )

    superiority = claim["route2_superiority_metric"]
    derived_superiority = _derive_route2_evidence(
        record,
        trusted_preregistration,
    )
    superiority_complete = (
        derived_superiority is not None
        and _summaries_match(record, derived_superiority)
    )
    superiority_record = {**record, **(derived_superiority or {})}
    record["route2_superiority_metric"] = (
        superiority_complete
        and all(
            evaluate_term(term, superiority_record)
            for term in superiority["evaluator"]["terms"]
        )
    )

    derived_partition = _derive_partition_evidence(
        record,
        trusted_preregistration,
    )
    partition_metric = claim["material_partition_bias"]
    partition_match = (
        derived_partition is not None
        and _summaries_match(record, derived_partition)
    )
    partition_record = {**record, **(derived_partition or {})}
    record[partition_metric["gate_field"]] = (
        partition_match
        and all(
            evaluate_term(term, partition_record)
            for term in partition_metric["evaluator"]["terms"]
        )
    )

    reference_evidence = record.get("reference_cycle_evidence", {})
    ablation = derive_reference_cycle_gates(
        reference_cycle_contract,
        reference_evidence,
        trusted_preregistration,
        allow_test_evidence=allow_test_evidence,
    )
    lineage_matches = (
        isinstance(reference_evidence, dict)
        and record.get("reference_cycle_ablation_manifest_hash")
        == reference_evidence.get("reference_cycle_ablation_manifest_hash")
        and record.get("development_selection_artifact_hash")
        == reference_evidence.get("development_selection_artifact_hash")
        and record.get("holdout_lock_hash")
        == reference_evidence.get("holdout_lock_hash")
        and record.get("cycle_artifact_hashes")
        == reference_evidence.get("cycle_artifact_hashes")
        and record.get("artifact_hashes")
        == reference_evidence.get("artifact_hashes")
        and record.get("adjudication_hash")
        == reference_evidence.get("adjudication_hash")
    )
    if not lineage_matches:
        ablation = None
    for field in reference_cycle_contract["claim_gate_fields"]:
        record[field] = bool(ablation and ablation.get(field))

    trusted = _trusted_scientific_evidence(trusted_preregistration)
    trusted_holdout = trusted.get("frozen_holdout", {}) if trusted else {}
    expected_order = trusted_holdout.get("molecule_ids")
    absolute_order = [
        row.get("molecule_id")
        for row in record.get("frozen_absolute_rows", [])
    ]
    route2_order = [
        row.get("molecule_id")
        for row in record.get("route2_paired_rows", [])
    ]
    reference_holdout = (
        reference_evidence.get("holdout", {})
        if isinstance(reference_evidence, Mapping)
        else {}
    )
    record["paired_same_molecule_coverage"] = bool(
        derived_absolute is not None
        and derived_superiority is not None
        and ablation is not None
        and isinstance(expected_order, list)
        and absolute_order == route2_order == expected_order
        and reference_holdout.get("molecule_ids") == expected_order
        and record.get("frozen_evaluated_molecule_hash")
        == canonical_sha256(expected_order)
    )

    def evaluate_lattice_entry(target_status: str) -> bool:
        entry = next(
            item
            for item in claim["status_lattice"]
            if item["status"] == target_status
        )
        return all(evaluate_term(term, record) for term in entry["required"])

    record["thermodynamic-closure-passed"] = evaluate_lattice_entry(
        "thermodynamic-closure-passed"
    )
    record["route2-superiority-demonstrated"] = evaluate_lattice_entry(
        "route2-superiority-demonstrated"
    )
    return evaluate_lattice_entry(status)
