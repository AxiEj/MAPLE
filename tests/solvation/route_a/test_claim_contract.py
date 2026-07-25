from __future__ import annotations

import copy

from .conftest import DOCS_DIR, FIXTURE_DIR, load_json
from .validators import (
    canonical_sha256,
    deterministic_bootstrap_indices,
    evaluate_claim_status,
)

CLAIM_PATH = DOCS_DIR / "claim-contract-v1.json"
PROTOCOL_PATH = DOCS_DIR / "protocol-v1.json"
FAILURE_PATH = DOCS_DIR / "failure-contract-v1.json"
STATE_PATH = DOCS_DIR / "state-contract-v1.json"
RESTRAINT_PATH = DOCS_DIR / "restraint-contract-v1.json"
OUTER_ADAPTER_PATH = DOCS_DIR / "outer-adapter-contract-v1.json"
REFERENCE_CYCLE_PATH = DOCS_DIR / "reference-cycle-ablation-contract-v1.json"
REFERENCE_EVIDENCE_PATH = (
    FIXTURE_DIR / "reference_cycle" / "valid_ablation_evidence_v1.json"
)
PREREGISTRATION_PATH = (
    FIXTURE_DIR / "reference_cycle" / "preregistration_registry_v1.json"
)


def _eval_term(term, record):
    field = term["field"]
    op = term["op"]
    if field not in record:
        return False
    value = record[field]
    target = term["value"]

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


def _eval_required(required, record):
    for term in required:
        if not _eval_term(term, record):
            return False
    return True


def _trusted_contract_hashes():
    protocol = load_json(PROTOCOL_PATH)
    return {
        "protocol_hash": protocol["protocol_sha256"],
        "failure_contract_hash": canonical_sha256(load_json(FAILURE_PATH)),
        "state_contract_hash": canonical_sha256(load_json(STATE_PATH)),
        "restraint_contract_hash": canonical_sha256(load_json(RESTRAINT_PATH)),
        "outer_adapter_contract_hash": canonical_sha256(
            load_json(OUTER_ADAPTER_PATH)
        ),
        "reference_cycle_ablation_contract_hash": canonical_sha256(
            load_json(REFERENCE_CYCLE_PATH)
        ),
    }


def _evaluate_status(
    contract,
    status,
    record,
    *,
    allow_test_evidence=True,
):
    return evaluate_claim_status(
        contract,
        status,
        record,
        load_json(REFERENCE_CYCLE_PATH),
        load_json(PREREGISTRATION_PATH),
        _trusted_contract_hashes(),
        allow_test_evidence=allow_test_evidence,
    )


def _evaluate_superiority_metric(contract, record):
    evaluator = contract["evaluator"]
    assert evaluator["op"] == "and"
    return all(_eval_term(term, record) for term in evaluator["terms"])


def test_claim_contract_status_lattice_is_compact_and_staged():
    claim = load_json(CLAIM_PATH)
    preregistration = load_json(PREREGISTRATION_PATH)
    trusted = preregistration["trusted_scientific_evidence"]
    absolute_rows = [
        {"molecule_id": "m1", "prediction_kcal": 0.5, "experiment_kcal": 0.0, "status": "ok"},
        {"molecule_id": "m2", "prediction_kcal": -0.6, "experiment_kcal": 0.0, "status": "ok"},
        {"molecule_id": "m3", "prediction_kcal": 0.7, "experiment_kcal": 0.0, "status": "ok"},
        {"molecule_id": "m4", "prediction_kcal": -0.4, "experiment_kcal": 0.0, "status": "ok"},
    ]
    route2_rows = [
        {"molecule_id": "m1", "route_a_prediction_kcal": 0.5, "route2_prediction_kcal": 1.0, "experiment_kcal": 0.0, "status": "ok"},
        {"molecule_id": "m2", "route_a_prediction_kcal": -0.6, "route2_prediction_kcal": -1.1, "experiment_kcal": 0.0, "status": "ok"},
        {"molecule_id": "m3", "route_a_prediction_kcal": 0.7, "route2_prediction_kcal": 1.2, "experiment_kcal": 0.0, "status": "ok"},
        {"molecule_id": "m4", "route_a_prediction_kcal": -0.4, "route2_prediction_kcal": -0.9, "experiment_kcal": 0.0, "status": "ok"},
    ]
    route2_bootstrap = deterministic_bootstrap_indices(
        trusted["route2_bootstrap"]
    )
    partition_rows = [
        {"molecule_id": "m1", "route_a_prediction_kcal": 0.5, "full_explicit_reference_kcal": 0.4, "status": "ok"},
        {"molecule_id": "m2", "route_a_prediction_kcal": -0.6, "full_explicit_reference_kcal": -0.5, "status": "ok"},
        {"molecule_id": "m3", "route_a_prediction_kcal": 0.7, "full_explicit_reference_kcal": 0.5, "status": "ok"},
        {"molecule_id": "m4", "route_a_prediction_kcal": -0.4, "full_explicit_reference_kcal": -0.2, "status": "ok"},
    ]
    partition_bootstrap = deterministic_bootstrap_indices(
        trusted["partition_bias_bootstrap"]
    )

    assert claim["result_order"] == [
        "failed",
        "engineering-ready",
        "thermodynamic-closure-passed",
        "route2-superiority-demonstrated",
        "scientifically-validated",
    ]
    statuses = [row["status"] for row in claim["status_lattice"]]
    assert statuses == claim["result_order"]

    contract_hashes = _trusted_contract_hashes()
    evidence_class = preregistration["evidence_class"]
    engineering_evidence = {
        "validator_id": "route-a-engineering-stage-validator-v1",
        "evidence_class": evidence_class,
        "protocol_hash": contract_hashes["protocol_hash"],
        "tests_passed": True,
        "compileall_passed": True,
        "test_artifact_hash": "a" * 64,
    }
    closure_evidence = {
        "validator_id": "route-a-closure-stage-validator-v1",
        "evidence_class": evidence_class,
        "protocol_hash": contract_hashes["protocol_hash"],
        "state_contract_hash": contract_hashes["state_contract_hash"],
        "state_table_artifact_hash": "b" * 64,
        "derived_gates": {
            "internal_closure_ledger": True,
            "internal_closure_statistics": True,
            "internal_closure_tail_gates": True,
            "thermodynamic_state_contract": True,
        },
    }
    benchmark_lock_evidence = {
        "validator_id": "route-a-benchmark-lock-validator-v1",
        "evidence_class": evidence_class,
        "protocol_hash": contract_hashes["protocol_hash"],
        "frozen_holdout_hash": trusted["frozen_holdout_hash"],
        "frozen_reference": True,
        "protocol_locked_reference_set": True,
        "holdout_partition_lock": True,
    }
    full_explicit_evidence = {
        "validator_id": "route-a-full-explicit-subset-validator-v1",
        "evidence_class": evidence_class,
        "protocol_hash": contract_hashes["protocol_hash"],
        "subset_hash": canonical_sha256(
            trusted["full_explicit_reference_by_molecule"]
        ),
        "complete": True,
    }

    base = {
        **contract_hashes,
        "engineering_evidence": engineering_evidence,
        "engineering_evidence_hash": canonical_sha256(engineering_evidence),
        "closure_evidence": closure_evidence,
        "closure_evidence_hash": canonical_sha256(closure_evidence),
        "benchmark_lock_evidence": benchmark_lock_evidence,
        "benchmark_lock_evidence_hash": canonical_sha256(
            benchmark_lock_evidence
        ),
        "full_explicit_evidence": full_explicit_evidence,
        "full_explicit_evidence_hash": canonical_sha256(
            full_explicit_evidence
        ),
        "engineering": True,
        "internal_closure_ledger": True,
        "internal_closure_statistics": True,
        "internal_closure_tail_gates": True,
        "thermodynamic_state_contract": True,
        "thermodynamic-closure-passed": True,
        "frozen_reference": True,
        "protocol_locked_reference_set": True,
        "paired_same_molecule_coverage": True,
        "frozen_mae_kcal": 0.55,
        "frozen_rmse_kcal": 0.5612486080160912,
        "frozen_mse_kcal": 0.049999999999999975,
        "frozen_fail_count": 0,
        "frozen_abstain_count": 0,
        "frozen_absolute_rows": absolute_rows,
        "frozen_absolute_rows_hash": canonical_sha256(absolute_rows),
        "frozen_evaluated_molecule_hash": canonical_sha256(
            [row["molecule_id"] for row in absolute_rows]
        ),
        "route2_delta_ci_95_upper": -0.5,
        "route2_delta_ci_95_lower": -0.5,
        "route2_fail_count": 0,
        "route2_abstain_count": 0,
        "route2_d_i_abs_error": [-0.5, -0.5000000000000001, -0.5, -0.5],
        "route2_sample_count": 4,
        "route2_paired_rows": route2_rows,
        "route2_paired_rows_hash": canonical_sha256(route2_rows),
        "route2_bootstrap_indices": route2_bootstrap,
        "route2_bootstrap_indices_hash": canonical_sha256(route2_bootstrap),
        "holdout_partition_lock": True,
        "full_explicit_reference_subset": True,
        "partition_bias_mean_signed_kcal": 0.0,
        "partition_bias_ci_95_lower_kcal": -0.15000000000000002,
        "partition_bias_ci_95_upper_kcal": 0.15000000000000002,
        "partition_bias_fail_count": 0,
        "partition_bias_abstain_count": 0,
        "partition_bias_rows": partition_rows,
        "partition_bias_rows_hash": canonical_sha256(partition_rows),
        "partition_bias_bootstrap_indices": partition_bootstrap,
        "partition_bias_bootstrap_indices_hash": canonical_sha256(
            partition_bootstrap
        ),
        "full_explicit_subset_hash": canonical_sha256(
            trusted["full_explicit_reference_by_molecule"]
        ),
        "reference_cycle_ablation_completed": True,
        "reference_cycle_ablation_same_frozen_holdout": True,
        "reference_cycle_ablation_same_n_by_molecule": True,
        "reference_cycle_ablation_nonreference_controls_locked": True,
        "reference_cycle_ablation_both_cycles_complete": True,
        "reference_cycle_ablation_artifact_verified": True,
        "reference_cycle_evidence": load_json(REFERENCE_EVIDENCE_PATH),
        "reference_cycle_ablation_manifest_hash": load_json(
            REFERENCE_EVIDENCE_PATH
        )["reference_cycle_ablation_manifest_hash"],
        "development_selection_artifact_hash": load_json(
            REFERENCE_EVIDENCE_PATH
        )["development_selection_artifact_hash"],
        "holdout_lock_hash": load_json(REFERENCE_EVIDENCE_PATH)[
            "holdout_lock_hash"
        ],
        "cycle_artifact_hashes": load_json(REFERENCE_EVIDENCE_PATH)[
            "cycle_artifact_hashes"
        ],
        "artifact_hashes": load_json(REFERENCE_EVIDENCE_PATH)[
            "artifact_hashes"
        ],
        "adjudication_hash": load_json(REFERENCE_EVIDENCE_PATH)[
            "adjudication_hash"
        ],
    }

    # engineering prerequisite is required and sufficient for engineering-ready
    assert _evaluate_status(claim, "engineering-ready", base)
    base_no_eng = dict(base)
    base_no_eng["engineering"] = False
    assert _evaluate_status(claim, "engineering-ready", base_no_eng)
    bad_engineering = copy.deepcopy(base)
    bad_engineering["engineering_evidence"]["tests_passed"] = False
    bad_engineering["engineering_evidence_hash"] = canonical_sha256(
        bad_engineering["engineering_evidence"]
    )
    assert not _evaluate_status(
        claim,
        "engineering-ready",
        bad_engineering,
    )

    # thermodynamic closure strictly requires all internal closure gates
    assert _evaluate_status(claim, "thermodynamic-closure-passed", base)
    for field in ["internal_closure_ledger", "internal_closure_statistics", "internal_closure_tail_gates", "thermodynamic_state_contract"]:
        bad = copy.deepcopy(base)
        bad["closure_evidence"]["derived_gates"][field] = False
        bad["closure_evidence_hash"] = canonical_sha256(
            bad["closure_evidence"]
        )
        assert not _evaluate_status(claim, "thermodynamic-closure-passed", bad)

    # route-2 superiority requires closure + protocol/holdout protections
    assert _evaluate_status(claim, "route2-superiority-demonstrated", base)
    bad = copy.deepcopy(base)
    bad["benchmark_lock_evidence"]["frozen_reference"] = False
    bad["benchmark_lock_evidence_hash"] = canonical_sha256(
        bad["benchmark_lock_evidence"]
    )
    assert not _evaluate_status(claim, "route2-superiority-demonstrated", bad)

    # Final scientific status derives absolute and partition gates from numeric
    # evidence; caller-supplied booleans cannot override failed metrics.
    assert _evaluate_status(claim, "scientifically-validated", base)
    assert not _evaluate_status(
        claim,
        "scientifically-validated",
        base,
        allow_test_evidence=False,
    )
    assert claim["production_gate"]["enabled"] is False
    assert claim["production_gate"]["maximum_status"] == "engineering-ready"
    for field in claim["evidence_derivation"]["required_result_fields"]:
        missing = dict(base)
        missing.pop(field)
        assert not _evaluate_status(
            claim,
            "scientifically-validated",
            missing,
        ), field
    for rows_field, value_field in [
        ("frozen_absolute_rows", "prediction_kcal"),
        ("route2_paired_rows", "route_a_prediction_kcal"),
        ("partition_bias_rows", "route_a_prediction_kcal"),
    ]:
        tampered = copy.deepcopy(base)
        tampered[rows_field][0][value_field] = 99.0
        assert not _evaluate_status(
            claim,
            "scientifically-validated",
            tampered,
        ), rows_field
    for field in [
        "reference_cycle_ablation_manifest_hash",
        "development_selection_artifact_hash",
        "holdout_lock_hash",
        "adjudication_hash",
    ]:
        tampered = copy.deepcopy(base)
        tampered[field] = "0" * 64
        assert not _evaluate_status(
            claim,
            "scientifically-validated",
            tampered,
        ), field
    tampered_cycle_hashes = copy.deepcopy(base)
    tampered_cycle_hashes["cycle_artifact_hashes"] = {
        cycle_id: {
            "per_molecule_result_artifact_hash": "0" * 64,
            "uncertainty_artifact_hash": "0" * 64,
        }
        for cycle_id in tampered_cycle_hashes["cycle_artifact_hashes"]
    }
    assert not _evaluate_status(
        claim,
        "scientifically-validated",
        tampered_cycle_hashes,
    )
    disjoint_route2 = copy.deepcopy(base)
    for index, row in enumerate(disjoint_route2["route2_paired_rows"]):
        row["molecule_id"] = f"other-{index}"
    disjoint_route2["route2_paired_rows_hash"] = canonical_sha256(
        disjoint_route2["route2_paired_rows"]
    )
    assert not _evaluate_status(
        claim,
        "scientifically-validated",
        disjoint_route2,
    )
    cherry_picked = copy.deepcopy(base)
    cherry_picked["route2_bootstrap_indices"] = [
        [0, 0, 0, 0]
        for _ in cherry_picked["route2_bootstrap_indices"]
    ]
    cherry_picked["route2_bootstrap_indices_hash"] = canonical_sha256(
        cherry_picked["route2_bootstrap_indices"]
    )
    assert not _evaluate_status(
        claim,
        "scientifically-validated",
        cherry_picked,
    )
    rewritten_experiment = copy.deepcopy(base)
    rewritten_experiment["frozen_absolute_rows"][0][
        "experiment_kcal"
    ] = 99.0
    rewritten_experiment["frozen_absolute_rows_hash"] = canonical_sha256(
        rewritten_experiment["frozen_absolute_rows"]
    )
    assert not _evaluate_status(
        claim,
        "scientifically-validated",
        rewritten_experiment,
    )
    rewritten_route2 = copy.deepcopy(base)
    rewritten_route2["route2_paired_rows"][0][
        "route2_prediction_kcal"
    ] = 99.0
    rewritten_route2["route2_paired_rows_hash"] = canonical_sha256(
        rewritten_route2["route2_paired_rows"]
    )
    assert not _evaluate_status(
        claim,
        "scientifically-validated",
        rewritten_route2,
    )
    forged_contract = copy.deepcopy(base)
    forged_contract["state_contract_hash"] = "0" * 64
    assert not _evaluate_status(
        claim,
        "scientifically-validated",
        forged_contract,
    )
    bad = dict(base)
    bad["partition_bias_ci_95_upper_kcal"] = 0.5000001
    bad["material_partition_bias_passed"] = True
    assert not _evaluate_status(claim, "scientifically-validated", bad)
    bad = dict(base)
    bad["frozen_mae_kcal"] = 9.0
    bad["frozen_absolute_accuracy_passed"] = True
    assert not _evaluate_status(claim, "route2-superiority-demonstrated", bad)
    assert not _evaluate_status(claim, "scientifically-validated", bad)
    for field in [
        "reference_cycle_ablation_completed",
        "reference_cycle_ablation_same_frozen_holdout",
        "reference_cycle_ablation_same_n_by_molecule",
        "reference_cycle_ablation_nonreference_controls_locked",
        "reference_cycle_ablation_both_cycles_complete",
        "reference_cycle_ablation_artifact_verified",
    ]:
        bad = dict(base)
        bad[field] = False
        # Raw booleans are ignored because valid evidence is still present.
        assert _evaluate_status(claim, "scientifically-validated", bad)

    spoofed_missing_evidence = dict(base)
    spoofed_missing_evidence.pop("reference_cycle_evidence")
    for field in [
        "reference_cycle_ablation_completed",
        "reference_cycle_ablation_same_frozen_holdout",
        "reference_cycle_ablation_same_n_by_molecule",
        "reference_cycle_ablation_nonreference_controls_locked",
        "reference_cycle_ablation_both_cycles_complete",
        "reference_cycle_ablation_artifact_verified",
    ]:
        spoofed_missing_evidence[field] = True
    assert not _evaluate_status(
        claim,
        "scientifically-validated",
        spoofed_missing_evidence,
    )


def test_absolute_accuracy_and_partition_bias_contracts_are_numeric_and_strict():
    claim = load_json(CLAIM_PATH)
    absolute = claim["frozen_absolute_accuracy"]
    partition = claim["material_partition_bias"]

    assert absolute["direct_boolean_gate_input_allowed"] is False
    assert absolute["thresholds"] == {
        "mae_max_kcal": 1.0,
        "rmse_max_kcal": 1.5,
        "abs_mse_max_kcal": 0.5,
        "fail_count": 0,
        "abstain_count": 0,
    }
    assert partition["direct_boolean_gate_input_allowed"] is False
    assert partition["equivalence_interval_kcal"] == [-0.5, 0.5]
    assert "95% CI" in partition["definition"]
    assert "material_partition_bias_assessed" not in str(claim)

    numeric_gates = {
        absolute["gate_field"],
        partition["gate_field"],
    }
    assert numeric_gates.issubset(
        set(claim["evidence_derivation"]["derived_gate_fields"])
    )


def test_route2_superiority_metric_evaluator_and_strict_bounds():
    claim = load_json(CLAIM_PATH)
    metric = claim["route2_superiority_metric"]

    assert metric["required_fields"] == [
        "route2_delta_ci_95_upper",
        "route2_delta_ci_95_lower",
        "route2_fail_count",
        "route2_abstain_count",
        "route2_d_i_abs_error",
        "route2_sample_count",
        "route2_paired_rows",
        "route2_paired_rows_hash",
        "route2_bootstrap_indices",
        "route2_bootstrap_indices_hash",
    ]
    assert metric["formula"] == "d_i = abs(err_A(i)) - abs(err_R2(i))"
    assert "max(" not in str(metric).lower()

    base = {
        "route2_delta_ci_95_upper": -0.01,
        "route2_fail_count": 0,
        "route2_abstain_count": 0,
        "route2_delta_ci_95_lower": -0.06,
        "route2_d_i_abs_error": [0.1, -0.2, -0.3],
        "route2_sample_count": 12,
    }

    assert _evaluate_superiority_metric(metric, base)

    fail_upper_eq = dict(base)
    fail_upper_eq["route2_delta_ci_95_upper"] = 0.0
    assert not _evaluate_superiority_metric(metric, fail_upper_eq)

    fail_count = dict(base)
    fail_count["route2_fail_count"] = 1
    assert not _evaluate_superiority_metric(metric, fail_count)

    fail_abstain = dict(base)
    fail_abstain["route2_abstain_count"] = 1
    assert not _evaluate_superiority_metric(metric, fail_abstain)


def test_claim_contract_contains_no_arbitrary_sample_count_floor():
    claim = load_json(CLAIM_PATH)
    evaluator = claim["route2_superiority_metric"]["evaluator"]
    assert all(
        not (
            term.get("field") == "route2_sample_count"
            and term.get("op") in {"ge", "gt"}
            and term.get("value") == 3
        )
        for term in evaluator["terms"]
    )
