"""Canonical source-bound software contracts for Route 1 advanced methods."""

from __future__ import annotations

from typing import NamedTuple

from maple.function.dispatcher.method_registry import CapabilityKey

EXPECTED_CAPABILITIES = frozenset(key.value for key in CapabilityKey)
MATRIX_CORE_MANIFEST_SHA256 = (
    "7f0b0883b4f0e449f887691ae3cee0c9265f51fd370cebdd68ea6eca91597ad6"
)


class ContractRow(NamedTuple):
    row_id: str
    capability: str | None
    test_node: str
    implementation: str


CANONICAL_ROWS = (
    ContractRow(
        "capability.irc.gs",
        "irc.gs",
        "tests/solvation/test_aimnet2025_implicit_paths.py::test_gs_zero_budget_returns_exact_input_ts_once_and_restores_geometry",
        "maple/function/dispatcher/irc/algorithm/gs.py",
    ),
    ContractRow(
        "capability.irc.hpc",
        "irc.hpc",
        "tests/test_irc_lifecycle_contract.py::test_exact_physical_order_and_ts_index",
        "maple/function/dispatcher/irc/algorithm/hpc.py",
    ),
    ContractRow(
        "capability.irc.lqa",
        "irc.lqa",
        "tests/test_irc_lifecycle_contract.py::test_exact_physical_order_and_ts_index",
        "maple/function/dispatcher/irc/algorithm/lqa.py",
    ),
    ContractRow(
        "capability.irc.eulerpc",
        "irc.eulerpc",
        "tests/test_irc_lifecycle_contract.py::test_eulerpc_predicted_gradient_cannot_claim_real_force_convergence",
        "maple/function/dispatcher/irc/algorithm/eulerpc.py",
    ),
    ContractRow(
        "capability.ts.prfo",
        "ts.prfo",
        "tests/test_prfo_result_contract.py::test_certified_success",
        "maple/function/dispatcher/ts/algorithm/PRFO.py",
    ),
    ContractRow(
        "capability.ts.neb.base",
        "ts.neb.base",
        "tests/test_neb_stage_contract.py::test_stage_api_zero_and_one_step_budgets_are_truthful",
        "maple/function/dispatcher/ts/algorithm/neb.py",
    ),
    ContractRow(
        "capability.ts.neb.ci",
        "ts.neb.ci",
        "tests/test_neb_stage_contract.py::test_ci_uses_its_memory_budget_and_returns_final_state",
        "maple/function/dispatcher/ts/algorithm/neb.py",
    ),
    ContractRow(
        "capability.ts.neb.ts",
        "ts.neb.ts",
        "tests/test_neb_stage_contract.py::test_only_certified_prfo_writes_nebts_file",
        "maple/function/dispatcher/ts/algorithm/neb.py",
    ),
    ContractRow(
        "capability.ts.string.base",
        "ts.string.base",
        "tests/test_string_2007_contract.py::test_ordinary_uses_full_force_then_equal_arc_reparameterization",
        "maple/function/dispatcher/ts/algorithm/string.py",
    ),
    ContractRow(
        "capability.ts.string.ci",
        "ts.string.ci",
        "tests/test_string_2007_contract.py::test_ci_formula_freezes_tau0_and_anchors_latest_ci_during_split_reparam",
        "maple/function/dispatcher/ts/algorithm/string.py",
    ),
    ContractRow(
        "capability.ts.string.ts",
        "ts.string.ts",
        "tests/test_string_2007_contract.py::test_stringts_uses_typed_prfo_params_reports_actual_force_and_writes_no_false_ts",
        "maple/function/dispatcher/ts/algorithm/string.py",
    ),
    ContractRow(
        "capability.ts.autoneb.base",
        "ts.autoneb.base",
        "tests/solvation/test_autoneb_standard_policy.py::test_insertions_preserve_one_calculator_metadata_and_endpoint_roles",
        "maple/function/dispatcher/ts/algorithm/autoneb.py",
    ),
    ContractRow(
        "capability.ts.autoneb.ci",
        "ts.autoneb.ci",
        "tests/solvation/test_autoneb_standard_policy.py::test_ci_uses_separate_budget_only_after_all_growth_stages_converge",
        "maple/function/dispatcher/ts/algorithm/autoneb.py",
    ),
    ContractRow(
        "capability.frequency.analytic",
        "frequency.analytic",
        "tests/test_curvature_contract.py::test_composed_obc_n32",
        "maple/function/dispatcher/curvature.py",
    ),
    ContractRow(
        "hostile.unknown-key",
        None,
        "tests/test_method_registry.py::test_unknown_public_method_is_rejected",
        "maple/function/dispatcher/method_registry.py",
    ),
    ContractRow(
        "hostile.forged-binding",
        None,
        "tests/solvation/test_advanced_method_capability.py::test_forged_binding_refuses",
        "maple/function/dispatcher/method_registry.py",
    ),
    ContractRow(
        "hostile.source-drift",
        None,
        "tests/test_curvature_contract.py::test_drifted_exact_aimnet_claim_refuses_instead_of_downgrading",
        "maple/function/dispatcher/curvature.py",
    ),
    ContractRow(
        "hostile.wrong-checkpoint-state",
        None,
        "tests/solvation/test_aimnet2025_candidate_validation.py::test_rejects_per_operation_precision_or_checkpoint_spoof",
        "maple/function/calculator/aimnet/_aimnet2025_calculator.py",
    ),
    ContractRow(
        "hostile.unsupported-charge-spin",
        None,
        "tests/test_curvature_contract.py::test_bound_profile_refuses_unqualified_charge_and_spin",
        "maple/function/dispatcher/curvature.py",
    ),
    ContractRow(
        "hostile.unsupported-provider-gb",
        None,
        "tests/test_curvature_contract.py::test_composed_binding_refuses_nonpolar_drift",
        "maple/function/dispatcher/curvature.py",
    ),
    ContractRow(
        "hostile.stale-energy-force",
        None,
        "tests/test_prfo_result_contract.py::test_nonfinite_final_state_never_publishes_ts",
        "maple/function/dispatcher/ts/algorithm/PRFO.py",
    ),
    ContractRow(
        "hostile.bounded-outcome",
        None,
        "tests/test_prfo_result_contract.py::test_budget_returns_typed_result",
        "maple/function/dispatcher/ts/algorithm/PRFO.py",
    ),
    ContractRow(
        "hostile.false-ts",
        None,
        "tests/test_neb_stage_contract.py::test_bounded_prfo_candidate_stays_separate_and_writes_no_ts",
        "maple/function/dispatcher/ts/algorithm/neb.py",
    ),
    ContractRow(
        "hostile.unknown-stage",
        None,
        "tests/test_neb_stage_contract.py::test_unknown_stage_refuses_before_evaluation",
        "maple/function/dispatcher/ts/algorithm/neb.py",
    ),
)
CANONICAL_ROWS_BY_ID = {row.row_id: row for row in CANONICAL_ROWS}
HOSTILE_ROW_IDS = frozenset(
    row.row_id for row in CANONICAL_ROWS if row.capability is None
)
if (
    len(CANONICAL_ROWS_BY_ID) != len(CANONICAL_ROWS)
    or {row.capability for row in CANONICAL_ROWS if row.capability is not None}
    != EXPECTED_CAPABILITIES
):
    raise RuntimeError("canonical algorithm contract map is incomplete or duplicated")


def expected_outcome(spec: ContractRow) -> str:
    return (
        f"capability-contract-pass:{spec.capability}"
        if spec.capability is not None
        else f"hostile-guard-pass:{spec.row_id}"
    )


def fixture_identity(spec: ContractRow) -> dict:
    return {"kind": "deterministic-cpu-pytest-node", "node": spec.test_node}


def validate_sealed_contracts(
    section: dict,
    *,
    source_git_sha: str,
    source_inventory_aggregate_sha256: str,
    qualification_sources: dict[str, str],
) -> frozenset[str]:
    """Validate sealed rows against the canonical map and frozen source inventory."""
    if not isinstance(qualification_sources, dict):
        raise ValueError("algorithm contract qualification source inventory is missing")
    if not isinstance(section, dict) or section.get("schema_version") != 1:
        raise ValueError("algorithm contract evidence is missing")
    if (
        section.get("source_git_sha") != source_git_sha
        or section.get("source_inventory_aggregate_sha256")
        != source_inventory_aggregate_sha256
    ):
        raise ValueError("algorithm contract source binding mismatch")
    if section.get("capability_keys") != sorted(EXPECTED_CAPABILITIES):
        raise ValueError("algorithm contract capability set is malformed")
    rows = section.get("rows")
    if not isinstance(rows, dict) or set(rows) != set(CANONICAL_ROWS_BY_ID):
        raise ValueError("algorithm contract canonical rows are missing or changed")
    for row_id, spec in CANONICAL_ROWS_BY_ID.items():
        row = rows[row_id]
        test_file = spec.test_node.split("::", 1)[0]
        outcome = expected_outcome(spec)
        if (
            spec.implementation not in qualification_sources
            or test_file not in qualification_sources
        ):
            raise ValueError(f"algorithm contract frozen source is missing: {row_id}")
        if not isinstance(row, dict) or row.get("passed") is not True:
            raise ValueError(f"algorithm contract row did not pass: {row_id}")
        if row.get("capability_key") != spec.capability:
            raise ValueError(f"algorithm contract capability binding changed: {row_id}")
        if row.get("implementation_files_sha256") != {
            spec.implementation: qualification_sources.get(spec.implementation)
        }:
            raise ValueError(
                f"algorithm contract implementation binding changed: {row_id}"
            )
        if (
            row.get("test_file") != test_file
            or row.get("test_node") != spec.test_node
            or row.get("test_file_sha256") != qualification_sources.get(test_file)
        ):
            raise ValueError(f"algorithm contract test binding changed: {row_id}")
        if row.get("fixture_identity") != fixture_identity(spec):
            raise ValueError(f"algorithm contract fixture binding changed: {row_id}")
        if row.get("expected") != outcome or row.get("observed") != outcome:
            raise ValueError(f"algorithm contract outcome mismatch: {row_id}")
    return EXPECTED_CAPABILITIES
