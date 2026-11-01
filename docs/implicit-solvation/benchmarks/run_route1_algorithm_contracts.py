#!/usr/bin/env python3
"""Run deterministic pre-seal Route 1 algorithm contract nodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for directory in (ROOT, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import validate_independent_obc2_default as qualification  # noqa: E402
import validate_route1_algorithm_contracts as validator  # noqa: E402


ROWS = [
    ("capability.irc.gs", "irc.gs", "tests/solvation/test_aimnet2025_implicit_paths.py::test_gs_zero_budget_returns_exact_input_ts_once_and_restores_geometry", "maple/function/dispatcher/irc/algorithm/gs.py"),
    ("capability.irc.hpc", "irc.hpc", "tests/test_irc_lifecycle_contract.py::test_exact_physical_order_and_ts_index", "maple/function/dispatcher/irc/algorithm/hpc.py"),
    ("capability.irc.lqa", "irc.lqa", "tests/test_irc_lifecycle_contract.py::test_exact_physical_order_and_ts_index", "maple/function/dispatcher/irc/algorithm/lqa.py"),
    ("capability.irc.eulerpc", "irc.eulerpc", "tests/test_irc_lifecycle_contract.py::test_eulerpc_predicted_gradient_cannot_claim_real_force_convergence", "maple/function/dispatcher/irc/algorithm/eulerpc.py"),
    ("capability.ts.prfo", "ts.prfo", "tests/test_prfo_result_contract.py::test_certified_success", "maple/function/dispatcher/ts/algorithm/PRFO.py"),
    ("capability.ts.neb.base", "ts.neb.base", "tests/test_neb_stage_contract.py::test_stage_api_zero_and_one_step_budgets_are_truthful", "maple/function/dispatcher/ts/algorithm/neb.py"),
    ("capability.ts.neb.ci", "ts.neb.ci", "tests/test_neb_stage_contract.py::test_ci_uses_its_memory_budget_and_returns_final_state", "maple/function/dispatcher/ts/algorithm/neb.py"),
    ("capability.ts.neb.ts", "ts.neb.ts", "tests/test_neb_stage_contract.py::test_only_certified_prfo_writes_nebts_file", "maple/function/dispatcher/ts/algorithm/neb.py"),
    ("capability.ts.string.base", "ts.string.base", "tests/test_string_2007_contract.py::test_ordinary_uses_full_force_then_equal_arc_reparameterization", "maple/function/dispatcher/ts/algorithm/string.py"),
    ("capability.ts.string.ci", "ts.string.ci", "tests/test_string_2007_contract.py::test_ci_formula_freezes_tau0_and_anchors_latest_ci_during_split_reparam", "maple/function/dispatcher/ts/algorithm/string.py"),
    ("capability.ts.string.ts", "ts.string.ts", "tests/test_string_2007_contract.py::test_stringts_uses_typed_prfo_params_reports_actual_force_and_writes_no_false_ts", "maple/function/dispatcher/ts/algorithm/string.py"),
    ("capability.ts.autoneb.base", "ts.autoneb.base", "tests/solvation/test_autoneb_standard_policy.py::test_insertions_preserve_one_calculator_metadata_and_endpoint_roles", "maple/function/dispatcher/ts/algorithm/autoneb.py"),
    ("capability.ts.autoneb.ci", "ts.autoneb.ci", "tests/solvation/test_autoneb_standard_policy.py::test_ci_uses_separate_budget_only_after_all_growth_stages_converge", "maple/function/dispatcher/ts/algorithm/autoneb.py"),
    ("capability.frequency.analytic", "frequency.analytic", "tests/test_curvature_contract.py::test_composed_obc_n32", "maple/function/dispatcher/curvature.py"),
    ("hostile.unknown-key", None, "tests/test_method_registry.py::test_unknown_public_method_is_rejected", "maple/function/dispatcher/method_registry.py"),
    ("hostile.forged-binding", None, "tests/solvation/test_advanced_method_capability.py::test_forged_binding_refuses", "maple/function/dispatcher/method_registry.py"),
    ("hostile.source-drift", None, "tests/test_curvature_contract.py::test_drifted_exact_aimnet_claim_refuses_instead_of_downgrading", "maple/function/dispatcher/curvature.py"),
    ("hostile.wrong-checkpoint-state", None, "tests/solvation/test_aimnet2025_candidate_validation.py::test_rejects_per_operation_precision_or_checkpoint_spoof", "maple/function/calculator/aimnet/_aimnet2025_calculator.py"),
    ("hostile.unsupported-charge-spin", None, "tests/test_curvature_contract.py::test_bound_profile_refuses_unqualified_charge_and_spin", "maple/function/dispatcher/curvature.py"),
    ("hostile.unsupported-provider-gb", None, "tests/test_curvature_contract.py::test_composed_binding_refuses_nonpolar_drift", "maple/function/dispatcher/curvature.py"),
    ("hostile.stale-energy-force", None, "tests/test_prfo_result_contract.py::test_nonfinite_final_state_never_publishes_ts", "maple/function/dispatcher/ts/algorithm/PRFO.py"),
    ("hostile.bounded-outcome", None, "tests/test_prfo_result_contract.py::test_budget_returns_typed_result", "maple/function/dispatcher/ts/algorithm/PRFO.py"),
    ("hostile.false-ts", None, "tests/test_neb_stage_contract.py::test_bounded_prfo_candidate_stays_separate_and_writes_no_ts", "maple/function/dispatcher/ts/algorithm/neb.py"),
    ("hostile.unknown-stage", None, "tests/test_neb_stage_contract.py::test_unknown_stage_refuses_before_evaluation", "maple/function/dispatcher/ts/algorithm/neb.py"),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = qualification.source_snapshot(require_clean=True)
    cache = {}
    rows = {}
    for row_id, capability, node, implementation in ROWS:
        if node not in cache:
            cache[node] = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", node], cwd=ROOT,
                text=True, capture_output=True,
            )
        completed = cache[node]
        test_file = node.split("::", 1)[0]
        expected = (
            f"capability-contract-pass:{capability}"
            if capability is not None
            else f"hostile-guard-pass:{row_id}"
        )
        rows[row_id] = {
            "capability_key": capability,
            "implementation_files_sha256": {implementation: _sha256(ROOT / implementation)},
            "test_file": test_file,
            "test_node": node,
            "test_file_sha256": _sha256(ROOT / test_file),
            "fixture_identity": {"kind": "deterministic-cpu-pytest-node", "node": node},
            "expected": expected,
            "observed": expected if completed.returncode == 0 else completed.stdout + completed.stderr,
            "passed": completed.returncode == 0,
        }
    after = qualification.source_snapshot(require_clean=True)
    artifact = {
        "schema_version": validator.SCHEMA_VERSION,
        "artifact_type": "route1-advanced-algorithm-contracts",
        "source_before": before,
        "source_after": after,
        "capability_keys": sorted(validator.EXPECTED_CAPABILITIES),
        "rows": rows,
    }
    artifact["content_sha256"] = hashlib.sha256(validator.canonical_payload(artifact)).hexdigest()
    validator.validate_artifact(artifact, expected_source_snapshot=before)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
