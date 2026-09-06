#!/usr/bin/env python3
"""Freeze the one-time analytic/L-BFGS A3 optimization closure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tools.route2_release import (  # noqa: E402
    continue_vqm24_mdp_p13_a3_lbfgs as closure,
)


SELF_REPO_PATH = (
    "tools/route2_release/create_vqm24_mdp_p13_a3_o2_preregistration.py"
)
ARTIFACT = closure.PREREGISTRATION_ARTIFACT
DEFAULT_PILOT = Path(
    "/home/axie/.cache/maple-route2-hybrid-runs/"
    "vqm24-mdp-p13-passive-response-pilot-v1"
)
DEFAULT_OBSERVABLE_PARENT = SOURCE_ROOT / (
    "docs/route2/preregistrations/"
    "vqm24-observable-training-batch-dense-v2.json"
)
DEFAULT_SELECTION = SOURCE_ROOT / (
    "docs/route2/preregistrations/"
    "vqm24-nonuniform-qm-geometry-selection-v1.json"
)
DEFAULT_MDP_FEATURES = Path(
    "/home/axie/.cache/maple-route2-hybrid-runs/"
    "vqm24-frozen-mdp-features-v1"
)
DEFAULT_PILOT_PREREGISTRATION = SOURCE_ROOT / (
    "docs/route2/preregistrations/"
    "vqm24-mdp-p13-block-passive-response-pilot-v1.json"
)
DEFAULT_PRO_EVIDENCE = SOURCE_ROOT / (
    "docs/route2/evidence/"
    "mdp-polar-p13-postpilot-pro-20260827/answer.md"
)
DEFAULT_OUTPUT = SOURCE_ROOT / (
    "docs/route2/preregistrations/"
    "vqm24-mdp-p13-a3-o2-lbfgs-closure-v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _grouped_folds(records: list[dict[str, object]]) -> list[list[str]]:
    by_stratum: dict[str, list[dict[str, object]]] = {}
    for record in records:
        if record["split"] != "train":
            continue
        by_stratum.setdefault(str(record["stratum"]), []).append(record)
    folds: list[list[str]] = [[] for _ in range(4)]
    offset = 0
    for stratum in sorted(by_stratum):
        members = sorted(
            by_stratum[stratum], key=lambda item: str(item["record_sha256"])
        )
        for index, record in enumerate(members):
            folds[(offset + index) % len(folds)].append(str(record["record_sha256"]))
        offset = (offset + len(members)) % len(folds)
    for fold in folds:
        fold.sort()
    if sorted(value for fold in folds for value in fold) != sorted(
        str(record["record_sha256"])
        for record in records
        if record["split"] == "train"
    ):
        raise RuntimeError("Grouped folds do not cover the train split exactly.")
    return folds


def run(args: argparse.Namespace) -> dict[str, object]:
    pilot_root = args.pilot_result.expanduser().resolve(strict=True)
    observable_parent_path = args.observable_parent.expanduser().resolve(strict=True)
    selection_path = args.selection.expanduser().resolve(strict=True)
    mdp_root = args.mdp_features.expanduser().resolve(strict=True)
    pilot_preregistration_path = (
        args.pilot_preregistration.expanduser().resolve(strict=True)
    )
    pro_path = args.pro_evidence.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    pilot_result_path = pilot_root / "result.json"
    pilot_checkpoint_path = pilot_root / "A3" / "head.pt"
    pilot_result = json.loads(pilot_result_path.read_text())
    pilot_preregistration = json.loads(pilot_preregistration_path.read_text())
    observable_parent = json.loads(observable_parent_path.read_text())
    selection = json.loads(selection_path.read_text())
    if (
        pilot_result["status"] != "fail-development-response-pilot"
        or pilot_result["selected_candidate_for_grouped_cv"] is not None
        or pilot_result["candidates"]["A3"]["gate_passed"] is not False
    ):
        raise RuntimeError("O2 requires the exact failed A3 development pilot.")
    folds = _grouped_folds(selection["records"])
    response_path = "maple/solvation/models/passive_p13_response.py"
    factor_path = "maple/solvation/models/sparse_passive_factor.py"
    tests = [
        "tests/route2_vnext/test_passive_p13_response.py",
        "tests/route2_vnext/test_sparse_passive_factor.py",
    ]
    payload: dict[str, object] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-o2-continuation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_pilot": {
            "result_path": str(pilot_result_path),
            "result_sha256": _sha256(pilot_result_path),
            "result_payload_sha256": pilot_result["result_sha256"],
            "a3_checkpoint_path": str(pilot_checkpoint_path),
            "a3_checkpoint_sha256": _sha256(pilot_checkpoint_path),
            "pilot_preregistration_path": str(
                pilot_preregistration_path.relative_to(SOURCE_ROOT)
            ),
            "pilot_preregistration_file_sha256": _sha256(
                pilot_preregistration_path
            ),
            "pilot_preregistration_sha256": pilot_preregistration[
                "preregistration_sha256"
            ],
        },
        "observable_parent": {
            "path": str(observable_parent_path.relative_to(SOURCE_ROOT)),
            "file_sha256": _sha256(observable_parent_path),
            "preregistration_sha256": observable_parent["preregistration_sha256"],
        },
        "inputs": {
            "mdp_feature_manifest_path": str(mdp_root / "manifest.json"),
            "mdp_feature_manifest_sha256": _sha256(mdp_root / "manifest.json"),
            "train_record_ids": [
                record["record_id"] for record in observable_parent["records"]
            ],
            "validation_and_blind_data_unopened": True,
        },
        "source": {
            "preregistration_builder_path": SELF_REPO_PATH,
            "preregistration_builder_sha256": _sha256(
                SOURCE_ROOT / SELF_REPO_PATH
            ),
            "runner_path": closure.SELF_REPO_PATH,
            "runner_sha256": _sha256(SOURCE_ROOT / closure.SELF_REPO_PATH),
            "analytic_response": {
                "path": response_path,
                "sha256": _sha256(SOURCE_ROOT / response_path),
            },
            "factor": {
                "path": factor_path,
                "sha256": _sha256(SOURCE_ROOT / factor_path),
            },
            "tests": [
                {"path": path, "sha256": _sha256(SOURCE_ROOT / path)}
                for path in tests
            ],
            "pro_answer_path": str(pro_path.relative_to(SOURCE_ROOT)),
            "pro_answer_sha256": _sha256(pro_path),
        },
        "backend_gates": {
            "source_atol": 1.0e-11,
            "source_rtol": 1.0e-10,
            "energy_work_identity_atol": 1.0e-11,
            "energy_work_identity_rtol": 1.0e-10,
            "parameter_gradient_atol": 1.0e-9,
            "parameter_gradient_rtol": 1.0e-8,
            "gradcheck_required": True,
            "gradgradcheck_required": True,
            "rotation_and_edge_orientation_tests_required": True,
        },
        "optimizer": {
            "name": "torch.optim.LBFGS",
            "line_search_fn": "strong_wolfe",
            "learning_rate": closure.LBFGS_LEARNING_RATE,
            "history_size": closure.LBFGS_HISTORY_SIZE,
            "maximum_accepted_outer_steps": closure.MAXIMUM_STEPS,
            "maximum_full_objective_evaluations": (
                closure.MAXIMUM_OBJECTIVE_EVALUATIONS
            ),
            "maximum_evaluations_per_step": closure.LBFGS_MAX_EVAL_PER_STEP,
            "tolerance_grad": closure.LBFGS_TOLERANCE_GRAD,
            "tolerance_change": closure.LBFGS_TOLERANCE_CHANGE,
            "same_data_objective": "0.82 response-MEP + 0.18 dipole-response",
            "same_regularization": (
                "0.5 * 1e-4 * sum(theta^2), exactly equivalent to prior "
                "torch.optim.Adam weight_decay=1e-4"
            ),
            "audit_metrics_used_for_early_stopping": False,
        },
        "plateau": {
            "block_steps": closure.PLATEAU_BLOCK_STEPS,
            "consecutive_nonoverlapping_blocks": closure.PLATEAU_BLOCK_COUNT,
            "maximum_relative_best_objective_improvement_per_block": (
                closure.PLATEAU_RELATIVE_IMPROVEMENT
            ),
            "select_lowest_total_training_objective_only": True,
            "hard_cap_without_plateau_is_terminal": True,
        },
        "gates": pilot_preregistration["gates"],
        "conditional_A4": {
            "authorized_only_after_plateau_and_failed_A3_gate": True,
            "precommitted_parameter_cap": 4096,
            "parameter_cap_precommitted_before_A3_results": True,
            "maximum_multiplicity_PCA_channels": {
                "even_scalars_l0": 8,
                "odd_vectors_l1": 4,
            },
            "PCA_same_mixing_for_all_m_components": True,
            "PCA_must_be_recomputed_inside_each_future_CV_fold": True,
            "opposite_parity_mixing_forbidden": True,
            "O3_reflection_gate_required": True,
            "new_latent_couplings_initialize_to_zero": True,
            "new_message_passing_forbidden": True,
            "frozen_weights_must_remain_coordinate_differentiable": True,
            "field_dependent_or_post_source_features_forbidden": True,
            "upstream_provenance_audit_required_before_A4": True,
            "one_A4_run_only": True,
            "A4_failure_terminates_branch": True,
        },
        "cross_factor_C": {
            "authorized": False,
            "reason": (
                "four input modes identify susceptibility action on rank<=4 "
                "only; cross and diagonal factor contributions are confounded"
            ),
        },
        "future_grouped_train_cv": {
            "status": "fold-identities-frozen-but-not-run",
            "fold_record_sha256s": {
                f"fold-{index}": fold for index, fold in enumerate(folds)
            },
            "all_modes_and_probe_frames_stay_with_formula": True,
            "not_used_to_choose_A3_vs_A4": True,
            "run_only_after_candidate_search_closes": True,
        },
        "opening_policy": {
            "validation_opens_only_after_train_gates_grouped_CV_and_ddPCM_math": True,
            "blind_opens_only_after_validation_passes": True,
            "validation_or_blind_informed_repair_forbidden": True,
        },
        "claim_boundary": {
            "same_A3_architecture_and_objective": True,
            "development_train_formulas_only": True,
            "validation_or_blind_formula_opened": False,
            "experimental_solvation_target_read": False,
            "pcm_or_cavity_used": False,
            "model_selected_or_admitted": False,
            "maple_capability_admitted": False,
        },
    }
    payload["preregistration_sha256"] = _canonical_sha256(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output_path.chmod(0o444)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-result", type=Path, default=DEFAULT_PILOT)
    parser.add_argument(
        "--observable-parent", type=Path, default=DEFAULT_OBSERVABLE_PARENT
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--mdp-features", type=Path, default=DEFAULT_MDP_FEATURES)
    parser.add_argument(
        "--pilot-preregistration",
        type=Path,
        default=DEFAULT_PILOT_PREREGISTRATION,
    )
    parser.add_argument("--pro-evidence", type=Path, default=DEFAULT_PRO_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
