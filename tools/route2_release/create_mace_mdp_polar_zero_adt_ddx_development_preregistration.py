#!/usr/bin/env python3
"""Freeze the zero-training v3 hybrid before its 505 development evaluation.

The creator is permitted only after the exact twelve-case Stage-B mechanism
screen has passed.  It freezes one immutable prediction method and one exact
505 development partition.  The prediction runner may parse the historical
MNSol table to recover geometries, but it is forbidden to use or emit target
values; target comparison occurs only after all 505 prediction records close.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

ARTIFACT_ID = "route2-mace-mdp-polar-zero-adt-ddx-development-prereg-v1"
EXPECTED_RECORD_COUNT = 505
EXPECTED_STAGE_B_COUNT = 12
EXPECTED_PROFILE_ID = (
    "route2-research-macepolar-zero-point-mdp-alpha-polar-residual-"
    "role-separated-adt-ddx-operational-v3"
)
INPUT_FILES = {
    "dataset_zip_sha256": "MNSolDatabase_v2012.zip",
    "development_selection_sha256": (
        "route2-mnsol-development-selection-v1.private.json"
    ),
    "protocol_sha256": "route2-mnsol-protocol-v1.json",
    "pilot_selection_sha256": "route2-mnsol-pilot-selection-v1.json",
}
SOURCE_FILES = (
    "maple/function/calculator/extra_correction/implicit/pyscf_smd_cds.py",
    "maple/function/calculator/extra_correction/implicit/smd_cds.py",
    "maple/function/route2_solvents.py",
    "maple/solvation/api/profiles.py",
    "maple/solvation/continuum/separated_source_adt_ddx.py",
    "maple/solvation/continuum/separated_source_ddx.py",
    "maple/solvation/coupling/adt_radial_shape.py",
    "maple/solvation/coupling/atomic_displacement_lift.py",
    "maple/solvation/coupling/neutral_atom_penetration.py",
    "maple/solvation/experimental/mace_mdp_polar_adt_ddx.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_adt.py",
    "maple/solvation/models/mace_mdp_polar_hybrid.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_separated.py",
    "maple/solvation/release/uniform_response_manifold.py",
    "maple/solvation/release/uniform_susceptibility_replacement.py",
    "maple/solvation/solvent_terms.py",
    "tools/route2_release/aggregate_mace_mdp_polar_zero_adt_ddx_development.py",
    "tools/route2_release/create_mace_mdp_polar_zero_adt_ddx_development_preregistration.py",
    "tools/route2_release/create_mace_mdp_polar_zero_adt_coupled_screen_preregistration.py",
    "tools/route2_release/create_mace_mdp_polar_zero_adt_stage_b_preregistration.py",
    "tools/route2_release/run_mace_mdp_polar_zero_adt_ddx_development.py",
    "tools/route2_release/run_mace_mdp_polar_zero_adt_coupled_screen.py",
    "tools/route2_release/run_mace_mdp_polar_zero_adt_stage_b.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *args), text=True, stderr=subprocess.DEVNULL
    ).strip()


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_selection(source_root: Path, input_root: Path) -> None:
    benchmark_root = source_root / "docs/implicit-solvation/benchmarks"
    sys.path.insert(0, str(benchmark_root))
    from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
    from mnsol_partition import validate_frozen_mnsol_partition_selection

    protocol = load_mnsol_protocol(input_root / INPUT_FILES["protocol_sha256"])
    dataset = load_mnsol_v2012(input_root / INPUT_FILES["dataset_zip_sha256"], protocol)
    selection = json.loads(
        (input_root / INPUT_FILES["development_selection_sha256"]).read_text()
    )
    pilot = json.loads((input_root / INPUT_FILES["pilot_selection_sha256"]).read_text())
    rows = validate_frozen_mnsol_partition_selection(
        selection, dataset, protocol, pilot
    )
    _require(
        len(rows) == EXPECTED_RECORD_COUNT,
        "Frozen development selection is not exactly 505 rows.",
    )


def _validate_stage_b(
    *,
    source_root: Path,
    stage_b_preregistration: Path,
    stage_b_aggregate: Path,
    stage_b_records_dir: Path,
    stage_a_dataset: Path,
    mdp_checkpoint: Path,
    polar_checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Revalidate and exactly reproduce the frozen Stage-B pass."""

    sys.path.insert(0, str(source_root))
    from tools.route2_release import (
        run_mace_mdp_polar_zero_adt_stage_b as stage_b_runner,
    )

    prereg = stage_b_runner._load_preregistration(stage_b_preregistration)
    stage_b_runner._validate_lineage_and_inputs(
        prereg,
        dataset=stage_a_dataset,
        mdp_checkpoint=mdp_checkpoint,
        polar_checkpoint=polar_checkpoint,
    )
    aggregate = json.loads(stage_b_aggregate.read_text())
    digest = aggregate.pop("aggregate_sha256", None)
    _require(
        digest == stage_b_runner._canonical_sha256(aggregate),
        "Stage-B aggregate self hash changed.",
    )
    aggregate["aggregate_sha256"] = digest
    _require(
        aggregate.get("artifact") == f"{stage_b_runner.ARTIFACT_ID}-aggregate"
        and aggregate.get("status") == "pass-stage-b"
        and aggregate.get("record_count") == EXPECTED_STAGE_B_COUNT
        and aggregate.get("pass_count") == EXPECTED_STAGE_B_COUNT
        and aggregate.get("failure_count") == 0
        and aggregate.get("stage_b_passed") is True,
        "The exact twelve-case Stage-B mechanism screen did not pass.",
    )
    _require(
        aggregate.get("preregistration_file_sha256") == _sha256(stage_b_preregistration)
        and aggregate.get("preregistration_artifact_sha256")
        == prereg.get("preregistration_sha256"),
        "Stage-B aggregate belongs to another preregistration.",
    )
    _require(
        aggregate.get("accuracy_claim_made") is False
        and aggregate.get("capabilities_admitted")
        == {tier: False for tier in ("E", "F", "H", "V", "M")},
        "Stage B exceeded its mechanism-only claim boundary.",
    )
    with tempfile.TemporaryDirectory(prefix="maple-stage-b-reaggregate-") as temporary:
        regenerated = stage_b_runner.aggregate(
            SimpleNamespace(
                preregistration=stage_b_preregistration,
                dataset=stage_a_dataset,
                mdp_checkpoint=mdp_checkpoint,
                polar_checkpoint=polar_checkpoint,
                output_dir=stage_b_records_dir,
                aggregate_output=Path(temporary) / "aggregate.json",
            )
        )
    _require(
        regenerated == aggregate,
        "Stage-B aggregate no longer regenerates from its twelve records.",
    )
    return prereg, aggregate


def create(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    input_root = args.input_root.expanduser().resolve(strict=True)
    records_dir = args.records_dir.expanduser().resolve()
    preregistration = args.output.expanduser().resolve()
    runner = args.runner.expanduser().resolve(strict=True)
    aggregator = args.aggregator.expanduser().resolve(strict=True)
    stage_b_preregistration = args.stage_b_preregistration.expanduser().resolve(
        strict=True
    )
    stage_b_aggregate = args.stage_b_aggregate.expanduser().resolve(strict=True)
    stage_b_records_dir = args.stage_b_records_dir.expanduser().resolve(strict=True)
    stage_a_dataset = args.stage_a_dataset.expanduser().resolve(strict=True)
    mdp_checkpoint = args.mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.polar_checkpoint.expanduser().resolve(strict=True)

    if preregistration.exists():
        raise FileExistsError(preregistration)
    if records_dir.exists() and any(records_dir.iterdir()):
        raise ValueError("Records directory must not contain pre-evaluation files.")
    if _git(source_root, "status", "--porcelain=v1"):
        raise ValueError("Source snapshot must be a clean Git worktree.")
    source_head = _git(source_root, "rev-parse", "HEAD")
    _require(
        runner.parent.parent.parent == source_root,
        "Runner must belong to the frozen source snapshot.",
    )
    _require(
        aggregator.parent.parent.parent == source_root,
        "Aggregator must belong to the frozen source snapshot.",
    )

    stage_b_prereg, stage_b_result = _validate_stage_b(
        source_root=source_root,
        stage_b_preregistration=stage_b_preregistration,
        stage_b_aggregate=stage_b_aggregate,
        stage_b_records_dir=stage_b_records_dir,
        stage_a_dataset=stage_a_dataset,
        mdp_checkpoint=mdp_checkpoint,
        polar_checkpoint=polar_checkpoint,
    )
    input_hashes = {
        key: _sha256(input_root / relative) for key, relative in INPUT_FILES.items()
    }
    _validate_selection(source_root, input_root)
    source_hashes = {name: _sha256(source_root / name) for name in SOURCE_FILES}
    payload: dict[str, object] = {
        "artifact_id": ARTIFACT_ID,
        "schema_version": 1,
        "status": "locked-after-stage-b-pass-before-zero-adt-505",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "development",
        "record_count": EXPECTED_RECORD_COUNT,
        "confirmation_partition_opened": False,
        "development_targets_previously_opened": True,
        "profile_selected_from_experimental_targets": False,
        "prediction_runner_uses_experimental_targets": False,
        "prediction_runner_emits_experimental_targets": False,
        "do_not_commit_private_outputs": True,
        "fitting_or_calibration_permitted": False,
        "checkpoint_or_method_selection_after_results_permitted": False,
        "hard_accuracy_target": {
            "metric": "mean_absolute_error_kcal_mol",
            "comparison": "<=",
            "threshold_kcal_mol": 1.5,
        },
        "provider_failure_rule": {
            "required_success_count": EXPECTED_RECORD_COUNT,
            "maximum_provider_failures": 0,
        },
        "decision_rule": (
            "primary pass iff all 505 frozen predictions succeed and the later "
            "stock-CDS M1 aggregate MAE is <=1.5 kcal/mol; separately report "
            "the 1.0 kcal/mol stretch target"
        ),
        "reporting_metrics": [
            "MAE",
            "RMSE",
            "maximum absolute error",
            "q95 absolute error",
            "mean signed error",
            "per-solvent metrics",
            "provider failure count",
        ],
        **input_hashes,
        "input_root": str(input_root),
        "source_root": str(source_root),
        "records_path": str(records_dir),
        "preregistration_path": str(preregistration),
        "source_git_head": source_head,
        "source_files_sha256": source_hashes,
        "runner_path": str(runner),
        "runner_sha256": _sha256(runner),
        "aggregator_path": str(aggregator),
        "aggregator_sha256": _sha256(aggregator),
        "stage_b_preregistration_path": str(stage_b_preregistration),
        "stage_b_preregistration_file_sha256": _sha256(stage_b_preregistration),
        "stage_b_preregistration_artifact_sha256": stage_b_prereg[
            "preregistration_sha256"
        ],
        "stage_b_aggregate_path": str(stage_b_aggregate),
        "stage_b_aggregate_file_sha256": _sha256(stage_b_aggregate),
        "stage_b_aggregate_artifact_sha256": stage_b_result["aggregate_sha256"],
        "mdp_checkpoint_sha256": _sha256(mdp_checkpoint),
        "polar_checkpoint_sha256": _sha256(polar_checkpoint),
        "selection_history": {
            "same_frozen_505_development_partition_as_prior_profiles": True,
            "development_targets_already_known": True,
            "zero_adt_profile_selected_from_stage_a_b_without_solvation_targets": True,
            "zero_adt_505_results_used_for_method_selection": False,
            "confirmation_partition_remains_sealed": True,
        },
        "method": {
            "profile_id": EXPECTED_PROFILE_ID,
            "scalar": (
                "G_ddX,pol(MACE-POLAR zero-field permanent point l<=1 + "
                "MACE-POLAR nonuniform Gaussian residual l<=1 with its uniform "
                "tangent removed + canonical MDP-alpha ADT dipoles) + PySCF "
                "stock SMD CDS"
            ),
            "permanent_source": "official frozen MACE-POLAR zero-field point q/p",
            "nonuniform_response": (
                "official frozen MACE-POLAR-1-M zero-anchored residual response "
                "minus its exact zero-field uniform tangent"
            ),
            "uniform_response": (
                "canonical ADT lift of the official frozen MACE-MDP molecular "
                "polarizability"
            ),
            "mace_polar_long_range_evaluator": (
                "graph-longrange-analytic-gaussian-multipole-realspace-v1"
            ),
            "continuum_model": "pcm",
            "lmax": 12,
            "n_lebedev": 1202,
            "solver_tolerance": 1.0e-12,
            "eta": 0.1,
            "n_proc": 1,
            "mdp_device": "cpu",
            "polar_device": "cuda",
            "standard_state": "MNSol protocol gas-to-solution 1M convention",
            "cds": "unmodified PySCF 2.13.1 stock SMD CDS",
            "fitting_or_calibration": False,
        },
        "claim_boundary": (
            "Prospectively frozen zero-training v3 development energy decision. "
            "The 505 development targets were seen by older profiles, but this "
            "exact identity was selected only from source diagnostics plus the "
            "target-free Stage-A/B mechanism screens. Prediction records contain "
            "no target or error values. The 148-row confirmation partition remains "
            "sealed. No force, virial, Hessian, frequency, optimization, MD, or "
            "release accuracy is established."
        ),
    }
    payload["preregistration_sha256"] = _canonical_sha256(payload)
    preregistration.parent.mkdir(parents=True, exist_ok=True)
    with preregistration.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    preregistration.chmod(0o444)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--aggregator", type=Path, required=True)
    parser.add_argument("--stage-b-preregistration", type=Path, required=True)
    parser.add_argument("--stage-b-aggregate", type=Path, required=True)
    parser.add_argument("--stage-b-records-dir", type=Path, required=True)
    parser.add_argument("--stage-a-dataset", type=Path, required=True)
    parser.add_argument("--mdp-checkpoint", type=Path, required=True)
    parser.add_argument("--polar-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = create(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(_sha256(args.output.expanduser().resolve()))


if __name__ == "__main__":
    main()
