#!/usr/bin/env python3
"""Freeze the target-free canonical-ADT hybrid before its 505 evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ARTIFACT_ID = "route2-mace-mdp-polar-canonical-adt-ddx-development-prereg-v1"
EXPECTED_RECORD_COUNT = 505
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
    "maple/solvation/continuum/separated_source_adt_ddx.py",
    "maple/solvation/continuum/separated_source_ddx.py",
    "maple/solvation/coupling/atomic_displacement_lift.py",
    "maple/solvation/experimental/mace_mdp_polar_adt_ddx.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_adt.py",
    "maple/solvation/models/mace_mdp_polar_hybrid.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_separated.py",
    "maple/solvation/solvent_terms.py",
    "tools/route2_release/aggregate_mace_mdp_polar_canonical_adt_ddx_development.py",
    "tools/route2_release/create_mace_mdp_polar_canonical_adt_ddx_development_preregistration.py",
    "tools/route2_release/run_mace_mdp_polar_canonical_adt_ddx_development.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *args), text=True, stderr=subprocess.DEVNULL
    ).strip()


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
    if len(rows) != EXPECTED_RECORD_COUNT:
        raise ValueError("Frozen development selection is not exactly 505 rows.")


def create(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    input_root = args.input_root.expanduser().resolve(strict=True)
    records_dir = args.records_dir.expanduser().resolve()
    preregistration = args.output.expanduser().resolve()
    runner = args.runner.expanduser().resolve(strict=True)
    aggregator = args.aggregator.expanduser().resolve(strict=True)
    prequalification = args.prequalification.expanduser().resolve(strict=True)
    mdp_checkpoint = args.mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.polar_checkpoint.expanduser().resolve(strict=True)

    if preregistration.exists():
        raise FileExistsError(preregistration)
    if records_dir.exists() and any(records_dir.iterdir()):
        raise ValueError("Records directory must not contain pre-evaluation files.")
    if _git(source_root, "status", "--porcelain=v1"):
        raise ValueError("Source snapshot must be a clean Git worktree.")
    source_head = _git(source_root, "rev-parse", "HEAD")
    if runner.parent.parent.parent != source_root:
        raise ValueError("Runner must belong to the frozen source snapshot.")
    if aggregator.parent.parent.parent != source_root:
        raise ValueError("Aggregator must belong to the frozen source snapshot.")
    prequalification_payload = json.loads(prequalification.read_text())
    gates = prequalification_payload.get("gates")
    decision = prequalification_payload.get("decision")
    configuration = prequalification_payload.get("configuration")
    if (
        prequalification_payload.get("status") != "pass-target-free-prequalification"
        or not isinstance(gates, dict)
        or not gates
        or any(value is not True for value in gates.values())
        or not isinstance(decision, dict)
        or decision.get("target_free_prequalification_passed") is not True
        or decision.get("accuracy_claim_made") is not False
        or not isinstance(configuration, dict)
        or configuration.get("profile_id")
        != "route2-research-mace-mdp-point-polar-residual-adt-ddx-operational-v1"
        or configuration.get("lmax") != 8
        or configuration.get("n_lebedev") != 1202
    ):
        raise ValueError("Target-free prequalification is not an exact pass.")

    input_hashes = {
        key: _sha256(input_root / relative) for key, relative in INPUT_FILES.items()
    }
    _validate_selection(source_root, input_root)
    source_hashes = {name: _sha256(source_root / name) for name in SOURCE_FILES}
    payload: dict[str, object] = {
        "artifact_id": ARTIFACT_ID,
        "schema_version": 1,
        "status": (
            "locked-after-target-free-prequalification-before-canonical-adt-505"
        ),
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "development",
        "record_count": EXPECTED_RECORD_COUNT,
        "confirmation_partition_opened": False,
        "development_targets_previously_opened": True,
        "profile_selected_from_experimental_targets": False,
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
            "primary pass iff all 505 frozen development records succeed and "
            "stock-CDS M1 MAE <= 1.5 kcal/mol; separately report the 1.0 "
            "kcal/mol stretch target"
        ),
        "reporting_metrics": [
            "MAE",
            "RMSE",
            "maximum absolute error",
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
        "target_free_prequalification_path": str(prequalification),
        "target_free_prequalification_file_sha256": _sha256(prequalification),
        "target_free_prequalification_artifact_sha256": (
            prequalification_payload["artifact_sha256"]
        ),
        "target_free_prequalification_measurement_sha256": (
            prequalification_payload["measurement_sha256"]
        ),
        "mdp_checkpoint_sha256": _sha256(mdp_checkpoint),
        "polar_checkpoint_sha256": _sha256(polar_checkpoint),
        "selection_history": {
            "same_frozen_505_development_partition_as_prior_profiles": True,
            "development_targets_already_known": True,
            "canonical_adt_profile_selected_from_bound_target_free_canary": True,
            "canonical_adt_505_results_used_for_method_selection": False,
            "confirmation_partition_remains_sealed": True,
        },
        "method": {
            "profile_id": (
                "route2-research-mace-mdp-point-polar-residual-adt-ddx-"
                "operational-v1"
            ),
            "scalar": (
                "G_ddX,pol(MACE-MDP permanent point l<=1 + MACE-POLAR "
                "nonuniform Gaussian residual l<=1 + canonical MDP ADT dipoles) "
                "+ PySCF stock SMD CDS"
            ),
            "permanent_source": "official frozen MACE-MDP q/p checkpoint",
            "nonuniform_response": (
                "official frozen MACE-POLAR-1-M zero-anchored residual response"
            ),
            "uniform_response": (
                "canonical free-atom-density ADT lift of the same official "
                "MACE-MDP molecular polarizability"
            ),
            "mace_polar_long_range_evaluator": (
                "graph-longrange-analytic-gaussian-multipole-realspace-v1"
            ),
            "continuum_model": "pcm",
            "lmax": 8,
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
            "Development energy-accuracy decision only. The 505 development "
            "targets were previously opened by older profiles; the canonical-ADT "
            "method itself is frozen here from a target-free canary before its "
            "505 evaluation. The 148-row confirmation partition remains sealed. "
            "No force, virial, Hessian, frequency, optimization, MD, or release "
            "accuracy is established."
        ),
    }
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
    parser.add_argument("--prequalification", type=Path, required=True)
    parser.add_argument("--mdp-checkpoint", type=Path, required=True)
    parser.add_argument("--polar-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = create(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(_sha256(args.output.expanduser().resolve()))


if __name__ == "__main__":
    main()
