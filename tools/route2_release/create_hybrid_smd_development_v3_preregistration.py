#!/usr/bin/env python3
"""Create the post-reboot, durable hybrid MNSol development preregistration.

The v3 scientific method is intentionally identical to v2.  The only changes
are evidence engineering: all inputs, source, records, logs, and decision tools
live on persistent storage, and the 505 identities are reconstructed directly
from the already-frozen private development selection rather than from a
destroyed temporary aggregate panel.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ARTIFACT_ID = "route2-hybrid-smd-development-prereg-v3"
EXPECTED_RECORD_COUNT = 505
INTERRUPTED_V2_PREREGISTRATION_SHA256 = (
    "7e20caafa840cc1f7c2ed84cde7396a8e0e6219876ad4a4408689335093a8018"
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
    "maple/solvation/continuum/separated_source_ddx.py",
    "maple/solvation/experimental/mace_mdp_polar_ddx.py",
    "maple/solvation/experimental/mace_mdp_polar_solvated_ddx.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_hybrid.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_separated.py",
    "maple/solvation/solvent_terms.py",
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
    verifier = args.verifier.expanduser().resolve(strict=True)
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
    if verifier.parent.parent.parent != source_root:
        raise ValueError("Verifier must belong to the frozen source snapshot.")

    input_hashes = {
        key: _sha256(input_root / relative) for key, relative in INPUT_FILES.items()
    }
    _validate_selection(source_root, input_root)
    source_hashes = {name: _sha256(source_root / name) for name in SOURCE_FILES}
    payload: dict[str, object] = {
        "artifact_id": ARTIFACT_ID,
        "schema_version": 3,
        "status": "locked-before-first-v3-hybrid-evaluation",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "development",
        "record_count": EXPECTED_RECORD_COUNT,
        "confirmation_partition_opened": False,
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
            "pass iff all 505 frozen development records succeed and same-panel "
            "MAE <= 1.5 kcal/mol"
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
        "verifier_path": str(verifier),
        "verifier_sha256": _sha256(verifier),
        "mdp_checkpoint_sha256": _sha256(mdp_checkpoint),
        "polar_checkpoint_sha256": _sha256(polar_checkpoint),
        "v2_interruption": {
            "parent_preregistration_sha256": (INTERRUPTED_V2_PREREGISTRATION_SHA256),
            "last_observed_record_count": 396,
            "completed_shards": 1,
            "active_shards_at_loss": 7,
            "host_reboot_time": "2026-08-16T17:07:33+08:00",
            "partial_results_used_for_tuning": False,
            "partial_records_reused": False,
        },
        "v3_change_boundary": {
            "scientific_method_changed": False,
            "checkpoint_changed": False,
            "continuum_or_cavity_changed": False,
            "accuracy_results_used_for_tuning": False,
            "evidence_storage_changed_to_persistent": True,
            "selection_identity_source": (
                "frozen private selection plus frozen MNSol dataset"
            ),
        },
        "method": {
            "scalar": (
                "G_ddX,pol(MACE-MDP permanent point l<=1 + MACE-POLAR "
                "induced Gaussian l<=1) + PySCF SMD CDS"
            ),
            "permanent_source": "official frozen MACE-MDP q/p checkpoint",
            "induced_response": (
                "official frozen MACE-POLAR-1-M zero-anchored response"
            ),
            "mace_polar_long_range_evaluator": (
                "graph-longrange-analytic-gaussian-multipole-realspace-v1"
            ),
            "continuum_model": "pcm",
            "lmax": 15,
            "n_lebedev": 1202,
            "solver_tolerance": 1.0e-12,
            "eta": 0.1,
            "n_proc": 1,
            "device": "cpu",
            "standard_state": "MNSol protocol gas-to-solution 1M convention",
        },
        "claim_boundary": (
            "Development energy-accuracy decision only. It does not open the "
            "148-row confirmation partition and does not establish force, virial, "
            "Hessian, frequency, optimization, MD, or release accuracy."
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
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--mdp-checkpoint", type=Path, required=True)
    parser.add_argument("--polar-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = create(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(_sha256(args.output.expanduser().resolve()))


if __name__ == "__main__":
    main()
