#!/usr/bin/env python3
"""Prospectively freeze Stage B for the target-free v3 hybrid profile.

Stage B may be created only from a complete passing Stage-A aggregate.  It
continues to read geometry and checkpoint state only: no MNSol, FreeSolv, or
other experimental solvation target is opened.  The resulting artifact fixes
the five-start replay, finite-field, projected local-root, and rigid-motion
tests before any v3 accuracy panel is evaluated.
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
from typing import Any, Mapping

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tools.route2_release import (
    create_mace_mdp_polar_zero_adt_coupled_screen_preregistration as stage_a_creator,
)
from tools.route2_release import (
    run_mace_mdp_polar_zero_adt_coupled_screen as stage_a_runner,
)

ARTIFACT_ID = "route2-mace-mdp-polar-zero-adt-stage-b-prereg-v1"
EXPECTED_STAGE_A_AGGREGATE_ARTIFACT = f"{stage_a_runner.ARTIFACT_ID}-aggregate"
EXPECTED_RECORD_COUNT = stage_a_creator.EXPECTED_SCREEN_RECORD_COUNT

STAGE_B_SOURCE_FILES = (
    "tools/route2_release/create_mace_mdp_polar_zero_adt_stage_b_preregistration.py",
    "tools/route2_release/run_mace_mdp_polar_zero_adt_stage_b.py",
)

ROTATION_MATRIX = (
    (-0.664894784009872, 0.12761976496184402, 0.7359538856392766),
    (0.7124449411044197, -0.18759949407880958, 0.6761868349177573),
    (0.22435938155375984, 0.9739197222628044, 0.033811869163890604),
)
TRANSLATION_ANGSTROM = (0.71, -0.43, 0.29)
FINITE_FIELD_STEPS = (2.0e-3, 1.0e-3, 5.0e-4)
PROJECTED_RANDOM_PROBE_COUNT = 5
PROJECTED_ROOT_RANDOM_SEED = 20260820

CLAIM_BOUNDARY = {
    "accuracy_panel_opened": False,
    "capability_admitted": False,
    "experimental_solvation_targets_read": False,
    "fitting_calibration_or_post_training_performed": False,
    "hybrid_line_only": True,
    "local_root_test_is_projected_diagnostic_not_full_space_proof": True,
    "mnsol_or_freesolv_imported": False,
    "pure_mace_polar_in_scope": False,
    "rigid_test_is_numerical_not_structural_so3_proof": True,
    "stage_b_can_admit_release": False,
}

STAGE_B_GATES = {
    "maximum_five_start_residual_eV": 1.0e-10,
    "maximum_five_start_energy_span_eV": 2.0e-10,
    # The evaluator enforces this componentwise bound while constructing a
    # state.  Stage B binds that implementation contract exactly; the state
    # currently stores converged-field hashes rather than the achieved span.
    "maximum_five_start_field_span_eV": 3.0e-9,
    "exact_cold_replay_required": True,
    "maximum_jvp_vjp_relative_defect": 1.0e-8,
    "finite_field_steps": list(FINITE_FIELD_STEPS),
    "maximum_finest_finite_field_error": 1.0e-6,
    "strict_finite_field_error_decrease_required": True,
    "projected_random_probe_count": PROJECTED_RANDOM_PROBE_COUNT,
    "projected_root_random_seed": PROJECTED_ROOT_RANDOM_SEED,
    "maximum_sampled_state_map_gain": 0.95,
    "minimum_projected_residual_singular_value": 0.10,
    "minimum_projected_residual_symmetric_margin": 0.05,
    # 4.34e-4 eV = 0.0100 kcal/mol, negligible relative to the 1 kcal/mol goal.
    "maximum_rigid_total_energy_error_eV": 4.34e-4,
    "maximum_rigid_polarization_energy_error_eV": 4.34e-4,
    "maximum_rigid_source_relative_error": 1.0e-3,
    "maximum_rigid_field_relative_error": 1.0e-3,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return stage_a_creator._canonical_sha256(payload)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *args),
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_stage_a_aggregate(
    payload: Mapping[str, Any],
    *,
    stage_a_prereg: Mapping[str, Any],
) -> str:
    """Validate the terminal Stage-A decision and return its self digest."""

    aggregate = dict(payload)
    digest = aggregate.pop("aggregate_sha256", None)
    _require(digest == _canonical_sha256(aggregate), "Stage-A aggregate hash changed.")
    _require(
        aggregate.get("artifact") == EXPECTED_STAGE_A_AGGREGATE_ARTIFACT,
        "Wrong Stage-A aggregate artifact.",
    )
    _require(
        aggregate.get("status") == "pass-stage-a-zero-start-coupled-screen"
        and aggregate.get("stage_a_passed") is True
        and aggregate.get("record_count") == EXPECTED_RECORD_COUNT
        and aggregate.get("pass_count") == EXPECTED_RECORD_COUNT
        and aggregate.get("provider_or_gate_failure_count") == 0,
        "Stage A did not pass all twelve frozen records.",
    )
    _require(
        aggregate.get("preregistration_artifact_sha256")
        == stage_a_prereg.get("preregistration_sha256"),
        "Stage-A aggregate belongs to another preregistration.",
    )
    records = aggregate.get("record_sha256s")
    _require(
        isinstance(records, list)
        and len(records) == EXPECTED_RECORD_COUNT
        and len(set(records)) == EXPECTED_RECORD_COUNT
        and all(isinstance(item, str) and len(item) == 64 for item in records),
        "Stage-A aggregate has no exact twelve-record closure.",
    )
    _require(
        aggregate.get("accuracy_claim_made") is False
        and aggregate.get("capabilities_admitted")
        == {tier: False for tier in ("E", "F", "H", "V", "M")},
        "Stage-A aggregate exceeded its rejection-only claim boundary.",
    )
    return str(digest)


def create(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    _require(
        source_root == SOURCE_ROOT, "Stage-B creator must run from its own source root."
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)

    dataset = args.dataset.expanduser().resolve(strict=True)
    mdp_checkpoint = args.mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.polar_checkpoint.expanduser().resolve(strict=True)
    stage_a_prereg_path = args.stage_a_preregistration.expanduser().resolve(strict=True)
    stage_a_aggregate_path = args.stage_a_aggregate.expanduser().resolve(strict=True)
    stage_a_records_dir = args.stage_a_records_dir.expanduser().resolve(strict=True)
    stage_a_prereg = stage_a_runner._load_preregistration(stage_a_prereg_path)
    stage_a_runner._validate_source_and_inputs(
        stage_a_prereg,
        dataset=dataset,
        mdp_checkpoint=mdp_checkpoint,
        polar_checkpoint=polar_checkpoint,
    )
    stage_a_aggregate = json.loads(stage_a_aggregate_path.read_text())
    stage_a_aggregate_sha = validate_stage_a_aggregate(
        stage_a_aggregate,
        stage_a_prereg=stage_a_prereg,
    )
    _require(
        stage_a_aggregate.get("preregistration_file_sha256")
        == _sha256_file(stage_a_prereg_path),
        "Stage-A aggregate preregistration file binding changed.",
    )
    with tempfile.TemporaryDirectory(prefix="maple-stage-a-reaggregate-") as temporary:
        regenerated = stage_a_runner.aggregate(
            SimpleNamespace(
                preregistration=stage_a_prereg_path,
                dataset=dataset,
                mdp_checkpoint=mdp_checkpoint,
                polar_checkpoint=polar_checkpoint,
                output_dir=stage_a_records_dir,
                aggregate_output=Path(temporary) / "aggregate.json",
            )
        )
    _require(
        regenerated == stage_a_aggregate,
        "Stage-A aggregate does not regenerate exactly from its twelve records.",
    )
    expected_record_names = {
        f"record-{index:03d}.json" for index in range(EXPECTED_RECORD_COUNT)
    }
    observed_record_names = {
        path.name for path in stage_a_records_dir.glob("record-*.json")
    }
    _require(
        observed_record_names == expected_record_names,
        "Stage-A record directory is not exactly closed.",
    )
    stage_a_record_file_hashes = {
        name: _sha256_file(stage_a_records_dir / name)
        for name in sorted(expected_record_names)
    }

    inherited_source_hashes = stage_a_prereg.get("source_files_sha256")
    _require(
        isinstance(inherited_source_hashes, dict), "Missing Stage-A source manifest."
    )
    for name, expected in inherited_source_hashes.items():
        _require(
            _sha256_file((source_root / name).resolve(strict=True)) == expected,
            f"Stage-A bound source changed before Stage B: {name}.",
        )
    stage_b_source_hashes = {
        name: _sha256_file((source_root / name).resolve(strict=True))
        for name in STAGE_B_SOURCE_FILES
    }

    runtime_sha = stage_a_aggregate.get("runtime_identity_sha256")
    _require(
        isinstance(runtime_sha, str) and len(runtime_sha) == 64,
        "Stage-A runtime identity is missing.",
    )
    payload: dict[str, object] = {
        "artifact": ARTIFACT_ID,
        "schema_version": 1,
        "status": "locked-after-stage-a-pass-before-any-stage-b-evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_head": _git(source_root, "rev-parse", "HEAD"),
        "source_worktree_dirty_at_lock": bool(
            _git(source_root, "status", "--porcelain=v1")
        ),
        "stage_a_preregistration": {
            "path": str(stage_a_prereg_path),
            "file_sha256": _sha256_file(stage_a_prereg_path),
            "artifact_sha256": stage_a_prereg["preregistration_sha256"],
        },
        "stage_a_aggregate": {
            "path": str(stage_a_aggregate_path),
            "file_sha256": _sha256_file(stage_a_aggregate_path),
            "aggregate_sha256": stage_a_aggregate_sha,
            "records_path": str(stage_a_records_dir),
            "record_files_sha256": stage_a_record_file_hashes,
            "record_sha256s": stage_a_aggregate["record_sha256s"],
            "runtime_identity_sha256": runtime_sha,
        },
        "inherited_stage_a_source_files_sha256": dict(inherited_source_hashes),
        "stage_b_source_files_sha256": stage_b_source_hashes,
        "inputs_sha256": dict(stage_a_prereg["inputs_sha256"]),
        "selection": stage_a_prereg["selection"],
        "selection_sha256": stage_a_prereg["selection_sha256"],
        "method": {
            **dict(stage_a_prereg["method"]),
            "solve_stage": "five-start-cold-replay-rigid-finite-field-stage-b",
            "base_solve_count_per_geometry": 5,
            "cold_replay_solve_count_per_geometry": 5,
            "rigid_solve_count_per_geometry": 5,
            "rotation_matrix": [list(row) for row in ROTATION_MATRIX],
            "translation_angstrom": list(TRANSLATION_ANGSTROM),
            "finite_field_steps": list(FINITE_FIELD_STEPS),
            "projected_local_root_probe_count": 3 + PROJECTED_RANDOM_PROBE_COUNT,
        },
        "stage_b_gates": STAGE_B_GATES,
        "claim_boundary": CLAIM_BOUNDARY,
        "next_if_stage_b_passes": (
            "Freeze this exact v3 identity for a target-blind 505 energy run; "
            "Stage B itself makes no accuracy or release claim."
        ),
        "next_if_stage_b_fails": (
            "Reject the zero-training v3 profile and move to a separately named "
            "scalar-first electrostatic head; do not fit a solvation-target residual."
        ),
    }
    payload["preregistration_sha256"] = _canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output.chmod(0o444)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--stage-a-preregistration", type=Path, required=True)
    parser.add_argument("--stage-a-aggregate", type=Path, required=True)
    parser.add_argument("--stage-a-records-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mdp-checkpoint", type=Path, required=True)
    parser.add_argument("--polar-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    payload = create(_parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
