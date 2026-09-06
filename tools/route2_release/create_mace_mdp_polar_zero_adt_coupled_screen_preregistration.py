#!/usr/bin/env python3
"""Freeze the source-selected v3 coupled screen before any panel result.

The twelve geometries are selected only from the already frozen prospective
SPICE/MBIS electrostatic-source diagnostics.  No MNSol, FreeSolv, or other
experimental solvation target is read.  This is deliberately an adversarial
mechanism screen rather than an independent source-accuracy benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

ARTIFACT_ID = "route2-mace-mdp-polar-zero-adt-coupled-screen-prereg-v2"
SOURCE_ARTIFACT_ID = "route2-mdp-polar-uniform-response-source-gate-v1-record"
SOURCE_AGGREGATE_ARTIFACT_ID = (
    "route2-mdp-polar-uniform-response-source-gate-v1-aggregate"
)
EXPECTED_SOURCE_RECORD_COUNT = 60
EXPECTED_SCREEN_RECORD_COUNT = 12

SOURCE_EVIDENCE_RELATIVE = Path(
    "docs/route2/evidence/mdp-polar-uniform-response-source-gate-20260817"
)
SUPERSEDED_PREREGISTRATION_RELATIVE = Path(
    "docs/route2/preregistrations/mace-mdp-polar-zero-adt-coupled-screen-v1.json"
)
PROVIDER_REMEDIATION_RELATIVE = Path(
    "docs/route2/evidence/"
    "macepolar-zero-mdp-alpha-coupled-screen-provider-remediation-20260819/"
    "record-002-v1.json"
)
SOURCE_FILES = (
    "maple/function/calculator/extra_correction/implicit/smd_cds.py",
    "maple/solvation/api/profiles.py",
    "maple/solvation/continuum/separated_source_adt_ddx.py",
    "maple/solvation/continuum/separated_source_ddx.py",
    "maple/solvation/coupling/adt_radial_shape.py",
    "maple/solvation/coupling/atomic_displacement_lift.py",
    "maple/solvation/experimental/mace_mdp_polar_adt_ddx.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_adt.py",
    "maple/solvation/models/mace_polar_separated.py",
    "tools/route2_release/create_mace_mdp_polar_zero_adt_coupled_screen_preregistration.py",
    "tools/route2_release/run_mace_mdp_polar_zero_adt_coupled_screen.py",
)

CLAIM_BOUNDARY = {
    "capability_admitted": False,
    "experimental_solvation_targets_read": False,
    "fitting_calibration_or_post_training_performed": False,
    "hybrid_line_only": True,
    "mbis_source_and_mep_diagnostics_used_for_adversarial_selection": True,
    "mnsol_or_freesolv_imported": False,
    "panel_is_independent_source_accuracy_evidence": False,
    "pure_mace_polar_in_scope": False,
    "single_water_v3_canary_preceded_this_selection": True,
    "stage_a_zero_start_screen_can_admit_release": False,
    "prior_array_identity_failure_before_coupled_evaluation_disclosed": True,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *args),
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_source_record(path: Path, expected_index: int) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    digest = payload.pop("record_sha256", None)
    _require(
        digest == _canonical_sha256(payload),
        f"Source record {expected_index} hash mismatch.",
    )
    payload["record_sha256"] = digest
    _require(
        payload.get("artifact") == SOURCE_ARTIFACT_ID,
        "Source artifact identity changed.",
    )
    _require(
        payload.get("selection_index") == expected_index,
        "Source selection index changed.",
    )
    boundary = payload.get("claim_boundary")
    _require(isinstance(boundary, dict), "Source record has no claim boundary.")
    _require(
        boundary.get("experimental_solvation_targets_read") is False,
        "Source record read solvation targets.",
    )
    _require(
        boundary.get("mnsol_or_freesolv_imported") is False,
        "Source record imported a solvation benchmark.",
    )
    _require(
        boundary.get("post_training_or_fitting_performed") is False,
        "Source record used fitted parameters.",
    )
    return payload


def load_source_records(source_evidence: Path) -> list[dict[str, Any]]:
    records_dir = source_evidence / "records"
    expected_names = {
        f"record-{index:03d}.json" for index in range(EXPECTED_SOURCE_RECORD_COUNT)
    }
    observed_names = {path.name for path in records_dir.glob("record-*.json")}
    _require(
        observed_names == expected_names,
        "Frozen sixty-record source evidence is incomplete.",
    )
    return [
        _load_source_record(records_dir / f"record-{index:03d}.json", index)
        for index in range(EXPECTED_SOURCE_RECORD_COUNT)
    ]


def _dipole_norm(record: Mapping[str, Any]) -> float:
    values = record["source_closure"]["polar_zero_point"][
        "target_molecular_dipole_eangstrom"
    ]
    return float(sum(float(value) ** 2 for value in values) ** 0.5)


def _metric_functions() -> (
    tuple[tuple[str, int, str, bool, Callable[[Mapping[str, Any]], float]], ...]
):
    return (
        (
            "largest_fixed_source_pcm_error",
            4,
            "polar_zero_point_absolute_error_kcal_mol",
            True,
            lambda record: float(
                record["fixed_source_ddpcm_energy"][
                    "polar_zero_point_absolute_error_kcal_mol"
                ]
            ),
        ),
        (
            "largest_surface_mep_relative_rmse",
            2,
            "radius_1.00_reference_qpqo_relative_root_mean_square_error",
            True,
            lambda record: float(
                record["mep_by_radius_scale"]["1.00"]["metrics"]["polar_zero_point"][
                    "reference_qpqo"
                ]["relative_root_mean_square_error"]
            ),
        ),
        (
            "largest_atom_count",
            2,
            "atom_count",
            True,
            lambda record: float(record["selection_identity"]["atom_count"]),
        ),
        (
            "largest_molecular_dipole",
            1,
            "target_molecular_dipole_norm_eangstrom",
            True,
            _dipole_norm,
        ),
        (
            "smallest_molecular_dipole",
            1,
            "target_molecular_dipole_norm_eangstrom",
            False,
            _dipole_norm,
        ),
        (
            "largest_uniform_response_condition_number",
            1,
            "maximum_jacobian_condition_number",
            True,
            lambda record: float(
                record["uniform_response"]["maximum_jacobian_condition_number"]
            ),
        ),
        (
            "largest_uniform_response_source_shift",
            1,
            "source_l2_shift",
            True,
            lambda record: float(record["uniform_response"]["source_l2_shift"]),
        ),
    )


def select_screen_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _require(
        len(records) == EXPECTED_SOURCE_RECORD_COUNT,
        "Selection requires the complete sixty-record source panel.",
    )
    selected_source_indices: set[int] = set()
    result: list[dict[str, Any]] = []
    for category, quota, metric_name, descending, metric in _metric_functions():
        ranked = sorted(
            records,
            key=lambda record: (
                -metric(record) if descending else metric(record),
                int(record["selection_index"]),
            ),
        )
        accepted = 0
        for rank, record in enumerate(ranked, start=1):
            source_index = int(record["selection_index"])
            if source_index in selected_source_indices:
                continue
            identity = dict(record["selection_identity"])
            result.append(
                {
                    "screen_index": len(result),
                    "source_record_index": source_index,
                    "source_record_sha256": record["record_sha256"],
                    "selection_identity": identity,
                    "selection_category": category,
                    "selection_metric": metric_name,
                    "selection_metric_value": metric(record),
                    "selection_rank_before_duplicate_filter": rank,
                }
            )
            selected_source_indices.add(source_index)
            accepted += 1
            if accepted == quota:
                break
        _require(accepted == quota, f"Category {category} could not fill its quota.")
    _require(
        len(result) == EXPECTED_SCREEN_RECORD_COUNT,
        "Adversarial selection did not produce exactly twelve unique records.",
    )
    return result


def create(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    source_evidence = (source_root / SOURCE_EVIDENCE_RELATIVE).resolve(strict=True)
    superseded_path = (source_root / SUPERSEDED_PREREGISTRATION_RELATIVE).resolve(
        strict=True
    )
    remediation_path = (source_root / PROVIDER_REMEDIATION_RELATIVE).resolve(
        strict=True
    )
    dataset = args.dataset.expanduser().resolve(strict=True)
    mdp_checkpoint = args.mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.polar_checkpoint.expanduser().resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)

    aggregate_path = source_evidence / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text())
    aggregate_digest = aggregate.pop("aggregate_sha256", None)
    _require(
        aggregate_digest == _canonical_sha256(aggregate),
        "Source aggregate self hash changed.",
    )
    aggregate["aggregate_sha256"] = aggregate_digest
    _require(
        aggregate.get("artifact") == SOURCE_AGGREGATE_ARTIFACT_ID
        and aggregate.get("record_count") == EXPECTED_SOURCE_RECORD_COUNT,
        "Wrong source aggregate identity or count.",
    )
    boundary = aggregate.get("claim_boundary")
    _require(
        isinstance(boundary, dict)
        and boundary.get("experimental_solvation_targets_read") is False
        and boundary.get("mnsol_or_freesolv_imported") is False,
        "Source aggregate is not target-free with respect to solvation data.",
    )

    records = load_source_records(source_evidence)
    input_bindings = {
        tuple(sorted(record["input_sha256"].items())) for record in records
    }
    _require(len(input_bindings) == 1, "Source records disagree on input hashes.")
    source_inputs = dict(input_bindings.pop())
    _require(
        _sha256_file(dataset) == source_inputs["dataset"],
        "SPICE dataset hash differs from the frozen source evidence.",
    )
    _require(
        _sha256_file(mdp_checkpoint) == source_inputs["mace_mdp_checkpoint"],
        "MACE-MDP checkpoint hash differs from the frozen source evidence.",
    )
    _require(
        _sha256_file(polar_checkpoint) == source_inputs["mace_polar_checkpoint"],
        "MACE-POLAR checkpoint hash differs from the frozen source evidence.",
    )

    selected = select_screen_records(records)
    superseded = json.loads(superseded_path.read_text())
    _require(
        superseded.get("artifact")
        == "route2-mace-mdp-polar-zero-adt-coupled-screen-prereg-v1"
        and superseded.get("selection") == selected
        and superseded.get("method")
        == {
            "profile_id": (
                "route2-research-macepolar-zero-point-mdp-alpha-polar-residual-"
                "role-separated-adt-ddx-operational-v3"
            ),
            "permanent_source": "MACE-POLAR zero-field point q/p",
            "uniform_response": (
                "canonical ADT lift of public MACE-MDP molecular alpha"
            ),
            "nonuniform_response": (
                "MACE-POLAR response minus its zero-field uniform tangent; "
                "1.5 A Gaussian"
            ),
            "continuum_model": "pcm",
            "dielectric": 78.39,
            "lmax": 12,
            "n_lebedev": 1202,
            "solver_tolerance": 1.0e-12,
            "eta": 0.1,
            "n_proc": 1,
            "solve_stage": "one-zero-start-rejection-screen",
            "full_five_start_replay_required_after_stage_a_pass": True,
        },
        "Superseded preregistration does not bind the unchanged selection/method.",
    )
    superseded_digest = superseded.pop("preregistration_sha256", None)
    _require(
        superseded_digest == _canonical_sha256(superseded),
        "Superseded preregistration self hash changed.",
    )
    remediation = json.loads(remediation_path.read_text())
    remediation_digest = remediation.pop("record_sha256", None)
    _require(
        remediation_digest == _canonical_sha256(remediation)
        and remediation.get("status") == "provider-failure"
        and remediation.get("failure_type") == "ValueError"
        and remediation.get("failure_message") == "Atomic-number identity changed."
        and remediation.get("stage_a_passed") is False,
        "Provider-remediation record is not the frozen pre-evaluation failure.",
    )
    source_record_files = {
        f"record-{index:03d}.json": _sha256_file(
            source_evidence / "records" / f"record-{index:03d}.json"
        )
        for index in range(EXPECTED_SOURCE_RECORD_COUNT)
    }
    source_hashes = {
        name: _sha256_file((source_root / name).resolve(strict=True))
        for name in SOURCE_FILES
    }
    payload: dict[str, object] = {
        "artifact": ARTIFACT_ID,
        "schema_version": 1,
        "status": (
            "locked-after-array-identity-remediation-before-any-selected-v3-"
            "coupled-state-evaluation"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_head": _git(source_root, "rev-parse", "HEAD"),
        "source_worktree_dirty_at_lock": bool(
            _git(source_root, "status", "--porcelain=v1")
        ),
        "source_files_sha256": source_hashes,
        "superseded_preregistration": {
            "relative_path": str(SUPERSEDED_PREREGISTRATION_RELATIVE),
            "file_sha256": _sha256_file(superseded_path),
            "artifact_sha256": superseded_digest,
            "selection_and_method_unchanged": True,
        },
        "provider_remediation": {
            "relative_path": str(PROVIDER_REMEDIATION_RELATIVE),
            "file_sha256": _sha256_file(remediation_path),
            "record_sha256": remediation_digest,
            "failure_type": "ValueError",
            "failure_message": "Atomic-number identity changed.",
            "failed_before_coupled_state_evaluation": True,
            "scientific_configuration_changed": False,
            "selection_changed": False,
            "remediation": (
                "restore the frozen NUL separator between array identity header "
                "and bytes"
            ),
        },
        "source_evidence": {
            "relative_path": str(SOURCE_EVIDENCE_RELATIVE),
            "aggregate_file_sha256": _sha256_file(aggregate_path),
            "aggregate_artifact_sha256": aggregate_digest,
            "record_files_sha256": source_record_files,
            "record_count": EXPECTED_SOURCE_RECORD_COUNT,
            "source_gate_decision": aggregate["decision"],
            "source_gate_selected_candidate": aggregate["selected_candidate"],
            "interpretation": (
                "The prior source gate rejected both candidates under its full tail "
                "criteria. Its frozen diagnostics are used here only to select a "
                "mechanistically adversarial coupled-state screen."
            ),
        },
        "inputs_sha256": {
            "spice_dataset": _sha256_file(dataset),
            "mace_mdp_checkpoint": _sha256_file(mdp_checkpoint),
            "mace_polar_checkpoint": _sha256_file(polar_checkpoint),
        },
        "selection_policy": {
            "category_order": [spec[0] for spec in _metric_functions()],
            "category_quotas": {spec[0]: spec[1] for spec in _metric_functions()},
            "ranking_tie_break": "ascending frozen source selection_index",
            "duplicate_rule": "skip already selected and take the next ranked record",
            "source_record_count": EXPECTED_SOURCE_RECORD_COUNT,
            "screen_record_count": EXPECTED_SCREEN_RECORD_COUNT,
            "uses_experimental_solvation_targets": False,
            "uses_qm_source_or_mep_diagnostics": True,
        },
        "selection": selected,
        "selection_sha256": _canonical_sha256(selected),
        "method": {
            "profile_id": (
                "route2-research-macepolar-zero-point-mdp-alpha-polar-residual-"
                "role-separated-adt-ddx-operational-v3"
            ),
            "permanent_source": "MACE-POLAR zero-field point q/p",
            "uniform_response": "canonical ADT lift of public MACE-MDP molecular alpha",
            "nonuniform_response": (
                "MACE-POLAR response minus its zero-field uniform tangent; 1.5 A Gaussian"
            ),
            "continuum_model": "pcm",
            "dielectric": 78.39,
            "lmax": 12,
            "n_lebedev": 1202,
            "solver_tolerance": 1.0e-12,
            "eta": 0.1,
            "n_proc": 1,
            "solve_stage": "one-zero-start-rejection-screen",
            "full_five_start_replay_required_after_stage_a_pass": True,
        },
        "stage_a_gates": {
            "all_records_succeed": True,
            "maximum_primal_residual_eV": 1.0e-10,
            "maximum_absolute_permanent_charge_e": 1.0e-8,
            "maximum_absolute_radial_response_charge_e": 1.0e-8,
            "maximum_jvp_vjp_relative_defect": 1.0e-8,
            "maximum_uniform_left_inverse_error": 3.0e-14,
            "maximum_zero_field_uniform_radial_cancellation": 1.0e-8,
            "maximum_adt_molecular_dipole_closure_eangstrom": 1.0e-10,
            "minimum_mdp_alpha_eigenvalue": 1.0e-8,
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "next_if_stage_a_passes": (
            "Run the same twelve geometries with five starts, cold replay, a fixed "
            "rigid rotation, local-root diagnostics, and finite-field checks before "
            "any v3 505 energy panel."
        ),
        "next_if_stage_a_fails": (
            "Reject this zero-training v3 response profile without opening MNSol or "
            "FreeSolv targets and move to a separately trained scalar-first "
            "electrostatic head rather than a solvation-target residual patch."
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
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mdp-checkpoint", type=Path, required=True)
    parser.add_argument("--polar-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = create(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
