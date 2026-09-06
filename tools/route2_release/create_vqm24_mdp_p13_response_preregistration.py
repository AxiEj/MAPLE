#!/usr/bin/env python3
"""Freeze the MDP-only block-passive P13 response development pilot."""

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
    train_vqm24_mdp_p13_passive_response_pilot as train,
)


SELF_REPO_PATH = (
    "tools/route2_release/create_vqm24_mdp_p13_response_preregistration.py"
)
ARTIFACT = train.PREREGISTRATION_ARTIFACT
DEFAULT_PARENT = SOURCE_ROOT / (
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
DEFAULT_OUTPUT = SOURCE_ROOT / (
    "docs/route2/preregistrations/"
    "vqm24-mdp-p13-block-passive-response-pilot-v1.json"
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


def run(args: argparse.Namespace) -> dict[str, object]:
    parent_path = args.parent.expanduser().resolve(strict=True)
    selection_path = args.selection.expanduser().resolve(strict=True)
    mdp_root = args.mdp_features.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    parent = json.loads(parent_path.read_text())
    selection = json.loads(selection_path.read_text())
    if parent.get("artifact") != train.PARENT_ARTIFACT:
        raise ValueError("Dense response parent has the wrong identity.")
    train_records = parent["records"]
    if len(train_records) != 32 or any(
        record["selection_record"]["split"] != "train"
        for record in train_records
    ):
        raise RuntimeError("P13 pilot requires exactly 32 frozen train formulas.")
    selected_by_split: dict[str, list[str]] = {
        role: sorted(
            record["record_sha256"]
            for record in selection["records"]
            if record["split"] == role
        )
        for role in ("train", "validation", "blind")
    }
    if sorted(
        record["selection_record"]["record_sha256"] for record in train_records
    ) != selected_by_split["train"]:
        raise RuntimeError("Dense response records differ from the frozen train split.")
    sources = {
        "factor": "maple/solvation/models/sparse_passive_factor.py",
        "response": "maple/solvation/models/passive_p13_response.py",
        "radial_coupling": "maple/solvation/coupling/exact_gto.py",
        "l2_coupling": "maple/solvation/coupling/gaussian_quadrupole.py",
    }
    payload: dict[str, object] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-training",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent": {
            "path": str(parent_path.relative_to(SOURCE_ROOT)),
            "file_sha256": _sha256(parent_path),
            "preregistration_sha256": parent["preregistration_sha256"],
        },
        "selection": {
            "path": str(selection_path.relative_to(SOURCE_ROOT)),
            "file_sha256": _sha256(selection_path),
            "selection_sha256": selection["selection_sha256"],
            "record_sha256s_by_split": selected_by_split,
            "validation_and_blind_data_unopened": True,
        },
        "inputs": {
            "mdp_feature_manifest_path": str(mdp_root / "manifest.json"),
            "mdp_feature_manifest_sha256": _sha256(mdp_root / "manifest.json"),
            "qm_targets": [
                "nonuniform exterior-MEP linear response",
                "molecular dipole linear response",
            ],
            "forbidden_targets": [
                "per-molecule source coefficients",
                "MACE-MDP atomic partitions",
                "original MACE-POLAR source or response",
                "PCM energies or fields",
                "experimental solvation energies",
                "VQM24 total energies",
                "finite-difference energy curvature",
            ],
        },
        "source": {
            "preregistration_builder_path": SELF_REPO_PATH,
            "preregistration_builder_sha256": _sha256(
                SOURCE_ROOT / SELF_REPO_PATH
            ),
            "training_runner_path": train.SELF_REPO_PATH,
            "training_runner_sha256": _sha256(
                SOURCE_ROOT / train.SELF_REPO_PATH
            ),
            **{
                name: {"path": path, "sha256": _sha256(SOURCE_ROOT / path)}
                for name, path in sources.items()
            },
        },
        "train_record_ids": [record["record_id"] for record in train_records],
        "candidate_sequence": ["A2", "A3"],
        "architecture": {
            "permanent_source": "frozen existing MACE-MDP point q/p affine term",
            "response_scalar": "-1/2 ||C_theta(R) eta_P13||^2",
            "response_source": "gradient of sole scalar; no independent head",
            "source_space": "radial8 plus Gaussian-STF-l2 sigma=1.5 angstrom",
            "A2": "atom radial-null + two atom vector factor rows + local l2 rows",
            "A3": "A2 plus symmetric local edge charge/dipole factor row",
            "parameter_cap": 2500,
            "backbones_frozen": True,
            "new_message_passing": False,
            "edge_cutoff": "C2 quintic, 5.0 angstrom",
        },
        "optimizer": {
            "seed": train.SEED,
            "epochs": train.EPOCHS,
            "learning_rate": train.LEARNING_RATE,
            "weight_decay": train.WEIGHT_DECAY,
            "gradient_norm_clip": 20.0,
            "loss_weights": {
                "response_mep": train.MEP_LOSS_WEIGHT,
                "dipole_response": train.DIPOLE_LOSS_WEIGHT,
            },
            "audit_points_used_in_gradient": False,
        },
        "gates": {
            "mean_audit_mep_relative_maximum": 0.08,
            "maximum_audit_mep_relative": 0.20,
            "maximum_audit_mep_absolute_hartree_per_e_per_source_e": 0.002,
            "mean_dipole_response_relative_maximum": 0.08,
            "maximum_dipole_response_relative": 0.20,
            "mean_source_point_response_relative_maximum": 0.05,
            "maximum_source_point_response_relative": 0.20,
            "maximum_source_point_reciprocity_relative": 1.0e-10,
            "maximum_source_point_symmetric_eigenvalue_hartree_per_e2": 1.0e-10,
            "maximum_induced_charge_residual_e": 1.0e-12,
            "mean_audit_to_fit_ratio_maximum": 1.5,
            "maximum_individual_audit_to_fit_ratio": 2.0,
        },
        "decision": {
            "A2_pass_stops_before_A3": True,
            "A3_runs_only_if_A2_fails": True,
            "passing_candidate_only_authorizes_grouped_train_CV": True,
            "validation_and_blind_remain_unopened": True,
            "failure_of_both_candidates_authorizes_only_preregistered_A4": True,
            "no_MAPLE_capability_or_solvation_accuracy_claim": True,
        },
        "claim_boundary": {
            "development_train_formulas_only": True,
            "response_architecture_pilot": True,
            "grouped_formula_cv_completed": False,
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
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--mdp-features", type=Path, default=DEFAULT_MDP_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
