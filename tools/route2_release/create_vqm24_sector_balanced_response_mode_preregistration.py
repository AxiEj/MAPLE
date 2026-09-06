#!/usr/bin/env python3
"""Freeze the target-independent twelve-mode P13 input expansion."""

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
    prepare_vqm24_sector_balanced_response_modes as prepare,
)


SELF_REPO_PATH = (
    "tools/route2_release/"
    "create_vqm24_sector_balanced_response_mode_preregistration.py"
)
DEFAULT_PARENT = SOURCE_ROOT / (
    "docs/route2/preregistrations/"
    "vqm24-observable-training-batch-dense-v2.json"
)
DEFAULT_OUTPUT = SOURCE_ROOT / (
    "docs/route2/preregistrations/"
    "vqm24-sector-balanced-p13-response-modes-v1.json"
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
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    parent = json.loads(parent_path.read_text())
    if len(parent["records"]) != 32 or any(
        record["selection_record"]["split"] != "train"
        for record in parent["records"]
    ):
        raise RuntimeError("Mode expansion requires the frozen 32 train formulas.")
    selector_path = "maple/solvation/reference/sector_balanced_response_modes.py"
    radial_path = "maple/solvation/coupling/exact_gto.py"
    l2_path = "maple/solvation/coupling/gaussian_quadrupole.py"
    payload: dict[str, object] = {
        "artifact": prepare.PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-mode-generation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent": {
            "path": str(parent_path.relative_to(SOURCE_ROOT)),
            "file_sha256": _sha256(parent_path),
            "preregistration_sha256": parent["preregistration_sha256"],
        },
        "source": {
            "builder_path": SELF_REPO_PATH,
            "builder_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "runner_path": prepare.SELF_REPO_PATH,
            "runner_sha256": _sha256(SOURCE_ROOT / prepare.SELF_REPO_PATH),
            "selector": {
                "path": selector_path,
                "sha256": _sha256(SOURCE_ROOT / selector_path),
            },
            "radial_coupling": {
                "path": radial_path,
                "sha256": _sha256(SOURCE_ROOT / radial_path),
            },
            "l2_coupling": {
                "path": l2_path,
                "sha256": _sha256(SOURCE_ROOT / l2_path),
            },
        },
        "record_ids": [record["record_id"] for record in parent["records"]],
        "protocol": {
            "mode_count": 12,
            "frozen_prefix_mode_count": 4,
            "frozen_prefix": "existing farthest exterior fit-frame point charges",
            "candidate_points": "dense-v2 fit partition only",
            "constant_potential_gauge_projected": True,
            "radial_and_l2_sector_scaling": "separate geometry-only candidate RMS",
            "selection": "greedy maximum residual row volume with deterministic ties",
            "l2_sigma_angstrom": 1.5,
            "relative_rank_tolerance": 1.0e-11,
            "QM_or_model_target_used": False,
        },
        "gates": {
            "condition_number_maximum": 10.0,
            "minimum_greedy_residual_norm": 0.5,
            "radial_fraction_minimum": 0.4,
            "radial_fraction_maximum": 0.6,
            "every_record_rank_equals_12": True,
            "old_four_mode_prefix_preserved": True,
        },
        "decision": {
            "all_32_train_records_must_pass": True,
            "failed_mode_generation_stops_response_expansion": True,
            "success_authorizes_only_response_generator_prequalification": True,
            "no_QM_execution_or_training_authorized_by_this_artifact": True,
        },
        "claim_boundary": {
            "target_independent_input_design_only": True,
            "qm_response_or_energy_read": False,
            "model_prediction_read": False,
            "pcm_or_cavity_used": False,
            "experimental_solvation_target_read": False,
            "validation_or_blind_formula_opened": False,
            "model_fit_or_selection_performed": False,
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
