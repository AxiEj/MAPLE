#!/usr/bin/env python3
"""Freeze twelve-mode sector-balanced CPKS response generation."""

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

from tools.route2_release import run_vqm24_sector_balanced_cpks_record as record_runner  # noqa: E402
from tools.route2_release import seal_vqm24_sector_balanced_cpks as sealer  # noqa: E402


SELF_REPO_PATH = (
    "tools/route2_release/create_vqm24_sector_balanced_cpks_preregistration.py"
)
DEFAULT_PARENT = SOURCE_ROOT / (
    "docs/route2/preregistrations/"
    "vqm24-observable-training-batch-dense-v2.json"
)
DEFAULT_MODES = Path(
    "/home/axie/.cache/maple-route2-hybrid-runs/"
    "vqm24-sector-balanced-p13-response-modes-v1"
)
DEFAULT_COVERAGE = SOURCE_ROOT / (
    "docs/route2/evidence/"
    "vqm24-cpks-response-only-coverage-20260827/result.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/axie/.cache/maple-route2-hybrid-runs/"
    "vqm24-sector-balanced-twelve-mode-cpks-v1"
)
DEFAULT_OUTPUT = SOURCE_ROOT / (
    "docs/route2/preregistrations/"
    "vqm24-sector-balanced-twelve-mode-cpks-v1.json"
)
GROUPED_RUNNER = "tools/route2_release/run_vqm24_grouped_static_cpks_response.py"


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
    modes_root = args.modes.expanduser().resolve(strict=True)
    coverage_path = args.coverage.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() or output_root.exists():
        raise FileExistsError(output_path if output_path.exists() else output_root)
    parent = json.loads(parent_path.read_text())
    mode_manifest = json.loads((modes_root / "manifest.json").read_text())
    coverage = json.loads(coverage_path.read_text())
    if (
        coverage["status"] != "pass-response-only-cpks-coverage"
        or coverage["decision"]["twelve_mode_generation_authorized"] is not True
    ):
        raise RuntimeError("Complete response-only CPKS coverage did not pass.")
    if mode_manifest["status"] != "pass-target-independent-mode-generation":
        raise RuntimeError("Sector-balanced response modes did not pass.")
    modes_by_id = {record["record_id"]: record for record in mode_manifest["records"]}
    records = []
    for record in parent["records"]:
        record_id = record["record_id"]
        modes_record = modes_by_id[record_id]
        paths = {
            "gas_checkpoint": Path(record["outputs"]["gas_directory"]) / "gas.chk",
            "surface": Path(record["inputs"]["surface"]["path"]),
            "four_modes": Path(record["inputs"]["modes"]["path"]),
            "twelve_modes": Path(modes_record["modes_path"]),
            "finite_field_observable": Path(record["outputs"]["observable_npz"]),
        }
        for path in paths.values():
            path.resolve(strict=True)
        records.append(
            {
                "record_id": record_id,
                "selection_record_sha256": record["selection_record"][
                    "record_sha256"
                ],
                "inputs": {
                    name: {"path": str(path), "sha256": _sha256(path)}
                    for name, path in paths.items()
                },
                "output_directory": str(output_root / record_id),
            }
        )
    grouped_path = SOURCE_ROOT / GROUPED_RUNNER
    payload: dict[str, object] = {
        "artifact": record_runner.PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-twelve-mode-execution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent": {
            "path": str(parent_path.relative_to(SOURCE_ROOT)),
            "file_sha256": _sha256(parent_path),
            "preregistration_sha256": parent["preregistration_sha256"],
        },
        "coverage": {
            "path": str(coverage_path.relative_to(SOURCE_ROOT)),
            "file_sha256": _sha256(coverage_path),
            "payload_sha256": coverage["result_sha256"],
        },
        "modes": {
            "manifest_path": str(modes_root / "manifest.json"),
            "manifest_sha256": _sha256(modes_root / "manifest.json"),
            "payload_sha256": mode_manifest["result_sha256"],
            "mode_count": 12,
            "frozen_prefix_count": 4,
        },
        "source": {
            "builder_path": SELF_REPO_PATH,
            "builder_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "record_runner_path": record_runner.SELF_REPO_PATH,
            "record_runner_sha256": _sha256(
                SOURCE_ROOT / record_runner.SELF_REPO_PATH
            ),
            "grouped_runner_path": GROUPED_RUNNER,
            "grouped_runner_sha256": _sha256(grouped_path),
            "sealer_path": sealer.SELF_REPO_PATH,
            "sealer_sha256": _sha256(SOURCE_ROOT / sealer.SELF_REPO_PATH),
        },
        "output_root": str(output_root),
        "records": records,
        "gates": {
            "prefix_mep_symmetric_relative_maximum": 5.8e-4,
            "prefix_dipole_symmetric_relative_maximum": 5.8e-4,
            "direct_cpks_residual_relative_frobenius_maximum": 1.0e-5,
            "direct_cpks_residual_relative_infinity_maximum": 2.0e-5,
            "electron_number_derivative_abs_maximum": 1.0e-10,
            "reciprocity_relative_frobenius_maximum": 1.0e-6,
            "passivity_maximum_eigenvalue_hartree_per_e2": 1.0e-8,
        },
        "decision": {
            "all_32_records_must_pass_every_gate": True,
            "any_failure_stops_response_model_training": True,
            "energy_curvature_target_or_gate": False,
            "success_authorizes_only_train_split_response_model_development": True,
        },
        "claim_boundary": {
            "independent_train_split_QM_response_generation": True,
            "model_fit_or_accuracy_measured": False,
            "experimental_solvation_target_read": False,
            "pcm_or_cavity_used": False,
            "validation_or_blind_formula_opened": False,
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
    parser.add_argument("--modes", type=Path, default=DEFAULT_MODES)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
