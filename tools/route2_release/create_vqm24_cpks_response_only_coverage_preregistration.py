#!/usr/bin/env python3
"""Freeze 32-record dense CPKS-versus-finite-field response coverage."""

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
    run_vqm24_cpks_response_only_coverage_record as record_runner,
)
from tools.route2_release import (  # noqa: E402
    seal_vqm24_cpks_response_only_coverage as coverage_sealer,
)


SELF_REPO_PATH = (
    "tools/route2_release/"
    "create_vqm24_cpks_response_only_coverage_preregistration.py"
)
DEFAULT_PARENT = SOURCE_ROOT / (
    "docs/route2/preregistrations/"
    "vqm24-observable-training-batch-dense-v2.json"
)
DEFAULT_PILOT = SOURCE_ROOT / (
    "docs/route2/evidence/vqm24-cpks-response-only-pilot-20260827/result.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/axie/.cache/maple-route2-hybrid-runs/"
    "vqm24-cpks-response-only-coverage-v1"
)
DEFAULT_OUTPUT = SOURCE_ROOT / (
    "docs/route2/preregistrations/"
    "vqm24-cpks-response-only-coverage-v1.json"
)
CPKS_RUNNER = "tools/route2_release/run_vqm24_static_cpks_response.py"


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
    pilot_path = args.pilot.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() or output_root.exists():
        raise FileExistsError(output_path if output_path.exists() else output_root)
    parent = json.loads(parent_path.read_text())
    pilot = json.loads(pilot_path.read_text())
    if (
        pilot["status"] != "pass-response-only-pilot"
        or pilot["decision"][
            "full_32_record_coverage_preregistration_authorized"
        ]
        is not True
    ):
        raise RuntimeError("Response-only pilot did not authorize coverage.")
    records = []
    for record in parent["records"]:
        gas_checkpoint = Path(record["outputs"]["gas_directory"]) / "gas.chk"
        inputs = {
            "gas_checkpoint": gas_checkpoint,
            "surface": Path(record["inputs"]["surface"]["path"]),
            "modes": Path(record["inputs"]["modes"]["path"]),
            "finite_field_observable": Path(record["outputs"]["observable_npz"]),
        }
        for path in inputs.values():
            path.resolve(strict=True)
        records.append(
            {
                "record_id": record["record_id"],
                "selection_record_sha256": record["selection_record"][
                    "record_sha256"
                ],
                "inputs": {
                    name: {"path": str(path), "sha256": _sha256(path)}
                    for name, path in inputs.items()
                },
                "output_directory": str(output_root / record["record_id"]),
            }
        )
    payload: dict[str, object] = {
        "artifact": record_runner.PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-coverage-execution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent": {
            "path": str(parent_path.relative_to(SOURCE_ROOT)),
            "file_sha256": _sha256(parent_path),
            "preregistration_sha256": parent["preregistration_sha256"],
        },
        "pilot": {
            "path": str(pilot_path.relative_to(SOURCE_ROOT)),
            "file_sha256": _sha256(pilot_path),
            "payload_sha256": pilot["result_sha256"],
        },
        "source": {
            "builder_path": SELF_REPO_PATH,
            "builder_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "record_runner_path": record_runner.SELF_REPO_PATH,
            "record_runner_sha256": _sha256(
                SOURCE_ROOT / record_runner.SELF_REPO_PATH
            ),
            "sealer_path": coverage_sealer.SELF_REPO_PATH,
            "sealer_sha256": _sha256(
                SOURCE_ROOT / coverage_sealer.SELF_REPO_PATH
            ),
            "cpks_runner_path": CPKS_RUNNER,
            "cpks_runner_sha256": _sha256(SOURCE_ROOT / CPKS_RUNNER),
        },
        "output_root": str(output_root),
        "records": records,
        "gates": {
            "mep_symmetric_relative_maximum": 5.8e-4,
            "dipole_symmetric_relative_maximum": 5.8e-4,
            "direct_cpks_residual_relative_frobenius_maximum": 1.0e-5,
            "direct_cpks_residual_relative_infinity_maximum": 2.0e-5,
            "checkpoint_canonical_fock_residual_relative_maximum": 1.0e-5,
            "checkpoint_rebuilt_energy_abs_hartree_maximum": 1.0e-5,
            "energy_identity_abs_hartree_per_e2_maximum": 1.0e-10,
            "electron_number_derivative_abs_maximum": 1.0e-10,
            "reciprocity_relative_frobenius_maximum": 1.0e-6,
            "passivity_maximum_eigenvalue_hartree_per_e2": 1.0e-8,
        },
        "decision": {
            "all_32_records_must_pass_every_gate": True,
            "any_provider_failure_stops_expanded_response_generation": True,
            "energy_curvature_target_or_gate": False,
            "success_authorizes_only_twelve_mode_cpks_generation": True,
        },
        "claim_boundary": {
            "train_split_QM_response_generator_coverage_only": True,
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
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
