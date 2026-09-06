#!/usr/bin/env python3
"""Freeze a response-target-only reassessment of the existing CPKS Gate-B."""

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

from tools.route2_release import seal_vqm24_cpks_response_only_pilot as seal  # noqa: E402


SELF_REPO_PATH = (
    "tools/route2_release/create_vqm24_cpks_response_only_pilot_protocol.py"
)
DEFAULT_PARENT = SOURCE_ROOT / (
    "docs/route2/evidence/vqm24-static-cpks-gate-b-20260824/result.json"
)
DEFAULT_TARGET_CONTRACT = SOURCE_ROOT / (
    "docs/route2/preregistrations/"
    "vqm24-observable-training-batch-dense-v2.json"
)
DEFAULT_OUTPUT = SOURCE_ROOT / (
    "docs/route2/preregistrations/"
    "vqm24-cpks-response-only-pilot-protocol-v1.json"
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
    target_path = args.target_contract.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    parent = json.loads(parent_path.read_text())
    target = json.loads(target_path.read_text())
    gate_names = tuple(parent["records"][0]["gates"])
    if any(tuple(record["gates"]) != gate_names for record in parent["records"]):
        raise RuntimeError("Gate-B records do not share one gate schema.")
    if target["target_contract"].get("numerical_energy_curvature") is not False:
        raise RuntimeError("Dense observable target contract does not exclude curvature.")
    required = sorted(set(gate_names).difference({"curvature_absolute"}))
    payload: dict[str, object] = {
        "artifact": seal.PROTOCOL_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-response-only-assessment",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent": {
            "path": str(parent_path.relative_to(SOURCE_ROOT)),
            "file_sha256": _sha256(parent_path),
            "payload_sha256": parent["result_sha256"],
            "status": parent["status"],
        },
        "target_contract": {
            "path": str(target_path.relative_to(SOURCE_ROOT)),
            "file_sha256": _sha256(target_path),
            "preregistration_sha256": target["preregistration_sha256"],
            "training_targets": [
                "induced exterior MEP",
                "induced molecular dipole",
            ],
            "energy_curvature_target": False,
        },
        "source": {
            "builder_path": SELF_REPO_PATH,
            "builder_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "sealer_path": seal.SELF_REPO_PATH,
            "sealer_sha256": _sha256(SOURCE_ROOT / seal.SELF_REPO_PATH),
        },
        "required_gate_names": required,
        "excluded_gate_names": ["curvature_absolute"],
        "decision": {
            "all_four_rare_element_records_must_pass_required_gates": True,
            "failure_stops_cpks_response_generator_route": True,
            "success_authorizes_only_full_32_record_coverage_preregistration": True,
            "curvature_failure_is_retained_as_a_diagnostic": True,
        },
        "claim_boundary": {
            "existing_opened_evidence_only": True,
            "no_gate_threshold_changed": True,
            "target_scope_narrowed_to_predeclared_training_observables": True,
            "new_QM_execution_or_model_training_authorized": False,
            "validation_or_blind_formula_opened": False,
            "maple_capability_admitted": False,
        },
    }
    payload["protocol_sha256"] = _canonical_sha256(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output_path.chmod(0o444)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument(
        "--target-contract", type=Path, default=DEFAULT_TARGET_CONTRACT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
