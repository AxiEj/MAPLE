#!/usr/bin/env python3
"""Replace the heavy-element-incompatible SG1 NLC profile before QM execution."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SELF_REPO_PATH = (
    "tools/route2_release/create_vqm24_qm_record_level3_preregistration.py"
)
RESPONSE_RUNNER = SOURCE_ROOT / (
    "tools/route2_release/run_vqm24_localized_qm_response.py"
)
ARTIFACT = "route2-vqm24-localized-field-qm-record-level3-prereg-v1"


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


def create(args: argparse.Namespace) -> dict[str, Any]:
    parent_path = args.parent.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    parent = json.loads(parent_path.read_text())
    if (
        parent.get("artifact")
        != "route2-vqm24-localized-field-qm-record-prereg-v1"
        or parent.get("status") != "locked-before-record-qm-execution"
    ):
        raise ValueError("parent VQM24 record preregistration is invalid.")
    for record in parent["input_files"].values():
        path = Path(str(record["path"])).resolve(strict=True)
        if _sha256(path) != record["sha256"]:
            raise ValueError("parent VQM24 record input drifted.")
    gas_directory = Path(parent["outputs"]["gas_directory"])
    response_directory = Path(parent["outputs"]["response_directory"])
    if response_directory.exists():
        raise ValueError("response output exists before level-3 preregistration.")
    if gas_directory.exists() and any(gas_directory.iterdir()):
        raise ValueError(
            "nonempty gas output must be archived or removed before level-3 lock."
        )
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-level3-record-qm-execution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent": {
            "path": str(parent_path),
            "sha256": _sha256(parent_path),
            "preregistration_sha256": parent["preregistration_sha256"],
            "disposition": (
                "pre-QM SG1 NLC grid rejected because PySCF sg1_prune supports "
                "atomic numbers only through Ar"
            ),
        },
        "selection": parent["selection"],
        "input_files": parent["input_files"],
        "probe_protocol": parent["probe_protocol"],
        "source": {
            "creator_path": SELF_REPO_PATH,
            "creator_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "clean_snapshot": parent["source"]["clean_snapshot"],
            "clean_snapshot_git_head": parent["source"]["clean_snapshot_git_head"],
            "gas_runner": parent["source"]["gas_runner"],
            "response_runner": {
                "path": str(RESPONSE_RUNNER),
                "sha256": _sha256(RESPONSE_RUNNER),
            },
        },
        "qm_protocol": {
            "method": "omegaB97M-V",
            "basis": "def2-tzvpd",
            "reference": "RKS density fitting",
            "semilocal_grid_level": 3,
            "nonlocal_grid_profile": "PySCF level 3",
            "nonlocal_prune": "PySCF level default",
            "threads": 8,
            "maximum_memory_mb": 8000,
        },
        "outputs": parent["outputs"],
        "claim_boundary": parent["claim_boundary"],
    }
    payload["preregistration_sha256"] = _canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    output.chmod(0o444)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
