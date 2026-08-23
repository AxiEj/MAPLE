#!/usr/bin/env python3
"""Prepare the private label-free MNSol-10 geometry/solvent input bundle.

The captured distribution bytes are used only to reconstruct and validate the
frozen selection.  The emitted bundle contains coordinates plus opaque
identities and no experimental target, row name, or model output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.route2_release._mnsol_label_free_inputs import (  # noqa: E402
    derive_mnsol10_label_free_inputs,
)
from tools.route2_release._secure_artifacts import (  # noqa: E402
    SecureArtifactError,
    StabilityGuard,
    capture_clean_repository,
    capture_file,
    canonical_sha256,
    load_json_bytes,
    publish_json_noreplace,
)

BENCHMARK_DIR = REPO_ROOT / "docs" / "implicit-solvation" / "benchmarks"
PREREGISTRATION = BENCHMARK_DIR / (
    "route2-mace-mdp-polar-hybrid-mnsol10-fullsolv-prereg-v1.json"
)
PROTOCOL = BENCHMARK_DIR / "route2-mnsol-protocol-v1.json"
SELECTION = BENCHMARK_DIR / "route2-mnsol-pilot-selection-v1.json"
OUTPUT = REPO_ROOT / ".omx/route2/hybrid-mnsol10/label-free-input-v1.json"
ARTIFACT_ID = "route2-mace-mdp-polar-hybrid-mnsol10-label-free-input-v1"


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SecureArtifactError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def prepare(source: str | Path) -> tuple[dict[str, object], StabilityGuard]:
    repository = capture_clean_repository(REPO_ROOT)
    captures = {
        "source": capture_file(source, role="MNSol distribution"),
        "preregistration": capture_file(PREREGISTRATION, role="preregistration"),
        "protocol": capture_file(PROTOCOL, role="MNSol protocol"),
        "selection": capture_file(SELECTION, role="MNSol selection"),
    }
    stability = StabilityGuard(repository, tuple(captures.items()))
    preregistration = load_json_bytes(
        captures["preregistration"].data, role="preregistration"
    )
    selection_payload = load_json_bytes(
        captures["selection"].data, role="MNSol selection"
    )
    dataset_contract = _mapping(
        preregistration.get("dataset_contract"), name="dataset_contract"
    )
    if {
        "protocol_sha256": dataset_contract.get("protocol_sha256"),
        "selection_sha256": dataset_contract.get("selection_sha256"),
        "selection_fingerprint": dataset_contract.get("selection_fingerprint"),
        "record_count": dataset_contract.get("record_count"),
    } != {
        "protocol_sha256": captures["protocol"].sha256,
        "selection_sha256": captures["selection"].sha256,
        "selection_fingerprint": selection_payload.get("selection_fingerprint"),
        "record_count": 10,
    }:
        raise SecureArtifactError("preregistration dataset identity drifted")
    dataset, records = derive_mnsol10_label_free_inputs(
        source_bytes=captures["source"].data,
        protocol_path=captures["protocol"].path,
        selection_payload=selection_payload,
        benchmark_directory=BENCHMARK_DIR,
    )
    bundle: dict[str, object] = {
        "schema_id": "maple-route2-label-free-accuracy-input-v1",
        "artifact_id": ARTIFACT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": repository.as_dict(),
        "preregistration_sha256": captures["preregistration"].sha256,
        "protocol_sha256": captures["protocol"].sha256,
        "selection_sha256": captures["selection"].sha256,
        "selection_fingerprint": str(selection_payload["selection_fingerprint"]),
        "dataset": dataset,
        "record_count": len(records),
        "records": records,
        "redistribution_guard": {
            "raw_rows_emitted": False,
            "entry_numbers_emitted": False,
            "geometry_handles_emitted": False,
            "solute_names_emitted": False,
            "formulas_emitted": False,
            "experimental_values_emitted": False,
        },
        "claim_boundary": (
            "Private label-free coordinates and solvent identities for the exact "
            "frozen MNSol-10 known-panel regression. This artifact contains no "
            "experimental target, error, prediction, fit, or capability claim."
        ),
    }
    bundle["content_sha256"] = canonical_sha256(bundle)
    accuracy = importlib.import_module("maple.solvation.release.accuracy_admission")
    accuracy.validate_label_free_input_bundle(bundle, expected_record_count=10)
    stability.assert_stable()
    return bundle, stability


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        bundle, stability = prepare(args.source)
        publish_json_noreplace(
            OUTPUT,
            bundle,
            root=REPO_ROOT,
            stability=stability.assert_stable,
        )
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": os.fspath(OUTPUT),
                "content_sha256": bundle["content_sha256"],
                "record_count": bundle["record_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
