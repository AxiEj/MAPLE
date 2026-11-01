#!/usr/bin/env python3
"""Run deterministic pre-seal Route 1 algorithm contract nodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for directory in (ROOT, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import validate_independent_obc2_default as qualification  # noqa: E402
import validate_route1_algorithm_contracts as validator  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = qualification.source_snapshot(require_clean=True)
    cache = {}
    rows = {}
    for spec in validator.CANONICAL_ROWS:
        row_id = spec.row_id
        capability = spec.capability
        node = spec.test_node
        implementation = spec.implementation
        if node not in cache:
            cache[node] = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", node], cwd=ROOT,
                text=True, capture_output=True,
            )
        completed = cache[node]
        test_file = node.split("::", 1)[0]
        expected = validator.expected_outcome(spec)
        rows[row_id] = {
            "capability_key": capability,
            "implementation_files_sha256": {implementation: _sha256(ROOT / implementation)},
            "test_file": test_file,
            "test_node": node,
            "test_file_sha256": _sha256(ROOT / test_file),
            "fixture_identity": validator.fixture_identity(spec),
            "expected": expected,
            "observed": expected if completed.returncode == 0 else completed.stdout + completed.stderr,
            "passed": completed.returncode == 0,
        }
    after = qualification.source_snapshot(require_clean=True)
    artifact = {
        "schema_version": validator.SCHEMA_VERSION,
        "artifact_type": "route1-advanced-algorithm-contracts",
        "source_before": before,
        "source_after": after,
        "capability_keys": sorted(validator.EXPECTED_CAPABILITIES),
        "rows": rows,
    }
    artifact["content_sha256"] = hashlib.sha256(validator.canonical_payload(artifact)).hexdigest()
    validator.validate_artifact(artifact, expected_source_snapshot=before)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
