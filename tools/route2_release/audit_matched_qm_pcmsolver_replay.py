#!/usr/bin/env python3
"""Audit two complete matched QM/PCMSolver panel executions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from maple.solvation.reference.replay import audit_complete_reference_replay
from maple.solvation.release import (
    RepositorySnapshot,
    collect_loaded_repository_sources,
    committed_source_hashes,
    write_external_json_artifact,
)

PREREGISTRATION_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-matched-qm-pcmsolver-four-prereg-v3.json"
)
REQUIRED_SOURCE_PATHS = (
    "maple/solvation/reference/replay.py",
    "tools/route2_release/audit_matched_qm_pcmsolver_replay.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-run", type=Path, required=True)
    parser.add_argument("--second-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    snapshot = RepositorySnapshot.capture(repo_root)
    preregistration = snapshot.root / PREREGISTRATION_RELATIVE_PATH
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    try:
        output.relative_to(snapshot.root)
    except ValueError:
        pass
    else:
        raise RuntimeError("replay audit output must be outside the checkout.")

    audit = audit_complete_reference_replay(
        args.first_run,
        args.second_run,
        preregistration,
    )
    source_paths = collect_loaded_repository_sources(
        snapshot.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    result = {
        **audit,
        "repository": snapshot.as_dict(),
        "source_sha256": committed_source_hashes(snapshot, source_paths),
    }
    snapshot.assert_unchanged()
    write_external_json_artifact(snapshot, output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
